"""Тесты стейт-машины пайплайна из test_main.py.

Сеть не используется: вместо LLM-провайдера подставляются заглушки агентов,
которые возвращают заранее заданные ответы (валидный код, код с ошибкой,
вердикты APPROVED/REJECTED).

Проверяется главное требование: пайплайн обязан завершаться (а не крутиться
в цикле 5 <-> 6 или 5 <-> 7) и корректно маршрутизировать шаги.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import unittest
from unittest import mock

import test_main
from core.logging_setup import close_logging_handlers
from test_main import (
    FINISH_OK,
    LIMIT_EXIT,
    MAX_REVIEW_ATTEMPTS,
    AgentType,
    process_step_5_coder,
    process_step_6_compiler,
    process_step_7_tester,
)
from tests.test_code_validation import (
    BROKEN_MISSING_COLON,
    VALID_PROJECT_SCANNER,
)


class FakeAgent:
    """Заглушка агента: отдаёт ответы из очереди, не обращаясь к LLM."""

    def __init__(self, name: str, responses=None, default=None):
        self.name = name
        self.responses = list(responses or [])
        self.default = default
        self.calls: list[dict] = []

    async def execute_task(self, prompt: str, task_type: str = "chat"):
        self.calls.append({"prompt": prompt, "task_type": task_type})
        if self.responses:
            return self.responses.pop(0)
        return self.default


APPROVED = "Всё хорошо.\n<verdict>APPROVED</verdict>"
REJECTED = "Нашёл баги.\n<verdict>REJECTED</verdict>\nДобавь обработку пустых файлов."


def build_agents(
    coder_responses=None,
    coder_default=None,
    tester_responses=None,
    tester_default=None,
    spec_writer_default="ТЗ: скрипт для Obsidian.",
    reviewer_default=APPROVED,
    architect_default="Архитектура: сканер, анализатор, генератор md.",
    compiler_default="Статус сборки: SUCCESS",
):
    """Собирает словарь агентов в том же формате, что create_agents()."""
    return {
        AgentType.SPEC_WRITER: FakeAgent("analyst", default=spec_writer_default),
        AgentType.SPEC_REVIEWER: FakeAgent("validator", default=reviewer_default),
        AgentType.ARHITEKTOR: FakeAgent("architect", default=architect_default),
        AgentType.CODER: FakeAgent(
            "coder", responses=coder_responses, default=coder_default
        ),
        AgentType.TESTER: FakeAgent(
            "tester", responses=tester_responses, default=tester_default
        ),
        AgentType.COMPILER: FakeAgent("compiler", default=compiler_default),
    }


def run_pipeline(agents, context, max_iterations=100):
    """Прогоняет стейт-машину main() без LLM. Возвращает (шаг, число итераций)."""

    async def _run():
        step = 2
        spec_attempts = 0
        code_attempts = 0
        iterations = 0
        while step <= 7:
            iterations += 1
            if iterations > max_iterations:
                # Пайплайн зациклился — это провал требования о завершении.
                return None, iterations
            if step == 2:
                step = await test_main.process_step_2_spec_writer(agents, context)
            elif step == 3:
                step, spec_attempts = await test_main.process_step_3_spec_reviewer(
                    agents, context, spec_attempts
                )
            elif step == 4:
                step = await test_main.process_step_4_arhitektor(agents, context)
            elif step == 5:
                step = await process_step_5_coder(agents, context)
            elif step == 6:
                step, code_attempts = await process_step_6_compiler(
                    agents, context, code_attempts
                )
            elif step == 7:
                step, code_attempts = await process_step_7_tester(
                    agents, context, code_attempts
                )
        return step, iterations

    return asyncio.run(_run())


def new_context():
    """Контекст в том виде, в котором его создаёт main()."""
    return {"user_idea": test_main.MY_PROMPT}


def context_with_artifacts(code=None):
    """Контекст с ТЗ и архитектурой — для точечных вызовов шагов 5/6/7."""
    context = new_context()
    context[AgentType.SPEC_WRITER] = "ТЗ: скрипт для Obsidian."
    context[AgentType.ARHITEKTOR] = "Архитектура: сканер, анализатор, генератор md."
    if code is not None:
        context[AgentType.CODER] = code
    return context


class TestPipelineHappyPath(unittest.TestCase):
    """Успешный сценарий: код компилируется и согласован тестировщиком."""

    def test_full_pipeline_finishes_ok(self):
        agents = build_agents(
            coder_default=VALID_PROJECT_SCANNER, tester_default=APPROVED
        )
        context = new_context()

        step, iterations = run_pipeline(agents, context)

        self.assertEqual(step, FINISH_OK)
        # Ровно один проход по шагам 2,3,4,5,6,7 — без лишних перегенераций.
        self.assertEqual(iterations, 6)
        self.assertEqual(len(agents[AgentType.CODER].calls), 1)
        self.assertEqual(len(agents[AgentType.TESTER].calls), 1)

    def test_context_is_filled_with_all_artifacts(self):
        agents = build_agents(
            coder_default=VALID_PROJECT_SCANNER, tester_default=APPROVED
        )
        context = new_context()

        run_pipeline(agents, context)

        self.assertTrue(context[AgentType.SPEC_WRITER])
        self.assertTrue(context[AgentType.ARHITEKTOR])
        self.assertEqual(context[AgentType.CODER], VALID_PROJECT_SCANNER.strip())
        # Компиляция прошла — замечаний сборки нет.
        self.assertEqual(context[AgentType.COMPILER], "")

    def test_coder_prompt_contains_spec_and_architecture(self):
        agents = build_agents(
            coder_default=VALID_PROJECT_SCANNER, tester_default=APPROVED
        )
        context = new_context()

        run_pipeline(agents, context)

        prompt = agents[AgentType.CODER].calls[0]["prompt"]
        self.assertIn(context[AgentType.SPEC_WRITER], prompt)
        self.assertIn(context[AgentType.ARHITEKTOR], prompt)
        # Для генерации кода должен использоваться task_type="code".
        self.assertEqual(agents[AgentType.CODER].calls[0]["task_type"], "code")


class TestPipelineTermination(unittest.TestCase):
    """Пайплайн обязан завершаться при любом поведении агентов."""

    def test_perpetual_code_rejection_terminates(self):
        """Компиляция успешна, тестировщик всегда отклоняет — цикл 5<->6<->7.

        Счётчик правок кода не должен сбрасываться на успешной компиляции,
        иначе счётчик всегда равен 1 и выход по лимиту не сработает.
        """
        agents = build_agents(
            coder_default=VALID_PROJECT_SCANNER, tester_default=REJECTED
        )
        context = new_context()

        step, iterations = run_pipeline(agents, context)

        self.assertIsNotNone(step, "Пайплайн зациклился: лимит итераций исчерпан")
        self.assertEqual(step, LIMIT_EXIT)
        self.assertEqual(len(agents[AgentType.CODER].calls), MAX_REVIEW_ATTEMPTS)
        self.assertEqual(len(agents[AgentType.TESTER].calls), MAX_REVIEW_ATTEMPTS)
        self.assertLess(iterations, 40)

    def test_perpetual_syntax_error_terminates(self):
        """Кодер всегда отдаёт код с синтаксической ошибкой."""
        agents = build_agents(coder_default=BROKEN_MISSING_COLON)
        context = new_context()

        step, iterations = run_pipeline(agents, context)

        self.assertEqual(step, LIMIT_EXIT)
        self.assertEqual(len(agents[AgentType.CODER].calls), MAX_REVIEW_ATTEMPTS)
        self.assertLess(iterations, 40)

    def test_empty_code_terminates(self):
        """Кодер возвращает None (ошибка LLM) — выход, а не бесконечный цикл."""
        agents = build_agents(coder_default=None)
        context = new_context()

        step, _ = run_pipeline(agents, context)

        self.assertEqual(step, LIMIT_EXIT)

    def test_compiler_feedback_is_passed_back_to_coder(self):
        """Ошибка компиляции обязана попадать в промпт следующей попытки."""
        agents = build_agents(
            coder_responses=[BROKEN_MISSING_COLON, VALID_PROJECT_SCANNER],
            tester_default=APPROVED,
        )
        context = new_context()

        step, _ = run_pipeline(agents, context)

        self.assertEqual(step, FINISH_OK)
        self.assertEqual(len(agents[AgentType.CODER].calls), 2)
        second_prompt = agents[AgentType.CODER].calls[1]["prompt"]
        self.assertIn("Ошибки компилятора", second_prompt)
        self.assertIn("SyntaxError", second_prompt)


class TestStep6Compiler(unittest.TestCase):
    """Точечные проверки шага 6."""

    def test_success_returns_step_7_and_keeps_attempts(self):
        agents = build_agents()
        context = context_with_artifacts(VALID_PROJECT_SCANNER)

        step, attempts = asyncio.run(process_step_6_compiler(agents, context, 3))

        self.assertEqual(step, 7)
        self.assertEqual(attempts, 3, "Счётчик правок кода не должен сбрасываться")

    def test_verdict_tag_from_coder_does_not_break_compilation(self):
        """Ответ кодера содержит <verdict>APPROVED</verdict> (требование CODER_PROMPT).

        Тег — не Python-код, поэтому перед валидацией он должен отрезаться,
        иначе ни один корректный ответ кодера не пройдёт шаг компиляции.
        """
        agents = build_agents()
        context = new_context()
        context[AgentType.CODER] = (
            f"{VALID_PROJECT_SCANNER}\n<verdict>APPROVED</verdict>"
        )

        step, _ = asyncio.run(process_step_6_compiler(agents, context, 0))

        self.assertEqual(step, 7)
        self.assertNotIn("verdict", context[AgentType.CODER])

    def test_markdown_fenced_code_compiles(self):
        agents = build_agents()
        context = new_context()
        context[AgentType.CODER] = f"```python\n{VALID_PROJECT_SCANNER}```"

        step, _ = asyncio.run(process_step_6_compiler(agents, context, 0))

        self.assertEqual(step, 7)

    def test_syntax_error_returns_step_5_with_feedback(self):
        agents = build_agents()
        context = new_context()
        context[AgentType.CODER] = BROKEN_MISSING_COLON

        step, attempts = asyncio.run(process_step_6_compiler(agents, context, 0))

        self.assertEqual(step, 5)
        self.assertEqual(attempts, 1)
        self.assertIn("SyntaxError", context[AgentType.COMPILER])

    def test_missing_code_exits_on_limit(self):
        agents = build_agents()
        context = new_context()

        step, attempts = asyncio.run(
            process_step_6_compiler(agents, context, MAX_REVIEW_ATTEMPTS - 1)
        )

        self.assertEqual(step, LIMIT_EXIT)
        self.assertEqual(attempts, MAX_REVIEW_ATTEMPTS)


class TestStep7Tester(unittest.TestCase):
    """Точечные проверки шага 7."""

    def test_approved_finishes_pipeline(self):
        agents = build_agents(tester_default=APPROVED)
        context = context_with_artifacts(VALID_PROJECT_SCANNER)

        step, attempts = asyncio.run(process_step_7_tester(agents, context, 0))

        self.assertEqual(step, FINISH_OK)
        self.assertEqual(attempts, 0)

    def test_rejection_returns_to_coder_with_feedback(self):
        agents = build_agents(tester_default=REJECTED)
        context = context_with_artifacts(VALID_PROJECT_SCANNER)

        step, attempts = asyncio.run(process_step_7_tester(agents, context, 0))

        self.assertEqual(step, 5)
        self.assertEqual(attempts, 1)
        # В контекст кладётся фидбек, очищенный от служебного тега вердикта.
        feedback = context[AgentType.TESTER]
        self.assertNotIn("verdict", feedback)
        self.assertIn("Добавь обработку пустых файлов", feedback)

    def test_missing_code_exits(self):
        agents = build_agents()
        context = new_context()

        step, _ = asyncio.run(process_step_7_tester(agents, context, 0))

        self.assertEqual(step, LIMIT_EXIT)


class TestStep5Coder(unittest.TestCase):
    """Шаг 5: кодера нельзя кормить его же markdown-обёртками."""

    def test_previous_code_is_cleaned_before_prompting(self):
        agents = build_agents(coder_default=VALID_PROJECT_SCANNER)
        context = new_context()
        context[AgentType.ARHITEKTOR] = "Архитектура."
        context[AgentType.SPEC_WRITER] = "ТЗ."
        context[AgentType.CODER] = f"```python\n{VALID_PROJECT_SCANNER}```"

        asyncio.run(process_step_5_coder(agents, context))

        prompt = agents[AgentType.CODER].calls[0]["prompt"]
        # Код передаётся в ровно одной обёртке, без вложенных ```.
        self.assertEqual(prompt.count("```python"), 1)
        self.assertEqual(prompt.count("```"), 2)

    def test_empty_coder_response_stops_pipeline(self):
        agents = build_agents(coder_default=None)
        context = context_with_artifacts()

        step = asyncio.run(process_step_5_coder(agents, context))

        self.assertEqual(step, LIMIT_EXIT)


class TestSaveResults(unittest.TestCase):
    """save_results(): файлы сохраняются всегда, даже при пустом контексте."""

    def test_saves_tz_code_and_feedback(self):
        context = new_context()
        context[AgentType.SPEC_WRITER] = "ТЗ проекта."
        context[AgentType.CODER] = VALID_PROJECT_SCANNER
        context[AgentType.TESTER] = "Замечания тестировщика."

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"PROJECT_DIR": tmpdir}):
                project_dir = test_main.save_results(context)

            self.assertEqual(project_dir, tmpdir)
            with open(
                os.path.join(tmpdir, "result.py"), encoding="utf-8"
            ) as handle:
                self.assertEqual(handle.read(), VALID_PROJECT_SCANNER)
            with open(os.path.join(tmpdir, "TZ.txt"), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "ТЗ проекта.")
            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, "Review_Feedback.txt"))
            )

    def test_empty_context_does_not_raise(self):
        """Контекст без ТЗ и кода: файлы создаются пустыми, исключения нет."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"PROJECT_DIR": tmpdir}):
                project_dir = test_main.save_results(new_context())

            self.assertEqual(project_dir, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "result.py")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "TZ.txt")))
            # Замечаний нет — отдельный файл не создаётся.
            self.assertFalse(
                os.path.exists(os.path.join(tmpdir, "Review_Feedback.txt"))
            )

    def test_unwritable_dir_returns_none(self):
        context = new_context()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Родитель пути — обычный файл: создать в нём папку нельзя.
            blocker = os.path.join(tmpdir, "blocker.txt")
            with open(blocker, "w", encoding="utf-8") as handle:
                handle.write("not a directory")
            bad_dir = os.path.join(blocker, "subdir")
            with mock.patch.dict(os.environ, {"PROJECT_DIR": bad_dir}):
                self.assertIsNone(test_main.save_results(context))


class TestMainIntegration(unittest.TestCase):
    """Полный main() с подменённым провайдером: без сети и без реальных затрат."""

    def setUp(self):
        # main() вызывает setup_logging(): хендлеры логгера меняются, поэтому
        # сохраняем исходные и закрываем свои после теста. Временную папку
        # создаём через mkdtemp + addCleanup: cleanup идёт ПОСЛЕ tearDown,
        # то есть когда agent.log уже не заблокирован (иначе на Windows
        # удаление падает с PermissionError).
        self._root = logging.getLogger()
        self._saved_handlers = self._root.handlers[:]
        self._saved_level = self._root.level
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_tmpdir)

    def tearDown(self):
        close_logging_handlers()
        self._root.handlers[:] = self._saved_handlers
        self._root.setLevel(self._saved_level)

    def _cleanup_tmpdir(self):
        close_logging_handlers()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_main(self, agents, capture=True):
        """Прогоняет main() с заглушками.

        capture=True перехватывает записи логгера test_main через assertLogs.
        Для проверки ФАЙЛА перехват не нужен: assertLogs отключает propagate
        у логгера test_main, и записи не доходят до файлового хендлера корня.
        """
        env = {
            "PROJECT_DIR": self.tmpdir,
            "LOG_DIR": os.path.join(self.tmpdir, "logs"),
        }
        with mock.patch.dict(os.environ, env):
            with mock.patch.object(test_main, "create_agents", return_value=agents):
                with mock.patch.object(
                    test_main, "CloudAPIProvider"
                ) as provider_cls:
                    if not capture:
                        asyncio.run(test_main.main())
                        return provider_cls, None
                    with self.assertLogs("test_main", level="INFO") as captured:
                        asyncio.run(test_main.main())
        return provider_cls, captured

    def test_main_saves_results_and_returns(self):
        agents = build_agents(
            coder_default=VALID_PROJECT_SCANNER, tester_default=APPROVED
        )

        provider_cls, captured = self._run_main(agents)

        provider_cls.assert_called_once()
        logs = "\n".join(captured.output)
        self.assertIn("Пайплайн завершён успешно", logs)
        self.assertIn("Старт пайплайна", logs)
        result_path = os.path.join(self.tmpdir, "result.py")
        self.assertTrue(os.path.exists(result_path))
        with open(result_path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), VALID_PROJECT_SCANNER.strip())

    def test_main_terminates_when_tester_always_rejects(self):
        """Даже при вечных отклонениях main() обязан дойти до конца."""
        agents = build_agents(
            coder_default=VALID_PROJECT_SCANNER, tester_default=REJECTED
        )

        _, captured = self._run_main(agents)

        logs = "\n".join(captured.output)
        self.assertIn("остановлен по лимиту попыток", logs)
        self.assertIn("Работа завершена", logs)

    def test_main_writes_log_file(self):
        """Логи пишутся не только в консоль, но и в файл logs/agent.log."""
        agents = build_agents(
            coder_default=VALID_PROJECT_SCANNER, tester_default=APPROVED
        )

        self._run_main(agents, capture=False)
        for handler in logging.getLogger().handlers:
            handler.flush()
        log_path = os.path.join(self.tmpdir, "logs", "agent.log")
        with open(log_path, encoding="utf-8") as handle:
            content = handle.read()

        # DEBUG-детали шагов попадают в файл, даже если консоль на INFO.
        self.assertIn("Шаг 6: проверка синтаксиса", content)
        self.assertIn("Шаг 5: ответ кодера", content)
        # Секреты в лог не попадают.
        self.assertNotIn("cloud_key", content)


if __name__ == "__main__":
    unittest.main()
