"""Тесты борьбы с «молчаливой обрезкой» ответа LLM (P0 root cause).

Что проверяется:
  - ``LLMResponse`` корректно хранит finish_reason и его свойство ``truncated``;
  - ``coerce_response()`` совместим со строкой и с None (обратная совместимость
    с FakeAgent из test_pipeline_flow);
  - ``Helper.is_response_complete()`` отличает оборванный ответ от полного;
  - шаг 5 переспрашивает кодера при обрыве (finish_reason=length ИЛИ
    структурная незавершённость) и выходит по лимиту, если обрыв повторяется.

Это регрессионные тесты на корень проблемы: 44/50 ответов кодера не имели
закрывающего ``<verdict>`` из-за обрезки на 8192 токенах, и пайплайн крутился
в цикле 5 <-> 6 <-> 7 бесконечно.
"""

import asyncio
import unittest

import test_main
from core.llm_types import LLMResponse, coerce_response, to_text
from test_main import LIMIT_EXIT, AgentType, process_step_5_coder
from tests.test_code_validation import VALID_PROJECT_SCANNER
from tests.test_pipeline_flow import build_agents, context_with_artifacts
from utils.helpers import Helper


class TruncationMetaAgent:
    """FakeAgent, но с метаданными последнего ответа (finish_reason).

    ``items`` — очередь кортежей ``(текст, finish_reason)``: каждый вызов
    возвращает следующий элемент и выставляет ``last_response_meta``.
    Это позволяет эмулировать «первый ответ обрезан, повторный ок».
    """

    def __init__(self, items):
        self.name = "coder"  # шаг 5 логирует agents[CODED].name
        self.items = list(items)
        self.calls = []
        self.last_response_meta = None

    async def execute_task(self, prompt: str, task_type: str = "chat"):
        self.calls.append({"prompt": prompt, "task_type": task_type})
        if self.items:
            text, reason = self.items.pop(0)
            self.last_response_meta = LLMResponse(text=text, finish_reason=reason)
            return text
        self.last_response_meta = LLMResponse(text="")
        return None


class TestLLMResponseType(unittest.TestCase):
    def test_truncated_flag_is_true_on_length(self):
        response = LLMResponse(text="код...", finish_reason="length")
        self.assertTrue(response.truncated)

    def test_truncated_flag_is_false_on_stop(self):
        response = LLMResponse(text="код", finish_reason="stop")
        self.assertFalse(response.truncated)

    def test_truncated_flag_is_false_when_reason_missing(self):
        # API без finish_reason (None) НЕ должны считаться обрезанными:
        # иначе сломаем провайдеры, которые не отдают причину.
        response = LLMResponse(text="код", finish_reason=None)
        self.assertFalse(response.truncated)

    def test_usage_tokens_reach_pipeline(self):
        response = LLMResponse(
            text="код", finish_reason="length", prompt_tokens=10, completion_tokens=8192
        )
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 8192)


class TestCoerceResponse(unittest.TestCase):
    def test_coerce_keeps_llm_response_as_is(self):
        original = LLMResponse(text="x", finish_reason="length")
        self.assertIs(coerce_response(original), original)

    def test_coerce_wraps_str(self):
        wrapped = coerce_response("просто строка")
        self.assertIsInstance(wrapped, LLMResponse)
        self.assertEqual(wrapped.text, "просто строка")
        self.assertFalse(wrapped.truncated)

    def test_coerce_none_to_empty(self):
        wrapped = coerce_response(None)
        self.assertEqual(wrapped.text, "")
        self.assertTrue(wrapped.is_empty)

    def test_to_text_digests_both_forms(self):
        self.assertEqual(to_text("строка"), "строка")
        self.assertEqual(to_text(LLMResponse(text="обёрнуто")), "обёрнуто")
        self.assertEqual(to_text(None), "")


class TestIsResponseComplete(unittest.TestCase):
    def test_full_valid_code_is_complete(self):
        self.assertTrue(Helper.is_response_complete(VALID_PROJECT_SCANNER))

    def test_empty_is_incomplete(self):
        self.assertFalse(Helper.is_response_complete(""))
        self.assertFalse(Helper.is_response_complete("   \n "))

    def test_opened_verdict_tag_is_incomplete(self):
        # Реальный паттерн обрыва: модель начала тег, но не закрыла.
        broken = "def f():\n    pass\n<verdict>APPROVED"
        self.assertFalse(Helper.is_response_complete(broken))

    def test_closed_verdict_tag_is_complete(self):
        ok = "def f():\n    pass\n<verdict>APPROVED</verdict>"
        self.assertTrue(Helper.is_response_complete(ok))

    def test_unclosed_parenthesis_in_tail_is_incomplete(self):
        broken = "app = Obsidian(\n    vault='x'"
        self.assertFalse(Helper.is_response_complete(broken))

    def test_dangling_docstring_is_incomplete(self):
        broken = 'def f():\n    """незакрытый docstring'
        self.assertFalse(Helper.is_response_complete(broken))

    def test_unclosed_markdown_fence_is_incomplete(self):
        broken = "line1\n```python\nprint(1)"
        self.assertFalse(Helper.is_response_complete(broken))
class TestStep5TruncationHandling(unittest.TestCase):
    """Шаг 5 не должен принимать оборванный код как полноценный."""

    def build_coder(self, items):
        agents = build_agents(coder_default=VALID_PROJECT_SCANNER)
        agents[AgentType.CODER] = TruncationMetaAgent(items)
        return agents

    def test_step5_retries_once_on_provider_truncation(self):
        """finish_reason=length -> один повтор -> полный код -> шаг 6."""
        agents = self.build_coder(
            [
                ("def f():\n    pas", "length"),
                (VALID_PROJECT_SCANNER, "stop"),
            ]
        )
        context = context_with_artifacts()

        step = asyncio.run(process_step_5_coder(agents, context))

        self.assertEqual(step, 6)
        self.assertEqual(len(agents[AgentType.CODER].calls), 2)
        # В повторный промпт попала директива об обрыве.
        self.assertIn("ОБОРВАЛСЯ", agents[AgentType.CODER].calls[1]["prompt"])
        # В контекст сохранён ПОЛНЫЙ код, а не огрызок.
        self.assertEqual(context[AgentType.CODER], VALID_PROJECT_SCANNER.strip())

    def test_step5_exits_after_repeated_truncation(self):
        """Два подряд обрыва по лимиту -> эскалация, а не бесконечный цикл."""
        agents = self.build_coder(
            [
                ("def f():\n    pas", "length"),
                ("def g():\n    aga", "length"),
            ]
        )
        context = context_with_artifacts()

        step = asyncio.run(process_step_5_coder(agents, context))

        self.assertEqual(step, LIMIT_EXIT)
        self.assertEqual(len(agents[AgentType.CODER].calls), 2)
        # Огрызок не должен попадать в контекст.
        self.assertNotIn(AgentType.CODER, context)

    def test_step5_retries_on_structural_incompleteness(self):
        """Провайдер НЕ сигналит обрыв (reason=stop), но текст оборван —
        структурный детектор ловит и делает целевой повтор."""
        agents = self.build_coder(
            [
                (f"{VALID_PROJECT_SCANNER}\n<verdict>APPROVED", "stop"),
                (VALID_PROJECT_SCANNER, "stop"),
            ]
        )
        context = context_with_artifacts()

        step = asyncio.run(process_step_5_coder(agents, context))

        self.assertEqual(step, 6)
        self.assertEqual(len(agents[AgentType.CODER].calls), 2)

    def test_polnyi_kod_bez_metki_ne_trigerit_perespros(self):
        """Полный код без <verdict> (валидный) НЕ должен переспрашиваться:
        старые тесты и фейковые агенты отвечают именно так."""
        agents = self.build_coder([])
        context = context_with_artifacts()
        # Провайдер вернул чистый валидный код с finish_reason="stop":
        # метаданные не должны вызывать повторный запрос.
        agents[AgentType.CODER].last_response_meta = LLMResponse(
            text=VALID_PROJECT_SCANNER, finish_reason="stop"
        )

        async def _fake_execute(prompt, task_type="chat"):
            agents[AgentType.CODER].calls.append(
                {"prompt": prompt, "task_type": task_type}
            )
            return VALID_PROJECT_SCANNER

        agents[AgentType.CODER].execute_task = _fake_execute

        step = asyncio.run(process_step_5_coder(agents, context))

        self.assertEqual(step, 6)
        self.assertEqual(len(agents[AgentType.CODER].calls), 1)


if __name__ == "__main__":
    unittest.main()