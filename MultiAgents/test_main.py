import asyncio
import logging
import time

import re, os, shutil
from enum import Enum
from providers.ollama_providers import AsyncOllamaClient
from providers.cloud_api_providers import CloudAPIProvider
from core.base_agent import BaseAgent
from core.logging_setup import get_log_file_path, preview, setup_logging
from config.prompts import (
    ARHITEKTOR_PROMPT,
    CODER_PROMPT,
    TESTER_PROMPT,
    COMPILER_AGENT_PROMPT,
    SPEC_WRITER_PROMPT,
    SPEC_REVIEWER_PROMPT,
    MY_PROMPT,
)
from utils.helpers import Helper

# Логгер на модуль: в каждой записи видно, откуда она пришла (%(name)s).
logger = logging.getLogger(__name__)

MAX_REVIEW_ATTEMPTS = 50  # Максимальное количество правок (отдельно для ТЗ и для кода)
MAX_SPEC_ATTEMPTS = 5  # Максимальное количество правок ТЗ

# Целевой язык генерируемого кода: "python" (obsidian-скрипт из MY_PROMPT)
# или "cpp" (Arduino-скетч). От него зависит способ проверки синтаксиса и
# имя файла результата. Меняется через переменную окружения TARGET_LANG.
TARGET_LANG = os.getenv("TARGET_LANG", "python").lower()
CODE_EXTENSIONS = {"python": "py", "cpp": "ino"}

# Шаги-терминаторы стейт-машины:
LIMIT_EXIT = 8  # Остановка по лимиту правок / из-за отсутствия данных
FINISH_OK = 9  # Успешное завершение (код согласован тестировщиком)


class AgentType(str, Enum):
    """Строгий перечень типов агентов для предотвращения опечаток."""

    SPEC_WRITER = "spec_writer"
    SPEC_REVIEWER = "spec_reviewer"
    ARHITEKTOR = "arhitektor"
    CODER = "coder"
    TESTER = "tester"
    COMPILER = "compiler"
    FEEDBACK = "spec_feedback"


def create_agents(llm_client) -> dict[str, BaseAgent]:
    """Фабрика для создания и инициализации всех агентов системы."""
    return {
        AgentType.SPEC_WRITER: BaseAgent(  # Шаг 2
            name_agent="Системный аналитик",
            role_prompt=SPEC_WRITER_PROMPT,
            llm=llm_client,
        ),
        AgentType.SPEC_REVIEWER: BaseAgent(  # Шаг 3
            name_agent="Главный валидатор",
            role_prompt=SPEC_REVIEWER_PROMPT,
            llm=llm_client,
        ),
        AgentType.ARHITEKTOR: BaseAgent(  # Шаг 4
            name_agent="Архитектор",
            role_prompt=ARHITEKTOR_PROMPT,
            llm=llm_client,
        ),
        AgentType.CODER: BaseAgent(  # Шаг 5
            name_agent="Программист",
            role_prompt=CODER_PROMPT,
            llm=llm_client,
        ),
        AgentType.TESTER: BaseAgent(  # Шаг 6
            name_agent="Тестировщик",
            role_prompt=TESTER_PROMPT,
            llm=llm_client,
        ),
        AgentType.COMPILER: BaseAgent(  # Шаг 7
            name_agent="Компилятор",
            role_prompt=COMPILER_AGENT_PROMPT,
            llm=llm_client,
        ),
    }


# Вспомогательная функция для парсинга вердикта
def parse_verdict(response_text: str | None) -> tuple[str, str]:
    """
    Ищет тег <verdict>APPROVED</verdict> или <verdict>REJECTED</verdict> в тексте.
    Если тегов несколько — берётся последний (итоговый).
    Если тега нет или ответ пустой (ошибка LLM) — считается REJECTED.
    Возвращает кортеж: (статус, чистый_текст_ответа)
    """
    if not response_text or not response_text.strip():
        return (
            "REJECTED",
            "Пустой ответ агента (ошибка LLM/сети). Требуется повторная проверка.",
        )

    tag_pattern = r"<\s*verdict\s*>\s*(APPROVED|REJECTED)\s*<\s*/\s*verdict\s*>"
    matches = list(re.finditer(tag_pattern, response_text, re.IGNORECASE | re.DOTALL))

    if matches:
        status = matches[-1].group(1).upper()
        # Отрезаем все теги вердикта из фидбека для чистоты
        clean_feedback = re.sub(
            tag_pattern, "", response_text, flags=re.IGNORECASE | re.DOTALL
        ).strip()
        return status, clean_feedback

    # Нет тега — принять ответ нельзя (никаких "APPROVED" подстроками:
    # модель часто цитирует инструкцию и это даёт ложные согласования)
    return "REJECTED", response_text.strip()


def strip_verdict_tag(response_text: str) -> str:
    """Убирает служебный тег <verdict>...</verdict> из ответа агента.

    CODER_PROMPT требует завершать ответ вердиктом, поэтому ответ кодера
    выглядит как «код + <verdict>APPROVED</verdict>». Такой текст нельзя
    подавать на валидацию синтаксиса: тег не является кодом.
    Всё, что идёт после тега, тоже отбрасывается (там пояснения модели).
    """
    if not response_text:
        return ""

    tag_pattern = r"<\s*verdict\s*>\s*(?:APPROVED|REJECTED)\s*<\s*/\s*verdict\s*>"
    match = re.search(tag_pattern, response_text, re.IGNORECASE | re.DOTALL)
    if match:
        # Берём текст до тега: код всегда идёт первым.
        return response_text[: match.start()].strip()
    return response_text.strip()


def check_code_syntax(code: str) -> tuple[bool, str | None]:
    """Проверка синтаксиса в зависимости от целевого языка (TARGET_LANG)."""
    if TARGET_LANG == "cpp":
        if shutil.which("avr-g++") is None:
            # Компилятор не установлен — не блокируем пайплайн
            logger.warning("⚠️ avr-g++ не найден, пропускаю проверку Arduino-кода.")
            return True, None
        return Helper.compile_arduino_sketch(code)
    return Helper.validate_syntax_python(code)


# =====================================================================
# МЕТОДЫ ДЛЯ КАЖДОГО ШАГА СТЕЙТ-МАШИНЫ
# =====================================================================
async def process_step_2_spec_writer(
    agents: dict[AgentType, BaseAgent], context: dict
) -> int:
    logger.info(
        "📋 Шаг 2: составление ТЗ (агент «%s»)", agents[AgentType.SPEC_WRITER].name
    )
    logger.debug(
        "📋 Шаг 2: пользовательская идея: %s", preview(context.get("user_idea"))
    )
    spec_text = await agents[AgentType.SPEC_WRITER].execute_task(
        prompt=f"Составь ТЗ для моей идеи : {context['user_idea']}",
        task_type="review",
    )
    if not spec_text:
        logger.error("❌ Аналитик не вернул ТЗ (пустой ответ LLM). Прерываю пайплайн.")
        return LIMIT_EXIT
    context[AgentType.SPEC_WRITER] = spec_text
    logger.info(
        "✅ Шаг 2: ТЗ получено (%d симв.), перехожу к проверке.", len(spec_text)
    )
    logger.debug("📋 Шаг 2: ТЗ: %s", preview(spec_text))
    return 3  # next step 3


async def process_step_3_spec_reviewer(
    agents: dict[AgentType, BaseAgent], context: dict, attempts: int
) -> tuple[int, int]:
    logger.info(
        "✔ Шаг 3: проверка технического задания (попытка %d/%d).",
        attempts + 1,
        MAX_SPEC_ATTEMPTS,
    )
    if not context.get(AgentType.SPEC_WRITER):
        logger.error("❌ Нет ТЗ для проверки. Прерываю пайплайн.")
        return LIMIT_EXIT, attempts

    prompt_for_review = (
        f"Проверь следующее техническое задание :\n\n{context[AgentType.SPEC_WRITER]}"
    )
    if context.get(AgentType.FEEDBACK):
        prompt_for_review += f"\n\nПредыдущие замечания , которые должны быть исправлены : \n{context[AgentType.FEEDBACK]}"

    logger.debug("✔ Шаг 3: промпт для ревью: %s", preview(prompt_for_review))
    reviewer_response = await agents[AgentType.SPEC_REVIEWER].execute_task(
        prompt=prompt_for_review,
        task_type="review",
    )

    status, feedback = parse_verdict(reviewer_response)
    logger.info("✔ Шаг 3: вердикт валидатора ТЗ — %s.", status)
    logger.debug("✔ Шаг 3: замечания валидатора: %s", preview(feedback))

    if status == "APPROVED":
        logger.info("💚 ТЗ успешно согласовано, перехожу к проектированию архитектуры.")
        context[AgentType.FEEDBACK] = ""  # clear feedback
        return 4, 0  # Идем к Архитектору
    else:
        attempts += 1
        logger.warning(
            "⚠️ ТЗ отклонено. Попытка правки %d/%d. Замечания: %s",
            attempts,
            MAX_SPEC_ATTEMPTS,
            preview(feedback),
        )
        if attempts >= MAX_SPEC_ATTEMPTS:
            logger.error(
                "❌ Превышено максимальное количество правок ТЗ (%d).",
                MAX_SPEC_ATTEMPTS,
            )
            return LIMIT_EXIT, attempts  # exit while
        context[AgentType.FEEDBACK] = feedback
        context["user_idea"] = (
            f"Переделай техническое задание. Замечания Валидатора : \n{feedback}\n\nОригинальная идея : {MY_PROMPT}"
        )
        return 2, attempts  # next step 2


async def process_step_4_arhitektor(
    agents: dict[AgentType, BaseAgent], context: dict
) -> int:
    logger.info(
        "📐 Шаг 4: проектирование архитектуры (агент «%s»).",
        agents[AgentType.ARHITEKTOR].name,
    )
    prompt_for_arhi = f"Спроектируй архитектуру согласно данному техническому заданию : \n\n{context[AgentType.SPEC_WRITER]}"
    logger.debug("📐 Шаг 4: промпт архитектора: %s", preview(prompt_for_arhi))
    architecture = await agents[AgentType.ARHITEKTOR].execute_task(
        prompt=prompt_for_arhi,
        task_type="boss",
    )
    if not architecture:
        logger.error(
            "❌ Архитектор не вернул архитектуру (пустой ответ LLM). Прерываю пайплайн."
        )
        return LIMIT_EXIT
    context[AgentType.ARHITEKTOR] = architecture
    logger.info(
        "✅ Шаг 4: архитектура получена (%d симв.), перехожу к написанию кода.",
        len(architecture),
    )
    logger.debug("📐 Шаг 4: архитектура: %s", preview(architecture))
    return 5


async def process_step_5_coder(
    agents: dict[AgentType, BaseAgent], context: dict
) -> int:
    logger.info("💻 Шаг 5: написание кода (агент «%s»).", agents[AgentType.CODER].name)

    if context.get(AgentType.CODER):
        # Отдаём модели уже очищенный от markdown и вердикта код
        context[AgentType.CODER] = Helper.clean_code(
            strip_verdict_tag(context[AgentType.CODER])
        )

    prompt_for_coder = f"Напиши код по архитектуре:\n{context[AgentType.ARHITEKTOR]}\n\nИ ТЗ:\n{context[AgentType.SPEC_WRITER]}"

    if context.get(AgentType.TESTER):  # есть замечания от тестера
        prompt_for_coder += (
            f"\n\nИсправь ошибки из отчета тестировщика:\n{context[AgentType.TESTER]}"
        )
        logger.info("💻 Шаг 5: в промпт добавлены замечания тестировщика.")
    if context.get(AgentType.COMPILER):  # есть замечания от компилера
        prompt_for_coder += (
            f"\n\nИсправь ошибки из отчета компилятора:\n{context[AgentType.COMPILER]}"
        )
        logger.info("💻 Шаг 5: в промпт добавлены замечания компилятора.")
    if context.get(AgentType.CODER):  # есть предыдущий код
        fence = "cpp" if TARGET_LANG == "cpp" else "python"
        prompt_for_coder += (
            f"\n\nТвой предыдущий код :\n```{fence}\n{context[AgentType.CODER]}\n```"
        )

    logger.debug("💻 Шаг 5: промпт кодера: %s", preview(prompt_for_coder))
    code = await agents[AgentType.CODER].execute_task(
        prompt=prompt_for_coder,
        task_type="code",
    )
    if not code:
        logger.error(
            "❌ Программист не вернул код (пустой ответ LLM). Прерываю пайплайн."
        )
        return LIMIT_EXIT
    # CODER_PROMPT обязывает модель выводить <verdict>...</verdict>:
    # без отрезания тега код не пройдёт проверку синтаксиса на шаге 6.
    context[AgentType.CODER] = strip_verdict_tag(code)
    logger.info(
        "✅ Шаг 5: код получен (%d симв.), перехожу к проверке синтаксиса.", len(code)
    )
    logger.debug("💻 Шаг 5: ответ кодера: %s", preview(code))
    return 6


async def process_step_7_tester(
    agents: dict[AgentType, BaseAgent], context: dict, attempts: int
) -> tuple[int, int]:
    logger.info(
        "🧪 Шаг 7: тестирование кода (попытка правки %d/%d).",
        attempts + 1,
        MAX_REVIEW_ATTEMPTS,
    )

    if not context.get(AgentType.CODER):
        logger.error("❌ Нет кода для тестирования. Прерываю пайплайн.")
        return LIMIT_EXIT, attempts

    tester_prompt = f"Протестируй код :\n{context[AgentType.CODER]}\n\nТехническое задание :\n{context[AgentType.SPEC_WRITER]}"

    logger.debug("🧪 Шаг 7: промпт тестировщика: %s", preview(tester_prompt))
    tester_response = await agents[AgentType.TESTER].execute_task(
        prompt=tester_prompt, task_type="review"
    )

    status, feedback = parse_verdict(tester_response)
    logger.info("🧪 Шаг 7: вердикт тестировщика — %s.", status)
    logger.debug("🧪 Шаг 7: замечания тестировщика: %s", preview(feedback))

    if status == "APPROVED":
        logger.info("💚 Код успешно согласован тестировщиком.")
        context[AgentType.TESTER] = ""
        return FINISH_OK, 0  # Успешный финиш
    else:
        attempts += 1
        logger.warning(
            "⚠️ Код отклонён тестировщиком. Попытка правки %d/%d. Замечания: %s",
            attempts,
            MAX_REVIEW_ATTEMPTS,
            preview(feedback),
        )
        if attempts >= MAX_REVIEW_ATTEMPTS:
            logger.error(
                "❌ Превышено максимальное количество правок кода (%d).",
                MAX_REVIEW_ATTEMPTS,
            )
            return LIMIT_EXIT, attempts  # exit while
        context[AgentType.TESTER] = feedback
        return 5, attempts  # next step 5


async def process_step_6_compiler(
    agents: dict[AgentType, BaseAgent], context: dict, attempts: int
) -> tuple[int, int]:
    logger.info("🔧 Шаг 6: проверка синтаксиса (TARGET_LANG=%s).", TARGET_LANG)

    if not context.get(AgentType.CODER):
        logger.error("❌ Шаг 6: код отсутствует в контексте.")
        attempts += 1
        context[AgentType.COMPILER] = "Код отсутствует"
        if attempts >= MAX_REVIEW_ATTEMPTS:
            logger.error("❌ Превышено максимальное количество правок кода (нет кода).")
            return LIMIT_EXIT, attempts  # exit while
        return 5, attempts

    cleaned_code = Helper.clean_code(strip_verdict_tag(context[AgentType.CODER]))
    context[AgentType.CODER] = cleaned_code

    if not cleaned_code.strip():
        logger.warning(
            "⚠️ Шаг 6: после очистки кода не осталось (только markdown/вердикт)."
        )
        attempts += 1
        context[AgentType.COMPILER] = "Код пустой после очистки от markdown."
        if attempts >= MAX_REVIEW_ATTEMPTS:
            logger.error(
                "❌ Превышено максимальное количество правок кода (пустой код)."
            )
            return LIMIT_EXIT, attempts
        return 5, attempts

    compile_ok, compile_errors = check_code_syntax(cleaned_code)

    if not compile_ok:
        errors_text = compile_errors or "неизвестная ошибка"
        logger.warning(
            "⚠️ Шаг 6: проверка синтаксиса не удалась: %s", preview(errors_text)
        )
        feedback = f"Код не скомпилировался. Ошибки компилятора:\n{errors_text}"
        attempts += 1
        context[AgentType.COMPILER] = feedback
        if attempts >= MAX_REVIEW_ATTEMPTS:
            logger.error(
                "❌ Превышено максимальное количество правок кода (компиляция)."
            )
            return LIMIT_EXIT, attempts  # exit while
        return 5, attempts

    context[AgentType.COMPILER] = ""
    logger.info("✅ Шаг 6: синтаксис в порядке, перехожу к тестированию.")
    # attempts НЕ сбрасываем: иначе при вечном REJECTED от тестировщика
    # счётчик правок кода каждый раз обнуляется и лимит никогда не сработает.
    return 7, attempts


def save_results(context: dict) -> str | None:
    """Сохраняет ТЗ и код в файлы. Вызывается всегда, даже при падении."""
    project_dir = os.getenv("PROJECT_DIR", r"C:\Work\Source-NSU\ML\ResultSoft")
    try:
        os.makedirs(project_dir, exist_ok=True)

        tz_text = context.get(AgentType.SPEC_WRITER) or ""
        code_text = context.get(AgentType.CODER) or ""
        code_ext = CODE_EXTENSIONS.get(TARGET_LANG, "txt")

        with open(os.path.join(project_dir, "TZ.txt"), "w", encoding="utf-8") as f:
            f.write(tz_text)

        with open(
            os.path.join(project_dir, f"result.{code_ext}"), "w", encoding="utf-8"
        ) as f:
            f.write(code_text)

        feedback_text = context.get(AgentType.COMPILER) or context.get(AgentType.TESTER)
        if feedback_text:
            with open(
                os.path.join(project_dir, "Review_Feedback.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(feedback_text)

        logger.info("✅ Проект сохранён в %s", project_dir)
        logger.info("📄 ТЗ: %s", os.path.join(project_dir, "TZ.txt"))
        logger.info("📄 Код: %s", os.path.join(project_dir, f"result.{code_ext}"))
        if feedback_text:
            logger.info(
                "📄 Замечания: %s", os.path.join(project_dir, "Review_Feedback.txt")
            )
        return project_dir
    except OSError:
        # logger.exception сам приложит traceback — руками его собирать не нужно.
        logger.exception("❌ Не удалось сохранить результаты в %s", project_dir)
        return None


async def main():
    # Логирование настраивается один раз здесь: дальше все модули просто
    # пишут в свой logger = logging.getLogger(__name__). Заодно setup_logging()
    # переводит консоль в UTF-8 — сообщения содержат эмодзи-маркеры этапов
    # (📋 Шаг 2, 💻 Шаг 5, ...), а cp1251 их не кодирует.
    setup_logging()

    # ollama = AsyncOllamaClient()
    cloud = CloudAPIProvider(model_name="deepseek-ai/DeepSeek-V4-Flash")
    # 1 Пишу своими словами ТЗ
    # 2 SPEC_WRITER_PROMPT составляет ТЗ
    # 3 SPEC_REVIEWER_PROMPT согласовывает ТЗ, если ег
    # 4 ARHITEKTOR_PROMPT проектирует архитектуру
    # 5 CODER_PROMPT пишет код
    # 6 TESTER_PROMPT проверяет код
    # 7 COMPILER_AGENT_PROMPT запускает компилятор

    agents = create_agents(llm_client=cloud)
    step = 2
    context = {"user_idea": MY_PROMPT}
    # Два независимых счетчика: правки ТЗ и правки кода
    spec_attempts = 0
    code_attempts = 0
    started_at = time.perf_counter()

    logger.info(
        "🚀 Старт пайплайна: модель=%s, целевой язык=%s.",
        getattr(cloud, "model_name", type(cloud).__name__),
        TARGET_LANG,
    )
    if not context["user_idea"]:
        logger.warning(
            "⚠️ Идея пользователя пуста — ТЗ может получиться бессодержательным."
        )

    try:
        while step <= 7:
            if step == 2:
                step = await process_step_2_spec_writer(
                    agents, context
                )  # Формирует ТЗ для проверки
            elif step == 3:
                step, spec_attempts = await process_step_3_spec_reviewer(  # Проверка
                    agents, context, spec_attempts
                )
            elif step == 4:
                step = await process_step_4_arhitektor(
                    agents, context
                )  # Формирует архитектуру для проверки
            elif step == 5:
                step = await process_step_5_coder(agents, context)
            elif step == 6:
                step, code_attempts = await process_step_6_compiler(
                    agents, context, code_attempts
                )  # -> 5 || 7 || finish
            elif step == 7:
                step, code_attempts = await process_step_7_tester(
                    agents, context, code_attempts
                )  # -> 8 || -> 5
        if step == LIMIT_EXIT:
            logger.warning(
                "⚠️ Пайплайн остановлен по лимиту попыток "
                "(правок ТЗ: %d, правок кода: %d). Сохраняю текущий результат.",
                spec_attempts,
                code_attempts,
            )
        elif step == FINISH_OK:
            logger.info("✅ Пайплайн завершён успешно: код согласован тестировщиком.")
    except Exception:
        # Падение внутри шага: пишем traceback и всё равно сохраняем артефакты,
        # накопленные в context (finally ниже).
        logger.exception("❌ Пайплайн упал с ошибкой (шаг %d).", step)
        raise
    finally:
        # Сохраняем всегда, чтобы не потерять ТЗ и код при падении
        save_results(context)

    logger.info(
        "🎉 Работа завершена. Итоговый шаг: %d, время: %.2f сек.",
        step,
        time.perf_counter() - started_at,
    )
    log_file = get_log_file_path()
    if log_file:
        logger.info("📄 Подробный лог: %s", log_file)


if __name__ == "__main__":
    asyncio.run(main())
