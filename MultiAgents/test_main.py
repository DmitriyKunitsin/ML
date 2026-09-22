import asyncio
import logging
import time

import re, os, shutil
from enum import Enum
from providers.ollama_providers import AsyncOllamaClient
from providers.cloud_api_providers import CloudAPIProvider
from core.base_agent import BaseAgent
from core.logging_setup import (
    close_run_loggers,
    get_log_file_path,
    get_run_log_dir,
    preview,
    setup_logging,
)
from config.prompts import (
    ARHITEKTOR_PROMPT,
    CODER_PROMPT,
    TESTER_PROMPT,
    COMPILER_AGENT_PROMPT,
    SPEC_WRITER_PROMPT,
    SPEC_REVIEWER_PROMPT,
    MY_PROMPT,
    PROMPTS_BY_TARGET,
)
from utils.helpers import Helper

# Логгер на модуль: в каждой записи видно, откуда она пришла (%(name)s).
logger = logging.getLogger(__name__)

# Общий анимированный статус «Работает…» — приём от пользователя: без него
# экран замирает на время долгого запроса к LLM и кажется, что всё зависло.
# Спиннер живёт в фоновом потоке и пишет в консоль (stdout), см. utils/progress_spinner.
from utils.progress_spinner import notify as spinner_notify
from utils.progress_spinner import stop as spinner_stop

# Тесты и CI не выводят статус: спиннер пишет '\r' — мусор в протоколе.
# Отключить можно через PROGRESS_SPINNER=0 (например, для CI).
PROGRESS_SPINNER = os.getenv("PROGRESS_SPINNER", "1").lower() not in ("0", "false", "off")

MAX_REVIEW_ATTEMPTS = 15  # Максимальное количество правок кода (эскалация человеку)
MAX_SYNTAX_ATTEMPTS = 10  # Лимит ПРОВЕРОК СИНТАКСИСА (шаг 6) — отдельный от ревью.
# Раньше синтаксис и ревью делили один счётчик: 15 неудачных компиляций
# «съедали» весь лимит правок кода и наоборот. Теперь это независимые лимиты.
MAX_SPEC_ATTEMPTS = 5  # Максимальное количество правок ТЗ

# --- Борьба с «циклом смерти»: изоляция контекста между итерациями ---
# Кодеру передаётся НЕ вся история правок, а только последние N фидбэков
# (кольцевой буфер). Иначе к середине цикла его контекст — это простыня из
# старых замечаний, и модель «захлёбывается» (генерирует невалидный код).
FEEDBACK_HISTORY_LIMIT = 3
# Если подряд приходит N одинаковых ошибок (SyntaxError) — код деградирует,
# а не чинится. Переключаем кодера на режим «перепиши с нуля по ТЗ».
RESET_THRESHOLD = 3
# Доля доступного контекста (после SAFE_LIMIT), выделяемая на «предыдущий код».
# Остальное — ТЗ, архитектура и фидбеки. Предотвращает переполнение контекста.
CODE_TOKEN_BUDGET_PCT = 0.35

# Целевой язык генерируемого кода: "python" (obsidian-скрипт из MY_PROMPT)
# или "cpp" (Arduino-скетч). От него зависит способ проверки синтаксиса и
# имя файла результата. Меняется через переменную окружения TARGET_LANG.
TARGET_LANG = os.getenv("TARGET_LANG", "python").lower()
CODE_EXTENSIONS = {"python": "py", "cpp": "ino"}

# Резерв контекста под генерацию ответа модели (аналогично core.base_agent.SAFE_LIMIT).
SAFE_LIMIT = 2000

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
    """Фабрика для создания и инициализации всех агентов системы.

    Роли выбираются из PROMPTS_BY_TARGET по TARGET_LANG: раньше промпты были
    жёстко прибиты под Arduino (compiler/avr-g++), хотя дефолтный язык —
    python. Теперь кодер/компилятор получают профиль под целевой язык.
    """
    prompts = PROMPTS_BY_TARGET.get(TARGET_LANG, PROMPTS_BY_TARGET["python"])
    return {
        AgentType.SPEC_WRITER: BaseAgent(  # Шаг 2
            name_agent="Системный аналитик",
            role_prompt=prompts["spec_writer"],
            llm=llm_client,
        ),
        AgentType.SPEC_REVIEWER: BaseAgent(  # Шаг 3
            name_agent="Главный валидатор",
            role_prompt=prompts["spec_reviewer"],
            llm=llm_client,
        ),
        AgentType.ARHITEKTOR: BaseAgent(  # Шаг 4
            name_agent="Архитектор",
            role_prompt=prompts["arhitektor"],
            llm=llm_client,
        ),
        AgentType.CODER: BaseAgent(  # Шаг 5
            name_agent="Программист",
            role_prompt=prompts["coder"],
            llm=llm_client,
        ),
        AgentType.TESTER: BaseAgent(  # Шаг 6
            name_agent="Тестировщик",
            role_prompt=prompts["tester"],
            llm=llm_client,
        ),
        AgentType.COMPILER: BaseAgent(  # Шаг 7
            name_agent="Компилятор",
            role_prompt=prompts["compiler"],
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
    обычно выглядит как «код + <verdict>APPROVED</verdict>». Такой текст нельзя
    подавать на валидацию синтаксиса: тег не является кодом.

    Модель иногда нарушает порядок и ставит вердикт ПЕРЕД кодом
    («<verdict>REJECTED</verdict> вот код...»). Тогда чистый код лежит после
    тега — берём его. При нескольких тегах оставляем последний блок кода.
    Если тег один и код идёт до него — берём дотеговую часть.
    """
    if not response_text:
        return ""

    tag_pattern = r"<\s*verdict\s*>\s*(?:APPROVED|REJECTED)\s*<\s*/\s*verdict\s*>"
    matches = list(re.finditer(tag_pattern, response_text, re.IGNORECASE | re.DOTALL))
    if not matches:
        return response_text.strip()

    last = matches[-1]
    before = response_text[: last.start()].strip()
    after = response_text[last.end() :].strip()

    # Если после последнего тега что-то есть — вероятно, это запрошенный
    # правкой код/пояснение. В противном случае берём дотеговую часть.
    if before and not _looks_like_explanation(after):
        return before
    return after or before


def _looks_like_explanation(text: str) -> bool:
    """Примерная эвристика: является ли текст после тега «бла-бла», а не кодом."""
    if not text:
        return False
    # Код почти всегда содержит рав или фигурные скобки/отступы, пояснение — нет.
    code_hints = ("def ", "class ", "import ", "=", "```", "{", "}", "return ")
    lowered = text.strip().lower()
    if any(hint in lowered for hint in code_hints):
        return False
    # Слишком длинный «код без признаков кода» — это пояснение.
    return len(text) < 400


# =====================================================================
# Управление контекстом цикла правок (борьба с «контекстной помойкой»)
# =====================================================================
def feedback_history_init(context: dict) -> None:
    """Гарантирует наличие кольцевого буфера последних замечаний в context.

    Ключ «feedback_history» не совпадает ни с одним AgentType, поэтому
    не пересекается с данными ТЗ/кода/архитектуры.
    """
    context.setdefault("feedback_history", [])


def feedback_history_push(context: dict, source: str, text: str) -> None:
    """Добавляет замечание в кольцевой буфер, ограничивая его глубину.

    Это и есть «изоляция контекста между итерациями»: кодер видит НЕ всю
    историю замечаний, а только последние FEEDBACK_HISTORY_LIMIT штук.
    Многолетние «старые» замечания не накапливаются и не загрязняют промпт.
    """
    if not text or not text.strip():
        return
    feedback_history_init(context)
    history = context["feedback_history"]
    history.append({"source": source, "text": text.strip()})
    if len(history) > FEEDBACK_HISTORY_LIMIT:
        # Отбрасываем самую старую запись: счётчик неизменен внутри буфера.
        del history[: len(history) - FEEDBACK_HISTORY_LIMIT]


def feedback_snapshot(context: dict) -> str:
    """Форматирует последние замечания для включения в промпт кодера.

    Каждый фидбек снабжается префиксом источника («тестировщик» /
    «компилятор»), чтобы кодер понимал, что чинить. Возвращает пустую
    строку, если замечаний нет (или фидбеки почищены после APPROVED).
    """
    history = context.get("feedback_history") or []
    if not history:
        return ""
    blocks = []
    for entry in history:
        source_label = (
            "отчет тестировщика"
            if entry["source"] == AgentType.TESTER
            else "отчет компилятора"
        )
        blocks.append(f"Исправь ошибки из {source_label}:\n{entry['text']}")
    return "\n\n".join(blocks)


def _is_negative_feedback(text: str) -> bool:
    """Является ли фидбек «отрицательным» (код не принят и требует правок).

    Фиксирует маркеры, которыми пайплайн и промпты помечают неудачу:
    вердикт REJECTED, ошибки компиляции/синтаксиса, «отклон», «не скомпилировался».
    """
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "rejected",
            "<verdict>rejected</verdict>",
            "syntaxerror",
            "не скомпилировался",
            "не удалось",
            "ошибки компилятора",
        )
    )


def _should_rewrite_from_scratch(context: dict) -> bool:
    """Пора ли переписать код с нуля вместо рискованного «латания».

    Срабатывает, если в последних ``RESET_THRESHOLD`` фидбэках подряд НЕТ
    успеха — то есть код систематически не принимается (компиляция или ревью),
    особенно когда фидбеки повторяются дословно (кодер «топчется на месте»).

    Раньше критерий был узким (три подряд ``SyntaxError``), из-за чего вечный
    REJECTED от тестировщика НЕ детектился как деградация, и пайплайн молотил
    цикл до исчерпания лимита. Расширен на любой отрицательный фидбек.
    """
    history = context.get("feedback_history") or []
    if len(history) < RESET_THRESHOLD:
        return False

    seq = [entry.get("text", "") for entry in history[-RESET_THRESHOLD:]]
    # Три подряд отрицательных фидбека — код не чинится, пора переписать.
    return all(_is_negative_feedback(text) for text in seq)


def truncate_code_for_context(coder, code: str, max_tokens: int) -> str:
    """Обрезает «предыдущий код» по токенному бюджету, если он не влезает.

    Не отдаём модели весь огромный код из прошлой итерации — это переполняет
    контекст и «токсично». Показываем хвост (последние строки: в них обычно
    и сидят исправляемые конструкции) с явной пометкой обрезки.
    """
    if not code:
        return ""
    try:
        tokens = coder.count_tokens(code)
    except Exception:
        # Не смогли посчитать токены (нет кэша tiktoken?) — режем по символам.
        tokens = len(code) // 3  # грубая оценка: ~3 символа на токен
    if tokens <= max_tokens:
        return code

    # Не пропорция токенов->символов (это слишком грубо и часто даёт расщепление
    # посреди строки). Берём последние max_tokens символов с запасом и затем
    # отрезаем по границе строки.
    tail_chars = max(200, int(len(code) * (max_tokens / max(tokens, 1))))
    tail = code[-tail_chars:].lstrip("\n")
    # Не выходим за границу '```', если обрезка попала в markdown-блок.
    marker = "\n... [предыдущий код обрезан по лимиту токенов, показан хвост] ...\n"
    return marker + tail


async def _code_context_budget(coder) -> int:
    """Сколько токенов можно отдать под «предыдущий код» в промпте кодера."""
    try:
        limit = await coder.llm._get_context_limit("code")
    except Exception:
        limit = 8192
    return max(500, int((limit - SAFE_LIMIT) * CODE_TOKEN_BUDGET_PCT))


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
    feedback_history_init(context)

    if context.get(AgentType.CODER):
        # Отдаём модели уже очищенный от markdown и вердикта код
        context[AgentType.CODER] = Helper.clean_code(
            strip_verdict_tag(context[AgentType.CODER])
        )

    prompt_for_coder = (
        f"Напиши код по архитектуре:\n{context[AgentType.ARHITEKTOR]}"
        f"\n\nИ ТЗ:\n{context[AgentType.SPEC_WRITER]}"
    )

    # Изоляция контекста: кодер видит ТОЛЬКО последние замечания из
    # feedback_history (кольцевой буфер), а не всю накопленную простыню.
    feedback_text = feedback_snapshot(context)
    if feedback_text:
        # Если много подряд повторяющихся SyntaxError — код деградировал.
        # Переключаемся на «перепиши с нуля по ТЗ» и не показываем старый код.
        if _should_rewrite_from_scratch(context):
            prompt_for_coder += (
                "\n\nКРИТИЧЕСКИ ВАЖНО: предыдущие итерации только ломали код "
                "(повторяющиеся SyntaxError). НЕ пытайся «латать» старый код — "
                "НАПИШИ ПОЛНУЮ РЕАЛИЗАЦИЮ С НУЛЯ по ТЗ и архитектуре выше. "
                "Игнорируй предыдущий код и старые замечания."
            )
            logger.warning(
                "💻 Шаг 5: детектирована деградация кода — просим кодера "
                "переписать с нуля, старый код из контекста исключён."
            )
        else:
            prompt_for_coder += f"\n\n{feedback_text}"
            logger.info("💻 Шаг 5: в промпт добавлены последние замечания ревью.")

    previous_code = context.get(AgentType.CODER)
    if previous_code and not _should_rewrite_from_scratch(context):
        # Отсечение контекста по токенам: не впихиваем весь прошлый код,
        # если он не влезает в бюджет модельки.
        code_budget = await _code_context_budget(agents[AgentType.CODER])
        trimmed_previous = truncate_code_for_context(
            agents[AgentType.CODER], previous_code, code_budget
        )
        fence = "cpp" if TARGET_LANG == "cpp" else "python"
        prompt_for_coder += (
            f"\n\nТвой предыдущий код :\n```{fence}\n{trimmed_previous}\n```"
        )

    logger.debug("💻 Шаг 5: промпт кодера: %s", preview(prompt_for_coder))
    code = await agents[AgentType.CODER].execute_task(
        prompt=prompt_for_coder,
        task_type="code",
    )

    # --- Борьба с «молчаливой обрезкой» ответа (корень цикла 5<->6<->7) ---
    # Провайдер сообщил finish_reason="length" (упёрлись в max_tokens) или
    # текст физически оборван посреди конструкции — такой код нельзя
    # отправлять на проверку: он гарантированно не пройдёт и будет крутить
    # цикл. Переспрашиваем кодера один раз (с компактной директивой).
    coder_meta = getattr(agents[AgentType.CODER], "last_response_meta", None)
    truncated_by_provider = bool(coder_meta and coder_meta.truncated)
    structurally_incomplete = not Helper.is_response_complete(code or "")
    if truncated_by_provider or structurally_incomplete:
        reason = (
            "обрыв по лимиту токенов (finish_reason=length)"
            if truncated_by_provider
            else "текст ответа оборван (нет закрывающего тега/незакрытая конструкция)"
        )
        logger.warning(
            "💻 Шаг 5: ответ кодера неполный (%s). Делаю одну целевую "
            "повторную генерацию с укороченным контекстом.",
            reason,
        )
        code = await agents[AgentType.CODER].execute_task(
            prompt=(
                prompt_for_coder
                + "\n\nВНИМАНИЕ: твой предыдущий ответ ОБОРВАЛСЯ (не был дописан "
                "до конца) и был отброшен целиком. Напиши полный код ЗАНОВО. "
                "Сократи комментарии, не повторяй ТЗ, экономь токены. "
                "Обязательно закончи ответ тегом <verdict>APPROVED</verdict> "
                "или <verdict>REJECTED</verdict>."
            ),
            task_type="code",
        )
        # После повтора перечитываем метаданные: повторный ответ тоже мог
        # упереться в max_tokens (finish_reason=length).
        coder_meta = getattr(agents[AgentType.CODER], "last_response_meta", None)
        if (
            (coder_meta and coder_meta.truncated)
            or not code
            or not Helper.is_response_complete(code or "")
        ):
            logger.error(
                "❌ Программист повторно вернул неполный/пустой код после "
                "уведомления об обрыве. Прерываю пайплайн (лимит: эскалация человеку)."
            )
            return LIMIT_EXIT
    if not code:
        logger.error(
            "❌ Программист не вернул код (пустой ответ LLM). Прерываю пайплайн."
        )
        return LIMIT_EXIT
    # CODER_PROMPT обязывает модель выводить <verdict>...</verdict>:
    # без отрезания тега код не пройдёт проверку синтаксиса на шаге 6.
    cleaned_code = Helper.clean_code(strip_verdict_tag(code))
    context[AgentType.CODER] = cleaned_code

    logger.info(
        "✅ Шаг 5: код получен (%d симв.), перехожу к sandbox-проверке синтаксиса.",
        len(cleaned_code),
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
        context["feedback_history"] = []  # чистим историю правок
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
                "❌ Превышено максимальное количество правок кода (%d). "
                "Эскалирую человеку: сохраняю последнюю версию кода и замечания.",
                MAX_REVIEW_ATTEMPTS,
            )
            return LIMIT_EXIT, attempts  # exit while
        # Замечание попадает в кольцевой буфер (последние N), а не копится
        # бесконечно: кодер на следующей итерации видит только актуальный срез.
        feedback_history_push(context, AgentType.TESTER, feedback)
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
        if attempts >= MAX_SYNTAX_ATTEMPTS:
            logger.error("❌ Превышен лимит проверок синтаксиса (нет кода).")
            return LIMIT_EXIT, attempts  # exit while
        return 5, attempts

    cleaned_code = Helper.clean_code(strip_verdict_tag(context[AgentType.CODER]))
    context[AgentType.CODER] = cleaned_code
    feedback_history_init(context)

    if not cleaned_code.strip():
        logger.warning(
            "⚠️ Шаг 6: после очистки кода не осталось (только markdown/вердикт)."
        )
        attempts += 1
        feedback = "Код пустой после очистки от markdown."
        feedback_history_push(context, AgentType.COMPILER, feedback)
        context[AgentType.COMPILER] = feedback
        if attempts >= MAX_SYNTAX_ATTEMPTS:
            logger.error("❌ Превышен лимит проверок синтаксиса (пустой код).")
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
        feedback_history_push(context, AgentType.COMPILER, feedback)
        context[AgentType.COMPILER] = feedback
        if attempts >= MAX_SYNTAX_ATTEMPTS:
            logger.error("❌ Превышен лимит проверок синтаксиса (компиляция).")
            return LIMIT_EXIT, attempts  # exit while
        return 5, attempts

    context[AgentType.COMPILER] = ""
    logger.info("✅ Шаг 6: синтаксис в порядке, перехожу к тестированию.")
    # attempts НЕ сбрасываем: иначе при вечном REJECTED от тестировщика
    # счётчик правок кода каждый раз обнуляется и лимит никогда не сработает.
    return 7, attempts


def _atomic_write(path: str, content: str) -> None:
    """Пишет файл атомарно: сначала во временный файл, затем переименование.

    Если процесс упадёт посреди записи, целевой файл не останется
    «полупустым» — пайплайн сохраняет артефакты, и читатель не должен
    увидеть оборванный на середине JSON/код.
    """
    import tempfile

    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except Exception:
        # Не оставляем мусорный tmp-файл.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def save_results(context: dict, escalation: bool = False) -> str | None:
    """Сохраняет ТЗ, код и фидбек в файлы. Вызывается всегда, даже при падении.

    ``escalation=True`` — пайплайн остановлен по лимиту правок: пишем
    ``escalation.md`` с пояснением для человека и ссылками на артефакты.
    Записи атомарные (см. ``_atomic_write``).
    """
    project_dir = os.getenv("PROJECT_DIR", r"C:\Work\Source-NSU\ML\ResultSoft")
    try:
        os.makedirs(project_dir, exist_ok=True)

        tz_text = context.get(AgentType.SPEC_WRITER) or ""
        code_text = context.get(AgentType.CODER) or ""
        code_ext = CODE_EXTENSIONS.get(TARGET_LANG, "txt")

        _atomic_write(os.path.join(project_dir, "TZ.txt"), tz_text)
        _atomic_write(os.path.join(project_dir, f"result.{code_ext}"), code_text)

        feedback_text = context.get(AgentType.COMPILER) or context.get(AgentType.TESTER)
        if feedback_text:
            _atomic_write(
                os.path.join(project_dir, "Review_Feedback.txt"), feedback_text
            )

        if escalation:
            reason = feedback_text or (
                "Код так и не был согласован тестировщиком, деталей нет."
            )
            _atomic_write(
                os.path.join(project_dir, "escalation.md"),
                (
                    "# Эскалация человеку\n\n"
                    "Пайплайн исчерпал лимит попыток и НЕ смог согласовать код "
                    "автоматически. Требуется ручное вмешательство.\n\n"
                    "## Причина остановки\n"
                    f"{reason}\n\n"
                    "## Артефакты\n"
                    f"- ТЗ: `TZ.txt`\n"
                    f"- Код: `result.{code_ext}`\n"
                ),
            )

        logger.info("✅ Проект сохранён в %s", project_dir)
        logger.info("📄 ТЗ: %s", os.path.join(project_dir, "TZ.txt"))
        logger.info("📄 Код: %s", os.path.join(project_dir, f"result.{code_ext}"))
        if feedback_text:
            logger.info(
                "📄 Замечания: %s", os.path.join(project_dir, "Review_Feedback.txt")
            )
        if escalation:
            logger.info(
                "🚨 Эскалация человеку: %s",
                os.path.join(project_dir, "escalation.md"),
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
    feedback_history_init(context)
    # Три независимых счётчика: правки ТЗ, проверки синтаксиса (шаг 6) и
    # правки кода/ревью (шаг 7). Раньше синтаксис и ревью делили один лимит:
    # 15 неудачных компиляций «съедали» весь бюджет правок кода и наоборот.
    spec_attempts = 0
    syntax_attempts = 0
    review_attempts = 0
    started_at = time.perf_counter()
    # Приём от пользователя о «зависшем» пайплайне: включаем анимированный
    # статус «Работает…» сразу после старта.
    if PROGRESS_SPINNER:
        spinner_notify("pipeline", "Работает… 0:00")

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
                step, syntax_attempts = await process_step_6_compiler(
                    agents, context, syntax_attempts
                )  # -> 5 || 7 || finish
            elif step == 7:
                step, review_attempts = await process_step_7_tester(
                    agents, context, review_attempts
                )  # -> 8 || -> 5
        if step == LIMIT_EXIT:
            logger.warning(
                "⚠️ Пайплайн остановлен по лимиту попыток "
                "(правок ТЗ: %d, проверок синтаксиса: %d, правок кода: %d). "
                "Сохраняю текущий результат.",
                spec_attempts,
                syntax_attempts,
                review_attempts,
            )
        elif step == FINISH_OK:
            logger.info("✅ Пайплайн завершён успешно: код согласован тестировщиком.")
    except Exception:
        # Падение внутри шага: пишем traceback и всё равно сохраняем артефакты,
        # накопленные в context (finally ниже).
        logger.exception("❌ Пайплайн упал с ошибкой (шаг %d).", step)
        raise
    finally:
        # Сохраняем всегда, чтобы не потерять ТЗ и код при падении.
        # Если пайплайн вышел по лимиту правок — дописываем escalation.md.
        save_results(context, escalation=(step == LIMIT_EXIT))
        # Останавливаем анимацию «Работает…», чтобы финальные сообщения
        # (время, путь к файлу лога) выглядели чисто, без «\r»-хвостов.
        if PROGRESS_SPINNER:
            spinner_stop()

    logger.info(
        "🎉 Работа завершена. Итоговый шаг: %d, время: %.2f сек.",
        step,
        time.perf_counter() - started_at,
    )
    # Логи этого запуска: папка вида logs/2026-..-.._HH-MM-SS с run.log и
    # agents/*.log. Файлы агентов закрываем сейчас (на Windows открытый файл
    # нельзя удалить/переместить), чтобы папка осталась цельной.
    run_dir = get_run_log_dir()
    log_file = get_log_file_path()
    if log_file:
        logger.info("📄 Логи этого запуска: %s", run_dir or os.path.dirname(log_file))
    close_run_loggers()


if __name__ == "__main__":
    asyncio.run(main())
