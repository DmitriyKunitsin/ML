import asyncio
import re, os, sys, shutil
from enum import Enum
from providers.ollama_providers import AsyncOllamaClient
from providers.cloud_api_providers import CloudAPIProvider
from core.base_agent import BaseAgent
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

MAX_REVIEW_ATTEMPTS = 5  # Максимальное количество правок (отдельно для ТЗ и для кода)
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
            print("⚠️ avr-g++ не найден, пропускаю проверку Arduino-кода.")
            return True, None
        return Helper.compile_arduino_sketch(code)
    return Helper.validate_syntax_python(code)


# =====================================================================
# МЕТОДЫ ДЛЯ КАЖДОГО ШАГА СТЕЙТ-МАШИНЫ
# =====================================================================
async def process_step_2_spec_writer(
    agents: dict[AgentType, BaseAgent], context: dict
) -> int:
    print("📋 [Шаг 2] Составление ТЗ...")
    spec_text = await agents[AgentType.SPEC_WRITER].execute_task(
        prompt=f"Составь ТЗ для моей идеи : {context['user_idea']}",
        task_type="review",
    )
    if not spec_text:
        print("❌ Аналитик не вернул ТЗ (пустой ответ LLM). Прерываю пайплайн.")
        return 8
    context[AgentType.SPEC_WRITER] = spec_text
    return 3  # next step 3


async def process_step_3_spec_reviewer(
    agents: dict[AgentType, BaseAgent], context: dict, attempts: int
) -> tuple[int, int]:
    print("✔ [Шаг 3] Проверка технического задания...")
    if not context.get(AgentType.SPEC_WRITER):
        print("❌ Нет ТЗ для проверки. Прерываю пайплайн.")
        return 8, attempts

    prompt_for_review = (
        f"Проверь следующее техническое задание :\n\n{context[AgentType.SPEC_WRITER]}"
    )
    if context.get(AgentType.FEEDBACK):
        prompt_for_review += f"\n\nПредыдущие замечания , которые должны быть исправлены : \n{context[AgentType.FEEDBACK]}"

    reviewer_response = await agents[AgentType.SPEC_REVIEWER].execute_task(
        prompt=prompt_for_review,
        task_type="review",
    )

    status, feedback = parse_verdict(reviewer_response)

    if status == "APPROVED":
        print("💚 ТЗ Успешно согласовано!")
        context[AgentType.FEEDBACK] = ""  # clear feedback
        return 4, 0  # Идем к Архитектору
    else:
        attempts += 1
        print(f"⚠️ТЗ Отклонено. Попытка правки {attempts}/{MAX_SPEC_ATTEMPTS}")
        print(f"замечания : {feedback[:500]}")
        if attempts >= MAX_SPEC_ATTEMPTS:
            print("❌ Превышено максимальное количество правок ТЗ!")
            return 8, attempts  # exit while
        context[AgentType.FEEDBACK] = feedback
        context["user_idea"] = (
            f"Переделай техническое задание. Замечания Валидатора : \n{feedback}\n\nОригинальная идея : {MY_PROMPT}"
        )
        return 2, attempts  # next step 2


async def process_step_4_arhitektor(
    agents: dict[AgentType, BaseAgent], context: dict
) -> int:
    print("📐 [Шаг 4] Проектирование архитектуры...")
    prompt_for_arhi = f"Спроектируй архитектуру согласно данному техническому заданию : \n\n{context[AgentType.SPEC_WRITER]}"
    architecture = await agents[AgentType.ARHITEKTOR].execute_task(
        prompt=prompt_for_arhi,
        task_type="boss",
    )
    if not architecture:
        print("❌ Архитектор не вернул архитектуру (пустой ответ LLM). Прерываю пайплайн.")
        return 8
    context[AgentType.ARHITEKTOR] = architecture
    return 5


async def process_step_5_coder(
    agents: dict[AgentType, BaseAgent], context: dict
) -> int:
    print("💻 [Шаг 5] Написание кода программистом...")

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
    if context.get(AgentType.COMPILER):  # есть замечания от компилера
        prompt_for_coder += (
            f"\n\nИсправь ошибки из отчета компилятора:\n{context[AgentType.COMPILER]}"
        )
    if context.get(AgentType.CODER):  # есть предыдущий код
        fence = "cpp" if TARGET_LANG == "cpp" else "python"
        prompt_for_coder += (
            f"\n\nТвой предыдущий код :\n```{fence}\n{context[AgentType.CODER]}\n```"
        )

    code = await agents[AgentType.CODER].execute_task(
        prompt=prompt_for_coder,
        task_type="code",
    )
    if not code:
        print("❌ Программист не вернул код (пустой ответ LLM). Прерываю пайплайн.")
        return 8
    # CODER_PROMPT обязывает модель выводить <verdict>...</verdict>:
    # без отрезания тега код не пройдёт проверку синтаксиса на шаге 6.
    context[AgentType.CODER] = strip_verdict_tag(code)
    return 6


async def process_step_7_tester(
    agents: dict[AgentType, BaseAgent], context: dict, attempts: int
) -> tuple[int, int]:
    print("🧪 [Шаг 7] Тестирование кода...")

    if not context.get(AgentType.CODER):
        print("❌ Нет кода для тестирования. Прерываю пайплайн.")
        return 8, attempts

    tester_prompt = f"Протестируй код :\n{context[AgentType.CODER]}\n\nТехническое задание :\n{context[AgentType.SPEC_WRITER]}"

    tester_response = await agents[AgentType.TESTER].execute_task(
        prompt=tester_prompt, task_type="review"
    )

    status, feedback = parse_verdict(tester_response)

    if status == "APPROVED":
        print("💚 Код Успешно согласован!")
        context[AgentType.TESTER] = ""
        return FINISH_OK, 0  # Успешный финиш
    else:
        attempts += 1
        print(f"⚠️КОД Отклонен. Попытка правки {attempts}/{MAX_REVIEW_ATTEMPTS}")
        print(f"замечания : {feedback[:500]}")
        if attempts >= MAX_REVIEW_ATTEMPTS:
            print("❌ Превышено максимальное количество правок КОДА!")
            return 8, attempts  # exit while
        context[AgentType.TESTER] = feedback
        return 5, attempts  # next step 5


async def process_step_6_compiler(
    agents: dict[AgentType, BaseAgent], context: dict, attempts: int
) -> tuple[int, int]:
    print("🔧 [Шаг 6] Компиляция...")

    if not context.get(AgentType.CODER):
        print("❌ Отсутствие кода")
        attempts += 1
        context[AgentType.COMPILER] = "Код отсутствует"
        if attempts >= MAX_REVIEW_ATTEMPTS:
            print("❌ Превышено максимальное количество правок КОДА (нет кода)!")
            return 8, attempts  # exit while
        return 5, attempts

    cleaned_code = Helper.clean_code(
        strip_verdict_tag(context[AgentType.CODER])
    )
    context[AgentType.CODER] = cleaned_code

    if not cleaned_code.strip():
        print("❌ После очистки кода не осталось.")
        attempts += 1
        context[AgentType.COMPILER] = "Код пустой после очистки от markdown."
        if attempts >= MAX_REVIEW_ATTEMPTS:
            print("❌ Превышено максимальное количество правок КОДА (пустой код)!")
            return 8, attempts
        return 5, attempts

    compile_ok, compile_errors = check_code_syntax(cleaned_code)

    if not compile_ok:
        errors_text = compile_errors or "неизвестная ошибка"
        print(f"❌ Компиляция не удалась, ошибки:\n{errors_text[:500]}...")
        feedback = f"Код не скомпилировался. Ошибки компилятора:\n{errors_text}"
        attempts += 1
        context[AgentType.COMPILER] = feedback
        if attempts >= MAX_REVIEW_ATTEMPTS:
            print("❌ Превышено максимальное количество правок КОДА (компиляция)!")
            return 8, attempts  # exit while
        return 5, attempts

    context[AgentType.COMPILER] = ""
    print("✅ Компиляция успешна!")
    # attempts НЕ сбрасываем: иначе при вечном REJECTED от тестировщика
    # счётчик правок кода каждый раз обнуляется и лимит никогда не сработает.
    return 7, attempts


def save_results(context: dict) -> str | None:
    """Сохраняет ТЗ и код в файлы. Вызывается всегда, даже при падении."""
    project_dir = os.getenv(
        "PROJECT_DIR", r"C:\Work\Source-NSU\Arduino\llama3.1_8b_Project"
    )
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

        feedback_text = context.get(AgentType.COMPILER) or context.get(
            AgentType.TESTER
        )
        if feedback_text:
            with open(
                os.path.join(project_dir, "Review_Feedback.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(feedback_text)

        print(f"\n✅ Проект сохранён в {project_dir}")
        print(f"📄 ТЗ: {project_dir}\\TZ.txt")
        print(f"📄 Код: {project_dir}\\result.{code_ext}")
        if feedback_text:
            print(f"📄 Замечания: {project_dir}\\Review_Feedback.txt")
        return project_dir
    except OSError as e:
        print(f"❌ Не удалось сохранить результаты: {e}")
        return None


async def main():
    # UTF-8 для консоли Windows (иначе UnicodeEncodeError на эмодзи)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

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
            print(
                f"\n⚠️ Пайплайн остановлен по лимиту попыток "
                f"(правок ТЗ: {spec_attempts}, правок кода: {code_attempts}). "
                f"Сохраняю текущий результат."
            )
        elif step == FINISH_OK:
            print("\n✅ Пайплайн завершён успешно: код согласован тестировщиком.")
    finally:
        # Сохраняем всегда, чтобы не потерять ТЗ и код при падении
        save_results(context)

    print("\n🎉 Работа завершена!")


if __name__ == "__main__":
    asyncio.run(main())
