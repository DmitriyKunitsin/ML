import asyncio
import re, os
from enum import Enum
from providers.ollama_providers import AsyncOllamaClient
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

MAX_REVIEW_ATTEMPTS = 300  # Максимальное количество правок


class AgentType(str, Enum):
    """Строгий перечень типов агентов для предотвращения опечаток."""

    SPEC_WRITER = "spec_writer"
    SPEC_REVIEWER = "spec_reviewer"
    ARHITEKTOR = "arhitektor"
    CODER = "coder"
    TESTER = "tester"
    COMPILER = "compiler"
    FEEDBACK = "spec_feedback"


def create_agents(ollama_client: AsyncOllamaClient) -> dict[str, BaseAgent]:
    """Фабрика для создания и инициализации всех агентов системы."""
    return {
        AgentType.SPEC_WRITER: BaseAgent(  # Шаг 2
            name_agent="Системный аналитик",
            role_prompt=SPEC_WRITER_PROMPT,
            llm=ollama_client,
        ),
        AgentType.SPEC_REVIEWER: BaseAgent(  # Шаг 3
            name_agent="Главный валидатор",
            role_prompt=SPEC_REVIEWER_PROMPT,
            llm=ollama_client,
        ),
        AgentType.ARHITEKTOR: BaseAgent(  # Шаг 4
            name_agent="Архитектор",
            role_prompt=ARHITEKTOR_PROMPT,
            llm=ollama_client,
        ),
        AgentType.CODER: BaseAgent(  # Шаг 5
            name_agent="Программист",
            role_prompt=CODER_PROMPT,
            llm=ollama_client,
        ),
        AgentType.TESTER: BaseAgent(  # Шаг 6
            name_agent="Тестировщик",
            role_prompt=TESTER_PROMPT,
            llm=ollama_client,
        ),
        AgentType.COMPILER: BaseAgent(  # Шаг 7
            name_agent="Компилятор",
            role_prompt=COMPILER_AGENT_PROMPT,
            llm=ollama_client,
        ),
    }


# Вспомогательная функция для парсинга вердикта
def parse_verdict(response_text: str) -> tuple[str, str]:
    """
    Ищет тег <verdict> в тексте.
    Возвращает кортеж: (статус, чистый_текст_ответа)
    """
    match = re.search(
        r"<verdict>(APPROVED|REJECTED)</verdict>", response_text, re.IGNORECASE
    )
    if match:
        status = match.group(1).upper()
        # Отрезаем сам тег вердикта из фидбека для чистоты
        clean_feedback = re.sub(
            r"<verdict>.*?</verdict>", "", response_text, flags=re.DOTALL
        ).strip()
        return status, clean_feedback

    # Фолбек на случай, если модель забыла тег, но написала ключевое слово
    if "APPROVED" in response_text.upper():
        return "APPROVED", response_text
    return "REJECTED", response_text


# =====================================================================
# МЕТОДЫ ДЛЯ КАЖДОГО ШАГА СТЕЙТ-МАШИНЫ
# =====================================================================
async def process_step_2_spec_writer(
    agents: dict[AgentType, BaseAgent], context: dict
) -> int:
    print("📋 [Шаг 2] Составление ТЗ...")
    context[AgentType.SPEC_WRITER] = await agents[AgentType.SPEC_WRITER].execute_task(
        prompt=f"Составь ТЗ для моей идеи : {context['user_idea']}",
        task_type="review",
    )
    return 3  # next step 3


async def process_step_3_spec_reviewer(
    agents: dict[AgentType, BaseAgent], context: dict, attempts: int
) -> tuple[int, int]:
    print("✔ [Шаг 3] Проверка технического задания...")
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
        print(f"⚠️ТЗ Отклонено. Попытка правки {attempts}/{MAX_REVIEW_ATTEMPTS}")
        print(f"замечания : {feedback[:500]}")
        if attempts >= MAX_REVIEW_ATTEMPTS:
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
    context[AgentType.ARHITEKTOR] = await agents[AgentType.ARHITEKTOR].execute_task(
        prompt=prompt_for_arhi,
        task_type="boss",
    )
    return 5


async def process_step_5_coder(
    agents: dict[AgentType, BaseAgent], context: dict
) -> int:
    print("💻 [Шаг 5] Написание кода программистом...")
    prompt_for_coder = f"Напиши код по архитектуре:\n{context[AgentType.ARHITEKTOR]}\n\nИ ТЗ:\n{context[AgentType.SPEC_WRITER]}"

    if context.get(AgentType.TESTER):  # есть замечания от тестера
        prompt_for_coder += (
            f"\n\nИсправь ошибки из отчета тестировщика:\n{context[AgentType.TESTER]}"
        )
    if context.get(AgentType.COMPILER):  # есть замечания от компилера
        prompt_for_coder += (
            f"\n\nИсправь ошибки из отчета компилятора:\n{context[AgentType.COMPILER]}"
        )

    code = await agents[AgentType.CODER].execute_task(
        prompt=prompt_for_coder,
        task_type="code",
    )
    context[AgentType.CODER] = code
    return 6


async def process_step_6_tester(
    agents: dict[AgentType, BaseAgent], context: dict, attempts: int
) -> tuple[int, int]:
    print("🧪 [Шаг 6] Тестирование кода...")

    tester_prompt = f"Протестируй код :\n{context[AgentType.CODER]}\n\nТехническое задание :\n{context[AgentType.SPEC_WRITER]}"

    tester_response = await agents[AgentType.TESTER].execute_task(
        prompt=tester_prompt, task_type="review"
    )

    status, feedback = parse_verdict(tester_response)

    if status == "APPROVED":
        print("💚 Код Успешно согласован!")
        context[AgentType.TESTER] = ""
        return 7, 0  # Идем дальше
    else:
        attempts += 1
        print(f"⚠️КОД Отклонен. Попытка правки {attempts}/{MAX_REVIEW_ATTEMPTS}")
        print(f"замечания : {feedback[:500]}")
        if attempts >= MAX_REVIEW_ATTEMPTS:
            print("❌ Превышено максимальное количество правок ТЗ!")
            return 8, attempts  # exit while
        context[AgentType.TESTER] = feedback
        return 5, attempts  # next step 5


async def process_step_7_compiler(
    agents: dict[AgentType, BaseAgent], context: dict, attempts: int
) -> tuple[int, int]:
    print("🔧 [Шаг 7] Компиляция...")

    if context.get(AgentType.CODER):
        compile_ok, compile_errors = Helper().validate_syntax_python(
            context[AgentType.CODER]
        )
    else:
        print("❌ Отсутсвие кода")
        feedback = f"Код отсутствует"
        attempts += 1
        context[AgentType.COMPILER] = feedback
        return 5, attempts

    if not compile_ok:
        print(f"❌ Компиляция не удалась, ошибки:\n{compile_errors[:500]}...")
        feedback = f"Код не скомпилировался. Ошибки компилятора:\n{compile_errors}"
        attempts += 1
        context[AgentType.COMPILER] = feedback
        return 5, attempts
    else:
        context[AgentType.COMPILER] = ""
        print("✅ Компиляция успешна! Записываю код в файл")
        return 8, 0


async def main():
    ollama = AsyncOllamaClient()

    # 1 Пишу своими словами ТЗ
    # 2 SPEC_WRITER_PROMPT составляет ТЗ
    # 3 SPEC_REVIEWER_PROMPT согласовывает ТЗ, если ег
    # 4 ARHITEKTOR_PROMPT проектирует архитектуру
    # 5 CODER_PROMPT пишет код
    # 6 TESTER_PROMPT проверяет код
    # 7 COMPILER_AGENT_PROMPT запускает компилятор

    agents = create_agents(ollama_client=ollama)
    step = 2
    context = {"user_idea": MY_PROMPT}
    # Счетчик итераций
    review_attempts = 0
    while step <= 7:
        if step == 2:
            step = await process_step_2_spec_writer(
                agents, context
            )  # Формирует ТЗ для проверки
        elif step == 3:
            step, review_attempts = await process_step_3_spec_reviewer(  # Проверка
                agents, context, review_attempts
            )
        elif step == 4:
            step = await process_step_4_arhitektor(
                agents, context
            )  # Формирует архитектуру для проверки
        elif step == 5:
            step, review_attempts = await process_step_5_coder(agents, context)
        elif step == 6:
            step, review_attempts = await process_step_6_tester(
                agents, context, review_attempts
            )  # -> 7 || -> 5
        elif step == 7:
            step, review_attempts = await process_step_7_compiler(
                agents, context, review_attempts
            )  # -> 5 || finish

    # --- Шаг 4: Сохраняем всё в файлы ---
    project_dir = "/mnt/c/Work/Source-NSU/Arduino/llama3.1_8b_Project"
    os.makedirs(project_dir, exist_ok=True)

    with open(f"{project_dir}/TZ.txt", "w", encoding="utf-8") as f:
        f.write(context[AgentType.SPEC_WRITER])

    with open(f"{project_dir}/clock.ino", "w", encoding="utf-8") as f:
        f.write(context[AgentType.CODER])

    print(f"\n✅ Проект сохранён в {project_dir}")
    print(f"📄 ТЗ: {project_dir}/TZ.txt")
    print(f"📄 Код: {project_dir}/clock.ino")
    if os.path.exists(f"{project_dir}/Review_Feedback.txt"):
        print(f"📄 Замечания: {project_dir}/Review_Feedback.txt")
    print("\n🎉 Работа завершена!")


if __name__ == "__main__":
    asyncio.run(main())
