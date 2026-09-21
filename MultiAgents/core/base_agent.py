import logging
import time

import tiktoken
import os

from core.base_llm import BaseLLM
from core.logging_setup import preview
from utils.progress_spinner import notify as spinner_notify

# Принудительно заставляем tiktoken не лезть в сеть, если файл уже скачан
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"

logger = logging.getLogger(__name__)

SAFE_LIMIT = 2000


class BaseAgent:
    """Базовый класс агента, работающий с асбтрактной LLM"""

    def __init__(
        self, name_agent: str, role_prompt: str, llm: BaseLLM, context_limit: int = 8192
    ):
        self.name = name_agent  # Просто имя агента
        self.role_prompt = role_prompt  # роль агента
        self.llm = llm  # LLM

    async def execute_task(self, prompt: str, task_type: str = "chat") -> str | None:
        """Выполняет задачу, передавая ее llm"""
        # context_limit = await self.llm._get_context_limit(task_type)
        # if not self.validate_prompt(prompt, context_limit):
        #     raise ValueError(
        #         f"Ошибка: Промпт для агента '{self.name}' слишком огромный ({self.count_tokens(self.role_prompt + prompt)} токенов)! "
        #         f"Он превышает безопасный лимит контекста ({context_limit - SAFE_LIMIT} токенов)."
        #     )
        model = getattr(self.llm, "model_name", type(self.llm).__name__)
        logger.info(
            "🤖 Агент «%s»: старт задачи (task_type=%s, модель=%s).",
            self.name,
            task_type,
            model,
        )
        logger.debug("🤖 Агент «%s»: system-промпт: %s", self.name, preview(self.role_prompt))
        logger.debug("🤖 Агент «%s»: user-промпт: %s", self.name, preview(prompt))
        spinner_notify(self.name, f"Агент «{self.name}» выполняет задачу…")

        started_at = time.perf_counter()
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.role_prompt,
                task_type=task_type,
            )
        except Exception:
            # logger.exception сам приложит traceback — руками его собирать не нужно.
            elapsed = time.perf_counter() - started_at
            logger.exception(
                "❌ Агент «%s»: вызов LLM упал за %.2f сек (task_type=%s).",
                self.name,
                elapsed,
                task_type,
            )
            return None

        elapsed = time.perf_counter() - started_at
        if not response:
            logger.warning(
                "⚠️ Агент «%s»: пустой ответ LLM за %.2f сек (task_type=%s).",
                self.name,
                elapsed,
                task_type,
            )
            return None

        logger.info(
            "✅ Агент «%s»: ответ получен (%d симв.) за %.2f сек.",
            self.name,
            len(response),
            elapsed,
        )
        logger.debug("🤖 Агент «%s»: ответ: %s", self.name, preview(response))
        return response

        """
        architect = BaseAgent(
            name="Архитектор",
            role_prompt="Ты C# системный архитектор. Твоя задача — проектировать структуру классов.",
            llm=llm_provider,
        )
        """

    def count_tokens(sels, text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def validate_prompt(self, prompt: str, context_limit: int) -> bool:
        # Считаем токены системных инструкций + самого запроса
        total_text = self.role_prompt + prompt
        prompt_tokens = self.count_tokens(total_text)

        # Задаем безопасный порог (оставляем минимум 2000 токенов на генерацию ответа модели)
        available_space = context_limit - SAFE_LIMIT

        logger.debug(
            "📊 [%s] Размер запроса: %d токенов. Доступно: %d",
            self.name,
            prompt_tokens,
            available_space,
        )

        if prompt_tokens > available_space:
            logger.warning(
                "⚠️ [%s] Промпт не влезает в контекст: %d > %d токенов.",
                self.name,
                prompt_tokens,
                available_space,
            )
            return False
        return True
