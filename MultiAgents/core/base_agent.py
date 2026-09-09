import tiktoken
import os

from core.base_llm import BaseLLM

# Принудительно заставляем tiktoken не лезть в сеть, если файл уже скачан
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"

SAFE_LIMIT = 2000


class BaseAgent:
    """Базовый класс агента, работающий с асбтрактной LLM"""

    def __init__(
        self, role_name: str, role_prompt: str, llm: BaseLLM, context_limit: int = 8192
    ):
        self.role_name = role_name  # Просто имя агента
        self.role_prompt = role_prompt  # роль агента
        self.llm = llm  # LLM

    async def execute_task(self, prompt: str, task_type: str = "chat") -> str | None:
        """Выполняет задачу, передавая ее llm"""
        context_limit = await self.llm._get_context_limit(task_type)
        if not self.validate_prompt(prompt, context_limit):
            raise ValueError(
                f"Ошибка: Промпт для агента '{self.role_name}' слишком огромный ({self.count_tokens(self.role_prompt + prompt)} токенов)! "
                f"Он превышает безопасный лимит контекста ({context_limit - SAFE_LIMIT} токенов)."
            )
        return await self.llm.generate(
            prompt=prompt,
            model_role=self.role_name,
            system_prompt=self.role_prompt,
            task_type=task_type,
        )

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

        print(
            f"[{self.role_name}] Размер запроса: {prompt_tokens} токенов. Доступно: {available_space}"
        )

        if prompt_tokens > available_space:
            return False
        return True
