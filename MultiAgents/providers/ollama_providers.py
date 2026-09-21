import logging

import httpx
from core.base_llm import BaseLLM
from core.logging_setup import preview

logger = logging.getLogger(__name__)


class AsyncOllamaClient(BaseLLM):
    """Асинхронный клиент для Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        boss_model: str = "llama3.1:8b-instruct-q4_K_M",
        worker_model: str = "codellama:7b-instruct-q4_K_M",
        timeout: float = 900.0,  # 15 minut
    ):
        super().__init__(timeout=timeout)
        self.chat_url = f"{base_url.rstrip('/')}/api/chat"
        self.models = {"boss": boss_model, "worker": worker_model}
        # Имя основной модели нужно для логов BaseAgent (getattr llm.model_name).
        self.model_name = boss_model

        self._task_configs = {
            # Больше контекста для диалога, достаточно токенов для развёрнутого ответа.
            "chat": ("llama3.2:3b", 16384, 512, 0.5),
            # Максимальный контекст для большого кода, много токенов для полного скетча, низкая температура для строгого синтаксиса.
            "code": ("codellama:7b-instruct-q4_K_M", 16384, 8192, 0.1),
            # Большой контекст для анализа, много токенов для детального ТЗ.
            "boss": ("llama3.1:8b-instruct-q4_K_M", 32768, 4096, 0.3),
            # Максимальный контекст для чтения большого кода и ТЗ, много токенов для развёрнутых замечаний.
            "review": ("llama3.1:8b-instruct-q4_K_M", 65536, 4096, 0.3),
        }
        self._default_config = ("llama3.2:3b", 8192, 64, 0.5)

    def _get_config(self, task_type: str) -> tuple:
        return self._task_configs.get(task_type.lower(), self._default_config)

    async def _get_context_limit(self, task_type: str) -> int:
        return self._get_config(task_type)[1]

    async def _generate(
        self,
        prompt: str,
        system_prompt: str = "",
        task_type: str = "chat",
    ) -> str | None:
        """Реализация _generate() для Ollama."""
        model_name, ctx, predict, temp = self._get_config(task_type)
        logger.debug(
            "🤖 Параметры вызова Ollama: модель=%s, task_type=%s, num_ctx=%s, num_predict=%s, temperature=%s.",
            model_name,
            task_type,
            ctx,
            predict,
            temp,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": ctx,
                "num_predict": predict,
                "temperature": temp,
                "num_thread": 8,
                "use_mmap": True,
                "use_mlock": True,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.debug(
                "🤖 Запрос к Ollama %s (timeout=%.0f сек): user=%s",
                self.chat_url,
                self.timeout,
                preview(prompt),
            )
            response = await client.post(self.chat_url, json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            logger.debug("🤖 Ответ Ollama: %s", preview(content))
            return content
