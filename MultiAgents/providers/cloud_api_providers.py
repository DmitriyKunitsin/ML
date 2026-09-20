import asyncio

from openai import AsyncOpenAI
from config.key_llm import cloud_key

from core.base_llm import BaseLLM


class CloudAPIProvider(BaseLLM):
    """Асинхронный клиент для облачного OpenAI-совместимого API."""

    def __init__(
        self,
        model_name: str,
        url: str = "https://foundation-models.api.cloud.ru/v1",
        timeout: float = 180.0,
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ):
        super().__init__(timeout=timeout)
        self.model_name = model_name
        self.url = url
        self.max_retries = max_retries  # Сколько раз повторить запрос при ошибке
        self.retry_delay = retry_delay  # Базовая пауза между попытками (сек)
        self.client = AsyncOpenAI(
            api_key=cloud_key,
            base_url=self.url,
            timeout=self.timeout,
        )

        # (лимит_контекста, max_tokens_ответа, temperature) для каждого типа задачи.
        # task_type приходит из BaseAgent.execute_task(task_type=...).
        self._task_configs = {
            # Развёрнутый, но не гигантский ответ (ТЗ, архитектура, ревью).
            "chat": (16384, 4096, 0.5),
            # Код: низкая температура для строгого синтаксиса, максимум токенов.
            "code": (32768, 8192, 0.2),
            # Проектирование архитектуры — компромисс объёма и стабильности.
            "boss": (32768, 4096, 0.3),
            # Ревью больших ТЗ и скетчей — нужен большой запас токенов.
            "review": (32768, 4096, 0.3),
        }
        self._default_config = (16384, 4096, 0.5)

    def _get_config(self, task_type: str) -> tuple[int, int, float]:
        return self._task_configs.get(task_type.lower(), self._default_config)

    async def _get_context_limit(self, task_type: str) -> int:
        return self._get_config(task_type)[0]

    async def _generate(
        self,
        prompt: str,
        system_prompt: str = "",
        task_type: str = "chat",
        **kwargs,
    ) -> str | None:
        """Реализация _generate() для облачного API (с повторными попытками)."""
        _, max_tokens, temperature = self._get_config(task_type)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        total_attempts = self.max_retries + 1
        last_error: str | None = None

        for attempt in range(1, total_attempts + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    presence_penalty=0,
                    top_p=0.95,
                    messages=messages,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                last_error = "пустой ответ модели"
                print(f"⚠️ Пустой ответ LLM (попытка {attempt}/{total_attempts}).")
            except Exception as ex:
                last_error = f"{type(ex).__name__}: {ex}"
                print(f"⚠️ Ошибка LLM (попытка {attempt}/{total_attempts}): {last_error}")

            if attempt < total_attempts:
                await asyncio.sleep(self.retry_delay * attempt)

        print(f"❌ LLM недоступна, запрос не выполнен: {last_error}")
        return None
