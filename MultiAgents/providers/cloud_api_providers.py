import asyncio
import logging
import os
import random

from openai import AsyncOpenAI
from config.key_llm import cloud_key

from core.base_llm import BaseLLM
from core.llm_types import LLMResponse
from core.logging_setup import mask_secret, preview

logger = logging.getLogger(__name__)


class CloudAPIProvider(BaseLLM):
    """Асинхронный клиент для облачного OpenAI-совместимого API."""

    def __init__(
        self,
        model_name: str,
        url: str = "https://foundation-models.api.cloud.ru/v1",
        timeout: float | None = None,
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ):
        # Облако умеет думать дольше минуты: ставим 10 минут по умолчанию,
        # а переопределять можно через LLM_TIMEOUT без правки кода.
        default_timeout = float(os.getenv("LLM_TIMEOUT", "600"))
        super().__init__(timeout=timeout if timeout is not None else default_timeout)
        self.model_name = model_name
        self.url = url
        self.max_retries = max_retries  # Сколько раз повторить запрос при ошибке
        self.retry_delay = retry_delay  # Базовая пауза между попытками (сек)
        self.client = AsyncOpenAI(
            api_key=cloud_key,
            base_url=self.url,
            timeout=self.timeout,
        )
        # В лог уходит только маска: ключ в логах = утечка в git и в переписку.
        logger.info(
            "🚀 LLM-провайдер: %s (модель=%s, url=%s, ключ=%s, timeout=%.0f сек).",
            type(self).__name__,
            self.model_name,
            self.url,
            mask_secret(cloud_key),
            self.timeout,
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
    ) -> LLMResponse | None:
        """Реализация _generate() для облачного API (с повторными попытками).

        Возвращает ``LLMResponse`` с ``finish_reason``: пайплайн должен знать,
        не обрезала ли модель ответ по ``max_tokens`` (иначе оборванный код
        уходит дальше и провоцирует бесконечный цикл правок).
        """
        _, max_tokens, temperature = self._get_config(task_type)
        logger.debug(
            "🤖 Параметры вызова: модель=%s, task_type=%s, max_tokens=%d, temperature=%.2f.",
            self.model_name,
            task_type,
            max_tokens,
            temperature,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        total_attempts = self.max_retries + 1
        last_error: str | None = None

        for attempt in range(1, total_attempts + 1):
            try:
                logger.debug(
                    "🤖 Запрос к LLM (попытка %d/%d): system=%s | user=%s",
                    attempt,
                    total_attempts,
                    preview(system_prompt),
                    preview(prompt),
                )
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    presence_penalty=0,
                    top_p=0.95,
                    messages=messages,
                )
                choice = response.choices[0]
                content = choice.message.content
                # Причина остановки генерации: "stop" | "length" | ...
                # API без причины (None) НЕ помечаем как обрезанный — это
                # избегает ложных срабатываний на нестандартных эндпоинтах.
                finish_reason = getattr(choice, "finish_reason", None)
                usage = getattr(response, "usage", None)
                prompt_tokens = (
                    getattr(usage, "prompt_tokens", None) if usage else None
                )
                completion_tokens = (
                    getattr(usage, "completion_tokens", None) if usage else None
                )
                if content and content.strip():
                    if finish_reason == "length":
                        logger.warning(
                            "⚠️ Ответ обрезан по max_tokens=%d (finish_reason=length, "
                            "потоков: %d/%d). Код может быть недописан.",
                            max_tokens,
                            completion_tokens if completion_tokens is not None else -1,
                            max_tokens,
                        )
                    return LLMResponse(
                        text=content.strip(),
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                last_error = "пустой ответ модели"
                logger.warning(
                    "⚠️ Пустой ответ LLM (попытка %d/%d).", attempt, total_attempts
                )
            except Exception as ex:
                last_error = f"{type(ex).__name__}: {ex}"
                # logger.exception — только при последней попытке: иначе traceback
                # от ретраев, которые заведомо повторятся, забивает лог.
                if attempt < total_attempts:
                    logger.warning(
                        "⚠️ Ошибка LLM (попытка %d/%d): %s",
                        attempt,
                        total_attempts,
                        last_error,
                    )
                else:
                    logger.exception(
                        "❌ Ошибка LLM (попытка %d/%d): %s",
                        attempt,
                        total_attempts,
                        last_error,
                    )

            if attempt < total_attempts:
                # Случайная пауза (не суммарная экспонента): если в будущем
                # появятся параллельные агенты — попытки не совпадут по времени.
                await asyncio.sleep(self.retry_delay + random.uniform(0, self.retry_delay))

        logger.error("❌ LLM недоступна, запрос не выполнен: %s", last_error)
        return None
