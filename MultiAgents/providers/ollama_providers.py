import logging
import os

import httpx
from core.base_llm import BaseLLM
from core.base_agent import SAFE_LIMIT
from core.llm_types import LLMResponse
from core.logging_setup import preview

logger = logging.getLogger(__name__)


class AsyncOllamaClient(BaseLLM):
    """Асинхронный клиент для Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        boss_model: str = "llama3.1:8b-instruct-q4_K_M",
        worker_model: str = "codellama:7b-instruct-q4_K_M",
        timeout: float | None = None,
    ):
        # Локальная Ollama не подключена к интернету, но модель может думать
        # долго. По умолчанию 15 минут; переопределить можно через LLM_TIMEOUT
        # без правки кода — единообразно с облачным провайдером.
        default_timeout = float(os.getenv("LLM_TIMEOUT", "900"))
        super().__init__(timeout=timeout if timeout is not None else default_timeout)
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
        **kwargs,
    ) -> LLMResponse | None:
        """Реализация _generate() для Ollama.

        Возвращает ``LLMResponse`` с ``done_reason``: Ollama сама сообщает,
        упёрлась ли генерация в ``num_predict`` ("length"). Обрезанный ответ
        нельзя подавать в sandbox/ревью — пайплайн должен знать об обрезке.
        """
        model_name, ctx, predict, temp = self._get_config(task_type)
        # Динамический num_predict: не больше, чем реально осталось места
        # в контексте (num_ctx) после промпта. Иначе модель обрежется по
        # num_predict раньше, чем допишет ответ.
        prompt_tokens_observed = kwargs.get("prompt_tokens")
        if prompt_tokens_observed is not None:
            predict = min(predict, max(1, ctx - SAFE_LIMIT - prompt_tokens_observed))
        logger.debug(
            "🤖 Параметры вызова Ollama: модель=%s, task_type=%s, num_ctx=%s, "
            "num_predict=%s (динамический), temperature=%s.",
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
            data = response.json()
            content = data["message"]["content"]
            # done_reason у Ollama: "stop" | "length" | null. "length" значит,
            # что ответ не поместился в num_predict — такой текст нельзя
            # считать полным кодом.
            done_reason = data.get("done_reason")
            logger.debug("🤖 Ответ Ollama: %s", preview(content))
            if not content or not content.strip():
                logger.warning(
                    "⚠️ Пустой ответ Ollama (done_reason=%s).", done_reason
                )
                return None
            if done_reason == "length":
                logger.warning(
                    "⚠️ Ответ Ollama обрезан (done_reason=length). Код может быть недописан."
                )
            return LLMResponse(
                text=content.strip(),
                finish_reason=done_reason,
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
            )
