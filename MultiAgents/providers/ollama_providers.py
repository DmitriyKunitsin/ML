import httpx
from core.base_llm import BaseLLM


class AsyncOllamaClient(BaseLLM):
    """Асинхронный клиент для Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        boss_model: str = "llama3.1:8b-instruct-q4_K_M",
        worker_model: str = "codellama:7b-instruct-q4_K_M",
        timeout: float = 300.0,
    ):
        super().__init__(timeout=timeout)
        self.chat_url = f"{base_url.rstrip('/')}/api/chat"
        self.models = {"boss": boss_model, "worker": worker_model}

        self._task_configs = {
            "short": (8192, 32, 0.5),
            "chat": (8192, 64, 0.5),
            "long": (65536, 2048, 0.5),
            "code": (8192, 4096, 0.2),
            "boss": (8192, 2048, 0.3),
            "review": (32768, 2048, 0.3),
        }
        self._default_config = (8192, 64, 0.5)

    def _get_config(self, task_type: str) -> tuple:
        return self._task_configs.get(task_type.lower(), self._default_config)

    async def _get_context_limit(self, task_type):
        return self._get_config(task_type)[0]

    async def _generate(
        self,
        model_role: str,
        prompt: str,
        system_prompt: str = "",
        task_type: str = "chat",
    ) -> str | None:
        """Реализация _generate() для Ollama."""
        model_name = self.models.get(model_role, model_role)
        ctx, predict, temp = self._get_config(task_type)

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
            response = await client.post(self.chat_url, json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]
