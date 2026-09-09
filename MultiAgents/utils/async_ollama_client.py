import asyncio
import sys
import time
import httpx


class AsyncOllamaClient:
    """Асинхронный клиент для Ollama с красивым таймером генерации."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        boss_model: str = "llama3.1:8b-instruct-q4_K_M",
        worker_model: str = "codellama:7b-instruct-q4_K_M",
    ):
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

    async def _track_time(self):
        """Асинхронный фоновый таймер."""
        start_time = time.time()
        try:
            while True:
                elapsed = time.time() - start_time
                sys.stdout.write(f"\r   ⏳ Генерация... прошло {elapsed:.1f} сек")
                sys.stdout.flush()
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            # Срабатывает, когда задачу принудительно отменяют (генерация завершена)
            sys.stdout.write("\r" + " " * 40 + "\r")
            sys.stdout.flush()

    def _get_config(self, task_type: str) -> tuple:
        return self._task_configs.get(task_type.lower(), self._default_config)

    async def ask(
        self,
        model_role: str,
        prompt: str,
        system_prompt: str = "",
        task_type: str = "chat",
    ) -> str | None:
        """Асинправляет запрос к Ollama."""
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

        # Запускаем фоновый таймер как отдельную асинхронную задачу
        timer_task = asyncio.create_task(self._track_time())
        start_call = time.time()

        # Используем httpx.AsyncClient с большим таймаутом (5 минут)
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(self.chat_url, json=payload)
                response.raise_for_status()
                result = response.json()["message"]["content"]

                total_time = time.time() - start_call
                print(f"   ⏱️ Время генерации: {total_time:.2f} сек")
                return result

            except Exception as e:
                total_time = time.time() - start_call
                print(
                    f"❌ Ошибка при запросе к {model_name}: {e} (прошло {total_time:.2f} сек)"
                )
                return None

            finally:
                timer_task.cancel()
                try:
                    await timer_task
                except asyncio.CancelledError:
                    pass


# --- Как запускать асинхронный код ---
async def main():
    ollama = AsyncOllamaClient()

    print("--- Тест 1: Последовательные запросы ---")
    # Сначала босс думает над планом
    plan = await ollama.ask("boss", "Придумай тему для короткого рассказа.", "boss")

    if plan:
        # Потом воркер пишет по этому плану
        await ollama.ask("worker", f"Напиши рассказ по теме: {plan}", "code")

    print("\n--- Тест 2: Параллельные запросы (магия async) ---")
    print("Запускаем обе модели одновременно...")

    # Эти два запроса уйдут в Ollama одновременно (если сервер и железо потянут)
    task1 = ollama.ask("boss", "Напиши стих про Python", "chat")
    task2 = ollama.ask("worker", "Напиши стих про JavaScript", "chat")

    # Ждем выполнения обоих
    results = await asyncio.gather(task1, task2)
    print(
        f"Оба ответа получены! Количество символов: {len(results[0] or '')} и {len(results[1] or '')}"
    )


if __name__ == "__main__":
    asyncio.run(main())
