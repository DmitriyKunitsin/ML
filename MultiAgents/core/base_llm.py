from abc import ABC, abstractmethod
import asyncio
import sys
import time


class BaseLLM(ABC):
    """Абстрактный базовый класс для всех LLM-провайдеров с встроенным таймером."""

    def __init__(self, timeout: float = 300.0):
        self.timeout = timeout

    async def _track_time(self):
        """Асинхронный таймер, запускаемый в фоне."""
        start_time = time.time()
        try:
            while True:
                elapsed = time.time() - start_time
                sys.stdout.write(f"\r   ⏳ Генерация... прошло {elapsed:.1f} сек")
                sys.stdout.flush()
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            sys.stdout.write("\r" + " " * 40 + "\r")
            sys.stdout.flush()

    async def generate_with_timer(self, *args, **kwargs):
        """
        Обёртка, которая запускает таймер, вызывает _generate(),
        и возвращает результат. Наследники должны реализовать _generate().
        """
        timer_task = asyncio.create_task(self._track_time())
        start_call = time.time()

        try:
            result = await self._generate(*args, **kwargs)
            total_time = time.time() - start_call
            print(f"   ⏱️ Время генерации: {total_time:.2f} сек")
            return result
        except Exception as e:
            total_time = time.time() - start_call
            print(f"❌ Ошибка: {e} (прошло {total_time:.2f} сек)")
            return None
        finally:
            timer_task.cancel()
            try:
                await timer_task
            except asyncio.CancelledError:
                pass

    @abstractmethod
    async def _generate(self, *args, **kwargs):
        """
        Реальная логика запроса к API.
        Каждый провайдер реализует этот метод.
        """
        pass

    @abstractmethod
    async def _get_context_limit(self):
        """Сообщает информацию о своей модели"""
        pass

    async def generate(self, *args, **kwargs):
        """Вызывается извне. Запускает таймер и вызывает _generate()"""
        return await self.generate_with_timer(*args, **kwargs)
