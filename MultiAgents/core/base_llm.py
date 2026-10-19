import asyncio
import logging
import time

from abc import ABC, abstractmethod

from utils.progress_spinner import notify as spinner_notify
from utils.progress_spinner import stop as spinner_stop

logger = logging.getLogger(__name__)

# Как часто таймер напоминает о себе в лог (сек). Пишем на уровне DEBUG:
# в консоль (INFO) прогресс не сыплется, в файле видно, что запрос жив.
PROGRESS_TICK_SECONDS = 15.0


class BaseLLM(ABC):
    """Абстрактный базовый класс для всех LLM-провайдеров с встроенным таймером."""

    def __init__(self, timeout: float = 300.0):
        self.timeout = timeout

    async def _track_time(self, context: str = ""):
        """Асинхронный таймер, запускаемый в фоне.

        Каждые ``PROGRESS_TICK_SECONDS`` делает DEBUG-запись в файл («запрос
        жив, уже N сек») и обновляет спиннер в консоли с живым временем
        (``Работает… 0:12``). ``context`` — что именно выполняется.
        """
        started_at = time.perf_counter()
        try:
            while True:
                await asyncio.sleep(PROGRESS_TICK_SECONDS)
                elapsed = time.perf_counter() - started_at
                logger.debug(
                    "⏳ Генерация продолжается: %.1f сек (timeout=%.0f сек).",
                    elapsed,
                    self.timeout,
                )
                spinner_notify(
                    context or "LLM",
                    f"Работает… {int(elapsed // 60)}:{int(elapsed % 60):02d}",
                )
        except asyncio.CancelledError:
            raise

    async def generate_with_timer(self, *args, **kwargs):
        """
        Обёртка, которая запускает таймер, вызывает _generate(),
        и возвращает результат. Наследники должны реализовать _generate().
        """
        timer_task = asyncio.create_task(self._track_time())
        start_call = time.perf_counter()

        try:
            result = await self._generate(*args, **kwargs)
            logger.info(
                "⏱️ Время генерации: %.2f сек.", time.perf_counter() - start_call
            )
            return result
        except Exception:
            # logger.exception приложит traceback целиком — собирать его вручную
            # через str(e) (как было раньше) бессмысленно: теряется стек.
            logger.exception(
                "❌ Ошибка генерации (прошло %.2f сек).",
                time.perf_counter() - start_call,
            )
            return None
        finally:
            timer_task.cancel()
            try:
                await timer_task
            except asyncio.CancelledError:
                pass
            # Обновляем спиннер «Работает…», чтобы финальные сообщения
            # (время, пути файлов) выглядели чисто, без "\r"-хвостов.
            spinner_stop()

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
