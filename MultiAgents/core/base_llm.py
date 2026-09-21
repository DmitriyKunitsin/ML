from abc import ABC, abstractmethod
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# Как часто таймер напоминает о себе в лог (сек). Пишем на уровне DEBUG:
# в консоль (INFO) прогресс не сыплется, в файле видно, что запрос жив.
PROGRESS_TICK_SECONDS = 15.0


class BaseLLM(ABC):
    """Абстрактный базовый класс для всех LLM-провайдеров с встроенным таймером."""

    def __init__(self, timeout: float = 300.0):
        self.timeout = timeout

    async def _track_time(self):
        """Асинхронный таймер, запускаемый в фоне.

        Раньше он перерисовывал строку в stdout («⏳ Генерация... 12.5 сек»).
        Теперь это DEBUG-запись в файл раз в PROGRESS_TICK_SECONDS — консоль
        остаётся чистой, а по логу видно, что долгий запрос не завис.
        """
        started_at = time.perf_counter()
        try:
            while True:
                await asyncio.sleep(PROGRESS_TICK_SECONDS)
                logger.debug(
                    "⏳ Генерация продолжается: %.1f сек (timeout=%.0f сек).",
                    time.perf_counter() - started_at,
                    self.timeout,
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
