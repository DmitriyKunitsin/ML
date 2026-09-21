"""Единая настройка логирования для всего проекта.

Договорённость по уровням (соблюдается во всех модулях):

  * DEBUG     — тяжёлые детали: полные промпты, payload, сырые ответы LLM
                (всегда через preview(), иначе лог распухает на мегабайты);
  * INFO      — жизненный цикл: старт/финиш шага, агент, модель, вердикт,
                время выполнения;
  * WARNING   — странность, но не смертельно: пустой ответ LLM, ретрай,
                замечания ревьюера, исчерпание лимита попыток;
  * ERROR     — вызов упал или файл не записался (logger.exception даёт traceback);
  * CRITICAL  — сломался весь пайплайн.

Консоль по умолчанию показывает INFO, файл — DEBUG: в проде консоль остаётся
читаемой, а logs/agent.log хранит всё для разбора инцидента.
Один форматтер на оба хендлера, дата %Y-%m-%d %H:%M:%S — читается глазами и
режется grep'ом.

Файл ротируется (logging.handlers.RotatingFileHandler), поэтому лог не растёт
бесконечно. Уровни и папку можно менять из окружения: LOG_LEVEL, LOG_FILE_LEVEL,
LOG_DIR.

Папка лога по умолчанию — абсолютная и лежит рядом с проектом (MultiAgents/logs),
а не относительно текущего каталога: иначе запуск `python test_main.py` из другого
места отправлял бы лог куда угодно, и найти его было бы нельзя. Куда именно пишется
лог, видно в консоли: setup_logging() сообщает это на уровне INFO.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path

# Один формат на весь проект.
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Корень проекта (MultiAgents/): core/logging_setup.py -> core -> MultiAgents.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LOG_DIR = str(PROJECT_ROOT / "logs")
LOG_FILE_NAME = "agent.log"
MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 МБ на файл
LOG_BACKUP_COUNT = 3  # agent.log.1 ... agent.log.3

DEFAULT_CONSOLE_LEVEL = "INFO"  # прод: консоль не шумит
DEFAULT_FILE_LEVEL = "DEBUG"  # файл: всё, включая промпты — для отладки

# Секреты (API-ключи, токены) в лог попадать не должны:
#   sk-XXXX...            — OpenAI-совместимые ключи;
#   Bearer XXXX           — токены авторизации;
#   base64.base64         — формат ключей cloud.ru (см. config/key_llm.py).
SECRET_PATTERN = re.compile(
    r"(?:sk-|Bearer\s)[A-Za-z0-9_\-.]{6,}"
    r"|[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}"
)

# Сторонние библиотеки не должны забивать лог своими DEBUG-простынями.
NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3")

# Промпты и ответы моделей огромные: в лог идут только первые 500 символов.
PREVIEW_LIMIT = 500

# --- ANCHOR ---


def mask_secret(secret: str | None) -> str:
    """Маскирует секрет для лога: ``sk-***`` вместо ключа целиком.

    Оставляем 4 первых символа (чтобы отличать ключи друг от друга), всё
    остальное прячем — логи утекают в git и в переписку чаще, чем кажется.
    """
    if not secret:
        return "<пусто>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}***{secret[-2:]}"


class _SecretScrubbingFormatter(logging.Formatter):
    """Страховка: чистит секреты, даже если о них забыли в вызывающем коде.

    Достаточно один раз настроить форматтер, чем надеяться на дисциплину
    во всех местах логирования.
    """

    def format(self, record: logging.LogRecord) -> str:
        return SECRET_PATTERN.sub("***", super().format(record))


def preview(text: str | None, limit: int = PREVIEW_LIMIT) -> str:
    """Готовит длинный текст (промпт, ответ LLM) для лога.

    Обрезает до ``limit`` символов и экранирует переводы строк, чтобы одна
    запись лога оставалась одной строкой — иначе grep по логу бесполезен.
    """
    if not text:
        return "<пусто>"
    flat = "\\n".join(text.splitlines())
    if len(flat) <= limit:
        return flat
    return f"{flat[:limit]}... [обрезано, всего {len(text)} симв.]"


def _ensure_utf8_console() -> None:
    """Переводит stdout/stderr в UTF-8, иначе эмодзи этапов ломают вывод.

    В логе есть маркеры этапов (📋 Шаг 2, 💻 Шаг 5, ...), а консоль Windows по
    умолчанию cp1251: хендлер получит UnicodeEncodeError и запись потеряется.
    Поток может не поддерживать reconfigure (перехват в тестах) — молча выходим.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            # Поток уже закрыт или перенаправлен в файл без поддержки — не беда.
            continue


def setup_logging(
    console_level: str | None = None,
    file_level: str | None = None,
    log_dir: str | None = None,
) -> logging.Logger:
    """Инициализирует корневой логгер. Вызывается ОДИН раз в main().

    Консоль получает ``LOG_LEVEL`` (по умолчанию INFO), файл
    ``MultiAgents/logs/agent.log`` — ``LOG_FILE_LEVEL`` (по умолчанию DEBUG).
    Папка лога берётся абсолютная (рядом с проектом), поэтому не зависит от
    текущего каталога запуска, и создаётся при необходимости.

    Функция идемпотентна: повторный вызов не добавляет вторые хендлеры, иначе
    каждая запись дублировалась бы (классическая ошибка при
    ``logging.basicConfig`` в каждом модуле).

    Куда пишется лог, сообщается в консоль на уровне INFO — иначе файл
    приходится искать наугад.

    Если файл открыть нельзя (нет прав, каталог занят файлом) — остаёмся с
    консолью: логирование не должно ронять пайплайн.
    """
    console_level = console_level or os.getenv("LOG_LEVEL", DEFAULT_CONSOLE_LEVEL)
    file_level = file_level or os.getenv("LOG_FILE_LEVEL", DEFAULT_FILE_LEVEL)
    log_dir = log_dir or os.getenv("LOG_DIR", DEFAULT_LOG_DIR)

    # Эмодзи-маркеры этапов требуют UTF-8 в консоли — до первой записи.
    _ensure_utf8_console()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # фильтруют хендлеры, а не логгер

    # Идемпотентность: помечаем свои хендлеры атрибутом.
    root.handlers[:] = [
        h for h in root.handlers if not getattr(h, "_agent_pipeline", False)
    ]

    formatter = _SecretScrubbingFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(_resolve_level(console_level))
    console.setFormatter(formatter)
    console._agent_pipeline = True
    root.addHandler(console)

    log_path: str | None = None
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, LOG_FILE_NAME)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(_resolve_level(file_level))
        file_handler.setFormatter(formatter)
        file_handler._agent_pipeline = True
        file_handler._log_path = os.path.abspath(log_path)
        root.addHandler(file_handler)
    except OSError as exc:
        # Логирование не должно ломать работу — предупреждаем и живём дальше.
        root.warning(
            "❌ Не удалось открыть файл лога %s: %s Буду писать только в консоль.",
            log_dir,
            exc,
        )

    # Библиотеки-зависимости говорят на INFO только о важном.
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Куда пишется лог — видно в консоли (INFO), а не только в DEBUG-файле:
    # иначе искать файл приходится наугад.
    if log_path:
        root.info(
            "📄 Логи пишутся в файл: %s (консоль=%s, файл=%s).",
            os.path.abspath(log_path),
            console_level,
            file_level,
        )
    return root


def get_log_file_path() -> str | None:
    """Путь к файлу лога или None, если файловый хендлер не поднялся.

    Нужна там, где пользователю надо показать, куда смотреть: сообщения
    «куда писать отчёт» и «где искать подробности» не должны хардкодить путь.
    """
    for handler in logging.getLogger().handlers:
        path = getattr(handler, "_log_path", None)
        if path:
            return str(path)
    return None


def _resolve_level(level: str | int | None) -> int:
    """Приводит уровень из строки ('info', 'DEBUG') или числа к int."""
    if isinstance(level, int):
        return level
    if not level:
        return logging.NOTSET
    value = logging.getLevelName(str(level).strip().upper())
    return value if isinstance(value, int) else logging.NOTSET


def close_logging_handlers() -> None:
    """Закрывает и убирает хендлеры, созданные setup_logging().

    Нужна там, где логирование переключают на другое место (тесты, повторный
    запуск в одном процессе). На Windows это ещё и обязательно: открытый
    файл лога нельзя удалить, поэтому временная папка с логом не чистится,
    пока хендлер не закрыт.
    """
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if getattr(handler, "_agent_pipeline", False):
            root.removeHandler(handler)
            handler.close()