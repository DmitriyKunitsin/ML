"""Тесты настройки логирования (core.logging_setup).

Проверяется контракт, на который опирается весь проект:

1. setup_logging() даёт один формат на консоль и файл и идемпотентен —
   повторный вызов не дублирует хендлеры (иначе каждая запись идёт дважды).
2. Файл logs/agent.log пишется в UTF-8 и получает DEBUG-записи, даже когда
   консоль настроена на INFO. Путь к логу — абсолютный и не зависит от того,
   из какого каталога запущена программа, а сам путь виден в консоли (INFO).
3. Формат записи — "%Y-%m-%d %H:%M:%S | LEVEL | name | message".
4. Секреты маскируются и не попадают в лог целиком.
5. preview() не даёт логу распухнуть: обрезает и склеивает в одну строку.
6. В коде проекта не осталось print() — договорённость «логируем, не печатаем».
7. Каждый шаг пайплайна помечен эмодзи-маркером: по логу видно этап работы.

Особенность Windows: открытый файл лога нельзя удалить. Поэтому временная
папка создаётся через mkdtemp + addCleanup — cleanup выполняется после
tearDown, когда хендлеры уже закрыты (с TemporaryDirectory файл остаётся
заблокированным и удаление падает с PermissionError).
"""

import logging
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from core.logging_setup import (
    DEFAULT_LOG_DIR,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    PREVIEW_LIMIT,
    _SecretScrubbingFormatter,
    close_logging_handlers,
    get_log_file_path,
    mask_secret,
    preview,
    setup_logging,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Модули, которые обязаны писать в logging, а не в stdout.
# core/logging_setup.py здесь нет: он настраивает корневой логгер и не
# логирует от своего имени — своего logger у него и не должно быть.
LOGGED_MODULES = (
    "test_main.py",
    "core/base_agent.py",
    "core/base_llm.py",
    "providers/cloud_api_providers.py",
    "providers/ollama_providers.py",
    "utils/helpers.py",
)


class LoggingTestBase(unittest.TestCase):
    """Изолирует глобальный логгер и временную папку с логами."""

    def setUp(self):
        self._root = logging.getLogger()
        self._saved_handlers = self._root.handlers[:]
        self._saved_level = self._root.level
        self._env = {
            key: os.environ.get(key)
            for key in ("LOG_LEVEL", "LOG_FILE_LEVEL", "LOG_DIR")
        }
        self.tmpdir = tempfile.mkdtemp()
        # addCleanup выполняется ПОСЛЕ tearDown, то есть когда хендлеры уже
        # закрыты и файл лога больше не заблокирован.
        self.addCleanup(self._cleanup_tmpdir)

    def tearDown(self):
        close_logging_handlers()
        self._root.handlers[:] = self._saved_handlers
        self._root.setLevel(self._saved_level)
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _cleanup_tmpdir(self):
        close_logging_handlers()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _setup(self, console_level="INFO", file_level="DEBUG"):
        """Настраивает логирование в изолированную папку, возвращает её путь."""
        log_dir = os.path.join(self.tmpdir, "logs")
        setup_logging(
            console_level=console_level, file_level=file_level, log_dir=log_dir
        )
        return log_dir

    def _flush(self):
        for handler in logging.getLogger().handlers:
            handler.flush()

    def _read_log(self, log_dir):
        with open(os.path.join(log_dir, "agent.log"), encoding="utf-8") as handle:
            return handle.read()

    def _console_handler(self):
        """Консольный хендлер проекта (у файлового есть baseFilename)."""
        return next(
            handler
            for handler in logging.getLogger().handlers
            if getattr(handler, "_agent_pipeline", False)
            and not hasattr(handler, "baseFilename")
        )


class TestSetupLoggingHandlers(LoggingTestBase):
    def test_creates_console_and_file_handlers(self):
        log_dir = self._setup()

        self.assertTrue(os.path.isdir(log_dir))
        self.assertTrue(os.path.exists(os.path.join(log_dir, "agent.log")))
        self.assertEqual(self._console_handler().level, logging.INFO)

    def test_is_idempotent(self):
        """Повторный вызов не должен удваивать хендлеры (иначе логи дублируются)."""
        log_dir = self._setup()
        handlers_first = len(logging.getLogger().handlers)

        setup_logging(console_level="INFO", file_level="DEBUG", log_dir=log_dir)

        self.assertEqual(len(logging.getLogger().handlers), handlers_first)

    def test_single_record_is_written_once(self):
        """Если хендлеры задвоены, одна и та же запись попадает в файл дважды."""
        log_dir = self._setup()
        setup_logging(console_level="INFO", file_level="DEBUG", log_dir=log_dir)

        logging.getLogger("duplicate.probe").warning("единственная запись")
        self._flush()

        self.assertEqual(self._read_log(log_dir).count("единственная запись"), 1)

    def test_close_logging_handlers_releases_file(self):
        """После закрытия хендлеров файл лога освобождается (важно для Windows)."""
        log_dir = self._setup()
        logging.getLogger("release.probe").info("запись перед закрытием")

        close_logging_handlers()

        self.assertEqual(logging.getLogger().handlers, self._saved_handlers)
        # Файл лога остался на диске и доступен на чтение.
        self.assertTrue(os.path.exists(os.path.join(log_dir, "agent.log")))

    def test_unwritable_log_dir_does_not_raise(self):
        """Каталог логов занят файлом: остаёмся с консолью, пайплайн не падает."""
        blocker = os.path.join(self.tmpdir, "logs")
        with open(blocker, "w", encoding="utf-8") as handle:
            handle.write("not a directory")

        # Исключения быть не должно: логирование не ломает работу.
        setup_logging(console_level="INFO", file_level="DEBUG", log_dir=blocker)

        self.assertEqual(self._console_handler().level, logging.INFO)
        # Файлового хендлера нет — путь к логу сообщить нечего.
        self.assertIsNone(get_log_file_path())

    def test_default_log_dir_is_absolute_near_project(self):
        """Папка логов по умолчанию не зависит от текущего каталога запуска.

        Иначе `python test_main.py` из другого места писал бы лог куда угодно,
        и найти его было бы нельзя.
        """
        self.assertTrue(os.path.isabs(DEFAULT_LOG_DIR))
        self.assertEqual(DEFAULT_LOG_DIR, str(PROJECT_ROOT / "logs"))

    def test_get_log_file_path_points_to_written_file(self):
        log_dir = self._setup()

        log_path = get_log_file_path()

        self.assertTrue(os.path.isabs(log_path))
        self.assertEqual(log_path, os.path.join(log_dir, "agent.log"))
        logging.getLogger("path.probe").info("проверка пути")
        self._flush()
        self.assertTrue(os.path.exists(log_path))

    def test_setup_reports_log_file_in_console(self):
        """Куда пишется лог — видно в консоли, иначе файл ищут наугад."""
        log_dir = self._setup()
        log_path = os.path.join(log_dir, "agent.log")

        with self.assertLogs(level="INFO") as captured:
            setup_logging(
                console_level="INFO", file_level="DEBUG", log_dir=log_dir
            )

        self.assertIn(os.path.abspath(log_path), "\n".join(captured.output))


class TestLogFormat(LoggingTestBase):
    def test_file_line_matches_project_format(self):
        log_dir = self._setup()
        logging.getLogger("format.probe").info("проверка формата")
        self._flush()

        lines = [
            line
            for line in self._read_log(log_dir).splitlines()
            if "проверка формата" in line
        ]

        self.assertEqual(len(lines), 1)
        # 2026-09-21 12:00:00 | INFO     | format.probe | проверка формата
        pattern = (
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| "
            r"INFO\s+\| format\.probe \| проверка формата"
        )
        self.assertRegex(lines[0].strip(), pattern)

    def test_format_constants_are_stable(self):
        # Меняя формат, придётся править грепы и парсеры — пусть тест заметит.
        self.assertEqual(
            LOG_FORMAT, "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        self.assertEqual(LOG_DATE_FORMAT, "%Y-%m-%d %H:%M:%S")

    def test_debug_goes_to_file_not_console(self):
        """DEBUG пишется в файл, но не в консоль: в проде консоль не шумит."""
        log_dir = self._setup(console_level="INFO", file_level="DEBUG")

        logging.getLogger("levels.probe").debug("только в файл")
        self._flush()

        self.assertIn("только в файл", self._read_log(log_dir))
        self.assertEqual(self._console_handler().level, logging.INFO)

    def test_env_overrides_levels(self):
        log_dir = os.path.join(self.tmpdir, "logs")
        os.environ["LOG_LEVEL"] = "WARNING"
        try:
            setup_logging(log_dir=log_dir)
        finally:
            os.environ.pop("LOG_LEVEL", None)

        self.assertEqual(self._console_handler().level, logging.WARNING)


class TestStageMarkers(unittest.TestCase):
    """Эмодзи-маркеры этапов: по логу должно быть видно, какой шаг идёт.

    Без маркеров INFO-строки шагов сливаются в однородную простыню, а с ними
    этап находится глазами моментально (📋 ТЗ, 💻 код, 🔧 компиляция, 🧪 тесты).
    """

    # Шаг -> эмодзи, которым он помечен в test_main.py.
    STAGE_MARKERS = {
        "Шаг 2": "📋",
        "Шаг 3": "✔",
        "Шаг 4": "📐",
        "Шаг 5": "💻",
        "Шаг 6": "🔧",
        "Шаг 7": "🧪",
    }

    def setUp(self):
        self.source = (PROJECT_ROOT / "test_main.py").read_text(encoding="utf-8")

    def test_each_stage_has_emoji_marker(self):
        for stage, marker in self.STAGE_MARKERS.items():
            self.assertRegex(
                self.source,
                re.escape(marker) + r"[^\n]*" + re.escape(stage),
                f"{stage}: нет эмодзи-маркера {marker} в логирующей строке",
            )

    def test_outcome_markers_are_used(self):
        """Итог шага тоже читается с одного взгляда: ✅ ок, ❌ провал, ⚠️ странность."""
        for marker in ("✅", "❌", "⚠️", "💚", "🎉", "🚀"):
            self.assertIn(marker, self.source, f"маркер {marker} не используется")

    def test_source_file_is_utf8_without_bom(self):
        """BOM ломает первый импорт и портит grep — файл должен быть чистым UTF-8."""
        raw = (PROJECT_ROOT / "test_main.py").read_bytes()

        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "файл начинается с BOM")


class TestSecretMasking(unittest.TestCase):
    def test_mask_secret_hides_body(self):
        secret = (
            "M2UzMGU3MjYtMGE5Zi00ODA1LWEzYTUtNDRjYTY5ZjRkMWVi"
            ".b7764057f6636c67b6f8359bbdf04d90"
        )
        masked = mask_secret(secret)

        self.assertNotIn(secret, masked)
        self.assertTrue(masked.startswith(secret[:4]))
        self.assertIn("***", masked)

    def test_mask_secret_handles_empty(self):
        self.assertEqual(mask_secret(None), "<пусто>")
        self.assertEqual(mask_secret(""), "<пусто>")

    def test_mask_secret_handles_short_value(self):
        # Короткое значение маскируем целиком: 4+2 символа уже раскрывают секрет.
        self.assertEqual(mask_secret("abc12345"), "***")

    def test_formatter_scrubs_secret_from_log_record(self):
        """Страховка форматтера: ключ не утечёт, даже если его забыли замаскировать."""
        formatter = _SecretScrubbingFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        record = logging.LogRecord(
            name="secret.probe",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="ключ: %s",
            args=("sk-abcdef1234567890abcdef",),
            exc_info=None,
        )
        rendered = formatter.format(record)

        self.assertNotIn("sk-abcdef1234567890abcdef", rendered)
        self.assertIn("***", rendered)


class TestPreview(unittest.TestCase):
    def test_short_text_is_kept(self):
        self.assertEqual(preview("короткий текст"), "короткий текст")

    def test_long_text_is_truncated(self):
        result = preview("x" * 5000)

        self.assertLess(len(result), 5000)
        self.assertIn("обрезано", result)
        self.assertIn("5000", result)

    def test_multiline_text_becomes_single_line(self):
        """Одна запись лога = одна строка, иначе grep по логу бесполезен."""
        result = preview("первая\nвторая\r\nтретья")

        self.assertNotIn("\n", result)
        self.assertIn("\\n", result)

    def test_empty_text(self):
        self.assertEqual(preview(""), "<пусто>")
        self.assertEqual(preview(None), "<пусто>")

    def test_limit_is_configurable(self):
        self.assertEqual(len(preview("y" * 100, limit=PREVIEW_LIMIT)), 100)
        self.assertIn("обрезано", preview("y" * 100, limit=10))


class TestNoPrintInPipeline(unittest.TestCase):
    """print — только для быстрой отладки; в пайплайне используем logging."""

    def test_modules_do_not_use_print(self):
        offenders = []
        for relative in LOGGED_MODULES:
            source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            for match in re.finditer(r"(?<![\w.])print\s*\(", source):
                line = source[: match.start()].count("\n") + 1
                offenders.append(f"{relative}:{line}")

        self.assertEqual(offenders, [], f"print() найден в: {', '.join(offenders)}")

    def test_modules_declare_module_level_logger(self):
        for relative in LOGGED_MODULES:
            source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(
                "logging.getLogger(__name__)",
                source,
                f"{relative}: нет logger = logging.getLogger(__name__)",
            )


if __name__ == "__main__":
    unittest.main()