# Пакет тестов. Сделан пакетом, чтобы pytest/unittest корректно добавляли
# корень проекта в sys.path и `import test_main` работал без правок окружения.
#
# Логи и сообщения пайплайна — на русском и с эмодзи-маркерами этапов, а консоль
# Windows по умолчанию cp1251: без этого переключения
# `python -m unittest discover -s tests -t .` падает на UnicodeEncodeError ещё до
# проверок. В самом пайплайне то же делает setup_logging().
import os
import sys

# Тесты не выводят консольный спиннер: он пишет '\r'-кадры с Брайлем в stdout,
# что замусоривает протокол CI/IDE. В пайплайне спиннер включается повторно,
# если явно задать PROGRESS_SPINNER=1. Тесты спиннера (TestProgressSpinner)
# проверяют его поведение через _stdout_write, не через реальный stdout.
os.environ.setdefault("PROGRESS_SPINNER", "0")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
