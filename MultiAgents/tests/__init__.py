# Пакет тестов. Сделан пакетом, чтобы pytest/unittest корректно добавляли
# корень проекта в sys.path и `import test_main` работал без правок окружения.
#
# test_main печатает эмодзи, а консоль Windows по умолчанию cp1251: без этого
# переключения `python -m unittest discover -s tests -t .` падает на UnicodeEncodeError
# ещё до проверок. Тот же приём используется в main() самого test_main.
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
