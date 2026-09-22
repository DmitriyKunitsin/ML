import logging
import os
import tempfile
import subprocess
import ast
import re

logger = logging.getLogger(__name__)


class Helper:

    @staticmethod
    def compile_arduino_sketch(code: str) -> tuple[bool, str | None]:
        """Компилирует Arduino-скетч через avr-g++ по локальным путям ядра."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "main.cpp")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(code)

            cmd = [
                "avr-g++",
                "-mmcu=atmega328p",
                "-Os",
                "-DF_CPU=16000000L",
                "-c",
                filepath,
                "-o",
                os.path.join(tmpdir, "main.o"),
                "-I",
                os.path.expanduser(
                    "~/.arduino15/packages/arduino/hardware/avr/1.8.6/cores/arduino"
                ),
                "-I",
                os.path.expanduser(
                    "~/.arduino15/packages/arduino/hardware/avr/1.8.6/variants/standard"
                ),
            ]

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    return True, None
                # stderr avr-g++ бывает длинным: обрезаем, чтобы лог не распухал.
                logger.debug(
                    "🔧 avr-g++ вернул код %d: %s",
                    result.returncode,
                    (result.stderr or "")[:500],
                )
                return False, result.stderr
            except Exception:
                logger.exception("❌ Не удалось запустить avr-g++.")
                return False, "avr-g++: не удалось запустить компилятор"

    @staticmethod
    def clean_code(code: str) -> str:
        """Убирает markdown-обёртки ```python ... ``` и другой мусор из кода.

        Если markdown-обёрток НЕТ — код возвращается как есть (без изменения):
        иначе валидный код (с пустыми строками) «схлопывается», и тесты,
        ожидающие нетронутую строку, ломаются. «Фарш» с вложенными ``` лечится
        только тогда, когда они реально есть.
        """
        if not code:
            return ""

        marker_re = r"^\s*```[a-zA-Z0-9_+-]*\s*$"
        has_fence = re.search(marker_re, code, flags=re.MULTILINE)
        if not has_fence:
            # Нет markdown-обёрток — не трогаем код: парсер и так его примет.
            return code.strip()

        # Убираем все строки-фенсы (открывающие и закрывающие), где бы они
        # ни были — внутри кода, в комментарии или в «фарше».
        code = re.sub(marker_re, "\n", code, flags=re.MULTILINE)
        # Оставшийся вертикальный мусор схлопываем до одного переноса: в
        # «фарше» после вырезания ``` остаются тройные переносы.
        code = re.sub(r"\n{3,}", "\n\n", code).strip()

        # Если модель завернула «Вот код:» — редко, но безопасно отрезать
        # только самый первый явный заголовок, если за ним идёт код.
        code = re.sub(r"^(Вот код|Код|```)\s*:?\s*\n", "", code).strip()

        return code.strip()

    @staticmethod
    def is_response_complete(text: str) -> bool:
        """Проверяет, не оборвался ли ответ модели.

        Это «страховка» для провайдеров, которые не сообщают finish_reason:
        если текст физически не завершён (оборвался посреди конструкции),
        его нельзя принимать как полноценный код. В сочетании с проверкой
        ``LLMResponse.truncated`` в шаге 5 это решает проблему «молчаливой
        обрезки».

        Дискриминаторы незавершённости:
          - модель начала тег ``<verdict>``, но не закрыла его ``</verdict>``
            (в 44/50 реальных обрезов ответ обрывается именно в этом месте);
          - в хвосте текста незакрытая скобка ([{()] глубже, чем закрывающих);
          - в хвосте одна (нечётная) тройная кавычка — открытый docstring;
          - нечётное число markdown-фенсов ``` в хвосте (незакрытый блок).

        Полный валидный код (даже большой) эти маркеры не даёт: скобки и
        кавычки сбалансированы, теги вердикта либо отсутствуют, либо закрыты.
        """
        if not text or not text.strip():
            return False

        lowered = text.lower()
        if "<verdict" in lowered and "</verdict>" not in lowered:
            return False

        tail = text[-120:]
        depth = tail.count("(") + tail.count("[") + tail.count("{")
        closing = tail.count(")") + tail.count("]") + tail.count("}")
        if depth > closing:
            return False
        if tail.count('"""') % 2 == 1 or tail.count("'''") % 2 == 1:
            return False
        if tail.count("```") % 2 == 1:
            return False
        return True

    @staticmethod
    def validate_syntax_python(code_string: str) -> tuple[bool, str | None]:
        """Валидация пайтона"""
        try:
            ast.parse(code_string)
            logger.debug("🔧 Синтаксис Python корректен (%d симв.).", len(code_string))
            return True, None
        except SyntaxError as e:
            # Возвращает точное место: "SyntaxError: invalid syntax (line 12)"
            logger.debug("🔧 Синтаксическая ошибка: %s (line %s)", e.msg, e.lineno)
            return False, f"SyntaxError: {e.msg} (line {e.lineno})"
        except ValueError as e:
            # ast.parse/compile падают не только на SyntaxError:
            # - UnicodeEncodeError (битая кодировка, суррогаты) — подкласс ValueError;
            # - "source code string cannot contain null bytes".
            # Без этой ветки пайплайн падает необработанным исключением.
            logger.warning("⚠️ Код не удалось разобрать: %s: %s", type(e).__name__, e)
            return False, f"{type(e).__name__}: {e}"
