import os
import tempfile
import subprocess
import ast
import re


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
                return False, result.stderr
            except Exception as e:
                return False, str(e)

    @staticmethod
    def clean_code(code: str) -> str:
        """Убирает markdown-обёртки ```python ... ``` из кода."""
        # Убираем открывающий блок ```python или ```
        code = re.sub(r"^```\w*\s*\n", "", code, flags=re.MULTILINE)
        # Убираем закрывающий блок ```
        code = re.sub(r"\n```\s*$", "", code, flags=re.MULTILINE)
        return code.strip()

    @staticmethod
    def validate_syntax_python(code_string: str) -> tuple[bool, str | None]:
        """Валидация пайтона"""
        try:
            ast.parse(code_string)
            return True, None
        except SyntaxError as e:
            # Возвращает точное место: "SyntaxError: invalid syntax (line 12)"
            return False, f"SyntaxError: {e.msg} (line {e.lineno})"
