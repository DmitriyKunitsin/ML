"""Тесты проверки и очистки кода (utils.helpers + check_code_syntax из test_main).

Проверяются:
1. Helper.clean_code — снятие markdown-обёрток, включая реальные ответы LLM.
2. Helper.validate_syntax_python — обнаружение синтаксических ошибок.
3. test_main.check_code_syntax — маршрутизация по целевому языку.
4. Совместимость с реальным форматом ответа кодера (код + <verdict>...).

Внешних зависимостей нет: только stdlib (unittest).
"""

import ast
import unittest

from test_main import TARGET_LANG, check_code_syntax
from utils.helpers import Helper


# =============================================================================
# Эталонные исходники
# =============================================================================

# Минимальный, но полноценный модуль в духе задачи из MY_PROMPT
# (сканер проекта -> markdown для Obsidian).
VALID_PROJECT_SCANNER = '''"""Сканер проекта для генерации базы знаний Obsidian."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path

IGNORED_DIRS = {".git", "node_modules", "venv", "__pycache__", ".venv"}
SUPPORTED_SUFFIXES = {".py", ".js", ".ts", ".go"}


@dataclass
class FileInfo:
    """Описание одного исходного файла проекта."""

    path: Path
    docstring: str = ""
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)


def iter_source_files(root: Path) -> list[Path]:
    """Рекурсивно собирает файлы поддерживаемых языков, пропуская мусор."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix in SUPPORTED_SUFFIXES:
                found.append(path)
    return found


def analyze_python_file(path: Path) -> FileInfo:
    """Разбирает Python-файл: docstring, классы, функции, импорты."""
    info = FileInfo(path=path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return info

    info.docstring = ast.get_docstring(tree) or ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            info.classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.functions.append(node.name)
        elif isinstance(node, ast.Import):
            info.imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            info.imports.add(node.module.split(".")[0])
    return info


def build_markdown(info: FileInfo, known_files: set[str]) -> str:
    """Генерирует markdown-заметку с вики-ссылками [[Имя_Файла]]."""
    links = sorted(f"[[{name}]]" for name in known_files if name in info.imports)
    lines = [f"# {info.path.name}", "", info.docstring, "", "## Зависимости", ""]
    lines.extend(f"- {link}" for link in links)
    return "\\n".join(lines)
'''

# Модуль, который кодер вполне может вернуть: логика верна, но пропущено
# двоеточие — обязано быть отловлено валидатором.
BROKEN_MISSING_COLON = "def main()\n    return 1\n"

# Неизвестная кодировка: в исходнике литерал-строка с битым байтом.
BROKEN_UNDECODABLE = 'text = "\udcff\udcfe"\nprint(text)\n'

# Синтаксически корректно, но исполнять нельзя: `await` вне функции.
# Именно такие конструкции ловит compile(), а ast.parse() пропускает.
COMPILE_ONLY_ERROR = "result = await fetch_data()\n"

# Валидный Python, но внутри docstring есть markdown-блок с ``` —
# наивная регулярка может отрезать лишнее.
DOCSTRING_WITH_FENCE = '"""Модуль генерации отчёта.\n\n```python\nprint(1)\n```\n"""\n\n\n' \
    'def build_report(project_dir: str) -> str:\n' \
    '    """Возвращает текст отчёта по проекту."""\n' \
    '    return f"report: {project_dir}"\n'

class TestCleanCode(unittest.TestCase):
    """Helper.clean_code: снятие markdown-обёрток."""

    def test_plain_code_is_untouched(self):
        cleaned = Helper.clean_code(VALID_PROJECT_SCANNER)
        self.assertEqual(cleaned, VALID_PROJECT_SCANNER.strip())
        self.assertTrue(cleaned.startswith('"""Сканер проекта'))

    def test_removes_python_fence(self):
        raw = f"```python\n{VALID_PROJECT_SCANNER}```"
        cleaned = Helper.clean_code(raw)
        self.assertNotIn("```", cleaned)
        self.assertTrue(ast.parse(cleaned))

    def test_removes_bare_fence(self):
        raw = f"```\n{VALID_PROJECT_SCANNER}```"
        cleaned = Helper.clean_code(raw)
        self.assertNotIn("```", cleaned)
        self.assertTrue(ast.parse(cleaned))

    def test_removes_fence_with_leading_prose(self):
        # Модели часто добавляют строку-преамбулу перед обёрткой.
        raw = f"Вот код:\n```python\n{VALID_PROJECT_SCANNER}```"
        cleaned = Helper.clean_code(raw)
        self.assertNotIn("```", cleaned)
        # Преамбула не является кодом, поэтому проверяем именно тело модуля.
        self.assertIn("def iter_source_files", cleaned)

    def test_code_with_docstring_fence_stays_parseable(self):
        cleaned = Helper.clean_code(DOCSTRING_WITH_FENCE)
        self.assertTrue(ast.parse(cleaned))
        self.assertIn("build_report", cleaned)

    def test_coder_output_with_verdict_is_not_valid_code(self):
        """Реальный формат ответа кодера: код + <verdict>APPROVED</verdict>.

        CODER_PROMPT требует выводить вердикт, поэтому ответ кодера нельзя
        подавать на валидацию как есть: тег не является Python-кодом.
        Тест фиксирует контракт: вердикт обязан отрезаться в пайплайне.
        """
        raw = f"{VALID_PROJECT_SCANNER}\n<verdict>APPROVED</verdict>\n"
        cleaned = Helper.clean_code(raw)
        ok, error = Helper.validate_syntax_python(cleaned)
        self.assertFalse(ok)
        self.assertIsNotNone(error)


class TestValidateSyntaxPython(unittest.TestCase):
    """Helper.validate_syntax_python: обнаружение ошибок."""

    def test_valid_code_passes(self):
        ok, error = Helper.validate_syntax_python(VALID_PROJECT_SCANNER)
        self.assertTrue(ok)
        self.assertIsNone(error)

    def test_empty_string_passes(self):
        # Пустой файл синтаксически корректен; «пустоту» ловит шаг 6 отдельно.
        ok, error = Helper.validate_syntax_python("")
        self.assertTrue(ok)
        self.assertIsNone(error)

    def test_missing_colon_reports_line(self):
        ok, error = Helper.validate_syntax_python(BROKEN_MISSING_COLON)
        self.assertFalse(ok)
        self.assertIn("SyntaxError", error)
        self.assertIn("line 1", error)

    def test_undecodable_literal_does_not_crash(self):
        # Гарантия отсутствия падения на непечатаемых символах.
        ok, error = Helper.validate_syntax_python(BROKEN_UNDECODABLE)
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(error, (str, type(None)))

    def test_compile_only_error_is_currently_missed(self):
        """Документируем ограничение: ast.parse() пропускает 'await' вне функции.

        Строгий разбор через compile() такую ошибку находит, поэтому тест
        фиксирует разницу между парсингом и компиляцией.
        """
        ok, _ = Helper.validate_syntax_python(COMPILE_ONLY_ERROR)
        self.assertTrue(ok)  # текущее поведение
        with self.assertRaises(SyntaxError):
            compile(COMPILE_ONLY_ERROR, "<coder_output>", "exec")


class TestCheckCodeSyntax(unittest.TestCase):
    """test_main.check_code_syntax: маршрутизация по TARGET_LANG."""

    def test_target_lang_default_is_python(self):
        # MY_PROMPT описывает Python/Obsidian-скрипт, не Arduino.
        self.assertEqual(TARGET_LANG, "python")

    def test_python_target_routes_to_python_validator(self):
        ok, error = check_code_syntax(VALID_PROJECT_SCANNER)
        self.assertTrue(ok)
        self.assertIsNone(error)

        ok, error = check_code_syntax(BROKEN_MISSING_COLON)
        self.assertFalse(ok)
        self.assertIn("SyntaxError", error)

    def test_cpp_target_does_not_require_avr_toolchain(self):
        """При TARGET_LANG=cpp без avr-g++ проверка не блокирует пайплайн."""
        import test_main

        original = test_main.TARGET_LANG
        test_main.TARGET_LANG = "cpp"
        try:
            ok, error = check_code_syntax("void setup() {}\nvoid loop() {}\n")
        finally:
            test_main.TARGET_LANG = original
        self.assertTrue(ok)
        self.assertIsNone(error)

    def test_result_extension_follows_target_lang(self):
        from test_main import CODE_EXTENSIONS

        self.assertEqual(CODE_EXTENSIONS["python"], "py")
        self.assertEqual(CODE_EXTENSIONS["cpp"], "ino")


if __name__ == "__main__":
    unittest.main()
