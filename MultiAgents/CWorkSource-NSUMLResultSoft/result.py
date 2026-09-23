import argparse
import logging
import sys
from pathlib import Path
from typing import List, Set, Iterator, Tuple, Optional, Sequence, Dict, DefaultDict
from collections import defaultdict
from dataclasses import dataclass, field
import fnmatch
import re
import ast

# ---------- cli.py ----------

def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Obsidian Knowledge Base from source code."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Root directory of the project to scan"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./obsidian_vault"),
        help="Output directory for the generated vault (default: ./obsidian_vault)"
    )
    parser.add_argument(
        "--ignore",
        type=str,
        default="",
        help="Comma‑separated list of glob patterns to ignore"
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Use flat file structure instead of hierarchical"
    )
    parser.add_argument(
        "--max-size",
        type=positive_int,
        default=10,
        help="Maximum file size in MB (default: 10)"
    )
    args = parser.parse_args(argv)
    ignore_list = [p.strip() for p in args.ignore.split(",") if p.strip()]
    args.ignore = ignore_list
    return args

# ---------- scanner.py ----------

logger = logging.getLogger(__name__)

EXCLUDED_DIRS: Set[str] = {
    ".git", "node_modules", "venv", "__pycache__",
    "dist", "build", ".svn", ".hg"
}

@dataclass
class FileInfo:
    relative_path: Path
    extension: str
    full_path: Path
    size: int

def scan_files(
    project_root: Path,
    ignore_patterns: List[str],
    supported_extensions: Set[str],
    max_size_bytes: int
) -> Iterator[FileInfo]:
    for entry in project_root.rglob("*"):
        if not entry.is_file():
            continue
        try:
            relative = entry.relative_to(project_root)
        except ValueError:
            logger.debug("Cannot compute relative path for %s, skipping", entry)
            continue
        parts = relative.parts
        if any(part.lower() in EXCLUDED_DIRS for part in parts):
            continue
        ext = entry.suffix.lower()
        if ext not in supported_extensions:
            continue
        try:
            size = entry.stat().st_size
        except OSError as e:
            logger.warning("Cannot stat %s: %s, skipping", entry, e)
            continue
        if size > max_size_bytes:
            continue
        try:
            with open(entry, "rb") as f:
                header = f.read(512)
            if b"\x00" in header:
                continue
        except OSError:
            continue
        rel_str = str(relative).replace("\\", "/")
        if any(fnmatch.fnmatch(rel_str, pat) for pat in ignore_patterns):
            continue
        yield FileInfo(
            relative_path=relative,
            extension=ext,
            full_path=entry.resolve(),
            size=size
        )

# ---------- parser.py ----------

@dataclass
class FunctionInfo:
    name: str
    lineno: int

@dataclass
class ClassInfo:
    name: str
    lineno: int

@dataclass
class ImportInfo:
    name: str
    source: str
    is_local: Optional[bool] = None

@dataclass
class FileMetadata:
    file_info: "FileInfo"
    docstring: str
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)

def detect_encoding(file_path: Path) -> str:
    try:
        with open(file_path, "rb") as f:
            raw = f.read(4096)
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        import chardet
    except ImportError:
        print("ERROR: chardet package is required. Install it with: pip install chardet")
        sys.exit(1)
    with open(file_path, "rb") as f:
        raw = f.read(32768)
    result = chardet.detect(raw)
    if result["encoding"] and result["confidence"] > 0.8:
        return result["encoding"]
    else:
        return "utf-8"

def _strip_comments_js_ts(content: str) -> str:
    content = re.sub(r"//[^\n]*", "", content)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    return content

def _strip_comments_go(content: str) -> str:
    content = re.sub(r"//[^\n]*", "", content)
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    return content

def _parse_python(content: str) -> Tuple[str, List[ImportInfo], List[FunctionInfo], List[ClassInfo]]:
    try:
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree) or "No documentation found"
        imports: List[ImportInfo] = []
        functions: List[FunctionInfo] = []
        classes: List[ClassInfo] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(name=alias.name, source=f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    imports.append(ImportInfo(name=full_name, source=f"from {module} import {alias.name}"))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(FunctionInfo(name=node.name, lineno=node.lineno))
            elif isinstance(node, ast.ClassDef):
                classes.append(ClassInfo(name=node.name, lineno=node.lineno))
        return docstring, imports, functions, classes
    except SyntaxError:
        docstring = "No documentation found"
        imports: List[ImportInfo] = []
        for match in re.finditer(
            r"^(?:from\s+(\S+)\s+import\s+(\S+(?:\s*,\s*\S+)*)|import\s+(\S+(?:\s*,\s*\S+)*))",
            content, re.MULTILINE
        ):
            if match.group(1) is not None:
                module = match.group(1)
                names_str = match.group(2)
                names = [n.strip() for n in names_str.split(",")]
                for name in names:
                    imports.append(ImportInfo(
                        name=f"{module}.{name}",
                        source=match.group(0).strip()
                    ))
            else:
                names_str = match.group(3)
                names = [n.strip() for n in names_str.split(",")]
                for name in names:
                    imports.append(ImportInfo(name=name, source=match.group(0).strip()))
        return docstring, imports, [], []

def _parse_js_ts(content: str) -> Tuple[str, List[ImportInfo], List[FunctionInfo], List[ClassInfo]]:
    docstring = "No documentation found"
    imports: List[ImportInfo] = []
    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []

    cleaned = _strip_comments_js_ts(content)

    for match in re.finditer(r'''import\s+.*?\s+from\s+['"]([^'"]+)['"]''', cleaned):
        imports.append(ImportInfo(name=match.group(1), source=match.group(0).strip()))
    for match in re.finditer(r'''require\s*\(\s*['"]([^'"]+)['"]\s*\)''', cleaned):
        imports.append(ImportInfo(name=match.group(1), source=match.group(0).strip()))

    for match in re.finditer(r"function\s+(\w+)\s*\(", cleaned):
        functions.append(FunctionInfo(name=match.group(1), lineno=0))
    for match in re.finditer(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(.*?\)\s*=>", cleaned):
        functions.append(FunctionInfo(name=match.group(1), lineno=0))

    for match in re.finditer(r"class\s+(\w+)", cleaned):
        classes.append(ClassInfo(name=match.group(1), lineno=0))

    return docstring, imports, functions, classes

def _parse_go(content: str) -> Tuple[str, List[ImportInfo], List[FunctionInfo], List[ClassInfo]]:
    docstring = "No documentation found"
    imports: List[ImportInfo] = []
    functions: List[FunctionInfo] = []
    classes: List[ClassInfo] = []

    cleaned = _strip_comments_go(content)

    multi_match = re.search(r"import\s*\(([^)]*)\)", cleaned, re.DOTALL)
    if multi_match:
        inner = multi_match.group(1)
        for line in inner.split("\n"):
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                imports.append(ImportInfo(name=line.strip('"'), source=f'import {line}'))
            elif '"' in line:
                parts = line.split('"')
                if len(parts) >= 2:
                    path = parts[1]
                    imports.append(ImportInfo(name=path, source=line.strip()))
    for match in re.finditer(r'^import\s+"([^"]+)"', cleaned, re.MULTILINE):
        imports.append(ImportInfo(name=match.group(1), source=match.group(0).strip()))

    for match in re.finditer(r"func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(", cleaned):
        functions.append(FunctionInfo(name=match.group(1), lineno=0))

    for match in re.finditer(r"type\s+(\w+)\s+struct\s*\{", cleaned):
        classes.append(ClassInfo(name=match.group(1), lineno=0))

    return docstring, imports, functions, classes

_PARSERS = {
    ".py": _parse_python,
    ".js": _parse_js_ts,
    ".ts": _parse_js_ts,
    ".go": _parse_go,
}

def parse_file(file_info: "FileInfo") -> FileMetadata:
    if file_info.size == 0:
        return FileMetadata(
            file_info=file_info,
            docstring="Empty file",
            functions=[],
            classes=[],
            imports=[]
        )

    encoding = detect_encoding(file_info.full_path)
    try:
        with open(file_info.full_path, "r", encoding=encoding, errors="replace") as f:
            content = f.read()
    except OSError as e:
        return FileMetadata(
            file_info=file_info,
            docstring="No documentation found",
            functions=[],
            classes=[],
            imports=[]
        )
    ext = file_info.extension.lower()
    parser = _PARSERS.get(ext, _parse_python)
    try:
        docstring, imports, functions, classes = parser(content)
    except Exception as e:
        logger.warning("Failed to parse %s: %s, using fallback empty metadata", file_info.relative_path, e)
        docstring = "No documentation found"
        imports, functions, classes = [], [], []
    return FileMetadata(
        file_info=file_info,
        docstring=docstring,
        functions=functions,
        classes=classes,
        imports=imports
    )

# ---------- generator.py ----------

_EXT_SUFFIX = {
    ".py": "_py",
    ".js": "_js",
    ".ts": "_ts",
    ".go": "_go"
}

def _resolve_link_name(target_relative_path: Path, flat_mode: bool) -> str:
    if flat_mode:
        name = str(target_relative_path).replace("\\", "/").replace("/", "_")
        name = Path(name).stem
        return name
    else:
        rel = str(target_relative_path).replace("\\", "/")
        name = rel[:rel.rfind(".")] if "." in rel else rel
        return name

def _generate_md_content(
    metadata: "FileMetadata",
    all_targets: frozenset[str],
    flat_mode: bool
) -> str:
    fi = metadata.file_info
    lines = []
    lines.append(f"# {Path(fi.relative_path).stem}")
    lines.append("")
    lines.append(f"**Путь:** `{fi.relative_path}`")
    lines.append("")
    lines.append("## Описание")
    lines.append(metadata.docstring)
    lines.append("")
    lines.append("## Зависимости (импорты)")
    if not metadata.imports:
        lines.append("- Нет импортов")
    else:
        for imp in metadata.imports:
            imp_name = imp.name
            as_path = imp_name.replace(".", "/")
            if as_path in all_targets:
                link = f"[[{as_path}]]"
                lines.append(f"- {link}")
            else:
                lines.append(f"- {imp.name}")
    lines.append("")
    lines.append("## Функции")
    if metadata.functions:
        for func in metadata.functions:
            lines.append(f"- `{func.name}` (строка {func.lineno})")
    else:
        lines.append("- Нет функций")
    lines.append("")
    lines.append("## Классы")
    if metadata.classes:
        for cls in metadata.classes:
            lines.append(f"- `{cls.name}` (строка {cls.lineno})")
    else:
        lines.append("- Нет классов")
    lines.append("")
    return "\n".join(lines)

def generate_vault(
    metadata_list: Sequence["FileMetadata"],
    output_dir: Path,
    flat_mode: bool = False
) -> None:
    all_targets: Set[str] = set()
    target_map: Dict[str, "FileMetadata"] = {}
    used_names: Set[str] = set()

    base_groups: DefaultDict[str, List["FileMetadata"]] = defaultdict(list)

    for meta in metadata_list:
        fi = meta.file_info
        base = _resolve_link_name(fi.relative_path, flat_mode)
        base_groups[base].append(meta)

    for base, metas in base_groups.items():
        if flat_mode:
            if len(metas) == 1:
                link_name = base
                used_names.add(link_name)
                all_targets.add(link_name)
                target_map[link_name] = metas[0]
            else:
                ext_groups: DefaultDict[str, List["FileMetadata"]] = defaultdict(list)
                for m in metas:
                    ext_groups[m.file_info.extension.lower()].append(m)
                for ext, ext_metas in ext_groups.items():
                    if len(ext_metas) == 1:
                        candidate = base + _EXT_SUFFIX[ext]
                        final_name = candidate
                        counter = 1
                        while final_name in used_names:
                            final_name = f"{candidate}_{counter}"
                            counter += 1
                        used_names.add(final_name)
                        all_targets.add(final_name)
                        target_map[final_name] = ext_metas[0]
                    else:
                        for idx, m in enumerate(ext_metas, start=1):
                            candidate = base + _EXT_SUFFIX[ext] + f"_{idx}"
                            final_name = candidate
                            counter = 1
                            while final_name in used_names:
                                final_name = f"{candidate}_{counter}"
                                counter += 1
                            used_names.add(final_name)
                            all_targets.add(final_name)
                            target_map[final_name] = m
        else:
            for m in metas:
                link_name = _resolve_link_name(m.file_info.relative_path, flat_mode=False)
                if link_name in used_names:
                    counter = 1
                    while f"{link_name}_{counter}" in used_names:
                        counter += 1
                    link_name = f"{link_name}_{counter}"
                used_names.add(link_name)
                all_targets.add(link_name)
                target_map[link_name] = m

    all_targets_frozen = frozenset(all_targets)

    output_dir.mkdir(parents=True, exist_ok=True)

    for link_name, meta in target_map.items():
        if flat_mode:
            out_path = output_dir / f"{link_name}.md"
        else:
            rel = meta.file_info.relative_path
            parent = rel.parent
            stem = rel.stem
            out_dir = output_dir / parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}.md"

        content = _generate_md_content(meta, all_targets_frozen, flat_mode)
        out_path.write_text(content, encoding="utf-8")

    obsidian_dir = output_dir / ".obsidian"
    obsidian_dir.mkdir(parents=True, exist_ok=True)
    app_config = '{"showInlineTitle": false}'
    (obsidian_dir / "app.json").write_text(app_config, encoding="utf-8")

# ---------- main.py ----------

def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler("code2obsidian.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        args = parse_args()
    except Exception as e:
        logger.error(f"Failed to parse arguments: {e}")
        sys.exit(1)

    project_root = args.input
    output_dir = args.output
    ignore_patterns = args.ignore
    flat_mode = args.flat
    max_size_mb = args.max_size
    max_size_bytes = max_size_mb * 1024 * 1024

    if not project_root.exists() or not project_root.is_dir():
        logger.error(f"Input path does not exist or is not a directory: {project_root}")
        sys.exit(1)

    supported_extensions = {".py", ".js", ".ts", ".go"}

    logger.info(f"Scanning project: {project_root}")
    logger.info(f"Output vault: {output_dir}")
    logger.info(f"Ignore patterns: {ignore_patterns}")
    logger.info(f"Flat mode: {flat_mode}")
    logger.info(f"Max file size: {max_size_mb} MB")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create output directory {output_dir}: {e}")
        sys.exit(1)

    file_infos: List[FileInfo] = []
    try:
        file_infos = list(scan_files(
            project_root, ignore_patterns, supported_extensions, max_size_bytes
        ))
    except Exception as e:
        logger.error(f"Scanning failed: {e}")
        sys.exit(1)
    logger.info(f"Found {len(file_infos)} files to process")

    metadata_list: List[FileMetadata] = []
    skipped = 0
    for fi in file_infos:
        try:
            meta = parse_file(fi)
            metadata_list.append(meta)
        except Exception as e:
            logger.warning(f"Failed to parse {fi.relative_path}: {e}")
            skipped += 1

    logger.info(f"Parsed {len(metadata_list)} files, skipped {skipped}")

    try:
        generate_vault(metadata_list, output_dir, flat_mode)
    except Exception as e:
        logger.error(f"Vault generation failed: {e}")
        sys.exit(1)

    logger.info("Vault generation completed successfully.")
    logger.info(f"Total files processed: {len(metadata_list)}")
    if skipped:
        logger.info(f"Files skipped due to errors: {skipped}")

if __name__ == "__main__":
    main()