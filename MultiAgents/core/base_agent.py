import logging
import time

import tiktoken
import os
from pathlib import Path

from core.base_llm import BaseLLM
from core.llm_types import LLMResponse, coerce_response
from core.logging_setup import (
    _AGENT_HANDLER_MARKER,
    get_run_log_dir,
    preview,
    register_agent_logger,
)
from utils.progress_spinner import notify as spinner_notify

# Принудительно заставляем tiktoken не лезть в сеть, если файл уже скачан
os.environ["TIKTOKEN_CACHE_DIR"] = "./tiktoken_cache"

logger = logging.getLogger(__name__)

SAFE_LIMIT = 2000


def _to_agent_id(name: str) -> str:
    """Транслитерирует имя агента в безопасное имя файла agents/<id>.log."""
    replacements = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
        "А": "a", "Б": "b", "В": "v", "Г": "g", "Д": "d", "Е": "e", "Ё": "e",
        "Ж": "zh", "З": "z", "И": "i", "Й": "y", "К": "k", "Л": "l", "М": "m",
        "Н": "n", "О": "o", "П": "p", "Р": "r", "С": "s", "Т": "t", "У": "u",
        "Ф": "f", "Х": "kh", "Ц": "ts", "Ч": "ch", "Ш": "sh", "Щ": "sch",
        "Ъ": "", "Ы": "y", "Ь": "", "Э": "e", "Ю": "yu", "Я": "ya",
    }
    result = []
    for ch in name:
        if ch.isalnum():
            result.append(replacements.get(ch, ch))
        else:
            result.append("_")
    cleaned = "".join(result).strip("_").lower()
    return cleaned or "agent"


def _find_file_handler(agent_logger: logging.Logger) -> logging.FileHandler | None:
    """Возвращает file-хендлер логгера агента (или None)."""
    for handler in agent_logger.handlers:
        if getattr(handler, _AGENT_HANDLER_MARKER, None) == agent_logger.name:
            return handler
    return None


class BaseAgent:
    """Базовый класс агента, работающий с асбтрактной LLM"""

    def __init__(
        self, name_agent: str, role_prompt: str, llm: BaseLLM, context_limit: int = 8192
    ):
        self.name = name_agent  # Просто имя агента
        self.role_prompt = role_prompt  # роль агента
        self.llm = llm  # LLM
        # Метаданные последнего ответа LLM (finish_reason, токены). Шаг 5
        # использует это, чтобы НЕ принимать обрезанный код как полноценный.
        # Для FakeAgent (возвращает str) значение остаётся None — старые
        # тесты на урезание не срабатывают.
        self.last_response_meta: LLMResponse | None = None
        # Стабильный идентификатор для файла agents/<id>.log и имени логгера.
        self.agent_id = _to_agent_id(name_agent)
        # Отдельный логгер: пишет в свой файл (полные тексты) и НЕ дублирует
        # записи в run.log (propagate=False задаёт register_agent_logger).
        self.agent_logger = logging.getLogger(f"agent.{self.agent_id}")
        register_agent_logger(
            self.agent_logger.name, self.name, file_stem=self.agent_id
        )

    def _agent_log_path(self) -> str | None:
        """Абсолютный путь к файлу лога этого агента (None — если нет)."""
        run_dir = get_run_log_dir()
        if not run_dir:
            return None
        path = Path(run_dir) / "agents" / f"{self.agent_id}.log"
        handler = _find_file_handler(self.agent_logger)
        return str(path) if (handler and path.exists()) else None

    def _file_line_count(self) -> int:
        """Сколько строк уже в файле агента (0 — файла нет)."""
        path = self._agent_log_path()
        if not path:
            return 0
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return sum(1 for _ in fh)
        except OSError:
            return 0

    def _append_agent_protocol(
        self,
        kind: str,
        model: str,
        task_type: str,
        text: str,
        system_prompt: str | None = None,
    ) -> int | None:
        """Пишет в файл агента ПОЛНЫЙ запрос или ответ (без preview).

        Возвращает номер строки, С КОТОРОЙ начинается этот блок («ЗАПРОС») или
        которой блок заканчивается («ОТВЕТ») — для ссылки «см. строки N..M».
        system_prompt передаётся только для запроса (в ответах его нет).
        None — если файла агента нет или текст пустой.
        """
        if not text:
            return None
        if _find_file_handler(self.agent_logger) is None:
            return None
        start = self._file_line_count() + 1
        try:
            with open(self._agent_log_path() or "", "a", encoding="utf-8") as fh:
                header = (
                    f"{'=' * 20} {kind} «{self.name}» "
                    f"(task_type={task_type}, модель={model}) {'=' * 20}"
                )
                fh.write(header + "\n")
                if system_prompt is not None:
                    fh.write(f"--- SYSTEM ---\n{system_prompt}\n")
                fh.write(f"--- USER ---\n{text}\n")
                fh.write(f"{'=' * 88}\n")
        except OSError:
            return None
        if kind == "ОТВЕТ":
            return self._file_line_count()
        return start

    async def execute_task(self, prompt: str, task_type: str = "chat") -> str | None:
        """Выполняет задачу, передавая ее llm"""
        # context_limit = await self.llm._get_context_limit(task_type)
        # if not self.validate_prompt(prompt, context_limit):
        #     raise ValueError(
        #         f"Ошибка: Промпт для агента '{self.name}' слишком огромный ({self.count_tokens(self.role_prompt + prompt)} токенов)! "
        #         f"Он превышает безопасный лимит контекста ({context_limit - SAFE_LIMIT} токенов)."
        #     )
        model = getattr(self.llm, "model_name", type(self.llm).__name__)
        logger.info(
            "🤖 Агент «%s»: старт задачи (task_type=%s, модель=%s).",
            self.name,
            task_type,
            model,
        )
        logger.debug("🤖 Агент «%s»: system-промпт: %s", self.name, preview(self.role_prompt))
        logger.debug("🤖 Агент «%s»: user-промпт: %s", self.name, preview(prompt))
        spinner_notify(self.name, f"Агент «{self.name}» выполняет задачу…")

        # Полный протокол агента (без preview, с сохранением многострочности):
        # живёт в logs/<запуск>/agents/<id>.log. В базовом логе остаётся
        # одна INFO-строка со ссылкой «см. файл агента, строки N..M».
        agent_path = self._agent_log_path()
        start_line = self._append_agent_protocol(
            "ЗАПРОС", model, task_type, prompt, system_prompt=self.role_prompt
        )

        started_at = time.perf_counter()
        try:
            raw_response = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.role_prompt,
                task_type=task_type,
            )
        except Exception:
            # logger.exception сам приложит traceback — руками его собирать не нужно.
            elapsed = time.perf_counter() - started_at
            logger.exception(
                "❌ Агент «%s»: вызов LLM упал за %.2f сек (task_type=%s).",
                self.name,
                elapsed,
                task_type,
            )
            return None

        elapsed = time.perf_counter() - started_at
        # Совместимость: провайдеры возвращают LLMResponse, заглушки тестов —
        # строку. Приводим к единому конверту и сохраняем метаданные: шагу 5
        # нужно знать, не обрезан ли ответ (finish_reason == "length").
        meta = coerce_response(raw_response)
        self.last_response_meta = meta
        response = meta.text
        if not response:
            logger.warning(
                "⚠️ Агент «%s»: пустой ответ LLM за %.2f сек (task_type=%s).",
                self.name,
                elapsed,
                task_type,
            )
            return None

        end_line = self._append_agent_protocol(
            "ОТВЕТ", model, task_type, response, system_prompt=None
        )
        if agent_path is not None and start_line is not None and end_line is not None:
            logger.info(
                "✅ Агент «%s»: ответ получен (%d симв.) за %.2f сек. "
                "Полный запрос/ответ: %s, строки %d–%d.",
                self.name,
                len(response),
                elapsed,
                agent_path,
                start_line,
                end_line,
            )
        else:
            logger.info(
                "✅ Агент «%s»: ответ получен (%d симв.) за %.2f сек.",
                self.name,
                len(response),
                elapsed,
            )
        logger.debug("🤖 Агент «%s»: ответ: %s", self.name, preview(response))
        return response

        """
        architect = BaseAgent(
            name="Архитектор",
            role_prompt="Ты C# системный архитектор. Твоя задача — проектировать структуру классов.",
            llm=llm_provider,
        )
        """

    def count_tokens(sels, text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def validate_prompt(self, prompt: str, context_limit: int) -> bool:
        # Считаем токены системных инструкций + самого запроса
        total_text = self.role_prompt + prompt
        prompt_tokens = self.count_tokens(total_text)

        # Задаем безопасный порог (оставляем минимум 2000 токенов на генерацию ответа модели)
        available_space = context_limit - SAFE_LIMIT

        logger.debug(
            "📊 [%s] Размер запроса: %d токенов. Доступно: %d",
            self.name,
            prompt_tokens,
            available_space,
        )

        if prompt_tokens > available_space:
            logger.warning(
                "⚠️ [%s] Промпт не влезает в контекст: %d > %d токенов.",
                self.name,
                prompt_tokens,
                available_space,
            )
            return False
        return True
