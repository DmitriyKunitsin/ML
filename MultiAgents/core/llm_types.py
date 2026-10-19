"""Типы, разделяемые между провайдерами LLM и ядром пайплайна.

Проблема, которую решает этот модуль: «молчаливая обрезка» ответа модели.
Провайдеры (OpenAI-совместимый API, Ollama) отдают ``finish_reason``
("stop" | "length" | ...), но до сих пор пайплайн читал только текст и
принимал обрезанный по ``max_tokens`` ответ как полноценный. Из-за этого
кодер «успешно» возвращал оборванный код без закрывающего ``<verdict>``,
а пайплайн крутился в цикле 5 <-> 6 <-> 7 по 50 итераций.

``LLMResponse`` — это конверт ответа: текст + служебная метаинформация.
Провайдеры возвращают именно его, а ``base_agent`` и шаги стейт-машины
решают, как трактовать метаданные.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """Ответ LLM с метаданными о завершении генерации.

    ``finish_reason`` — почему модель остановилась:
      - "stop" — штатное завершение (запрос выполнен полностью);
      - "length" — упёрлись в ``max_tokens``: ответ ОБРЕЗАН и доверять ему
        нельзя;
      - иное / None — провайдер не дал причину; считаем ответ полным
        (без ложных срабатываний на API, которые не отдают reason).

    ``prompt_tokens`` / ``completion_tokens`` — счётчики из usage (если
    провайдер их присылает); используются для того, чтобы не считать
    токены заново и для диагностики в логах.
    """

    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def truncated(self) -> bool:
        """True, если модель остановилась из-за лимита токенов."""
        return self.finish_reason == "length"

    @property
    def is_empty(self) -> bool:
        """Пустой текст (или только пробельные символы) — считать провалом."""
        return not self.text or not self.text.strip()


def coerce_response(value) -> LLMResponse:
    """Приводит результат генерации к ``LLMResponse``.

    Нужен для обратной совместимости: заглушки агентов в тестах возвращают
    обычную строку, а реальные провайдеры — ``LLMResponse``. Без этой
    функции любой новый код, читающий ``.truncated``, упал бы на ``str``.
    """
    if isinstance(value, LLMResponse):
        return value
    if value is None:
        return LLMResponse(text="")
    return LLMResponse(text=str(value))


def to_text(value) -> str:
    """Достаёт «чистый текст» из ответа (str или LLMResponse)."""
    if isinstance(value, LLMResponse):
        return value.text
    if value is None:
        return ""
    return str(value)