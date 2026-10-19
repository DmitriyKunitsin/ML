"""Консольный анимированный индикатор прогресса длинных операций.

Задача: пользователь должен видеть, что пайплайн жив, пока LLM «думает».
Решение — один общий спиннер, живущий в фоновом потоке:

- ``notify(section, message)`` обновляет строку спиннера в консоли (stdout);
- ``stop()`` останавливает поток и дописывает итоговую строку статуса.

Поток пишет анимацию в ``stdout`` (каретка возвращается в начало — ``\\r``,
строка переписывается поверх). Это работает на Windows-консоли, в IDLE и при
перенаправлении в файл без ``curses``. Если вывод не поддерживает ``\\r``
(например, бобы в спиннере ломают cp1251) — кадры молча пропадают, ничего не
роняя. Все важные вехи всё равно дублируются в лог через logging.

Поток НЕ трогает stderr и НЕ пишет в лог-файл: лог остаётся grepable, а каждая
запись — одной строкой. ``notify()`` и ``stop()`` безопасно вызывать из любого
потока и повторно (при повторном запуске реализует тот же спиннер).
"""

from __future__ import annotations

import atexit
import itertools
import sys
import threading
import time

# Как часто перерисовывать анимацию (сек).
_TICK = 0.15
# Наборы кадров: «брайлевский» кёрлинг-спиннер → обычные \ | / -.
_FRAMES = (
    "⠿⠼⠻⠽⠾⠁⠃⠄⠅⠆⠇⠈⠉⠊⠋⠌⠍⠎⠏⠐⠑⠒⠓⠔⠕⠖⠗⠘⠙⠚⠛⠜⠝⠞⠟⠠⠡⠢⠣⠤⠥⠦⠧⠨⠩⠪⠫⠬⠭⠮⠯⠰⠱⠲⠳⠴⠵⠶⠷⠸⠹⠺⠻⠼⠽⠾",
    "\\|/-",
)
_SPINNER_CHARS = itertools.cycle(_FRAMES[0])

PROGRESS_SPINNER_STARTED = False
_stdout_write = None  # перехваченный sys.stdout.write (тесты, REPL-хостинг)
_spinner_lock = threading.Lock()
_stop_event = threading.Event()
_stopped_ack = threading.Event()  # поток сигналит, что закончил рисовать
_last_net_message: dict = {"section": "", "message": "", "at": 0.0}
_shutdown = False

Lock = threading.Lock


def _spinner_frame() -> str:
    return next(_SPINNER_CHARS)


def _should_run() -> bool:
    """Спиннер нужен только на живом терминале или там, где явно перехватили
    вывод (_stdout_write). При перенаправлении в файл/CI не-TTY молча не
    запускаемся: \r-кадры — мусор в протоколе.
    """
    if _stdout_write is not None:
        return True
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _start_spinner() -> None:
    """Запускает фоновый поток-анимацию, если ещё не запущен.

    Вызывается ТОЛЬКО из notify() под _spinner_lock — сам блокировку не берёт:
    threading.Lock не реентерабельный, повторный захват тем же потоком = deadlock.
    """
    global PROGRESS_SPINNER_STARTED, _shutdown

    if PROGRESS_SPINNER_STARTED or _shutdown:
        return
    if not _should_run():
        return
    # Свежий поток — свежее подтверждение остановки: стop() дождётся именно его.
    _stopped_ack.clear()
    threading.Thread(
        target=_spinner_loop,
        name="progress-spinner",
        daemon=True,
    ).start()
    PROGRESS_SPINNER_STARTED = True


def _spinner_loop() -> None:
    """Фоновый цикл: рисует спиннер, перечитывая сообщение из _last_net_message."""
    while not _stop_event.is_set():
        frame = _spinner_frame()
        with _spinner_lock:
            text = _last_net_message["message"] if not _shutdown else ""
            if text and not _shutdown:
                _write_status(frame, text, keep_line=True)
        # Не жрём CPU на 100%: между кадрами пауза.
        _stop_event.wait(_TICK)
    # Быстрый финальный кадр, чтобы пользователь увидел статус без задержки.
    if not _shutdown:
        with _spinner_lock:
            text = _last_net_message["message"]
            if text:
                _write_status(" ", text, keep_line=False)
    # Сигнализируем stop(): финальный кадр нарисован, можно продолжать.
    _stopped_ack.set()


def _write_status(frame: str, text: str, keep_line: bool) -> None:
    """Пишет кадр в stdout: либо перезаписывая строку (\\r), либо отдельной.

    keep_line=True — кадр продолжает текущую строку (\\r + frame + ' | ' + text),
    keep_line=False — финальный статус пишется отдельной строкой без \\r.
    """
    write = _stdout_write or sys.stdout.write
    try:
        if keep_line:
            # \\r — вернуть каретку в начало строки и переписать поверх.
            write(f"\r\033[2K{frame} | {text}")
        else:
            write(f"{frame} | {text}\n")
    except (UnicodeEncodeError, OSError):
        pass
    try:
        sys.stdout.flush()
    except (ValueError, OSError):
        pass


def notify(section: str, message: str) -> None:
    """Обновляет статус спиннера. Вызывается из кода пайплайна.

    Сообщение живёт до нового ``notify()`` или ``stop()``. Частые обновления
    подряд перезаписывают одну строку — экран не заливается.
    """
    global PROGRESS_SPINNER_STARTED, _last_net_message

    if _shutdown:
        return
    with _spinner_lock:
        _last_net_message = {"section": section, "message": message, "at": time.perf_counter()}
        if not PROGRESS_SPINNER_STARTED:
            _start_spinner()


def stop() -> None:
    """Останавливает поток спиннера и печатает итоговую строку текущего статуса."""
    global PROGRESS_SPINNER_STARTED, _shutdown

    if not PROGRESS_SPINNER_STARTED:
        return
    _stop_event.set()
    # Ждём финальный кадр: поток завершает цикл и ставит _stopped_ack.
    # Таймаут — страховка: если поток уже вышел (редкая гонка), не виснуть.
    _stopped_ack.wait(2.0)
    _shutdown = True
    PROGRESS_SPINNER_STARTED = False


# Гарантируем остановку потока даже при выходе через исключение/KeyboardInterrupt.
atexit.register(stop)