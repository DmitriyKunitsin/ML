#!/usr/bin/env python3
import requests
import json
import os
import time
import threading
import sys
import subprocess
import tempfile
import re

# --- Конфигурация ---
OLLAMA_URL = "http://localhost:11434/api/chat"

# Модели
BOSS_MODEL = "llama3.1:8b-instruct-q4_K_M"
WORKER_MODEL = "codellama:7b-instruct-q4_K_M"
MAX_REVIEW_ATTEMPTS = 300  # Максимальное количество правок кода


def compile_arduino_sketch(code):
    """Компилирует Arduino-скетч через avr-g++."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "main.cpp")
        with open(filepath, "w") as f:
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return True, None
            return False, result.stderr
        except Exception as e:
            return False, str(e)


# --- Функция живого таймера в консоли ---
def track_time(stop_event):
    start_time = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        sys.stdout.write(f"\r   ⏳ Прошло времени: {elapsed:.1f} сек...")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 40 + "\r")
    sys.stdout.flush()


# --- Функция запроса к Ollama ---
def ask_ollama(model, prompt, system_prompt="", task_type="chat"):
    """Отправляет запрос к Ollama с автоматическим выбором параметров и замером времени."""

    if task_type == "short":
        ctx, predict, temp = 8192, 32, 0.5
    elif task_type == "chat":
        ctx, predict, temp = 8192, 64, 0.5
    elif task_type == "long":
        ctx, predict, temp = 65536, 2048, 0.5
    elif task_type == "code":
        ctx, predict, temp = 8192, 4096, 0.2
    elif task_type == "boss":
        ctx, predict, temp = 8192, 2048, 0.3
    elif task_type == "review":
        ctx, predict, temp = 32768, 2048, 0.3
    else:
        ctx, predict, temp = 8192, 64, 0.5

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": ctx,
            "num_predict": predict,
            "temperature": temp,
            "num_thread": 8,
            "use_mmap": True,
            "use_mlock": True,
        },
    }

    print(f"   Модель: {model} ({task_type})")

    stop_event = threading.Event()
    timer_thread = threading.Thread(target=track_time, args=(stop_event,))
    start_call = time.time()
    timer_thread.start()

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()["message"]["content"]
    except Exception as e:
        result = None
        error_msg = e
    finally:
        stop_event.set()
        timer_thread.join()

    total_time = time.time() - start_call

    if result:
        print(f"   ⏱️ Время генерации: {total_time:.2f} сек")
        return result
    else:
        print(
            f"❌ Ошибка при запросе к {model}: {error_msg} (прошло {total_time:.2f} сек)"
        )
        return None


# --- Прогрев моделей ---
print("🔥 Прогрев моделей...")
_ = ask_ollama(BOSS_MODEL, "Привет", task_type="short")
_ = ask_ollama(WORKER_MODEL, "Привет", task_type="short")
print("✅ Модели загружены в VRAM\n")

# --- Шаг 1: Босс генерирует ТЗ ---
print("🧠 Босс (llama3.1) думает над ТЗ...")
boss_prompt = """
Ты — архитектор встроенных систем. Составь ЧЁТКОЕ ТЕХНИЧЕСКОЕ ЗАДАНИЕ (ТЗ) для Arduino-проекта "Умные часы" ТЗ только на код программы.

Функции:
- Отображение времени на LCD-дисплее (16x2).
- Одна кнопка для переключения режимов (время/секундомер).
- Секундомер с точностью до 0.1 секунды.

ТЗ должно содержать:
1. Список необходимых библиотек.
2. Описание логики работы (конечный автомат).
3. Примерный псевдокод для основных функций.
4. Требования по код-стайлу.
5. Требования по архитектуре кода.
"""

tz_text = ask_ollama(BOSS_MODEL, boss_prompt, task_type="boss")
if not tz_text:
    print("❌ Босс не смог сгенерировать ТЗ")
    exit()

print("✅ ТЗ готово.\n")

# --- Шаг 2 и Шаг 3: Цикл разработки и ревью ---
code = ""
feedback = ""
attempt = 1
code_is_perfect = False

while attempt <= MAX_REVIEW_ATTEMPTS:
    print(f"🔨 [Попытка {attempt}/{MAX_REVIEW_ATTEMPTS}] Разработка и проверка кода...")

    if attempt == 1:
        # Первая генерация
        worker_prompt = f"""
Ты — опытный Arduino-разработчик. Напиши полный скетч (.ino) для проекта по этому ТЗ:
{tz_text}

Требования:
- Код должен быть готов к компиляции.
- Добавь комментарии на русском языке.
- Используй библиотеку LiquidCrystal_I2C.
- Выдавай ТОЛЬКО код внутри блока ```cpp ... ```
"""
    else:
        # Исправление ошибок на основе код-ревью
        worker_prompt = f"""
Ты — Arduino-разработчик. Твой предыдущий код не прошёл ревью Тимлида.
Исправь ошибки и перепиши код с учётом замечаний.

Замечания Тимлида:
{feedback}

Оригинальное ТЗ:
{tz_text}

Выдавай ТОЛЬКО исправленный код внутри блока ```cpp ... ```
"""

    # Исполнитель пишет/правит код
    code = ask_ollama(WORKER_MODEL, worker_prompt, task_type="code")
    if not code:
        print("❌ Исполнитель не смог написать код.")
        exit()
    # --- Тестировщик: компиляция ---
    print(f"🧪 Тестировщик компилирует код (Итерация {attempt})...")
    compile_ok, compile_errors = compile_arduino_sketch(code)
    if not compile_ok:
        print(f"❌ Компиляция не удалась, ошибки:\n{compile_errors[:500]}...")
        feedback = f"Код не скомпилировался. Ошибки компилятора:\n{compile_errors}"
        attempt += 1
        continue  # Пропускаем Тимлида, идём на следующую итерацию
    else:
        print("✅ Компиляция успешна! Передаю код Тимлиду.")

    # Тимлид делает ревью
    print(f"🔎 Тимлид проверяет код (Итерация {attempt})...")
    review_prompt = f"""
Ты — опытный тимлид и эксперт по Arduino с 10-летним стажем. Ты проверяешь код на продакшн-уровень.
Ты НЕ принимаешь код, если он не соответствует жёстким критериям.

Проверь этот код по следующим пунктам (каждый пункт должен быть проверен):

1. **Компилируемость:** Все ли библиотеки стандартны для Arduino? Есть ли `#include` для всех используемых классов?
2. **Соответствие ТЗ:** Все ли функции из ТЗ реализованы? Работает ли секундомер с точностью 0.1 секунды?
3. **Архитектура:** Код разделён на функции? Есть ли дублирование? Используются ли константы?
4. **Логика:** Правильно ли реализован конечный автомат? Нет ли гонок состояний?
5. **Безопасность:** Нет ли конфликтов с прерываниями? Не используется ли `delay()`?
6. **Реализуемость:** Можно ли загрузить этот код на реальную плату? Не требует ли он нестандартных библиотек?

Инструкция для ответа:
1. Если код безупречен по ВСЕМ пунктам — напиши строго "ИДЕАЛЬНО".
2. Если есть хотя бы один пункт, где код не соответствует — напиши подробный список замечаний с указанием конкретных строк и предложением исправления.

Исходное ТЗ:
{tz_text}

Код Исполнителя:
{code}
"""

    feedback = ask_ollama(BOSS_MODEL, review_prompt, task_type="review")

    if feedback and "ИДЕАЛЬНО" in feedback.upper():
        print("🎉 Тимлид утвердил код! Проверка пройдена успешно.")
        code_is_perfect = True
        break
    else:
        print(f"⚠️ Тимлид отклонил код и отправил на доработку.")
        print(f"📋 Замечания Тимлида:\n{feedback}\n" + "-" * 40)
        attempt += 1

if not code_is_perfect:
    print(
        f"🚨 Достигнут лимит попыток ({MAX_REVIEW_ATTEMPTS}). Сохраняем последнюю версию кода, хоть она и не идеальна."
    )


# --- Шаг 4: Сохраняем всё в файлы ---
project_dir = "/mnt/c/Work/Source-NSU/Arduino/llama3.1_8b_Project"
os.makedirs(project_dir, exist_ok=True)

with open(f"{project_dir}/TZ.txt", "w", encoding="utf-8") as f:
    f.write(tz_text)

with open(f"{project_dir}/clock.ino", "w", encoding="utf-8") as f:
    f.write(code)

if not code_is_perfect and feedback:
    with open(f"{project_dir}/Review_Feedback.txt", "w", encoding="utf-8") as f:
        f.write(feedback)

print(f"\n✅ Проект сохранён в {project_dir}")
print(f"📄 ТЗ: {project_dir}/TZ.txt")
print(f"📄 Код: {project_dir}/clock.ino")
if os.path.exists(f"{project_dir}/Review_Feedback.txt"):
    print(f"📄 Замечания: {project_dir}/Review_Feedback.txt")
print("\n🎉 Работа завершена!")
