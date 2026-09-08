import requests
import json
import time

# --- Конфигурация ---
OLLAMA_URL = "http://localhost:11434/api/chat"
BOSS_MODEL = "llama3.1:8b"
WORKER_MODEL = "codellama:7b-instruct-q4_K_M"  # Исполнитель

def ask_ollama(model, prompt, system_prompt=""):
    """Отправляет запрос к Ollama и возвращает ответ."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3}  # Низкая температура для точных ответов
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()['message']['content']
    except Exception as e:
        print(f"❌ Ошибка при запросе к {model}: {e}")
        return None

# --- Шаг 1: Босс (llama3.1) генерирует ТЗ ---
print("🧠 Босс (llama3.1) думает...")
boss_prompt = """
Ты — архитектор встроенных систем. Составь ЧЁТКОЕ ТЕХНИЧЕСКОЕ ЗАДАНИЕ (ТЗ) для Arduino-проекта "Умные часы".

Функции:
- Отображение времени на LCD-дисплее (16x2).
- Одна кнопка для переключения режимов (время/секундомер).
- Секундомер с точностью до 0.1 секунды.

ТЗ должно содержать:
1. Список необходимых компонентов.
2. Описание логики работы (конечный автомат).
3. Примерный псевдокод для основных функций.
"""

tz_text = ask_ollama(BOSS_MODEL, boss_prompt)
if tz_text:
    print("✅ ТЗ готово:\n", tz_text[:300], "...\n")
else:
    print("❌ Босс не смог сгенерировать ТЗ")
    exit()

# --- Шаг 2: Исполнитель (Codellama) пишет код по ТЗ ---
print("👷 Исполнитель (Codellama) пишет код...")
worker_prompt = f"""
Ты — опытный Arduino-разработчик. Напиши полный скетч (.ino) для проекта по этому ТЗ:

{tz_text}

Требования:
- Код должен быть готов к компиляции.
- Добавь комментарии на русском языке.
- Используй библиотеку LiquidCrystal_I2C.
"""

code = ask_ollama(WORKER_MODEL, worker_prompt)
if code:
    print("✅ Код готов:\n", code[:500], "...\n")
else:
    print("❌ Исполнитель не смог написать код")
    exit()

# --- Шаг 3: Сохраняем всё в файлы ---
import os
project_dir = "/mnt/c/Work/Source-NSU/Arduino/Clock/llama3.1_8b_Project"
os.makedirs(project_dir, exist_ok=True)

with open(f"{project_dir}/TZ.txt", "w", encoding="utf-8") as f:
    f.write(tz_text)

with open(f"{project_dir}/clock.ino", "w", encoding="utf-8") as f:
    f.write(code)

print(f"✅ Проект сохранён в {project_dir}")
print(f"📄 ТЗ: {project_dir}/TZ.txt")
print(f"📄 Код: {project_dir}/clock.ino")