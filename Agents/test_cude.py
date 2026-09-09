import os
import subprocess

print("=== Диагностика GPU в WSL ===\n")

# 1. Проверяем переменную CUDA_VISIBLE_DEVICES
print(
    f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'НЕ УСТАНОВЛЕНА')}\n"
)

# 2. Проверяем, что nvidia-smi работает
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    print("GPU, видимые через nvidia-smi:")
    print(result.stdout)
except Exception as e:
    print(f"Ошибка при вызове nvidia-smi: {e}\n")

# 3. Проверяем через PyTorch (если установлен)
try:
    import torch

    print(f"PyTorch видит CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Количество GPU: {torch.cuda.device_count()}")
        print(f"Текущий GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch не установлен. Установи: pip install torch")
