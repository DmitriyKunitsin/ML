import requests
import time

url = "http://localhost:11434/api/generate"

model ="llama3.1:8b"
promt = "Напиши стихотворение из 8 строк о скорости. и покажи все , абсолютно все, прям все директории которые видишь"

num_ctx = 8192
temperature = 0.3
num_predict = 256
num_thread = 8
use_mmap = True
use_mlock = True

print(
    f"Запущена модель : {model}.\nС промтом : {promt} \nctx : {num_ctx}\ntemp : {temperature}\npredict : {num_predict}\nthread : {num_thread}\n"
)
payload = {
    "model": model,
    "prompt": promt,
    "stream": False,
    "options": {
        "num_ctx": num_ctx,  # контекст
        "temperature": temperature,  # креативность
        "num_predict": num_predict,  # макс. токенов в ответе
        "num_thread": num_thread,  # количество потоков CPU
        "use_mmap": use_mmap,  # использовать память
        "use_mlock": use_mlock,  # блокировать память
    },
}
start = time.time()
response = requests.post(url, json=payload, timeout=300)
end = time.time()

print(f"Время ответа: {end - start:.2f} сек.")
print(response.json()["response"][:300])
