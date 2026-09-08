import requests
import time

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3.1:8b",
    "prompt": "Напиши стихотворение из 8 строк о скорости.",
    "stream": False,
}

start = time.time()
response = requests.post(url, json=payload, timeout=300)
end = time.time()

print(f"Время ответа: {end - start:.2f} сек.")
print(response.json()["response"][:300])
