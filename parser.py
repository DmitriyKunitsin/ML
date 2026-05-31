import re
from math import sqrt

def parse_sensor_file(filename):
    """
    Парсит файл с данными датчиков и возвращает массив записей
    """
    records = []

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        # Пропускаем пустые строки
        if not lines[i].strip():
            i += 1
            continue

        # Парсим строку с ds18
        ds18_match = re.match(r'ds18:bytearray\(b\'(.*?)\'\)\s+(\d+):(\d+)\s+temp=([\d.]+)', lines[i])
        if ds18_match:
            record = {
                'ds18_id': ds18_match.group(1),
                'hour': int(ds18_match.group(2)),
                'minute': int(ds18_match.group(3)),
                'ds18_temp': float(ds18_match.group(4))
            }

            # Проверяем следующую строку с данными dht
            if i + 1 < len(lines):
                dht_match = re.match(r'\s+dht\.temperature\(\)=([\d.]+)\s+dht\.humidity\(\)=([\d.]+)', lines[i + 1])
                if dht_match:
                    record['dht_temp'] = float(dht_match.group(1))
                    record['dht_humidity'] = float(dht_match.group(2))

                    # Добавляем запись в массив
                    records.append(record)

            i += 2  # Пропускаем обе строки (ds18 и dht)
        else:
            i += 1

    return records


# Использование
filename = 'temp.txt'  # Замените на имя вашего файла
result = parse_sensor_file(filename)

# Вывод результата для проверки
for record in result:
    print(record)

ds_mean = 0
dht_mean = 0

for i in result:
    t_ds = i["ds18_temp"]
    ds_mean += t_ds
    
    t_dht = i["dht_temp"]
    dht_mean += t_dht

ds_mean /= len(result)
dht_mean /= len(result)

delta_dht = 0
for i in result:
    delta_dht += abs(ds_mean - i["dht_temp"])

i=5
print(f"{len(result)=}")
print(f"{dht_mean=}")
print(f"{ds_mean=}")
print(f"Абсолютная погрешность для {i=} ds: {abs(result[i]["dht_temp"] - ds_mean)}")
print(f"Относительная погрешность для {i=}: {(abs(result[i]["dht_temp"] - ds_mean))/ds_mean}")
sig_sum = 0
print(f"{type(result)=}")
for I in result:
    sig_sum += (I["dht_temp"] - dht_mean) ** 2
    
sig = sig_sum / (len(result) - 1)
sig = sqrt(sig)

print(f"σ: {sig}")

