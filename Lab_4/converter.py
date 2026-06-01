import pandas as pd

# Если файл называется house_prices.txt
df = pd.read_csv('house_prices.txt', sep=',', na_values=['?', ''])

# Посмотрим первые строки
print(df.head())
print(df.shape)  # должно быть (1460, 81) — 80 признаков + SalePrice

df.to_csv('house_prices.csv', index=False)