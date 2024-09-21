import pandas as pd

# df = pd.read_parquet('ohlc_data.parquet 20-54-40-843.parquet')
df = pd.read_parquet('data/MC_2023-01-01_to_2024-01-01.parquet')
print(df.isnull().sum())