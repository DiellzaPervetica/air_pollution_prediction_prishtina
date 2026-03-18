import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "1E3_gases_imputed.csv")

df = df.ffill().bfill()

output_path = BASE_DIR / "data" / "1B_cleaned_data_hourly_2023_2025.csv"
df.to_csv(output_path, index=False)

print(f"Procesi përfundoi. Dataseti final i pastruar: {output_path}")
print("Vlera Null të mbetura:", df.isnull().sum().sum())