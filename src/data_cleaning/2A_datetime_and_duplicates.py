import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "1A_merged_data_hourly_2023_2025.csv"
output_path = BASE_DIR / "data" / "2A_cleaned_no_duplicates.csv"

df = pd.read_csv(input_path)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

duplicate_count = df.duplicated().sum()
print(f"Numri i duplikateve: {duplicate_count}")

df = df.drop_duplicates().reset_index(drop=True)

df.to_csv(output_path, index=False)