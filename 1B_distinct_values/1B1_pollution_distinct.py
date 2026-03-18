import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "1A_merged_data_hourly_2023_2025.csv"
output_dir = BASE_DIR / "data" / "distinct_values"

output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)
pollution_cols = ["co", "no2", "o3", "pm10", "pm25", "so2"]

for col in pollution_cols:
    distinct_vals = pd.DataFrame(df[col].dropna().unique(), columns=[col])
    distinct_vals = distinct_vals.sort_values(by=col)
    distinct_vals.to_csv(output_dir / f"1B1_distinct_{col}.csv", index=False)