import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "1A_merged_data_hourly_2023_2025.csv"
output_dir = BASE_DIR / "data" / "distinct_values"

output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)
energy_cols = ["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW", "total_generation_mw"]

for col in energy_cols:
    distinct_vals = pd.DataFrame(df[col].dropna().unique(), columns=[col])
    distinct_vals = distinct_vals.sort_values(by=col)
    distinct_vals.to_csv(output_dir / f"1B3_distinct_{col}.csv", index=False)