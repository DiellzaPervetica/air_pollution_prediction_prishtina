import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "2A_cleaned_no_duplicates.csv"
output_path = BASE_DIR / "data" / "2B_quality_checked.csv"

df = pd.read_csv(input_path)

pollution_cols = ["pm10", "pm25", "co", "no2", "o3", "so2"]
for col in pollution_cols:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        df[col] = df[col].apply(lambda x: np.nan if pd.notnull(x) and x < 0 else x)
        print(f"{col}: {negative_count} vlera negative u kthyen ne NaN")

wind_col = "wind_direction_10m (°)"
if wind_col in df.columns:
    df[wind_col] = df[wind_col].apply(lambda x: x % 360 if pd.notnull(x) else x)
    print("wind_direction_10m (°): u normalizua ne intervalin 0-359")

for col in ["rain (mm)", "snowfall (cm)"]:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        df[col] = df[col].clip(lower=0)
        print(f"{col}: {negative_count} vlera negative u korrigjuan ne 0")

energy_cols = ["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW", "total_generation_mw"]
for col in energy_cols:
    if col in df.columns:
        negative_count = (df[col] < 0).sum()
        df[col] = df[col].clip(lower=0)
        print(f"{col}: {negative_count} vlera negative u korrigjuan ne 0")

if "relative_humidity_2m (%)" in df.columns:
    below_zero = (df["relative_humidity_2m (%)"] < 0).sum()
    above_hundred = (df["relative_humidity_2m (%)"] > 100).sum()
    df["relative_humidity_2m (%)"] = df["relative_humidity_2m (%)"].clip(0, 100)
    print(f"relative_humidity_2m (%): {below_zero + above_hundred} vlera u kufizuan ne intervalin 0-100")

energy_units = ["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW"]
if all(col in df.columns for col in energy_units) and "total_generation_mw" in df.columns:
    original_total = df["total_generation_mw"].copy()
    recalculated_total = df[energy_units].sum(axis=1)
    mismatch_count = (original_total.round(3) != recalculated_total.round(3)).sum()
    print(f"total_generation_mw: {mismatch_count} raste me mospërputhje u korrigjuan")
    df["total_generation_mw"] = recalculated_total

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = df[numeric_cols].round(3)

df.to_csv(output_path, index=False)
print(f"Dataseti final u ruajt ne: {output_path}")