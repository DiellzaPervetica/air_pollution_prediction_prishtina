import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "1D2_wind_quality_checked.csv")

df["rain (mm)"] = df["rain (mm)"].clip(lower=0)
df["snowfall (cm)"] = df["snowfall (cm)"].clip(lower=0)

energy_cols = ["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW", "total_generation_mw"]
for col in energy_cols:
    df[col] = df[col].clip(lower=0)

df.to_csv(BASE_DIR / "data" / "1D3_all_quality_checked.csv", index=False)