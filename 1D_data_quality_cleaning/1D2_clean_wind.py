import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "1D1_pollution_quality_checked.csv")

df["wind_direction_10m (°)"] = df["wind_direction_10m (°)"].apply(lambda x: x % 360 if pd.notnull(x) else x)
df.to_csv(BASE_DIR / "data" / "1D2_wind_quality_checked.csv", index=False)