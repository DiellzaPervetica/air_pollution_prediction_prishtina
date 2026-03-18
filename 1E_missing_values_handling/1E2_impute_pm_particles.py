import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "1D5_logically_validated.csv")

before_pm10 = df['pm10'].isnull().sum()

df['pm10'] = df['pm10'].bfill()
df['pm25'] = df['pm25'].bfill()

df.to_csv(BASE_DIR / "data" / "1E2_pm_imputed.csv", index=False)
print(f"PM10: U mbushën {before_pm10} vlera.")