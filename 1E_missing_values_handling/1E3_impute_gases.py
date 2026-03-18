import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "1E2_pm_imputed.csv")

gases = ['co', 'no2', 'o3', 'so2']
for col in gases:
    df[col] = df[col].ffill()

df.to_csv(BASE_DIR / "data" / "1E3_gases_imputed.csv", index=False)
print("Gazrat u mbushën me sukses.")