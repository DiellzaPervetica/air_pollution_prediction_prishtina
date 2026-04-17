import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

input_path = BASE_DIR / "data" / "2B_quality_checked.csv"
output_path = BASE_DIR / "data" / "2C_missing_values_handled.csv"

df = pd.read_csv(input_path)

missing_count = df.isnull().sum().reset_index()
missing_count.columns = ["Column", "Missing_Values"]
missing_count["Percentage"] = (missing_count["Missing_Values"] / len(df)) * 100

before_pm10 = df["pm10"].isnull().sum() if "pm10" in df.columns else 0
before_pm25 = df["pm25"].isnull().sum() if "pm25" in df.columns else 0

if "pm10" in df.columns:
    df["pm10"] = df["pm10"].bfill()

if "pm25" in df.columns:
    df["pm25"] = df["pm25"].bfill()

print(f"PM10: U mbushën {before_pm10} vlera.")
print(f"PM25: U mbushën {before_pm25} vlera.")

gases = ["co", "no2", "o3", "so2"]
for col in gases:
    if col in df.columns:
        before_missing = df[col].isnull().sum()
        df[col] = df[col].ffill()
        print(f"{col}: U mbushën {before_missing} vlera.")

df = df.ffill().bfill()

df.to_csv(output_path, index=False)

print(f"Dataseti final u ruajt te: {output_path}")
print("Vlera Null të mbetura:", df.isnull().sum().sum())