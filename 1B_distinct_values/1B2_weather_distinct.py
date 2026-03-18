import pandas as pd
from pathlib import Path

# RREGULLIMI: Dy here .parent nese skripta eshte brenda nje folderi (si 1B_distinct_values)
BASE_DIR = Path(__file__).resolve().parent.parent
input_path = BASE_DIR / "data" / "1A_merged_data_hourly_2023_2025.csv"
output_dir = BASE_DIR / "data" / "distinct_values"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

# EMRI I SAKTË: Duhet të jetë "temperature_2m (°C)" si në 1A.py
weather_cols = [
    "temperature_2m (°C)", 
    "rain (mm)", 
    "snowfall (cm)", 
    "relative_humidity_2m (%)", 
    "wind_direction_10m (°)", 
    "wind_speed_10m (km/h)"
]

for col in weather_cols:
    if col in df.columns:
        distinct_vals = pd.DataFrame(df[col].dropna().unique(), columns=[col])
        distinct_vals = distinct_vals.sort_values(by=col)
        file_name = col.replace(" (°C)", "").replace(" (mm)", "").replace(" (%)", "").replace(" (°)", "").replace(" (km/h)", "").replace(" ", "_")
        distinct_vals.to_csv(output_dir / f"1B2_distinct_{file_name}.csv", index=False)
    else:
        print(f"Kujdes: Kolona '{col}' nuk u gjet ne dataset!")