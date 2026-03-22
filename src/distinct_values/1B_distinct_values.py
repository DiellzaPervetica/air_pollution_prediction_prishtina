import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "1A_merged_data_hourly_2023_2025.csv"
output_dir = BASE_DIR / "data" / "1B_distinct_values"

output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

pollution_cols = ["co", "no2", "o3", "pm10", "pm25", "so2"]

weather_cols = [
    "temperature_2m (째C)",
    "rain (mm)",
    "snowfall (cm)",
    "relative_humidity_2m (%)",
    "wind_direction_10m (째)",
    "wind_speed_10m (km/h)"
]

energy_cols = ["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW", "total_generation_mw"]

all_cols = pollution_cols + weather_cols + energy_cols

def clean_name(col):
    return (
        col.replace(" (째C)", "")
           .replace(" (mm)", "")
           .replace(" (cm)", "")
           .replace(" (%)", "")
           .replace(" (째)", "")
           .replace(" (km/h)", "")
           .replace(" ", "_")
           .lower()
    )

for col in all_cols:
    if col in df.columns:
        distinct_vals = pd.DataFrame(df[col].dropna().unique(), columns=[col])
        distinct_vals = distinct_vals.sort_values(by=col)

        file_name = clean_name(col)
        distinct_vals.to_csv(output_dir / f"distinct_{file_name}.csv", index=False)
    else:
        print(f"Kolona '{col}' nuk u gjet ne dataset!")