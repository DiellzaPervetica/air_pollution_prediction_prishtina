import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

air_path = RAW_DATA_DIR / "prishtina_air_quality_2023_2025.csv"
weather_path = RAW_DATA_DIR / "prishtina_weather_2023_2026.csv"
energy_path = RAW_DATA_DIR / "prishtina_energy_production_2023_2026.csv"
output_path = DATA_DIR / "1A_merged_data_hourly_2023_2025.csv"

air = pd.read_csv(air_path)
weather = pd.read_csv(weather_path, skiprows=3)
energy_raw = pd.read_csv(energy_path, header=None)

header_idx = None
for i in range(min(10, len(energy_raw))):
    row_text = " ".join(map(str, energy_raw.iloc[i].tolist())).lower()
    if "hour" in row_text and "date" in row_text:
        header_idx = i
        break

if header_idx is None:
    raise ValueError("Header row for energy dataset was not found.")

energy = energy_raw.iloc[header_idx:].copy()
energy.columns = energy.iloc[0]
energy = energy.iloc[1:].copy()
energy.columns = [" ".join(str(col).replace("\n", " ").split()) for col in energy.columns]

energy = energy.rename(columns={
    "Ora Hour": "hour",
    "A3 (MW)": "A3_MW",
    "A4 (MW)": "A4_MW",
    "A5 (MW)": "A5_MW",
    "B1 (MW)": "B1_MW",
    "B2 (MW)": "B2_MW",
    "DATA Date": "date"
})

air["datetime"] = pd.to_datetime(air["datetime"], errors="coerce", utc=True)
air["datetime"] = air["datetime"].dt.tz_convert("Europe/Belgrade").dt.tz_localize(None)
air = air.dropna(subset=["datetime"])
air = air.drop_duplicates(subset=["datetime"])
air = air.sort_values("datetime").reset_index(drop=True)

weather = weather.rename(columns={"time": "datetime"})
weather.columns = [" ".join(str(col).split()) for col in weather.columns]

weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce", utc=True)
weather["datetime"] = weather["datetime"].dt.tz_convert("Europe/Belgrade").dt.tz_localize(None)

weather = weather.dropna(subset=["datetime"])
weather = weather.drop_duplicates(subset=["datetime"])
weather = weather.sort_values("datetime").reset_index(drop=True)

if "date" not in energy.columns or "hour" not in energy.columns:
    raise ValueError(f"Energy columns after cleaning: {list(energy.columns)}")

energy["date"] = pd.to_datetime(energy["date"], dayfirst=True, errors="coerce")
energy["hour"] = pd.to_numeric(energy["hour"], errors="coerce")

for col in ["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW"]:
    energy[col] = pd.to_numeric(energy[col], errors="coerce")

energy = energy.dropna(subset=["date", "hour"])
energy["hour_zero_based"] = energy["hour"] - 1
energy["datetime"] = energy["date"] + pd.to_timedelta(energy["hour_zero_based"], unit="h")
energy["datetime"] = energy["datetime"] + pd.Timedelta(hours=1)
energy["total_generation_mw"] = energy[["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW"]].sum(axis=1)

energy = energy.drop(columns=["hour_zero_based"])
energy = energy.drop_duplicates(subset=["datetime"])
energy = energy.sort_values("datetime").reset_index(drop=True)

air = air[["datetime", "co", "no2", "o3", "pm10", "pm25", "so2"]]
weather = weather[
    [
        "datetime",
        "temperature_2m (°C)",
        "rain (mm)",
        "snowfall (cm)",
        "relative_humidity_2m (%)",
        "wind_direction_10m (°)",
        "wind_speed_10m (km/h)"
    ]
]
energy = energy[
    [
        "datetime",
        "A3_MW",
        "A4_MW",
        "A5_MW",
        "B1_MW",
        "B2_MW",
        "total_generation_mw"
    ]
]

merged = air.merge(weather, on="datetime", how="inner")
merged = merged.merge(energy, on="datetime", how="inner")
merged = merged.sort_values("datetime").reset_index(drop=True)
merged["date"] = merged["datetime"].dt.date
merged["hour"] = merged["datetime"].dt.hour
merged["interval_start"] = merged["datetime"] - pd.Timedelta(hours=1)

final_cols = [
    "datetime",
    "date",
    "hour",
    "interval_start",
    "A3_MW",
    "A4_MW",
    "A5_MW",
    "B1_MW",
    "B2_MW",
    "total_generation_mw",
    "co",
    "no2",
    "o3",
    "pm10",
    "pm25",
    "so2",
    "temperature_2m (°C)",
    "rain (mm)",
    "snowfall (cm)",
    "relative_humidity_2m (%)",
    "wind_direction_10m (°)",
    "wind_speed_10m (km/h)"
]

merged = merged[final_cols]

print("Air quality range:", air["datetime"].min(), "->", air["datetime"].max())
print("Weather range:", weather["datetime"].min(), "->", weather["datetime"].max())
print("Energy range:", energy["datetime"].min(), "->", energy["datetime"].max())
print("Merged shape:", merged.shape)
print(merged.head())

merged.to_csv(output_path, index=False)
print(output_path)