import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT = BASE_DIR / "data" / "2D_validated_final_dataset.csv"
OUTPUT = BASE_DIR / "data" / "3B_engineered_dataset.csv"

TARGET = "pm25"
ENERGY_COLS = ["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW", "total_generation_mw"]
WEATHER_COLS = [
    "temperature_2m (°C)",
    "rain (mm)",
    "snowfall (cm)",
    "relative_humidity_2m (%)",
    "wind_direction_10m (°)",
    "wind_speed_10m (km/h)",
]

df = pd.read_csv(INPUT)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

LAG_COLS = ["total_generation_mw", "wind_speed_10m (km/h)", "temperature_2m (°C)"]

for col in LAG_COLS:
    for lag in [1, 3, 6]:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

for window in [12, 24]:
    df[f"total_gen_rolling_sum_{window}h"] = df["total_generation_mw"].rolling(window=window, min_periods=window).sum()

df["temp_wind_interact"] = df["temperature_2m (°C)"] * df["wind_speed_10m (km/h)"]
df["generation_humidity_interact"] = df["total_generation_mw"] * df["relative_humidity_2m (%)"]

df["pollution_stagnation_index"] = df["total_generation_mw"] / (df["wind_speed_10m (km/h)"] + 0.1)

wv = df["wind_speed_10m (km/h)"]
wd_rad = df["wind_direction_10m (°)"] * np.pi / 180
df["wind_x_vector"] = wv * np.cos(wd_rad)
df["wind_y_vector"] = wv * np.sin(wd_rad)

df = df.dropna().reset_index(drop=True)

final_cols = [
    "datetime", "date", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "temp_wind_interact", "generation_humidity_interact",
    "pollution_stagnation_index", "wind_x_vector", "wind_y_vector",
    "total_generation_mw"
] + WEATHER_COLS + [c for c in df.columns if "lag_" in c] + [c for c in df.columns if "rolling_" in c] + [TARGET]

final_cols = list(dict.fromkeys(final_cols))
df = df[[c for c in final_cols if c in df.columns]]

df.to_csv(OUTPUT, index=False)

print(df.shape)