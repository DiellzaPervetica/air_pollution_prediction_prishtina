import matplotlib
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PHASE1_DATA_DIR = BASE_DIR / "data" / "phase_1"
PHASE1_PICTURES_DIR = BASE_DIR / "pictures" / "phase_1"

INPUT = PHASE1_DATA_DIR / "2D_validated_final_dataset.csv"

POLLUTANTS = ["co", "no2", "o3", "pm10", "pm25", "so2"]

ENERGY_FEATURES = ["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW", "total_generation_mw"]
WEATHER_FEATURES = [
    "temperature_2m (°C)",
    "rain (mm)",
    "snowfall (cm)",
    "relative_humidity_2m (%)",
    "wind_direction_10m (°)",
    "wind_speed_10m (km/h)",
]

if __name__ == "__main__":
    df = pd.read_csv(INPUT)

    pictures_dir = PHASE1_PICTURES_DIR
    pictures_dir.mkdir(parents=True, exist_ok=True)

    print("\nCandidate pollutant columns:", POLLUTANTS)

    summary_stats = df[POLLUTANTS].describe().T
    print("\n=== Pollutant summary statistics ===")
    print(summary_stats)

    predictors = ENERGY_FEATURES + WEATHER_FEATURES
    subset = df[POLLUTANTS + predictors].dropna()
    corr = subset[POLLUTANTS + predictors].corr()

    corr_pollutant_predictors = corr.loc[POLLUTANTS, predictors].round(3)
    print("\n=== Correlation with energy + weather predictors ===")
    print(corr_pollutant_predictors)

    plt.figure(figsize=(12,6))
    sns.heatmap(corr_pollutant_predictors, annot=True, cmap="coolwarm", center=0)
    plt.title("Pollutant vs Energy & Weather Predictors Correlation")
    
    plt.savefig(pictures_dir / "pollutant_vs_predictors_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    corr_pollutant = corr.loc[POLLUTANTS, POLLUTANTS].round(3)
    print("\n=== Pollutant <-> Pollutant correlation ===")
    print(corr_pollutant)

    plt.figure(figsize=(8,6))
    sns.heatmap(corr_pollutant, annot=True, cmap="coolwarm", center=0)
    plt.title("Pollutant <-> Pollutant Correlation")
    
    plt.savefig(pictures_dir / "pollutant_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
