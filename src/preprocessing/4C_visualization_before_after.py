import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_BEFORE = BASE_DIR / "data" / "3B_engineered_dataset.csv"
INPUT_OUTLIERS = BASE_DIR / "data" / "4A_outliers_handled.csv"
INPUT_SKEWNESS = BASE_DIR / "data" / "4B_skewness_handled.csv"

PLOTS_DIR = BASE_DIR / "pictures" / "4C_visualization_before_after"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "pm25",
    "total_generation_mw",
    "pollution_stagnation_index",
    "rain (mm)",
    "temp_wind_interact"
]

def clean_filename(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("°", "deg")
        .replace(".", "")
        .replace("-", "_")
        .lower()
    )

df_before = pd.read_csv(INPUT_BEFORE)
df_outliers = pd.read_csv(INPUT_OUTLIERS)
df_skewness = pd.read_csv(INPUT_SKEWNESS)

for col in FEATURES:
    if col not in df_before.columns or col not in df_outliers.columns or col not in df_skewness.columns:
        continue

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df_before[col].dropna(), bins=50)
    axes[0].set_title("Before")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frequency")

    axes[1].hist(df_outliers[col].dropna(), bins=50)
    axes[1].set_title("After Outliers")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("Frequency")

    axes[2].hist(df_skewness[col].dropna(), bins=50)
    axes[2].set_title("After Skewness")
    axes[2].set_xlabel(col)
    axes[2].set_ylabel("Frequency")

    fig.suptitle(col)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{clean_filename(col)}_distribution_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

print(f"Saved plots in: {PLOTS_DIR}")