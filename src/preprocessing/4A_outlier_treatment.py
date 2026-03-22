import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "data" / "3B_engineered_dataset.csv"
OUTPUT = BASE_DIR / "data" / "4A_outliers_handled.csv"

NON_FEATURE_COLS = {"datetime", "date"}

EXCLUDED_COLS = {
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "wind_x_vector",
    "wind_y_vector"
}

LOWER_Q = 0.005
UPPER_Q = 0.995

df = pd.read_csv(INPUT)

candidate_cols = [
    col for col in df.columns
    if col not in NON_FEATURE_COLS
    and col not in EXCLUDED_COLS
    and pd.api.types.is_numeric_dtype(df[col])
]

summary = []

for col in candidate_cols:
    original = df[col]

    lower = original.quantile(LOWER_Q)
    upper = original.quantile(UPPER_Q)

    low_count = int((original < lower).sum())
    high_count = int((original > upper).sum())

    df[col] = original.clip(lower=lower, upper=upper)

    summary.append({
        "feature": col,
        "capped_low": low_count,
        "capped_high": high_count,
        "total_capped": low_count + high_count
    })

summary_df = pd.DataFrame(summary).sort_values(
    by="total_capped",
    ascending=False
)

df.to_csv(OUTPUT, index=False)

print("Outlier capping completed.")
print(f"Dataset saved: {OUTPUT.name}")
print(f"Shape: {df.shape}")

print("\nTop features with most capped values:")
print(summary_df.head(10))