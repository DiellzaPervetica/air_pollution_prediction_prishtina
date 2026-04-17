import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import PowerTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT = BASE_DIR / "data" / "4A_outliers_handled.csv"
OUTPUT = BASE_DIR / "data" / "4B_skewness_handled.csv"

NON_FEATURE_COLS = {"datetime", "date"}

EXCLUDED_COLS = {
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "month_sin", "month_cos",
    "low_wind_flag", "rain_flag", "cold_stagnation"
}

SKEW_THRESHOLD = 1.0

df = pd.read_csv(INPUT)
df_transformed = df.copy()

candidate_cols = [
    col for col in df.columns
    if col not in NON_FEATURE_COLS
    and col not in EXCLUDED_COLS
    and pd.api.types.is_numeric_dtype(df[col])
]

skew_before_all = df[candidate_cols].skew()
results = []

for col in candidate_cols:
    original = df[col].copy()
    skew_before = original.skew()
    method = "none"

    if abs(skew_before) > SKEW_THRESHOLD:
        if (original >= 0).all():
            transformed = np.log1p(original)
            method = "log1p"
        else:
            transformer = PowerTransformer(method="yeo-johnson", standardize=False)
            transformed = transformer.fit_transform(original.to_frame()).flatten()
            method = "yeo-johnson"

        df_transformed[col] = transformed
    else:
        df_transformed[col] = original

    skew_after = df_transformed[col].skew()

    results.append({
        "feature": col,
        "method": method,
        "skew_before": skew_before,
        "skew_after": skew_after
    })

results_df = pd.DataFrame(results)
transformed_df = results_df[results_df["method"] != "none"].copy()

print("\nTransformed columns:")
if transformed_df.empty:
    print("No columns required skewness transformation.")
else:
    print(
        transformed_df[["feature", "method", "skew_before", "skew_after"]]
        .sort_values(by="skew_before", key=lambda s: s.abs(), ascending=False)
        .to_string(index=False)
    )

mean_abs_skew_before = skew_before_all.abs().mean()
mean_abs_skew_after = df_transformed[candidate_cols].skew().abs().mean()

median_abs_skew_before = skew_before_all.abs().median()
median_abs_skew_after = df_transformed[candidate_cols].skew().abs().median()

print("\nOverall skewness:")
print(f"Mean absolute skew BEFORE:  {mean_abs_skew_before:.4f}")
print(f"Mean absolute skew AFTER:   {mean_abs_skew_after:.4f}")
print(f"Median absolute skew BEFORE:{median_abs_skew_before:.4f}")
print(f"Median absolute skew AFTER: {median_abs_skew_after:.4f}")

df_transformed.to_csv(OUTPUT, index=False)

print(f"\nSaved dataset: {OUTPUT.name}")
print(f"Shape: {df_transformed.shape}")