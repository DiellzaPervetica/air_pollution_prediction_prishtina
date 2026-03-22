import pandas as pd
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "data" / "4D_scaled_dataset.csv"
OUTPUT = BASE_DIR / "data" / "4E_selected_dataset.csv"

TARGET = "pm25"
KEEP_COLS = ["datetime", "date"]

POLLUTANTS_TO_DROP = ["co", "no2", "o3", "pm10", "so2"]
STRUCTURAL_TO_DROP = ["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW", "hour", "month", "day_of_week"]

def calculate_vif(df_numeric):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_numeric.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_numeric.values, i)
        for i in range(df_numeric.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)

df = pd.read_csv(INPUT)

df_datetime = df[KEEP_COLS].copy()
df_numeric = df.drop(columns=KEEP_COLS)

cols_to_drop = [c for c in df_numeric.columns if "lag" in c.lower()]
cols_to_drop += [c for c in df_numeric.columns if "pm25" in c and c != TARGET]
cols_to_drop += [c for c in POLLUTANTS_TO_DROP + STRUCTURAL_TO_DROP if c in df_numeric.columns]

df_numeric = df_numeric.drop(columns=list(set(cols_to_drop)))

VIF_THRESHOLD = 7.0
X = df_numeric.drop(columns=[TARGET])

constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)

near_constant_cols = [col for col in X.columns if X[col].std() < 1e-8]
if near_constant_cols:
    X = X.drop(columns=near_constant_cols)

while True:
    vif_results = calculate_vif(X)
    vif_results = vif_results.replace([float("inf"), -float("inf")], pd.NA).dropna()

    if vif_results.empty:
        break

    max_vif = vif_results.iloc[0]["VIF"]

    if max_vif > VIF_THRESHOLD:
        feature_to_drop = vif_results.iloc[0]["Feature"]
        X = X.drop(columns=[feature_to_drop])
    else:
        break

final_features = X.columns.tolist() + [TARGET]
df_selected = df_numeric[final_features].copy()
df_final = pd.concat([df_datetime, df_selected], axis=1)

df_final.to_csv(OUTPUT, index=False)

correlations = df_numeric.corr(numeric_only=True)[TARGET].drop(TARGET).abs()
correlations = correlations.dropna().sort_values(ascending=False)

print("\n" + "=" * 70)
print("RAPORTI FINAL I ZGJEDHJES SË TIPAREVE")
print("=" * 70)
print(f"Dataset fillestar: {df.shape}")
print(f"Dataset përfundimtar: {df_final.shape}")
print(f"Numri i tipareve finale: {len(final_features) - 1}\n")

print("TIPARET E ZGJEDHURA (të renditura sipas korrelacionit):\n")
for feat in correlations.index:
    if feat in final_features:
        print(f"  [SELECTED] {feat:40s} | Korrelacion: {correlations[feat]:7.4f}")

print("\n" + "-" * 70)