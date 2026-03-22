import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "data" / "4B_skewness_handled.csv"
OUTPUT = BASE_DIR / "data" / "4D_scaled_dataset.csv"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

NON_NUMERIC_COLS = ["datetime", "date"]

df = pd.read_csv(INPUT)

df_datetime = df[NON_NUMERIC_COLS].copy()
df_numeric = df.drop(columns=NON_NUMERIC_COLS)

scaler = StandardScaler()
scaler.fit(df_numeric)
df_numeric_scaled = pd.DataFrame(
    scaler.transform(df_numeric),
    columns=df_numeric.columns,
    index=df_numeric.index,
)

df_scaled = pd.concat([df_datetime, df_numeric_scaled], axis=1)

df_scaled.to_csv(OUTPUT, index=False)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)