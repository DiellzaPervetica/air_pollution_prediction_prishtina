import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "1C3_no_duplicates.csv")

for col in ["pm10", "pm25", "co", "no2", "o3", "so2"]:
    df[col] = df[col].apply(lambda x: np.nan if x < 0 else x)

df.to_csv(BASE_DIR / "data" / "1D1_pollution_quality_checked.csv", index=False)