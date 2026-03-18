import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "1C1_datetime_fixed.csv")

df_clean = df.drop_duplicates().reset_index(drop=True)
df_clean.to_csv(BASE_DIR / "data" / "1C3_no_duplicates.csv", index=False)