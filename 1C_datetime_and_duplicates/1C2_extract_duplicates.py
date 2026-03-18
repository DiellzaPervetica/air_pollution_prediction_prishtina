import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "1C1_datetime_fixed.csv")

duplicates = df[df.duplicated(keep=False)]
duplicates.to_csv(BASE_DIR / "data" / "1C2_only_duplicates.csv", index=False)