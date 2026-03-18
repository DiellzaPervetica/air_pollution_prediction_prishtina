import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "1F1_pm_consistent.csv")

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

time_diff = df['datetime'].diff()

gaps = df[time_diff > pd.Timedelta(hours=1)].copy()
gaps['gap_duration'] = time_diff[time_diff > pd.Timedelta(hours=1)]

gaps.to_csv(BASE_DIR / "data" / "1F2_timeline_gaps_report.csv", index=False)

