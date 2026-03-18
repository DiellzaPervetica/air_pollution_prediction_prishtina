import pandas as pd
from pathlib import Path

# Rruga drejt projektit
BASE_DIR = Path(__file__).resolve().parent.parent
input_path = BASE_DIR / "data" / "1B_cleaned_data_hourly_2023_2025.csv"
output_path = BASE_DIR / "data" / "1F1_pm_consistent.csv"

df = pd.read_csv(input_path)

bad_ratio_mask = df['pm25'] > df['pm10']
anomalies = df[bad_ratio_mask].copy()

if not anomalies.empty:
    anomalies.to_csv(BASE_DIR / "data" / "1F1_pm_anomalies_report.csv", index=False)
    print(f"U gjeten {len(anomalies)} raste ku PM2.5 > PM10. Raporti u ruajt.")
else:
    print("Nuk u gjet asnje anomali ne relacionin PM2.5/PM10.")

df.loc[bad_ratio_mask, 'pm25'] = df['pm10']

df.to_csv(output_path, index=False)