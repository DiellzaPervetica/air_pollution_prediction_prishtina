import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "2C_missing_values_handled.csv"
output_path = BASE_DIR / "data" / "2D_validated_final_dataset.csv"

df = pd.read_csv(input_path)

if "pm10" in df.columns and "pm25" in df.columns:
    bad_ratio_mask = df["pm25"] > df["pm10"]
    pm_anomaly_count = bad_ratio_mask.sum()
    df.loc[bad_ratio_mask, "pm25"] = df.loc[bad_ratio_mask, "pm10"]
    print(f"Raste ku PM2.5 > PM10 u korrigjuan: {pm_anomaly_count}")
else:
    print("Kolonat pm10 dhe pm25 nuk u gjeten.")

if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    time_diff = df["datetime"].diff()
    gap_count = (time_diff > pd.Timedelta(hours=1)).sum()

    if gap_count == 0:
        print("Nuk ka gaps ne timeline.")
    else:
        print(f"U gjeten {gap_count} gaps ne timeline.")
else:
    print("Kolona datetime nuk u gjet.")

total_nulls = df.isnull().sum().sum()
if total_nulls == 0:
    print("Nuk ka vlera NULL ne dataset.")
else:
    print(f"Ka ende {total_nulls} vlera NULL ne dataset.")

df.to_csv(output_path, index=False)

print(f"Dataseti final u ruajt nÃ«: {output_path.name}")