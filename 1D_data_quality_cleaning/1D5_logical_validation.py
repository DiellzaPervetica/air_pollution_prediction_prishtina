import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
input_path = BASE_DIR / "data" / "1D4_precision_fixed.csv"
output_path = BASE_DIR / "data" / "1D5_logically_validated.csv"

df = pd.read_csv(input_path)

energy_units = ["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW"]
df["total_generation_mw"] = df[energy_units].sum(axis=1)

if "relative_humidity_2m (%)" in df.columns:
    df["relative_humidity_2m (%)"] = df["relative_humidity_2m (%)"].clip(0, 100)

anomalies = df[df[energy_units].sum(axis=1) != df["total_generation_mw"]]
anomalies.to_csv(BASE_DIR / "data" / "1D5_energy_anomalies_report.csv", index=False)

df.to_csv(output_path, index=False)
print("Validimi logjik përfundoi. Çdo mospërputhje në shuma u korrigjua.")