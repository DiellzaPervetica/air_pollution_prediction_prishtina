import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "1F1_pm_consistent.csv")

null_report = df.isnull().sum().reset_index()
null_report.columns = ['Column', 'Null_Count']
null_report.to_csv(BASE_DIR / "data" / "1F3_final_null_check.csv", index=False)

final_stats = df.describe()
final_stats.to_csv(BASE_DIR / "data" / "1F3_final_statistical_audit.csv")

final_output = BASE_DIR / "data" / "1F_CLEANED_FINAL.csv"
df.to_csv(final_output, index=False)

print("-" * 30)
print("PASTRIMI PERFUNDOI ME SUKSES")
print(f"Dataseti gati per Outliers: {final_output.name}")
print("-" * 30)