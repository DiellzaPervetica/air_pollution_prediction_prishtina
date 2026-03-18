import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
input_path = BASE_DIR / "data" / "1D5_logically_validated.csv"
df = pd.read_csv(input_path)

missing_count = df.isnull().sum().reset_index()
missing_count.columns = ['Column', 'Missing_Values']
missing_count['Percentage'] = (missing_count['Missing_Values'] / len(df)) * 100

missing_count.to_csv(BASE_DIR / "data" / "1E1_missing_report.csv", index=False)
