import pandas as pd
from pathlib import Path

# Rruga drejt projektit
BASE_DIR = Path(__file__).resolve().parent.parent
input_path = BASE_DIR / "data" / "1D3_all_quality_checked.csv"
output_path = BASE_DIR / "data" / "1D4_precision_fixed.csv"

df = pd.read_csv(input_path)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

df[numeric_cols] = df[numeric_cols].round(3)

df.to_csv(output_path, index=False)
