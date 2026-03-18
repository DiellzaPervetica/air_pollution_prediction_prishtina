import pandas as pd

print(pd.Timestamp("2023-07-31 23:00:00", tz="UTC").tz_convert("Europe/Belgrade"))