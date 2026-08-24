"""Inspect Nifty Options CSV files for Open Interest (OI) column."""

import pandas as pd
from pathlib import Path

path = next(Path("C:/Websites/ammu/nifty_options").rglob("*.csv"))
df = pd.read_csv(path, nrows=5)
print("Columns in nifty_options CSV:")
print(list(df.columns))
