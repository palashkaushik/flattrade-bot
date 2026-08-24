"""Inspect intraday Dow Jones (DowJones1m.csv) and India VIX (INDIA VIX_minute.csv)."""

import pandas as pd
from pathlib import Path

desktop_dir = Path("C:/Users/user/Desktop/nifty50 data")

print("Checking DowJones1m.csv...")
df_dow = pd.read_csv(desktop_dir / "DowJones1m.csv", nrows=5)
print("Columns in DowJones1m.csv:", list(df_dow.columns))
print(df_dow.head(2))

print("\nChecking INDIA VIX_minute.csv...")
df_vix = pd.read_csv(desktop_dir / "INDIA VIX_minute.csv", nrows=5)
print("Columns in INDIA VIX_minute.csv:", list(df_vix.columns))
print(df_vix.head(2))
