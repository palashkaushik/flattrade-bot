"""Check overlapping trading dates across Nifty Options, India VIX, and Dow Jones."""

import pandas as pd
from pathlib import Path
from backtest_5y_optimized import option_files

# Load Nifty option dates
files = option_files("2020-01-01", "2024-12-31")
nifty_dates = set(files.keys())

# Load India VIX dates
vix_path = Path("c:/Websites/ammu/macro_data/INDIAVIX_day.csv")
df_vix = pd.read_csv(vix_path)
vix_dates = set(df_vix["date"].astype(str))

# Load Dow Jones dates
dji_path = Path("c:/Websites/ammu/macro_data/DJI_day.csv")
df_dji = pd.read_csv(dji_path)
dji_dates = set(df_dji["date"].astype(str))

# Overlap
overlap_all = sorted(nifty_dates & vix_dates & dji_dates)
missing_vix = sorted(nifty_dates - vix_dates)
missing_dji = sorted(nifty_dates - dji_dates)

print(f"Total Nifty Option Trading Days (2020-2024): {len(nifty_dates)}")
print(f"Total India VIX Trading Days              : {len(vix_dates)}")
print(f"Total Dow Jones Trading Days              : {len(dji_dates)}")
print(f"Total OVERLAPPING Trading Days           : {len(overlap_all)}")

print(f"\nNifty Days Missing India VIX ({len(missing_vix)} days):")
print(missing_vix[:10])

print(f"\nNifty Days Missing Dow Jones ({len(missing_dji)} days - due to US holidays):")
print(missing_dji[:10])
