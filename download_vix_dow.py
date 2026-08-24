"""Download historical India VIX (^INDIAVIX) and Dow Jones (^DJI) data (2020-2024)."""

import pandas as pd
import yfinance as yf
from pathlib import Path

out_dir = Path("c:/Websites/ammu/macro_data")
out_dir.mkdir(parents=True, exist_ok=True)

print("Downloading India VIX (^INDIAVIX)...")
try:
    vix = yf.download("^INDIAVIX", start="2020-01-01", end="2024-12-31")
    if not vix.empty:
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix = vix.reset_index()
        vix.to_csv(out_dir / "INDIAVIX_day.csv", index=False)
        print(f"[OK] Downloaded India VIX ({len(vix)} rows).")
    else:
        print("[WARNING] yfinance returned empty for ^INDIAVIX.")
except Exception as e:
    print(f"[ERROR] Failed downloading VIX: {e}")

print("Downloading Dow Jones Industrial Average (^DJI)...")
try:
    dji = yf.download("^DJI", start="2020-01-01", end="2024-12-31")
    if not dji.empty:
        if isinstance(dji.columns, pd.MultiIndex):
            dji.columns = dji.columns.get_level_values(0)
        dji = dji.reset_index()
        dji.to_csv(out_dir / "DJI_day.csv", index=False)
        print(f"[OK] Downloaded Dow Jones ({len(dji)} rows).")
    else:
        print("[WARNING] yfinance returned empty for ^DJI.")
except Exception as e:
    print(f"[ERROR] Failed downloading DJI: {e}")
