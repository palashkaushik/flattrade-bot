"""Fetch India VIX (^INDIAVIX) and Dow Jones (^DJI) daily historical data directly from Yahoo Finance API."""

import json
import urllib.request
from pathlib import Path
import pandas as pd

out_dir = Path("c:/Websites/ammu/macro_data")
out_dir.mkdir(parents=True, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def download_ticker(ticker: str, filename: str):
    # Timestamps for 2020-01-01 to 2024-12-31
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1=1577836800&period2=1735689600&interval=1d"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]
            
            df = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s").strftime("%Y-%m-%d"),
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "volume": quote.get("volume", [0]*len(timestamps))
            }).dropna(subset=["close"])
            
            df.to_csv(out_dir / filename, index=False)
            print(f"[OK] Downloaded {ticker} -> {filename} ({len(df)} rows).")
            return df
    except Exception as e:
        print(f"[ERROR] Failed fetching {ticker}: {e}")
        return None

if __name__ == "__main__":
    download_ticker("^INDIAVIX", "INDIAVIX_day.csv")
    download_ticker("^DJI", "DJI_day.csv")
