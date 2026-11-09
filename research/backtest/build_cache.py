"""Build a single Parquet cache from the per-day option CSVs (one row group/day).

Keeps the same rows the six-year runner needs: band-filtered strikes, parsed
minute, true volume.  Run once; the parallel runner reads day slices from it.
"""
import os
import re
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = r"C:\Users\user\Desktop\nifty50 data\nifty_options"
INDEX = r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
OUT = r"C:\Websites\FLATTRADE BOT\research\backtest\sixyear_cache.parquet"
START, END = "2020-01-01", "2026-04-30"
BAND = 400

ix = pd.read_csv(INDEX, usecols=["date", "close"])
ix["iso"] = ix["date"].str.slice(0, 10)
ix["min"] = ix["date"].str.slice(11, 13).astype(int) * 60 + ix["date"].str.slice(14, 16).astype(int)
SPOT = {}
for d, m, c in zip(ix["iso"], ix["min"], ix["close"]):
    SPOT.setdefault(d, {})[int(m)] = float(c)
del ix


def parse_day(p):
    m = re.search(r"nifty_options_(\d{2})_(\d{2})_(\d{4})\.csv$", p)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None


def files():
    out = []
    for d, _, fs in os.walk(ROOT):
        for f in fs:
            if f.startswith("nifty_options_") and f.endswith(".csv"):
                day = parse_day(f)
                if day and START <= day <= END:
                    out.append((day, os.path.join(d, f)))
    out.sort()
    return out


writer = None
n_days = n_rows = 0
for day, path in files():
    try:
        df = pd.read_csv(path, usecols=["date", "time", "symbol", "open", "high", "low", "close", "volume"])
    except Exception as e:
        print("skip", path, e); continue
    df = df[df.date == day].copy()  # no strike band: cache must be row-identical to the CSVs
    df["time"] = df["time"].astype(str).str.strip()
    df = df[df["time"].str.match(r"^\d{2}:\d{2}", na=False)]
    if len(df) < 500:
        continue
    df["min"] = df["time"].map(lambda t: int(t[:2]) * 60 + int(t[3:5])).astype(int)
    df = df.drop(columns=["time", "strike"], errors="ignore")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["min"] = df["min"].astype("int64")
    df["date"] = df["date"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df = df[["date", "symbol", "open", "high", "low", "close", "volume", "min"]]

    tbl = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUT, tbl.schema)
    writer.write_table(tbl)
    n_days += 1; n_rows += len(df)
    if n_days % 100 == 0:
        print(f"{n_days} days, {n_rows:,} rows", flush=True)

if writer:
    writer.close()
print("wrote", OUT, n_days, "days", n_rows, "rows")


