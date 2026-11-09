"""Diff one day: sequential CSV pipeline vs Parquet-cache pipeline."""
import os
from datetime import datetime, timedelta

import pandas as pd
import pyarrow.parquet as pq

import bt
from sixyear import files, SPOT, FrameData, expiry

CACHE = r"C:\Websites\FLATTRADE BOT\research\backtest\sixyear_cache.parquet"
TARGET = "2020-01-02"

# --- sequential: read CSVs for the 12 days before + target
allf = files()
idx = next(i for i, (d, _) in enumerate(allf) if d == TARGET)
use = allf[max(0, idx - 12): idx + 1]
frames = []
for d, p in use:
    df = pd.read_csv(p, usecols=["date", "time", "symbol", "open", "high", "low", "close", "volume"])
    df = df[df.date == d].copy()
    df["time"] = df["time"].astype(str).str.strip()
    df = df[df["time"].str.match(r"^\d{2}:\d{2}", na=False)]
    df["min"] = df["time"].map(lambda t: int(t[:2]) * 60 + int(t[3:5])).astype(int)
    frames.append(df)
seq_df = pd.concat(frames, ignore_index=True)

# --- cache slice for the same dates
tbl = pq.read_table(CACHE, filters=[("date", ">=", use[0][0]), ("date", "<=", TARGET)])
cache_df = tbl.to_pandas()

print("seq rows", len(seq_df), "cache rows", len(cache_df))
print("seq dtypes:\n", seq_df.dtypes)
print("cache dtypes:\n", cache_df.dtypes)

# compare per-day row counts and symbol sets
for d, _ in use[-3:]:
    a = seq_df[seq_df.date == d]
    b = cache_df[cache_df.date == d]
    print(d, "seq", len(a), "cache", len(b),
          "same syms", set(a.symbol) == set(b.symbol))
    if set(a.symbol) != set(b.symbol):
        print("  only seq:", list(set(a.symbol) - set(b.symbol))[:5])
        print("  only cache:", list(set(b.symbol) - set(a.symbol))[:5])

# run precompute on both
def ctx_for(df):
    data = FrameData(df, SPOT)
    ddf = df[df.date == TARGET]
    exps = {expiry(s) for s in ddf.symbol.unique()}; exps.discard(None)
    dd = datetime.strptime(TARGET, "%Y-%m-%d").date()
    later = sorted(e for e in exps if e >= dd)
    nearest = f"NIFTY{later[0]:%d%b%y}".upper() if later else ""
    return bt.precompute(data, TARGET, nearest)

for name, df in (("seq", seq_df), ("cache", cache_df)):
    ctx = ctx_for(df)
    sigs = []
    for c in ctx:
        sigs.extend(bt.evaluate([c], cap=float("inf")))
    print(f"{name}: contract-days {len(ctx)} ctxs {sum(len(c['ctxs']) for c in ctx)} "
          f"signals {[(s['j'], s['sym'], round(s['pts'],2)) for s in sigs]}")
