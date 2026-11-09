"""Find the first day where CSV-pipeline and cache-pipeline disagree."""
import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

import bt
from sixyear import files, SPOT, FrameData, expiry

CACHE = r"C:\Websites\FLATTRADE BOT\research\backtest\sixyear_cache.parquet"
CONFIGS = [("cal", 0.9, 8.0, 20), ("wr", 0.9, 4.0, 99), ("dep", 1.0, 5.0, 20), ("t36", 1.2, 3.6, 20)]
allf = files()[:40]


def load_csv(day, path):
    df = pd.read_csv(path, usecols=["date", "time", "symbol", "open", "high", "low", "close", "volume"])
    df = df[df.date == day].copy()
    df["time"] = df["time"].astype(str).str.strip()
    df = df[df["time"].str.match(r"^\d{2}:\d{2}", na=False)]
    df["min"] = df["time"].map(lambda t: int(t[:2]) * 60 + int(t[3:5])).astype(int)
    return df


def day_book(df, day):
    data = FrameData(df, SPOT)
    ddf = df[df.date == day]
    exps = {expiry(s) for s in ddf.symbol.unique()}; exps.discard(None)
    dd = datetime.strptime(day, "%Y-%m-%d").date()
    later = sorted(e for e in exps if e >= dd)
    nearest = f"NIFTY{later[0]:%d%b%y}".upper() if later else ""
    ctx = bt.precompute(data, day, nearest) if nearest else []
    out = {}
    for name, sl, tp, mx in CONFIGS:
        bt.SL_ATR, bt.TP_MULT, bt.MAX_SL = sl, tp, mx
        bt.MIN_RISK, bt.ENTRY_END, bt.DAY_CAP = 3, 900, 40
        sigs = []
        for c in ctx:
            sigs.extend(bt.evaluate([c], cap=float("inf")))
        kept, last = [], -1
        for s in sorted(sigs, key=lambda x: x["j"]):
            if s["j"] <= last:
                continue
            kept.append(s); last = s["exit"]
        capped, cum = [], 0.0
        for s in kept:
            if cum <= -40:
                break
            capped.append(s); cum += s["pts"]
        out[name] = (len(capped), round(sum(x["pts"] for x in capped), 2))
    return out


cache = pq.read_table(CACHE).to_pandas()
by_day = {d: g for d, g in cache.groupby("date", sort=True)}

csv_frames = []
diffs = 0
for i, (day, path) in enumerate(allf):
    csv_frames.append(load_csv(day, path))
    seq = day_book(pd.concat(csv_frames[-13:], ignore_index=True), day)
    # cache buffer: same 13 days
    hist = [d for d, _ in allf[max(0, i - 12): i + 1]]
    cache_buf = pd.concat([by_day[d] for d in hist if d in by_day], ignore_index=True)
    cac = day_book(cache_buf, day)
    if seq != cac:
        diffs += 1
        print(f"{day}  seq {seq}  cache {cac}")
print("days compared", len(allf), "diffs", diffs)
