"""Parallel six-year runner (10 workers) over the Parquet cache.

Same per-day pipeline as sixyear.py (bt.precompute + bt.evaluate), only the
data access and scheduling change, so results stay congruent.
"""
import os
import re
import sys
from datetime import datetime
from multiprocessing import Pool

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt  # noqa: E402

ROOT = r"C:\Users\user\Desktop\nifty50 data\nifty_options"
INDEX = r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
CACHE = r"C:\Websites\FLATTRADE BOT\research\backtest\sixyear_cache.parquet"
START, END = "2020-01-01", "2026-04-30"
BUF_DAYS = 12
WORKERS = int(os.environ.get("SY_WORKERS", "10"))

CONFIGS = [
    ("cal SL0.9 TP8", 0.9, 8.0, 20),
    ("wr  SL0.9 TP4 nocap", 0.9, 4.0, 99),
    ("dep SL1.0 TP5", 1.0, 5.0, 20),
    ("SL1.2 TP3.6", 1.2, 3.6, 20),
]

# spot map (small)
_ix = pd.read_csv(INDEX, usecols=["date", "close"])
_ix["iso"] = _ix["date"].str.slice(0, 10)
_ix["min"] = _ix["date"].str.slice(11, 13).astype(int) * 60 + _ix["date"].str.slice(14, 16).astype(int)
SPOT = {}
for d, m, c in zip(_ix["iso"], _ix["min"], _ix["close"]):
    SPOT.setdefault(d, {})[int(m)] = float(c)
del _ix


def parse_day(p):
    m = re.search(r"nifty_options_(\d{2})_(\d{2})_(\d{4})\.csv$", p)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None


def all_days():
    days = set()
    for d, _, fs in os.walk(ROOT):
        for f in fs:
            if f.startswith("nifty_options_") and f.endswith(".csv"):
                day = parse_day(f)
                if day and START <= day <= END:
                    days.add(day)
    return sorted(days)


def expiry(sym):
    m = re.match(r"^NIFTY(\d{2}[A-Z]{3}\d{2})", sym)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%d%b%y").date()
    except ValueError:
        return None


class FrameData:
    def __init__(self, df, spot):
        self.df = df
        self.spot = spot

    def atm(self, day, minute):
        sp = self.spot.get(day, {})
        ks = [k for k in sp if k <= minute]
        return int(round(sp[max(ks)] / 50.0) * 50) if ks else None

    def contracts(self, day, nearest):
        t = self.df[(self.df.date == day) & self.df.symbol.str.startswith(nearest)]
        return sorted(t.symbol.unique())


def process_chunk(args):
    chunk, lead = args
    d0, d1 = min(lead + chunk), max(lead + chunk)
    tbl = pq.read_table(CACHE, filters=[("date", ">=", d0), ("date", "<=", d1)])
    df = tbl.to_pandas()
    by_day = {d: g for d, g in df.groupby("date", sort=True)}
    out = []
    buffer = [by_day[d] for d in lead if d in by_day]
    for day in chunk:
        if day not in by_day:
            continue
        buf = buffer + [by_day[day]]
        comb = pd.concat(buf, ignore_index=True)
        data = FrameData(comb, SPOT)
        exps = {expiry(s) for s in by_day[day].symbol.unique()}
        exps.discard(None)
        dd = datetime.strptime(day, "%Y-%m-%d").date()
        later = sorted(e for e in exps if e >= dd)
        nearest = f"NIFTY{later[0]:%d%b%y}".upper() if later else ""
        ctx = bt.precompute(data, day, nearest) if nearest else []
        res = {}
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
                if cum <= -bt.DAY_CAP:
                    break
                capped.append(s); cum += s["pts"]
            res[name] = [(s["j"], s["pts"]) for s in capped]
        out.append((day, res))
        buffer.append(by_day[day])
        if len(buffer) > BUF_DAYS:
            buffer.pop(0)
    return out


def main(limit=None):
    days = all_days()
    if limit:
        days = days[:limit]
    n = len(days)
    per = (n + WORKERS - 1) // WORKERS
    chunks = []
    for i in range(0, n, per):
        chunk = days[i:i + per]
        lead = days[max(0, i - BUF_DAYS):i]
        chunks.append((chunk, lead))
    t0 = datetime.now()
    print(f"{n} days, {len(chunks)} chunks, {WORKERS} workers", flush=True)
    with Pool(WORKERS) as pool:
        results = pool.map(process_chunk, chunks)
    merged = {}
    for part in results:
        for day, res in part:
            merged[day] = res
    print(f"done in {(datetime.now()-t0).seconds}s", flush=True)
    stats = {c[0]: [] for c in CONFIGS}
    for day in sorted(merged):
        for name in stats:
            pts = merged[day].get(name, [])
            stats[name].append((day, [{"pts": p, "j": j, "exit": j, "sym": "", "nm": "",
                                       "entry": 0, "sl": 0, "risk": 0, "tp": 0,
                                       "tp_pts": 0, "why": ""} for j, p in pts]))
    print(f"\n=== {len(merged)} sessions ({START} .. {END}) ===")
    for name, _, _, _ in CONFIGS:
        s = bt.summarise(stats[name])
        print(f"{name:22s} tr {s['trades']:5d} WR {s['wr']:5.1f} pts {s['pts']:10.2f} "
              f"Rs {s['rs']:+10,} DD {s['dd']:8.2f} Cal {s['calmar']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)


