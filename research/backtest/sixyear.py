"""Six-year (2020 .. Apr 2026) run of the order-block strategy, 4 configs.

Data: Desktop/nifty50 data/nifty_options/YYYY/M/nifty_options_DD_MM_YYYY.csv
(date,time,symbol,open,high,low,close,oi,volume) — TRUE VOLUME, so VWAP and
PDVWAP are real volume-weighted prices, congruent with the live engine.

Warm-up mirrors the bot: 12-session rolling buffer -> per-contract seed of the
last up-to-2000 prior bars, comb = prior session <=15:30, day levels from the
prior session, PC = 15:32 bar, virgin CPR from the buffer's oldest day.

    python sixyear.py        # full 2020-01-01 .. 2026-04-30
    python sixyear.py 20     # smoke
"""
import os
import re
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt  # noqa: E402

ROOT = r"C:\Users\user\Desktop\nifty50 data\nifty_options"
INDEX = r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
START, END = "2020-01-01", "2026-04-30"
BUF_DAYS = 12

bt.WAVE_MODE = "full"
bt.TRIGGER_MODE = "ob"
bt.EXTRA_EMAS = False
bt.MIN_RISK, bt.ENTRY_END, bt.DAY_CAP = 3, 900, 40

CONFIGS = [
    ("cal SL0.9 TP8", 0.9, 8.0, 20),
    ("wr  SL0.9 TP4 nocap", 0.9, 4.0, 99),
    ("dep SL1.0 TP5", 1.0, 5.0, 20),
    ("SL1.2 TP3.6", 1.2, 3.6, 20),
]

print("index ...", flush=True)
ix = pd.read_csv(INDEX, usecols=["date", "close"])
ix["iso"] = ix["date"].str.slice(0, 10)
ix["min"] = ix["date"].str.slice(11, 13).astype(int) * 60 + ix["date"].str.slice(14, 16).astype(int)
SPOT = {}
for d, m, c in zip(ix["iso"], ix["min"], ix["close"]):
    SPOT.setdefault(d, {})[int(m)] = float(c)
del ix
print("index days", len(SPOT), flush=True)


def parse_day(p):
    m = re.search(r"nifty_options_(\d{2})_(\d{2})_(\d{4})\.csv$", p)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None


def files():
    out = []
    for d, _, fs in os.walk(ROOT):
        for f in fs:
            if not f.startswith("nifty_options_") or not f.endswith(".csv"):
                continue
            day = parse_day(f)
            if day and START <= day <= END:
                out.append((day, os.path.join(d, f)))
    out.sort()
    return out


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


def run(limit=None):
    stats = {c[0]: [] for c in CONFIGS}
    buffer, n, t0 = [], 0, datetime.now()
    for day, path in files():
        ddf = pd.read_csv(path, usecols=["date", "time", "symbol", "open", "high", "low", "close", "volume"])
        ddf = ddf[ddf.date == day].copy()
        ddf["time"] = ddf["time"].astype(str).str.strip()
        ddf = ddf[ddf["time"].str.match(r"^\d{2}:\d{2}", na=False)]
        if len(ddf) < 500:
            continue
        ddf["min"] = ddf["time"].map(lambda t: int(t[:2]) * 60 + int(t[3:5])).astype(int)
        comb = pd.concat(buffer + [ddf], ignore_index=True)
        data = FrameData(comb, SPOT)
        exps = {expiry(s) for s in ddf.symbol.unique()}
        exps.discard(None)
        dd = datetime.strptime(day, "%Y-%m-%d").date()
        later = sorted(e for e in exps if e >= dd)
        nearest = f"NIFTY{later[0]:%d%b%y}".upper() if later else ""
        ctx = bt.precompute(data, day, nearest) if nearest else []
        for name, sl, tp, mx in CONFIGS:
            bt.SL_ATR, bt.TP_MULT, bt.MAX_SL = sl, tp, mx
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
            stats[name].append((day, capped))
        n += 1
        if n % 25 == 0:
            print(f"day {n} {day} elapsed {(datetime.now()-t0).seconds}s", flush=True)
        if limit and n >= limit:
            return stats, n
        buffer.append(ddf)
        if len(buffer) > BUF_DAYS:
            buffer.pop(0)
    return stats, n


def report(stats, n):
    print(f"\n=== {n} sessions ({START} .. {END}) ===")
    for name, _, _, _ in CONFIGS:
        s = bt.summarise(stats[name])
        print(f"{name:22s} tr {s['trades']:5d} WR {s['wr']:5.1f} pts {s['pts']:10.2f} "
              f"Rs {s['rs']:+10,} DD {s['dd']:8.2f} Cal {s['calmar']}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    st, nd = run(lim)
    report(st, nd)



