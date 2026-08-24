"""Backtest F6 champion (S3>=80 exit, 12-pt SL, no ATR) on 2026-08-18..20.

Options  : C:/Users/user/Desktop/nifty50 data/nifty_options  (user-provided)
Index    : C:/Websites/ammu/data/2026-08-XX/nifty50_index_1m_*.csv (per-day)
"""
import csv
import numpy as np

import grid_optimize_f6_atr as G
import backtest_5y_optimized as B5
import test_f6_champion_s3exit as m
from pathlib import Path

DESKTOP_OPTS = Path("C:/Users/user/Desktop/nifty50 data/nifty_options")
AMMU_DATA = Path("C:/Websites/ammu/data")

# 1) point option-file discovery at the Desktop folder
B5.OPTS_DIR = DESKTOP_OPTS

REQ = ["2026-08-18", "2026-08-19", "2026-08-20"]
files = B5.option_files("2026-08-14", "2026-08-20")  # include warm-up days
days_all = sorted(files.keys())
print("option files discovered:", days_all, flush=True)


# 2) build spot dict from the per-day ammu index files
def build_spot(daylist):
    out = {}
    for d in daylist:
        fp = AMMU_DATA / d / f"nifty50_index_1m_{d}.csv"
        if not fp.exists():
            print("MISSING INDEX FILE:", fp, flush=True)
            continue
        mins, closes = [], []
        with open(fp, newline="") as f:
            for row in csv.DictReader(f):
                ts = row["timestamp"].split("T")[1].split("+")[0]
                h, mm, _ = ts.split(":")
                mins.append(int(h) * 60 + int(mm))
                closes.append(float(row["close"]))
        out[d] = {"min": np.array(mins, dtype=np.int64),
                  "close": np.array(closes, dtype=float)}
    return out


spot = build_spot(REQ)
print("spot days built:", sorted(spot.keys()), flush=True)
m.init_worker_local(spot)

# 3) run
idxmap = {d: i for i, d in enumerate(days_all)}
tasks = []
for d in REQ:
    pi = idxmap[d]
    fprev = str(files[days_all[pi - 1]]) if pi > 0 else ""
    tasks.append((d, str(files[d]), fprev, m.CHAMPION))

allt = []
for t in tasks:
    allt += m.process_day(t)

print()


def hm(mn):
    return f"{mn // 60:02d}:{mn % 60:02d}"


hdr = f"{'DATE':12}{'ENT':>6}{'EX':>6} {'SIDE':>2} {'SYMBOL':>18} {'ENTRY':>8}{'EXIT':>8} {'PTS':>7}{'RS':>8}  {'REASON':>18} {'DUR':>4}"
print(hdr)
print("-" * 100)
for tr in allt:
    print(f"{tr['date']:12}{hm(tr['entry_min']):>6}{hm(tr['exit_min']):>6} {tr['side']:>2} "
          f"{tr['symbol']:>18} {tr['entry']:>8.2f}{tr['exit']:>8.2f} {tr['pts']:>7.2f}{tr['rs']:>8}  "
          f"{tr['reason']:>18} {tr['duration_min']:>4}")
print("-" * 100)
st, _ = m.stats_for(allt)
print(f"Days: {REQ}")
print(f"Trades {st['trades']} | WR {st['wr']:.1f}% | Net Rs {st['rs']:+,d} | PF {st['pf']:.2f}")
