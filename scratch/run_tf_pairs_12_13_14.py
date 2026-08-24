import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"c:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
import artifacts.f6_hybrid.marni_vsa_tf_pairs_matrix_7y as tf_matrix

spot_all = source.load_spot()
opt_map = source.option_day_files("2020-01-01", "2026-05-05")
all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))

# Get the last 3 days in dataset (2026-05) or available days
days = [d for d in ["2026-08-12", "2026-08-13", "2026-08-14"] if d in opt_map]
if not days:
    days = all_days[-3:]

previous = {day: max((c for c in all_days if c < day), default="") for day in days}

tf_matrix.init_worker(spot_all)

tf_pairs = [
    (1, 15, "1m LTF / 15m HTF (Reference Baseline)"),
    (1, 5,  "1m LTF / 5m HTF"),
    (3, 15, "3m LTF / 15m HTF"),
    (3, 5,  "3m LTF / 5m HTF"),
    (5, 15, "5m LTF / 15m HTF"),
    (5, 30, "5m LTF / 30m HTF"),
]

print(f"\n{'='*135}")
print(f"MARNI VSA ENGINE — TIMEFRAME BIAS PAIRS STUDY ({days[0]} to {days[-1]} | TP = 0.290)")
print(f"{'='*135}")
print(f"{'Timeframe Bias Pair':45s} | {'Trades':6s} | {'Win Rate':9s} | {'Net Points':12s} | {'Profit Factor':14s} | {'Net Realized P&L (Rs)':22s}")
print("-" * 135)

for ltf_p, htf_p, pair_label in tf_pairs:
    pair_trades = []
    for day in days:
        task = (day, opt_map[day], previous[day], ltf_p, htf_p, 20.0, True)
        res = tf_matrix.process_day(task)
        pair_trades.extend(res)
        
    st = tf_matrix.compute_stats(pair_trades, len(days))
    print(f"{pair_label:45s} | {st['trades']:6d} | {st['win_rate']:8.1f}% | {st['net_points']:+11.2f}p | {st['profit_factor']:13.2f} | Rs {st['net_rs']:+19,.2f}")

print("-" * 135)
