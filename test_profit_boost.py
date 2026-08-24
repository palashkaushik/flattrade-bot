"""Test Profit Optimization: SL -8.0 Pts, Session Cutoff 01:30 PM, Max 3 Trades/Day."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from backtest_5y_optimized import main as run_optimized_backtest, process_single_day, load_spot, option_files, init_worker, summarize, print_yearly_breakdown, SL_POINTS, TP_POINTS
from multiprocessing import Pool, cpu_count

def test_optimized_rules(sl_pts: float, tp_pts: float, cutoff_min: int, max_trades_per_day: int, label: str):
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    tasks = []
    for i in range(len(days)):
        day = days[i]
        curr_file = str(files[day])
        prev_file = str(files[days[i-1]]) if i > 0 else ""
        tasks.append((day, curr_file, prev_file))

    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker, initargs=(spot_all,)) as pool:
        results = pool.map(process_single_day, tasks)
        for res in results:
            all_trades.extend(res)

    df = pd.DataFrame(all_trades)
    
    # Filter 1: Entry minute cutoff
    df_filtered = df[df["entry_min"] <= cutoff_min].copy()

    # Filter 2: Max trades per day cap
    trades_by_day = []
    for day_val, g in df_filtered.groupby("date"):
        g_capped = g.iloc[:max_trades_per_day]
        trades_by_day.extend(g_capped.to_dict("records"))

    st = summarize(trades_by_day)

    print("\n" + "=" * 115)
    print(f"PROFIT OPTIMIZATION: {label}")
    print("=" * 115)
    print(f"Total Trades : {st['trades']}")
    print(f"Win Rate     : {st['wr']:.1f}%")
    print(f"Net Points   : {st['pts']:+.2f} pts")
    print(f"Net Profit   : Rs {st['rs']:+,d}")
    print(f"Profit Factor: {st['pf']:.2f}")

    print_yearly_breakdown(trades_by_day)

def main():
    print("Running Deep Profit Optimization Simulations...")
    test_optimized_rules(10.0, 15.0, 810, 3, "Cutoff 01:30 PM + Max 3 Trades/Day")

if __name__ == "__main__":
    main()
