"""Test Profit Optimization: Session Window 09:20 AM to 01:30 PM vs 02:00 PM."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from backtest_5y_optimized import main as run_optimized_backtest, process_single_day, load_spot, option_files, init_worker, summarize, print_yearly_breakdown
from multiprocessing import Pool, cpu_count

def test_cutoff(cutoff_minute: int, label: str):
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
    df_opt = df[df["entry_min"] <= cutoff_minute].copy()

    trades_opt = df_opt.to_dict("records")
    st = summarize(trades_opt)

    print("\n" + "=" * 115)
    print(f"SESSION WINDOW OPTIMIZATION: {label} (Cutoff Minute {cutoff_minute})")
    print("=" * 115)
    print(f"Total Trades : {st['trades']}")
    print(f"Win Rate     : {st['wr']:.1f}%")
    print(f"Net Points   : {st['pts']:+.2f} pts")
    print(f"Net Profit   : Rs {st['rs']:+,d}")
    print(f"Profit Factor: {st['pf']:.2f}")

    print_yearly_breakdown(trades_opt)

def main():
    print("Running Session Optimization Tests (09:20 to 01:30 PM)...")
    test_cutoff(810, "09:20 AM to 01:30 PM (Cutoff 13:30)")

if __name__ == "__main__":
    main()
