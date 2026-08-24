"""Deep Statistical Audit Engine to Identify Profit Leakage in 5-Year Dataset."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from backtest_5y_optimized import main as run_optimized_backtest, process_single_day, load_spot, option_files, init_worker
from multiprocessing import Pool, cpu_count
import time

def main():
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    tasks = []
    for i in range(len(days)):
        day = days[i]
        curr_file = str(files[day])
        prev_file = str(files[days[i-1]]) if i > 0 else ""
        tasks.append((day, curr_file, prev_file))

    print(f"Auditing all trades across {len(days)} days (2020-2024)...")
    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker, initargs=(spot_all,)) as pool:
        results = pool.map(process_single_day, tasks)
        for res in results:
            all_trades.extend(res)

    df = pd.DataFrame(all_trades)
    df["entry_hour"] = df["entry_min"] // 60
    df["hour_str"] = df["entry_hour"].apply(lambda h: f"{h:02d}:00")
    df["mode"] = df["is_rev"].apply(lambda r: "REVERSE" if r else "NORMAL")

    print("\n" + "=" * 115)
    print("1. BREAKDOWN BY EXIT REASON")
    print("=" * 115)
    print(f"{'REASON':25s} | {'COUNT':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'NET PROFIT (Rs)':16s} | {'AVG PTS/TRADE'}")
    print("-" * 115)
    for reason, g in df.groupby("reason"):
        wins = g[g["pts"] > 0]
        wr = len(wins) / len(g) * 100
        net_pts = g["pts"].sum()
        net_rs = g["rs"].sum()
        avg_pts = g["pts"].mean()
        print(f"{reason:25s} | {len(g):7d} | {wr:8.1f}% | {net_pts:+10.2f} | Rs {net_rs:+14,d} | {avg_pts:+11.2f}")

    print("\n" + "=" * 115)
    print("2. BREAKDOWN BY ENTRY HOUR")
    print("=" * 115)
    print(f"{'HOUR WINDOW':15s} | {'COUNT':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'NET PROFIT (Rs)':16s} | {'AVG PTS/TRADE'}")
    print("-" * 115)
    for hour, g in df.groupby("hour_str"):
        wins = g[g["pts"] > 0]
        wr = len(wins) / len(g) * 100
        net_pts = g["pts"].sum()
        net_rs = g["rs"].sum()
        avg_pts = g["pts"].mean()
        print(f"{hour:15s} | {len(g):7d} | {wr:8.1f}% | {net_pts:+10.2f} | Rs {net_rs:+14,d} | {avg_pts:+11.2f}")

    print("\n" + "=" * 115)
    print("3. BREAKDOWN BY EXECUTION MODE (NORMAL vs REVERSE)")
    print("=" * 115)
    print(f"{'MODE':15s} | {'COUNT':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'NET PROFIT (Rs)':16s} | {'AVG PTS/TRADE'}")
    print("-" * 115)
    for mode, g in df.groupby("mode"):
        wins = g[g["pts"] > 0]
        wr = len(wins) / len(g) * 100
        net_pts = g["pts"].sum()
        net_rs = g["rs"].sum()
        avg_pts = g["pts"].mean()
        print(f"{mode:15s} | {len(g):7d} | {wr:8.1f}% | {net_pts:+10.2f} | Rs {net_rs:+14,d} | {avg_pts:+11.2f}")

if __name__ == "__main__":
    main()
