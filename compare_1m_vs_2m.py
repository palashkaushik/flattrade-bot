"""Side-by-Side Comparison Engine: 1-Minute ONLY vs 2-Minute ONLY vs Combined (1m + 2m)."""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from backtest_5y_optimized import load_spot, option_files, init_worker, process_single_day, summarize, print_yearly_breakdown

def run_mode_backtest(tf_mode: str):
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
    if tf_mode == "1m_only":
        df_filtered = df[df["tf"] == "1m"].copy()
    elif tf_mode == "2m_only":
        df_filtered = df[df["tf"] == "2m"].copy()
    else:
        df_filtered = df.copy()

    trades = df_filtered.to_dict("records")
    return trades

def main():
    print("Running 5-Year Multi-Timeframe Comparison (2020-2024)...", flush=True)

    trades_combined = run_mode_backtest("combined")
    df = pd.DataFrame(trades_combined)

    df_1m = df[df["tf"] == "1m"].to_dict("records")
    df_2m = df[df["tf"] == "2m"].to_dict("records")

    st_1m = summarize(df_1m)
    st_2m = summarize(df_2m)
    st_comb = summarize(trades_combined)

    print("\n" + "=" * 115)
    print("5-YEAR TIMEFRAME COMPARISON SUMMARY (2020 - 2024)")
    print("=" * 115)
    print(f"{'TIMEFRAME MODE':25s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'PROFIT (Rs)':14s} | {'PROFIT FACTOR'}")
    print("-" * 115)
    pf_1m_s = f"{st_1m['pf']:.2f}" if st_1m['pf'] != float("inf") else "INF"
    pf_2m_s = f"{st_2m['pf']:.2f}" if st_2m['pf'] != float("inf") else "INF"
    pf_comb_s = f"{st_comb['pf']:.2f}" if st_comb['pf'] != float("inf") else "INF"

    print(f"{'1-Minute Timeframe ONLY':25s} | {st_1m['trades']:7d} | {st_1m['wr']:8.1f}% | {st_1m['pts']:+10.2f} | Rs {st_1m['rs']:+12,d} | {pf_1m_s:>13s}")
    print(f"{'2-Minute Timeframe ONLY':25s} | {st_2m['trades']:7d} | {st_2m['wr']:8.1f}% | {st_2m['pts']:+10.2f} | Rs {st_2m['rs']:+12,d} | {pf_2m_s:>13s}")
    print(f"{'1m + 2m COMBINED (DUAL)':25s} | {st_comb['trades']:7d} | {st_comb['wr']:8.1f}% | {st_comb['pts']:+10.2f} | Rs {st_comb['rs']:+12,d} | {pf_comb_s:>13s}")

    print("\n" + "=" * 115)
    print("YEARLY COMPARISON: 1-MINUTE ONLY vs 2-MINUTE ONLY vs COMBINED")
    print("=" * 115)
    df["year"] = pd.to_datetime(df["date"]).dt.year

    for year, g in df.groupby("year"):
        g_1m = g[g["tf"] == "1m"].to_dict("records")
        g_2m = g[g["tf"] == "2m"].to_dict("records")
        g_all = g.to_dict("records")

        s1 = summarize(g_1m)
        s2 = summarize(g_2m)
        sa = summarize(g_all)

        print(f"\n--- YEAR {year} ---")
        print(f"  1m Only  : {s1['trades']:4d} trades | WR: {s1['wr']:5.1f}% | Net P&L: {s1['pts']:+7.2f} pts (Rs {s1['rs']:+9,d})")
        print(f"  2m Only  : {s2['trades']:4d} trades | WR: {s2['wr']:5.1f}% | Net P&L: {s2['pts']:+7.2f} pts (Rs {s2['rs']:+9,d})")
        print(f"  Combined : {sa['trades']:4d} trades | WR: {sa['wr']:5.1f}% | Net P&L: {sa['pts']:+7.2f} pts (Rs {sa['rs']:+9,d})")

if __name__ == "__main__":
    main()
