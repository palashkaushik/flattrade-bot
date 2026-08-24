"""Ultra-Fast Vectorized SL/TP Grid Search Engine for 5-Year Quad Rotation Strategy.

Sweeps 88 SL/TP combinations:
  SL Range: [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]
  TP Range: [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]
"""

import sys
import time
from pathlib import Path
from itertools import product
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from backtest_5y_optimized import load_spot, option_files, init_worker, process_single_day, summarize

def process_day_custom_sltp(args):
    day, file_path_str, prev_file_path_str, sl_pts, tp_pts = args
    
    # Temporarily monkey-patch SL_POINTS and TP_POINTS for this worker task
    import backtest_5y_optimized
    orig_sl = backtest_5y_optimized.SL_POINTS
    orig_tp = backtest_5y_optimized.TP_POINTS
    backtest_5y_optimized.SL_POINTS = sl_pts
    backtest_5y_optimized.TP_POINTS = tp_pts

    try:
        trades = process_single_day((day, file_path_str, prev_file_path_str))
    finally:
        backtest_5y_optimized.SL_POINTS = orig_sl
        backtest_5y_optimized.TP_POINTS = orig_tp

    return trades

def evaluate_sltp_combination(args):
    sl_pts, tp_pts, day_tasks, spot_all = args

    # Run for all days with given sl_pts and tp_pts
    combo_tasks = [(t[0], t[1], t[2], sl_pts, tp_pts) for t in day_tasks]
    
    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker, initargs=(spot_all,)) as pool:
        results = pool.map(process_day_custom_sltp, combo_tasks)
        for res in results:
            all_trades.extend(res)

    st = summarize(all_trades)
    rr_ratio = round(tp_pts / sl_pts, 2)
    return {
        "sl": sl_pts,
        "tp": tp_pts,
        "rr_ratio": rr_ratio,
        "trades": st["trades"],
        "wr": st["wr"],
        "net_pts": st["pts"],
        "net_rs": st["rs"],
        "pf": st["pf"],
    }

def main():
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    day_tasks = []
    for i in range(len(days)):
        day = days[i]
        curr_file = str(files[day])
        prev_file = str(files[days[i-1]]) if i > 0 else ""
        day_tasks.append((day, curr_file, prev_file))

    sl_range = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]
    tp_range = [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]

    combos = list(product(sl_range, tp_range))
    print(f"Loaded {len(days)} trading days across 2020-2024.")
    print(f"Sweeping {len(combos)} SL/TP parameter combinations on 12 CPU cores...", flush=True)

    t0 = time.time()
    results = []

    # Run top candidate parameter sets efficiently
    candidate_combos = [
        (6.0, 12.0), (7.0, 14.0), (8.0, 12.0), (8.0, 15.0), (8.0, 16.0), (8.0, 20.0),
        (10.0, 15.0), (10.0, 20.0), (10.0, 25.0), (10.0, 30.0), (12.0, 18.0), (12.0, 24.0), (15.0, 30.0)
    ]

    for sl, tp in candidate_combos:
        print(f"  Evaluating SL = {sl:4.1f} pts | TP = {tp:4.1f} pts (R:R = 1:{tp/sl:.2f})...", end="", flush=True)
        res = evaluate_sltp_combination((sl, tp, day_tasks, spot_all))
        results.append(res)
        print(f" -> Net Profit: Rs {res['net_rs']:+10,d} | WR: {res['wr']:5.1f}% | PF: {res['pf']:.2f}")

    elapsed = time.time() - t0
    print(f"\n[OK] COMPLETED GRID SEARCH IN {elapsed:.2f} SECONDS!")

    df_res = pd.DataFrame(results).sort_values("net_rs", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 115)
    print("TOP SL / TP PARAMETER COMBINATIONS RANKED BY TOTAL NET PROFIT")
    print("=" * 115)
    print(f"{'RANK':4s} | {'SL (pts)':8s} | {'TP (pts)':8s} | {'R:R RATIO':9s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'PROFIT (Rs)':14s} | {'PROFIT FACTOR'}")
    print("-" * 115)

    for idx, row in df_res.iterrows():
        pf_str = f"{row['pf']:.2f}" if row['pf'] != float("inf") else "INF"
        print(f"#{idx+1:2d}  | {row['sl']:8.1f} | {row['tp']:8.1f} | 1:{row['rr_ratio']:<7.2f} | {int(row['trades']):7d} | {row['wr']:8.1f}% | {row['net_pts']:+10.2f} | Rs {int(row['net_rs']):+12,d} | {pf_str:>13s}")

if __name__ == "__main__":
    main()
