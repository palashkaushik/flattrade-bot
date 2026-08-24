"""Score Threshold & Session Breakdown for Ammu Rejection Strategy."""

import json
from pathlib import Path
import pandas as pd

import sys
ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.optimus_rejection_gpu_fast import load_or_build_signals, run_cuda_rejection_sweep

df_sig, candles_3m = load_or_build_signals()

grid = []
for sc in [0, 40, 50, 60, 70, 80, 90]:
    for sl in [0.3, 0.4, 0.5, 0.7, 1.0]:
        for tp in [1.5, 2.0, 2.5, 3.0, 4.0]:
            for tr in [6.0, 8.0, 10.0, 15.0]:
                for st in [2.0, 3.0, 5.0]:
                    grid.append({
                        "min_score": sc,
                        "sl_mult": sl,
                        "tp_mult": tp,
                        "trail_trigger": tr,
                        "trail_step": st,
                    })

print(f"Running Score Threshold Grid: {len(grid):,} combinations per session...")

sessions = [
    ("1. Morning Session (09:15-11:00)", 555, 660),
    ("2. Afternoon Session (13:30-15:00)", 810, 900),
    ("3. Combined Dual-Engine (09:15-11:00 + 13:30-15:00)", 555, 900),
]

for s_name, min_t, max_t in sessions:
    res = run_cuda_rejection_sweep(df_sig, candles_3m, s_name, min_t, max_t, grid, batch_size=100)
    df_r = pd.DataFrame(res)
    print(f"\n" + "=" * 145)
    print(f"SESSION: {s_name.upper()} — SCORE THRESHOLD COMPARISON (BEST CONFIG PER THRESHOLD)")
    print("=" * 145)
    print(f"{'Score':>6} | {'Trades':>7} | {'Daily WR':>8} | {'Trade WR':>8} | {'Net Points':>12} | {'Net Realized Rs':>17} | {'PF':>6} | {'Max DD':>10} | {'Calmar':>8} | {'Optimal Parameters'}")
    print("-" * 145)
    for sc in [0, 40, 50, 60, 70, 80, 90]:
        sub = df_r[df_r["min_score"] == sc]
        if sub.empty or sub["trades"].iloc[0] == 0:
            continue
        top = sub.sort_values(by="calmar_ratio", ascending=False).iloc[0]
        params_str = f"SL: {top['sl_mult']}x, TP: {top['tp_mult']}x, Trail: +{top['trail_trigger']:.0f}/{top['trail_step']:.0f} pts"
        print(f">={sc:<5} | {top['trades']:7d} | {top['daily_win_rate']:7.1f}% | {top['trade_win_rate']:7.1f}% | {top['net_points']:>+11.2f} | Rs {top['net_rs']:>+14,.2f} | {top['profit_factor']:6.3f} | Rs {top['max_drawdown']:>7,.2f} | {top['calmar_ratio']:8.2f} | {params_str}")
