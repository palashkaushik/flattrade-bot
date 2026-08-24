"""Extract Comprehensive Non-Walk-Forward 7-Year (2020-2026) Results With vs Without VIX.

Computes exact 7-year totals and yearly breakdown for all key configurations.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.optimus_vix_atr_gpu import (
    load_gpu_all_data,
    compute_quad_stochastics_gpu,
    compute_atr_gpu,
    simulate_vix_dynamic_batch_gpu,
    LOT_SIZE,
    FEE,
    BASE_SESSION_START,
    BASE_SESSION_END,
    DAILY_LOSS_RS,
)

d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
N_DAYS = len(days)

s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
d_atr = compute_atr_gpu(d_h, d_l, d_c, period=14)

prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)
super_setup = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_setup = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
entries_mask = super_setup | flag_setup

# Key configurations to evaluate side-by-side
configs_to_compare = [
    # 1. 3.0x SL / 6.0x TP
    {"desc": "Static ATR (No VIX)", "scaling_type": "static", "sl_mult": 3.0, "tp_mult": 6.0},
    {"desc": "Power VIX (gamma=1.00)", "scaling_type": "power", "sl_mult": 3.0, "tp_mult": 6.0, "gamma": 1.0},
    {"desc": "Power VIX (gamma=0.75)", "scaling_type": "power", "sl_mult": 3.0, "tp_mult": 6.0, "gamma": 0.75},
    {"desc": "Power VIX (gamma=0.50)", "scaling_type": "power", "sl_mult": 3.0, "tp_mult": 6.0, "gamma": 0.50},
    {"desc": "Power VIX (gamma=0.25)", "scaling_type": "power", "sl_mult": 3.0, "tp_mult": 6.0, "gamma": 0.25},
    {"desc": "3-Regime VIX (0.80/1.35)", "scaling_type": "discrete_regime", "sl_mult": 3.0, "tp_mult": 6.0, "low_scale": 0.80, "high_scale": 1.35, "tp_high_boost": 1.25},

    # 2. 2.5x SL / 5.0x TP
    {"desc": "Static ATR (No VIX)", "scaling_type": "static", "sl_mult": 2.5, "tp_mult": 5.0},
    {"desc": "Power VIX (gamma=1.00)", "scaling_type": "power", "sl_mult": 2.5, "tp_mult": 5.0, "gamma": 1.0},
    {"desc": "Power VIX (gamma=0.50)", "scaling_type": "power", "sl_mult": 2.5, "tp_mult": 5.0, "gamma": 0.50},
    {"desc": "3-Regime VIX (0.80/1.35)", "scaling_type": "discrete_regime", "sl_mult": 2.5, "tp_mult": 5.0, "low_scale": 0.80, "high_scale": 1.35, "tp_high_boost": 1.25},

    # 3. 2.5x SL / 4.0x TP
    {"desc": "Static ATR (No VIX)", "scaling_type": "static", "sl_mult": 2.5, "tp_mult": 4.0},
    {"desc": "Power VIX (gamma=1.00)", "scaling_type": "power", "sl_mult": 2.5, "tp_mult": 4.0, "gamma": 1.0},
    {"desc": "Power VIX (gamma=0.50)", "scaling_type": "power", "sl_mult": 2.5, "tp_mult": 4.0, "gamma": 0.50},

    # 4. 2.5x SL / 3.5x TP
    {"desc": "Static ATR (No VIX)", "scaling_type": "static", "sl_mult": 2.5, "tp_mult": 3.5},
    {"desc": "Power VIX (gamma=1.00)", "scaling_type": "power", "sl_mult": 2.5, "tp_mult": 3.5, "gamma": 1.0},
    {"desc": "Power VIX (gamma=0.50)", "scaling_type": "power", "sl_mult": 2.5, "tp_mult": 3.5, "gamma": 0.50},

    # 5. 2.0x SL / 5.0x TP
    {"desc": "Static ATR (No VIX)", "scaling_type": "static", "sl_mult": 2.0, "tp_mult": 5.0},
    {"desc": "Power VIX (gamma=1.00)", "scaling_type": "power", "sl_mult": 2.0, "tp_mult": 5.0, "gamma": 1.0},
    {"desc": "Power VIX (gamma=0.50)", "scaling_type": "power", "sl_mult": 2.0, "tp_mult": 5.0, "gamma": 0.50},
]

# Run full 7-year Non-Walk-Forward
res_full = simulate_vix_dynamic_batch_gpu(entries_mask, d_h, d_l, d_c, d_atr, d_vix, configs_to_compare, day_mask=None)

# Run yearly masks
years = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
yearly_results = {}
for y in years:
    y_mask = torch.tensor([d.startswith(y) for d in days], dtype=torch.bool, device=d_h.device)
    y_res = simulate_vix_dynamic_batch_gpu(entries_mask, d_h, d_l, d_c, d_atr, d_vix, configs_to_compare, day_mask=y_mask)
    yearly_results[y] = y_res

# Compile Output Table
print("=" * 135)
print("7-YEAR NON-WALK-FORWARD COMPARISON: WITH VIX vs WITHOUT VIX (2020-2026)")
print("=" * 135)
print(f"{'#':2s} | {'SLxTP':9s} | {'Model / Scaling Description':26s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
print("-" * 135)

for i, (cfg, r) in enumerate(zip(configs_to_compare, res_full), 1):
    sl_tp_str = f"{cfg['sl_mult']:.2f}x{cfg['tp_mult']:.2f}"
    print(f"{i:2d} | {sl_tp_str:9s} | {cfg['desc']:26s} | {r['trades']:7d} | {r['win_rate']:7.1f}% | {r['net_points']:+10.2f} | Rs {r['net_rs']:+12.2f} | {r['profit_factor']:6.3f} | Rs {r['max_drawdown_rs']:9.2f} | {r['calmar_ratio']:7.3f}")

# Print Year-by-Year Comparison Table for Top Winner vs No VIX
print("\n" + "=" * 135)
print("YEAR-BY-YEAR PnL BREAKDOWN (2020-2026): WITH VIX (Power gamma=1.0) vs WITHOUT VIX (Static ATR) for 3.00x SL / 6.00x TP")
print("=" * 135)
print(f"{'Year':6s} | {'WITHOUT VIX (Static ATR)':40s} | {'WITH VIX (Power gamma=1.00)':40s} | {'VIX Edge (Rs)':14s}")
print(f"{'':6s} | {'Trades':7s} {'WR':6s} {'Net Pts':9s} {'Net Rs':14s} | {'Trades':7s} {'WR':6s} {'Net Pts':9s} {'Net Rs':14s} |")
print("-" * 135)

for y in years:
    no_vix_res = yearly_results[y][0]
    with_vix_res = yearly_results[y][1]
    diff_rs = with_vix_res['net_rs'] - no_vix_res['net_rs']
    print(f"{y:6s} | {no_vix_res['trades']:7d} {no_vix_res['win_rate']:5.1f}% {no_vix_res['net_points']:+8.1f} Rs {no_vix_res['net_rs']:+11.2f} | {with_vix_res['trades']:7d} {with_vix_res['win_rate']:5.1f}% {with_vix_res['net_points']:+8.1f} Rs {with_vix_res['net_rs']:+11.2f} | Rs {diff_rs:+11.2f}")

tot_no_vix = res_full[0]
tot_with_vix = res_full[1]
tot_diff = tot_with_vix['net_rs'] - tot_no_vix['net_rs']
print("-" * 135)
print(f"{'TOTAL':6s} | {tot_no_vix['trades']:7d} {tot_no_vix['win_rate']:5.1f}% {tot_no_vix['net_points']:+8.1f} Rs {tot_no_vix['net_rs']:+11.2f} | {tot_with_vix['trades']:7d} {tot_with_vix['win_rate']:5.1f}% {tot_with_vix['net_points']:+8.1f} Rs {tot_with_vix['net_rs']:+11.2f} | Rs {tot_diff:+11.2f}")
print("=" * 135)

# Save JSON comparison
out_data = {
    "overall_comparison": [
        {**cfg, **r} for cfg, r in zip(configs_to_compare, res_full)
    ],
    "yearly_breakdown": {
        y: [{**cfg, **r} for cfg, r in zip(configs_to_compare, yearly_results[y])]
        for y in years
    }
}

out_file = ROOT / "artifacts" / "f6_hybrid" / "non_walk_forward_vix_comparison.json"
out_file.write_text(json.dumps(out_data, indent=2, default=float), encoding="utf-8")
print(f"\n[Saved Non-Walk-Forward Comparison JSON]: {out_file}")
