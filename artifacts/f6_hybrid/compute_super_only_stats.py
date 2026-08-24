"""Compute exact trade statistics for Super Setup Only Consistent Champion."""

from __future__ import annotations

import json
import sys
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
    LOT_SIZE,
    FEE,
    BASE_SESSION_START,
    BASE_SESSION_END,
)

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
N_DAYS = len(days)

s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
d_atr = compute_atr_gpu(d_h, d_l, d_c, period=14)

prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)
super_setup = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
entries_mask = super_setup

@torch.inference_mode()
def compute_super_only_stats():
    coords = torch.nonzero(entries_mask, as_tuple=False)
    d_idx = coords[:, 0]
    b_idx = coords[:, 1]
    ep = d_c[d_idx, b_idx]
    base_atr = d_atr[d_idx, b_idx].clamp(min=5.0, max=25.0)

    max_future = 345 - BASE_SESSION_START - 1
    col_offsets = torch.arange(max_future, device=device).unsqueeze(0)
    col_idx = (b_idx + 1).unsqueeze(1) + col_offsets
    valid = (col_idx < BASE_SESSION_END) & (col_idx < 375)
    col_safe = col_idx.clamp(max=374)

    d_exp = d_idx.unsqueeze(1).expand(-1, max_future)
    fut_h = d_h[d_exp, col_safe]
    fut_l = d_l[d_exp, col_safe]

    fut_h_m = torch.where(valid, fut_h, torch.tensor(-1e9, device=device))
    fut_l_m = torch.where(valid, fut_l, torch.tensor(1e9, device=device))
    eod_px = d_c[d_idx, BASE_SESSION_END - 1]

    fut_h_clean = torch.where(valid, fut_h, ep.unsqueeze(1))
    running_peaks = torch.cummax(fut_h_clean, dim=1).values
    ep_exp = ep.unsqueeze(1)

    sl_m = 1.50
    tp_m = 3.00
    trail_trig_m = 0.75
    trail_dist_m = 0.40

    sl_d = (sl_m * base_atr).clamp(min=5.0, max=30.0)
    tp_d = (tp_m * base_atr).clamp(min=8.0, max=60.0)

    init_sl = ep_exp - sl_d.unsqueeze(1)
    tp_barrier = ep_exp + tp_d.unsqueeze(1)

    trig_gain = (trail_trig_m * base_atr).unsqueeze(1)
    trail_d = (trail_dist_m * base_atr).unsqueeze(1)
    gains = running_peaks - ep_exp
    is_trailing = gains >= trig_gain
    trailing_sl_level = running_peaks - trail_d
    dyn_sl_barrier = torch.where(is_trailing, torch.maximum(init_sl, trailing_sl_level), init_sl)

    hit_sl = (fut_l_m <= dyn_sl_barrier)
    hit_tp = (fut_h_m >= tp_barrier)

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    exit_bar_offset = torch.where(sl_exits, sl_first, torch.where(tp_exits, tp_first, max_future - 1))
    duration_min = exit_bar_offset + 1

    exit_sl_px = dyn_sl_barrier.gather(1, sl_first.clamp(max=max_future-1).unsqueeze(1)).squeeze(1)
    exit_tp_px = tp_barrier.squeeze(1)
    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px))

    pts = (exit_px - ep) * 0.50
    rs_net = pts * LOT_SIZE - FEE

    pts_cpu = pts.cpu().numpy()
    rs_cpu = rs_net.cpu().numpy()
    d_idx_cpu = d_idx.cpu().numpy()
    sl_dist_cpu = (sl_d * 0.50).cpu().numpy()
    tp_dist_cpu = (tp_d * 0.50).cpu().numpy()
    dur_cpu = duration_min.cpu().numpy()
    sl_ex_cpu = sl_exits.cpu().numpy()
    tp_ex_cpu = tp_exits.cpu().numpy()

    trades_per_day = pd.Series(d_idx_cpu).value_counts().reindex(range(N_DAYS), fill_value=0)

    win_mask = rs_cpu > 0
    loss_mask = rs_cpu <= 0

    wins_pts = pts_cpu[win_mask]
    wins_rs = rs_cpu[win_mask]
    wins_dur = dur_cpu[win_mask]

    loss_pts = pts_cpu[loss_mask]
    loss_rs = rs_cpu[loss_mask]
    loss_dur = dur_cpu[loss_mask]

    print("=" * 90)
    print("EXACT TRADE METRICS: SUPER SETUP ONLY (CONSISTENT CHAMPION)")
    print("=" * 90)
    print(f"\n--- 1. TRADE FREQUENCY & VOLUME ---")
    print(f"  * Total Trades (7 Years):       {len(pts_cpu):,} trades across {N_DAYS:,} trading days")
    print(f"  * Average Trades Per Day:       {trades_per_day.mean():.2f} trades/day")
    print(f"  * Median Trades Per Day:        {trades_per_day.median():.0f} trades/day")
    print(f"  * Daily Trade Range (Min/Max):  {trades_per_day.min()} to {trades_per_day.max()} trades/day")
    print(f"  * Active Trading Days:          {(trades_per_day > 0).sum():,} days (Avg on active days: {trades_per_day[trades_per_day > 0].mean():.2f} trades/day)")
    print(f"  * Zero-Trade Days:              {(trades_per_day == 0).sum()} days ({(trades_per_day == 0).sum()/N_DAYS*100:.1f}%)")

    print(f"\n--- 2. STOP LOSS (SL) METRICS ---")
    print(f"  * Average Initial SL Distance:  {np.mean(sl_dist_cpu):.2f} option pts (~Rs {np.mean(sl_dist_cpu)*LOT_SIZE + FEE:.2f} max risk per lot)")
    print(f"  * Initial SL Range:             {np.min(sl_dist_cpu):.2f} to {np.max(sl_dist_cpu):.2f} option pts")
    print(f"  * Average Realized Loss:        {np.mean(loss_pts):.2f} option pts (-Rs {abs(np.mean(loss_rs)):.2f} net loss per losing trade)")

    print(f"\n--- 3. TAKE PROFIT (TP) METRICS ---")
    print(f"  * Average Initial TP Distance:  {np.mean(tp_dist_cpu):.2f} option pts (~Rs {np.mean(tp_dist_cpu)*LOT_SIZE - FEE:.2f} target reward per lot)")
    print(f"  * Initial TP Range:             {np.min(tp_dist_cpu):.2f} to {np.max(tp_dist_cpu):.2f} option pts")
    print(f"  * Average Realized Win:         +{np.mean(wins_pts):.2f} option pts (+Rs {np.mean(wins_rs):.2f} net profit per winning trade)")
    print(f"  * Trailing SL Trigger Gain:     +{0.75 * np.mean(base_atr.cpu().numpy()) * 0.50:.2f} option pts (trails tightly at {0.40 * np.mean(base_atr.cpu().numpy()) * 0.50:.2f} pts)")

    print(f"\n--- 4. HOLDING DURATION & EXITS ---")
    print(f"  * Average Trade Duration:       {np.mean(dur_cpu):.1f} minutes")
    print(f"  * Winning Trade Duration:       {np.mean(wins_dur):.1f} minutes (lets profits run)")
    print(f"  * Losing Trade Duration:        {np.mean(loss_dur):.1f} minutes (cuts losses fast)")
    print(f"  * Full Target TP Hits:          {tp_ex_cpu.sum():,} ({tp_ex_cpu.sum()/len(pts_cpu)*100:.1f}%)")
    print(f"  * Trailing SL / SL Hits:        {sl_ex_cpu.sum():,} ({sl_ex_cpu.sum()/len(pts_cpu)*100:.1f}%)")
    print(f"  * End of Day (EOD) Exits:       {(~sl_ex_cpu & ~tp_ex_cpu).sum():,} ({(~sl_ex_cpu & ~tp_ex_cpu).sum()/len(pts_cpu)*100:.1f}%)")
    print("=" * 90)

if __name__ == "__main__":
    compute_super_only_stats()
