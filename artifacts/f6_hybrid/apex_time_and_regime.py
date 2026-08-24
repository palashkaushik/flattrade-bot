"""Detailed Time-of-Day, Volatility, and Trade Frequency Sweet Spot Analyzer."""

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
prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)
super_setup = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_setup = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
entries_mask = super_setup | flag_setup


@torch.inference_mode()
def analyze_time_and_frequency():
    coords = torch.nonzero(entries_mask, as_tuple=False)
    d_idx = coords[:, 0]
    b_idx = coords[:, 1]
    ep = d_c[d_idx, b_idx]

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
    gains = running_peaks - ep_exp

    initial_sl_pts = 6.0
    lock_trigger_pts = 12.0
    locked_profit_pts = 10.0
    trail_dist_pts = 4.0
    hard_tp_pts = 20.0

    init_sl = ep_exp - (initial_sl_pts * 2.0)
    is_locked = gains >= (lock_trigger_pts * 2.0)
    locked_sl = ep_exp + (locked_profit_pts * 2.0)
    trail_sl = running_peaks - (trail_dist_pts * 2.0)

    dyn_sl = init_sl.clone()
    dyn_sl = torch.where(is_locked, torch.maximum(dyn_sl, locked_sl), dyn_sl)
    dyn_sl = torch.where(is_locked, torch.maximum(dyn_sl, trail_sl), dyn_sl)
    tp_barrier = ep_exp + (hard_tp_pts * 2.0)

    hit_sl = (fut_l_m <= dyn_sl)
    hit_tp = (fut_h_m >= tp_barrier)

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    exit_sl_px = dyn_sl.gather(1, sl_first.clamp(max=max_future-1).unsqueeze(1)).squeeze(1)
    exit_tp_px = tp_barrier.squeeze(1)
    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px))

    pts = (exit_px - ep) * 0.50
    rs_net = pts * LOT_SIZE - FEE

    df = pd.DataFrame({
        "day_idx": d_idx.cpu().numpy(),
        "bar_idx": b_idx.cpu().numpy(),
        "pts": pts.cpu().numpy(),
        "rs_net": rs_net.cpu().numpy(),
        "vix": d_vix[d_idx, b_idx].cpu().numpy(),
    })
    
    # Calculate time of day (bar 0 = 09:15, bar 15 = 09:30, bar 75 = 10:30, etc.)
    df["minute_of_day"] = df["bar_idx"] + 15  # from 09:00
    df["hour"] = 9 + (df["minute_of_day"] // 60)
    df["minute"] = df["minute_of_day"] % 60
    df["time_slot"] = pd.cut(
        df["bar_idx"],
        bins=[-1, 45, 105, 165, 225, 285, 375],
        labels=["09:15-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00", "13:00-14:00", "14:00-15:30"]
    )

    print("\n--- 4. HOURLY BREAKDOWN OF PERFORMANCE ---")
    print(f"{'Time Slot':15s} | {'Trades':7s} | {'Win Rate':9s} | {'Avg Win':9s} | {'Avg Loss':9s} | {'Avg Pts':8s} | {'Tot Pts':10s} | {'Tot Rs':14s} | {'PF':6s}")
    print("-" * 105)
    for slot, grp in df.groupby("time_slot", observed=True):
        w = grp[grp["rs_net"] > 0]
        l = grp[grp["rs_net"] <= 0]
        wr = len(w) / len(grp) * 100.0
        pf = w["rs_net"].sum() / abs(l["rs_net"].sum()) if len(l) > 0 and abs(l["rs_net"].sum()) > 0 else 99.0
        print(f"{str(slot):15s} | {len(grp):7d} | {wr:7.1f}% | +{w['pts'].mean():5.2f} pt | {l['pts'].mean():6.2f} pt | {grp['pts'].mean():+7.2f} | {grp['pts'].sum():+9.1f} | Rs {grp['rs_net'].sum():+11.1f} | {pf:6.3f}")

    # Daily Activity Tier Analysis (how many signals occur on that day)
    day_counts = df.groupby("day_idx").size()
    df["daily_activity_tier"] = df["day_idx"].map(
        lambda d: "Low (<8 trades/day)" if day_counts.get(d, 0) < 8
        else ("Medium (8-16 trades/day)" if day_counts.get(d, 0) <= 16
        else "High (>16 trades/day)")
    )

    print("\n--- 5. PERFORMANCE BY DAILY MARKET REGIME (DAILY SIGNAL INTENSITY) ---")
    print(f"{'Activity Regime':25s} | {'Trades':7s} | {'Win Rate':9s} | {'Avg Win':9s} | {'Avg Loss':9s} | {'Avg Pts':8s} | {'Tot Pts':10s} | {'Tot Rs':14s} | {'PF':6s}")
    print("-" * 115)
    for regime in ["Low (<8 trades/day)", "Medium (8-16 trades/day)", "High (>16 trades/day)"]:
        grp = df[df["daily_activity_tier"] == regime]
        w = grp[grp["rs_net"] > 0]
        l = grp[grp["rs_net"] <= 0]
        wr = len(w) / len(grp) * 100.0
        pf = w["rs_net"].sum() / abs(l["rs_net"].sum()) if len(l) > 0 and abs(l["rs_net"].sum()) > 0 else 99.0
        print(f"{regime:25s} | {len(grp):7d} | {wr:7.1f}% | +{w['pts'].mean():5.2f} pt | {l['pts'].mean():6.2f} pt | {grp['pts'].mean():+7.2f} | {grp['pts'].sum():+9.1f} | Rs {grp['rs_net'].sum():+11.1f} | {pf:6.3f}")


if __name__ == "__main__":
    analyze_time_and_frequency()
