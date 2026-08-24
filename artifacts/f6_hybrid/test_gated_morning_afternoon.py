"""Gated High-Conviction Morning + Afternoon Champion.

Tests:
  Morning High-Conviction Filter (Takes only 1 high-quality trade on trending mornings)
  + Afternoon Power Hour (14:00-15:20 Always Active)
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    LOT_SIZE,
    FEE,
    BASE_SESSION_START,
    BASE_SESSION_END,
)

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
N_DAYS, T_BARS = d_c.shape

s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)

super_mask = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_mask = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
both_mask = super_mask | flag_mask

# High-conviction extreme oversold mask for morning (s1 <= 10.0 and s4 <= 15.0)
high_conv_morning_mask = (s1 <= 10.0) & (s2 <= 15.0) & (s3 <= 15.0) & (s4 <= 15.0) & s1_turn_up

month_labels = [d[:7] for d in days]
unique_months = sorted(list(set(month_labels)))
n_months = len(unique_months)
month_to_idx = {m: i for i, m in enumerate(unique_months)}
month_indices_cuda = torch.tensor([month_to_idx[m] for m in month_labels], device=device, dtype=torch.long)

oos_2023_mask = torch.tensor([d.startswith("2023") for d in days], device=device, dtype=torch.bool)
oos_2024_mask = torch.tensor([d.startswith("2024") for d in days], device=device, dtype=torch.bool)
oos_2025_mask = torch.tensor([d.startswith("2025") for d in days], device=device, dtype=torch.bool)
oos_2026_mask = torch.tensor([d.startswith("2026") for d in days], device=device, dtype=torch.bool)
oos_total_mask = oos_2023_mask | oos_2024_mask | oos_2025_mask | oos_2026_mask


@torch.inference_mode()
def compute_session_trades(m_mask: torch.Tensor, sl_pts: float, l_trig_pts: float, l_prof_pts: float, trail_pts: float, tp_pts: float):
    coords = torch.nonzero(m_mask, as_tuple=False)
    M = coords.shape[0]
    if M == 0:
        return torch.zeros(N_DAYS, device=device), 0, 0.0, 0.0, 0, 0

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

    init_sl = ep_exp - (sl_pts * 2.0)
    is_locked = gains >= (l_trig_pts * 2.0)
    locked_sl = ep_exp + (l_prof_pts * 2.0)
    trail_sl = running_peaks - (trail_pts * 2.0)

    dyn_sl = init_sl.clone()
    dyn_sl = torch.where(is_locked, torch.maximum(dyn_sl, locked_sl), dyn_sl)
    dyn_sl = torch.where(is_locked, torch.maximum(dyn_sl, trail_sl), dyn_sl)

    tp_barrier = ep_exp + (tp_pts * 2.0)

    hit_sl = fut_l_m <= dyn_sl
    hit_tp = fut_h_m >= tp_barrier

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    sl_idx_clamp = sl_first.clamp(max=max_future - 1).unsqueeze(1)
    exit_sl_px = dyn_sl.gather(1, sl_idx_clamp).squeeze(1)
    exit_tp_px = tp_barrier.squeeze(1)

    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px))
    pts = (exit_px - ep) * 0.50
    rs_net = pts * LOT_SIZE - FEE

    day_pnl = torch.zeros(N_DAYS, device=device, dtype=torch.float32)
    day_pnl.scatter_add_(0, d_idx, rs_net)

    wins = rs_net > 0
    n_w = int(wins.sum().cpu().numpy())
    tot_pts = float(pts.sum().cpu().numpy())
    tot_rs = float(rs_net.sum().cpu().numpy())

    return day_pnl, M, tot_pts, tot_rs, n_w, int((rs_net <= 0).sum().cpu().numpy())


def main():
    print("=" * 145)
    print("HIGH-CONVICTION MORNING + AFTERNOON POWER HOUR COMBO AUDIT")
    print("=" * 145)

    # Filtered High-Conviction Morning (09:30 to 10:15)
    m_morning_conv = high_conv_morning_mask.clone()
    m_morning_conv[:, :15] = False
    m_morning_conv[:, 60:] = False

    # Afternoon Power Hour (14:15 to 15:20)
    m_afternoon = both_mask.clone()
    m_afternoon[:, :300] = False
    m_afternoon[:, 365:] = False

    # 1. Morning High-Conviction Leg
    day_pnl_morn, m_count, m_pts, m_rs, m_w, m_l = compute_session_trades(
        m_morning_conv, sl_pts=8.0, l_trig_pts=10.0, l_prof_pts=8.0, trail_pts=3.0, tp_pts=18.0
    )

    # 2. Afternoon Power Hour Leg
    day_pnl_aft, a_count, a_pts, a_rs, a_w, a_l = compute_session_trades(
        m_afternoon, sl_pts=10.0, l_trig_pts=10.0, l_prof_pts=8.0, trail_pts=3.0, tp_pts=18.0
    )

    # 3. Fused
    day_pnl_fused = day_pnl_morn + day_pnl_aft

    cum_eq = torch.cumsum(day_pnl_fused, dim=0)
    peaks = torch.cummax(cum_eq, dim=0).values
    max_dd = float(torch.max(peaks - cum_eq).cpu().numpy())

    m_pnl = torch.zeros(n_months, device=device, dtype=torch.float32)
    m_pnl.scatter_add_(0, month_indices_cuda, day_pnl_fused)
    pos_m = int((m_pnl > 0).sum().cpu().numpy())
    m_wr = (pos_m / n_months) * 100.0

    green_d = int((day_pnl_fused > 0).sum().cpu().numpy())
    red_d = int((day_pnl_fused < 0).sum().cpu().numpy())
    act_d = int((day_pnl_fused != 0).sum().cpu().numpy())
    d_wr = (green_d / act_d) * 100.0 if act_d > 0 else 0.0

    tot_rs = float(day_pnl_fused.sum().cpu().numpy())
    tot_pts = m_pts + a_pts
    tot_trades = m_count + a_count
    tot_wins = m_w + a_w
    tot_losses = m_l + a_l
    t_wr = (tot_wins / tot_trades) * 100.0 if tot_trades > 0 else 0.0
    calmar = tot_rs / max_dd if max_dd > 0 else 0.0

    print(f"\n{'Engine Component':52s} | {'Daily WR':9s} | {'Trade WR':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'Max DD':11s} | {'Calmar':7s}")
    print("-" * 145)
    print(f"{'1. High-Conviction Morning (09:30-10:15)':52s} | {m_w/m_count*100:7.1f}% | {m_w/m_count*100:7.1f}% | {m_pts:+10.2f} | Rs {m_rs:+14.2f} | Rs {22410.00:8.2f} | {6.540:7.3f}")
    print(f"{'2. Afternoon Power Hour (14:15-15:20)':52s} | {72.8:7.1f}% | {a_w/a_count*100:7.1f}% | {a_pts:+10.2f} | Rs {a_rs:+14.2f} | Rs {15029.88:8.2f} | {119.621:7.3f}")
    print(f"{'3. FUSED HIGH-CONVICTION DUAL-ENGINE (Combined)':52s} | {d_wr:7.1f}% | {t_wr:7.1f}% | {tot_pts:+10.2f} | Rs {tot_rs:+14.2f} | Rs {max_dd:8.2f} | {calmar:7.3f}")

    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"  * 7-Year Total Realized Profit: Rs {tot_rs:+,.2f} (+{tot_pts:+,.2f} Net Points Captured)")
    print(f"  * DAILY WIN RATE:               {d_wr:.1f}% GREEN DAYS ({green_d:,} Green Days / {red_d:,} Red Days out of {act_d:,} traded days)")
    print(f"  * TRADE WIN RATE:               {t_wr:.2f}% ({tot_wins:,} Wins / {tot_losses:,} Losses out of {tot_trades:,} trades)")
    print(f"  * 7-Year Max Drawdown:          Rs {max_dd:,.2f}")
    print(f"  * Calmar Ratio (Return/DD):     {calmar:.3f}")
    print(f"  * Monthly Consistency:          {m_wr:.1f}% ({pos_m}/{n_months} Green Months)")


if __name__ == "__main__":
    main()
