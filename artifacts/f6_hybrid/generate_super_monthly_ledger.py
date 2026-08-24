"""Extract Full 78-Month PnL Grid & Yearly Breakdown for V2_Super_Only Champion.

Outputs:
  - Month-by-Month Matrix (Jan-Dec for 2020, 2021, 2022, 2023, 2024, 2025, 2026)
  - % Positive Months per Year
  - Total Net Realized Profit, Max Drawdown, Profit Factor, Calmar Ratio
"""

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

# Super-Only Mask
entries_mask = super_setup

@torch.inference_mode()
def run_super_only_monthly():
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

    sl_m = 1.5
    tp_m = 3.0
    trail_trig_m = 0.75
    trail_dist_m = 0.4

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

    exit_sl_px = dyn_sl_barrier.gather(1, sl_first.clamp(max=max_future-1).unsqueeze(1)).squeeze(1)
    exit_tp_px = tp_barrier.squeeze(1)
    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px))

    pts = (exit_px - ep) * 0.50
    rs_net = pts * LOT_SIZE - FEE

    rs_cpu = rs_net.cpu().numpy()
    pts_cpu = pts.cpu().numpy()
    d_idx_cpu = d_idx.cpu().numpy()
    b_idx_cpu = b_idx.cpu().numpy()

    order = np.lexsort((b_idx_cpu, d_idx_cpu))
    days_sorted = d_idx_cpu[order]
    rs_sorted = rs_cpu[order]
    pts_sorted = pts_cpu[order]

    df_trades = pd.DataFrame({"day_idx": days_sorted, "rs": rs_sorted, "pts": pts_sorted})
    df_trades["date"] = [days[i] for i in df_trades["day_idx"]]
    df_trades["year"] = df_trades["date"].str[:4]
    df_trades["month"] = df_trades["date"].str[:7]
    df_trades["month_num"] = df_trades["date"].str[5:7].astype(int)

    monthly_pnl = df_trades.groupby("month")["rs"].sum()
    all_months = sorted(list(set(d[:7] for d in days)))
    monthly_pnl = monthly_pnl.reindex(all_months, fill_value=0.0)

    # Monthly Pivot Table
    records = []
    for m, val in monthly_pnl.items():
        records.append({"year": m[:4], "month_num": int(m[5:7]), "pnl": val})
    df_m = pd.DataFrame(records)
    pivot = df_m.pivot(index="year", columns="month_num", values="pnl").fillna(0.0)

    # Calculate Totals & Stats
    print("=" * 135)
    print("CHAMPION CONSISTENT STRATEGY: SUPER SETUP ONLY (TRAIL=0.75x/0.40x, SL=1.50x, TP=3.00x)")
    print("=" * 135)
    print("\n>>> MONTH-BY-MONTH REALIZED PNL MATRIX (Rs):")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    header = f"{'Year':6s} | " + " | ".join(f"{m:9s}" for m in month_names) + f" | {'Total (Rs)':14s} | {'% Green':7s}"
    print(header)
    print("-" * 155)

    year_totals = {}
    total_green = 0
    total_months_count = 0

    for yr in sorted(pivot.index):
        row_vals = pivot.loc[yr]
        row_strs = []
        yr_tot = 0.0
        yr_green = 0
        yr_mos = 0
        for m_idx in range(1, 13):
            if m_idx in row_vals and not (yr == "2026" and m_idx > 8):
                v = row_vals[m_idx]
                yr_tot += v
                yr_mos += 1
                total_months_count += 1
                if v > 0:
                    yr_green += 1
                    total_green += 1
                    row_strs.append(f"{v:+9.0f}")
                elif v < 0:
                    row_strs.append(f"{v:+9.0f}")
                else:
                    row_strs.append(f"{'0':>9s}")
            else:
                row_strs.append(f"{'-':>9s}")

        yr_pct = (yr_green / yr_mos * 100.0) if yr_mos > 0 else 0.0
        year_totals[yr] = yr_tot
        print(f"{yr:6s} | " + " | ".join(row_strs) + f" | Rs {yr_tot:+11.2f} | {yr_pct:6.1f}%")

    tot_pnl = sum(rs_sorted)
    wins = [r for r in rs_sorted if r > 0]
    losses = [r for r in rs_sorted if r <= 0]
    eq = np.cumsum(rs_sorted)
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max(peak - eq))
    calmar = tot_pnl / max_dd

    print("-" * 155)
    print(f"\n>>> 7-YEAR MASTER PERFORMANCE METRICS:")
    print(f"  • Total Trades:          {len(rs_sorted):,} trades (Avg: {len(rs_sorted)/N_DAYS:.1f} trades/day)")
    print(f"  • Win Rate:              {len(wins)/len(rs_sorted)*100:.2f}% ({len(wins):,} Wins / {len(losses):,} Losses)")
    print(f"  • Net Realized Profit:   Rs {tot_pnl:+,.2f} (+{sum(pts_sorted):+,.2f} net points)")
    print(f"  • Profit Factor:         {sum(wins)/abs(sum(losses)):.3f}")
    print(f"  • Max Drawdown:          Rs {max_dd:,.2f}")
    print(f"  • Calmar Ratio:          {calmar:.3f}")
    print(f"  • Profitable Months:     {total_green} out of {total_months_count} months ({total_green/total_months_count*100:.1f}% Monthly Win Rate)")
    print("=" * 135)


if __name__ == "__main__":
    run_super_only_monthly()
