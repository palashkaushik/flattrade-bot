"""Optimus Wide-Parameter GPU Optimizer: SL >= 10.0 pts & TP >= 10.0 pts.

Gives options room to absorb 1-minute market noise and ride large trend moves.
100% Pure CUDA Vectorized across 7 Years (1,588 Days, 2020–2026).
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import itertools
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

print("=" * 135, flush=True)
print("OPTIMUS WIDE-PARAMETER GPU OPTIMIZER (SL >= 10.0 pts & TP >= 10.0 pts)", flush=True)
print(f"Device: {torch.cuda.get_device_name(0)} | Precision: TensorFloat32", flush=True)
print("=" * 135, flush=True)

d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
N_DAYS, T_BARS = d_c.shape
ALL_MONTHS = sorted(list(set(d[:7] for d in days)))

# Precompute Quad Stochastics & Entry Setup Masks on CUDA
s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)

super_mask = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_mask = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
both_mask = super_mask | flag_mask

month_labels = [d[:7] for d in days]
unique_months = sorted(list(set(month_labels)))
n_months = len(unique_months)
month_to_idx = {m: i for i, m in enumerate(unique_months)}
month_indices_cuda = torch.tensor([month_to_idx[m] for m in month_labels], device=device, dtype=torch.long)


@torch.inference_mode()
def run_wide_sweep(session_name: str, start_bar: int, end_bar: int, mid_skip_s: int, mid_skip_e: int, param_grid: list[dict], batch_size: int = 25):
    active_mask = both_mask.clone()
    active_mask[:, :start_bar] = False
    active_mask[:, end_bar:] = False
    if mid_skip_s < mid_skip_e:
        active_mask[:, mid_skip_s:mid_skip_e] = False

    coords = torch.nonzero(active_mask, as_tuple=False)
    M = coords.shape[0]
    if M == 0:
        return []

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

    gains_3d = gains.unsqueeze(0)
    ep_3d = ep_exp.unsqueeze(0)
    peaks_3d = running_peaks.unsqueeze(0)
    fut_l_3d = fut_l_m.unsqueeze(0)
    fut_h_3d = fut_h_m.unsqueeze(0)

    BIG = 999999
    results = []

    for i in range(0, len(param_grid), batch_size):
        chunk = param_grid[i: i + batch_size]
        B = len(chunk)

        b_init_sl = torch.tensor([p["initial_sl"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_lock_trig = torch.tensor([p["lock_trigger"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_locked_sl = torch.tensor([p["locked_profit"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_trail = torch.tensor([p["trail_dist"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_tp = torch.tensor([p["hard_tp"] * 2.0 for p in chunk], device=device).view(B, 1, 1)

        init_sl_3d = ep_3d - b_init_sl
        is_locked_3d = gains_3d >= b_lock_trig
        locked_sl_3d = ep_3d + b_locked_sl
        trail_sl_3d = peaks_3d - b_trail

        dyn_sl_3d = init_sl_3d.expand(B, M, max_future).clone()
        dyn_sl_3d = torch.where(is_locked_3d, torch.maximum(dyn_sl_3d, locked_sl_3d), dyn_sl_3d)
        dyn_sl_3d = torch.where(is_locked_3d, torch.maximum(dyn_sl_3d, trail_sl_3d), dyn_sl_3d)

        tp_barrier_3d = ep_3d + b_tp

        hit_sl_3d = fut_l_3d <= dyn_sl_3d
        hit_tp_3d = fut_h_3d >= tp_barrier_3d

        sl_any = hit_sl_3d.any(dim=2)
        tp_any = hit_tp_3d.any(dim=2)

        sl_first = torch.where(sl_any, torch.argmax(hit_sl_3d.int(), dim=2), BIG)
        tp_first = torch.where(tp_any, torch.argmax(hit_tp_3d.int(), dim=2), BIG)

        sl_exits = sl_any & (sl_first <= tp_first)
        tp_exits = tp_any & (~sl_exits)

        sl_idx_clamp = sl_first.clamp(max=max_future - 1).unsqueeze(2)
        exit_sl_px = dyn_sl_3d.gather(2, sl_idx_clamp).squeeze(2)
        exit_tp_px = tp_barrier_3d.squeeze(2)
        eod_px_2d = eod_px.unsqueeze(0).expand(B, M)

        exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px_2d))
        pts_2d = (exit_px - ep.unsqueeze(0)) * 0.50
        rs_net_2d = pts_2d * LOT_SIZE - FEE

        # Pure GPU Vectorized Aggregations
        d_idx_exp = d_idx.unsqueeze(0).expand(B, M)
        day_pnl_cuda = torch.zeros((B, N_DAYS), device=device, dtype=torch.float32)
        day_pnl_cuda.scatter_add_(1, d_idx_exp, rs_net_2d)

        cum_eq_cuda = torch.cumsum(day_pnl_cuda, dim=1)
        peaks_cuda = torch.cummax(cum_eq_cuda, dim=1).values
        drawdowns_cuda = peaks_cuda - cum_eq_cuda
        max_dds_cuda = torch.max(drawdowns_cuda, dim=1).values

        month_idx_exp = month_indices_cuda.unsqueeze(0).expand(B, N_DAYS)
        m_pnl_cuda = torch.zeros((B, n_months), device=device, dtype=torch.float32)
        m_pnl_cuda.scatter_add_(1, month_idx_exp, day_pnl_cuda)
        pos_months_cuda = (m_pnl_cuda > 0).sum(dim=1)
        month_wrs_cuda = (pos_months_cuda.float() / float(n_months)) * 100.0

        green_days_cuda = (day_pnl_cuda > 0).sum(dim=1)
        red_days_cuda = (day_pnl_cuda < 0).sum(dim=1)
        active_days_cuda = (day_pnl_cuda != 0).sum(dim=1)
        daily_wrs_cuda = (green_days_cuda.float() / active_days_cuda.clamp(min=1).float()) * 100.0

        tot_rs_cuda = day_pnl_cuda.sum(dim=1)
        tot_pts_cuda = pts_2d.sum(dim=1)
        calmar_cuda = tot_rs_cuda / max_dds_cuda.clamp(min=1.0)

        wins_mask = rs_net_2d > 0
        loss_mask = rs_net_2d <= 0
        n_wins_cuda = wins_mask.sum(dim=1)
        trade_wrs_cuda = (n_wins_cuda.float() / float(M)) * 100.0

        win_sums_cuda = (torch.where(wins_mask, rs_net_2d, torch.zeros_like(rs_net_2d))).sum(dim=1)
        loss_sums_cuda = (torch.where(loss_mask, rs_net_2d.abs(), torch.zeros_like(rs_net_2d))).sum(dim=1)
        pfs_cuda = win_sums_cuda / loss_sums_cuda.clamp(min=1.0)

        tot_rs_cpu = tot_rs_cuda.cpu().numpy()
        tot_pts_cpu = tot_pts_cuda.cpu().numpy()
        max_dds_cpu = max_dds_cuda.cpu().numpy()
        calmar_cpu = calmar_cuda.cpu().numpy()
        daily_wrs_cpu = daily_wrs_cuda.cpu().numpy()
        green_days_cpu = green_days_cuda.cpu().numpy()
        red_days_cpu = red_days_cuda.cpu().numpy()
        active_days_cpu = active_days_cuda.cpu().numpy()
        month_wrs_cpu = month_wrs_cuda.cpu().numpy()
        pos_months_cpu = pos_months_cuda.cpu().numpy()
        trade_wrs_cpu = trade_wrs_cuda.cpu().numpy()
        pfs_cpu = pfs_cuda.cpu().numpy()

        for b_i, p in enumerate(chunk):
            results.append({
                "session": session_name,
                **p,
                "trades": M,
                "trade_win_rate": round(float(trade_wrs_cpu[b_i]), 2),
                "daily_win_rate": round(float(daily_wrs_cpu[b_i]), 1),
                "green_days": int(green_days_cpu[b_i]),
                "red_days": int(red_days_cpu[b_i]),
                "traded_days": int(active_days_cpu[b_i]),
                "net_points": round(float(tot_pts_cpu[b_i]), 2),
                "net_rs": round(float(tot_rs_cpu[b_i]), 2),
                "profit_factor": round(float(pfs_cpu[b_i]), 3),
                "max_drawdown": round(float(max_dds_cpu[b_i]), 2),
                "calmar_ratio": round(float(calmar_cpu[b_i]), 3),
                "month_win_rate": round(float(month_wrs_cpu[b_i]), 1),
                "pos_months": int(pos_months_cpu[b_i]),
                "tot_months": n_months,
            })

    return results


def main():
    # Build Wide-Parameter Grid (SL >= 10.0 & TP >= 10.0)
    initial_sls = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    lock_triggers = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0]
    locked_profits = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    trail_dists = [3.0, 4.0, 5.0, 6.0, 8.0]
    hard_tps = [15.0, 20.0, 25.0, 30.0, 40.0, 999.0]

    valid_wide_params = []
    for sl, l_trig, l_prof, trail, tp in itertools.product(initial_sls, lock_triggers, locked_profits, trail_dists, hard_tps):
        if l_prof >= l_trig:
            continue
        if l_prof > tp:
            continue
        valid_wide_params.append({
            "initial_sl": sl,
            "lock_trigger": l_trig,
            "locked_profit": l_prof,
            "trail_dist": trail,
            "hard_tp": tp,
        })

    print(f"Generated {len(valid_wide_params):,} Valid Wide-Parameter Sets (SL >= 10.0 & TP >= 10.0)\n", flush=True)

    sessions = [
        ("Full-Day (09:15-15:00)", 5, 345, 999, 999),
        ("Opening Bell + Afternoon (09:30-10:15 + 14:00-15:00)", 15, 345, 60, 285),
        ("Afternoon Power Session (14:00-15:30)", 285, 345, 999, 999),
        ("Golden Window (13:30-15:15)", 255, 360, 999, 999),
    ]

    all_results = []
    t0 = time.time()

    for s_name, s_start, s_end, m_start, m_end in sessions:
        t_s = time.time()
        print(f">>> Running Pure GPU Sweep: [{s_name}] ({len(valid_wide_params):,} configs)...", flush=True)
        res = run_wide_sweep(s_name, s_start, s_end, m_start, m_end, valid_wide_params, batch_size=25)
        all_results.extend(res)
        print(f"    Completed in {time.time()-t_s:.2f}s ({len(res):,} evals | {len(res)/(time.time()-t_s):.1f} configs/sec)", flush=True)

    total_time = time.time() - t0
    print("\n" + "=" * 145, flush=True)
    print(f"WIDE SWEEP COMPLETED in {total_time:.2f}s ({len(all_results):,} evaluations | {len(all_results)/total_time:.1f} configs/sec)", flush=True)
    print("=" * 145, flush=True)

    df = pd.DataFrame(all_results)

    # 1. Global Pareto Champion (High Profit + High Daily WR + Least DD)
    top_pareto_pool = df[(df["daily_win_rate"] >= 65.0) & (df["month_win_rate"] >= 90.0)]
    champ_pareto = (top_pareto_pool.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict()
                    if len(top_pareto_pool) > 0 else df.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict())

    # 2. Maximum Profit Champion
    champ_profit = df.sort_values(by="net_rs", ascending=False).iloc[0].to_dict()

    # 3. Maximum Daily Win Rate King
    champ_daily = df.sort_values(by=["daily_win_rate", "net_rs"], ascending=[False, False]).iloc[0].to_dict()

    print("\n--- TOP 3 WIDE-PARAMETER SWEET SPOT CHAMPIONS ---", flush=True)
    for title, c in [
        ("1. GLOBAL WIDE SWEET SPOT CHAMPION (Highest Calmar + Consistency)", champ_pareto),
        ("2. MAXIMUM NET PROFIT CHAMPION (Highest Net Rs)", champ_profit),
        ("3. DAILY INCOME CHAMPION (Highest Daily Win Rate % Green Days)", champ_daily),
    ]:
        print(f"\n{title}:", flush=True)
        print(f"  * Session Window:             {c['session']}", flush=True)
        print(f"  * Parameters:                 SL = -{c['initial_sl']} pt | Lock +{c['locked_profit']} pt @ +{c['lock_trigger']} pt Gain | Trail = {c['trail_dist']} pt | TP = +{c['hard_tp']} pt", flush=True)
        print(f"  * DAILY WIN RATE:             {c['daily_win_rate']}% GREEN DAYS ({c['green_days']:,} Green Days / {c['red_days']:,} Red Days out of {c['traded_days']:,} traded days)", flush=True)
        print(f"  * TRADE WIN RATE:             {c['trade_win_rate']}% ({c['trades']:,} trades)", flush=True)
        print(f"  * 7-Year Net Realized Profit: Rs {c['net_rs']:+,.2f} (+{c['net_points']:+,.2f} Net Points Captured)", flush=True)
        print(f"  * Profit Factor:              {c['profit_factor']:.3f}", flush=True)
        print(f"  * 7-Year Max Drawdown:        Rs {c['max_drawdown']:,.2f}", flush=True)
        print(f"  * Calmar Ratio (Return/DD):   {c['calmar_ratio']:.3f}", flush=True)
        print(f"  * Monthly Consistency:        {c['month_win_rate']:.1f}% ({c['pos_months']}/{c['tot_months']} Green Months)", flush=True)

    out_file = ROOT / "artifacts" / "f6_hybrid" / "wide_sl_tp_gpu_champions.json"
    out_file.write_text(json.dumps({
        "champ_pareto": champ_pareto,
        "champ_profit": champ_profit,
        "champ_daily": champ_daily,
        "top_50": df.sort_values(by="calmar_ratio", ascending=False).head(50).to_dict(orient="records"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Wide Parameter Champions JSON]: {out_file}", flush=True)


if __name__ == "__main__":
    main()
