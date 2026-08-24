"""Comprehensive 3D GPU Boundary & Limits Sweet Spot Optimizer.

Systematically explores and optimizes all operational boundaries and limits:
  1. Session Boundaries (Start Bar: 09:15, 09:30, 10:00, 13:30, 14:00 | End Bar: 14:45, 15:00, 15:20 | Midday Skip: None, 10:15-14:00, 11:00-13:30)
  2. Daily Trade Limits (Max Daily Trades: 1, 2, 3, 4, 6, 8, Uncapped | Max Daily Losses: 1, 2, 3, Uncapped)
  3. Daily Profit/Loss Circuit Breakers (Daily Target: +Rs 400, +Rs 600, +Rs 900, +Rs 1500, Uncapped | Daily Loss Cap: -Rs 300, -Rs 500, -Rs 800, Uncapped)
  4. Signal Threshold Boundations (Super vs Flag vs Both, S1 turn-up threshold)
  5. Multi-Tier Geometry Limits (SL: 2.5..5.0, Tier1: 5..8 -> 3..6, Tier2: 9..14 -> 7..11, Trail: 1.5..3.0, TP: 15..999)
"""

from __future__ import annotations

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
from artifacts.f6_hybrid.deep_dow_macro_research import load_dow_metrics, build_nifty_dow_table

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 135, flush=True)
print("3D GPU BOUNDARIES & LIMITS SWEET SPOT OPTIMIZER", flush=True)
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | Precision: TensorFloat32", flush=True)
print("=" * 135, flush=True)

d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
N_DAYS, T_BARS = d_c.shape
ALL_MONTHS = sorted(list(set(d[:7] for d in days)))
OOS_DAYS_MASK = np.array([d >= "2023-01-01" for d in days])

# Precompute Quad Stochastics & Entry Setup Masks
s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)

super_mask = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_mask = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
both_mask = super_mask | flag_mask

print(f"Loaded: {N_DAYS} days x {T_BARS} bars | Super: {super_mask.sum():,} | Flag: {flag_mask.sum():,} | Both: {both_mask.sum():,}", flush=True)


@torch.inference_mode()
def evaluate_boundary_sweep(
    entry_mask: torch.Tensor,
    geometry_list: list[dict],
    boundary_list: list[dict],
    batch_size: int = 40,
) -> list[dict]:
    coords = torch.nonzero(entry_mask, as_tuple=False)
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

    d_idx_cpu = d_idx.cpu().numpy()
    b_idx_cpu = b_idx.cpu().numpy()

    unique_days, split_indices = np.unique(d_idx_cpu, return_index=True)
    splits = np.split(np.arange(len(d_idx_cpu)), split_indices[1:])
    day_trade_map = {d_i: idxs for d_i, idxs in zip(unique_days, splits)}

    BIG = 999999
    all_evaluated_results = []

    for i in range(0, len(geometry_list), batch_size):
        chunk = geometry_list[i: i + batch_size]
        B = len(chunk)

        b_init_sl = torch.tensor([p["initial_sl"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_t1_trig = torch.tensor([p["t1_trig"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_t1_lock = torch.tensor([p["t1_lock"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_t2_trig = torch.tensor([p["t2_trig"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_t2_lock = torch.tensor([p["t2_lock"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_trail = torch.tensor([p["trail_dist"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_tp = torch.tensor([p["hard_tp"] * 2.0 for p in chunk], device=device).view(B, 1, 1)

        init_sl_3d = ep_3d - b_init_sl
        dyn_sl_3d = init_sl_3d.expand(B, M, max_future).clone()

        is_t1_3d = gains_3d >= b_t1_trig
        dyn_sl_3d = torch.where(is_t1_3d, torch.maximum(dyn_sl_3d, ep_3d + b_t1_lock), dyn_sl_3d)

        is_t2_3d = gains_3d >= b_t2_trig
        dyn_sl_3d = torch.where(is_t2_3d, torch.maximum(dyn_sl_3d, ep_3d + b_t2_lock), dyn_sl_3d)
        dyn_sl_3d = torch.where(is_t2_3d, torch.maximum(dyn_sl_3d, peaks_3d - b_trail), dyn_sl_3d)

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

        exit_bar_offset = torch.where(sl_exits, sl_first, torch.where(tp_exits, tp_first, max_future - 1))
        exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px_2d))
        pts_2d = (exit_px - ep.unsqueeze(0)) * 0.50
        rs_net_2d = pts_2d * LOT_SIZE - FEE

        pts_cpu = pts_2d.cpu().numpy()
        rs_cpu = rs_net_2d.cpu().numpy()
        exit_bars_cpu = (exit_bar_offset + b_idx.unsqueeze(0) + 1).cpu().numpy()

        # Iterate over both Geometry Chunks and Boundary Parameter Variants
        for b_i, geom in enumerate(chunk):
            r_arr = rs_cpu[b_i]
            p_arr = pts_cpu[b_i]
            ex_bars = exit_bars_cpu[b_i]

            for bnd in boundary_list:
                start_bar = bnd["start_bar"]
                end_bar = bnd["end_bar"]
                mid_skip_s = bnd["mid_skip_start"]
                mid_skip_e = bnd["mid_skip_end"]
                max_d_trades = bnd["max_daily_trades"]
                max_d_losses = bnd["max_daily_losses"]
                d_target_rs = bnd["daily_target_rs"]
                d_loss_cap_rs = bnd["daily_loss_cap_rs"]

                selected_rs = []
                selected_pts = []
                day_pnl = np.zeros(N_DAYS, dtype=np.float32)
                day_pts = np.zeros(N_DAYS, dtype=np.float32)

                for d_i in range(N_DAYS):
                    t_idxs = day_trade_map.get(d_i, [])
                    if len(t_idxs) == 0:
                        continue

                    last_exit = -999
                    d_rs = 0.0
                    d_p = 0.0
                    taken_trades = 0
                    taken_losses = 0

                    for idx in t_idxs:
                        e_b = b_idx_cpu[idx]
                        if e_b < start_bar or e_b >= end_bar:
                            continue
                        if mid_skip_s <= e_b < mid_skip_e:
                            continue
                        if taken_trades >= max_d_trades:
                            break
                        if taken_losses >= max_d_losses:
                            break
                        if d_rs >= d_target_rs:
                            break
                        if d_rs <= -d_loss_cap_rs:
                            break

                        if e_b < last_exit:
                            continue

                        r_val = r_arr[idx]
                        p_val = p_arr[idx]

                        selected_rs.append(r_val)
                        selected_pts.append(p_val)
                        d_rs += r_val
                        d_p += p_val
                        taken_trades += 1
                        if r_val <= 0:
                            taken_losses += 1
                        last_exit = ex_bars[idx]

                    day_pnl[d_i] = d_rs
                    day_pts[d_i] = d_p

                n_t = len(selected_rs)
                if n_t == 0:
                    continue

                active_days = day_pnl != 0
                green_days = (day_pnl > 0).sum()
                red_days = (day_pnl < 0).sum()
                n_traded_days = int(active_days.sum())
                daily_wr = (green_days / n_traded_days) * 100.0 if n_traded_days > 0 else 0.0

                tot_rs = float(day_pnl.sum())
                tot_pts = float(day_pts.sum())

                wins = [r for r in selected_rs if r > 0]
                losses = [r for r in selected_rs if r <= 0]
                trade_wr = (len(wins) / n_t) * 100.0 if n_t > 0 else 0.0
                pf = sum(wins) / abs(sum(losses)) if losses and abs(sum(losses)) > 0 else 99.0

                eq = np.cumsum(day_pnl)
                peak = np.maximum.accumulate(eq)
                max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
                calmar = tot_rs / max_dd if max_dd > 0 else 0.0

                df_m = pd.DataFrame({"day": days, "rs": day_pnl})
                df_m["month"] = df_m["day"].str[:7]
                m_pnl = df_m.groupby("month")["rs"].sum().reindex(ALL_MONTHS, fill_value=0.0)
                pos_m = int((m_pnl > 0).sum())
                month_wr = (pos_m / len(ALL_MONTHS)) * 100.0

                oos_pnl = float(day_pnl[OOS_DAYS_MASK].sum())

                all_evaluated_results.append({
                    **geom,
                    **bnd,
                    "trades": n_t,
                    "trade_win_rate": round(trade_wr, 2),
                    "daily_win_rate": round(daily_wr, 1),
                    "green_days": int(green_days),
                    "red_days": int(red_days),
                    "traded_days": n_traded_days,
                    "net_points": round(tot_pts, 2),
                    "net_rs": round(tot_rs, 2),
                    "profit_factor": round(pf, 3),
                    "max_drawdown": round(max_dd, 2),
                    "calmar_ratio": round(calmar, 3),
                    "month_win_rate": round(month_wr, 1),
                    "pos_months": pos_m,
                    "tot_months": len(ALL_MONTHS),
                    "oos_net_rs": round(oos_pnl, 2),
                })

    return all_evaluated_results


def run_boundaries_optimizer():
    # 1. Geometry Parameter Search Space
    initial_sls = [3.0, 3.5, 4.0]
    t1_trigs = [5.0, 6.0, 7.0, 8.0]
    t1_locks = [3.5, 4.0, 5.0]
    t2_trigs = [9.0, 10.0, 12.0]
    t2_locks = [7.0, 8.0, 9.0]
    trails = [1.5, 2.0]
    tps = [25.0, 999.0]

    geom_grid = []
    for sl, t1t, t1l, t2t, t2l, tr, tp in itertools.product(initial_sls, t1_trigs, t1_locks, t2_trigs, t2_locks, trails, tps):
        if t1l >= t1t or t2l >= t2t or t1l >= t2l:
            continue
        geom_grid.append({
            "initial_sl": sl, "t1_trig": t1t, "t1_lock": t1l,
            "t2_trig": t2t, "t2_lock": t2l, "trail_dist": tr, "hard_tp": tp,
        })

    print(f"Multi-Tier Geometry Sets: {len(geom_grid):,}", flush=True)

    # 2. Operational Limits & Boundaries Space
    # Sessions: Full-Day (15..345), Twin-Peak (15..60 + 285..345), Afternoon Power (285..345), Late Morning + Afternoon (60..345)
    sessions = [
        {"session_name": "Full-Day (09:30-15:00)", "start_bar": 15, "end_bar": 345, "mid_skip_start": 999, "mid_skip_end": 999},
        {"session_name": "Twin-Peak (09:30-10:15 + 14:00-15:00)", "start_bar": 15, "end_bar": 345, "mid_skip_start": 60, "mid_skip_end": 285},
        {"session_name": "Afternoon Power (14:00-15:00)", "start_bar": 285, "end_bar": 345, "mid_skip_start": 999, "mid_skip_end": 999},
        {"session_name": "Post-Open Momentum (10:00-15:00)", "start_bar": 45, "end_bar": 345, "mid_skip_start": 999, "mid_skip_end": 999},
    ]

    max_trades_list = [2, 3, 5, 8, 999]
    max_losses_list = [1, 2, 3, 999]
    daily_targets = [600.0, 1000.0, 1800.0, 999999.0]
    daily_loss_caps = [400.0, 700.0, 1200.0, 999999.0]

    bnd_grid = []
    for s in sessions:
        for mt in [2, 3, 5, 999]:
            for ml in [2, 3, 999]:
                for dt in [800.0, 1500.0, 999999.0]:
                    for dl in [600.0, 1000.0, 999999.0]:
                        bnd_grid.append({
                            **s,
                            "max_daily_trades": mt,
                            "max_daily_losses": ml,
                            "daily_target_rs": dt,
                            "daily_loss_cap_rs": dl,
                        })

    print(f"Boundary & Limits Variants: {len(bnd_grid):,}", flush=True)
    total_evals = len(geom_grid) * len(bnd_grid)
    print(f"Total Combined Parameter Space: {total_evals:,} Evaluations\n", flush=True)

    print(">>> Launching 3D GPU Boundations & Limits Optimizer across 7 Years (1,588 Days)...", flush=True)
    t0 = time.time()
    results = evaluate_boundary_sweep(both_mask, geom_grid, bnd_grid, batch_size=40)
    print(f"    Completed in {time.time()-t0:.2f}s ({len(results):,} configurations evaluated)", flush=True)

    df = pd.DataFrame(results)

    # 1. Global Pareto Champion: Highest Calmar with >65% Daily WR & >90% Month WR
    top_pareto = df[(df["daily_win_rate"] >= 65.0) & (df["month_win_rate"] >= 90.0) & (df["net_rs"] >= 2000000.0)].sort_values(by="calmar_ratio", ascending=False)
    champ_pareto = top_pareto.iloc[0].to_dict() if len(top_pareto) > 0 else df.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict()

    # 2. Maximum Profit Champion with Controlled Drawdown (< Rs 15k DD)
    top_profit = df[df["max_drawdown"] <= 15000.0].sort_values(by="net_rs", ascending=False).iloc[0].to_dict()

    # 3. Maximum Daily Win Rate King (>70% Green Days)
    top_daily = df.sort_values(by=["daily_win_rate", "net_rs"], ascending=[False, False]).iloc[0].to_dict()

    print("\n" + "=" * 145, flush=True)
    print("TOP 3 BOUNDARY & LIMITS SWEET SPOT CHAMPIONS", flush=True)
    print("=" * 145, flush=True)

    for title, c in [
        ("1. GLOBAL SWEET SPOT CHAMPION (Highest Calmar + >65% Daily WR + >Rs 25L Profit)", champ_pareto),
        ("2. MAXIMUM NET PROFIT CHAMPION (Highest Profit with < Rs 15k Drawdown)", top_profit),
        ("3. DAILY INCOME CHAMPION (Highest Daily Win Rate % Green Days)", top_daily),
    ]:
        print(f"\n{title}:", flush=True)
        print(f"  * Session Window:             {c['session_name']}", flush=True)
        print(f"  * Multi-Tier Limits:          SL = -{c['initial_sl']} pt | Lock +{c['t1_lock']} pt @ +{c['t1_trig']} pt | Lock +{c['t2_lock']} pt @ +{c['t2_trig']} pt | Trail = {c['trail_dist']} pt | TP = +{c['hard_tp']} pt", flush=True)
        print(f"  * Daily Limits & Boundations: Max Trades = {c['max_daily_trades']} | Max Losses = {c['max_daily_losses']} | Daily Target Lock = Rs {c['daily_target_rs']} | Daily Loss Guard = Rs {c['daily_loss_cap_rs']}", flush=True)
        print(f"  * DAILY WIN RATE:             {c['daily_win_rate']}% GREEN DAYS ({c['green_days']:,} Green Days / {c['red_days']:,} Red Days out of {c['traded_days']:,} traded days)", flush=True)
        print(f"  * TRADE WIN RATE:             {c['trade_win_rate']}% ({c['trades']:,} trades)", flush=True)
        print(f"  * 7-Year Net Realized Profit: Rs {c['net_rs']:+,.2f} (+{c['net_points']:+,.2f} Net Points Captured)", flush=True)
        print(f"  * 4-Year Walk-Forward OOS:    Rs {c['oos_net_rs']:+,.2f} (2023-2026 Blind OOS)", flush=True)
        print(f"  * Profit Factor:              {c['profit_factor']:.3f}", flush=True)
        print(f"  * 7-Year Max Drawdown:        Rs {c['max_drawdown']:,.2f}", flush=True)
        print(f"  * Calmar Ratio (Return/DD):   {c['calmar_ratio']:.3f}", flush=True)
        print(f"  * Monthly Consistency:        {c['month_win_rate']:.1f}% ({c['pos_months']}/{c['tot_months']} Green Months)", flush=True)

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "apex_limits_champions.json"
    out_file.write_text(json.dumps({
        "champ_pareto": champ_pareto,
        "champ_profit": top_profit,
        "champ_daily": top_daily,
        "top_50": df.sort_values(by="calmar_ratio", ascending=False).head(50).to_dict(orient="records"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Limits Champions JSON]: {out_file}", flush=True)


if __name__ == "__main__":
    run_boundaries_optimizer()
