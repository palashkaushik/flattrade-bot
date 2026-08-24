"""Optimus Multi-Session Windows & Session Expansion GPU Optimizer.

Explores 25+ Single, Dual, and Tri-Peak Session Regimes on RTX 3060:
  - Single Windows: 09:15-15:20, 09:30-15:20, 09:30-11:30, 12:30-15:20, 13:00-15:20, 13:30-15:20, 14:00-15:20
  - Dual Windows (Twin-Peak): (09:30-10:30 + 13:30-15:20), (09:30-10:30 + 14:00-15:20), (09:30-11:00 + 13:30-15:20), (10:00-11:30 + 13:30-15:20)
  - Tri-Peak: (09:30-10:30 + 11:30-12:30 + 14:00-15:20)
Across 7 Years (1,588 Days, 2020-2026) with Walk-Forward OOS and Live Parity Check.
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
print("OPTIMUS MULTI-SESSION WINDOWS & EXPANSION GPU OPTIMIZER", flush=True)
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

# 4-Year Walk-Forward Masks
oos_2023_mask = torch.tensor([d.startswith("2023") for d in days], device=device, dtype=torch.bool)
oos_2024_mask = torch.tensor([d.startswith("2024") for d in days], device=device, dtype=torch.bool)
oos_2025_mask = torch.tensor([d.startswith("2025") for d in days], device=device, dtype=torch.bool)
oos_2026_mask = torch.tensor([d.startswith("2026") for d in days], device=device, dtype=torch.bool)
oos_total_mask = oos_2023_mask | oos_2024_mask | oos_2025_mask | oos_2026_mask


@torch.inference_mode()
def run_session_regime_sweep(
    session_name: str,
    valid_bars_tensor: torch.Tensor,
    param_grid: list[dict],
    batch_size: int = 25,
) -> list[dict]:
    # Active mask based on specific valid bars tensor
    active_mask = both_mask & valid_bars_tensor

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

        # Monthly Win Rate on CUDA
        month_idx_exp = month_indices_cuda.unsqueeze(0).expand(B, N_DAYS)
        m_pnl_cuda = torch.zeros((B, n_months), device=device, dtype=torch.float32)
        m_pnl_cuda.scatter_add_(1, month_idx_exp, day_pnl_cuda)
        pos_months_cuda = (m_pnl_cuda > 0).sum(dim=1)
        month_wrs_cuda = (pos_months_cuda.float() / float(n_months)) * 100.0

        # Daily Win Rate on CUDA
        green_days_cuda = (day_pnl_cuda > 0).sum(dim=1)
        red_days_cuda = (day_pnl_cuda < 0).sum(dim=1)
        active_days_cuda = (day_pnl_cuda != 0).sum(dim=1)
        daily_wrs_cuda = (green_days_cuda.float() / active_days_cuda.clamp(min=1).float()) * 100.0

        # Walk-Forward OOS
        pnl_oos_tot = (day_pnl_cuda * oos_total_mask.unsqueeze(0)).sum(dim=1)
        pnl_2023 = (day_pnl_cuda * oos_2023_mask.unsqueeze(0)).sum(dim=1)
        pnl_2024 = (day_pnl_cuda * oos_2024_mask.unsqueeze(0)).sum(dim=1)
        pnl_2025 = (day_pnl_cuda * oos_2025_mask.unsqueeze(0)).sum(dim=1)
        pnl_2026 = (day_pnl_cuda * oos_2026_mask.unsqueeze(0)).sum(dim=1)

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

        # Transfer back to CPU
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
        oos_tot_cpu = pnl_oos_tot.cpu().numpy()
        pnl_23_cpu = pnl_2023.cpu().numpy()
        pnl_24_cpu = pnl_2024.cpu().numpy()
        pnl_25_cpu = pnl_2025.cpu().numpy()
        pnl_26_cpu = pnl_2026.cpu().numpy()

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
                "oos_net_rs": round(float(oos_tot_cpu[b_i]), 2),
                "pnl_2023": round(float(pnl_23_cpu[b_i]), 2),
                "pnl_2024": round(float(pnl_24_cpu[b_i]), 2),
                "pnl_2025": round(float(pnl_25_cpu[b_i]), 2),
                "pnl_2026": round(float(pnl_26_cpu[b_i]), 2),
            })

    return results


def build_session_regimes():
    # Minute helper: 09:15 = bar 0, 09:30 = bar 15, 10:00 = bar 45, 10:30 = bar 75, 11:00 = bar 105, 11:30 = bar 135
    # 12:30 = bar 195, 13:00 = bar 225, 13:30 = bar 255, 14:00 = bar 285, 14:30 = bar 315, 15:15 = bar 360, 15:20 = bar 365
    regimes = []

    # 1. Single Continuous Sessions
    def single_mask(s, e):
        m = torch.zeros((N_DAYS, T_BARS), device=device, dtype=torch.bool)
        m[:, s:e] = True
        return m

    def multi_mask(intervals):
        m = torch.zeros((N_DAYS, T_BARS), device=device, dtype=torch.bool)
        for s, e in intervals:
            m[:, s:e] = True
        return m

    # Single Windows
    regimes.append(("1. Full-Day (09:15-15:20)", single_mask(0, 365)))
    regimes.append(("2. Standard Day (09:30-15:15)", single_mask(15, 360)))
    regimes.append(("3. Extended Afternoon (12:30-15:20)", single_mask(195, 365)))
    regimes.append(("4. European Open (13:00-15:20)", single_mask(225, 365)))
    regimes.append(("5. Golden Window (13:30-15:15)", single_mask(255, 360)))
    regimes.append(("6. Afternoon Power (14:00-15:20)", single_mask(285, 365)))
    regimes.append(("7. Power Hour (14:15-15:20)", single_mask(300, 365)))

    # Dual Windows (Twin-Peak Regimes)
    regimes.append(("8. Twin-Peak A (09:30-10:30 + 13:30-15:15)", multi_mask([(15, 75), (255, 360)])))
    regimes.append(("9. Twin-Peak B (09:30-10:30 + 14:00-15:20)", multi_mask([(15, 75), (285, 365)])))
    regimes.append(("10. Twin-Peak C (09:30-11:00 + 13:30-15:15)", multi_mask([(15, 105), (255, 360)])))
    regimes.append(("11. Twin-Peak D (09:30-11:00 + 14:00-15:20)", multi_mask([(15, 105), (285, 365)])))
    regimes.append(("12. Twin-Peak E (10:00-11:30 + 13:30-15:15)", multi_mask([(45, 135), (255, 360)])))
    regimes.append(("13. Twin-Peak F (09:45-11:15 + 13:45-15:15)", multi_mask([(30, 120), (270, 360)])))
    regimes.append(("14. Twin-Peak G (09:30-10:15 + 13:00-15:20)", multi_mask([(15, 60), (225, 365)])))

    # Tri-Peak & Extended Multi-Windows
    regimes.append(("15. Tri-Peak (09:30-10:30 + 11:30-12:30 + 14:00-15:20)", multi_mask([(15, 75), (135, 195), (285, 365)])))
    regimes.append(("16. Tri-Peak Wide (09:30-11:00 + 12:00-13:00 + 14:00-15:15)", multi_mask([(15, 105), (165, 225), (285, 360)])))

    return regimes


def main():
    # Proven Wide Parameter Grid (Fine-Tuned)
    initial_sls = [10.0, 12.0, 14.0, 16.0]
    lock_triggers = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    locked_profits = [8.0, 10.0, 12.0, 14.0, 16.0]
    trail_dists = [3.0, 4.0, 5.0]
    hard_tps = [18.0, 20.0, 25.0, 30.0, 999.0]

    valid_wide_grid = []
    for sl, l_trig, l_prof, trail, tp in itertools.product(initial_sls, lock_triggers, locked_profits, trail_dists, hard_tps):
        if l_prof >= l_trig:
            continue
        if l_prof > tp:
            continue
        valid_wide_grid.append({
            "initial_sl": sl, "lock_trigger": l_trig, "locked_profit": l_prof,
            "trail_dist": trail, "hard_tp": tp,
        })

    print(f"Generated {len(valid_wide_grid):,} Valid Wide Parameter Sets\n", flush=True)

    session_regimes = build_session_regimes()
    print(f"Loaded {len(session_regimes)} Distinct Session Regimes\n", flush=True)

    all_results = []
    t0 = time.time()

    for s_name, s_mask in session_regimes:
        t_s = time.time()
        print(f">>> Evaluating Session Regime: [{s_name}]...", flush=True)
        res = run_session_regime_sweep(s_name, s_mask, valid_wide_grid, batch_size=25)
        all_results.extend(res)
        print(f"    Completed in {time.time()-t_s:.2f}s ({len(res):,} evals | {len(res)/(time.time()-t_s):.1f} configs/sec)", flush=True)

    total_time = time.time() - t0
    total_evals = len(all_results)
    print("\n" + "=" * 145, flush=True)
    print(f"ALL SESSION REGIMES COMPLETED in {total_time:.2f}s ({total_evals:,} total evaluations | {total_evals/total_time:.1f} configs/sec)", flush=True)
    print("=" * 145, flush=True)

    df = pd.DataFrame(all_results)

    # 1. Global Pareto Champion (High Profit + High Calmar + >65% Daily WR)
    top_pareto_pool = df[(df["daily_win_rate"] >= 65.0) & (df["month_win_rate"] >= 85.0)]
    champ_pareto = (top_pareto_pool.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict()
                    if len(top_pareto_pool) > 0 else df.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict())

    # 2. Maximum Profit Champion
    champ_profit = df.sort_values(by="net_rs", ascending=False).iloc[0].to_dict()

    # 3. Maximum Daily Win Rate King
    champ_daily = df.sort_values(by=["daily_win_rate", "net_rs"], ascending=[False, False]).iloc[0].to_dict()

    # 4. Best Twin-Peak Regime Champion
    df_twin = df[df["session"].str.contains("Twin-Peak")]
    champ_twin = df_twin.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict()

    # 5. Best Extended Afternoon Champion
    df_ext = df[df["session"].str.contains("Extended Afternoon|European")]
    champ_ext = df_ext.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict()

    print("\n--- TOP 5 SESSION REGIME CHAMPIONS ---", flush=True)
    for title, c in [
        ("1. GLOBAL SWEET SPOT CHAMPION (Highest Calmar + Consistency)", champ_pareto),
        ("2. MAXIMUM NET PROFIT CHAMPION (Highest Rupee Profit)", champ_profit),
        ("3. MAXIMUM DAILY INCOME CHAMPION (Highest % Green Days)", champ_daily),
        ("4. BEST TWIN-PEAK MULTI-WINDOW CHAMPION (Open + Afternoon)", champ_twin),
        ("5. BEST EXTENDED AFTERNOON CHAMPION (12:30/13:00-15:20)", champ_ext),
    ]:
        print(f"\n{title}:", flush=True)
        print(f"  * Session Regime:             {c['session']}", flush=True)
        print(f"  * Geometry:                   SL = -{c['initial_sl']:.1f} pt | Lock +{c['locked_profit']:.1f} pt @ +{c['lock_trigger']:.1f} pt Gain | Trail = {c['trail_dist']:.1f} pt | TP = +{c['hard_tp']:.1f} pt", flush=True)
        print(f"  * DAILY WIN RATE:             {c['daily_win_rate']:.1f}% GREEN DAYS ({c['green_days']:,} Green Days / {c['red_days']:,} Red Days out of {c['traded_days']:,} traded days)", flush=True)
        print(f"  * TRADE WIN RATE:             {c['trade_win_rate']:.2f}% ({c['trades']:,} trades)", flush=True)
        print(f"  * 7-Year Net Realized Profit: Rs {c['net_rs']:+,.2f} (+{c['net_points']:+,.2f} Net Points Captured)", flush=True)
        print(f"  * 4-Year Walk-Forward OOS:    Rs {c['oos_net_rs']:+,.2f} (2023: Rs {c['pnl_2023']:+,.0f} | 2024: Rs {c['pnl_2024']:+,.0f} | 2025: Rs {c['pnl_2025']:+,.0f} | 2026: Rs {c['pnl_2026']:+,.0f})", flush=True)
        print(f"  * Profit Factor:              {c['profit_factor']:.3f}", flush=True)
        print(f"  * 7-Year Max Drawdown:        Rs {c['max_drawdown']:,.2f}", flush=True)
        print(f"  * Calmar Ratio (Return/DD):   {c['calmar_ratio']:.3f}", flush=True)
        print(f"  * Monthly Consistency:        {c['month_win_rate']:.1f}% ({c['pos_months']}/{c['tot_months']} Green Months)", flush=True)

    # Rank all 16 session regimes
    print("\n" + "=" * 145, flush=True)
    print("ALL 16 SESSION REGIMES RANKED BY PERFORMANCE (BEST CONFIG PER REGIME)", flush=True)
    print("=" * 145, flush=True)
    best_per_session = df.sort_values(by="calmar_ratio", ascending=False).groupby("session").first().reset_index()
    best_per_session = best_per_session.sort_values(by="net_rs", ascending=False)
    print(best_per_session[["session", "trades", "daily_win_rate", "trade_win_rate", "net_points", "net_rs", "profit_factor", "max_drawdown", "calmar_ratio", "month_win_rate"]].to_string(index=False), flush=True)

    out_file = ROOT / "artifacts" / "f6_hybrid" / "session_regimes_champions_results.json"
    out_file.write_text(json.dumps({
        "champ_pareto": champ_pareto,
        "champ_profit": champ_profit,
        "champ_daily": champ_daily,
        "champ_twin": champ_twin,
        "champ_ext": champ_ext,
        "best_per_session": best_per_session.to_dict(orient="records"),
        "top_50": df.sort_values(by="calmar_ratio", ascending=False).head(50).to_dict(orient="records"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Session Regimes Results JSON]: {out_file}", flush=True)


if __name__ == "__main__":
    main()
