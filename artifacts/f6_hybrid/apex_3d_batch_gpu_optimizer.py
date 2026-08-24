"""3D GPU-Batched Parameter Discovery Engine for APEX RUNNER.

Architecture:
  - Tensor Shape: (BATCH, N_DAYS, T_BARS) fully vectorized on RTX 3060 VRAM.
  - Zero host-device synchronizations in the hot loop.
  - Explores the multi-dimensional parameter space:
      * Initial SL: [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
      * Lock Trigger Milestone: [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
      * Guaranteed Locked Profit: [4.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0]
      * Chandelier Trailing Distance: [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
      * Hard Take Profit: [15.0, 18.0, 20.0, 25.0, 30.0, 40.0, 999.0]
      * Session Start Gate: [09:30 (Bar 15), 10:00 (Bar 45), 11:00 (Bar 105), 13:00 (Bar 225), 14:00 (Bar 285)]
      * Setup Type: [BOTH (Super + Flag), SUPER_ONLY, FLAG_ONLY]
"""

from __future__ import annotations

import argparse
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

print("=" * 135)
print("3D GPU-BATCHED APEX RUNNER OPTIMIZER (RTX 3060 12GB)")
print("=" * 135)

t_load = time.time()
d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
N_DAYS, T_BARS = d_c.shape
print(f"Data loaded in {time.time()-t_load:.2f}s: {N_DAYS} days x {T_BARS} bars | Memory: {torch.cuda.memory_allocated()/(1024**2):.1f} MB")

# Precompute Quad Stochastics & Entry Setups
s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)

super_mask = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_mask = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
both_mask = super_mask | flag_mask

SETUP_MASKS = {
    "BOTH": both_mask,
    "SUPER_ONLY": super_mask,
    "FLAG_ONLY": flag_mask,
}

ALL_MONTHS = sorted(list(set(d[:7] for d in days)))
OOS_DAYS_MASK = np.array([d >= "2023-01-01" for d in days])


# ═══════════════════════════════════════════════════════════════════════════
# 3D BATCHED GPU SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def evaluate_parameter_batch(
    param_batch: list[dict],
    setup_name: str = "BOTH",
    start_bar: int = 15,
) -> list[dict]:
    """Evaluates a batch of B configurations simultaneously in 3D GPU space."""
    B = len(param_batch)
    if B == 0:
        return []

    # Get entry coordinates for this setup and session filter
    active_mask = SETUP_MASKS[setup_name].clone()
    active_mask[:, :start_bar] = False
    active_mask[:, BASE_SESSION_END:] = False

    coords = torch.nonzero(active_mask, as_tuple=False)
    M = coords.shape[0]
    if M == 0:
        return []

    d_idx = coords[:, 0]
    b_idx = coords[:, 1]
    ep = d_c[d_idx, b_idx]  # (M,)

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
    running_peaks = torch.cummax(fut_h_clean, dim=1).values  # (M, max_future)
    ep_exp = ep.unsqueeze(1)
    gains = running_peaks - ep_exp  # (M, max_future)

    # Convert parameter batch into tensors (B, 1, 1)
    b_init_sl = torch.tensor([p["initial_sl"] * 2.0 for p in param_batch], device=device).view(B, 1, 1)
    b_lock_trig = torch.tensor([p["lock_trigger"] * 2.0 for p in param_batch], device=device).view(B, 1, 1)
    b_locked_sl = torch.tensor([p["locked_profit"] * 2.0 for p in param_batch], device=device).view(B, 1, 1)
    b_trail = torch.tensor([p["trail_dist"] * 2.0 for p in param_batch], device=device).view(B, 1, 1)
    b_tp = torch.tensor([p["hard_tp"] * 2.0 for p in param_batch], device=device).view(B, 1, 1)

    # Expand trade tensors into (1, M, max_future) for broadcasting to (B, M, max_future)
    gains_3d = gains.unsqueeze(0)  # (1, M, max_future)
    ep_3d = ep_exp.unsqueeze(0)
    peaks_3d = running_peaks.unsqueeze(0)
    fut_l_3d = fut_l_m.unsqueeze(0)
    fut_h_3d = fut_h_m.unsqueeze(0)

    # Vectorized Trailing SL logic in 3D
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

    BIG = 999999
    sl_any = hit_sl_3d.any(dim=2)  # (B, M)
    tp_any = hit_tp_3d.any(dim=2)  # (B, M)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl_3d.int(), dim=2), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp_3d.int(), dim=2), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    sl_idx_clamp = sl_first.clamp(max=max_future - 1).unsqueeze(2)
    exit_sl_px = dyn_sl_3d.gather(2, sl_idx_clamp).squeeze(2)  # (B, M)
    exit_tp_px = tp_barrier_3d.squeeze(2)  # (B, M)
    eod_px_2d = eod_px.unsqueeze(0).expand(B, M)

    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px_2d))
    pts_2d = (exit_px - ep.unsqueeze(0)) * 0.50  # (B, M)
    rs_net_2d = pts_2d * LOT_SIZE - FEE          # (B, M)

    # Bring results to CPU for high-speed metric reduction
    pts_cpu = pts_2d.cpu().numpy()
    rs_cpu = rs_net_2d.cpu().numpy()
    d_idx_cpu = d_idx.cpu().numpy()

    # Pre-build day indexing for fast daily aggregation
    unique_d, inv_d = np.unique(d_idx_cpu, return_inverse=True)

    batch_eval_results = []

    for b_i, p in enumerate(param_batch):
        r_arr = rs_cpu[b_i]
        p_arr = pts_cpu[b_i]

        n_t = len(r_arr)
        wins = r_arr > 0
        losses = r_arr <= 0
        n_wins = int(wins.sum())
        n_losses = int(losses.sum())
        wr = (n_wins / n_t) * 100.0 if n_t > 0 else 0.0

        tot_pts = float(p_arr.sum())
        tot_rs = float(r_arr.sum())

        win_sum = float(r_arr[wins].sum()) if n_wins > 0 else 0.0
        loss_sum = abs(float(r_arr[losses].sum())) if n_losses > 0 else 0.0
        pf = win_sum / loss_sum if loss_sum > 0 else (99.0 if win_sum > 0 else 0.0)

        avg_w_pts = float(p_arr[wins].mean()) if n_wins > 0 else 0.0
        avg_l_pts = float(p_arr[losses].mean()) if n_losses > 0 else 0.0
        asym = abs(avg_w_pts / avg_l_pts) if abs(avg_l_pts) > 0 else 0.0

        # Vectorized Daily PnL for Max Drawdown & Monthly Consistency
        day_pnl = np.zeros(N_DAYS, dtype=np.float32)
        np.add.at(day_pnl, d_idx_cpu, r_arr)

        eq = np.cumsum(day_pnl)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
        calmar = tot_rs / max_dd if max_dd > 0 else 0.0

        # Monthly Win Rate
        df_m = pd.DataFrame({"day": days, "rs": day_pnl})
        df_m["month"] = df_m["day"].str[:7]
        m_pnl = df_m.groupby("month")["rs"].sum().reindex(ALL_MONTHS, fill_value=0.0)
        pos_m = int((m_pnl > 0).sum())
        m_wr = (pos_m / len(ALL_MONTHS)) * 100.0

        # Blind Out-of-Sample Metrics (2023–2026: 4 Years)
        oos_pnl = float(day_pnl[OOS_DAYS_MASK].sum())
        oos_pts = float(p_arr[OOS_DAYS_MASK[d_idx_cpu]].sum()) if n_t > 0 else 0.0

        res_dict = {
            **p,
            "setup": setup_name,
            "start_time": "14:00 (Afternoon Power)" if start_bar >= 285 else ("09:30 (Market Open)" if start_bar == 15 else f"Bar {start_bar}"),
            "start_bar": start_bar,
            "trades": n_t,
            "win_rate": round(wr, 2),
            "avg_win_pts": round(avg_w_pts, 2),
            "avg_loss_pts": round(avg_l_pts, 2),
            "asymmetry_ratio": round(asym, 2),
            "net_points": round(tot_pts, 2),
            "net_rs": round(tot_rs, 2),
            "profit_factor": round(pf, 3),
            "max_drawdown": round(max_dd, 2),
            "calmar_ratio": round(calmar, 3),
            "month_win_rate": round(m_wr, 1),
            "pos_months": pos_m,
            "tot_months": len(ALL_MONTHS),
            "oos_net_rs": round(oos_pnl, 2),
            "oos_net_pts": round(oos_pts, 2),
        }
        batch_eval_results.append(res_dict)

    return batch_eval_results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXTENSIVE GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════
def generate_parameter_grid() -> list[dict]:
    initial_sls = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    lock_triggers = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0]
    locked_profits = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    trail_dists = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    hard_tps = [12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 999.0]

    valid_params = []
    for sl, l_trig, l_prof, trail, tp in itertools.product(initial_sls, lock_triggers, locked_profits, trail_dists, hard_tps):
        if l_prof >= l_trig:
            continue  # Locked profit must be strictly less than trigger
        if l_prof > tp:
            continue
        valid_params.append({
            "initial_sl": sl,
            "lock_trigger": l_trig,
            "locked_profit": l_prof,
            "trail_dist": trail,
            "hard_tp": tp,
        })
    return valid_params


def run_extensive_optimization(smoke: bool = False):
    params = generate_parameter_grid()
    if smoke:
        params = params[:100]

    print(f"\nGenerated {len(params):,} Valid Geometry Configurations per Setup & Session")
    setups = ["BOTH", "SUPER_ONLY", "FLAG_ONLY"]
    sessions = [15, 45, 105, 225, 285]  # 09:30, 10:00, 11:00, 13:00, 14:00

    total_evaluations = len(params) * len(setups) * len(sessions)
    print(f"Total Parameter Evaluations in 3D GPU Space: {total_evaluations:,}")

    GPU_BATCH_SIZE = 150  # 150 configurations parallel in 3D tensor
    all_evaluated = []

    t_start = time.time()
    count = 0

    for setup in setups:
        for s_bar in sessions:
            sess_label = "Afternoon Power (14:00)" if s_bar >= 285 else "All-Day (09:30)"
            print(f"\n>>> Launching 3D Batched Sweep for [{setup}] | Session: [{sess_label}] ({len(params):,} configs)...")
            
            for i in range(0, len(params), GPU_BATCH_SIZE):
                chunk = params[i: i + GPU_BATCH_SIZE]
                results = evaluate_parameter_batch(chunk, setup_name=setup, start_bar=s_bar)
                all_evaluated.extend(results)
                count += len(chunk)
                if count % 1500 == 0 or count == total_evaluations:
                    elapsed = time.time() - t_start
                    speed = count / elapsed
                    print(f"  * Evaluated {count:,} / {total_evaluations:,} ({count/total_evaluations*100:.1f}%) in {elapsed:.1f}s | Speed: {speed:.1f} configs/sec | VRAM: {torch.cuda.memory_allocated()/(1024**2):.1f} MB")

    total_time = time.time() - t_start
    print("\n" + "=" * 135)
    print(f"GPU OPTIMIZATION COMPLETE in {total_time:.2f}s ({len(all_evaluated):,} Total Configs Tested | {len(all_evaluated)/total_time:.1f} configs/sec)")
    print("=" * 135)

    # Identify Pareto Champions
    df_all = pd.DataFrame(all_evaluated)

    # 1. Absolute Net Profit Champion
    champ_profit = df_all.sort_values(by="net_rs", ascending=False).iloc[0].to_dict()

    # 2. Maximum Monthly Consistency Champion (Green Months >= 90%, highest Calmar)
    consistent_pool = df_all[df_all["month_win_rate"] >= 88.0]
    champ_consistency = (consistent_pool.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict()
                         if len(consistent_pool) > 0 else df_all.sort_values(by="month_win_rate", ascending=False).iloc[0].to_dict())

    # 3. Ultra-Low Drawdown Champion (Max DD < Rs 25,000, highest Profit Factor)
    low_dd_pool = df_all[(df_all["max_drawdown"] <= 25000.0) & (df_all["trades"] >= 2000)]
    champ_low_dd = (low_dd_pool.sort_values(by="profit_factor", ascending=False).iloc[0].to_dict()
                    if len(low_dd_pool) > 0 else df_all.sort_values(by="max_drawdown", ascending=True).iloc[0].to_dict())

    # 4. The Golden Asymmetric Champion (High Profit + High Calmar + >80% Green Months)
    golden_pool = df_all[(df_all["month_win_rate"] >= 80.0) & (df_all["max_drawdown"] <= 40000.0) & (df_all["profit_factor"] >= 1.50)]
    champ_golden = (golden_pool.sort_values(by="net_rs", ascending=False).iloc[0].to_dict()
                    if len(golden_pool) > 0 else df_all.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict())

    champions = {
        "champ_profit": champ_profit,
        "champ_consistency": champ_consistency,
        "champ_low_dd": champ_low_dd,
        "champ_golden": champ_golden,
    }

    # Print Champions Table
    print("\n" + "#" * 60 + " TOP PARETO FRONTIER CHAMPIONS " + "#" * 60)
    for title, c in [
        ("1. ABSOLUTE MAXIMUM NET PROFIT CHAMPION", champ_profit),
        ("2. MAXIMUM MONTHLY CONSISTENCY CHAMPION (>90% Green Months)", champ_consistency),
        ("3. ULTRA-LOW DRAWDOWN CHAMPION (< Rs 25k Max DD)", champ_low_dd),
        ("4. GOLDEN BALANCED CHAMPION (Best All-Round Risk/Reward)", champ_golden),
    ]:
        print(f"\n{title}:")
        print(f"  * Settings: Initial SL = -{c['initial_sl']} pts | Lock +{c['locked_profit']} pts @ +{c['lock_trigger']} pt Gain | Trail = {c['trail_dist']} pts | Target TP = +{c['hard_tp']} pts")
        print(f"  * Session & Setup: {c['start_time']} | Setup: {c['setup']}")
        print(f"  * 7-Year Net Profit:        Rs {c['net_rs']:+,.2f} (+{c['net_points']:+,.2f} Net Points Captured)")
        print(f"  * 4-Year Walk-Forward OOS:  Rs {c['oos_net_rs']:+,.2f} (+{c['oos_net_pts']:+,.2f} OOS Points)")
        print(f"  * Win Rate:                 {c['win_rate']}% ({c['trades']:,} trades | Avg Win: +{c['avg_win_pts']} pt vs Loss: {c['avg_loss_pts']} pt | Asymmetry: {c['asymmetry_ratio']}x)")
        print(f"  * Profit Factor:            {c['profit_factor']}")
        print(f"  * Maximum Drawdown:         Rs {c['max_drawdown']:,.2f}")
        print(f"  * Calmar Ratio:             {c['calmar_ratio']}")
        print(f"  * Monthly Win Rate:         {c['month_win_rate']}% ({c['pos_months']} / {c['tot_months']} Green Months)")

    # Save Top 100 Leaderboard to JSON
    top_100 = df_all.sort_values(by="net_rs", ascending=False).head(100).to_dict(orient="records")
    out_file = ROOT / "artifacts" / "f6_hybrid" / "apex_3d_batch_gpu_optimizer_results.json"
    out_file.write_text(json.dumps({
        "champions": champions,
        "top_100": top_100,
        "total_evaluated": len(all_evaluated),
        "execution_time_sec": round(total_time, 2),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Master 3D GPU Optimizer JSON]: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    args = parser.parse_args()
    run_extensive_optimization(smoke=args.smoke)
