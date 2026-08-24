"""Theta-Beating Tiered Ratchet & Dynamic Trailing Stop Optimizer on GPU.

Implements the 4-Tier Algorithmic Exit Framework:
  Tier 1: Wide Initial Protective SL (Breathing room against option noise)
  Tier 2: Instant Breakeven Lock at Entry + 0.5/1.0 pt once Gain >= +1.0x ATR
  Tier 3: Asymmetric Profit Ratchet (Trails peak price at 0.75x-1.25x ATR once Gain >= +2.0x ATR)
  Tier 4: Time-Decay Penalty (If in trade > N mins with no momentum, tighten SL to cut Theta bleed)

Evaluates on RTX 3060 Tensor Cores across 7 Years (1,588 days) & August 18-20.
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
flag_setup = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
entries_mask = super_setup | flag_setup


@torch.inference_mode()
def simulate_tiered_ratchet_gpu(
    entries_mask: torch.Tensor,
    configs: list[dict],
):
    coords = torch.nonzero(entries_mask, as_tuple=False)
    N_trades = coords.shape[0]
    if N_trades == 0:
        return []

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

    time_offsets = torch.arange(1, max_future + 1, device=device).unsqueeze(0).expand(N_trades, -1)

    results = []

    for p in configs:
        sl_m = p["sl_mult"]
        be_trig_m = p["be_trig_atr"]        # Gain at which BE is locked (e.g. +1.0x ATR)
        trail_trig_m = p["trail_trig_atr"]  # Gain at which trailing activates (e.g. +2.0x ATR)
        trail_dist_m = p["trail_dist_atr"]  # Trailing distance (e.g. 0.75x ATR)
        time_stop_min = p["time_stop_min"]  # Minutes before time decay penalty (e.g. 20-30 mins)
        target_tp_m = p["target_tp_atr"]    # Hard target TP multiplier (e.g. 4.0x ATR)

        eff_atr = base_atr
        sl_d = (sl_m * eff_atr).clamp(min=5.0, max=30.0)
        tp_d = (target_tp_m * eff_atr).clamp(min=8.0, max=60.0)

        init_sl = ep_exp - sl_d.unsqueeze(1)
        tp_barrier = ep_exp + tp_d.unsqueeze(1)

        # Dynamic Barrier Construction across the 4 Tiers
        gains = running_peaks - ep_exp

        # Tier 2: Breakeven Lock Level (Entry + 0.5 pt)
        be_level = ep_exp + 0.5
        is_be_reached = gains >= (be_trig_m * eff_atr).unsqueeze(1)

        # Tier 3: Asymmetric Trailing Stop (Peak - Trail Distance)
        trail_level = running_peaks - (trail_dist_m * eff_atr).unsqueeze(1)
        is_trail_reached = gains >= (trail_trig_m * eff_atr).unsqueeze(1)

        # Tier 4: Time Decay Tightening (If held > time_stop_min and gain < BE, tighten SL to Entry - 0.5x ATR)
        time_penalty_sl = ep_exp - (0.5 * eff_atr).unsqueeze(1)
        is_time_decay_active = (time_offsets >= time_stop_min) & (~is_be_reached)

        # Combine Barriers dynamically
        dyn_sl = init_sl.clone()
        dyn_sl = torch.where(is_time_decay_active, torch.maximum(dyn_sl, time_penalty_sl), dyn_sl)
        dyn_sl = torch.where(is_be_reached, torch.maximum(dyn_sl, be_level), dyn_sl)
        dyn_sl = torch.where(is_trail_reached, torch.maximum(dyn_sl, trail_level), dyn_sl)

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

        rs_cpu = rs_net.cpu().numpy()
        pts_cpu = pts.cpu().numpy()
        d_idx_cpu = d_idx.cpu().numpy()

        n_t = len(rs_cpu)
        wins = [r for r in rs_cpu if r > 0]
        losses = [r for r in rs_cpu if r <= 0]
        net_rs_tot = float(np.sum(rs_cpu))
        net_pts_tot = float(np.sum(pts_cpu))
        wr = len(wins) / n_t * 100.0 if n_t > 0 else 0.0
        pf = sum(wins) / abs(sum(losses)) if abs(sum(losses)) > 0 else 99.0

        # Calculate monthly win rate
        df_trades = pd.DataFrame({"day_idx": d_idx_cpu, "rs": rs_cpu})
        df_trades["date"] = [days[i] for i in df_trades["day_idx"]]
        df_trades["month"] = df_trades["date"].str[:7]
        all_months = sorted(list(set(d[:7] for d in days)))
        monthly_pnl = df_trades.groupby("month")["rs"].sum().reindex(all_months, fill_value=0.0)
        pos_months = (monthly_pnl > 0).sum()
        month_wr = (pos_months / len(all_months)) * 100.0

        # Drawdown
        eq = np.cumsum(rs_cpu)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
        calmar = net_rs_tot / max_dd if max_dd > 0 else 0.0

        results.append({
            "config": p,
            "trades": n_t,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(wr, 2),
            "net_points": round(net_pts_tot, 2),
            "net_rs": round(net_rs_tot, 2),
            "profit_factor": round(pf, 3),
            "max_drawdown_rs": round(max_dd, 2),
            "calmar_ratio": round(calmar, 3),
            "positive_months": int(pos_months),
            "total_months": len(all_months),
            "month_win_rate": round(month_wr, 1),
            "avg_month_rs": round(float(monthly_pnl.mean()), 2),
        })

    return results


def main():
    print("=" * 145)
    print("THETA-BEATING TIERED RATCHET & ASYMMETRIC TRAILING STOP GPU OPTIMIZATION")
    print("=" * 145)

    configs = []
    for sl_m in [1.5, 2.0, 2.5]:
        for be_trig in [0.75, 1.0, 1.25]:
            for trail_trig in [1.5, 2.0, 2.5]:
                for trail_dist in [0.5, 0.75, 1.0]:
                    for time_stop in [15, 20, 25, 30, 45]:
                        for tp_m in [3.5, 4.0, 5.0, 6.0]:
                            configs.append({
                                "sl_mult": sl_m,
                                "be_trig_atr": be_trig,
                                "trail_trig_atr": trail_trig,
                                "trail_dist_atr": trail_dist,
                                "time_stop_min": time_stop,
                                "target_tp_atr": tp_m,
                            })

    print(f"Sweeping {len(configs)} Tiered Ratchet Parameter Combinations on RTX 3060 Tensor Cores...", flush=True)
    t0 = time.time()
    all_res = simulate_tiered_ratchet_gpu(entries_mask, configs)
    print(f"  Completed {len(all_res)} evaluations in {time.time()-t0:.2f}s on GPU!", flush=True)

    # 1. Top 5 Ranked by Net Realized Profit
    top_profit = sorted(all_res, key=lambda x: x["net_rs"], reverse=True)[:5]
    print("\n" + "=" * 145)
    print(">>> TOP 5 THETA-BEATING RATCHET SETTINGS RANKED BY 7-YEAR NET REALIZED PROFIT:")
    print("=" * 145)
    print(f"{'Rank':4s} | {'SL':4s} | {'BE Trig':7s} | {'Trail (Trig/Dist)':17s} | {'Time Stop':9s} | {'TP':4s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Month WR':9s}")
    print("-" * 145)
    for r, item in enumerate(top_profit, 1):
        cfg = item["config"]
        tr_str = f"{cfg['trail_trig_atr']}x / {cfg['trail_dist_atr']}x"
        print(f"{r:4d} | {cfg['sl_mult']:4.2f} | {cfg['be_trig_atr']:5.2f}x  | {tr_str:17s} | {cfg['time_stop_min']:4d} min  | {cfg['target_tp_atr']:4.2f} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['month_win_rate']:7.1f}%")

    # 2. Top 5 Ranked by Highest Monthly Win Rate (Consistency Champion)
    top_month_wr = sorted(all_res, key=lambda x: (x["month_win_rate"], x["net_rs"]), reverse=True)[:5]
    print("\n" + "=" * 145)
    print(">>> TOP 5 THETA-BEATING RATCHET SETTINGS RANKED BY HIGHEST MONTHLY WIN RATE (% GREEN MONTHS):")
    print("=" * 145)
    print(f"{'Rank':4s} | {'SL':4s} | {'BE Trig':7s} | {'Trail (Trig/Dist)':17s} | {'Time Stop':9s} | {'TP':4s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Month WR':9s}")
    print("-" * 145)
    for r, item in enumerate(top_month_wr, 1):
        cfg = item["config"]
        tr_str = f"{cfg['trail_trig_atr']}x / {cfg['trail_dist_atr']}x"
        print(f"{r:4d} | {cfg['sl_mult']:4.2f} | {cfg['be_trig_atr']:5.2f}x  | {tr_str:17s} | {cfg['time_stop_min']:4d} min  | {cfg['target_tp_atr']:4.2f} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['month_win_rate']:7.1f}%")

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "theta_beating_ratchet_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "top_profit": top_profit,
        "top_month_wr": top_month_wr,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Theta-Beating JSON Ledger]: {out_file}")
    print("=" * 145)


if __name__ == "__main__":
    main()
