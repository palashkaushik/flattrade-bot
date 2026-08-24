"""GPU Optimization WITHOUT Daily Loss Halt (Continuous Trading).

Evaluates 7-year performance across trailing stops, VIX scaling, and trade limits
WITHOUT stopping trading on losses.

Hardware: NVIDIA RTX 3060 (12 GB VRAM · CUDA 12.1)
"""

from __future__ import annotations

import argparse
import json
import os
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
base_entries = super_setup | flag_setup


@torch.inference_mode()
def simulate_no_halt_trailing_gpu(
    entries_mask: torch.Tensor,
    d_h: torch.Tensor,
    d_l: torch.Tensor,
    d_c: torch.Tensor,
    d_atr: torch.Tensor,
    d_vix: torch.Tensor,
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
    trade_vix = d_vix[d_idx, b_idx].clamp(min=8.0, max=80.0)

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

    # Running cumulative high after entry (for Trailing SL)
    fut_h_clean = torch.where(valid, fut_h, ep.unsqueeze(1))
    running_peaks = torch.cummax(fut_h_clean, dim=1).values

    ep_exp = ep.unsqueeze(1)
    results = []

    for p in configs:
        sl_m = p["sl_mult"]
        tp_m = p["tp_mult"]
        gamma = p.get("gamma", 0.5)
        trail_trig_m = p.get("trail_trigger_atr", None)
        trail_dist_m = p.get("trail_dist_atr", 0.75)
        max_trades = p.get("max_trades_day", 999)  # NO artificial loss halt

        vix_scale = torch.pow(trade_vix / 15.0, gamma).clamp(min=0.6, max=2.0)
        eff_atr = base_atr * vix_scale
        sl_d = (sl_m * eff_atr).clamp(min=5.0, max=30.0)
        tp_d = (tp_m * eff_atr).clamp(min=8.0, max=60.0)

        init_sl = ep_exp - sl_d.unsqueeze(1)
        tp_barrier = ep_exp + tp_d.unsqueeze(1)

        # Dynamic Trailing SL Floor using Parallel Prefix Scan
        if trail_trig_m is not None:
            trig_gain = (trail_trig_m * eff_atr).unsqueeze(1)
            trail_d = (trail_dist_m * eff_atr).unsqueeze(1)
            gains = running_peaks - ep_exp
            is_trailing = gains >= trig_gain
            trailing_sl_level = running_peaks - trail_d
            dyn_sl_barrier = torch.where(is_trailing, torch.maximum(init_sl, trailing_sl_level), init_sl)
        else:
            dyn_sl_barrier = init_sl.expand(-1, max_future)

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

        kept_rs, kept_pts = [], []
        last_d = None
        trades_today = 0

        for r, pt, d in zip(rs_sorted, pts_sorted, days_sorted):
            if d != last_d:
                last_d = d
                trades_today = 0

            if trades_today >= max_trades:
                continue

            trades_today += 1
            # CONTINUOUS TRADING: NO HALT ON LOSS
            kept_rs.append(r)
            kept_pts.append(pt)

        if not kept_rs:
            continue

        n_t = len(kept_rs)
        wins = [r for r in kept_rs if r > 0]
        losses = [r for r in kept_rs if r <= 0]
        win_tot = sum(wins)
        loss_tot = abs(sum(losses))
        net_rs_tot = sum(kept_rs)
        net_pts_tot = sum(kept_pts)
        wr = len(wins) / n_t * 100.0
        pf = win_tot / loss_tot if loss_tot > 0 else (99.0 if win_tot > 0 else 0.0)

        eq = np.cumsum(kept_rs)
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
            "fees_rs": round(n_t * FEE, 2),
        })

    return results


def main():
    # Sweep across both Full Day and Prime Session Window
    bar_indices = torch.arange(375, device=device).unsqueeze(0).expand(N_DAYS, -1)
    prime_window = ((bar_indices >= 5) & (bar_indices <= 120)) | ((bar_indices >= 255) & (bar_indices <= 330))

    configs = []
    for window_type in ["full_day", "prime_window"]:
        for max_tr in [1, 2, 3, 999]:  # 999 = unlimited trades per day
            for trig in [0.75, 1.0, 1.25, 1.5, None]:
                for dist in [0.5, 0.75, 1.0]:
                    for sl_m in [1.5, 2.0, 2.5, 3.0]:
                        for tp_m in [2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
                            for gamma in [0.0, 0.5, 1.0]:
                                configs.append({
                                    "window_type": window_type,
                                    "max_trades_day": max_tr,
                                    "trail_trigger_atr": trig,
                                    "trail_dist_atr": dist,
                                    "sl_mult": sl_m,
                                    "tp_mult": tp_m,
                                    "gamma": gamma,
                                })

    print(f"Sweeping {len(configs)} configurations WITHOUT HALT on GPU...", flush=True)
    t0 = time.time()

    # Split execution by window type
    full_day_cfgs = [c for c in configs if c["window_type"] == "full_day"]
    prime_cfgs = [c for c in configs if c["window_type"] == "prime_window"]

    res_full_day = simulate_no_halt_trailing_gpu(base_entries, d_h, d_l, d_c, d_atr, d_vix, full_day_cfgs)
    res_prime = simulate_no_halt_trailing_gpu(base_entries & prime_window, d_h, d_l, d_c, d_atr, d_vix, prime_cfgs)
    all_results = res_full_day + res_prime

    print(f"  Finished {len(all_results)} evaluations in {time.time()-t0:.2f}s on RTX 3060", flush=True)

    # Filter positive profit
    profitable = [r for r in all_results if r["net_rs"] > 0]

    # 1. Top 5 Ranked by Max Net Profit
    top_5_profit = sorted(profitable, key=lambda x: x["net_rs"], reverse=True)[:5]
    print("\n" + "=" * 135)
    print(">>> TOP 5 SETTINGS (NO LOSS HALT) RANKED BY MAX NET POINTS & PROFIT:")
    print("=" * 135)
    print(f"{'Rank':4s} | {'Window':12s} | {'Trades/Day':10s} | {'Trail (Trig/Dist)':17s} | {'SL':4s} | {'TP':4s} | {'gamma':5s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
    print("-" * 145)
    for r, item in enumerate(top_5_profit, 1):
        cfg = item["config"]
        tr_str = f"{cfg['trail_trigger_atr']}x / {cfg['trail_dist_atr']}x" if cfg['trail_trigger_atr'] else "None (Fixed TP)"
        tr_day_str = f"Max {cfg['max_trades_day']}" if cfg['max_trades_day'] < 999 else "Unlimited"
        print(f"{r:4d} | {cfg['window_type']:12s} | {tr_day_str:10s} | {tr_str:17s} | {cfg['sl_mult']:4.2f} | {cfg['tp_mult']:4.2f} | {cfg['gamma']:5.2f} | {item['trades']:7d} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['calmar_ratio']:7.3f}")

    # 2. Top 5 Ranked by Least Max Drawdown
    top_5_lowest_dd = sorted(profitable, key=lambda x: x["max_drawdown_rs"])[:5]
    print("\n" + "=" * 135)
    print(">>> TOP 5 SETTINGS (NO LOSS HALT) RANKED BY LEAST MAX DRAWDOWN:")
    print("=" * 135)
    print(f"{'Rank':4s} | {'Window':12s} | {'Trades/Day':10s} | {'Trail (Trig/Dist)':17s} | {'SL':4s} | {'TP':4s} | {'gamma':5s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
    print("-" * 145)
    for r, item in enumerate(top_5_lowest_dd, 1):
        cfg = item["config"]
        tr_str = f"{cfg['trail_trigger_atr']}x / {cfg['trail_dist_atr']}x" if cfg['trail_trigger_atr'] else "None (Fixed TP)"
        tr_day_str = f"Max {cfg['max_trades_day']}" if cfg['max_trades_day'] < 999 else "Unlimited"
        print(f"{r:4d} | {cfg['window_type']:12s} | {tr_day_str:10s} | {tr_str:17s} | {cfg['sl_mult']:4.2f} | {cfg['tp_mult']:4.2f} | {cfg['gamma']:5.2f} | {item['trades']:7d} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['calmar_ratio']:7.3f}")

    # 3. Top 5 Ranked by Calmar Ratio (Best Risk-Adjusted)
    top_5_calmar = sorted(profitable, key=lambda x: x["calmar_ratio"], reverse=True)[:5]
    print("\n" + "=" * 135)
    print(">>> TOP 5 SETTINGS (NO LOSS HALT) RANKED BY CALMAR RATIO (RETURN / DRAWDOWN):")
    print("=" * 135)
    print(f"{'Rank':4s} | {'Window':12s} | {'Trades/Day':10s} | {'Trail (Trig/Dist)':17s} | {'SL':4s} | {'TP':4s} | {'gamma':5s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
    print("-" * 145)
    for r, item in enumerate(top_5_calmar, 1):
        cfg = item["config"]
        tr_str = f"{cfg['trail_trigger_atr']}x / {cfg['trail_dist_atr']}x" if cfg['trail_trigger_atr'] else "None (Fixed TP)"
        tr_day_str = f"Max {cfg['max_trades_day']}" if cfg['max_trades_day'] < 999 else "Unlimited"
        print(f"{r:4d} | {cfg['window_type']:12s} | {tr_day_str:10s} | {tr_str:17s} | {cfg['sl_mult']:4.2f} | {cfg['tp_mult']:4.2f} | {cfg['gamma']:5.2f} | {item['trades']:7d} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['calmar_ratio']:7.3f}")

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "no_halt_optimization_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "top_profit": top_5_profit,
        "top_lowest_dd": top_5_lowest_dd,
        "top_calmar": top_5_calmar,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved No-Halt JSON]: {out_file}")
    print("=" * 135)


if __name__ == "__main__":
    main()
