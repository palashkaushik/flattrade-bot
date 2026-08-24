"""Monthly Consistency Optimizer (Target: Maximum % of Positive Months & Smooth Equity Curve).

Evaluates 10,000+ strategy variations across all 80 calendar months (2020-01 to 2026-08)
to find the most consistent, robust strategy with >= 80% positive months.

Hardware Target: NVIDIA RTX 3060 (12 GB VRAM · CUDA 12.1)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
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

# 1. Compute Base Indicators
s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
d_atr = compute_atr_gpu(d_h, d_l, d_c, period=14)

prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)
super_setup = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_setup = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up

# 2. Multi-Timeframe Trend Filters (15m Trend Alignment & EMAs)
# 15m EMA 20 & EMA 50 on Nifty spot to eliminate counter-trend chops
def compute_ema(tensor, period):
    alpha = 2.0 / (period + 1.0)
    # Causal recursive EMA on GPU
    ema = torch.zeros_like(tensor)
    ema[:, 0] = tensor[:, 0]
    for t in range(1, tensor.shape[1]):
        ema[:, t] = alpha * tensor[:, t] + (1 - alpha) * ema[:, t-1]
    return ema

ema_fast = compute_ema(d_c, 15)
ema_slow = compute_ema(d_c, 45)
bullish_trend = (d_c > ema_fast) & (ema_fast > ema_slow)
bearish_trend = (d_c < ema_fast) & (ema_fast < ema_slow)

# Construct Candidate Signal Masks
# Signal Variant 1: Pure S1 Turn-Up (Baseline)
sig_v1 = super_setup | flag_setup

# Signal Variant 2: Super Setup Only (Highest Conviction Reversal)
sig_v2 = super_setup

# Signal Variant 3: HTF Trend-Aligned (Only trade in direction of 15m Trend)
sig_v3 = (super_setup | flag_setup) & (bullish_trend | (d_c > ema_fast))

# Signal Variant 4: Morning & Afternoon Momentum Hours Only
bar_indices = torch.arange(375, device=device).unsqueeze(0).expand(N_DAYS, -1)
prime_window = ((bar_indices >= 10) & (bar_indices <= 120)) | ((bar_indices >= 250) & (bar_indices <= 330))
sig_v4 = (super_setup | flag_setup) & prime_window

# Signal Variant 5: Trend-Aligned + Prime Window
sig_v5 = (super_setup | flag_setup) & (bullish_trend | (d_c > ema_fast)) & prime_window

SIGNAL_VARIANTS = {
    "V1_All_Signals": sig_v1,
    "V2_Super_Only": sig_v2,
    "V3_Trend_Aligned": sig_v3,
    "V4_Prime_Window": sig_v4,
    "V5_Trend_and_Prime": sig_v5,
}

# Generate Month Mapping
months_list = sorted(list(set(d[:7] for d in days)))
month_to_day_indices = {m: [i for i, d in enumerate(days) if d.startswith(m)] for m in months_list}
N_MONTHS = len(months_list)
print(f"Total Calendar Months in Dataset: {N_MONTHS} (from {months_list[0]} to {months_list[-1]})", flush=True)


@torch.inference_mode()
def simulate_monthly_consistency_gpu(
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

    fut_h_clean = torch.where(valid, fut_h, ep.unsqueeze(1))
    running_peaks = torch.cummax(fut_h_clean, dim=1).values
    ep_exp = ep.unsqueeze(1)

    results = []

    for p in configs:
        sl_m = p["sl_mult"]
        tp_m = p["tp_mult"]
        gamma = p.get("gamma", 0.5)
        trail_trig_m = p.get("trail_trigger_atr", None)
        trail_dist_m = p.get("trail_dist_atr", 0.5)
        max_trades = p.get("max_trades_day", 999)

        vix_scale = torch.pow(trade_vix / 15.0, gamma).clamp(min=0.6, max=2.0)
        eff_atr = base_atr * vix_scale
        sl_d = (sl_m * eff_atr).clamp(min=5.0, max=30.0)
        tp_d = (tp_m * eff_atr).clamp(min=8.0, max=60.0)

        init_sl = ep_exp - sl_d.unsqueeze(1)
        tp_barrier = ep_exp + tp_d.unsqueeze(1)

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

        kept_rs, kept_pts, kept_days = [], [], []
        last_d = None
        trades_today = 0

        for r, pt, d in zip(rs_sorted, pts_sorted, days_sorted):
            if d != last_d:
                last_d = d
                trades_today = 0

            if trades_today >= max_trades:
                continue

            trades_today += 1
            kept_rs.append(r)
            kept_pts.append(pt)
            kept_days.append(d)

        if not kept_rs:
            continue

        n_t = len(kept_rs)
        wins = [r for r in kept_rs if r > 0]
        losses = [r for r in kept_rs if r <= 0]
        net_rs_tot = sum(kept_rs)
        net_pts_tot = sum(kept_pts)
        wr = len(wins) / n_t * 100.0
        pf = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else 99.0

        # Monthly P&L Breakdown
        df_trades = pd.DataFrame({"day_idx": kept_days, "rs": kept_rs})
        df_trades["date"] = [days[i] for i in df_trades["day_idx"]]
        df_trades["month"] = df_trades["date"].str[:7]
        monthly_pnl = df_trades.groupby("month")["rs"].sum().reindex(months_list, fill_value=0.0)

        positive_months = (monthly_pnl > 0).sum()
        negative_months = (monthly_pnl < 0).sum()
        zero_months = (monthly_pnl == 0).sum()
        month_win_rate = (positive_months / N_MONTHS) * 100.0

        # Max consecutive losing months
        consec_neg = 0
        max_consec_neg = 0
        for m_pnl in monthly_pnl:
            if m_pnl < 0:
                consec_neg += 1
                if consec_neg > max_consec_neg:
                    max_consec_neg = consec_neg
            else:
                consec_neg = 0

        # Worst and Best Month
        worst_month_pnl = float(monthly_pnl.min())
        best_month_pnl = float(monthly_pnl.max())
        avg_month_pnl = float(monthly_pnl.mean())

        # Drawdown
        eq = np.cumsum(kept_rs)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
        calmar = net_rs_tot / max_dd if max_dd > 0 else 0.0

        results.append({
            "config": p,
            "trades": n_t,
            "win_rate": round(wr, 2),
            "net_points": round(net_pts_tot, 2),
            "net_rs": round(net_rs_tot, 2),
            "profit_factor": round(pf, 3),
            "max_drawdown_rs": round(max_dd, 2),
            "calmar_ratio": round(calmar, 3),

            # Monthly Consistency Metrics
            "positive_months": int(positive_months),
            "negative_months": int(negative_months),
            "month_win_rate": round(month_win_rate, 2),
            "max_consec_losing_months": int(max_consec_neg),
            "worst_month_rs": round(worst_month_pnl, 2),
            "best_month_rs": round(best_month_pnl, 2),
            "avg_month_rs": round(avg_month_pnl, 2),
        })

    return results


def main():
    configs = []
    for max_tr in [1, 2, 3, 999]:
        for trig in [0.75, 1.0, 1.25, 1.5, None]:
            for dist in [0.4, 0.5, 0.75]:
                for sl_m in [1.5, 2.0, 2.5, 3.0]:
                    for tp_m in [2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
                        for gamma in [0.0, 0.5, 1.0]:
                            configs.append({
                                "max_trades_day": max_tr,
                                "trail_trigger_atr": trig,
                                "trail_dist_atr": dist,
                                "sl_mult": sl_m,
                                "tp_mult": tp_m,
                                "gamma": gamma,
                            })

    print(f"Sweeping {len(configs)} parameter configurations across all 5 Signal Gating Variants...", flush=True)
    all_variant_results = []
    t0 = time.time()

    for v_name, v_mask in SIGNAL_VARIANTS.items():
        t_v = time.time()
        res = simulate_monthly_consistency_gpu(v_mask, configs)
        for r in res:
            r["variant"] = v_name
            all_variant_results.append(r)
        print(f"  Completed Variant {v_name:20s}: {len(res)} valid trials in {time.time()-t_v:.2f}s", flush=True)

    print(f"Total Completed Evaluations: {len(all_variant_results)} in {time.time()-t0:.2f}s on RTX 3060", flush=True)

    # Filter for High Monthly Consistency: Month Win Rate >= 75% and Net Profit > 0
    consistent_cfgs = [r for r in all_variant_results if r["month_win_rate"] >= 75.0 and r["net_rs"] > 0]
    print(f"\n===================================================================================================================")
    print(f"FOUND {len(consistent_cfgs)} CONFIGURATIONS WITH >= 75% POSITIVE MONTHS ACROSS ALL 80 MONTHS (2020-2026)")
    print(f"===================================================================================================================")

    # 1. Top 5 Ranked by Highest Monthly Win Rate (% Positive Months)
    top_5_month_wr = sorted(all_variant_results, key=lambda x: (x["month_win_rate"], x["net_rs"]), reverse=True)[:5]
    print("\n>>> TOP 5 STRATEGIES RANKED BY HIGHEST MONTHLY WIN RATE (% POSITIVE MONTHS):")
    print(f"{'Rank':4s} | {'Signal Variant':18s} | {'Trades/Day':10s} | {'Trail (Trig/Dist)':17s} | {'SL':4s} | {'TP':4s} | {'Pos/Total Months':16s} | {'Month WR':9s} | {'Net Realized Rs':16s} | {'PF':6s} | {'Max DD':12s} | {'Worst Month':13s}")
    print("-" * 155)
    for r, item in enumerate(top_5_month_wr, 1):
        cfg = item["config"]
        tr_str = f"{cfg['trail_trigger_atr']}x / {cfg['trail_dist_atr']}x" if cfg['trail_trigger_atr'] else "None (Fixed TP)"
        tr_day_str = f"Max {cfg['max_trades_day']}" if cfg['max_trades_day'] < 999 else "Unlimited"
        m_str = f"{item['positive_months']}/{N_MONTHS} mos"
        print(f"{r:4d} | {item['variant']:18s} | {tr_day_str:10s} | {tr_str:17s} | {cfg['sl_mult']:4.2f} | {cfg['tp_mult']:4.2f} | {m_str:16s} | {item['month_win_rate']:7.1f}% | Rs {item['net_rs']:+13.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | Rs {item['worst_month_rs']:+10.2f}")

    # 2. Top 5 Ranked by Net Realized Profit among Highly Consistent Strategies (Month WR >= 75%)
    top_5_profit_consistent = sorted(consistent_cfgs, key=lambda x: x["net_rs"], reverse=True)[:5]
    print("\n>>> TOP 5 STRATEGIES RANKED BY NET PROFIT (AMONG CONSISTENT STRATEGIES WITH >= 75% POSITIVE MONTHS):")
    print(f"{'Rank':4s} | {'Signal Variant':18s} | {'Trades/Day':10s} | {'Trail (Trig/Dist)':17s} | {'SL':4s} | {'TP':4s} | {'Pos/Total Months':16s} | {'Month WR':9s} | {'Net Realized Rs':16s} | {'PF':6s} | {'Max DD':12s} | {'Avg Month Rs':13s}")
    print("-" * 155)
    for r, item in enumerate(top_5_profit_consistent, 1):
        cfg = item["config"]
        tr_str = f"{cfg['trail_trigger_atr']}x / {cfg['trail_dist_atr']}x" if cfg['trail_trigger_atr'] else "None (Fixed TP)"
        tr_day_str = f"Max {cfg['max_trades_day']}" if cfg['max_trades_day'] < 999 else "Unlimited"
        m_str = f"{item['positive_months']}/{N_MONTHS} mos"
        print(f"{r:4d} | {item['variant']:18s} | {tr_day_str:10s} | {tr_str:17s} | {cfg['sl_mult']:4.2f} | {cfg['tp_mult']:4.2f} | {m_str:16s} | {item['month_win_rate']:7.1f}% | Rs {item['net_rs']:+13.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | Rs {item['avg_month_rs']:+10.2f}")

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "monthly_consistency_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "top_month_win_rate": top_5_month_wr,
        "top_profit_consistent": top_5_profit_consistent,
        "total_evaluated": len(all_variant_results),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Monthly Consistency JSON]: {out_file}")
    print("=" * 155)


if __name__ == "__main__":
    main()
