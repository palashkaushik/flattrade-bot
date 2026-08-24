"""Forensic Diagnostic & Strategy Discovery Engine on RTX 3060 GPU.

Fixes bar indexing: 09:15 AM = Bar 0 (Min 555), 09:30 AM = Bar 15 (Min 570).
Investigates:
  1. Opening Session Filter (Avoid 09:15–09:30 AM Opening Chop)
  2. Post-Trade Cooldown (5, 10, 15, 20 mins pause)
  3. Max Trades per Day (2, 3, 4, 6, No Limit)
  4. Trend Alignment (Option price > EMA20 > EMA50)
  5. Multi-Tier Exit Geometry (SL 1.50-2.50x, BE Lock 1.0-1.25x, Trail 0.40-0.75x, Time Stop 15-25m)
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

# Compute Option EMA 20 & EMA 50 for Trend Alignment
def compute_ema_gpu(prices: torch.Tensor, period: int) -> torch.Tensor:
    alpha = 2.0 / (period + 1.0)
    ema = prices.clone()
    for t in range(1, prices.shape[1]):
        ema[:, t] = alpha * prices[:, t] + (1.0 - alpha) * ema[:, t - 1]
    return ema

ema20 = compute_ema_gpu(d_c, 20)
ema50 = compute_ema_gpu(d_c, 50)
trend_bullish = (d_c > ema20) & (ema20 > ema50)


@torch.inference_mode()
def run_discovery_sweep_gpu():
    print("=" * 145)
    print("DIAGNOSTIC & DISCOVERY: SYSTEMATIC ROOT-CAUSE REMEDIATION SWEEP ON GPU")
    print("=" * 145)

    all_months = sorted(list(set(d[:7] for d in days)))

    # Parameter grid for structural improvements
    configs = []
    for start_bar in [5, 15, 20]:  # 09:20 AM (Bar 5), 09:30 AM (Bar 15), 09:35 AM (Bar 20)
        for setup_type in ["super_only", "super_plus_flag", "trend_super"]:
            for max_trades in [2, 3, 4, 6, 999]:  # Max trades per day cap
                for cooldown in [0, 5, 10, 15]:     # Cooldown minutes
                    for sl_m in [1.50, 2.00, 2.50]:
                        for be_trig in [1.00, 1.25]:
                            for trail_dist in [0.40, 0.50]:
                                for time_stop in [15, 20]:
                                    configs.append({
                                        "start_bar": start_bar,
                                        "setup_type": setup_type,
                                        "max_trades": max_trades,
                                        "cooldown": cooldown,
                                        "sl_m": sl_m,
                                        "be_trig": be_trig,
                                        "trail_dist": trail_dist,
                                        "time_stop": time_stop,
                                    })

    print(f"Evaluating {len(configs)} Structural Architecture Variations across 1,588 Days...", flush=True)
    t0 = time.time()

    base_mask_map = {
        "super_only": super_setup,
        "super_plus_flag": super_setup | flag_setup,
        "trend_super": (super_setup | flag_setup) & trend_bullish,
    }

    results = []

    for cfg in configs:
        st_bar = cfg["start_bar"]
        stype = cfg["setup_type"]
        max_tr = cfg["max_trades"]
        cd_min = cfg["cooldown"]
        sl_mult = cfg["sl_m"]
        be_mult = cfg["be_trig"]
        tr_dist = cfg["trail_dist"]
        t_stop = cfg["time_stop"]

        raw_mask = base_mask_map[stype].clone()
        raw_mask[:, :st_bar] = False
        raw_mask[:, 345:] = False

        coords = torch.nonzero(raw_mask, as_tuple=False)
        if coords.shape[0] == 0:
            continue

        d_arr = coords[:, 0].cpu().numpy()
        b_arr = coords[:, 1].cpu().numpy()

        unique_days, split_indices = np.unique(d_arr, return_index=True)
        splits = np.split(np.arange(len(d_arr)), split_indices[1:])

        day_rs_totals = np.zeros(N_DAYS, dtype=np.float32)
        all_day_wins = 0
        all_day_losses = 0

        for d_i, idxs in zip(unique_days, splits):
            day_bars = b_arr[idxs]
            last_exit_bar = -999
            trades_done = 0
            day_pnl = 0.0

            for b in day_bars:
                if trades_done >= max_tr:
                    break
                if b < last_exit_bar + cd_min:
                    continue

                ep = float(d_c[d_i, b])
                atr_val = float(d_atr[d_i, b].clamp(min=5.0, max=25.0))
                sl_d = max(5.0, min(30.0, sl_mult * atr_val))
                tp_d = max(8.0, min(60.0, 3.5 * atr_val))
                init_sl = ep - sl_d
                tp_barrier = ep + tp_d

                curr_sl = init_sl
                peak = ep
                is_be = False

                exit_px = float(d_c[d_i, 344])
                exit_bar = 344

                for fut_b in range(b + 1, 345):
                    dur = fut_b - b
                    fh = float(d_h[d_i, fut_b])
                    fl = float(d_l[d_i, fut_b])

                    if fh > peak:
                        peak = fh

                    gain = peak - ep

                    # Tier 4: Time Stop
                    if dur >= t_stop and not is_be:
                        theta_sl = ep - (0.5 * atr_val)
                        if theta_sl > curr_sl:
                            curr_sl = theta_sl

                    # Tier 2: Breakeven Lock
                    if gain >= be_mult * atr_val:
                        be_sl = ep + 0.5
                        if be_sl > curr_sl:
                            curr_sl = be_sl
                            is_be = True

                    # Tier 3: Trailing Stop
                    if gain >= 1.5 * atr_val:
                        trail_sl = peak - (tr_dist * atr_val)
                        if trail_sl > curr_sl:
                            curr_sl = trail_sl

                    if fl <= curr_sl:
                        exit_px = curr_sl
                        exit_bar = fut_b
                        break
                    elif fh >= tp_barrier:
                        exit_px = tp_barrier
                        exit_bar = fut_b
                        break

                pts = (exit_px - ep) * 0.50
                net_r = pts * LOT_SIZE - FEE
                day_pnl += net_r
                trades_done += 1
                last_exit_bar = exit_bar

                if net_r > 0:
                    all_day_wins += 1
                else:
                    all_day_losses += 1

            day_rs_totals[d_i] = day_pnl

        tot_trades = all_day_wins + all_day_losses
        if tot_trades == 0:
            continue

        tot_rs = float(np.sum(day_rs_totals))
        wr = (all_day_wins / tot_trades) * 100.0

        # Monthly metrics
        df_m = pd.DataFrame({"day": days, "rs": day_rs_totals})
        df_m["month"] = df_m["day"].str[:7]
        m_pnl = df_m.groupby("month")["rs"].sum().reindex(all_months, fill_value=0.0)
        pos_m = int((m_pnl > 0).sum())
        month_wr = (pos_m / len(all_months)) * 100.0

        # Drawdown
        eq = np.cumsum(day_rs_totals)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
        calmar = tot_rs / max_dd if max_dd > 0 else 0.0

        results.append({
            "config": cfg,
            "trades": tot_trades,
            "wins": all_day_wins,
            "losses": all_day_losses,
            "win_rate": round(wr, 2),
            "net_rs": round(tot_rs, 2),
            "max_dd": round(max_dd, 2),
            "calmar": round(calmar, 3),
            "pos_months": pos_m,
            "month_wr": round(month_wr, 1),
            "trades_per_day": round(tot_trades / N_DAYS, 2),
        })

    print(f"Sweep Completed in {time.time()-t0:.2f}s! Found {len(results)} valid configurations.", flush=True)

    # 1. Top 10 Ranked by Highest Monthly Consistency (% Green Months) & Calmar Ratio
    top_consistent = sorted(results, key=lambda x: (x["month_wr"], x["calmar"], x["net_rs"]), reverse=True)[:10]
    print("\n" + "=" * 155)
    print(">>> TOP 10 MOST CONSISTENT STRATEGIES RANKED BY HIGHEST MONTHLY WIN RATE (% GREEN MONTHS):")
    print("=" * 155)
    print(f"{'Rank':4s} | {'Start':8s} | {'Setup Type':16s} | {'Max Tr':6s} | {'Cooldown':8s} | {'SL':4s} | {'BE':5s} | {'Trail':5s} | {'Win Rate':9s} | {'Net Rs':15s} | {'Max DD':12s} | {'Calmar':7s} | {'Month WR':10s} | {'Tr/Day':6s}")
    print("-" * 155)
    for r, item in enumerate(top_consistent, 1):
        c = item["config"]
        st_str = "09:20 AM" if c["start_bar"] == 5 else ("09:30 AM" if c["start_bar"] == 15 else "09:35 AM")
        cd_str = f"{c['cooldown']} min" if c["cooldown"] > 0 else "None"
        max_str = f"{c['max_trades']}" if c["max_trades"] < 999 else "No Limit"
        print(f"{r:4d} | {st_str:8s} | {c['setup_type']:16s} | {max_str:6s} | {cd_str:8s} | {c['sl_m']:4.2f} | {c['be_trig']:4.2f}x | {c['trail_dist']:4.2f}x | {item['win_rate']:7.1f}% | Rs {item['net_rs']:+12.2f} | Rs {item['max_dd']:9.2f} | {item['calmar']:7.3f} | {item['month_wr']:7.1f}% ({item['pos_months']}/{len(all_months)}) | {item['trades_per_day']:6.2f}")

    # 2. Top 10 Ranked by Maximum Realized Profit
    top_profit = sorted(results, key=lambda x: x["net_rs"], reverse=True)[:10]
    print("\n" + "=" * 155)
    print(">>> TOP 10 STRATEGIES RANKED BY 7-YEAR NET REALIZED PROFIT:")
    print("=" * 155)
    print(f"{'Rank':4s} | {'Start':8s} | {'Setup Type':16s} | {'Max Tr':6s} | {'Cooldown':8s} | {'SL':4s} | {'BE':5s} | {'Trail':5s} | {'Win Rate':9s} | {'Net Rs':15s} | {'Max DD':12s} | {'Calmar':7s} | {'Month WR':10s} | {'Tr/Day':6s}")
    print("-" * 155)
    for r, item in enumerate(top_profit, 1):
        c = item["config"]
        st_str = "09:20 AM" if c["start_bar"] == 5 else ("09:30 AM" if c["start_bar"] == 15 else "09:35 AM")
        cd_str = f"{c['cooldown']} min" if c["cooldown"] > 0 else "None"
        max_str = f"{c['max_trades']}" if c["max_trades"] < 999 else "No Limit"
        print(f"{r:4d} | {st_str:8s} | {c['setup_type']:16s} | {max_str:6s} | {cd_str:8s} | {c['sl_m']:4.2f} | {c['be_trig']:4.2f}x | {c['trail_dist']:4.2f}x | {item['win_rate']:7.1f}% | Rs {item['net_rs']:+12.2f} | Rs {item['max_dd']:9.2f} | {item['calmar']:7.3f} | {item['month_wr']:7.1f}% ({item['pos_months']}/{len(all_months)}) | {item['trades_per_day']:6.2f}")

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "consistent_discovery_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "top_consistent": top_consistent,
        "top_profit": top_profit,
    }, indent=2, default=float), encoding="utf-8")
    print("\n" + "=" * 155)
    print(f"[Saved Consistent Discovery JSON Ledger]: {out_file}")
    print("=" * 155)


if __name__ == "__main__":
    run_discovery_sweep_gpu()
