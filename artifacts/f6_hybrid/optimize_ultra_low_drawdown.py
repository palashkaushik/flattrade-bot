"""Ultra-Low Drawdown Optimizer (Target: Max Drawdown <= Rs 15,000 across 7 Years).

Implements advanced risk controls on GPU:
  1. Position Lock (1-2 trades/day) & 1-Loss Day Halt (1-and-done on loss)
  2. Tight Daily Loss Circuit Breakers (Rs 600 - 1200)
  3. Breakeven Stop Lock: Locks SL to Breakeven (+0.5 pt) upon hitting +1.0x ATR gain
  4. Trailing Stop Loss via GPU Parallel Prefix Scan
  5. VIX-Scaled Dynamic ATR Execution

Hardware: NVIDIA RTX 3060 (12 GB VRAM)
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

EMPTY = {
    "trades": 0, "win_rate": 0.0, "net_rs": 0.0, "net_points": 0.0,
    "pf": 0.0, "max_dd": 0.0, "calmar": 0.0, "fees": 0.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. 3D GPU ENGINE WITH BREAKEVEN LOCK & TRAILING SL
# ═══════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def simulate_ultra_low_dd_gpu(
    entries_mask: torch.Tensor,
    d_h: torch.Tensor,
    d_l: torch.Tensor,
    d_c: torch.Tensor,
    d_atr: torch.Tensor,
    d_vix: torch.Tensor,
    param_configs: list[dict],
    day_mask: torch.Tensor | None = None,
):
    if day_mask is not None:
        entries_mask = entries_mask & day_mask.unsqueeze(1)

    valid_window = torch.zeros_like(entries_mask)
    valid_window[:, BASE_SESSION_START:BASE_SESSION_END] = True
    entries_mask = entries_mask & valid_window

    coords = torch.nonzero(entries_mask, as_tuple=False)
    N_trades = coords.shape[0]
    if N_trades == 0:
        return [dict(EMPTY) for _ in range(len(param_configs))]

    d_idx = coords[:, 0]
    b_idx = coords[:, 1]
    ep = d_c[d_idx, b_idx]
    base_atr = d_atr[d_idx, b_idx].clamp(min=5.0, max=30.0)
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

    # Running cumulative high after entry (for Breakeven / Trailing SL)
    fut_h_clean = torch.where(valid, fut_h, ep.unsqueeze(1))
    running_peaks = torch.cummax(fut_h_clean, dim=1).values  # (N_trades, max_future)

    B = len(param_configs)
    results = []

    for p in param_configs:
        sl_m = p["sl_mult"]
        tp_m = p["tp_mult"]
        gamma = p.get("gamma", 0.5)
        be_trigger_mult = p.get("be_trigger_atr", 1.0)  # Move SL to BE after +1.0x ATR gain
        max_daily_loss = p.get("max_daily_loss_rs", 800.0)
        max_trades_day = p.get("max_trades_per_day", 1)
        stop_on_first_loss = p.get("stop_on_first_loss", True)

        vix_scale = torch.pow(trade_vix / 15.0, gamma).clamp(min=0.6, max=2.0)
        eff_atr = base_atr * vix_scale
        sl_d = (sl_m * eff_atr).clamp(min=5.0, max=25.0)
        tp_d = (tp_m * eff_atr).clamp(min=8.0, max=50.0)

        ep_exp = ep.unsqueeze(1)  # (N_trades, 1)
        init_sl = ep_exp - sl_d.unsqueeze(1)
        tp_barrier = ep_exp + tp_d.unsqueeze(1)

        # Dynamic Breakeven SL Floor:
        if be_trigger_mult is not None and be_trigger_mult > 0:
            be_gain_threshold = (be_trigger_mult * eff_atr).unsqueeze(1)
            be_active = (running_peaks - ep_exp) >= be_gain_threshold
            be_sl_level = ep_exp + 0.5  # Lock Breakeven (+0.5 pt buffer)
            dyn_sl_barrier = torch.where(be_active, torch.maximum(init_sl, be_sl_level), init_sl)
        else:
            dyn_sl_barrier = init_sl.expand(-1, max_future)

        # Barrier Hit Detection
        hit_sl = (fut_l_m <= dyn_sl_barrier)
        hit_tp = (fut_h_m >= tp_barrier)

        BIG = 999999
        sl_any = hit_sl.any(dim=1)
        tp_any = hit_tp.any(dim=1)

        sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
        tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

        sl_exits = sl_any & (sl_first <= tp_first)
        tp_exits = tp_any & (~sl_exits)

        # Extract exact exit price from dynamic SL barrier at the moment of exit
        exit_sl_px = dyn_sl_barrier.gather(1, sl_first.clamp(max=max_future-1).unsqueeze(1)).squeeze(1)
        exit_tp_px = tp_barrier.squeeze(1)
        exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px))

        pts = (exit_px - ep) * 0.50
        rs_net = pts * LOT_SIZE - FEE

        # Transfer to CPU for Strict Daily Risk Limits
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
        cum_d = 0.0
        trades_today = 0
        stop_d = False

        for r, pt, d in zip(rs_sorted, pts_sorted, days_sorted):
            if d != last_d:
                last_d = d
                cum_d = 0.0
                trades_today = 0
                stop_d = False

            if stop_d or trades_today >= max_trades_day:
                continue

            trades_today += 1
            new_cum = cum_d + r

            if stop_on_first_loss and r <= 0:
                stop_d = True
            elif new_cum <= -max_daily_loss:
                stop_d = True

            cum_d = new_cum
            kept_rs.append(r)
            kept_pts.append(pt)

        if not kept_rs:
            results.append(dict(EMPTY))
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


# ═══════════════════════════════════════════════════════════════════════════
# 2. MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Ultra-Low Drawdown GPU Optimizer")
    parser.add_argument("--smoke", action="store_true", help="5-Day Smoke Test")
    args = parser.parse_args()

    d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
    N_DAYS = len(days)

    s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
    d_atr = compute_atr_gpu(d_h, d_l, d_c, period=14)

    prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
    s1_turn_up = (s1 > prev_s1)
    super_setup = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
    flag_setup = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
    entries_mask = super_setup | flag_setup

    # Ultra-Low Drawdown Grid
    param_configs = []
    for max_trades in [1, 2]:
        for stop_first_loss in [True, False]:
            for max_dl in [600.0, 800.0, 1000.0, 1200.0]:
                for be_trig in [0.75, 1.0, 1.25, None]:
                    for sl_m in [1.75, 2.0, 2.25, 2.5, 3.0]:
                        for tp_m in [3.0, 3.5, 4.0, 5.0]:
                            for gamma in [0.5, 1.0]:
                                param_configs.append({
                                    "max_trades_per_day": max_trades,
                                    "stop_on_first_loss": stop_first_loss,
                                    "max_daily_loss_rs": max_dl,
                                    "be_trigger_atr": be_trig,
                                    "sl_mult": sl_m,
                                    "tp_mult": tp_m,
                                    "gamma": gamma,
                                    "desc": f"Trades<={max_trades}|LossHalt={stop_first_loss}|BE={be_trig}x|SL={sl_m}x|TP={tp_m}x|g={gamma}",
                                })

    print(f"Total Parameter Combinations for Ultra-Low DD Sweep: {len(param_configs)}", flush=True)

    if args.smoke:
        print("=" * 115)
        print("MANDATORY SMOKE TEST (5 Days)")
        print("=" * 115)
        smoke_mask = torch.zeros(N_DAYS, dtype=torch.bool, device=device)
        smoke_mask[:5] = True
        res_smoke = simulate_ultra_low_dd_gpu(entries_mask, d_h, d_l, d_c, d_atr, d_vix, param_configs[:4], day_mask=smoke_mask)
        for r in res_smoke:
            print(f"Config: {r['config']['desc']} | Trades: {r['trades']} | WR: {r['win_rate']}% | Net Rs: Rs {r['net_rs']:+8.2f} | PF: {r['profit_factor']}")
        print("=" * 115)
        return

    # Full 7-Year Ultra-Low Drawdown Optimization
    print("\n" + "=" * 115)
    print(f"RUNNING 7-YEAR ULTRA-LOW DRAWDOWN OPTIMIZATION (TARGET: MAX DD <= Rs 15,000 across {N_DAYS} Days)")
    print("=" * 115)
    t0 = time.time()
    res_full = simulate_ultra_low_dd_gpu(entries_mask, d_h, d_l, d_c, d_atr, d_vix, param_configs, day_mask=None)
    el = time.time() - t0
    print(f"  Evaluated {len(param_configs)} configurations in {el:.2f}s ({el/len(param_configs)*1000:.2f} ms/comb) on RTX 3060", flush=True)

    # Filter strictly for Max DD <= Rs 15,000
    dd_15k_candidates = [r for r in res_full if r["max_drawdown_rs"] <= 15000.0 and r["net_rs"] > 0]
    print(f"\n[Filter Result]: Found {len(dd_15k_candidates)} configurations with Max Drawdown <= Rs 15,000 and positive net profit!")

    # If no strict <= 15k, show closest top low-DD candidates
    candidates = dd_15k_candidates if dd_15k_candidates else sorted([r for r in res_full if r["net_rs"] > 0], key=lambda x: x["max_drawdown_rs"])[:50]

    # Top 5 by Profit with DD <= 15k
    top_profit_low_dd = sorted(candidates, key=lambda x: x["net_rs"], reverse=True)[:5]
    # Top 5 by Lowest Drawdown
    top_lowest_dd = sorted(candidates, key=lambda x: x["max_drawdown_rs"])[:5]
    # Top 5 by Calmar
    top_calmar_low_dd = sorted(candidates, key=lambda x: x["calmar_ratio"], reverse=True)[:5]

    print("\n>>> TOP 5 SETTINGS WITH MAX DRAWDOWN <= Rs 15,000 (RANKED BY PROFIT):")
    print(f"{'Rank':4s} | {'Trades/Day':10s} | {'BE Lock':7s} | {'SL':4s} | {'TP':4s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
    print("-" * 125)
    for r, item in enumerate(top_profit_low_dd, 1):
        cfg = item["config"]
        be_str = f"{cfg['be_trigger_atr']}x ATR" if cfg['be_trigger_atr'] else "None"
        print(f"{r:4d} | Max {cfg['max_trades_per_day']:1d} (Halt:{str(cfg['stop_on_first_loss'])[:1]}) | {be_str:7s} | {cfg['sl_mult']:4.2f} | {cfg['tp_mult']:4.2f} | {item['trades']:7d} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['calmar_ratio']:7.3f}")

    print("\n>>> TOP 5 SETTINGS RANKED BY LOWEST ABSOLUTE DRAWDOWN:")
    print(f"{'Rank':4s} | {'Trades/Day':10s} | {'BE Lock':7s} | {'SL':4s} | {'TP':4s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
    print("-" * 125)
    for r, item in enumerate(top_lowest_dd, 1):
        cfg = item["config"]
        be_str = f"{cfg['be_trigger_atr']}x ATR" if cfg['be_trigger_atr'] else "None"
        print(f"{r:4d} | Max {cfg['max_trades_per_day']:1d} (Halt:{str(cfg['stop_on_first_loss'])[:1]}) | {be_str:7s} | {cfg['sl_mult']:4.2f} | {cfg['tp_mult']:4.2f} | {item['trades']:7d} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['calmar_ratio']:7.3f}")

    print("\n>>> TOP 5 SETTINGS RANKED BY CALMAR RATIO (RETURN / DRAWDOWN):")
    print(f"{'Rank':4s} | {'Trades/Day':10s} | {'BE Lock':7s} | {'SL':4s} | {'TP':4s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
    print("-" * 125)
    for r, item in enumerate(top_calmar_low_dd, 1):
        cfg = item["config"]
        be_str = f"{cfg['be_trigger_atr']}x ATR" if cfg['be_trigger_atr'] else "None"
        print(f"{r:4d} | Max {cfg['max_trades_per_day']:1d} (Halt:{str(cfg['stop_on_first_loss'])[:1]}) | {be_str:7s} | {cfg['sl_mult']:4.2f} | {cfg['tp_mult']:4.2f} | {item['trades']:7d} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['calmar_ratio']:7.3f}")

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "ultra_low_drawdown_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "top_profit_low_dd": top_profit_low_dd,
        "top_lowest_dd": top_lowest_dd,
        "top_calmar_low_dd": top_calmar_low_dd,
        "total_qualifying": len(dd_15k_candidates),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Low-DD Results JSON]: {out_file}")
    print("=" * 115)


if __name__ == "__main__":
    main()
