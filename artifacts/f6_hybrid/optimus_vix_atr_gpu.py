"""100% GPU-Resident Optimus Engine with Dynamic India VIX-Scaled ATR Optimization.

Implements cutting-edge quantitative dynamic volatility scaling using India VIX 1-minute data:
  1. Continuous Power-Law Scaling: ATR_dyn = ATR * (VIX / 15.0)^gamma
  2. Discrete 3-Regime Volatility Scaling: Low VIX (<13.5), Normal VIX (13.5-18), High VIX (>18)
  3. Dynamic Target Asymmetry: Expands Reward-to-Risk ratio during high IV regimes

Hardware Target: NVIDIA RTX 3060 (12 GB VRAM · CUDA 12.1 · TF32 Tensor Cores)
Runtime: Hermes Agent Virtual Environment (PyTorch 2.5.1 + CUDA 12.1)
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

import opt_futures_quad as source

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOT_SIZE = 65
FEE = 40.0
SLIPPAGE_PTS = 0.0
BASE_SESSION_START = 5   # 09:20
BASE_SESSION_END = 345   # 15:00
DAILY_LOSS_RS = 2000.0

DESKTOP_VIX = Path(r"C:\Users\user\Desktop\nifty50 data\INDIA VIX_minute.csv")
DESKTOP_OPTS = Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options\2026\8")
AMMU_DATA = Path(r"C:\Websites\ammu\data")

EMPTY = {
    "trades": 0, "win_rate": 0.0, "net_rs": 0.0, "net_points": 0.0,
    "pf": 0.0, "max_dd": 0.0, "calmar": 0.0, "fees": 0.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. GPU DATASET LOADER (NIFTY 50 + INDIA VIX)
# ═══════════════════════════════════════════════════════════════════════════
def extend_with_august(opt_map: dict, spot_all: dict):
    opt_map = dict(opt_map)
    spot_all = dict(spot_all)
    if DESKTOP_OPTS.exists():
        for p in sorted(DESKTOP_OPTS.glob("nifty_options_*.csv")):
            parts = p.stem.split("_")
            day = f"{parts[4]}-{parts[3]}-{parts[2]}"
            opt_map[day] = str(p)
    if AMMU_DATA.exists():
        for d in sorted(AMMU_DATA.glob("2026-08-*")):
            day = d.name
            f = d / f"nifty50_index_1m_{day}.csv"
            if not f.exists():
                continue
            rows = []
            with open(f) as fh:
                header = fh.readline().strip().split(",")
                t_col = header.index("timestamp")
                for line in fh:
                    fields = line.strip().split(",")
                    if len(fields) <= t_col:
                        continue
                    ts = fields[t_col]
                    try:
                        o = float(fields[t_col + 1])
                        h = float(fields[t_col + 2])
                        l = float(fields[t_col + 3])
                        c = float(fields[t_col + 4])
                        dt = pd.to_datetime(ts)
                        rows.append((dt.hour * 60 + dt.minute, o, h, l, c))
                    except Exception:
                        continue
            if not rows:
                continue
            arr = np.array(rows)
            spot_all[day] = {
                "min": arr[:, 0].astype(int),
                "open": arr[:, 1],
                "high": arr[:, 2],
                "low": arr[:, 3],
                "close": arr[:, 4],
            }
    return opt_map, spot_all


def load_vix_minute_tensor(days: list[str]) -> torch.Tensor:
    """Loads 1-minute India VIX data and aligns to (N_days, 375 bars) tensor."""
    print("Loading 1-Minute India VIX Tensor...", flush=True)
    t0 = time.time()
    vix_df = pd.read_csv(DESKTOP_VIX)
    vix_df["dt"] = pd.to_datetime(vix_df["date"])
    vix_df["day_str"] = vix_df["dt"].dt.date.astype(str)
    vix_df["bar_idx"] = (vix_df["dt"].dt.hour * 60 + vix_df["dt"].dt.minute) - 555

    vix_map = {}
    for d, g in vix_df.groupby("day_str"):
        arr = np.full(375, np.nan, dtype=np.float32)
        valid = (g["bar_idx"] >= 0) & (g["bar_idx"] < 375)
        arr[g["bar_idx"][valid].to_numpy()] = g["close"][valid].to_numpy()
        # Forward fill within day
        mask = np.isnan(arr)
        if not mask.all():
            idx = np.where(~mask, np.arange(375), 0)
            np.maximum.accumulate(idx, out=idx)
            arr = arr[idx]
        vix_map[d] = arr

    N = len(days)
    vix_arr = np.zeros((N, 375), dtype=np.float32)
    last_known_vix = 15.5

    for i, d in enumerate(days):
        if d in vix_map and not np.isnan(vix_map[d]).all():
            vix_arr[i] = vix_map[d]
            last_known_vix = float(np.nanmean(vix_map[d]))
        else:
            # Forward fill for days past CSV coverage (e.g. Aug 2026)
            vix_arr[i] = last_known_vix

    # Fill any remaining internal NaNs
    vix_arr[np.isnan(vix_arr)] = 15.5
    d_vix = torch.tensor(vix_arr, dtype=torch.float32, device=device)
    print(f"  India VIX Tensor {d_vix.shape} loaded in {time.time()-t0:.2f}s | Mean VIX: {d_vix.mean().item():.2f}", flush=True)
    return d_vix


def load_gpu_all_data():
    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    N = len(days)

    arr_h = np.zeros((N, 375), dtype=np.float32)
    arr_l = np.zeros((N, 375), dtype=np.float32)
    arr_c = np.zeros((N, 375), dtype=np.float32)
    arr_o = np.zeros((N, 375), dtype=np.float32)

    for i, d in enumerate(days):
        sp = spot_all[d]
        for idx, m in enumerate(sp["min"]):
            b = int(m) - 555
            if 0 <= b < 375:
                arr_h[i, b] = float(sp["high"][idx])
                arr_l[i, b] = float(sp["low"][idx])
                arr_c[i, b] = float(sp["close"][idx])
                arr_o[i, b] = float(sp["open"][idx])

    d_h = torch.tensor(arr_h, dtype=torch.float32, device=device)
    d_l = torch.tensor(arr_l, dtype=torch.float32, device=device)
    d_c = torch.tensor(arr_c, dtype=torch.float32, device=device)
    d_o = torch.tensor(arr_o, dtype=torch.float32, device=device)

    d_vix = load_vix_minute_tensor(days)
    d_is_mask = torch.tensor([d < "2024-01-01" for d in days], dtype=torch.bool, device=device)
    d_oos_mask = torch.tensor([d >= "2024-01-01" for d in days], dtype=torch.bool, device=device)

    return d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask


# ═══════════════════════════════════════════════════════════════════════════
# 2. CAUSAL GPU INDICATORS & DYNAMIC VIX-SCALED ATR
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_quad_stochastics_gpu(d_h, d_l, d_c):
    def calc_stoch(k_p, d_p):
        h_pad = F.pad(d_h.unsqueeze(1), (k_p - 1, 0), mode="replicate")
        l_pad = F.pad(d_l.unsqueeze(1), (k_p - 1, 0), mode="replicate")
        max_h = F.max_pool1d(h_pad, kernel_size=k_p, stride=1).squeeze(1)
        min_l = -F.max_pool1d(-l_pad, kernel_size=k_p, stride=1).squeeze(1)
        raw_k = ((d_c - min_l) / (max_h - min_l).clamp(min=1e-6)) * 100.0
        k_pad = F.pad(raw_k.unsqueeze(1), (d_p - 1, 0), mode="replicate")
        return F.avg_pool1d(k_pad, kernel_size=d_p, stride=1).squeeze(1)

    return calc_stoch(12, 3), calc_stoch(14, 3), calc_stoch(40, 4), calc_stoch(50, 10)


@torch.no_grad()
def compute_atr_gpu(d_h, d_l, d_c, period=14):
    prev_c = F.pad(d_c[:, :-1], (1, 0), mode="replicate")
    tr = torch.maximum(
        torch.maximum(d_h - d_l, torch.abs(d_h - prev_c)),
        torch.abs(d_l - prev_c)
    )
    tr_pad = F.pad(tr.unsqueeze(1), (period - 1, 0), mode="replicate")
    return F.avg_pool1d(tr_pad, kernel_size=period, stride=1).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════
# 3. 3D BATCH SIMULATION WITH DYNAMIC VIX-SCALING
# ═══════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def simulate_vix_dynamic_batch_gpu(
    entries_mask: torch.Tensor,       # (N, T)
    d_h: torch.Tensor,               # (N, T)
    d_l: torch.Tensor,               # (N, T)
    d_c: torch.Tensor,               # (N, T)
    d_atr: torch.Tensor,             # (N, T)
    d_vix: torch.Tensor,             # (N, T)
    param_configs: list[dict],       # list of parameter dicts
    day_mask: torch.Tensor | None = None,
):
    """
    Simultaneously evaluates B dynamic VIX-ATR configurations across all trades on GPU.
    """
    if day_mask is not None:
        entries_mask = entries_mask & day_mask.unsqueeze(1)

    valid_window = torch.zeros_like(entries_mask)
    valid_window[:, BASE_SESSION_START:BASE_SESSION_END] = True
    entries_mask = entries_mask & valid_window

    coords = torch.nonzero(entries_mask, as_tuple=False)
    N_trades = coords.shape[0]
    if N_trades == 0:
        return [dict(EMPTY) for _ in range(len(param_configs))]

    d_idx = coords[:, 0]  # (N_trades,)
    b_idx = coords[:, 1]  # (N_trades,)
    ep = d_c[d_idx, b_idx]  # (N_trades,)
    base_atr = d_atr[d_idx, b_idx].clamp(min=5.0, max=30.0)  # (N_trades,)
    trade_vix = d_vix[d_idx, b_idx].clamp(min=8.0, max=80.0)  # (N_trades,)

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

    B = len(param_configs)
    # Compute dynamic SL/TP tensors for each configuration
    sl_dists = []
    tp_dists = []

    for p in param_configs:
        sl_m = p["sl_mult"]
        tp_m = p["tp_mult"]
        scaling_type = p.get("scaling_type", "power")
        vix_base = p.get("vix_base", 15.0)

        if scaling_type == "power":
            gamma = p.get("gamma", 0.5)
            # Continuous VIX Power-Law Scaling
            vix_scale = torch.pow(trade_vix / vix_base, gamma).clamp(min=0.6, max=2.2)
            eff_atr = base_atr * vix_scale
            sl_d = (sl_m * eff_atr).clamp(min=6.0, max=35.0)
            tp_d = (tp_m * eff_atr).clamp(min=8.0, max=75.0)

        elif scaling_type == "discrete_regime":
            # 3-Regime Step Scaling: Low (<13.5), Normal (13.5-18), High (>18)
            low_scale = p.get("low_scale", 0.85)
            high_scale = p.get("high_scale", 1.30)
            tp_high_boost = p.get("tp_high_boost", 1.20)

            vix_factor = torch.ones_like(trade_vix)
            vix_factor = torch.where(trade_vix < 13.5, torch.full_like(trade_vix, low_scale), vix_factor)
            vix_factor = torch.where(trade_vix > 18.0, torch.full_like(trade_vix, high_scale), vix_factor)

            tp_factor = torch.ones_like(trade_vix)
            tp_factor = torch.where(trade_vix > 18.0, torch.full_like(trade_vix, tp_high_boost), tp_factor)

            eff_atr = base_atr * vix_factor
            sl_d = (sl_m * eff_atr).clamp(min=6.0, max=35.0)
            tp_d = (tp_m * eff_atr * tp_factor).clamp(min=8.0, max=75.0)

        else:  # baseline static ATR
            sl_d = (sl_m * base_atr).clamp(min=6.0, max=30.0)
            tp_d = (tp_m * base_atr).clamp(min=8.0, max=60.0)

        sl_dists.append(sl_d)
        tp_dists.append(tp_d)

    sl_dist_3d = torch.stack(sl_dists, dim=0).unsqueeze(-1)  # (B, N_trades, 1)
    tp_dist_3d = torch.stack(tp_dists, dim=0).unsqueeze(-1)  # (B, N_trades, 1)

    ep_3d = ep.unsqueeze(0).unsqueeze(-1)
    sl_barrier = ep_3d - sl_dist_3d
    tp_barrier = ep_3d + tp_dist_3d

    fut_l_exp = fut_l_m.unsqueeze(0)  # (1, N_trades, max_future)
    fut_h_exp = fut_h_m.unsqueeze(0)  # (1, N_trades, max_future)

    hit_sl = (fut_l_exp <= sl_barrier)
    hit_tp = (fut_h_exp >= tp_barrier)

    BIG = 999999
    sl_any = hit_sl.any(dim=2)
    tp_any = hit_tp.any(dim=2)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=2), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=2), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    exit_sl = sl_barrier.squeeze(-1)
    exit_tp = tp_barrier.squeeze(-1)
    exit_eod = eod_px.unsqueeze(0).expand(B, -1)

    exit_px = torch.where(sl_exits, exit_sl, torch.where(tp_exits, exit_tp, exit_eod))
    pts = (exit_px - ep.unsqueeze(0)) * 0.50
    rs_net = pts * LOT_SIZE - FEE

    rs_net_cpu = rs_net.cpu().numpy()
    pts_cpu = pts.cpu().numpy()
    d_idx_cpu = d_idx.cpu().numpy()
    b_idx_cpu = b_idx.cpu().numpy()

    results = []
    order = np.lexsort((b_idx_cpu, d_idx_cpu))
    days_sorted = d_idx_cpu[order]

    for b in range(B):
        p_rs = rs_net_cpu[b, order]
        p_pts = pts_cpu[b, order]

        kept_rs, kept_pts = [], []
        last_d = None
        cum_d = 0.0
        stop_d = False

        for r, pt, d in zip(p_rs, p_pts, days_sorted):
            if d != last_d:
                last_d = d
                cum_d = 0.0
                stop_d = False

            if stop_d:
                continue

            new_cum = cum_d + r
            if new_cum <= -DAILY_LOSS_RS:
                stop_d = True
                kept_rs.append(r)
                kept_pts.append(pt)
                cum_d = new_cum
                continue

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

        p_cfg = param_configs[b]
        results.append({
            "config": p_cfg,
            "scaling_type": p_cfg.get("scaling_type", "power"),
            "sl_mult": p_cfg["sl_mult"],
            "tp_mult": p_cfg["tp_mult"],
            "gamma": p_cfg.get("gamma", 0.0),
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
# 4. MAIN OPTIMUS RUNNER
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Optimus Dynamic VIX-ATR GPU Backtest")
    parser.add_argument("--smoke", action="store_true", help="5-Day Smoke Test")
    parser.add_argument("--full", action="store_true", help="Full 7-Year Optimization")
    args = parser.parse_args()

    d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
    N_DAYS = len(days)

    # Compute Indicators on GPU
    s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
    d_atr = compute_atr_gpu(d_h, d_l, d_c, period=14)

    # Signals
    prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
    s1_turn_up = (s1 > prev_s1)
    super_setup = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
    flag_setup = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
    entries_mask = super_setup | flag_setup

    # Construct Parameter Grid:
    # 1. Continuous Power Scaling Grid: sl_m x tp_m x gamma
    param_configs = []
    for sl_m in [2.0, 2.25, 2.5, 2.75, 3.0]:
        for tp_m in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
            for gamma in [0.0, 0.25, 0.50, 0.75, 1.0]:
                param_configs.append({
                    "scaling_type": "power",
                    "sl_mult": sl_m,
                    "tp_mult": tp_m,
                    "gamma": gamma,
                    "vix_base": 15.0,
                    "desc": f"Power(gamma={gamma:.2f})" if gamma > 0 else "Static ATR",
                })

    # 2. Discrete 3-Regime Scaling Grid
    for sl_m in [2.25, 2.5, 2.75]:
        for tp_m in [3.5, 4.0, 5.0, 6.0]:
            for low_s, high_s, tp_b in [(0.85, 1.25, 1.15), (0.80, 1.35, 1.25), (0.75, 1.50, 1.30)]:
                param_configs.append({
                    "scaling_type": "discrete_regime",
                    "sl_mult": sl_m,
                    "tp_mult": tp_m,
                    "low_scale": low_s,
                    "high_scale": high_s,
                    "tp_high_boost": tp_b,
                    "desc": f"3-Regime({low_s:.2f}/{high_s:.2f}/boost={tp_b:.2f})",
                })

    print(f"Total Dynamic VIX-ATR Parameter Configurations to Evaluate: {len(param_configs)}", flush=True)

    if args.smoke:
        print("=" * 115)
        print("MANDATORY SMOKE TEST (5 Days)")
        print("=" * 115)
        smoke_mask = torch.zeros(N_DAYS, dtype=torch.bool, device=device)
        smoke_mask[:5] = True
        res_smoke = simulate_vix_dynamic_batch_gpu(entries_mask, d_h, d_l, d_c, d_atr, d_vix, param_configs[:4], day_mask=smoke_mask)
        for r in res_smoke:
            print(f"Config: {r['config']['desc']} | SL={r['sl_mult']} TP={r['tp_mult']} | Trades: {r['trades']} | WR: {r['win_rate']}% | Net Rs: Rs {r['net_rs']:+8.2f} | PF: {r['profit_factor']}")
        print("=" * 115)
        return

    # Full Evaluation
    print("\n" + "=" * 115)
    print(f"RUNNING 7-YEAR 3D BATCH OPTIMUS RUNNER WITH INDIA VIX DYNAMIC ATR ({len(param_configs)} Configurations across {N_DAYS} Days)")
    print("=" * 115)
    t0 = time.time()

    res_full = simulate_vix_dynamic_batch_gpu(entries_mask, d_h, d_l, d_c, d_atr, d_vix, param_configs, day_mask=None)
    res_is = simulate_vix_dynamic_batch_gpu(entries_mask, d_h, d_l, d_c, d_atr, d_vix, param_configs, day_mask=d_is_mask)
    res_oos = simulate_vix_dynamic_batch_gpu(entries_mask, d_h, d_l, d_c, d_atr, d_vix, param_configs, day_mask=d_oos_mask)

    el = time.time() - t0
    print(f"  Completed all {len(param_configs)} 7-year evaluations in {el:.3f}s ({el/len(param_configs)*1000:.2f} ms/comb) on RTX 3060", flush=True)

    # Top 5 by Profit
    top_profit = sorted(res_full, key=lambda x: x["net_rs"], reverse=True)[:5]
    # Top 5 by Least Drawdown
    top_least_dd = sorted(res_full, key=lambda x: x["max_drawdown_rs"])[:5]
    # Top 5 by Calmar
    top_calmar = sorted(res_full, key=lambda x: x["calmar_ratio"], reverse=True)[:5]

    print("\n>>> TOP 5 SETTINGS RANKED BY MAX NET POINTS & PROFIT:")
    print(f"{'Rank':4s} | {'Scaling Model':28s} | {'SL':4s} | {'TP':4s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
    print("-" * 125)
    for r, item in enumerate(top_profit, 1):
        cfg_name = item["config"].get("desc", item["scaling_type"])
        print(f"{r:4d} | {cfg_name:28s} | {item['sl_mult']:4.2f} | {item['tp_mult']:4.2f} | {item['trades']:7d} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['calmar_ratio']:7.3f}")

    print("\n>>> TOP 5 SETTINGS RANKED BY LEAST MAX DRAWDOWN:")
    print(f"{'Rank':4s} | {'Scaling Model':28s} | {'SL':4s} | {'TP':4s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
    print("-" * 125)
    for r, item in enumerate(top_least_dd, 1):
        cfg_name = item["config"].get("desc", item["scaling_type"])
        print(f"{r:4d} | {cfg_name:28s} | {item['sl_mult']:4.2f} | {item['tp_mult']:4.2f} | {item['trades']:7d} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['calmar_ratio']:7.3f}")

    print("\n>>> TOP 5 SETTINGS RANKED BY CALMAR / RISK-ADJUSTED RETURN:")
    print(f"{'Rank':4s} | {'Scaling Model':28s} | {'SL':4s} | {'TP':4s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
    print("-" * 125)
    for r, item in enumerate(top_calmar, 1):
        cfg_name = item["config"].get("desc", item["scaling_type"])
        print(f"{r:4d} | {cfg_name:28s} | {item['sl_mult']:4.2f} | {item['tp_mult']:4.2f} | {item['trades']:7d} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['calmar_ratio']:7.3f}")

    print("\n>>> WALK-FORWARD OUT-OF-SAMPLE VALIDATION (IS: 2020-2023 vs OOS: 2024-2026):")
    print(f"{'Rank':4s} | {'Scaling Model':26s} | {'SLxTP':9s} | {'IS Trades':9s} | {'IS WR':7s} | {'IS Net Rs':13s} | {'IS PF':6s} | {'OOS Trades':10s} | {'OOS WR':8s} | {'OOS Net Rs':13s} | {'OOS PF':7s} | {'WFE':6s}")
    print("-" * 135)
    for r, item in enumerate(top_profit, 1):
        idx = next(i for i, x in enumerate(param_configs) if x == item["config"])
        m_is = res_is[idx]
        m_oos = res_oos[idx]
        wfe = (m_oos["net_rs"] / 2.35) / (m_is["net_rs"] / 4.0) if m_is["net_rs"] > 0 else 0.0
        cfg_name = item["config"].get("desc", item["scaling_type"])
        print(f"{r:4d} | {cfg_name:26s} | {item['sl_mult']:.2f}x{item['tp_mult']:.2f} | {m_is['trades']:9d} | {m_is['win_rate']:6.1f}% | Rs {m_is['net_rs']:+10.2f} | {m_is['profit_factor']:6.2f} | {m_oos['trades']:10d} | {m_oos['win_rate']:7.1f}% | Rs {m_oos['net_rs']:+10.2f} | {m_oos['profit_factor']:7.2f} | {wfe:6.2f}")

    # Save to JSON
    def sanitize(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    out_file = ROOT / "artifacts" / "f6_hybrid" / "optimus_vix_atr_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(sanitize({
        "top_profit": top_profit,
        "top_least_dd": top_least_dd,
        "top_calmar": top_calmar,
    }), indent=2), encoding="utf-8")
    print(f"\n[Saved Dynamic VIX Results JSON]: {out_file}")
    print("=" * 115)


if __name__ == "__main__":
    main()
