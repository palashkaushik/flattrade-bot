"""
COMPREHENSIVE MULTI-STRATEGY GPU OPTUNA STUDY (100 TRIALS PER STRATEGY)
=======================================================================
Executes 100 Bayesian TPE trials per strategy family (500 trials total) across
the full 7-Year Dataset (1,574 days) on NVIDIA GeForce RTX 3060:

  1. Family 1: S1 Turn-Up + Trailing SL (Unlimited Profit Baseline)
  2. Family 2: Marni F6 Cross-Filter (Flag Immediate + ATR SL/TP)
  3. Family 3: 15m Elder Impulse HTF Gated Stochastics
  4. Family 4: Pure Marni VSA Golden Pocket Retracement
  5. Family 5: Adaptive ATR Dynamic Multiplier Breakout

Calculates Risk-Adjusted Quality Score: PF * (WR / 40.0) - 0.20 * (MaxDD / NetProfit).
Saves comprehensive ledger to artifacts/f6_hybrid/multi_strategy_study_results.json.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost

LOT_SIZE = 65
SESSION_START_IDX = 5
SESSION_END_IDX = 345
TRIALS_PER_FAMILY = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"CUDA Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

# 1. Load GPU Datasets
def load_gpu_data(start_date="2020-01-01", end_date="2026-05-05"):
    spot_all = source.load_spot()
    opt_map = source.option_day_files(start_date, end_date)
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

    prev_c = F.pad(d_c[:, :-1], (1, 0), mode="replicate")
    d_tr = torch.maximum(torch.maximum(d_h - d_l, torch.abs(d_h - prev_c)), torch.abs(d_l - prev_c))
    return d_h, d_l, d_c, d_o, d_tr, days

print("Pre-loading 7-Year Dataset into GPU VRAM...", flush=True)
t_load_0 = time.time()
d_high, d_low, d_close, d_open, d_tr, all_days = load_gpu_data()
print(f"Loaded {len(all_days)} days into GPU VRAM in {time.time()-t_load_0:.2f}s — Shape: {d_close.shape}", flush=True)

# Precomputed Causal Indicator Helpers
@torch.no_grad()
def get_stoch(k_period, d_period=3):
    h_pad = F.pad(d_high.unsqueeze(1), (k_period - 1, 0), mode="replicate")
    l_pad = F.pad(d_low.unsqueeze(1), (k_period - 1, 0), mode="replicate")
    max_h = F.max_pool1d(h_pad, kernel_size=k_period, stride=1).squeeze(1)
    min_l = -F.max_pool1d(-l_pad, kernel_size=k_period, stride=1).squeeze(1)
    denom = (max_h - min_l).clamp(min=1e-6)
    k = ((d_close - min_l) / denom) * 100.0
    return k

@torch.no_grad()
def get_atr(period=14):
    tr_pad = F.pad(d_tr.unsqueeze(1), (period - 1, 0), mode="replicate")
    return F.avg_pool1d(tr_pad, kernel_size=period, stride=1).squeeze(1)

# Fast Vectorized Exit Simulation
@torch.no_grad()
def simulate_signals(entries_mask, sl_tensor, tp_tensor, is_trailing=False, trail_trigger=10.0, trail_step=5.0):
    entry_coords = torch.nonzero(entries_mask, as_tuple=False)
    if entry_coords.shape[0] == 0:
        return {"trades": 0, "win_rate": 0.0, "net_pts": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    pnl_pts = []
    wins = 0
    for i in range(min(entry_coords.shape[0], 5000)):
        d_i, b_i = int(entry_coords[i, 0]), int(entry_coords[i, 1])
        ep = float(d_close[d_i, b_i])
        cur_sl = float(sl_tensor[d_i, b_i])
        cur_tp = float(tp_tensor[d_i, b_i]) if tp_tensor is not None else ep + 9999.0

        fut_h = d_high[d_i, b_i+1:SESSION_END_IDX]
        fut_l = d_low[d_i, b_i+1:SESSION_END_IDX]
        if fut_h.shape[0] == 0: continue

        exit_px = float(d_close[d_i, SESSION_END_IDX-1])
        for bar in range(fut_h.shape[0]):
            h_bar = float(fut_h[bar])
            l_bar = float(fut_l[bar])

            if is_trailing:
                gain = h_bar - ep
                if gain >= trail_trigger:
                    levels = int(gain // trail_trigger)
                    cur_sl = max(cur_sl, ep + (levels * trail_step) - (trail_trigger - trail_step))

            if l_bar <= cur_sl:
                exit_px = cur_sl
                break
            if h_bar >= cur_tp:
                exit_px = cur_tp
                break

        pts = (exit_px - ep) * 0.50
        if pts > 0: wins += 1
        pnl_pts.append(pts)

    if not pnl_pts:
        return {"trades": 0, "win_rate": 0.0, "net_pts": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    pts_t = torch.tensor(pnl_pts, dtype=torch.float32)
    rs_t = pts_t * LOT_SIZE - 30.0  # Fees & slippage
    pos_rs = rs_t[rs_t > 0].sum().item()
    neg_rs = abs(rs_t[rs_t <= 0].sum().item())
    pf = (pos_rs / neg_rs) if neg_rs > 0 else 0.0
    equity = torch.cumsum(rs_t, dim=0)
    peak = torch.cummax(equity, dim=0).values
    max_dd = float(torch.max(peak - equity))

    return {
        "trades": len(pnl_pts),
        "win_rate": round(wins / len(pnl_pts) * 100.0, 2),
        "net_pts": round(float(pts_t.sum()), 2),
        "net_rs": round(float(rs_t.sum()), 2),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 2),
    }

# ==============================================================================
# STRATEGY FAMILY OBJECTIVE FUNCTIONS
# ==============================================================================

# Family 1: S1 Turn-Up Trailing SL
def objective_family_1(trial):
    s1_k = trial.suggest_int("s1_k", 7, 14)
    s4_k = trial.suggest_int("s4_k", 50, 75, step=5)
    s4_ob = trial.suggest_float("s4_ob", 75.0, 85.0, step=2.5)
    s1_os = trial.suggest_float("s1_os", 15.0, 25.0, step=2.5)
    init_sl_pts = trial.suggest_float("init_sl_pts", 10.0, 25.0, step=2.5)
    trail_trig = trial.suggest_float("trail_trig", 8.0, 15.0, step=1.0)
    trail_step = trial.suggest_float("trail_step", 4.0, 8.0, step=1.0)

    k1 = get_stoch(s1_k)
    k4 = get_stoch(s4_k)
    k1_prev = F.pad(k1[:, :-1], (1, 0), mode="replicate")
    turn_up = k1 > k1_prev

    valid_window = torch.zeros_like(k1, dtype=torch.bool)
    valid_window[:, SESSION_START_IDX:SESSION_END_IDX] = True
    entries = (k4 >= s4_ob) & (k1 <= s1_os) & turn_up & valid_window

    sl_tensor = d_close - init_sl_pts
    res = simulate_signals(entries, sl_tensor, tp_tensor=None, is_trailing=True, trail_trigger=trail_trig, trail_step=trail_step)
    if res["trades"] < 50: raise optuna.TrialPruned("Too few trades")

    score = res["pf"] * (res["win_rate"] / 40.0) - (0.20 * (res["max_dd"] / max(res["net_rs"], 1.0)))
    for k, v in res.items(): trial.set_user_attr(k, v)
    return score

# Family 2: Marni F6 Cross-Filter (ATR SL/TP)
def objective_family_2(trial):
    s1_k = trial.suggest_int("s1_k", 7, 14)
    s4_k = trial.suggest_int("s4_k", 50, 75, step=5)
    s4_ob = trial.suggest_float("s4_ob", 75.0, 85.0, step=2.5)
    s1_os = trial.suggest_float("s1_os", 15.0, 25.0, step=2.5)
    atr_period = trial.suggest_int("atr_period", 10, 20, step=2)
    sl_mult = trial.suggest_float("sl_mult", 1.2, 2.5, step=0.1)
    tp_mult = trial.suggest_float("tp_mult", 3.0, 6.0, step=0.25)
    if tp_mult < 1.5 * sl_mult: raise optuna.TrialPruned("R:R < 1.5")

    k1 = get_stoch(s1_k)
    k4 = get_stoch(s4_k)
    atr = get_atr(atr_period)

    valid_window = torch.zeros_like(k1, dtype=torch.bool)
    valid_window[:, SESSION_START_IDX:SESSION_END_IDX] = True
    entries = (k4 >= s4_ob) & (k1 <= s1_os) & valid_window

    sl_tensor = d_close - (atr * sl_mult)
    tp_tensor = d_close + (atr * tp_mult)
    res = simulate_signals(entries, sl_tensor, tp_tensor=tp_tensor, is_trailing=False)
    if res["trades"] < 50: raise optuna.TrialPruned("Too few trades")

    score = res["pf"] * (res["win_rate"] / 40.0) - (0.20 * (res["max_dd"] / max(res["net_rs"], 1.0)))
    for k, v in res.items(): trial.set_user_attr(k, v)
    return score

# Family 3: 15m Elder Impulse HTF Gated Stochastics
def objective_family_3(trial):
    s1_k = trial.suggest_int("s1_k", 7, 14)
    s4_k = trial.suggest_int("s4_k", 50, 75, step=5)
    ema_period = trial.suggest_int("ema_period", 10, 30, step=5)
    atr_period = trial.suggest_int("atr_period", 10, 20, step=2)
    sl_mult = trial.suggest_float("sl_mult", 1.2, 2.5, step=0.1)
    tp_mult = trial.suggest_float("tp_mult", 3.0, 6.0, step=0.25)

    k1 = get_stoch(s1_k)
    k4 = get_stoch(s4_k)
    atr = get_atr(atr_period)

    alpha = 2.0 / (ema_period + 1)
    ema_htf = torch.zeros_like(d_close)
    ema_htf[:, 0] = d_close[:, 0]
    for t in range(1, d_close.shape[1]):
        ema_htf[:, t] = alpha * d_close[:, t] + (1 - alpha) * ema_htf[:, t-1]

    htf_bull = d_close >= ema_htf
    valid_window = torch.zeros_like(k1, dtype=torch.bool)
    valid_window[:, SESSION_START_IDX:SESSION_END_IDX] = True
    entries = htf_bull & (k4 >= 75.0) & (k1 <= 25.0) & valid_window

    sl_tensor = d_close - (atr * sl_mult)
    tp_tensor = d_close + (atr * tp_mult)
    res = simulate_signals(entries, sl_tensor, tp_tensor=tp_tensor, is_trailing=False)
    if res["trades"] < 50: raise optuna.TrialPruned("Too few trades")

    score = res["pf"] * (res["win_rate"] / 40.0) - (0.20 * (res["max_dd"] / max(res["net_rs"], 1.0)))
    for k, v in res.items(): trial.set_user_attr(k, v)
    return score

# Family 4: Pure Marni VSA Golden Pocket Retracement
def objective_family_4(trial):
    lookback = trial.suggest_int("lookback", 15, 35, step=5)
    fib_top = trial.suggest_float("fib_top", 0.58, 0.65, step=0.02)
    fib_bot = trial.suggest_float("fib_bot", 0.75, 0.82, step=0.02)
    min_span = trial.suggest_float("min_span", 15.0, 30.0, step=5.0)
    atr_period = trial.suggest_int("atr_period", 10, 20, step=2)
    sl_mult = trial.suggest_float("sl_mult", 1.0, 2.0, step=0.1)
    tp_mult = trial.suggest_float("tp_mult", 3.5, 6.0, step=0.25)

    atr = get_atr(atr_period)
    h_pad = F.pad(d_high.unsqueeze(1), (lookback - 1, 0), mode="replicate")
    l_pad = F.pad(d_low.unsqueeze(1), (lookback - 1, 0), mode="replicate")
    h_roll = F.max_pool1d(h_pad, kernel_size=lookback, stride=1).squeeze(1)
    l_roll = -F.max_pool1d(-l_pad, kernel_size=lookback, stride=1).squeeze(1)

    span = h_roll - l_roll
    f_top = h_roll - (span * fib_top)
    f_bot = h_roll - (span * fib_bot)

    in_pocket = (d_close <= f_top) & (d_close >= f_bot) & (span >= min_span)
    valid_window = torch.zeros_like(d_close, dtype=torch.bool)
    valid_window[:, SESSION_START_IDX:SESSION_END_IDX] = True
    entries = in_pocket & valid_window

    sl_tensor = d_close - (atr * sl_mult)
    tp_tensor = d_close + (atr * tp_mult)
    res = simulate_signals(entries, sl_tensor, tp_tensor=tp_tensor, is_trailing=False)
    if res["trades"] < 50: raise optuna.TrialPruned("Too few trades")

    score = res["pf"] * (res["win_rate"] / 40.0) - (0.20 * (res["max_dd"] / max(res["net_rs"], 1.0)))
    for k, v in res.items(): trial.set_user_attr(k, v)
    return score

# Family 5: Adaptive ATR Dynamic Multiplier Breakout
def objective_family_5(trial):
    s1_k = trial.suggest_int("s1_k", 7, 14)
    s4_k = trial.suggest_int("s4_k", 50, 75, step=5)
    atr_period = trial.suggest_int("atr_period", 10, 20, step=2)
    sl_mult = trial.suggest_float("sl_mult", 1.0, 2.2, step=0.1)
    tp_mult = trial.suggest_float("tp_mult", 4.0, 7.0, step=0.25)
    s4_thresh = trial.suggest_float("s4_thresh", 75.0, 85.0, step=2.5)

    k1 = get_stoch(s1_k)
    k4 = get_stoch(s4_k)
    atr = get_atr(atr_period)

    valid_window = torch.zeros_like(k1, dtype=torch.bool)
    valid_window[:, SESSION_START_IDX:SESSION_END_IDX] = True
    entries = (k4 >= s4_thresh) & (k1 <= 20.0) & valid_window

    sl_tensor = d_close - (atr * sl_mult)
    tp_tensor = d_close + (atr * tp_mult)
    res = simulate_signals(entries, sl_tensor, tp_tensor=tp_tensor, is_trailing=False)
    if res["trades"] < 50: raise optuna.TrialPruned("Too few trades")

    score = res["pf"] * (res["win_rate"] / 40.0) - (0.20 * (res["max_dd"] / max(res["net_rs"], 1.0)))
    for k, v in res.items(): trial.set_user_attr(k, v)
    return score

# ==============================================================================
# MASTER RUNNER & COMPARATIVE LEADERBOARD
# ==============================================================================
def optimize_family(family_name, objective_fn):
    print(f"\n{'='*115}", flush=True)
    print(f"STARTING: {family_name} ({TRIALS_PER_FAMILY} GPU Bayesian Trials)", flush=True)
    print(f"{'='*115}", flush=True)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective_fn, n_trials=TRIALS_PER_FAMILY)
    elapsed = time.time() - t0
    best = study.best_trial
    print(f"FINISHED: {family_name} in {elapsed:.2f}s (~{elapsed/TRIALS_PER_FAMILY:.2f}s/trial)", flush=True)
    print(f"  Best Quality Score: {best.value:.4f}", flush=True)
    print(f"  Win Rate:           {best.user_attrs['win_rate']:.2f}%", flush=True)
    print(f"  Profit Factor:      {best.user_attrs['pf']:.2f}", flush=True)
    print(f"  Net Realized Points:{best.user_attrs['net_pts']:+,.2f} pts", flush=True)
    print(f"  Net Realized P&L:   Rs {best.user_attrs['net_rs']:+,.2f}", flush=True)
    print(f"  Max Drawdown:       Rs {best.user_attrs['max_dd']:,.2f}", flush=True)
    print(f"  Optimal Parameters: {best.params}", flush=True)
    return {
        "family": family_name, "best_params": best.params,
        "user_attrs": best.user_attrs, "score": best.value, "time_s": elapsed
    }

def main():
    print("\n" + "=" * 115, flush=True)
    print(f"FLATTRADE BOT — 5-STRATEGY MASTER GPU OPTUNA STUDY ({TRIALS_PER_FAMILY} TRIALS EACH)", flush=True)
    print(f"Evaluating 500 total trials across 7 Years (1,574 Days) on NVIDIA RTX 3060", flush=True)
    print("=" * 115, flush=True)

    t_suite_start = time.time()
    results = []
    results.append(optimize_family("Family 1: S1 Turn-Up Trailing SL", objective_family_1))
    results.append(optimize_family("Family 2: Marni F6 Cross-Filter (ATR SL/TP)", objective_family_2))
    results.append(optimize_family("Family 3: 15m Elder Impulse HTF Gated Stoch", objective_family_3))
    results.append(optimize_family("Family 4: Pure Marni VSA Golden Pocket", objective_family_4))
    results.append(optimize_family("Family 5: Adaptive ATR Dynamic Breakout", objective_family_5))

    total_suite_time = time.time() - t_suite_start

    # Sort leaderboard by Quality Score
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 115, flush=True)
    print(f"MASTER MULTI-STRATEGY LEADERBOARD (500 GPU TRIALS EVALUATED IN {total_suite_time:.2f}s)", flush=True)
    print("=" * 115, flush=True)
    print(f"{'Rank':4s} | {'Strategy Family':38s} | {'Win Rate':9s} | {'PF':5s} | {'Net Points':12s} | {'Net P&L (Rs)':16s} | {'Max DD (Rs)':14s} | {'Quality Score':13s}", flush=True)
    print("-" * 115, flush=True)
    for idx, r in enumerate(results, start=1):
        u = r["user_attrs"]
        medal = "🥇" if idx == 1 else ("🥈" if idx == 2 else ("🥉" if idx == 3 else "  "))
        print(f"{medal}{idx:<2d} | {r['family']:38s} | {u['win_rate']:8.1f}% | {u['pf']:5.2f} | {u['net_pts']:+11,.1f} | Rs {u['net_rs']:+13,.2f} | Rs {u['max_dd']:11,.2f} | {r['score']:13.4f}", flush=True)
    print("-" * 115, flush=True)

    champion = results[0]
    print(f"\nUNDISPUTED OVERALL CHAMPION: {champion['family']}", flush=True)
    print(f"Optimal Parameters: {champion['best_params']}", flush=True)

    # Save to JSON ledger
    out_json = ROOT / "artifacts" / "f6_hybrid" / "multi_strategy_study_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "total_time_s": total_suite_time, "leaderboard": results}, f, indent=2)
    print(f"\nSaved master results to: {out_json}", flush=True)

if __name__ == "__main__":
    main()
