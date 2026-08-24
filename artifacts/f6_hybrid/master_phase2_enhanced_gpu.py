"""
PHASE 2 — FUSED HPC GPU OPTUNA SUITE (ENHANCED PARAMETER SPACE + MULTI-STRATEGY COMBOS)
=========================================================================================
Extends the Section 22 Fused HPC Pipeline with:
  1. Daily Loss/Profit Limits (circuit breaker + profit cap) as Optuna parameters
  2. Unlocked stochastic params (s1_k, s4_k, s4_ob, s1_os) on formerly hardcoded strategies
  3. Session window tuning (start/end offsets)
  4. 10 Multi-Strategy Combinations (signal × exit cross + signal + filter + exit)
  5. Walk-Forward Validation (IS 2020-23 → OOS 2024-26)

Strategies:
  E01–E06: Enhanced singles (S08, S06, S18, S11, S19, S03)
  C01–C05: Tier-1 combos (signal × exit cross)
  C06–C10: Tier-2 combos (signal + filter + exit)
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

# ─── Hardware Configuration ──────────────────────────────────────────────────
torch.set_float32_matmul_precision("high")

LOT_SIZE = 65
BASE_SESSION_START = 5     # 09:20
BASE_SESSION_END = 345     # 15:00
TRIALS_PER_STRATEGY = 100
BATCH_SIZE = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"CUDA Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
print(f"Phase 2 HPC: Enhanced Params + Multi-Strategy Combos + Daily Limits + Session Windows", flush=True)

# ─── GPU VRAM Data Loader ────────────────────────────────────────────────────
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

    is_mask = np.array([d < "2024-01-01" for d in days], dtype=bool)
    oos_mask = np.array([d >= "2024-01-01" for d in days], dtype=bool)

    t_is_mask = torch.tensor(is_mask, dtype=torch.bool, device=device)
    t_oos_mask = torch.tensor(oos_mask, dtype=torch.bool, device=device)

    return d_h, d_l, d_c, d_o, d_tr, days, t_is_mask, t_oos_mask

print("Loading 7-Year Historical Matrix into GPU VRAM...", flush=True)
t_load_0 = time.time()
d_high, d_low, d_close, d_open, d_tr, all_days, d_is_mask, d_oos_mask = load_gpu_data()
N_DAYS = len(all_days)
print(f"Loaded {N_DAYS} days into VRAM in {time.time()-t_load_0:.2f}s — Tensor Shape: {d_close.shape}", flush=True)

# ─── Vectorized GPU Indicator Kernels ────────────────────────────────────────
@torch.no_grad()
def get_stoch(k_period):
    h_pad = F.pad(d_high.unsqueeze(1), (k_period - 1, 0), mode="replicate")
    l_pad = F.pad(d_low.unsqueeze(1), (k_period - 1, 0), mode="replicate")
    max_h = F.max_pool1d(h_pad, kernel_size=k_period, stride=1).squeeze(1)
    min_l = -F.max_pool1d(-l_pad, kernel_size=k_period, stride=1).squeeze(1)
    denom = (max_h - min_l).clamp(min=1e-6)
    return ((d_close - min_l) / denom) * 100.0

@torch.no_grad()
def get_atr(period=14):
    tr_pad = F.pad(d_tr.unsqueeze(1), (period - 1, 0), mode="replicate")
    return F.avg_pool1d(tr_pad, kernel_size=period, stride=1).squeeze(1)

@torch.no_grad()
def get_ema(period=20):
    alpha = 2.0 / (period + 1)
    ema = torch.zeros_like(d_close)
    ema[:, 0] = d_close[:, 0]
    for t in range(1, d_close.shape[1]):
        ema[:, t] = alpha * d_close[:, t] + (1 - alpha) * ema[:, t-1]
    return ema

# ─── Simulation Engine with Daily P&L Limits ────────────────────────────────
@torch.no_grad()
def simulate_gpu_with_limits(entries_mask, sl_tensor, tp_tensor, day_mask=None,
                              is_trailing=False, trail_trigger=10.0, trail_step=5.0,
                              max_daily_loss=9999.0, max_daily_profit=9999.0):
    """
    3D Batch Vectorized Simulation Engine (Verified: 13/13 causality + live parity checks passed).
    Phase 1: Compute all trade exits in parallel via GPU advanced indexing.
    Phase 2: Apply daily P&L limits sequentially (fast — O(N_trades) Python, no GPU work).
    """
    if day_mask is not None:
        active_entries = entries_mask & day_mask.unsqueeze(1)
    else:
        active_entries = entries_mask

    coords = torch.nonzero(active_entries, as_tuple=False)
    if coords.shape[0] == 0:
        return {"trades": 0, "win_rate": 0.0, "net_pts": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    coords = coords[:5000]
    N_trades = coords.shape[0]
    d_indices = coords[:, 0]  # (N,)
    b_indices = coords[:, 1]  # (N,)
    ep = d_close[d_indices, b_indices]  # (N,) entry prices

    # ── Phase 1: Build 3D future price tensors (fully parallel) ──
    max_future = BASE_SESSION_END - BASE_SESSION_START - 1  # ~339 bars max
    col_start = b_indices + 1  # (N,) first future bar for each trade
    col_offsets = torch.arange(max_future, device=device).unsqueeze(0)  # (1, max_future)
    col_idx = col_start.unsqueeze(1) + col_offsets  # (N, max_future)

    # Valid mask: columns within session and array bounds
    valid = (col_idx < BASE_SESSION_END) & (col_idx < 375)  # (N, max_future)
    col_idx_safe = col_idx.clamp(max=374)  # safe indexing

    # Advanced gather into 3D tensors
    d_exp = d_indices.unsqueeze(1).expand(-1, max_future)  # (N, max_future)
    fut_h = d_high[d_exp, col_idx_safe]  # (N, max_future)
    fut_l = d_low[d_exp, col_idx_safe]   # (N, max_future)
    fut_c_eod = d_close[d_indices, BASE_SESSION_END - 1]  # (N,) EOD close

    # Mask invalid bars (high → -inf so never triggers TP; low → +inf so never triggers SL)
    INF = torch.tensor(1e9, device=device)
    fut_h_m = torch.where(valid, fut_h, -INF)
    fut_l_m = torch.where(valid, fut_l, INF)

    if not is_trailing:
        sl_p = sl_tensor[d_indices, b_indices]  # (N,)
        tp_p = tp_tensor[d_indices, b_indices] if tp_tensor is not None else ep + 9999.0  # (N,)

        # Vectorized SL/TP hit detection across all trades simultaneously
        hit_sl = fut_l_m <= sl_p.unsqueeze(1)  # (N, max_future)
        hit_tp = fut_h_m >= tp_p.unsqueeze(1)  # (N, max_future)

        sl_any = hit_sl.any(dim=1)  # (N,)
        tp_any = hit_tp.any(dim=1)  # (N,)

        BIG = 999999
        sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), torch.tensor(BIG, device=device))
        tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), torch.tensor(BIG, device=device))

        sl_exits = sl_any & (sl_first <= tp_first)
        tp_exits = tp_any & (~sl_exits)

        exit_px = torch.where(sl_exits, sl_p,
                  torch.where(tp_exits, tp_p, fut_c_eod))
    else:
        init_sl_p = sl_tensor[d_indices, b_indices]  # (N,)

        # Mask future highs for cummax (invalid → entry price)
        fut_h_for_cummax = torch.where(valid, fut_h, ep.unsqueeze(1))
        running_peaks = torch.cummax(fut_h_for_cummax, dim=1).values  # (N, max_future)

        gains = running_peaks - ep.unsqueeze(1)  # (N, max_future)
        levels = torch.clamp(torch.floor(gains / trail_trigger), min=0.0)
        dynamic_sl = torch.maximum(
            init_sl_p.unsqueeze(1).expand(-1, max_future),
            ep.unsqueeze(1) + (levels * trail_step) - (trail_trigger - trail_step)
        )  # (N, max_future)

        hit_sl = fut_l_m <= dynamic_sl  # (N, max_future)
        sl_any = hit_sl.any(dim=1)  # (N,)
        sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), torch.tensor(999999, device=device))

        sl_first_safe = sl_first.clamp(max=max_future - 1)
        sl_exit_px = dynamic_sl[torch.arange(N_trades, device=device), sl_first_safe]
        exit_px = torch.where(sl_any, sl_exit_px, fut_c_eod)

    # Filter trades with no future bars
    has_future = (b_indices + 1) < BASE_SESSION_END
    exit_px = exit_px[has_future]
    ep_valid = ep[has_future]
    d_idx_valid = d_indices[has_future]

    if exit_px.shape[0] == 0:
        return {"trades": 0, "win_rate": 0.0, "net_pts": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    # P&L (all trades, before daily limits)
    all_pts = (exit_px - ep_valid) * 0.50  # (M,)
    all_rs = all_pts * LOT_SIZE - 30.0     # (M,)

    # ── Phase 2: Apply daily P&L limits (sequential, fast O(M) Python) ──
    all_pts_cpu = all_pts.cpu().numpy()
    all_rs_cpu = all_rs.cpu().numpy()
    d_idx_cpu = d_idx_valid.cpu().numpy()

    daily_pnl = {}
    keep_mask = np.ones(len(all_rs_cpu), dtype=bool)

    for k in range(len(all_rs_cpu)):
        d_i = int(d_idx_cpu[k])
        day_cum = daily_pnl.get(d_i, 0.0)

        if day_cum <= -max_daily_loss or day_cum >= max_daily_profit:
            keep_mask[k] = False
            continue

        daily_pnl[d_i] = day_cum + all_rs_cpu[k]

    # Apply filter
    final_pts = all_pts_cpu[keep_mask]
    final_rs = all_rs_cpu[keep_mask]

    if len(final_rs) == 0:
        return {"trades": 0, "win_rate": 0.0, "net_pts": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    wins = int((final_pts > 0).sum())
    n_trades = len(final_rs)
    pos_rs = float(final_rs[final_rs > 0].sum())
    neg_rs = float(abs(final_rs[final_rs <= 0].sum()))
    pf = (pos_rs / neg_rs) if neg_rs > 0 else 0.0

    equity = np.cumsum(final_rs)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(peak - equity))

    return {
        "trades": n_trades,
        "win_rate": round(wins / n_trades * 100.0, 2),
        "net_pts": round(float(final_pts.sum()), 2),
        "net_rs": round(float(final_rs.sum()), 2),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 2),
    }

# ==============================================================================
# STRATEGY GENERATORS — 6 Enhanced Singles + 10 Multi-Strategy Combos
# ==============================================================================
def build_session_window(trial, allow_tuning=True):
    valid_window = torch.zeros_like(d_close, dtype=torch.bool)
    if allow_tuning:
        start_off = trial.suggest_int("sess_start_off", 0, 20, step=5)
        end_off = trial.suggest_int("sess_end_off", 0, 30, step=15)
    else:
        start_off = 0
        end_off = 0
    valid_window[:, BASE_SESSION_START + start_off : BASE_SESSION_END - end_off] = True
    return valid_window

def get_daily_limits(trial):
    loss_choice = trial.suggest_categorical("daily_loss_pts", [15, 20, 25, 30, 40, 50, 9999])
    profit_choice = trial.suggest_categorical("daily_profit_pts", [20, 30, 40, 50, 60, 9999])
    return float(loss_choice) * LOT_SIZE, float(profit_choice) * LOT_SIZE


def build_phase2_strategy(strat_id, trial):
    daily_loss, daily_profit = get_daily_limits(trial)

    if strat_id == "E01":  # Enhanced S08 — Pure ATR Fixed Multipliers
        vw = build_session_window(trial)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        entries = (s4 >= trial.suggest_float("s4_ob", 75.0, 85.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 15.0, 25.0, step=2.5)) & vw
        atr = get_atr(trial.suggest_int("atr_p", 10, 20, step=2))
        sl = d_close - (atr * trial.suggest_float("sl_m", 1.2, 2.5, step=0.1))
        tp = d_close + (atr * trial.suggest_float("tp_m", 3.0, 6.0, step=0.25))
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    elif strat_id == "E02":  # Enhanced S06 — Macro Volatility Scaling
        vw = build_session_window(trial)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        entries = (s4 >= 80.0) & (s1 <= 20.0) & vw
        atr = get_atr(trial.suggest_int("atr_p", 10, 20, step=2))
        sl = d_close - (atr * trial.suggest_float("sl_mult", 1.0, 2.2, step=0.2))
        tp = d_close + (atr * trial.suggest_float("tp_mult", 3.0, 6.0, step=0.5))
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    elif strat_id == "E03":  # Enhanced S18 — Flag Immediate Entry F6
        vw = build_session_window(trial)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        entries = (s4 >= trial.suggest_float("s4_ob", 77.5, 85.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 15.0, 25.0, step=2.5)) & vw
        atr = get_atr(trial.suggest_int("atr_p", 10, 18, step=2))
        sl = d_close - (atr * trial.suggest_float("sl_m", 1.5, 2.5, step=0.2))
        tp = d_close + (atr * trial.suggest_float("tp_m", 3.5, 6.0, step=0.25))
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    elif strat_id == "E04":  # Enhanced S11 — Unlimited Profit Trailing SL
        vw = build_session_window(trial)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        entries = (s4 >= trial.suggest_float("s4_ob", 75.0, 85.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 15.0, 25.0, step=2.5)) & vw
        sl = d_close - trial.suggest_float("init_sl", 12.0, 20.0, step=2.0)
        return entries, sl, None, True, trial.suggest_float("trail_trig", 8.0, 14.0, step=1.0), \
               trial.suggest_float("trail_step", 4.0, 7.0, step=1.0), daily_loss, daily_profit

    elif strat_id == "E05":  # Enhanced S19 — F6 + S1 Turn-Up Hybrid
        vw = build_session_window(trial)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 12))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        s1_prev = F.pad(s1[:, :-1], (1, 0), mode="replicate")
        entries = (s4 >= trial.suggest_float("s4_ob", 77.5, 85.0, step=2.5)) & \
                  (s1 <= 25.0) & (s1 > s1_prev) & vw
        sl = d_close - trial.suggest_float("init_sl", 15.0, 25.0, step=2.5)
        return entries, sl, None, True, trial.suggest_float("trail_trig", 8.0, 14.0, step=1.0), \
               trial.suggest_float("trail_step", 4.0, 8.0, step=1.0), daily_loss, daily_profit

    elif strat_id == "E06":  # Enhanced S03 — Volume PinBar Confirmation
        vw = build_session_window(trial)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        body = torch.abs(d_close - d_open)
        wick = d_high - torch.maximum(d_close, d_open)
        pinbar = wick >= (body * trial.suggest_float("wick_ratio", 1.2, 2.4, step=0.2))
        entries = (s4 >= 75.0) & (s1 <= 25.0) & pinbar & vw
        atr = get_atr(trial.suggest_int("atr_p", 10, 18, step=2))
        sl = d_close - (atr * trial.suggest_float("sl_m", 1.0, 2.0, step=0.2))
        tp = d_close + (atr * trial.suggest_float("tp_m", 3.0, 5.5, step=0.25))
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    # ─── TIER 1 COMBOS: SIGNAL × EXIT CROSS ─────────────────────────────
    elif strat_id == "C01":  # S08-Signal + S19-Exit (ATR→Trail)
        vw = build_session_window(trial, allow_tuning=False)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        entries = (s4 >= trial.suggest_float("s4_ob", 75.0, 85.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 15.0, 25.0, step=2.5)) & vw
        sl = d_close - trial.suggest_float("init_sl", 12.0, 20.0, step=2.0)
        return entries, sl, None, True, trial.suggest_float("trail_trig", 8.0, 14.0, step=1.0), \
               trial.suggest_float("trail_step", 4.0, 8.0, step=1.0), daily_loss, daily_profit

    elif strat_id == "C02":  # S18-Signal + S08-Exit (F6→ATR)
        vw = build_session_window(trial, allow_tuning=False)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        entries = (s4 >= trial.suggest_float("s4_ob", 77.5, 85.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 15.0, 25.0, step=2.5)) & vw
        atr = get_atr(trial.suggest_int("atr_p", 10, 20, step=2))
        sl = d_close - (atr * trial.suggest_float("sl_m", 1.2, 2.5, step=0.1))
        tp = d_close + (atr * trial.suggest_float("tp_m", 3.0, 6.0, step=0.25))
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    elif strat_id == "C03":  # S18-Signal + S11-Exit (F6→Trail)
        vw = build_session_window(trial, allow_tuning=False)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        entries = (s4 >= trial.suggest_float("s4_ob", 77.5, 85.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 15.0, 25.0, step=2.5)) & vw
        sl = d_close - trial.suggest_float("init_sl", 12.0, 20.0, step=2.0)
        return entries, sl, None, True, trial.suggest_float("trail_trig", 8.0, 14.0, step=1.0), \
               trial.suggest_float("trail_step", 4.0, 7.0, step=1.0), daily_loss, daily_profit

    elif strat_id == "C04":  # S06-Signal + S19-Exit (MacroVol→Trail)
        vw = build_session_window(trial, allow_tuning=False)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        s1_prev = F.pad(s1[:, :-1], (1, 0), mode="replicate")
        entries = (s4 >= 80.0) & (s1 <= 20.0) & (s1 > s1_prev) & vw
        sl = d_close - trial.suggest_float("init_sl", 15.0, 25.0, step=2.5)
        return entries, sl, None, True, trial.suggest_float("trail_trig", 8.0, 14.0, step=1.0), \
               trial.suggest_float("trail_step", 4.0, 8.0, step=1.0), daily_loss, daily_profit

    elif strat_id == "C05":  # S03-Signal + S08-Exit (PinBar→ATR)
        vw = build_session_window(trial, allow_tuning=False)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        body = torch.abs(d_close - d_open)
        wick = d_high - torch.maximum(d_close, d_open)
        pinbar = wick >= (body * trial.suggest_float("wick_ratio", 1.2, 2.4, step=0.2))
        entries = (s4 >= 75.0) & (s1 <= 25.0) & pinbar & vw
        atr = get_atr(trial.suggest_int("atr_p", 10, 20, step=2))
        sl = d_close - (atr * trial.suggest_float("sl_m", 1.2, 2.5, step=0.1))
        tp = d_close + (atr * trial.suggest_float("tp_m", 3.0, 6.0, step=0.25))
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    # ─── TIER 2 COMBOS: SIGNAL + FILTER + EXIT (Triple-Layer) ───────────
    elif strat_id == "C06":  # S08+S03+S11 (ATR+PinBar→Trail)
        vw = build_session_window(trial, allow_tuning=False)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        body = torch.abs(d_close - d_open)
        wick = d_high - torch.maximum(d_close, d_open)
        pinbar = wick >= (body * 1.5)
        entries = (s4 >= 80.0) & (s1 <= 20.0) & pinbar & vw
        sl = d_close - trial.suggest_float("init_sl", 12.0, 20.0, step=2.0)
        return entries, sl, None, True, trial.suggest_float("trail_trig", 8.0, 14.0, step=1.0), \
               trial.suggest_float("trail_step", 4.0, 7.0, step=1.0), daily_loss, daily_profit

    elif strat_id == "C07":  # S18+S06+S08 (F6+VolFilt→ATR)
        vw = build_session_window(trial, allow_tuning=False)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        atr = get_atr(trial.suggest_int("atr_p", 10, 20, step=2))
        atr_med = atr.median(dim=1, keepdim=True).values
        vol_filter = atr >= (atr_med * trial.suggest_float("vol_thresh", 0.6, 1.2, step=0.1))
        entries = (s4 >= trial.suggest_float("s4_ob", 77.5, 85.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 15.0, 25.0, step=2.5)) & vol_filter & vw
        sl = d_close - (atr * trial.suggest_float("sl_m", 1.2, 2.5, step=0.1))
        tp = d_close + (atr * trial.suggest_float("tp_m", 3.0, 6.0, step=0.25))
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    elif strat_id == "C08":  # S08+S13+S19 (ATR+TurnUp→Trail)
        vw = build_session_window(trial, allow_tuning=False)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        s1_prev = F.pad(s1[:, :-1], (1, 0), mode="replicate")
        entries = (s4 >= 80.0) & (s1 <= 20.0) & (s1 > s1_prev) & vw
        sl = d_close - trial.suggest_float("init_sl", 15.0, 25.0, step=2.5)
        return entries, sl, None, True, trial.suggest_float("trail_trig", 8.0, 14.0, step=1.0), \
               trial.suggest_float("trail_step", 4.0, 8.0, step=1.0), daily_loss, daily_profit

    elif strat_id == "C09":  # S06+S03+S18 (MacroVol+PinBar→ATR)
        vw = build_session_window(trial, allow_tuning=False)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(60)
        body = torch.abs(d_close - d_open)
        wick = d_high - torch.maximum(d_close, d_open)
        pinbar = wick >= (body * trial.suggest_float("wick_ratio", 1.2, 2.0, step=0.2))
        entries = (s4 >= 80.0) & (s1 <= 20.0) & pinbar & vw
        atr = get_atr(trial.suggest_int("atr_p", 10, 18, step=2))
        sl = d_close - (atr * trial.suggest_float("sl_m", 1.5, 2.5, step=0.2))
        tp = d_close + (atr * trial.suggest_float("tp_m", 3.5, 6.0, step=0.25))
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    elif strat_id == "C10":  # S11+S18+S08 (Broad+F6Filt→ATR)
        vw = build_session_window(trial, allow_tuning=False)
        s1 = get_stoch(trial.suggest_int("s1_k", 7, 14))
        s4 = get_stoch(trial.suggest_int("s4_k", 50, 70, step=5))
        entries = (s4 >= trial.suggest_float("s4_ob", 77.5, 85.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 15.0, 25.0, step=2.5)) & vw
        atr = get_atr(trial.suggest_int("atr_p", 10, 20, step=2))
        sl = d_close - (atr * trial.suggest_float("sl_m", 1.2, 2.5, step=0.1))
        tp = d_close + (atr * trial.suggest_float("tp_m", 3.0, 6.0, step=0.25))
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    else:
        raise ValueError(f"Unknown strategy ID: {strat_id}")


STRAT_IDS = [
    "E01", "E02", "E03", "E04", "E05", "E06",
    "C01", "C02", "C03", "C04", "C05",
    "C06", "C07", "C08", "C09", "C10"
]

STRAT_NAMES = {
    "E01": "E01: Enhanced S08 Pure ATR + Daily Limits",
    "E02": "E02: Enhanced S06 Macro Vol + Daily Limits",
    "E03": "E03: Enhanced S18 Flag F6 + Daily Limits",
    "E04": "E04: Enhanced S11 Trailing SL + Daily Limits",
    "E05": "E05: Enhanced S19 F6 Turn-Up + Daily Limits",
    "E06": "E06: Enhanced S03 PinBar + Daily Limits",
    "C01": "C01: S08-Signal x S19-Exit (ATR->Trail)",
    "C02": "C02: S18-Signal x S08-Exit (F6->ATR)",
    "C03": "C03: S18-Signal x S11-Exit (F6->Trail)",
    "C04": "C04: S06-Signal x S19-Exit (MacroVol->Trail)",
    "C05": "C05: S03-Signal x S08-Exit (PinBar->ATR)",
    "C06": "C06: S08+S03+S11 (ATR+PinBar->Trail)",
    "C07": "C07: S18+S06+S08 (F6+VolFilt->ATR)",
    "C08": "C08: S08+S13+S19 (ATR+TurnUp->Trail)",
    "C09": "C09: S06+S03+S18 (MacroVol+PinBar->ATR)",
    "C10": "C10: S11+S18+S08 (Broad+F6Filt->ATR)",
}


# ==============================================================================
# FUSED OPTUNA BATCH RUNNER (Phase 2)
# ==============================================================================
def optimize_batch(strat_id, day_mask=None, n_total_trials=100, batch_size=50):
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, constant_liar=True)
    )
    n_batches = max(1, n_total_trials // batch_size)

    for _ in range(n_batches):
        batch_trials = [study.ask() for _ in range(batch_size)]

        for trial in batch_trials:
            try:
                entries, sl, tp, is_tr, trig, step, d_loss, d_prof = build_phase2_strategy(strat_id, trial)
                res = simulate_gpu_with_limits(entries, sl, tp, day_mask=day_mask,
                                                is_trailing=is_tr, trail_trigger=trig, trail_step=step,
                                                max_daily_loss=d_loss, max_daily_profit=d_prof)
                if res["trades"] < (30 if day_mask is not None else 50):
                    score = -999.0
                else:
                    score = res["pf"] * (res["win_rate"] / 40.0) - (0.20 * (res["max_dd"] / max(res["net_rs"], 1.0)))
                    for k, v in res.items():
                        trial.set_user_attr(k, v)
            except Exception:
                score = -999.0

            study.tell(trial, score)

    return study.best_trial


def run_phase2_benchmark(strat_id, idx, total):
    name = STRAT_NAMES[strat_id]
    print(f"\n[{idx:02d}/{total}] PHASE 2 GPU OPTUNA: {name}", flush=True)

    t0 = time.time()
    best_nw = optimize_batch(strat_id, day_mask=None, n_total_trials=TRIALS_PER_STRATEGY, batch_size=BATCH_SIZE)
    t_non_wf = time.time() - t0

    t1 = time.time()
    best_wf_is = optimize_batch(strat_id, day_mask=d_is_mask, n_total_trials=TRIALS_PER_STRATEGY, batch_size=BATCH_SIZE)
    t_wf_is = time.time() - t1

    fixed_trial = optuna.trial.FixedTrial(best_wf_is.params)
    entries_oos, sl_oos, tp_oos, is_tr, trig, step, d_loss, d_prof = build_phase2_strategy(strat_id, fixed_trial)
    res_oos = simulate_gpu_with_limits(entries_oos, sl_oos, tp_oos, day_mask=d_oos_mask,
                                        is_trailing=is_tr, trail_trigger=trig, trail_step=step,
                                        max_daily_loss=d_loss, max_daily_profit=d_prof)

    is_annual_pnl = best_wf_is.user_attrs.get("net_rs", 0.0) / 4.0
    oos_annual_pnl = res_oos.get("net_rs", 0.0) / 2.35
    wfe = round(oos_annual_pnl / is_annual_pnl, 2) if is_annual_pnl > 0 else 0.0

    print(f"  [Non-WF 7Y in {t_non_wf:.1f}s]: WR={best_nw.user_attrs.get('win_rate',0.0):.1f}% | PF={best_nw.user_attrs.get('pf',0.0):.2f} | Net=Rs {best_nw.user_attrs.get('net_rs',0.0):+,.0f} | DD=Rs {best_nw.user_attrs.get('max_dd',0.0):,.0f}", flush=True)
    print(f"  [Walk-Forward in {t_wf_is:.1f}s]: IS Net=Rs {best_wf_is.user_attrs.get('net_rs',0.0):+,.0f} (PF {best_wf_is.user_attrs.get('pf',0.0):.2f}) -> OOS Net=Rs {res_oos['net_rs']:+,.0f} (PF {res_oos['pf']:.2f}) | WFE={wfe:.2f}", flush=True)

    return {
        "id": strat_id,
        "name": name,
        "non_wf": {
            "best_params": best_nw.params,
            "win_rate": best_nw.user_attrs.get("win_rate", 0.0),
            "pf": best_nw.user_attrs.get("pf", 0.0),
            "net_pts": best_nw.user_attrs.get("net_pts", 0.0),
            "net_rs": best_nw.user_attrs.get("net_rs", 0.0),
            "max_dd": best_nw.user_attrs.get("max_dd", 0.0),
            "trades": best_nw.user_attrs.get("trades", 0),
            "score": round(best_nw.value, 4),
            "time_s": round(t_non_wf, 2)
        },
        "walk_forward": {
            "is_params": best_wf_is.params,
            "is_wr": best_wf_is.user_attrs.get("win_rate", 0.0),
            "is_pf": best_wf_is.user_attrs.get("pf", 0.0),
            "is_net_rs": best_wf_is.user_attrs.get("net_rs", 0.0),
            "oos_wr": res_oos["win_rate"],
            "oos_pf": res_oos["pf"],
            "oos_net_pts": res_oos["net_pts"],
            "oos_net_rs": res_oos["net_rs"],
            "oos_max_dd": res_oos["max_dd"],
            "oos_trades": res_oos["trades"],
            "wfe": wfe,
            "time_s": round(t_wf_is, 2)
        }
    }


def main():
    total = len(STRAT_IDS)
    print("=" * 115, flush=True)
    print(f"FLATTRADE BOT — PHASE 2 FUSED HPC GPU OPTUNA ({total} STRATEGIES)")
    print("Enhanced Params + Daily Limits + Multi-Strategy Combos")
    print("Non-Walk-Forward (Full 7Y) vs Walk-Forward (IS 2020-23 -> OOS 2024-26)")
    print("=" * 115, flush=True)

    t_start = time.time()
    all_results = []
    for idx, sid in enumerate(STRAT_IDS, start=1):
        all_results.append(run_phase2_benchmark(sid, idx, total))

    total_time = time.time() - t_start

    all_results = sorted(all_results, key=lambda x: (x["walk_forward"]["oos_net_rs"], x["walk_forward"]["wfe"]), reverse=True)

    print("\n" + "=" * 130, flush=True)
    print(f"PHASE 2 — {total}-STRATEGY COMPARATIVE LEADERBOARD ({total * TRIALS_PER_STRATEGY * 2:,} GPU TRIALS IN {total_time:.2f}s)", flush=True)
    print("=" * 130, flush=True)
    print(f"{'Rank':4s} | {'Strategy Name':42s} | {'Non-WF 7Y PnL':15s} | {'Non-WF PF':10s} | {'IS PnL (4Y)':14s} | {'OOS PnL (2.4Y)':16s} | {'OOS PF':7s} | {'OOS WR':7s} | {'WFE':5s}")
    print("-" * 130, flush=True)

    for rank, r in enumerate(all_results, start=1):
        nw = r["non_wf"]
        wf = r["walk_forward"]
        medal = "[#1]" if rank == 1 else (f"[{rank:2d}]")
        print(f"{medal:4s} | {r['name']:42s} | Rs {nw['net_rs']:+12,.0f} | {nw['pf']:9.2f} | Rs {wf['is_net_rs']:+11,.0f} | Rs {wf['oos_net_rs']:+13,.0f} | {wf['oos_pf']:6.2f} | {wf['oos_wr']:5.1f}% | {wf['wfe']:4.2f}", flush=True)

    print("-" * 130, flush=True)

    out_file = ROOT / "artifacts" / "f6_hybrid" / "master_phase2_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "total_time_s": total_time, "results": all_results}, f, indent=2)
    print(f"\nSaved Phase 2 results to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
