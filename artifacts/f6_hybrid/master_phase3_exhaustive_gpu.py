"""
PHASE 3 — EXHAUSTIVE GPU SEARCH ON TOP 4 CHAMPIONS
====================================================
Deep Bayesian search on the 4 best Phase 2 strategies:
  E03: Enhanced S18 Flag F6 + Daily Limits
  C02: S18-Signal × S08-Exit (F6→ATR)
  C07: S18+S06+S08 (F6+VolFilt→ATR)
  C10: S11+S18+S08 (Broad+F6Filt→ATR)

Enhancements over Phase 2:
  - 5× more trials (500 vs 100)
  - Expanded parameter ranges (wider + finer granularity)
  - Session window tuning enabled on ALL strategies
  - Enhanced triple-weighted objective (WR + PnL + low DD)
  - R:R constraint: TP >= 1.5 * SL enforced
  - 3D Batch Vectorized Engine (verified 13/13 causal checks)
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
TRIALS_PER_STRATEGY = 500  # 5× more than Phase 2
BATCH_SIZE = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"CUDA Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
print(f"Phase 3 EXHAUSTIVE: 4 Champions x 500 Trials x 2 Modes = 4,000 GPU Trials", flush=True)

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

# ─── Vectorized GPU Indicator Kernels (Causal, No Lookahead) ─────────────────
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


# ─── 3D Batch Vectorized Simulation Engine (Verified 13/13) ──────────────────
@torch.no_grad()
def simulate_gpu_with_limits(entries_mask, sl_tensor, tp_tensor, day_mask=None,
                              is_trailing=False, trail_trigger=10.0, trail_step=5.0,
                              max_daily_loss=9999.0, max_daily_profit=9999.0):
    if day_mask is not None:
        active_entries = entries_mask & day_mask.unsqueeze(1)
    else:
        active_entries = entries_mask

    coords = torch.nonzero(active_entries, as_tuple=False)
    if coords.shape[0] == 0:
        return {"trades": 0, "win_rate": 0.0, "net_pts": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    coords = coords[:5000]
    N_trades = coords.shape[0]
    d_indices = coords[:, 0]
    b_indices = coords[:, 1]
    ep = d_close[d_indices, b_indices]

    max_future = BASE_SESSION_END - BASE_SESSION_START - 1
    col_start = b_indices + 1
    col_offsets = torch.arange(max_future, device=device).unsqueeze(0)
    col_idx = col_start.unsqueeze(1) + col_offsets

    valid = (col_idx < BASE_SESSION_END) & (col_idx < 375)
    col_idx_safe = col_idx.clamp(max=374)

    d_exp = d_indices.unsqueeze(1).expand(-1, max_future)
    fut_h = d_high[d_exp, col_idx_safe]
    fut_l = d_low[d_exp, col_idx_safe]
    fut_c_eod = d_close[d_indices, BASE_SESSION_END - 1]

    INF = torch.tensor(1e9, device=device)
    fut_h_m = torch.where(valid, fut_h, -INF)
    fut_l_m = torch.where(valid, fut_l, INF)

    if not is_trailing:
        sl_p = sl_tensor[d_indices, b_indices]
        tp_p = tp_tensor[d_indices, b_indices] if tp_tensor is not None else ep + 9999.0

        hit_sl = fut_l_m <= sl_p.unsqueeze(1)
        hit_tp = fut_h_m >= tp_p.unsqueeze(1)

        sl_any = hit_sl.any(dim=1)
        tp_any = hit_tp.any(dim=1)

        BIG = 999999
        sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), torch.tensor(BIG, device=device))
        tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), torch.tensor(BIG, device=device))

        sl_exits = sl_any & (sl_first <= tp_first)
        tp_exits = tp_any & (~sl_exits)

        exit_px = torch.where(sl_exits, sl_p,
                  torch.where(tp_exits, tp_p, fut_c_eod))
    else:
        init_sl_p = sl_tensor[d_indices, b_indices]

        fut_h_for_cummax = torch.where(valid, fut_h, ep.unsqueeze(1))
        running_peaks = torch.cummax(fut_h_for_cummax, dim=1).values

        gains = running_peaks - ep.unsqueeze(1)
        levels = torch.clamp(torch.floor(gains / trail_trigger), min=0.0)
        dynamic_sl = torch.maximum(
            init_sl_p.unsqueeze(1).expand(-1, max_future),
            ep.unsqueeze(1) + (levels * trail_step) - (trail_trigger - trail_step)
        )

        hit_sl = fut_l_m <= dynamic_sl
        sl_any = hit_sl.any(dim=1)
        sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), torch.tensor(999999, device=device))

        sl_first_safe = sl_first.clamp(max=max_future - 1)
        sl_exit_px = dynamic_sl[torch.arange(N_trades, device=device), sl_first_safe]
        exit_px = torch.where(sl_any, sl_exit_px, fut_c_eod)

    has_future = (b_indices + 1) < BASE_SESSION_END
    exit_px = exit_px[has_future]
    ep_valid = ep[has_future]
    d_idx_valid = d_indices[has_future]

    if exit_px.shape[0] == 0:
        return {"trades": 0, "win_rate": 0.0, "net_pts": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    all_pts = (exit_px - ep_valid) * 0.50
    all_rs = all_pts * LOT_SIZE - 30.0

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
# PHASE 3 STRATEGY GENERATORS — EXPANDED PARAMETER RANGES
# ==============================================================================
def build_session_window(trial):
    """Session window tuning — enabled for ALL strategies in Phase 3."""
    valid_window = torch.zeros_like(d_close, dtype=torch.bool)
    start_off = trial.suggest_int("sess_start_off", 0, 25, step=5)
    end_off = trial.suggest_int("sess_end_off", 0, 45, step=15)
    valid_window[:, BASE_SESSION_START + start_off : BASE_SESSION_END - end_off] = True
    return valid_window

def get_daily_limits(trial):
    loss_choice = trial.suggest_categorical("daily_loss_pts", [10, 15, 20, 25, 30, 40, 50, 75, 9999])
    profit_choice = trial.suggest_categorical("daily_profit_pts", [15, 20, 30, 40, 50, 60, 80, 9999])
    return float(loss_choice) * LOT_SIZE, float(profit_choice) * LOT_SIZE


def build_phase3_strategy(strat_id, trial):
    daily_loss, daily_profit = get_daily_limits(trial)

    if strat_id == "E03":  # Enhanced S18 Flag F6 + Daily Limits — EXPANDED
        vw = build_session_window(trial)
        s1 = get_stoch(trial.suggest_int("s1_k", 5, 16))
        s4 = get_stoch(trial.suggest_int("s4_k", 40, 80, step=5))
        entries = (s4 >= trial.suggest_float("s4_ob", 72.5, 90.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 10.0, 30.0, step=2.5)) & vw
        atr = get_atr(trial.suggest_int("atr_p", 8, 22, step=2))
        sl_m = trial.suggest_float("sl_m", 1.0, 3.0, step=0.1)
        tp_m = trial.suggest_float("tp_m", 2.5, 7.0, step=0.25)
        if tp_m < 1.5 * sl_m:
            raise optuna.TrialPruned("R:R constraint: TP must be >= 1.5 * SL")
        sl = d_close - (atr * sl_m)
        tp = d_close + (atr * tp_m)
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    elif strat_id == "C02":  # S18-Signal x S08-Exit (F6->ATR) — EXPANDED + SESSION
        vw = build_session_window(trial)
        s1 = get_stoch(trial.suggest_int("s1_k", 5, 16))
        s4 = get_stoch(trial.suggest_int("s4_k", 40, 80, step=5))
        entries = (s4 >= trial.suggest_float("s4_ob", 72.5, 90.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 10.0, 30.0, step=2.5)) & vw
        atr = get_atr(trial.suggest_int("atr_p", 8, 22, step=2))
        sl_m = trial.suggest_float("sl_m", 0.8, 3.0, step=0.1)
        tp_m = trial.suggest_float("tp_m", 2.0, 7.5, step=0.25)
        if tp_m < 1.5 * sl_m:
            raise optuna.TrialPruned("R:R constraint: TP must be >= 1.5 * SL")
        sl = d_close - (atr * sl_m)
        tp = d_close + (atr * tp_m)
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    elif strat_id == "C07":  # S18+S06+S08 (F6+VolFilt->ATR) — EXPANDED + SESSION
        vw = build_session_window(trial)
        s1 = get_stoch(trial.suggest_int("s1_k", 5, 16))
        s4 = get_stoch(trial.suggest_int("s4_k", 40, 80, step=5))
        atr = get_atr(trial.suggest_int("atr_p", 8, 22, step=2))
        atr_med = atr.median(dim=1, keepdim=True).values
        vol_filter = atr >= (atr_med * trial.suggest_float("vol_thresh", 0.4, 1.4, step=0.1))
        entries = (s4 >= trial.suggest_float("s4_ob", 72.5, 90.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 10.0, 30.0, step=2.5)) & vol_filter & vw
        sl_m = trial.suggest_float("sl_m", 0.8, 3.0, step=0.1)
        tp_m = trial.suggest_float("tp_m", 2.0, 7.5, step=0.25)
        if tp_m < 1.5 * sl_m:
            raise optuna.TrialPruned("R:R constraint: TP must be >= 1.5 * SL")
        sl = d_close - (atr * sl_m)
        tp = d_close + (atr * tp_m)
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    elif strat_id == "C10":  # S11+S18+S08 (Broad+F6Filt->ATR) — EXPANDED + SESSION
        vw = build_session_window(trial)
        s1 = get_stoch(trial.suggest_int("s1_k", 5, 16))
        s4 = get_stoch(trial.suggest_int("s4_k", 40, 80, step=5))
        entries = (s4 >= trial.suggest_float("s4_ob", 72.5, 90.0, step=2.5)) & \
                  (s1 <= trial.suggest_float("s1_os", 10.0, 30.0, step=2.5)) & vw
        atr = get_atr(trial.suggest_int("atr_p", 8, 22, step=2))
        sl_m = trial.suggest_float("sl_m", 0.8, 3.0, step=0.1)
        tp_m = trial.suggest_float("tp_m", 2.0, 7.5, step=0.25)
        if tp_m < 1.5 * sl_m:
            raise optuna.TrialPruned("R:R constraint: TP must be >= 1.5 * SL")
        sl = d_close - (atr * sl_m)
        tp = d_close + (atr * tp_m)
        return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit

    else:
        raise ValueError(f"Unknown strategy ID: {strat_id}")


STRAT_IDS = ["E03", "C02", "C07", "C10"]

STRAT_NAMES = {
    "E03": "E03: Enhanced S18 Flag F6 + Limits (Exhaustive)",
    "C02": "C02: S18-Signal x S08-Exit F6->ATR (Exhaustive)",
    "C07": "C07: S18+S06+S08 F6+VolFilt->ATR (Exhaustive)",
    "C10": "C10: S11+S18+S08 Broad+F6Filt->ATR (Exhaustive)",
}


# ==============================================================================
# ENHANCED OPTUNA BATCH RUNNER — Triple-Weighted Objective
# ==============================================================================
def optimize_batch(strat_id, day_mask=None, n_total_trials=500, batch_size=50):
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, constant_liar=True)
    )
    n_batches = max(1, n_total_trials // batch_size)

    for _ in range(n_batches):
        batch_trials = [study.ask() for _ in range(batch_size)]

        for trial in batch_trials:
            try:
                entries, sl, tp, is_tr, trig, step, d_loss, d_prof = build_phase3_strategy(strat_id, trial)
                res = simulate_gpu_with_limits(entries, sl, tp, day_mask=day_mask,
                                                is_trailing=is_tr, trail_trigger=trig, trail_step=step,
                                                max_daily_loss=d_loss, max_daily_profit=d_prof)

                n_tr = res["trades"]
                min_trades = 30 if day_mask is not None else 50

                if n_tr < min_trades or res["net_rs"] <= 0:
                    score = -999.0
                else:
                    # Enhanced triple-weighted objective: WR + PnL + low DD
                    pf_comp = res["pf"] * (res["win_rate"] / 40.0)
                    dd_penalty = 0.30 * (res["max_dd"] / max(res["net_rs"], 1.0))
                    freq_bonus = min(n_tr / 500.0, 1.0) * 0.10
                    score = pf_comp - dd_penalty + freq_bonus

                    for k, v in res.items():
                        trial.set_user_attr(k, v)
            except optuna.TrialPruned:
                score = -999.0
            except Exception:
                score = -999.0

            study.tell(trial, score)

    return study.best_trial


def run_phase3_benchmark(strat_id, idx, total):
    name = STRAT_NAMES[strat_id]
    print(f"\n[{idx:02d}/{total}] PHASE 3 EXHAUSTIVE GPU: {name}", flush=True)

    t0 = time.time()
    best_nw = optimize_batch(strat_id, day_mask=None, n_total_trials=TRIALS_PER_STRATEGY, batch_size=BATCH_SIZE)
    t_non_wf = time.time() - t0

    t1 = time.time()
    best_wf_is = optimize_batch(strat_id, day_mask=d_is_mask, n_total_trials=TRIALS_PER_STRATEGY, batch_size=BATCH_SIZE)
    t_wf_is = time.time() - t1

    fixed_trial = optuna.trial.FixedTrial(best_wf_is.params)
    entries_oos, sl_oos, tp_oos, is_tr, trig, step, d_loss, d_prof = build_phase3_strategy(strat_id, fixed_trial)
    res_oos = simulate_gpu_with_limits(entries_oos, sl_oos, tp_oos, day_mask=d_oos_mask,
                                        is_trailing=is_tr, trail_trigger=trig, trail_step=step,
                                        max_daily_loss=d_loss, max_daily_profit=d_prof)

    is_annual_pnl = best_wf_is.user_attrs.get("net_rs", 0.0) / 4.0
    oos_annual_pnl = res_oos.get("net_rs", 0.0) / 2.35
    wfe = round(oos_annual_pnl / is_annual_pnl, 2) if is_annual_pnl > 0 else 0.0

    print(f"  [Non-WF 7Y in {t_non_wf:.1f}s]: WR={best_nw.user_attrs.get('win_rate',0.0):.1f}% | PF={best_nw.user_attrs.get('pf',0.0):.2f} | Net=Rs {best_nw.user_attrs.get('net_rs',0.0):+,.0f} | DD=Rs {best_nw.user_attrs.get('max_dd',0.0):,.0f} | Trades={best_nw.user_attrs.get('trades',0)}", flush=True)
    print(f"  [Walk-Forward in {t_wf_is:.1f}s]: IS Net=Rs {best_wf_is.user_attrs.get('net_rs',0.0):+,.0f} (PF {best_wf_is.user_attrs.get('pf',0.0):.2f}) -> OOS Net=Rs {res_oos['net_rs']:+,.0f} (PF {res_oos['pf']:.2f}, WR {res_oos['win_rate']:.1f}%) | WFE={wfe:.2f}", flush=True)
    print(f"  Params NW:  {best_nw.params}", flush=True)
    print(f"  Params WF:  {best_wf_is.params}", flush=True)

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
            "is_net_pts": best_wf_is.user_attrs.get("net_pts", 0.0),
            "is_max_dd": best_wf_is.user_attrs.get("max_dd", 0.0),
            "is_trades": best_wf_is.user_attrs.get("trades", 0),
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
    print("=" * 130, flush=True)
    print(f"FLATTRADE BOT — PHASE 3 EXHAUSTIVE GPU SEARCH ({total} CHAMPION STRATEGIES)")
    print(f"500 Trials/Strategy x 2 Modes = {total * TRIALS_PER_STRATEGY * 2:,} Total GPU Trials")
    print("Expanded Ranges + Session Windows + R:R Constraint + Triple-Weighted Objective")
    print("Non-Walk-Forward (Full 7Y) vs Walk-Forward (IS 2020-23 -> OOS 2024-26)")
    print("=" * 130, flush=True)

    t_start = time.time()
    all_results = []
    for idx, sid in enumerate(STRAT_IDS, start=1):
        all_results.append(run_phase3_benchmark(sid, idx, total))

    total_time = time.time() - t_start

    # Sort by OOS PnL (primary), then WFE (secondary)
    all_results_oos = sorted(all_results, key=lambda x: (x["walk_forward"]["oos_net_rs"], x["walk_forward"]["wfe"]), reverse=True)

    print("\n" + "=" * 150, flush=True)
    print(f"PHASE 3 — EXHAUSTIVE CHAMPION LEADERBOARD ({total * TRIALS_PER_STRATEGY * 2:,} GPU TRIALS IN {total_time:.2f}s)", flush=True)
    print("=" * 150, flush=True)

    # NW vs OOS side-by-side comparison
    all_results_nw = sorted(all_results, key=lambda x: x["non_wf"]["net_rs"], reverse=True)

    print(f"\n{'='*60} NON-WALK-FORWARD (Full 7Y) {'='*60}", flush=True)
    print(f"{'NW#':4s} | {'Strategy':48s} | {'NW PnL':>14s} | {'NW PF':>7s} | {'NW WR':>7s} | {'Trades':>6s} | {'MaxDD':>12s} | {'Score':>7s}", flush=True)
    print("-" * 120, flush=True)
    for rank, r in enumerate(all_results_nw, 1):
        nw = r["non_wf"]
        print(f"[{rank:2d}] | {r['name']:48s} | Rs {nw['net_rs']:+11,.0f} | {nw['pf']:6.2f} | {nw['win_rate']:5.1f}% | {nw['trades']:5d} | Rs {nw['max_dd']:9,.0f} | {nw['score']:6.4f}", flush=True)

    print(f"\n{'='*60} WALK-FORWARD OOS (2024-2026) {'='*60}", flush=True)
    print(f"{'OOS#':5s} | {'Strategy':48s} | {'IS PnL':>14s} | {'OOS PnL':>14s} | {'OOS PF':>7s} | {'OOS WR':>7s} | {'OOS DD':>12s} | {'Trades':>6s} | {'WFE':>5s}", flush=True)
    print("-" * 140, flush=True)
    for rank, r in enumerate(all_results_oos, 1):
        wf = r["walk_forward"]
        print(f"[{rank:2d}]  | {r['name']:48s} | Rs {wf['is_net_rs']:+11,.0f} | Rs {wf['oos_net_rs']:+11,.0f} | {wf['oos_pf']:6.2f} | {wf['oos_wr']:5.1f}% | Rs {wf['oos_max_dd']:9,.0f} | {wf['oos_trades']:5d} | {wf['wfe']:4.2f}", flush=True)

    print(f"\n{'='*60} SIDE-BY-SIDE COMPARISON {'='*60}", flush=True)
    nw_rank_map = {r["id"]: i+1 for i, r in enumerate(all_results_nw)}
    oos_rank_map = {r["id"]: i+1 for i, r in enumerate(all_results_oos)}
    print(f"{'NW#':4s} {'OOS#':5s} | {'Strategy':48s} | {'NW PnL':>14s} | {'NW PF':>7s} | {'OOS PnL':>14s} | {'OOS PF':>7s} | {'OOS WR':>7s} | {'WFE':>5s}", flush=True)
    print("-" * 140, flush=True)
    for r in all_results_nw:
        nw = r["non_wf"]
        wf = r["walk_forward"]
        nw_r = nw_rank_map[r["id"]]
        oos_r = oos_rank_map[r["id"]]
        print(f"[{nw_r:2d}] [{oos_r:2d}]  | {r['name']:48s} | Rs {nw['net_rs']:+11,.0f} | {nw['pf']:6.2f} | Rs {wf['oos_net_rs']:+11,.0f} | {wf['oos_pf']:6.2f} | {wf['oos_wr']:5.1f}% | {wf['wfe']:4.2f}", flush=True)

    print("-" * 140, flush=True)

    # Save JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "master_phase3_exhaustive.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_s": total_time,
            "trials_per_strategy": TRIALS_PER_STRATEGY,
            "total_trials": total * TRIALS_PER_STRATEGY * 2,
            "strategies": STRAT_IDS,
            "results": all_results_oos
        }, f, indent=2)
    print(f"\nSaved Phase 3 results to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
