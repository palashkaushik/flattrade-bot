"""
PHASE 4 — ULTIMATE EXHAUSTIVE GPU SEARCH (NO BOUNDARY HITS)
=============================================================
Building on C02's breakthrough (+Rs 1,25,900 OOS), this phase:
  1. Expands ALL ranges so no parameter can hit a boundary
  2. Adds NEW entry filters from academic/web research:
     - RSI oversold confirmation
     - EMA trend filter (trade with trend only)
     - Bollinger Band squeeze filter
     - Double stochastic confirmation (medium period)
  3. 1000 trials per strategy (10x Phase 2)
  4. 5 strategy families → top 5 results

Strategies:
  F01: C02 Ultra-Wide (no boundary possible)
  F02: C02 + RSI Oversold Confirmation
  F03: C02 + EMA Trend Filter (price > EMA = long)
  F04: C02 + Bollinger Band Squeeze
  F05: C02 + Double Stochastic (3-period confirmation)
"""

import json
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

torch.set_float32_matmul_precision("high")

LOT_SIZE = 65
BASE_SESSION_START = 5
BASE_SESSION_END = 345
TRIALS_PER_STRATEGY = 1000
BATCH_SIZE = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"CUDA: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
print(f"Phase 4 ULTIMATE: 5 Strategies x {TRIALS_PER_STRATEGY} Trials x 2 Modes = {5*TRIALS_PER_STRATEGY*2:,} GPU Trials", flush=True)

# ─── Data Loader ─────────────────────────────────────────────────────────────
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
    return d_h, d_l, d_c, d_o, d_tr, days, \
           torch.tensor(is_mask, dtype=torch.bool, device=device), \
           torch.tensor(oos_mask, dtype=torch.bool, device=device)

print("Loading 7Y data...", flush=True)
t0 = time.time()
d_high, d_low, d_close, d_open, d_tr, all_days, d_is_mask, d_oos_mask = load_gpu_data()
N_DAYS = len(all_days)
print(f"Loaded {N_DAYS} days in {time.time()-t0:.1f}s", flush=True)

# ─── GPU Indicator Kernels ───────────────────────────────────────────────────
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

@torch.no_grad()
def get_rsi(period=14):
    """Causal RSI using left-padded EMA of gains/losses."""
    delta = d_close[:, 1:] - d_close[:, :-1]  # (N, 374)
    gains = torch.clamp(delta, min=0)
    losses = torch.clamp(-delta, min=0)
    # Pad to maintain 375 length
    gains = F.pad(gains, (1, 0), mode="constant", value=0)  # (N, 375)
    losses = F.pad(losses, (1, 0), mode="constant", value=0)
    # Rolling average via causal avg_pool1d
    g_pad = F.pad(gains.unsqueeze(1), (period - 1, 0), mode="replicate")
    l_pad = F.pad(losses.unsqueeze(1), (period - 1, 0), mode="replicate")
    avg_gain = F.avg_pool1d(g_pad, kernel_size=period, stride=1).squeeze(1)
    avg_loss = F.avg_pool1d(l_pad, kernel_size=period, stride=1).squeeze(1)
    rs = avg_gain / avg_loss.clamp(min=1e-8)
    return 100.0 - (100.0 / (1.0 + rs))

@torch.no_grad()
def get_bollinger(period=20, num_std=2.0):
    """Causal Bollinger Bands."""
    c_pad = F.pad(d_close.unsqueeze(1), (period - 1, 0), mode="replicate")
    sma = F.avg_pool1d(c_pad, kernel_size=period, stride=1).squeeze(1)
    # Rolling std via variance
    c_sq_pad = F.pad((d_close ** 2).unsqueeze(1), (period - 1, 0), mode="replicate")
    mean_sq = F.avg_pool1d(c_sq_pad, kernel_size=period, stride=1).squeeze(1)
    variance = (mean_sq - sma ** 2).clamp(min=0)
    std = torch.sqrt(variance)
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower


# ─── 3D Batch Simulation Engine (Verified 13/13) ────────────────────────────
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
        exit_px = torch.where(sl_exits, sl_p, torch.where(tp_exits, tp_p, fut_c_eod))
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
# PHASE 4 STRATEGY GENERATORS — ULTRA-WIDE + NEW RESEARCH FILTERS
# ==============================================================================
def build_session_window(trial):
    valid_window = torch.zeros_like(d_close, dtype=torch.bool)
    start_off = trial.suggest_int("sess_start_off", 0, 30, step=5)
    end_off = trial.suggest_int("sess_end_off", 0, 60, step=15)
    valid_window[:, BASE_SESSION_START + start_off : BASE_SESSION_END - end_off] = True
    return valid_window

def get_daily_limits(trial):
    loss_choice = trial.suggest_categorical("daily_loss_pts", [5, 8, 10, 15, 20, 25, 30, 40, 50, 9999])
    profit_choice = trial.suggest_categorical("daily_profit_pts", [15, 20, 30, 40, 50, 60, 80, 100, 9999])
    return float(loss_choice) * LOT_SIZE, float(profit_choice) * LOT_SIZE


def build_phase4_strategy(strat_id, trial):
    daily_loss, daily_profit = get_daily_limits(trial)
    vw = build_session_window(trial)

    # Common ultra-wide stochastic params
    s1_k = trial.suggest_int("s1_k", 3, 25)
    s4_k = trial.suggest_int("s4_k", 20, 120, step=5)
    s1 = get_stoch(s1_k)
    s4 = get_stoch(s4_k)
    s4_ob = trial.suggest_float("s4_ob", 65.0, 95.0, step=2.5)
    s1_os = trial.suggest_float("s1_os", 5.0, 40.0, step=2.5)

    # Common ultra-wide ATR/exit params
    atr_p = trial.suggest_int("atr_p", 5, 35, step=1)
    atr = get_atr(atr_p)
    sl_m = trial.suggest_float("sl_m", 0.5, 5.0, step=0.1)
    tp_m = trial.suggest_float("tp_m", 1.0, 10.0, step=0.25)

    if tp_m < 1.5 * sl_m:
        raise optuna.TrialPruned("R:R constraint")

    if strat_id == "F01":  # C02 Ultra-Wide — No Boundaries
        entries = (s4 >= s4_ob) & (s1 <= s1_os) & vw

    elif strat_id == "F02":  # C02 + RSI Oversold Confirmation
        rsi_p = trial.suggest_int("rsi_p", 7, 21, step=2)
        rsi_thresh = trial.suggest_float("rsi_thresh", 20.0, 45.0, step=2.5)
        rsi = get_rsi(rsi_p)
        entries = (s4 >= s4_ob) & (s1 <= s1_os) & (rsi <= rsi_thresh) & vw

    elif strat_id == "F03":  # C02 + EMA Trend Filter (price > EMA = bullish)
        ema_p = trial.suggest_int("ema_p", 10, 50, step=5)
        ema = get_ema(ema_p)
        entries = (s4 >= s4_ob) & (s1 <= s1_os) & (d_close > ema) & vw

    elif strat_id == "F04":  # C02 + Bollinger Band Touch (price near lower band)
        bb_p = trial.suggest_int("bb_p", 15, 30, step=5)
        bb_std = trial.suggest_float("bb_std", 1.5, 2.5, step=0.25)
        bb_prox = trial.suggest_float("bb_proximity", 0.0, 0.5, step=0.1)
        sma, upper, lower = get_bollinger(bb_p, bb_std)
        band_width = upper - lower
        proximity_level = lower + band_width * bb_prox
        entries = (s4 >= s4_ob) & (s1 <= s1_os) & (d_close <= proximity_level) & vw

    elif strat_id == "F05":  # C02 + Double Stochastic (medium period confirmation)
        s_mid_k = trial.suggest_int("s_mid_k", 20, 50, step=5)
        s_mid = get_stoch(s_mid_k)
        s_mid_ob = trial.suggest_float("s_mid_ob", 60.0, 85.0, step=5.0)
        entries = (s4 >= s4_ob) & (s1 <= s1_os) & (s_mid >= s_mid_ob) & vw

    else:
        raise ValueError(f"Unknown: {strat_id}")

    sl = d_close - (atr * sl_m)
    tp = d_close + (atr * tp_m)
    return entries, sl, tp, False, 0.0, 0.0, daily_loss, daily_profit


STRAT_IDS = ["F01", "F02", "F03", "F04", "F05"]
STRAT_NAMES = {
    "F01": "F01: C02 Ultra-Wide No-Boundary",
    "F02": "F02: C02 + RSI Oversold Confirmation",
    "F03": "F03: C02 + EMA Trend Filter",
    "F04": "F04: C02 + Bollinger Band Lower Touch",
    "F05": "F05: C02 + Double Stochastic Confirmation",
}


# ==============================================================================
# OPTUNA BATCH RUNNER — Triple-Weighted Objective
# ==============================================================================
def optimize_batch(strat_id, day_mask=None, n_total_trials=1000, batch_size=50):
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, constant_liar=True)
    )
    n_batches = max(1, n_total_trials // batch_size)

    for _ in range(n_batches):
        batch_trials = [study.ask() for _ in range(batch_size)]
        for trial in batch_trials:
            try:
                entries, sl, tp, is_tr, trig, step, d_loss, d_prof = build_phase4_strategy(strat_id, trial)
                res = simulate_gpu_with_limits(entries, sl, tp, day_mask=day_mask,
                                                is_trailing=is_tr, trail_trigger=trig, trail_step=step,
                                                max_daily_loss=d_loss, max_daily_profit=d_prof)
                n_tr = res["trades"]
                min_trades = 30 if day_mask is not None else 50

                if n_tr < min_trades or res["net_rs"] <= 0:
                    score = -999.0
                else:
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


def run_benchmark(strat_id, idx, total):
    name = STRAT_NAMES[strat_id]
    print(f"\n[{idx:02d}/{total}] PHASE 4: {name}", flush=True)

    t0 = time.time()
    best_nw = optimize_batch(strat_id, day_mask=None, n_total_trials=TRIALS_PER_STRATEGY)
    t_nw = time.time() - t0

    t1 = time.time()
    best_wf = optimize_batch(strat_id, day_mask=d_is_mask, n_total_trials=TRIALS_PER_STRATEGY)
    t_wf = time.time() - t1

    fixed = optuna.trial.FixedTrial(best_wf.params)
    ent, sl, tp, is_tr, trig, step, dl, dp = build_phase4_strategy(strat_id, fixed)
    oos = simulate_gpu_with_limits(ent, sl, tp, day_mask=d_oos_mask,
                                    is_trailing=is_tr, trail_trigger=trig, trail_step=step,
                                    max_daily_loss=dl, max_daily_profit=dp)

    is_ann = best_wf.user_attrs.get("net_rs", 0.0) / 4.0
    oos_ann = oos.get("net_rs", 0.0) / 2.35
    wfe = round(oos_ann / is_ann, 2) if is_ann > 0 else 0.0

    nw = best_nw.user_attrs
    print(f"  [NW 7Y {t_nw:.0f}s]: WR={nw.get('win_rate',0):.1f}% PF={nw.get('pf',0):.2f} Net=Rs {nw.get('net_rs',0):+,.0f} DD=Rs {nw.get('max_dd',0):,.0f} Trades={nw.get('trades',0)}", flush=True)
    print(f"  [WF {t_wf:.0f}s]: IS=Rs {best_wf.user_attrs.get('net_rs',0):+,.0f} -> OOS=Rs {oos['net_rs']:+,.0f} (PF {oos['pf']:.2f} WR {oos['win_rate']:.1f}%) WFE={wfe:.2f}", flush=True)
    print(f"  NW params: {best_nw.params}", flush=True)
    print(f"  WF params: {best_wf.params}", flush=True)

    return {
        "id": strat_id, "name": name,
        "non_wf": {
            "best_params": best_nw.params,
            "win_rate": nw.get("win_rate", 0.0), "pf": nw.get("pf", 0.0),
            "net_pts": nw.get("net_pts", 0.0), "net_rs": nw.get("net_rs", 0.0),
            "max_dd": nw.get("max_dd", 0.0), "trades": nw.get("trades", 0),
            "score": round(best_nw.value, 4), "time_s": round(t_nw, 2)
        },
        "walk_forward": {
            "is_params": best_wf.params,
            "is_wr": best_wf.user_attrs.get("win_rate", 0.0),
            "is_pf": best_wf.user_attrs.get("pf", 0.0),
            "is_net_rs": best_wf.user_attrs.get("net_rs", 0.0),
            "is_net_pts": best_wf.user_attrs.get("net_pts", 0.0),
            "is_max_dd": best_wf.user_attrs.get("max_dd", 0.0),
            "is_trades": best_wf.user_attrs.get("trades", 0),
            "oos_wr": oos["win_rate"], "oos_pf": oos["pf"],
            "oos_net_pts": oos["net_pts"], "oos_net_rs": oos["net_rs"],
            "oos_max_dd": oos["max_dd"], "oos_trades": oos["trades"],
            "wfe": wfe, "time_s": round(t_wf, 2)
        }
    }


def main():
    total = len(STRAT_IDS)
    print("=" * 140, flush=True)
    print(f"PHASE 4 — ULTIMATE EXHAUSTIVE GPU SEARCH ({total} STRATEGIES x {TRIALS_PER_STRATEGY} TRIALS)")
    print(f"Total: {total * TRIALS_PER_STRATEGY * 2:,} GPU Trials | Ultra-Wide Ranges | New Research Filters")
    print("=" * 140, flush=True)

    t_start = time.time()
    results = [run_benchmark(sid, i+1, total) for i, sid in enumerate(STRAT_IDS)]
    total_time = time.time() - t_start

    by_nw = sorted(results, key=lambda x: x["non_wf"]["net_rs"], reverse=True)
    by_oos = sorted(results, key=lambda x: (x["walk_forward"]["oos_net_rs"], x["walk_forward"]["wfe"]), reverse=True)

    nw_rank = {r["id"]: i+1 for i, r in enumerate(by_nw)}
    oos_rank = {r["id"]: i+1 for i, r in enumerate(by_oos)}

    print(f"\n{'='*140}")
    print(f"PHASE 4 LEADERBOARD ({total * TRIALS_PER_STRATEGY * 2:,} TRIALS IN {total_time:.1f}s)")
    print(f"{'='*140}")

    print(f"\n{'='*60} SIDE-BY-SIDE: NW vs OOS {'='*60}")
    print(f"{'NW#':4s} {'OOS#':5s} | {'Strategy':46s} | {'NW PnL':>14s} | {'NW PF':>7s} | {'NW WR':>7s} | {'OOS PnL':>14s} | {'OOS PF':>7s} | {'OOS WR':>7s} | {'OOS DD':>12s} | {'WFE':>5s}")
    print("-" * 155)
    for r in by_nw:
        nw = r["non_wf"]
        wf = r["walk_forward"]
        nr = nw_rank[r["id"]]
        or_ = oos_rank[r["id"]]
        flag = " ***" if or_ <= 2 else ""
        print(f"[{nr:2d}] [{or_:2d}]  | {r['name']:46s} | Rs {nw['net_rs']:+11,.0f} | {nw['pf']:6.2f} | {nw['win_rate']:5.1f}% | Rs {wf['oos_net_rs']:+11,.0f} | {wf['oos_pf']:6.2f} | {wf['oos_wr']:5.1f}% | Rs {wf['oos_max_dd']:9,.0f} | {wf['wfe']:4.2f}{flag}", flush=True)
    print("-" * 155)

    # Parameter summary for OOS champion
    champ = by_oos[0]
    print(f"\n{'='*60} OOS CHAMPION PARAMETERS {'='*60}")
    print(f"Strategy: {champ['name']}")
    print(f"WF Params: {champ['walk_forward']['is_params']}")
    print(f"NW Params: {champ['non_wf']['best_params']}")

    out = ROOT / "artifacts" / "f6_hybrid" / "master_phase4_ultimate.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_s": total_time,
            "trials_per_strategy": TRIALS_PER_STRATEGY,
            "total_trials": total * TRIALS_PER_STRATEGY * 2,
            "results": by_oos
        }, f, indent=2)
    print(f"\nSaved to: {out}", flush=True)


if __name__ == "__main__":
    main()
