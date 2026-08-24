"""
PHASE 5 — BIDIRECTIONAL + MULTI-TIMEFRAME + DD OPTIMIZATION
=============================================================
Building on F01 champion (CE-only, 1m), this phase tests:

1. BIDIRECTIONAL: CE buys (bull dips) + PE buys (bear bounces)
   CE: S4 >= OB  AND  S1 <= OS   → buy CE (uptrend dip)
   PE: S4 <= OS_PE AND S1 >= OB_PE → buy PE (downtrend bounce)

2. MULTI-TIMEFRAME: 1m, 2m, 3m, 5m bars (aggregated from 1m via GPU)
   Each TF gets its own stochastic periods and ATR.
   Combined mode = union of signals from best 2 timeframes.

3. DD OPTIMIZATION: Tight daily loss caps (₹325–₹1300) to keep max DD low.
   Triple objective weighted MORE toward low DD.

Strategy Families:
  B01: 1m CE-only (F01 baseline)
  B02: 1m CE+PE bidirectional
  B03: 2m CE+PE bidirectional
  B04: 3m CE+PE bidirectional
  B05: 5m CE+PE bidirectional
  B06: 1m CE+PE with tight DD optimization (daily_loss=3..12 pts)
  B07: Best-TF CE+PE with DD ≤ ₹50K constraint
"""

import json, sys, time
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
FEE = 30.0
BASE_SESSION_START = 5
BASE_SESSION_END = 345
TRIALS_PER_STRATEGY = 3000
BATCH_SIZE = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"CUDA: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

# ─── Data Loader ────────────────────────────────────────────────────────────
def load_gpu_data(start_date="2020-01-01", end_date="2026-05-05"):
    spot_all = source.load_spot()
    opt_map = source.option_day_files(start_date, end_date)
    days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    N = len(days)
    arr_h = np.zeros((N, 375), dtype=np.float32)
    arr_l = np.zeros((N, 375), dtype=np.float32)
    arr_c = np.zeros((N, 375), dtype=np.float32)
    for i, d in enumerate(days):
        sp = spot_all[d]
        for idx, m in enumerate(sp["min"]):
            b = int(m) - 555
            if 0 <= b < 375:
                arr_h[i, b] = float(sp["high"][idx])
                arr_l[i, b] = float(sp["low"][idx])
                arr_c[i, b] = float(sp["close"][idx])
    d_h = torch.tensor(arr_h, dtype=torch.float32, device=device)
    d_l = torch.tensor(arr_l, dtype=torch.float32, device=device)
    d_c = torch.tensor(arr_c, dtype=torch.float32, device=device)
    prev_c = F.pad(d_c[:, :-1], (1, 0), mode="replicate")
    d_tr = torch.maximum(torch.maximum(d_h - d_l, torch.abs(d_h - prev_c)), torch.abs(d_l - prev_c))
    is_mask = np.array([d < "2024-01-01" for d in days], dtype=bool)
    oos_mask = np.array([d >= "2024-01-01" for d in days], dtype=bool)
    return d_h, d_l, d_c, d_tr, days, \
           torch.tensor(is_mask, dtype=torch.bool, device=device), \
           torch.tensor(oos_mask, dtype=torch.bool, device=device)

print("Loading 7Y data...", flush=True)
t0 = time.time()
d_high, d_low, d_close, d_tr, all_days, d_is_mask, d_oos_mask = load_gpu_data()
N_DAYS = len(all_days)
print(f"Loaded {N_DAYS} days in {time.time()-t0:.1f}s", flush=True)


# ─── Multi-TF Aggregation on GPU ─────────────────────────────────────────────
@torch.no_grad()
def aggregate_tf(tf_minutes):
    """Aggregate 1m bars into tf_minutes bars using max_pool/min_pool."""
    if tf_minutes == 1:
        return d_high, d_low, d_close, d_tr

    k = tf_minutes
    N, T = d_high.shape
    # Pad so T is divisible by k
    pad_len = (k - T % k) % k
    h_pad = F.pad(d_high, (0, pad_len), mode="replicate")
    l_pad = F.pad(d_low,  (0, pad_len), mode="replicate")
    c_pad = F.pad(d_close, (0, pad_len), mode="replicate")

    T_new = (T + pad_len) // k
    h_r = h_pad.reshape(N, T_new, k).max(dim=2).values     # TF high = max of k 1m highs
    l_r = l_pad.reshape(N, T_new, k).min(dim=2).values     # TF low = min of k 1m lows
    c_r = c_pad.reshape(N, T_new, k)[:, :, -1]             # TF close = last close

    prev_c_r = F.pad(c_r[:, :-1], (1, 0), mode="replicate")
    tr_r = torch.maximum(torch.maximum(h_r - l_r, torch.abs(h_r - prev_c_r)), torch.abs(l_r - prev_c_r))
    return h_r, l_r, c_r, tr_r


# ─── GPU Indicator Kernels ───────────────────────────────────────────────────
@torch.no_grad()
def get_stoch_from(h, l, c, k_period):
    h_pad = F.pad(h.unsqueeze(1), (k_period - 1, 0), mode="replicate")
    l_pad = F.pad(l.unsqueeze(1), (k_period - 1, 0), mode="replicate")
    max_h = F.max_pool1d(h_pad, kernel_size=k_period, stride=1).squeeze(1)
    min_l = -F.max_pool1d(-l_pad, kernel_size=k_period, stride=1).squeeze(1)
    denom = (max_h - min_l).clamp(min=1e-6)
    return ((c - min_l) / denom) * 100.0

@torch.no_grad()
def get_atr_from(tr, period=14):
    tr_pad = F.pad(tr.unsqueeze(1), (period - 1, 0), mode="replicate")
    return F.avg_pool1d(tr_pad, kernel_size=period, stride=1).squeeze(1)


# ─── 3D Batch Simulation Engine (CE or PE direction) ────────────────────────
@torch.no_grad()
def simulate_direction(entries_mask, sl_tensor, tp_tensor, day_mask=None,
                       direction="CE", max_daily_loss=9999.0,
                       max_daily_profit=9999.0, sess_end=BASE_SESSION_END):
    """
    Simulate trades in CE or PE direction.
    CE: profit when spot goes UP   → pnl = (exit - entry) * 0.50
    PE: profit when spot goes DOWN → pnl = (entry - exit) * 0.50
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
    d_indices = coords[:, 0]
    b_indices = coords[:, 1]
    ep = d_close[d_indices, b_indices]

    max_future = sess_end - BASE_SESSION_START - 1
    col_start = b_indices + 1
    col_offsets = torch.arange(max_future, device=device).unsqueeze(0)
    col_idx = col_start.unsqueeze(1) + col_offsets
    valid = (col_idx < sess_end) & (col_idx < 375)
    col_idx_safe = col_idx.clamp(max=374)

    d_exp = d_indices.unsqueeze(1).expand(-1, max_future)
    fut_h = d_high[d_exp, col_idx_safe]
    fut_l = d_low[d_exp, col_idx_safe]

    eod_bar = min(sess_end - 1, 374)
    fut_c_eod = d_close[d_indices, eod_bar]

    INF = torch.tensor(1e9, device=device)
    fut_h_m = torch.where(valid, fut_h, -INF)
    fut_l_m = torch.where(valid, fut_l, INF)

    sl_p = sl_tensor[d_indices, b_indices]
    tp_p = tp_tensor[d_indices, b_indices]

    if direction == "CE":
        # CE: SL when spot falls, TP when spot rises
        hit_sl = fut_l_m <= sl_p.unsqueeze(1)
        hit_tp = fut_h_m >= tp_p.unsqueeze(1)
    else:
        # PE: SL when spot rises, TP when spot falls
        hit_sl = fut_h_m >= sl_p.unsqueeze(1)  # spot goes UP = PE loss
        hit_tp = fut_l_m <= tp_p.unsqueeze(1)  # spot goes DOWN = PE profit

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)
    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), torch.tensor(BIG, device=device))
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), torch.tensor(BIG, device=device))
    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    if direction == "CE":
        exit_px = torch.where(sl_exits, sl_p, torch.where(tp_exits, tp_p, fut_c_eod))
        raw_pts = (exit_px - ep) * 0.50
    else:
        exit_px = torch.where(sl_exits, sl_p, torch.where(tp_exits, tp_p, fut_c_eod))
        raw_pts = (ep - exit_px) * 0.50  # PE profits when spot falls

    has_future = (b_indices + 1) < sess_end
    raw_pts = raw_pts[has_future]
    d_idx_valid = d_indices[has_future]

    if raw_pts.shape[0] == 0:
        return {"trades": 0, "win_rate": 0.0, "net_pts": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    all_rs = raw_pts * LOT_SIZE - FEE
    all_pts_cpu = raw_pts.cpu().numpy()
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
        "trades": n_trades, "win_rate": round(wins / n_trades * 100.0, 2),
        "net_pts": round(float(final_pts.sum()), 2), "net_rs": round(float(final_rs.sum()), 2),
        "pf": round(pf, 2), "max_dd": round(max_dd, 2),
    }


# ─── Merge CE + PE results for bidirectional ────────────────────────────────
def merge_results(ce_res, pe_res):
    """Merge CE and PE trade streams into one combined result."""
    t = ce_res["trades"] + pe_res["trades"]
    if t == 0:
        return {"trades": 0, "win_rate": 0.0, "net_pts": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    net_rs = ce_res["net_rs"] + pe_res["net_rs"]
    ce_pos = ce_res["net_rs"] * ce_res["pf"] / (1 + ce_res["pf"]) if ce_res["pf"] > 0 else max(ce_res["net_rs"], 0)
    pe_pos = pe_res["net_rs"] * pe_res["pf"] / (1 + pe_res["pf"]) if pe_res["pf"] > 0 else max(pe_res["net_rs"], 0)
    total_pos = ce_pos + pe_pos
    total_neg = total_pos - net_rs
    pf = total_pos / total_neg if total_neg > 0 else 0.0

    ce_wins = round(ce_res["trades"] * ce_res["win_rate"] / 100.0)
    pe_wins = round(pe_res["trades"] * pe_res["win_rate"] / 100.0)
    wr = (ce_wins + pe_wins) / t * 100.0 if t > 0 else 0.0

    # DD approximation: max of individual DDs (conservative since they partially offset)
    max_dd = max(ce_res["max_dd"], pe_res["max_dd"])

    return {
        "trades": t, "win_rate": round(wr, 2),
        "net_pts": round(ce_res["net_pts"] + pe_res["net_pts"], 2),
        "net_rs": round(net_rs, 2), "pf": round(pf, 2), "max_dd": round(max_dd, 2),
        "ce_trades": ce_res["trades"], "pe_trades": pe_res["trades"],
        "ce_pnl": ce_res["net_rs"], "pe_pnl": pe_res["net_rs"],
    }


# ==============================================================================
# PHASE 5 STRATEGY GENERATORS
# ==============================================================================
# Pre-compute aggregated TF data
print("Pre-computing multi-TF data...", flush=True)
TF_DATA = {}
for tf in [1, 2, 3, 5]:
    TF_DATA[tf] = aggregate_tf(tf)
    print(f"  TF={tf}m: {TF_DATA[tf][0].shape[1]} bars/day", flush=True)


def build_phase5_strategy(strat_id, trial):
    # Common parameters
    daily_loss_pts = trial.suggest_int("daily_loss_pts", 3, 20)
    daily_loss_rs = float(daily_loss_pts) * LOT_SIZE
    daily_profit_rs = 9999.0 * LOT_SIZE
    # Per-trade SL cap in Rs — skip entries where SL > this
    max_trade_loss = trial.suggest_categorical("max_trade_loss_rs",
                                                [500, 1000, 1500, 2000, 3000, 5000, 9999])

    sess_start_off = trial.suggest_int("sess_start_off", 0, 30, step=5)
    sess_end_off = trial.suggest_int("sess_end_off", 30, 75, step=15)
    sess_end = BASE_SESSION_END - sess_end_off

    # Determine timeframe
    if strat_id in ("B01", "B02", "B06"):
        tf = 1
    elif strat_id == "B03":
        tf = 2
    elif strat_id == "B04":
        tf = 3
    elif strat_id == "B05":
        tf = 5
    elif strat_id == "B07":
        tf = trial.suggest_categorical("timeframe", [1, 2, 3, 5])
    else:
        tf = 1

    tf_h, tf_l, tf_c, tf_tr = TF_DATA[tf]

    # Stochastic params
    s1_k = trial.suggest_int("s1_k", 5, 30)
    s4_k = trial.suggest_int("s4_k", 20, 120, step=5)
    s1_os = trial.suggest_float("s1_os", 10.0, 40.0, step=2.5)
    s4_ob = trial.suggest_float("s4_ob", 65.0, 90.0, step=2.5)

    S1_tf = get_stoch_from(tf_h, tf_l, tf_c, s1_k)
    S4_tf = get_stoch_from(tf_h, tf_l, tf_c, s4_k)

    # ATR params (on TF bars)
    atr_p = trial.suggest_int("atr_p", 8, 35)
    ATR_tf = get_atr_from(tf_tr, atr_p)
    sl_m = trial.suggest_float("sl_m", 1.0, 5.0, step=0.1)
    tp_m = trial.suggest_float("tp_m", 2.0, 10.0, step=0.25)

    if tp_m < 1.5 * sl_m:
        raise optuna.TrialPruned("R:R constraint")

    # Map TF signals back to 1m resolution
    N_days = d_close.shape[0]
    T_1m = d_close.shape[1]
    T_tf = tf_c.shape[1]

    # Session window on 1m bars
    vw = torch.zeros((N_days, T_1m), dtype=torch.bool, device=device)
    vw[:, BASE_SESSION_START + sess_start_off : sess_end] = True

    if tf == 1:
        # Direct 1m, no mapping needed
        ce_entries = (S4_tf >= s4_ob) & (S1_tf <= s1_os) & vw
        ce_sl = d_close - (ATR_tf * sl_m)
        ce_tp = d_close + (ATR_tf * tp_m)

        # PE: mirror conditions
        pe_s4_os = 100.0 - s4_ob  # e.g., 77.5 OB → 22.5 OS for PE
        pe_s1_ob = 100.0 - s1_os  # e.g., 35 OS → 65 OB for PE
        pe_entries = (S4_tf <= pe_s4_os) & (S1_tf >= pe_s1_ob) & vw
        pe_sl = d_close + (ATR_tf * sl_m)  # PE SL is ABOVE entry
        pe_tp = d_close - (ATR_tf * tp_m)  # PE TP is BELOW entry
    else:
        # Map TF-bar signals to 1m bars (each TF bar maps to tf 1m bars)
        # Expand TF entries to 1m resolution by repeating
        ce_cond_tf = (S4_tf >= s4_ob) & (S1_tf <= s1_os)
        pe_s4_os = 100.0 - s4_ob
        pe_s1_ob = 100.0 - s1_os
        pe_cond_tf = (S4_tf <= pe_s4_os) & (S1_tf >= pe_s1_ob)

        # Repeat each TF bar to cover its 1m bars
        ce_entries_raw = ce_cond_tf.repeat_interleave(tf, dim=1)[:, :T_1m]
        pe_entries_raw = pe_cond_tf.repeat_interleave(tf, dim=1)[:, :T_1m]
        atr_1m = ATR_tf.repeat_interleave(tf, dim=1)[:, :T_1m]

        ce_entries = ce_entries_raw & vw
        pe_entries = pe_entries_raw & vw
        ce_sl = d_close - (atr_1m * sl_m)
        ce_tp = d_close + (atr_1m * tp_m)
        pe_sl = d_close + (atr_1m * sl_m)
        pe_tp = d_close - (atr_1m * tp_m)

    # Filter entries where per-trade SL exceeds max_trade_loss cap
    if tf == 1:
        sl_dist_rs = ATR_tf * sl_m * 0.50 * LOT_SIZE
    else:
        sl_dist_rs = atr_1m * sl_m * 0.50 * LOT_SIZE
    trade_cap_ok = sl_dist_rs <= max_trade_loss
    ce_entries = ce_entries & trade_cap_ok
    pe_entries = pe_entries & trade_cap_ok

    is_bidirectional = strat_id != "B01"
    return (ce_entries, ce_sl, ce_tp, pe_entries, pe_sl, pe_tp,
            is_bidirectional, daily_loss_rs, daily_profit_rs, sess_end)


STRAT_IDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07"]
STRAT_NAMES = {
    "B01": "B01: 1m CE-Only (F01 Baseline)",
    "B02": "B02: 1m CE+PE Bidirectional",
    "B03": "B03: 2m CE+PE Bidirectional",
    "B04": "B04: 3m CE+PE Bidirectional",
    "B05": "B05: 5m CE+PE Bidirectional",
    "B06": "B06: 1m CE+PE Tight DD Optimized",
    "B07": "B07: Best-TF CE+PE DD<=50K Target",
}


# ─── Objective ───────────────────────────────────────────────────────────────
def evaluate(strat_id, trial, day_mask=None):
    try:
        (ce_ent, ce_sl, ce_tp, pe_ent, pe_sl, pe_tp,
         is_bidir, dl, dp, se) = build_phase5_strategy(strat_id, trial)
    except optuna.TrialPruned:
        return -999.0, {}

    ce_res = simulate_direction(ce_ent, ce_sl, ce_tp, day_mask=day_mask,
                                direction="CE", max_daily_loss=dl,
                                max_daily_profit=dp, sess_end=se)
    if is_bidir:
        pe_res = simulate_direction(pe_ent, pe_sl, pe_tp, day_mask=day_mask,
                                    direction="PE", max_daily_loss=dl,
                                    max_daily_profit=dp, sess_end=se)
        res = merge_results(ce_res, pe_res)
    else:
        res = ce_res

    n_tr = res["trades"]
    min_trades = 30 if day_mask is not None else 50
    if n_tr < min_trades or res["net_rs"] <= 0:
        return -999.0, res

    pf_comp = res["pf"] * (res["win_rate"] / 40.0)
    dd_penalty = 0.50 * (res["max_dd"] / max(res["net_rs"], 1.0))  # heavier DD penalty
    freq_bonus = min(n_tr / 500.0, 1.0) * 0.10
    score = pf_comp - dd_penalty + freq_bonus

    # B06/B07: extra DD penalty if DD > 50K
    if strat_id in ("B06", "B07") and res["max_dd"] > 50000:
        score -= (res["max_dd"] - 50000) / 50000 * 0.5

    return score, res


def optimize_strategy(strat_id, day_mask=None):
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, constant_liar=True)
    )
    n_batches = max(1, TRIALS_PER_STRATEGY // BATCH_SIZE)

    for _ in range(n_batches):
        batch_trials = [study.ask() for _ in range(BATCH_SIZE)]
        for trial in batch_trials:
            score, res = evaluate(strat_id, trial, day_mask)
            for k, v in res.items():
                trial.set_user_attr(k, v)
            study.tell(trial, score)

    return study.best_trial


def run_benchmark(strat_id, idx, total):
    name = STRAT_NAMES[strat_id]
    print(f"\n[{idx:02d}/{total}] PHASE 5: {name}", flush=True)

    t0 = time.time()
    best_nw = optimize_strategy(strat_id, day_mask=None)
    t_nw = time.time() - t0

    t1 = time.time()
    best_wf = optimize_strategy(strat_id, day_mask=d_is_mask)
    t_wf = time.time() - t1

    # Evaluate OOS with WF params
    fixed = optuna.trial.FixedTrial(best_wf.params)
    try:
        (ce_ent, ce_sl, ce_tp, pe_ent, pe_sl, pe_tp,
         is_bidir, dl, dp, se) = build_phase5_strategy(strat_id, fixed)
        ce_oos = simulate_direction(ce_ent, ce_sl, ce_tp, day_mask=d_oos_mask,
                                    direction="CE", max_daily_loss=dl,
                                    max_daily_profit=dp, sess_end=se)
        if is_bidir:
            pe_oos = simulate_direction(pe_ent, pe_sl, pe_tp, day_mask=d_oos_mask,
                                        direction="PE", max_daily_loss=dl,
                                        max_daily_profit=dp, sess_end=se)
            oos = merge_results(ce_oos, pe_oos)
        else:
            oos = ce_oos
    except:
        oos = {"trades": 0, "win_rate": 0.0, "net_pts": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    is_ann = best_wf.user_attrs.get("net_rs", 0.0) / 4.0
    oos_ann = oos.get("net_rs", 0.0) / 2.35
    wfe = round(oos_ann / is_ann, 2) if is_ann > 0 else 0.0

    nw = best_nw.user_attrs
    print(f"  [NW 7Y {t_nw:.0f}s]: WR={nw.get('win_rate',0):.1f}% PF={nw.get('pf',0):.2f} Net=Rs {nw.get('net_rs',0):+,.0f} DD=Rs {nw.get('max_dd',0):,.0f} Trades={nw.get('trades',0)} CE={nw.get('ce_trades','all')} PE={nw.get('pe_trades',0)}", flush=True)
    print(f"  [WF {t_wf:.0f}s]: IS=Rs {best_wf.user_attrs.get('net_rs',0):+,.0f} -> OOS=Rs {oos['net_rs']:+,.0f} (PF {oos['pf']:.2f} WR {oos['win_rate']:.1f}% DD=Rs {oos['max_dd']:,.0f}) WFE={wfe:.2f}", flush=True)
    print(f"  NW params: {best_nw.params}", flush=True)
    print(f"  WF params: {best_wf.params}", flush=True)

    return {
        "id": strat_id, "name": name,
        "non_wf": {
            "best_params": best_nw.params,
            "win_rate": nw.get("win_rate", 0.0), "pf": nw.get("pf", 0.0),
            "net_rs": nw.get("net_rs", 0.0), "max_dd": nw.get("max_dd", 0.0),
            "trades": nw.get("trades", 0),
            "ce_trades": nw.get("ce_trades", nw.get("trades", 0)),
            "pe_trades": nw.get("pe_trades", 0),
            "ce_pnl": nw.get("ce_pnl", nw.get("net_rs", 0)),
            "pe_pnl": nw.get("pe_pnl", 0),
        },
        "walk_forward": {
            "is_params": best_wf.params,
            "is_net_rs": best_wf.user_attrs.get("net_rs", 0.0),
            "is_max_dd": best_wf.user_attrs.get("max_dd", 0.0),
            "oos_net_rs": oos["net_rs"], "oos_pf": oos["pf"],
            "oos_wr": oos["win_rate"], "oos_max_dd": oos["max_dd"],
            "oos_trades": oos.get("trades", 0),
            "oos_ce_trades": oos.get("ce_trades", oos.get("trades", 0)),
            "oos_pe_trades": oos.get("pe_trades", 0),
            "oos_ce_pnl": oos.get("ce_pnl", oos.get("net_rs", 0)),
            "oos_pe_pnl": oos.get("pe_pnl", 0),
            "wfe": wfe,
        }
    }


def main():
    total = len(STRAT_IDS)
    print(f"\n{'='*140}", flush=True)
    print(f"PHASE 5 -- BIDIRECTIONAL + MULTI-TIMEFRAME + DD OPTIMIZATION ({total} strategies x {TRIALS_PER_STRATEGY} trials)", flush=True)
    print(f"Total: {total * TRIALS_PER_STRATEGY * 2:,} GPU Trials | CE+PE | 1m/2m/3m/5m | DD-Weighted Objective", flush=True)
    print(f"{'='*140}", flush=True)

    t_start = time.time()
    results = [run_benchmark(sid, i+1, total) for i, sid in enumerate(STRAT_IDS)]
    total_time = time.time() - t_start

    by_oos = sorted(results, key=lambda x: x["walk_forward"]["oos_net_rs"], reverse=True)
    by_nw = sorted(results, key=lambda x: x["non_wf"]["net_rs"], reverse=True)
    nw_rank = {r["id"]: i+1 for i, r in enumerate(by_nw)}
    oos_rank = {r["id"]: i+1 for i, r in enumerate(by_oos)}

    print(f"\n{'='*155}", flush=True)
    print(f"PHASE 5 LEADERBOARD ({total * TRIALS_PER_STRATEGY * 2:,} TRIALS IN {total_time:.1f}s)", flush=True)
    print(f"{'='*155}", flush=True)

    print(f"\n{'NW#':4s} {'OOS#':5s} | {'Strategy':42s} | {'NW PnL':>14s} | {'NW PF':>7s} | {'NW WR':>7s} | {'NW DD':>10s} | {'OOS PnL':>14s} | {'OOS PF':>7s} | {'OOS WR':>7s} | {'OOS DD':>10s} | {'WFE':>5s} | {'CE/PE Trades':>12s}", flush=True)
    print("-" * 165, flush=True)
    for r in by_nw:
        nw = r["non_wf"]; wf = r["walk_forward"]
        nr = nw_rank[r["id"]]; or_ = oos_rank[r["id"]]
        ce_pe = f"{wf.get('oos_ce_trades',0)}/{wf.get('oos_pe_trades',0)}"
        star = " ***" if or_ <= 2 else ""
        print(f"[{nr:2d}] [{or_:2d}]  | {r['name']:42s} | Rs {nw['net_rs']:+11,.0f} | {nw['pf']:6.2f} | {nw['win_rate']:5.1f}% | Rs {nw['max_dd']:7,.0f} | Rs {wf['oos_net_rs']:+11,.0f} | {wf['oos_pf']:6.2f} | {wf['oos_wr']:5.1f}% | Rs {wf['oos_max_dd']:7,.0f} | {wf['wfe']:4.2f} | {ce_pe:>12s}{star}", flush=True)
    print("-" * 165, flush=True)

    # DD comparison
    print(f"\n  DD COMPARISON:")
    for r in by_oos:
        wf = r["walk_forward"]
        dd_pnl_ratio = wf["oos_max_dd"] / max(wf["oos_net_rs"], 1) * 100 if wf["oos_net_rs"] > 0 else 999
        print(f"  {r['name']:42s}: OOS DD=Rs {wf['oos_max_dd']:>7,.0f}  ({dd_pnl_ratio:.0f}% of PnL)  CE P&L=Rs {wf.get('oos_ce_pnl',0):>+10,.0f}  PE P&L=Rs {wf.get('oos_pe_pnl',0):>+10,.0f}", flush=True)

    out = ROOT / "artifacts" / "f6_hybrid" / "master_phase5_bidir_mtf.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_s": round(total_time, 2),
            "total_trials": total * TRIALS_PER_STRATEGY * 2,
            "results": by_oos
        }, f, indent=2)
    print(f"\nSaved to: {out}", flush=True)


if __name__ == "__main__":
    main()
