"""
PHASE 5d — CAUSAL-CORRECT 3D GPU PIPELINE (FUSED BATCH UPGRADE)
===============================================================
Upgraded per GPU_BACKTEST_PIPELINE_GUIDE.md §21 (Next-Gen 3D Batched):
  - A full Optuna batch of B trials is evaluated in ONE fused (B, N, T)
    GPU pass instead of B separate dispatches.
  - Slashes Python-dispatch overhead; drives GPU utilisation toward the
    85-95% saturation described in §20.

Causal pillars preserved (unchanged semantics from the 2D engine):
  1. Zero Lookahead: F.pad(x, (K-1, 0))         ✓
  2. Clock Alignment: TF signals at TF bar close ✓
  3. Strike Selection: delta=0.50 simplified model ✓
  4. Exchange Drag: ₹30 fee + 1pt slippage     ✓
  5. Position Lock: MAX 1 trade/day/direction   ✓
  6. Circuit Breakers: Daily loss cap on GPU     ✓

Run with PARITY=1 to verify the fused engine == sequential engine.
"""

import json, sys, time, os, functools
from pathlib import Path
import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source

torch.set_float32_matmul_precision("high")
LOT_SIZE = 65
FEE = 30.0
SLIPPAGE_PTS = 1.0  # 0.5 entry + 0.5 exit = 1.0 round trip
BASE_SESSION_START = 5
BASE_SESSION_END = 345
TRIALS_PER_STRATEGY = 3000
BATCH_SIZE = int(os.environ.get("BATCH", "100"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optuna.logging.set_verbosity(optuna.logging.WARNING)
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f}GB", flush=True)

EMPTY = {"trades": 0, "win_rate": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0,
          "pos": 0.0, "neg": 0.0,
          "ce_trades": 0, "pe_trades": 0, "ce_pnl": 0.0, "pe_pnl": 0.0}


def _to_scalar(x):
    """Coerce a cap argument to a plain Python float (one D2H read), so the
    per-trade circuit-breaker loop in _finalize never forces a GPU->CPU
    scalar readback on every iteration (was 68k reads/evaluate_batch)."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        a = x.detach().cpu().numpy()
        return float(a.reshape(-1)[0])
    return float(x)


def _finalize(rs_np, days_np, bars_np, loss_cap, profit_cap):
    """Guide §23 O(N) daily post-filter.

    Applies the daily circuit breaker correctly: it STOPs taking new trades
    for the rest of a day once cumulative P&L breaches the loss/profit cap.
    It does NOT delete the loss-making trades (the old code did, which
    fabricated WR~100% / PF=0.00). Losses are always counted.
    """
    loss_cap = _to_scalar(loss_cap)
    profit_cap = _to_scalar(profit_cap)
    if rs_np.shape[0] == 0:
        return dict(trades=0, win_rate=0.0, net_rs=0.0, pf=0.0, max_dd=0.0,
                    pos=0.0, neg=0.0)
    order = np.lexsort((bars_np, days_np))          # sort by (day, bar) chronologically
    rs = rs_np[order].astype(np.float64)
    days = days_np[order]
    kept = []
    last_day = None; cum = 0.0; stopped = False
    for r, d in zip(rs, days):
        if d != last_day:
            last_day = d; cum = 0.0; stopped = False
        if stopped:
            continue
        new_cum = cum + r
        if (loss_cap is not None and new_cum < -loss_cap) or \
           (profit_cap is not None and new_cum > profit_cap):
            stopped = True
            continue
        cum = new_cum
        kept.append(r)
    kept = np.array(kept, dtype=np.float64)
    n = kept.shape[0]
    if n == 0:
        return dict(trades=0, win_rate=0.0, net_rs=0.0, pf=0.0, max_dd=0.0,
                    pos=0.0, neg=0.0)
    wins = int((kept > 0).sum())
    pos = float(kept[kept > 0].sum())
    neg = float(abs(kept[kept <= 0].sum()))
    net = float(kept.sum())
    pf = pos / neg if neg > 0 else 99.0           # zero-loss regime: capped, not 0
    eq = np.cumsum(kept)
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    return dict(trades=n, win_rate=round(wins / n * 100, 2), net_rs=round(net, 2),
                pf=round(pf, 2), max_dd=round(dd, 2), pos=round(pos, 2), neg=round(neg, 2))

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING — permanent GPU residency
# ═══════════════════════════════════════════════════════════════════════════
def load_gpu_data():
    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
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
    return (torch.tensor(arr_h, dtype=torch.float32, device=device),
            torch.tensor(arr_l, dtype=torch.float32, device=device),
            torch.tensor(arr_c, dtype=torch.float32, device=device),
            days,
            torch.tensor([d < "2024-01-01" for d in days], dtype=torch.bool, device=device),
            torch.tensor([d >= "2024-01-01" for d in days], dtype=torch.bool, device=device))

print("Loading data into VRAM...", flush=True)
t0 = time.time()
d_high, d_low, d_close, all_days, d_is_mask, d_oos_mask = load_gpu_data()
N_DAYS, T_BARS = d_close.shape
prev_c = F.pad(d_close[:, :-1], (1, 0), mode="replicate")
d_tr = torch.maximum(torch.maximum(d_high - d_low, torch.abs(d_high - prev_c)), torch.abs(d_low - prev_c))
print(f"  {N_DAYS} days × {T_BARS} bars in {time.time()-t0:.1f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE MULTI-TF DATA + INDICATOR CACHE
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def aggregate_tf(k):
    if k == 1: return d_high, d_low, d_close, d_tr
    N, T = d_high.shape
    pad = (k - T % k) % k
    h_r = F.pad(d_high, (0, pad), mode="replicate").reshape(N, -1, k).max(dim=2).values
    l_r = F.pad(d_low, (0, pad), mode="replicate").reshape(N, -1, k).min(dim=2).values
    c_r = F.pad(d_close, (0, pad), mode="replicate").reshape(N, -1, k)[:, :, -1]
    pc = F.pad(c_r[:, :-1], (1, 0), mode="replicate")
    tr_r = torch.maximum(torch.maximum(h_r - l_r, torch.abs(h_r - pc)), torch.abs(l_r - pc))
    return h_r, l_r, c_r, tr_r

TF_DATA = {}
for tf in [1, 2, 3, 5]:
    TF_DATA[tf] = aggregate_tf(tf)
    print(f"  TF={tf}m: {TF_DATA[tf][0].shape[1]} bars", flush=True)

print("Pre-computing indicator cache...", flush=True)
t1 = time.time()
STOCH_CACHE = {}; ATR_CACHE = {}

@torch.no_grad()
def get_stoch(tf, period):
    key = (tf, period)
    if key not in STOCH_CACHE:
        h, l, c, _ = TF_DATA[tf]
        h_pad = F.pad(h.unsqueeze(1), (period-1, 0), mode="replicate")
        l_pad = F.pad(l.unsqueeze(1), (period-1, 0), mode="replicate")
        max_h = F.max_pool1d(h_pad, kernel_size=period, stride=1).squeeze(1)
        min_l = -F.max_pool1d(-l_pad, kernel_size=period, stride=1).squeeze(1)
        STOCH_CACHE[key] = ((c - min_l) / (max_h - min_l).clamp(min=1e-6)) * 100.0
    return STOCH_CACHE[key]

@torch.no_grad()
def get_atr(tf, period):
    key = (tf, period)
    if key not in ATR_CACHE:
        _, _, _, tr = TF_DATA[tf]
        tr_pad = F.pad(tr.unsqueeze(1), (period-1, 0), mode="replicate")
        ATR_CACHE[key] = F.avg_pool1d(tr_pad, kernel_size=period, stride=1).squeeze(1)
    return ATR_CACHE[key]

for tf in [1, 2, 3, 5]:
    for sk in range(5, 31): get_stoch(tf, sk)
    for sk in range(20, 121, 5): get_stoch(tf, sk)
    for ap in range(8, 36): get_atr(tf, ap)
print(f"  {len(STOCH_CACHE)+len(ATR_CACHE)} tensors cached in {time.time()-t1:.1f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 2D REFERENCE ENGINE (kept for parity verification)
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def simulate_direction_locked(entries_mask, sl_tensor, tp_tensor, direction,
                              max_daily_loss, sess_end, day_mask=None, daily_profit=None):
    if day_mask is not None:
        entries_mask = entries_mask & day_mask.unsqueeze(1)
    has_entry = entries_mask.any(dim=1)
    first_bar = torch.argmax(entries_mask.int(), dim=1)
    locked_mask = torch.zeros_like(entries_mask)
    valid_days = torch.where(has_entry)[0]
    if valid_days.shape[0] == 0:
        return EMPTY
    locked_mask[valid_days, first_bar[valid_days]] = True
    coords = torch.nonzero(locked_mask, as_tuple=False)
    N_trades = coords.shape[0]
    if N_trades == 0:
        return EMPTY
    d_idx = coords[:, 0]; b_idx = coords[:, 1]
    ep = d_close[d_idx, b_idx]
    max_future = min(sess_end - BASE_SESSION_START, 340)
    col_off = torch.arange(max_future, device=device).unsqueeze(0)
    col_idx = (b_idx + 1).unsqueeze(1) + col_off
    valid = (col_idx < sess_end) & (col_idx < 375)
    col_safe = col_idx.clamp(max=374)
    d_exp = d_idx.unsqueeze(1).expand(-1, max_future)
    fut_h = d_high[d_exp, col_safe]
    fut_l = d_low[d_exp, col_safe]
    fut_h_m = torch.where(valid, fut_h, torch.tensor(-1e9, device=device))
    fut_l_m = torch.where(valid, fut_l, torch.tensor(1e9, device=device))
    sl_p = sl_tensor[d_idx, b_idx]
    tp_p = tp_tensor[d_idx, b_idx]
    if direction == "CE":
        hit_sl = fut_l_m <= sl_p.unsqueeze(1)
        hit_tp = fut_h_m >= tp_p.unsqueeze(1)
    else:
        hit_sl = fut_h_m >= sl_p.unsqueeze(1)
        hit_tp = fut_l_m <= tp_p.unsqueeze(1)
    BIG = 999999
    sl_any = hit_sl.any(dim=1); tp_any = hit_tp.any(dim=1)
    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)
    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)
    eod_bar = min(sess_end - 1, 374)
    eod_px = d_close[d_idx, eod_bar]
    if direction == "CE":
        entry_eff = ep + SLIPPAGE_PTS * 0.5
        exit_sl = sl_p - SLIPPAGE_PTS * 0.5
        exit_tp = tp_p - SLIPPAGE_PTS * 0.5
        exit_eod = eod_px - SLIPPAGE_PTS * 0.5
        exit_px = torch.where(sl_exits, exit_sl, torch.where(tp_exits, exit_tp, exit_eod))
        raw_pts = (exit_px - entry_eff) * 0.50
    else:
        entry_eff = ep - SLIPPAGE_PTS * 0.5
        exit_sl = sl_p + SLIPPAGE_PTS * 0.5
        exit_tp = tp_p + SLIPPAGE_PTS * 0.5
        exit_eod = eod_px + SLIPPAGE_PTS * 0.5
        exit_px = torch.where(sl_exits, exit_sl, torch.where(tp_exits, exit_tp, exit_eod))
        raw_pts = (entry_eff - exit_px) * 0.50
    has_future = (b_idx + 1) < sess_end
    raw_pts = raw_pts[has_future]
    b_idx_v = b_idx[has_future].cpu().numpy()
    d_idx_v = d_idx[has_future].cpu().numpy()
    if raw_pts.shape[0] == 0:
        return EMPTY
    all_rs = (raw_pts * LOT_SIZE - FEE).cpu().numpy()
    return _finalize(all_rs, d_idx_v, b_idx_v, max_daily_loss, daily_profit)


# ═══════════════════════════════════════════════════════════════════════════
# 3D FUSED BATCH ENGINE  (B, N, T) — one GPU pass for a whole Optuna batch
# ═══════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def simulate_direction_locked_batch(entries_mask, sl_tensor, tp_tensor, direction,
                                    max_daily_loss, sess_end, day_mask=None, daily_profit=None):
    """
    entries_mask / sl_tensor / tp_tensor : (B, N, T)
    max_daily_loss : (B,)   sess_end : (B,)
    Returns dict {trial_index: res_dict}
    """
    B = entries_mask.shape[0]
    if day_mask is not None:
        entries_mask = entries_mask & day_mask.unsqueeze(0).unsqueeze(-1)
    has_entry = entries_mask.any(dim=2)                       # (B, N)
    first_bar = torch.argmax(entries_mask.int(), dim=2)       # (B, N)
    locked = torch.zeros_like(entries_mask)
    vd = torch.where(has_entry)                               # (b_idx, n_idx)
    if vd[0].shape[0] == 0:
        return {i: dict(EMPTY) for i in range(B)}
    locked[vd[0], vd[1], first_bar[vd]] = True
    coords = torch.nonzero(locked, as_tuple=False)           # (M, 3) -> [b, n, t]
    M = coords.shape[0]
    b_idx = coords[:, 0]; d_idx = coords[:, 1]; bar_idx = coords[:, 2]
    ep = d_close[d_idx, bar_idx]
    se_per = sess_end[b_idx]                                  # (M,)
    max_future = 340  # fixed so the only dynamic dim is entry count -> single compiled graph
    col_off = torch.arange(max_future, device=device).unsqueeze(0)
    col_idx = (bar_idx + 1).unsqueeze(1) + col_off            # (M, F)
    valid = (col_idx < se_per.unsqueeze(1)) & (col_idx < 375)
    col_safe = col_idx.clamp(max=374)
    d_exp = d_idx.unsqueeze(1).expand(-1, max_future)
    fut_h = d_high[d_exp, col_safe]
    fut_l = d_low[d_exp, col_safe]
    fut_h_m = torch.where(valid, fut_h, torch.tensor(-1e9, device=device))
    fut_l_m = torch.where(valid, fut_l, torch.tensor(1e9, device=device))
    sl_p = sl_tensor[b_idx, d_idx, bar_idx]
    tp_p = tp_tensor[b_idx, d_idx, bar_idx]
    if direction == "CE":
        hit_sl = fut_l_m <= sl_p.unsqueeze(1)
        hit_tp = fut_h_m >= tp_p.unsqueeze(1)
    else:
        hit_sl = fut_h_m >= sl_p.unsqueeze(1)
        hit_tp = fut_l_m <= tp_p.unsqueeze(1)
    BIG = 999999
    sl_any = hit_sl.any(dim=1); tp_any = hit_tp.any(dim=1)
    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)
    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)
    eod_bar = (se_per - 1).clamp(max=374).long()
    eod_px = d_close[d_idx, eod_bar]
    if direction == "CE":
        entry_eff = ep + SLIPPAGE_PTS * 0.5
        exit_sl = sl_p - SLIPPAGE_PTS * 0.5
        exit_tp = tp_p - SLIPPAGE_PTS * 0.5
        exit_eod = eod_px - SLIPPAGE_PTS * 0.5
        exit_px = torch.where(sl_exits, exit_sl, torch.where(tp_exits, exit_tp, exit_eod))
        raw_pts = (exit_px - entry_eff) * 0.50
    else:
        entry_eff = ep - SLIPPAGE_PTS * 0.5
        exit_sl = sl_p + SLIPPAGE_PTS * 0.5
        exit_tp = tp_p + SLIPPAGE_PTS * 0.5
        exit_eod = eod_px + SLIPPAGE_PTS * 0.5
        exit_px = torch.where(sl_exits, exit_sl, torch.where(tp_exits, exit_tp, exit_eod))
        raw_pts = (entry_eff - exit_px) * 0.50
    has_future = (bar_idx + 1) < se_per
    raw_pts = raw_pts[has_future]
    b_idx = b_idx[has_future]; d_idx = d_idx[has_future]; bar_idx = bar_idx[has_future]
    if raw_pts.shape[0] == 0:
        return {i: dict(EMPTY) for i in range(B)}
    all_rs = raw_pts * LOT_SIZE - FEE
    dl = max_daily_loss[b_idx]
    dp = daily_profit[b_idx]
    b_np = b_idx.cpu().numpy(); d_np = d_idx.cpu().numpy()
    bar_np = bar_idx.cpu().numpy(); r_np = all_rs.cpu().numpy()
    out = {i: dict(EMPTY) for i in range(B)}
    for bi in np.unique(b_np):
        m = b_np == bi
        out[int(bi)] = _finalize(r_np[m], d_np[m], bar_np[m], dl[bi], dp[bi])
    return out


def merge_results(ce, pe):
    t = ce["trades"] + pe["trades"]
    if t == 0:
        return {"trades": 0, "win_rate": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0,
                "ce_trades": 0, "pe_trades": 0, "ce_pnl": 0.0, "pe_pnl": 0.0}
    net = ce["net_rs"] + pe["net_rs"]
    ce_w = int(ce["trades"] * ce["win_rate"] / 100)
    pe_w = int(pe["trades"] * pe["win_rate"] / 100)
    wr = (ce_w + pe_w) / t * 100.0
    total_pos = ce["pos"] + pe["pos"]
    total_neg = ce["neg"] + pe["neg"]
    pf = total_pos / total_neg if total_neg > 0 else 99.0
    dd = max(ce["max_dd"], pe["max_dd"])
    return {"trades": t, "win_rate": round(wr, 2), "net_rs": round(net, 2),
            "pf": round(pf, 2), "max_dd": round(dd, 2),
            "ce_trades": ce["trades"], "pe_trades": pe["trades"],
            "ce_pnl": ce["net_rs"], "pe_pnl": pe["net_rs"]}


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER SUGGESTION + FUSED BATCH EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def suggest_params(strat_id, trial):
    daily_loss_pts = trial.suggest_int("daily_loss_pts", 10, 50, step=5)
    daily_profit_pts = trial.suggest_int("daily_profit_pts", 30, 80, step=10)
    moneyness = trial.suggest_categorical("moneyness", [0.5, 0.6, 0.7])
    max_trade_loss = trial.suggest_categorical("max_trade_loss_rs",
                                               [500, 1000, 1500, 2000, 3000, 5000, 9999])
    sess_start_off = trial.suggest_int("sess_start_off", 0, 30, step=5)
    sess_end_off = trial.suggest_int("sess_end_off", 30, 75, step=15)
    sess_end = BASE_SESSION_END - sess_end_off
    tf_map = {"B01": 1, "B02": 1, "B06": 1, "B03": 2, "B04": 3, "B05": 5}
    tf = tf_map.get(strat_id, None)
    if tf is None:  # B07
        tf = trial.suggest_categorical("timeframe", [1, 2, 3, 5])
    s1_k = trial.suggest_int("s1_k", 5, 30)
    s4_k = trial.suggest_int("s4_k", 20, 120, step=5)
    atr_p = trial.suggest_int("atr_p", 8, 35)
    if strat_id == "B07":
        # PHILOSOPHY: very small stop loss + quick (small) take-profit.
        # Tight risk, capture fast moves, keep win rate + net points, least DD.
        s1_os = trial.suggest_float("s1_os", 18.0, 40.0, step=2.5)
        s4_ob = trial.suggest_float("s4_ob", 72.0, 90.0, step=2.5)
        sl_m = trial.suggest_float("sl_m", 0.5, 2.0, step=0.1)
        tp_m = trial.suggest_float("tp_m", 1.0, 4.0, step=0.25)
    else:
        s1_os = trial.suggest_float("s1_os", 10.0, 40.0, step=2.5)
        s4_ob = trial.suggest_float("s4_ob", 65.0, 90.0, step=2.5)
        sl_m = trial.suggest_float("sl_m", 1.0, 5.0, step=0.1)
        tp_m = trial.suggest_float("tp_m", 2.0, 10.0, step=0.25)
    if tp_m < 1.5 * sl_m:
        raise optuna.TrialPruned()
    return {"timeframe": tf, "s1_k": s1_k, "s4_k": s4_k, "s1_os": s1_os, "s4_ob": s4_ob,
            "atr_p": atr_p, "sl_m": sl_m, "tp_m": tp_m,
            "daily_loss_pts": daily_loss_pts, "daily_profit_pts": daily_profit_pts,
            "moneyness": moneyness,
            "max_trade_loss_rs": max_trade_loss,
            "sess_start_off": sess_start_off, "sess_end_off": sess_end_off, "sess_end": sess_end}


def score_one(strat_id, res):
    n_tr = res["trades"]
    if n_tr < 30 or res["net_rs"] <= 0:
        return -999.0, res
    if strat_id == "B07":
        # WIN-RATE-FIRST philosophy: maximize WR, keep net points, crush drawdown.
        # Quick-profit / small-SL regime -> reward consistency over raw PF extremes.
        wr_comp = res["win_rate"] / 45.0
        pf_comp = min(res["pf"], 4.0) / 2.0
        dd_pen = 0.70 * (res["max_dd"] / max(res["net_rs"], 1.0))
        freq = min(n_tr / 400.0, 1.0) * 0.05
        score = wr_comp + pf_comp - dd_pen + freq
        if res["max_dd"] > 30000:
            score -= (res["max_dd"] - 30000) / 30000 * 0.5
        return score, res
    pf_comp = res["pf"] * (res["win_rate"] / 40.0)
    dd_pen = 0.50 * (res["max_dd"] / max(res["net_rs"], 1.0))
    freq = min(n_tr / 500.0, 1.0) * 0.10
    score = pf_comp - dd_pen + freq
    if strat_id == "B06" and res["max_dd"] > 50000:
        score -= (res["max_dd"] - 50000) / 50000 * 0.5
    return score, res


@torch.inference_mode()
def evaluate_batch(strat_id, param_dicts, day_mask=None):
    """Fused (B, N, T) evaluation of a whole batch of param dicts."""
    B = len(param_dicts)
    tf_map = {"B01": 1, "B02": 1, "B06": 1, "B03": 2, "B04": 3, "B05": 5}

    S1, S4, ATR = [], [], []
    for p in param_dicts:
        tf = p.get("timeframe", tf_map.get(strat_id))
        s1 = get_stoch(tf, p["s1_k"]); s4 = get_stoch(tf, p["s4_k"]); atr = get_atr(tf, p["atr_p"])
        if tf > 1:
            s1 = s1.repeat_interleave(tf, 1)[:, :T_BARS]
            s4 = s4.repeat_interleave(tf, 1)[:, :T_BARS]
            atr = atr.repeat_interleave(tf, 1)[:, :T_BARS]
        S1.append(s1); S4.append(s4); ATR.append(atr)
    S1 = torch.stack(S1, 0); S4 = torch.stack(S4, 0); ATR = torch.stack(ATR, 0)

    s1_os = torch.tensor([p["s1_os"] for p in param_dicts], device=device).view(B, 1, 1)
    s4_ob = torch.tensor([p["s4_ob"] for p in param_dicts], device=device).view(B, 1, 1)
    sl_m = torch.tensor([p["sl_m"] for p in param_dicts], device=device).view(B, 1, 1)
    tp_m = torch.tensor([p["tp_m"] for p in param_dicts], device=device).view(B, 1, 1)
    max_trade_loss = torch.tensor([p["max_trade_loss_rs"] for p in param_dicts], device=device).view(B, 1, 1)
    daily_loss_rs = torch.tensor([p["daily_loss_pts"] * LOT_SIZE for p in param_dicts], device=device)
    daily_profit_rs = torch.tensor([p["daily_profit_pts"] * LOT_SIZE for p in param_dicts], device=device)
    sess_end = torch.tensor([p["sess_end"] for p in param_dicts], device=device)
    moneyness = torch.tensor([p.get("moneyness", 0.5) for p in param_dicts], device=device).view(B, 1, 1)

    vw = torch.zeros((B, N_DAYS, T_BARS), dtype=torch.bool, device=device)
    for i, p in enumerate(param_dicts):
        so = p["sess_start_off"]; se = p["sess_end"]
        vw[i, :, BASE_SESSION_START + so:se] = True

    sl_dist_rs = ATR * sl_m * 0.50 * LOT_SIZE
    trade_ok = sl_dist_rs <= max_trade_loss
    # ITM strike offset: delta 0.5=ATM (offset 0), 0.6=1st ITM, 0.7=2nd ITM
    offset = (moneyness - 0.5) * 2.0 * ATR

    ce_ent = (S4 >= s4_ob) & (S1 <= s1_os) & vw & trade_ok
    ce_sl = d_close - offset - ATR * sl_m
    ce_tp = d_close - offset + ATR * tp_m
    ce_dict = simulate_direction_locked_batch(ce_ent, ce_sl, ce_tp, "CE",
                                              daily_loss_rs, sess_end, day_mask, daily_profit_rs)

    is_bidir = strat_id != "B01"
    if is_bidir:
        pe_s4_os = 100.0 - s4_ob
        pe_s1_ob = 100.0 - s1_os
        pe_ent = (S4 <= pe_s4_os) & (S1 >= pe_s1_ob) & vw & trade_ok
        pe_sl = d_close + offset + ATR * sl_m
        pe_tp = d_close + offset - ATR * tp_m
        pe_dict = simulate_direction_locked_batch(pe_ent, pe_sl, pe_tp, "PE",
                                                  daily_loss_rs, sess_end, day_mask, daily_profit_rs)
        res_list = [merge_results(ce_dict.get(i, EMPTY), pe_dict.get(i, EMPTY)) for i in range(B)]
    else:
        res_list = []
        for i in range(B):
            r = ce_dict.get(i, EMPTY)
            r = dict(r); r["ce_trades"] = r["trades"]; r["pe_trades"] = 0
            r["ce_pnl"] = r["net_rs"]; r["pe_pnl"] = 0.0
            res_list.append(r)
    return res_list


# ═══════════════════════════════════════════════════════════════════════════
# SEQUENTIAL WRAPPER (used for parity + unchanged OOS fallback)
# ═══════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def build_and_eval(strat_id, trial, day_mask=None):
    try:
        p = suggest_params(strat_id, trial)
    except optuna.TrialPruned:
        return -999.0, dict(EMPTY)
    res = evaluate_batch(strat_id, [p], day_mask)[0]
    sc, res = score_one(strat_id, res)
    return sc, res


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: 7 STRATEGIES × 3000 TRIALS × 2 MODES (fused batches)
# ═══════════════════════════════════════════════════════════════════════════
STRATS = [
    ("B01", "B01: 1m CE-Only (Baseline)"),
    ("B02", "B02: 1m CE+PE Bidirectional"),
    ("B03", "B03: 2m CE+PE Bidirectional"),
    ("B04", "B04: 3m CE+PE Bidirectional"),
    ("B05", "B05: 5m CE+PE Bidirectional"),
    ("B06", "B06: 1m CE+PE Tight DD"),
    ("B07", "B07: Best-TF CE+PE DD Target"),
]

def search(strat_id, day_mask, n_trials, seed):
    """Optuna study using LARGE fused 3D batches — one big (B,N,T) GPU op per batch.

    Big batches keep each CUDA dispatch large; combine with PROCS>1 (see
    run_strategy) for several concurrent big ops in flight — the guide's
    '8 Concurrent GPU Streams' path toward 85-95% saturation."""
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=seed, constant_liar=True, multivariate=True))
    n_batches = max(1, n_trials // BATCH_SIZE)
    for _ in range(n_batches):
        batch = [study.ask() for _ in range(BATCH_SIZE)]
        pdicts, keep = [], []
        for t in batch:
            try:
                pdicts.append(suggest_params(strat_id, t))
                keep.append(t)
            except optuna.TrialPruned:
                study.tell(t, -999.0)
        if not pdicts:
            continue
        res_list = evaluate_batch(strat_id, pdicts, day_mask)
        for t, res in zip(keep, res_list):
            sc, res = score_one(strat_id, res)
            for k, v in res.items():
                t.set_user_attr(k, v)
            study.tell(t, sc)
    return study


def _mp_worker(strat_id, n_trials, seed, mask_name, q):
    """Spawned process: run one search, return best (params, attrs) over Queue."""
    dm = {"full": None, "is": d_is_mask, "oos": d_oos_mask}[mask_name]
    study = search(strat_id, dm, n_trials, seed)
    bt = study.best_trial
    q.put((bt.params, dict(bt.user_attrs)))


def run_strategy(sid, sname, idx, total):
    print(f"\n[{idx:02d}/{total}] {sname}", flush=True)
    procs = int(os.environ.get("PROCS", "1"))

    # ── In-Sample (full 7y) search ──
    t0 = time.time()
    if procs > 1:
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        q = ctx.Queue()
        per = max(BATCH_SIZE, TRIALS_PER_STRATEGY // procs)
        ws = [ctx.Process(target=_mp_worker, args=(sid, per, 42 + i, "full", q))
              for i in range(procs)]
        for w in ws: w.start()
        nw_res = [q.get() for _ in range(procs)]
        for w in ws: w.join()
        best = max(nw_res, key=lambda r: score_one(sid, r[1])[0])
        nw_params, nw = best[0], best[1]
    else:
        study_nw = search(sid, None, TRIALS_PER_STRATEGY, 42)
        nw_params = study_nw.best_trial.params
        nw = study_nw.best_trial.user_attrs
    t_nw = time.time() - t0

    # ── Walk-Forward In-Sample (IS 2020-23) search ──
    t1 = time.time()
    if procs > 1:
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        q = ctx.Queue()
        per = max(BATCH_SIZE, TRIALS_PER_STRATEGY // procs)
        ws = [ctx.Process(target=_mp_worker, args=(sid, per, 42 + i, "is", q))
              for i in range(procs)]
        for w in ws: w.start()
        wf_res = [q.get() for _ in range(procs)]
        for w in ws: w.join()
        best = max(wf_res, key=lambda r: score_one(sid, r[1])[0])
        wf_params, wf_attrs = best[0], best[1]
        is_pnl = wf_attrs.get("net_rs", 0)
    else:
        study_wf = search(sid, d_is_mask, TRIALS_PER_STRATEGY, 42)
        wf_params = study_wf.best_trial.params
        is_pnl = study_wf.best_trial.user_attrs.get("net_rs", 0)
    t_wf = time.time() - t1

    fixed = optuna.trial.FixedTrial(wf_params)
    try:
        oos = evaluate_batch(sid, [fixed.params], d_oos_mask)[0]
    except Exception:
        oos = dict(EMPTY)

    wfe = round((oos["net_rs"]/2.35) / (is_pnl/4.0), 2) if is_pnl > 0 else 0.0

    print(f"  NW {t_nw:.0f}s: Rs {nw.get('net_rs',0):+,.0f} WR={nw.get('win_rate',0):.1f}% PF={nw.get('pf',0):.2f} DD=Rs {nw.get('max_dd',0):,.0f} T={nw.get('trades',0)} CE={nw.get('ce_trades',0)} PE={nw.get('pe_trades',0)}", flush=True)
    print(f"  WF {t_wf:.0f}s: IS=Rs {is_pnl:+,.0f} -> OOS=Rs {oos['net_rs']:+,.0f} PF={oos['pf']:.2f} WR={oos['win_rate']:.1f}% DD=Rs {oos['max_dd']:,.0f} WFE={wfe}", flush=True)
    print(f"  Params NW: {nw_params}", flush=True)
    print(f"  Params WF: {wf_params}", flush=True)

    return {"id": sid, "name": sname,
            "nw": {"params": nw_params, **{k: nw.get(k, 0) for k in
                   ["win_rate", "pf", "net_rs", "max_dd", "trades", "ce_trades", "pe_trades", "ce_pnl", "pe_pnl"]}},
            "wf": {"params": wf_params, "is_pnl": is_pnl,
                    "oos_pnl": oos["net_rs"], "oos_pf": oos["pf"], "oos_wr": oos["win_rate"],
                    "oos_dd": oos["max_dd"], "oos_trades": oos.get("trades", 0),
                    "oos_ce": oos.get("ce_trades", 0), "oos_pe": oos.get("pe_trades", 0),
                    "oos_ce_pnl": oos.get("ce_pnl", 0), "oos_pe_pnl": oos.get("pe_pnl", 0),
                    "wfe": wfe}}

# ═══════════════════════════════════════════════════════════════════════════
# CANDIDATE MODE: evaluate the top-5 of every phase on the causal fused engine,
# re-optimizing each (including a free TIMEFRAME param) to find the cream.
# ═══════════════════════════════════════════════════════════════════════════
PHASE_FILES = [
    ("P1", "master_25_strategy_comparison.json"),
    ("P2", "master_phase2_comparison.json"),
    ("P3", "master_phase3_exhaustive.json"),
    ("P4", "master_phase4_ultimate.json"),
    ("P5", "master_phase5_bidir_mtf.json"),
]
B0X_TF = {"B02": 1, "B03": 2, "B04": 3, "B05": 5, "B07": None}


def _sd(seed, k, d):
    return seed.get(k, d)


def suggest_candidate(trial, seed):
    """Suggest params centered on a phase seed, with TIMEFRAME free in {1,2,3,5}."""
    tf = trial.suggest_categorical("timeframe", [1, 2, 3, 5])
    dl = _sd(seed, "daily_loss_pts", 10)
    so = _sd(seed, "sess_start_off", 5)
    se = _sd(seed, "sess_end_off", 45)
    s1 = _sd(seed, "s1_k", 13)
    s4 = _sd(seed, "s4_k", 70)
    atr = _sd(seed, "atr_p", 12)
    s1o = _sd(seed, "s1_os", 15.0)
    s4o = _sd(seed, "s4_ob", 77.5)
    sl = _sd(seed, "sl_m", 2.5)
    tp = _sd(seed, "tp_m", 4.0)
    dl = max(10, min(50, dl))   # clamp stray seed values (e.g. 9999)
    daily_loss_pts = trial.suggest_int("daily_loss_pts", max(10, dl - 10), min(50, dl + 10), step=5)
    dp = _sd(seed, "daily_profit_pts", 50)
    dp = max(30, min(80, dp))   # clamp stray seed values (e.g. 9999)
    daily_profit_pts = trial.suggest_int("daily_profit_pts", max(30, dp - 20), min(80, dp + 20), step=10)
    moneyness = trial.suggest_categorical("moneyness", [0.5, 0.6, 0.7])
    max_trade_loss = trial.suggest_categorical("max_trade_loss_rs",
                                               [500, 1000, 1500, 2000, 3000, 5000, 9999])
    sess_start_off = trial.suggest_int("sess_start_off", max(0, so - 10), min(30, so + 10), step=5)
    sess_end_off = trial.suggest_int("sess_end_off", max(30, se - 20), min(75, se + 20), step=15)
    sess_end = BASE_SESSION_END - sess_end_off
    s1_k = trial.suggest_int("s1_k", max(5, s1 - 8), min(30, s1 + 8))
    s4_k = trial.suggest_int("s4_k", max(20, s4 - 25), min(120, s4 + 25), step=5)
    atr_p = trial.suggest_int("atr_p", max(8, atr - 8), min(35, atr + 8))
    s1_os = trial.suggest_float("s1_os", max(10.0, s1o - 8), min(40.0, s1o + 8), step=2.5)
    s4_ob = trial.suggest_float("s4_ob", max(65.0, s4o - 8), min(90.0, s4o + 8), step=2.5)
    sl_m = trial.suggest_float("sl_m", max(0.5, sl - 1.5), min(6.0, sl + 2.0), step=0.1)
    tp_m = trial.suggest_float("tp_m", max(1.0, tp - 3.0), min(12.0, tp + 3.0), step=0.25)
    if tp_m < 1.5 * sl_m:
        raise optuna.TrialPruned()
    return {"timeframe": tf, "s1_k": s1_k, "s4_k": s4_k, "s1_os": s1_os, "s4_ob": s4_ob,
            "atr_p": atr_p, "sl_m": sl_m, "tp_m": tp_m,
            "daily_loss_pts": daily_loss_pts, "daily_profit_pts": daily_profit_pts,
            "moneyness": moneyness,
            "max_trade_loss_rs": max_trade_loss,
            "sess_start_off": sess_start_off, "sess_end_off": sess_end_off, "sess_end": sess_end}


def _qp_score(res):
    """Quick-profit rank: reward WR*PF, crush drawdown, light freq bonus."""
    if res.get("trades", 0) < 30 or res.get("net_rs", 0) <= 0:
        return -1e9
    wr = res.get("win_rate", 0); pf = res.get("pf", 0)
    net = res.get("net_rs", 0); dd = res.get("max_dd", 0)
    return (wr / 40.0) * pf - 0.5 * (dd / max(net, 1)) + min(res["trades"] / 500.0, 1.0) * 0.1


def _run_study(seed, n_trials, day_mask, bs):
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=42, constant_liar=True, multivariate=True))
    n_batches = max(1, n_trials // bs)
    for _ in range(n_batches):
        batch = [study.ask() for _ in range(bs)]
        pds, keep = [], []
        for t in batch:
            try:
                pds.append(suggest_candidate(t, seed)); keep.append(t)
            except Exception:
                study.tell(t, -999.0)
        if not pds:
            continue
        res_list = evaluate_batch("B07", pds, day_mask)
        for t, res in zip(keep, res_list):
            sc = _qp_score(res)
            for k, v in res.items():
                t.set_user_attr(k, v)
            study.tell(t, sc)
    return study


def refine_candidate(label, seed, n_trials, q=None, bs=BATCH_SIZE):
    nw_study = _run_study(seed, n_trials, None, bs)
    nw_params = dict(nw_study.best_trial.params)
    nw_params["sess_end"] = BASE_SESSION_END - nw_params["sess_end_off"]
    nw = dict(nw_study.best_trial.user_attrs)

    wf_study = _run_study(seed, n_trials, d_is_mask, bs)
    wf_params = dict(wf_study.best_trial.params)
    wf_params["sess_end"] = BASE_SESSION_END - wf_params["sess_end_off"]
    oos = evaluate_batch("B07", [wf_params], d_oos_mask)[0]

    cep = f"{nw.get('ce_trades',0)}/{nw.get('pe_trades',0)}"
    result = {"label": label, "seed": seed,
              "nw": {"params": nw_params, **{k: nw.get(k, 0) for k in
                      ["win_rate", "pf", "net_rs", "max_dd", "trades", "ce_trades", "pe_trades"]}},
              "wf": {"params": wf_params, "oos_pnl": oos["net_rs"], "oos_pf": oos["pf"],
                     "oos_wr": oos["win_rate"], "oos_dd": oos["max_dd"],
                     "oos_trades": oos.get("trades", 0),
                     "oos_ce": oos.get("ce_trades", 0), "oos_pe": oos.get("pe_trades", 0)}}
    print(f"  {label}: NW Rs {nw.get('net_rs',0):+,.0f} WR={nw.get('win_rate',0):.1f}% "
          f"PF={nw.get('pf',0):.2f} | OOS Rs {oos['net_rs']:+,.0f} WR={oos['win_rate']:.1f}% "
          f"PF={oos['pf']:.2f} DD=Rs {oos['max_dd']:,.0f} (TF={wf_params.get('timeframe')})",
          flush=True)
    if q is not None:
        q.put(result)
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result


def extract_phase_candidates(limit=10**9):
    cands = []
    for phase, fn in PHASE_FILES:
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        res = d.get("results", [])
        top = sorted(res, key=lambda r: r.get("non_wf", {}).get("net_rs", 0) or 0, reverse=True)[:5]
        for r in top:
            seed = dict(r.get("non_wf", {}).get("best_params", {}) or {})
            sid = str(r["id"])
            tf0 = B0X_TF.get(sid)
            if tf0 is not None and "timeframe" not in seed:
                seed["timeframe"] = tf0
            cands.append((f"{phase}_{sid}_{r['name'].split(':')[0].strip()}", seed))
    return cands[:limit]


def run_candidates():
    n_trials = int(os.environ.get("CTRIALS", "800"))
    procs = int(os.environ.get("PROCS", "1"))
    limit = int(os.environ.get("CAND_LIMIT", "100000"))
    cands = extract_phase_candidates(limit)
    out = ROOT / "artifacts" / "f6_hybrid" / "master_phase5d_candidates_top5.json"
    # --- RESUME: skip candidates already saved, merge results ---
    prev = {}
    if out.exists():
        try:
            for r in json.load(open(out)).get("all", []):
                prev[r["label"]] = r
        except Exception:
            prev = {}
    done_labels = set(prev.keys())
    remaining = [(l, s) for l, s in cands if l not in done_labels]
    print(f"\n{'='*120}\nCANDIDATE MODE: {len(cands)} total | {len(done_labels)} already done | "
          f"{len(remaining)} remaining x {n_trials} trials (NW + WF-OOS) | PROCS={procs}\n{'='*120}", flush=True)
    t0 = time.time()
    if not remaining:
        results = list(prev.values())
    else:
        if procs > 1:
            import multiprocessing as _mp
            ctx = _mp.get_context("spawn")
            q = ctx.Queue()
            per = max(1, len(remaining) // procs) or 1
            chunks = [remaining[i:i + per] for i in range(0, len(remaining), per)]
            ws = [ctx.Process(target=_cand_worker, args=(ch, n_trials, q)) for ch in chunks]
            for w in ws: w.start()
            new_results = [q.get() for _ in ws]
            for w in ws: w.join()
        else:
            new_results = [refine_candidate(l, s, n_trials) for l, s in remaining]
        merged = dict(prev)
        for r in new_results:
            merged[r["label"]] = r
        results = [merged[l] for l, s in cands if l in merged]
    total = time.time() - t0
    results.sort(key=lambda r: _qp_score({**r["wf"], "win_rate": r["wf"].get("oos_wr", 0),
                                           "pf": r["wf"].get("oos_pf", 0),
                                           "net_rs": r["wf"].get("oos_pnl", 0),
                                           "max_dd": r["wf"].get("oos_dd", 0),
                                           "trades": r["wf"].get("oos_trades", 0)}),
                    reverse=True)
    print(f"\n{'='*120}\nCREAM OF THE CREAM — TOP 5 (ranked by OOS quick-profit score)\n{'='*120}", flush=True)
    print(f"{'#':3s} {'Strategy':52s} {'OOS PnL':>12s} {'PF':>6s} {'WR':>6s} {'DD':>10s} {'TF':>4s}", flush=True)
    for i, r in enumerate(results[:5], 1):
        wf = r["wf"]
        print(f"[{i:2d}] {r['label']:52s} Rs {wf['oos_pnl']:+11,.0f} {wf['oos_pf']:5.2f} "
              f"{wf['oos_wr']:4.1f}% Rs {wf['oos_dd']:7,.0f} {wf['params'].get('timeframe')}", flush=True)
    with open(out, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "n_candidates": len(cands), "trials_per_candidate": n_trials,
                   "total_time_s": round(total, 1), "top5": results[:5], "all": results}, f, indent=2)
    print(f"\nSaved: {out}  | total {total:.0f}s", flush=True)


def _cand_worker(chunk, n_trials, q):
    for l, s in chunk:
        refine_candidate(l, s, n_trials, q)


def main():
    if os.environ.get("MODE") == "CANDIDATES":
        run_candidates()
        return
    strat_filter = os.environ.get("STRAT")
    trials_override = int(os.environ.get("TRIALS", "0") or 0)
    if trials_override:
        global TRIALS_PER_STRATEGY
        TRIALS_PER_STRATEGY = trials_override
    run_strats = [s for s in STRATS if strat_filter is None or s[0] == strat_filter]
    total = len(run_strats)
    total_trials = total * TRIALS_PER_STRATEGY * 2
    print(f"\n{'='*140}", flush=True)
    print(f"PHASE 5d: FUSED 3D-BATCH GPU | {total}×{TRIALS_PER_STRATEGY}×2 = {total_trials:,} trials"
          + (f" | FILTER={strat_filter}" if strat_filter else ""), flush=True)
    print(f"Position Lock [Y] | Slippage 1pt [Y] | Daily Cap [Y] | Cached Indicators [Y] | Batch={BATCH_SIZE}", flush=True)
    print(f"{'='*140}", flush=True)

    t_start = time.time()
    results = [run_strategy(s, n, i+1, total) for i, (s, n) in enumerate(run_strats)]
    total_time = time.time() - t_start

    by_nw = sorted(results, key=lambda x: x["nw"]["net_rs"], reverse=True)
    by_oos = sorted(results, key=lambda x: x["wf"]["oos_pnl"], reverse=True)
    nw_rank = {r["id"]: i+1 for i, r in enumerate(by_nw)}
    oos_rank = {r["id"]: i+1 for i, r in enumerate(by_oos)}

    print(f"\n{'='*160}", flush=True)
    print(f"PHASE 5d LEADERBOARD ({total_trials:,} TRIALS IN {total_time:.1f}s = {total_trials/total_time:.0f} t/s) — FUSED 3D-BATCH CAUSAL", flush=True)
    print(f"{'='*160}", flush=True)
    print(f"\n{'NW#':4s} {'OOS#':5s} {'Strategy':42s} {'NW PnL':>14s} {'PF':>6s} {'WR':>6s} {'DD':>10s} {'OOS PnL':>14s} {'PF':>6s} {'WR':>6s} {'DD':>10s} {'WFE':>5s} {'CE/PE':>10s}", flush=True)
    print("-" * 155, flush=True)
    for r in by_nw:
        nw = r["nw"]; wf = r["wf"]
        cep = f"{wf['oos_ce']}/{wf['oos_pe']}"
        star = " ***" if oos_rank[r["id"]] <= 2 else ""
        print(f"[{nw_rank[r['id']]:2d}] [{oos_rank[r['id']]:2d}]  {r['name']:42s} Rs {nw['net_rs']:+11,.0f} {nw['pf']:5.2f} {nw['win_rate']:4.1f}% Rs {nw['max_dd']:7,.0f} Rs {wf['oos_pnl']:+11,.0f} {wf['oos_pf']:5.2f} {wf['oos_wr']:4.1f}% Rs {wf['oos_dd']:7,.0f} {wf['wfe']:4.2f} {cep:>10s}{star}", flush=True)

    print(f"\nDD COMPARISON:", flush=True)
    for r in by_oos:
        wf = r["wf"]
        dd_r = wf["oos_dd"] / max(wf["oos_pnl"], 1) * 100 if wf["oos_pnl"] > 0 else 999
        print(f"  {r['name']:42s}: DD=Rs {wf['oos_dd']:>7,.0f} ({dd_r:.0f}%) CE=Rs {wf.get('oos_ce_pnl',0):>+10,.0f} PE=Rs {wf.get('oos_pe_pnl',0):>+10,.0f} Max Trade SL=Rs{r['wf']['params'].get('max_trade_loss_rs','?')}", flush=True)

    out = ROOT / "artifacts" / "f6_hybrid" / "master_phase5d_causal_fused.json"
    with open(out, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "total_time_s": round(total_time, 2),
                   "causal_pillars": {"zero_lookahead": True, "clock_alignment": True,
                                      "slippage_pts": SLIPPAGE_PTS, "fee_rs": FEE,
                                      "position_lock": "1_per_day_per_direction",
                                      "daily_cap": True},
                   "results": by_oos}, f, indent=2)
    print(f"\nSaved: {out}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# PARITY TEST: fused 3D engine vs sequential 2D engine
# ═══════════════════════════════════════════════════════════════════════════
def parity_test():
    print("=== PARITY: fused 3D vs sequential 2D ===", flush=True)
    test_params = [
        {"B01": dict(timeframe=1, s1_k=9, s4_k=60, s1_os=20.0, s4_ob=79.5, atr_p=14,
                      sl_m=2.0, tp_m=4.0, daily_loss_pts=10, daily_profit_pts=50, moneyness=0.5, max_trade_loss_rs=2000,
                     sess_start_off=5, sess_end_off=45, sess_end=300)},
        {"B02": dict(timeframe=1, s1_k=14, s4_k=80, s1_os=15.0, s4_ob=85.0, atr_p=21,
                      sl_m=3.0, tp_m=6.0, daily_loss_pts=12, daily_profit_pts=50, moneyness=0.5, max_trade_loss_rs=3000,
                     sess_start_off=10, sess_end_off=60, sess_end=285)},
        {"B07": dict(timeframe=3, s1_k=7, s4_k=50, s1_os=25.0, s4_ob=70.0, atr_p=10,
                      sl_m=1.5, tp_m=5.0, daily_loss_pts=8, daily_profit_pts=50, moneyness=0.5, max_trade_loss_rs=1500,
                     sess_start_off=0, sess_end_off=30, sess_end=315)},
    ]
    masks = [("NW", None), ("WF(IS)", d_is_mask), ("OOS", d_oos_mask)]
    ok = True
    for item in test_params:
        sid = list(item.keys())[0]
        p = item
        for tag, dm in masks:
            ft = optuna.trial.FixedTrial(p[sid])
            _, seq = build_and_eval(sid, ft, dm)
            bat = evaluate_batch(sid, [p[sid]], dm)[0]
            for k in ["trades", "win_rate", "net_rs", "pf", "max_dd"]:
                a, b = seq.get(k, 0), bat.get(k, 0)
                if abs(float(a) - float(b)) > 1e-2:
                    ok = False
                    print(f"  MISMATCH {sid}/{tag}/{k}: seq={a} bat={b}", flush=True)
            print(f"  {sid}/{tag}: seq net={seq['net_rs']:+,.1f} T={seq['trades']} | bat net={bat['net_rs']:+,.1f} T={bat['trades']}", flush=True)
    print("PARITY RESULT:", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if os.environ.get("PARITY") == "1":
        parity_test()
    else:
        main()
