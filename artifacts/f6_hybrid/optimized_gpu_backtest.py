"""
OPTIMUS BACKTEST  (a.k.a. optimized_gpu_backtest — the fused 1D-vs-multi-TF engine)
=================================================================================
The hardened, regression-tested backtest engine for the F6 hybrid strategy.
Branded "Optimus Backtest" after the bug-hardening pass (batching daily-cap fix,
mixed-precision fp16 safety, ensemble SoC refactor). See test_optimus_regression.py
for the speed + accuracy regression guard.

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
import pandas as pd
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
    The trade that BREACHES the cap is counted (it happened in live trading),
    then the day halts. Earlier trades in the day are kept as-is.
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
            # HONEST accounting: the trade that BREACHES the cap still happened
            # (real loss/win in live trading). Count it, then halt the day.
            stopped = True
            cum = new_cum
            kept.append(r)
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
    global d_open
    d_open = torch.tensor(arr_o, dtype=torch.float32, device=device)
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
# B08 — OPENING-RANGE BREAKOUT (translation of "NIFTY BANK INTRADAY OPTION
# BUYING SINGLE SUCCESSFUL STRATEGY" by Akshay VG, applied to Nifty 50).
#
# Reuses the EXACT 3D fused sim + _finalize core (no arithmetic changes) — only
# the entry-mask / SL / TP generation differs. GPU does all the compute.
#
# Per-combo work is kept on-GPU; the only CPU loop is over the param batch B,
# and the 11 CPR levels are handled as ONE (N,11) tensor with masked min/max
# (NOT a Python loop over levels) to avoid GPU-starvation kernel launches.
# ═══════════════════════════════════════════════════════════════════════════
# Daily OHLC aggregates (N,) + previous-trading-day classic floor pivots.
_DH = d_high.amax(dim=1); _DL = d_low.amin(dim=1); _DC = d_close[:, -1]
_pH = torch.roll(_DH, 1); _pL = torch.roll(_DL, 1); _pC = torch.roll(_DC, 1)
_PP = (_pH + _pL + _pC) / 3.0
_R1 = 2 * _PP - _pL; _S1 = 2 * _PP - _pH
_HL = _pH - _pL
_R2 = _PP + _HL; _S2 = _PP - _HL
_R3 = _pH + 2 * (_PP - _pL); _S3 = _pL - 2 * (_pH - _PP)
_R4 = _PP + 3 * _HL; _S4 = _PP - 3 * _HL
# (N, 11): PP, R1..R4, S1..S4, PDH, PDL
CPR_LEVELS = torch.stack([_PP, _R1, _R2, _R3, _R4, _S1, _S2, _S3, _S4, _pH, _pL], dim=1)

# Precomputed opening ranges for the 4 (open_candles, range_mode) variants.
RANGE_CACHE = {}
for _K in (2, 3):
    _rk = 5 * _K
    for _rm in ("body", "wick"):
        if _rm == "body":
            _bh = torch.maximum(d_open, d_close); _bl = torch.minimum(d_open, d_close)
        else:
            _bh = d_high; _bl = d_low
        RANGE_CACHE[(_K, _rm)] = (_bh[:, :_rk].amax(1), _bl[:, :_rk].amin(1))

# Friday mask (book: "Do not trade on Fridays").
_dates = pd.to_datetime(all_days)
d_friday = torch.tensor(_dates.dayofweek.to_numpy() == 4, dtype=torch.bool, device=device)


def _bo_sl_tp(rng_hi, rng_lo, ep, is_bull, p, span):
    """Vectorized SL/TP from CPR levels. All (N,) tensors."""
    sl_buf = float(p["sl_buf"]); ride = float(p["ride_frac"]); pct = float(p["pct_target"])
    opp = (rng_lo - sl_buf) if is_bull else (rng_hi + sl_buf)
    L = CPR_LEVELS                                     # (N, 11)
    ep_e = ep.unsqueeze(1); rl = rng_lo.unsqueeze(1); rh = rng_hi.unsqueeze(1)
    sp = span.unsqueeze(1)
    if p["sl_mode"] == "opposite":
        sl = opp
    else:
        if p["sl_mode"] == "cpr_inside":
            mask = (L >= rl) & (L <= rh)
        else:  # cpr_respect: nearest level on the SL side, within 1.5x range span
            side = (L < ep_e) if is_bull else (L > ep_e)
            mask = ((L - ep_e).abs() <= 1.5 * sp) & side
        fill = float("-inf") if is_bull else float("inf")
        cand = torch.where(mask, L, torch.full_like(L, fill))
        agg = cand.max(dim=1).values if is_bull else cand.min(dim=1).values
        sl = torch.where(torch.isinf(agg), opp, agg)
    if p["target_mode"] == "level_ride":
        if is_bull:
            above = torch.where(L > ep_e, L, torch.full_like(L, float("inf")))
            nxt = above.min(dim=1).values
            nxt = torch.where(torch.isinf(nxt), ep + ride * span, nxt)
            tp = ep + ride * (nxt - ep)
        else:
            below = torch.where(L < ep_e, L, torch.full_like(L, float("-inf")))
            nxt = below.max(dim=1).values
            nxt = torch.where(torch.isinf(nxt), ep - ride * span, nxt)
            tp = ep - ride * (ep - nxt)
    else:
        tp = ep * (1 + pct) if is_bull else ep * (1 - pct)
    return sl, tp


def _breakout_one(p):
    """Build (N,T) entry/SL/TP tensors for one param combo (GPU)."""
    K = int(p["open_candles"]); rk = 5 * K
    rng_hi, rng_lo = RANGE_CACHE[(K, p["range_mode"])]
    span = (rng_hi - rng_lo).clamp(min=1.0)
    buf = float(p["break_buf"])
    eu_bar = 5 * int(p["entry_until"])
    t = torch.arange(T_BARS, device=device)
    window = (t >= rk) & (t <= eu_bar)
    if p["break_mode"] == "close":
        bull = d_close > (rng_hi[:, None] + buf)
        bear = d_close < (rng_lo[:, None] - buf)
    else:
        bull = d_high > (rng_hi[:, None] + buf)
        bear = d_low < (rng_lo[:, None] - buf)
    ce_ent = bull & window
    pe_ent = bear & window
    if p["direction"] == "bull":
        pe_ent = torch.zeros_like(pe_ent)
    elif p["direction"] == "bear":
        ce_ent = torch.zeros_like(ce_ent)
    if not p["allow_friday"]:
        nf = (~d_friday).unsqueeze(1)
        ce_ent = ce_ent & nf
        pe_ent = pe_ent & nf
    fb_ce = torch.argmax(ce_ent.int(), dim=1)
    fb_pe = torch.argmax(pe_ent.int(), dim=1)
    ar = torch.arange(N_DAYS, device=device)
    ep_ce = d_close[ar, fb_ce]; ep_pe = d_close[ar, fb_pe]
    sl_ce, tp_ce = _bo_sl_tp(rng_hi, rng_lo, ep_ce, True, p, span)
    sl_pe, tp_pe = _bo_sl_tp(rng_hi, rng_lo, ep_pe, False, p, span)
    ce_sl = sl_ce[:, None].expand(-1, T_BARS); ce_tp = tp_ce[:, None].expand(-1, T_BARS)
    pe_sl = sl_pe[:, None].expand(-1, T_BARS); pe_tp = tp_pe[:, None].expand(-1, T_BARS)
    return ce_ent, ce_sl, ce_tp, pe_ent, pe_sl, pe_tp


@torch.inference_mode()
def evaluate_breakout_batch(param_dicts, day_mask=None):
    """Single-mask fused (B,N,T) eval — used by search()/Optuna/parity."""
    B = len(param_dicts)
    ce_e, ce_s, ce_t, pe_e, pe_s, pe_t = [], [], [], [], [], []
    for p in param_dicts:
        a, b, c, d, e, f = _breakout_one(p)
        ce_e.append(a); ce_s.append(b); ce_t.append(c)
        pe_e.append(d); pe_s.append(e); pe_t.append(f)
    ce_ent = torch.stack(ce_e, 0); ce_sl = torch.stack(ce_s, 0); ce_tp = torch.stack(ce_t, 0)
    pe_ent = torch.stack(pe_e, 0); pe_sl = torch.stack(pe_s, 0); pe_tp = torch.stack(pe_t, 0)
    dl = T([p.get("daily_loss_pts", 50) * LOT_SIZE for p in param_dicts])
    dp = T([p.get("daily_profit_pts", 120) * LOT_SIZE for p in param_dicts])
    sess_end = T([BASE_SESSION_END for _ in param_dicts])
    ce_dict = simulate_direction_locked_batch(ce_ent, ce_sl, ce_tp, "CE", dl, sess_end, day_mask, dp)
    pe_dict = simulate_direction_locked_batch(pe_ent, pe_sl, pe_tp, "PE", dl, sess_end, day_mask, dp)
    res_list = [merge_results(ce_dict.get(i, EMPTY), pe_dict.get(i, EMPTY)) for i in range(B)]
    for i, p in enumerate(param_dicts):
        cap = 1.0 - 0.04 * int(p["otm_strikes"])  # slightly-OTM premium capture
        res_list[i]["net_rs"] = res_list[i]["net_rs"] * cap
        res_list[i]["max_dd"] = res_list[i]["max_dd"] * cap
    return res_list


@torch.inference_mode()
def evaluate_breakout_all(param_dicts):
    """Build masks ONCE, simulate under full/IS/OOS masks (avoids 3x rebuild)."""
    B = len(param_dicts)
    ce_e, ce_s, ce_t, pe_e, pe_s, pe_t = [], [], [], [], [], []
    for p in param_dicts:
        a, b, c, d, e, f = _breakout_one(p)
        ce_e.append(a); ce_s.append(b); ce_t.append(c)
        pe_e.append(d); pe_s.append(e); pe_t.append(f)
    ce_ent = torch.stack(ce_e, 0); ce_sl = torch.stack(ce_s, 0); ce_tp = torch.stack(ce_t, 0)
    pe_ent = torch.stack(pe_e, 0); pe_sl = torch.stack(pe_s, 0); pe_tp = torch.stack(pe_t, 0)
    dl = T([p.get("daily_loss_pts", 50) * LOT_SIZE for p in param_dicts])
    dp = T([p.get("daily_profit_pts", 120) * LOT_SIZE for p in param_dicts])
    sess_end = T([BASE_SESSION_END for _ in param_dicts])
    out = {}
    for name, msk in (("full", None), ("is", d_is_mask), ("oos", d_oos_mask)):
        ce_dict = simulate_direction_locked_batch(ce_ent, ce_sl, ce_tp, "CE", dl, sess_end, msk, dp)
        pe_dict = simulate_direction_locked_batch(pe_ent, pe_sl, pe_tp, "PE", dl, sess_end, msk, dp)
        res_list = [merge_results(ce_dict.get(i, EMPTY), pe_dict.get(i, EMPTY)) for i in range(B)]
        for i, p in enumerate(param_dicts):
            cap = 1.0 - 0.04 * int(p["otm_strikes"])
            res_list[i]["net_rs"] = res_list[i]["net_rs"] * cap
            res_list[i]["max_dd"] = res_list[i]["max_dd"] * cap
        out[name] = res_list
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MIXED-PRECISION PATH  (env HALF=1) — "master fp32 / compute fp16" recipe.
#
# The pips-based exit math (exit_px − entry_eff ≈ 20 pts at price ≈ 24,000)
# is catastrophic-cancellation-prone in float16, so ALL price/ATR data and the
# exit P&L stay float32. Only the bulky STOCH_CACHE (used solely for threshold
# comparisons, no subtraction) is cast to float16 — it upcasts losslessly to
# fp32 for the compare, giving ~0.4% entry-mask noise but EXACT exits. This
# halves the indicator-cache footprint without the 50%-trade divergence a full
# fp16 recast produced. Control tensors (T) stay float32 so exit SL/TP is exact.
# ═══════════════════════════════════════════════════════════════════════════
DTYPE = torch.float32
HALF_MODE = False


def T(x):
    """Build a control/scalar tensor in float32 (exit-critical: keeps SL/TP exact)."""
    if isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=DTYPE, device=device)
    return torch.tensor(x, dtype=DTYPE, device=device)


def enable_half():
    global HALF_MODE, STOCH_CACHE
    HALF_MODE = True
    # Only the stochastic cache is downcast: it feeds threshold compares only,
    # and fp16→fp32 upcast is lossless, so entry masks stay near-exact.
    STOCH_CACHE = {k: v.half() for k, v in STOCH_CACHE.items()}
    print("  [HALF] STOCH_CACHE recast to float16 (prices/ATR/exits stay fp32)",
          flush=True)


if os.environ.get("HALF") == "1":
    enable_half()


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
    fut_h_m = torch.where(valid, fut_h, T(-1e9))
    fut_l_m = torch.where(valid, fut_l, T(1e9))
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
                                     max_daily_loss, sess_end, day_mask=None, daily_profit=None,
                                     entry_px=None):
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
    ep = entry_px[b_idx, d_idx] if entry_px is not None else d_close[d_idx, bar_idx]
    se_per = sess_end[b_idx]                                  # (M,)
    max_future = 340  # fixed so the only dynamic dim is entry count -> single compiled graph
    col_off = torch.arange(max_future, device=device).unsqueeze(0)
    col_idx = (bar_idx + 1).unsqueeze(1) + col_off            # (M, F)
    valid = (col_idx < se_per.unsqueeze(1)) & (col_idx < 375)
    col_safe = col_idx.clamp(max=374)
    d_exp = d_idx.unsqueeze(1).expand(-1, max_future)
    fut_h = d_high[d_exp, col_safe]
    fut_l = d_low[d_exp, col_safe]
    fut_h_m = torch.where(valid, fut_h, T(-1e9))
    fut_l_m = torch.where(valid, fut_l, T(1e9))
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
    dl_param = max_daily_loss          # (B,) per-param daily loss cap
    dp_param = daily_profit            # (B,) per-param daily profit cap
    b_np = b_idx.cpu().numpy(); d_np = d_idx.cpu().numpy()
    bar_np = bar_idx.cpu().numpy(); r_np = all_rs.cpu().numpy()
    out = {i: dict(EMPTY) for i in range(B)}
    for bi in np.unique(b_np):
        m = b_np == bi
        res = _finalize(r_np[m], d_np[m], bar_np[m], dl_param[bi], dp_param[bi])
        out[int(bi)] = res
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


def suggest_breakout_params(trial):
    """Optuna suggestion space for the B08 opening-range breakout (book translation)."""
    p = {
        "open_candles": trial.suggest_int("open_candles", 2, 3),
        "range_mode": trial.suggest_categorical("range_mode", ["body", "wick"]),
        "break_mode": trial.suggest_categorical("break_mode", ["close", "high_low"]),
        "break_buf": trial.suggest_int("break_buf", 0, 5, step=1),
        "entry_until": trial.suggest_int("entry_until", 5, 12, step=1),
        "otm_strikes": trial.suggest_int("otm_strikes", 2, 4),
        "sl_mode": trial.suggest_categorical("sl_mode", ["opposite", "cpr_inside", "cpr_respect"]),
        "sl_buf": trial.suggest_int("sl_buf", 0, 5, step=1),
        "target_mode": trial.suggest_categorical("target_mode", ["level_ride", "pct"]),
        "ride_frac": trial.suggest_float("ride_frac", 0.5, 0.75, step=0.05),
        "pct_target": trial.suggest_float("pct_target", 0.04, 0.10, step=0.02),
        "direction": trial.suggest_categorical("direction", ["both", "bull", "bear"]),
        "allow_friday": trial.suggest_categorical("allow_friday", [False, True]),
        "daily_loss_pts": 50, "daily_profit_pts": 120,
    }
    return p


def suggest_marni_core_params(trial):
    """Optuna suggestion space for the B09 Marni Core 15m-HA UT Bot signal."""
    p = {
        "ut_key": trial.suggest_categorical("ut_key", [0.8, 1.0, 1.2, 1.5]),
        "atr_period": trial.suggest_categorical("atr_period", [10, 14, 20]),
        "sl_m": trial.suggest_categorical("sl_m", [1.0, 1.5, 2.0, 2.5]),
        "tp_m": trial.suggest_categorical("tp_m", [2.0, 3.0, 4.0, 5.0]),
        "direction": trial.suggest_categorical("direction", ["both", "bull", "bear"]),
        "allow_friday": trial.suggest_categorical("allow_friday", [False, True]),
        "daily_loss_pts": trial.suggest_categorical("daily_loss_pts", [20, 30, 50]),
        "daily_profit_pts": trial.suggest_categorical("daily_profit_pts", [30, 50, 80]),
    }
    return p


def score_one(strat_id, res):
    n_tr = res["trades"]
    if n_tr < 30 or res["net_rs"] <= 0:
        return -999.0, res
    if strat_id in ("B07", "B08", "B09"):
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


# ═══════════════════════════════════════════════════════════════════════════
# B09 — MARNI CORE ENGINE (15-min Heikin-Ashi + UT Bot color signal)
#
# Faithful GPU port of the Marni Core "Marni Core Engine" entry signal that is
# gated by the 15-minute Heikin-Ashi UT Bot color (`ut_col` in the reference
# marni_elder_impulse_7y.py StrictHTFBiasState, period=15):
#     CE entry when the 15m-HA UT Bot color flips GREEN
#     PE entry when the 15m-HA UT Bot color flips RED
# Reuses the EXACT 3D fused sim + _finalize core. The only new compute is the
# 15m HA candle aggregation + UT Bot trailing-stop scan (25 bars, vectorized
# across N days), which is cheap and stays fully on-GPU. ATR-based SL/TP
# (option-premium model: dist = m * ATR * 0.5) matches the engine's money model.
# ═══════════════════════════════════════════════════════════════════════════
# 15-min aggregation of the 1-min spot (375 = 25 * 15 bars).
HA_NB = 25
_o15 = d_open.reshape(N_DAYS, HA_NB, 15)
_h15 = d_high.reshape(N_DAYS, HA_NB, 15)
_l15 = d_low.reshape(N_DAYS, HA_NB, 15)
_c15 = d_close.reshape(N_DAYS, HA_NB, 15)
agg_o = _o15[:, :, 0]
agg_h = _h15.amax(2)
agg_l = _l15.amin(2)
agg_c = _c15[:, :, -1]


def _build_ha_close():
    ha_c = torch.zeros((N_DAYS, HA_NB), device=device)
    ha_o = torch.zeros((N_DAYS, HA_NB), device=device)
    ha_c[:, 0] = (agg_o[:, 0] + agg_c[:, 0]) / 2.0
    ha_o[:, 0] = (agg_o[:, 0] + agg_c[:, 0]) / 2.0
    for i in range(1, HA_NB):
        ha_c[:, i] = (agg_o[:, i] + agg_h[:, i] + agg_l[:, i] + agg_c[:, i]) / 4.0
        ha_o[:, i] = (ha_o[:, i - 1] + ha_c[:, i - 1]) / 2.0
    ha_h = torch.maximum(agg_h, torch.maximum(ha_o, ha_c))
    ha_l = torch.minimum(agg_l, torch.minimum(ha_o, ha_c))
    return ha_o, ha_h, ha_l, ha_c


HA_O, HA_H, HA_L, HA_C = _build_ha_close()


def _wilder_atr_ha(tr, period):
    """Wilder ATR over the 25 HA bars (vectorized across N days)."""
    atr = torch.zeros((N_DAYS, HA_NB), device=device)
    s = tr[:, :period].mean(dim=1)
    atr[:, period - 1] = s
    for i in range(period, HA_NB):
        s = (s * (period - 1) + tr[:, i]) / period
        atr[:, i] = s
    atr[:, :period - 1] = s.unsqueeze(1)
    return atr


def _ha_true_range():
    prev_c = torch.cat([HA_C[:, :1], HA_C[:, :-1]], dim=1)
    tr = torch.maximum(HA_H - HA_L, torch.maximum((HA_H - prev_c).abs(), (HA_L - prev_c).abs()))
    return tr


HA_TR = _ha_true_range()
ATR_HA_CACHE = {p: _wilder_atr_ha(HA_TR, p) for p in (10, 14, 20)}


def _utbot_colors(ha_c, atr_ha, key):
    """UT Bot trailing-stop color scan over 25 HA bars -> (N,25) int (1 green,-1 red,0 none).
    Mirrors IncrementalUTBotState: first candle only records previous_source (ts stays 0, color none)."""
    ts = torch.zeros(N_DAYS, device=device)
    prev = torch.zeros(N_DAYS, device=device)
    colors = torch.zeros((N_DAYS, HA_NB), dtype=torch.int8, device=device)
    ones = torch.ones(N_DAYS, dtype=torch.int8, device=device)
    negones = -torch.ones(N_DAYS, dtype=torch.int8, device=device)
    for i in range(HA_NB):
        price = ha_c[:, i]
        if i == 0:
            prev = price  # reference returns "none"; ts stays 0
            continue
        nloss = key * atr_ha[:, i]
        up = price > ts
        prop_up = price - nloss
        prop_dn = price + nloss
        ts = torch.where(up,
                         torch.where(prev > ts, torch.maximum(ts, prop_up), prop_up),
                         torch.where(prev < ts, torch.minimum(ts, prop_dn), prop_dn))
        held = colors[:, i - 1]
        col = torch.where((prev <= ts) & (price > ts), ones,
                 torch.where((prev >= ts) & (price < ts), negones,
                 torch.where(held != 0, held,
                             torch.where(price > prev, ones, negones))))
        colors[:, i] = col
        prev = price
    return colors


# Precompute UT Bot colors for every (ut_key, atr_period) pair ONCE (12 combos) so the
# per-combo hot loop is fully vectorized (no 25-step GPU scan per combo -> no starvation).
MARNI_UT_KEYS = [0.5, 0.6, 0.7, 0.8]
MARNI_COLORS = {(k, ap): _utbot_colors(HA_C, ATR_HA_CACHE[ap], float(k))
                for k in MARNI_UT_KEYS for ap in (10, 14, 20)}

# 3-phase UT Bot color SETUP (the "red, green, red" / "green, red, green" RANGE).
# A bullish CE setup = a GREEN run (>=MARNI_MIN_RUN buckets) bracketed by RED on both
# sides; a bearish PE setup = a RED run bracketed by GREEN. The run's high/low defines
# the Fibonacci range. Computed ONCE per (ut_key, atr_period) and cached.
MARNI_MIN_RUN = 3
MARNI_SETUPS = {}


def _marni_setups(col, G=MARNI_MIN_RUN):
    """Return (ce_H, ce_L, ce_pb, pe_H, pe_L, pe_pb) per day for the 3-phase setups."""
    N = col.shape[0]
    gl = (col == 1).long(); rd = (col == -1).long()
    glen = torch.zeros(N, HA_NB, device=device, dtype=torch.long)
    rlen = torch.zeros(N, HA_NB, device=device, dtype=torch.long)
    glen[:, 0] = (gl[:, 0] == 1).long()
    rlen[:, 0] = (rd[:, 0] == 1).long()
    for j in range(1, HA_NB):
        glen[:, j] = torch.where(gl[:, j] == 1, glen[:, j - 1] + 1, 0)
        rlen[:, j] = torch.where(rd[:, j] == 1, rlen[:, j - 1] + 1, 0)
    ce_pb = torch.full((N,), -1, device=device, dtype=torch.long)
    pe_pb = torch.full((N,), -1, device=device, dtype=torch.long)
    ar = torch.arange(N, device=device)
    for j in range(G - 1, HA_NB - 1):
        if j + 1 >= HA_NB:
            break
        # bull: green run len>=G ending at j, RED after (j+1), RED before run start
        lend = glen[:, j]; start = j - lend + 1; before = start - 1
        b_red = torch.where(before < 0, torch.ones(N, dtype=torch.bool, device=device), col[ar, before] == -1)
        ok = (lend >= G) & (col[:, j + 1] == -1) & b_red & (ce_pb < 0)
        ce_pb = torch.where(ok, j, ce_pb)
        # bear: red run len>=G ending at j, GREEN after (j+1), GREEN before run start
        lrd = rlen[:, j]; st2 = j - lrd + 1; bf2 = st2 - 1
        b_grn = torch.where(bf2 < 0, torch.ones(N, dtype=torch.bool, device=device), col[ar, bf2] == 1)
        ok2 = (lrd >= G) & (col[:, j + 1] == 1) & b_grn & (pe_pb < 0)
        pe_pb = torch.where(ok2, j, pe_pb)
    ce_len = glen[ar, ce_pb.clamp(min=0)]; ce_st = ce_pb - ce_len + 1
    pe_len = rlen[ar, pe_pb.clamp(min=0)]; pe_st = pe_pb - pe_len + 1
    idx = torch.arange(HA_NB, device=device).unsqueeze(0)
    NEG = torch.tensor(float("-inf"), device=device)
    POS = torch.tensor(float("inf"), device=device)
    def range_hl(st, pb):
        mask = (idx >= st.unsqueeze(1)) & (idx <= pb.clamp(min=0).unsqueeze(1))
        # fill masked-out buckets with -inf/+inf so only the run contributes to max/min
        hh = torch.where(mask, HA_H, torch.full((N, HA_NB), NEG, device=device)).amax(1)
        ll = torch.where(mask, HA_L, torch.full((N, HA_NB), POS, device=device)).amin(1)
        return hh, ll
    ce_H, ce_L = range_hl(ce_st, ce_pb)
    pe_H, pe_L = range_hl(pe_st, pe_pb)
    return ce_H, ce_L, ce_pb, pe_H, pe_L, pe_pb


def _get_marni_setups(uk, ap):
    key = (uk, ap)
    if key not in MARNI_SETUPS:
        MARNI_SETUPS[key] = _marni_setups(MARNI_COLORS[key], MARNI_MIN_RUN)
    return MARNI_SETUPS[key]


def _marni_core_one(p):
    """Marni Core: 3-bar UT Bot COLOR RANGE -> Fibonacci entry/SL/TP.

    Rule: detect a 3-bar UT Bot color sequence on the 15m HA candles:
        bullish CE range = GREEN-RED-GREEN ; bearish PE range = RED-GREEN-RED.
    The high/low of those 3 bars defines a range [L,H]. Fibonacci levels:
        entry = 0.786 retracement of the range (deep pullback to the level),
        SL    = 1.115 or 1.25 extension BEYOND the range on the adverse side,
        TP    = 0.29 extension beyond the range on the favorable side, or 0 (= range extreme).
    Entry fires on the first subsequent 1-min bar that TOUCHES the 0.786 level
    (limit fill) within the same session; once per day.
    """
    key = float(p["ut_key"]); ap = int(p["atr_period"])
    sl_m = float(p["sl_m"]); tp_m = float(p["tp_m"])

    # 3-phase UT Bot color setup (red-green-red / green-red-green), precomputed per (uk,ap).
    # NOTE: treat the 3-phase as a REVERSAL — GREEN-RED-GREEN (ends up) -> PE (sell the top);
    # RED-GREEN-RED (ends down) -> CE (buy the bottom). This is the opposite of a raw
    # continuation read and is what the win-rate data calls for.
    grn_H, grn_L, grn_pb, red_H, red_L, red_pb = _get_marni_setups(key, ap)   # (N,)
    ce_H, ce_L, ce_pb = red_H, red_L, red_pb          # CE (long) from RED-GREEN-RED range
    ce_any = ce_pb >= 0
    pe_H, pe_L, pe_pb = grn_H, grn_L, grn_pb          # PE (short) from GREEN-RED-GREEN range
    pe_any = pe_pb >= 0
    ce_R = (ce_H - ce_L).clamp(min=1.0)
    pe_R = (pe_H - pe_L).clamp(min=1.0)

    # Classic Fibonacci retracement / extension of [L,H] (retracement measured from the base).
    ce_entry_lvl = ce_L + 0.786 * ce_R      # bull: 0.786 retracement up from L (shallow pullback)
    ce_sl_lvl    = ce_L - (sl_m - 1.0) * ce_R  # below L
    ce_tp_lvl    = ce_H + tp_m * ce_R       # above H (or = H if tp_m==0)
    pe_entry_lvl = pe_H - 0.786 * pe_R      # bear: 0.786 retracement down from H (shallow rally)
    pe_sl_lvl    = pe_H + (sl_m - 1.0) * pe_R  # above H
    pe_tp_lvl    = pe_L - tp_m * pe_R       # below L (or = L if tp_m==0)

    tt = torch.arange(T_BARS, device=device)
    def build(any_p, pb, entry_lvl, sl_lvl, tp_lvl, is_call):
        start = torch.where(any_p, (pb + 1) * 15, T_BARS + 1)   # monitor after the color run ends
        # 1-point buffer on the retracement touch (limit fill)
        if is_call:
            touch = d_low <= (entry_lvl.unsqueeze(1) + 1.0)
        else:
            touch = d_high >= (entry_lvl.unsqueeze(1) - 1.0)
        cand = touch & (tt.unsqueeze(0) >= start.unsqueeze(1))
        first_t = torch.where(cand.any(1), torch.argmax(cand.int(), 1),
                              torch.full((N_DAYS,), -1, dtype=torch.long, device=device))
        has = cand.any(1)
        ent = torch.zeros((N_DAYS, T_BARS), dtype=torch.bool, device=device)
        ent[torch.arange(N_DAYS, device=device), first_t] = has
        epx = torch.where(has, entry_lvl, torch.zeros(N_DAYS, device=device))   # limit-fill price
        sl = sl_lvl.unsqueeze(1).expand(-1, T_BARS)
        tp = tp_lvl.unsqueeze(1).expand(-1, T_BARS)
        return ent, sl, tp, epx

    ce_ent, ce_sl, ce_tp, ce_epx = build(ce_any, ce_pb, ce_entry_lvl, ce_sl_lvl, ce_tp_lvl, True)
    pe_ent, pe_sl, pe_tp, pe_epx = build(pe_any, pe_pb, pe_entry_lvl, pe_sl_lvl, pe_tp_lvl, False)

    if p["direction"] == "bull":
        pe_ent = torch.zeros_like(pe_ent); pe_epx = torch.zeros_like(pe_epx)
    elif p["direction"] == "bear":
        ce_ent = torch.zeros_like(ce_ent); ce_epx = torch.zeros_like(ce_epx)
    if not p["allow_friday"]:
        nf = (~d_friday).unsqueeze(1)
        ce_ent = ce_ent & nf
        pe_ent = pe_ent & nf
    return ce_ent, ce_sl, ce_tp, ce_epx, pe_ent, pe_sl, pe_tp, pe_epx


@torch.inference_mode()
def evaluate_marni_core_batch(param_dicts, day_mask=None):
    B = len(param_dicts)
    ce_e, ce_s, ce_t, ce_x, pe_e, pe_s, pe_t, pe_x = [], [], [], [], [], [], [], []
    for p in param_dicts:
        a, b, c, d, e, f, g, h = _marni_core_one(p)
        ce_e.append(a); ce_s.append(b); ce_t.append(c); ce_x.append(d)
        pe_e.append(e); pe_s.append(f); pe_t.append(g); pe_x.append(h)
    ce_ent = torch.stack(ce_e, 0); ce_sl = torch.stack(ce_s, 0); ce_tp = torch.stack(ce_t, 0); ce_epx = torch.stack(ce_x, 0)
    pe_ent = torch.stack(pe_e, 0); pe_sl = torch.stack(pe_s, 0); pe_tp = torch.stack(pe_t, 0); pe_epx = torch.stack(pe_x, 0)
    dl = T([p.get("daily_loss_pts", 30) * LOT_SIZE for p in param_dicts])
    dp = T([p.get("daily_profit_pts", 50) * LOT_SIZE for p in param_dicts])
    sess_end = T([BASE_SESSION_END for _ in param_dicts])
    ce_dict = simulate_direction_locked_batch(ce_ent, ce_sl, ce_tp, "CE", dl, sess_end, day_mask, dp, entry_px=ce_epx)
    pe_dict = simulate_direction_locked_batch(pe_ent, pe_sl, pe_tp, "PE", dl, sess_end, day_mask, dp, entry_px=pe_epx)
    return [merge_results(ce_dict.get(i, EMPTY), pe_dict.get(i, EMPTY)) for i in range(B)]


@torch.inference_mode()
def evaluate_batch(strat_id, param_dicts, day_mask=None):
    """Fused (B, N, T) evaluation of a whole batch of param dicts."""
    if strat_id == "B08":
        return evaluate_breakout_batch(param_dicts, day_mask)
    if strat_id == "B09":
        return evaluate_marni_core_batch(param_dicts, day_mask)
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

    s1_os = T([p["s1_os"] for p in param_dicts]).view(B, 1, 1)
    s4_ob = T([p["s4_ob"] for p in param_dicts]).view(B, 1, 1)
    sl_m = T([p["sl_m"] for p in param_dicts]).view(B, 1, 1)
    tp_m = T([p["tp_m"] for p in param_dicts]).view(B, 1, 1)
    max_trade_loss = T([p["max_trade_loss_rs"] for p in param_dicts]).view(B, 1, 1)
    daily_loss_rs = T([p["daily_loss_pts"] * LOT_SIZE for p in param_dicts])
    daily_profit_rs = T([p["daily_profit_pts"] * LOT_SIZE for p in param_dicts])
    sess_end = torch.tensor([p["sess_end"] for p in param_dicts], device=device)
    moneyness = T([p.get("moneyness", 0.5) for p in param_dicts]).view(B, 1, 1)

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
        if strat_id == "B08":
            p = suggest_breakout_params(trial)
        elif strat_id == "B09":
            p = suggest_marni_core_params(trial)
        else:
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
    ("B08", "B08: 5m Opening-Range Breakout (book translation)"),
    ("B09", "B09: Marni Core 15m-HA UT Bot color signal"),
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
                if strat_id == "B08":
                    pdicts.append(suggest_breakout_params(t))
                elif strat_id == "B09":
                    pdicts.append(suggest_marni_core_params(t))
                else:
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

