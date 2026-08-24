"""
CROSS-STRATEGY META-CONFIRMATION ENSEMBLE  (built on optimized_gpu_backtest)
============================================================================

Extends the fused (B,N,T) causal engine with three levers that raise robustness
AND trade frequency:

  1. META-CONFIRMATION  — MAX_COMPONENT=4 component strategies each vote on the
     entry bar (CE + PE). A trade fires only when >= confirm_k components agree
     on the same direction. Confirmation filters false signals; the other two
     levers restore frequency.
  2. WIDER ENTRY BANDS  — `band_relax` widens the s4_ob / s1_os thresholds so
     more bars qualify as entries (band_relax=0 == exact thresholds).
  3. INTRADAY RE-ENTRIES — `reentry=True` lifts the "1 trade/day/direction"
     lock: after a position exits, a new entry may fire the same day.

All three feed the SAME optimized simulate / _finalize path, so the §5 rules in
OPTIMIZED_GPU_BACKTEST.md still apply (no scalar readback in loops; caps are
coerced via base._to_scalar).

Run:  python cross_strategy_ensemble_gpu.py            (short smoke study)
      PARITY=1 python cross_strategy_ensemble_gpu.py    (ensemble==single-strategy)
"""

import os, sys, json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import numpy as np
import torch
import torch.nn.functional as F
import optuna
from optuna.samplers import TPESampler

import optimized_gpu_backtest as base

device = base.device
N_DAYS = base.N_DAYS
T_BARS = base.T_BARS
LOT_SIZE = base.LOT_SIZE
SLIPPAGE_PTS = base.SLIPPAGE_PTS
BASE_SESSION_START = base.BASE_SESSION_START
get_stoch = base.get_stoch
get_atr = base.get_atr
d_close = base.d_close
d_high = base.d_high
d_low = base.d_low
d_open = base.d_open
EMPTY = base.EMPTY
merge_results = base.merge_results
_to_scalar = base._to_scalar

MAX_COMP = 4
TF_MAP = {"B01": 1, "B02": 1, "B06": 1, "B03": 2, "B04": 3, "B05": 5}
BIG = 999999


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR + ENTRY-MASK BUILDER (reused for components AND ensemble trade mgmt)
# ═══════════════════════════════════════════════════════════════════════════
def _build_indicators_and_masks(pdicts, band_relax=0.0, need_masks=True):
    """Return (S1, S4, ATR, s1_os, s4_ob, sl_m, tp_m, sess_end, moneyness, vw,
    trade_ok, ce_mask, pe_mask) for a list of B param dicts.

    `band_relax` (float) widens the entry thresholds to admit more bars:
      CE: S4 >= s4_ob - relax   AND   S1 <= s1_os + relax
      PE: S4 <= 100-s4_ob + relax   AND   S1 >= 100-s1_os - relax

    `need_masks=False` skips the stochastic/mask computation (which requires
    s1_k/s4_k/s1_os/s4_ob) and returns only the risk params (ATR/sl/tp/...),
    so ens_params may omit the stochastic keys without crashing.
    """
    B = len(pdicts)
    # ATR is always required (used for ensemble risk + trade_ok)
    ATR = []
    for p in pdicts:
        tf = p.get("timeframe", TF_MAP.get(p.get("strat_id", "B07")))
        atr = get_atr(tf, p["atr_p"])
        if tf > 1:
            atr = atr.repeat_interleave(tf, 1)[:, :T_BARS]
        ATR.append(atr)
    ATR = torch.stack(ATR, 0)

    s1_os = base.T([p.get("s1_os", 25.0) for p in pdicts]).view(B, 1, 1)
    s4_ob = base.T([p.get("s4_ob", 70.0) for p in pdicts]).view(B, 1, 1)
    sl_m = base.T([p["sl_m"] for p in pdicts]).view(B, 1, 1)
    tp_m = base.T([p["tp_m"] for p in pdicts]).view(B, 1, 1)
    max_trade_loss = base.T([p["max_trade_loss_rs"] for p in pdicts]).view(B, 1, 1)
    sess_end = torch.tensor([p["sess_end"] for p in pdicts], device=device)
    moneyness = base.T([p.get("moneyness", 0.5) for p in pdicts]).view(B, 1, 1)

    vw = torch.zeros((B, N_DAYS, T_BARS), dtype=torch.bool, device=device)
    for i, p in enumerate(pdicts):
        so = p["sess_start_off"]; se = p["sess_end"]
        vw[i, :, BASE_SESSION_START + so:se] = True

    sl_dist = ATR * sl_m * 0.5 * LOT_SIZE
    trade_ok = sl_dist <= max_trade_loss

    if need_masks:
        S1, S4 = [], []
        for p in pdicts:
            tf = p.get("timeframe", TF_MAP.get(p.get("strat_id", "B07")))
            s1 = get_stoch(tf, p["s1_k"]); s4 = get_stoch(tf, p["s4_k"])
            if tf > 1:
                s1 = s1.repeat_interleave(tf, 1)[:, :T_BARS]
                s4 = s4.repeat_interleave(tf, 1)[:, :T_BARS]
            S1.append(s1); S4.append(s4)
        S1 = torch.stack(S1, 0); S4 = torch.stack(S4, 0)
        br = float(band_relax)
        ce_mask = (S4 >= (s4_ob - br)) & (S1 <= (s1_os + br)) & vw & trade_ok
        pe_s4_os = 100.0 - s4_ob
        pe_s1_ob = 100.0 - s1_os
        pe_mask = (S4 <= (pe_s4_os + br)) & (S1 >= (pe_s1_ob - br)) & vw & trade_ok
        return S1, S4, ATR, s1_os, s4_ob, sl_m, tp_m, sess_end, moneyness, vw, trade_ok, ce_mask, pe_mask
    # risk-only path: stoch/masks not needed, so ens_params may omit
    # s1_k/s4_k/s1_os/s4_ob without crashing (SoC: don't require unused keys)
    dummy = torch.zeros((B, N_DAYS, T_BARS), dtype=torch.bool, device=device)
    return dummy, dummy, ATR, s1_os, s4_ob, sl_m, tp_m, sess_end, moneyness, vw, trade_ok, dummy, dummy


# ═══════════════════════════════════════════════════════════════════════════
# MARNI TREND FILTER  —  5m / 15m Heikin-Ashi + UT Bot(key=1.0, period=10) + LinReg(11)
# ---------------------------------------------------------------------------
# Port of artifacts.f6_hybrid.marny_option_chart_backtest.Option5mHTFBias to
# GPU tensors, computed on the UNDERLYING (NIFTY) at the requested HTF.
# Trend-confirmation gate applied to every strategy:
#   CE (call/long) allowed ONLY when  HA close > LinReg(11)  AND  UT Bot == green
#   PE (put/short) allowed ONLY when  HA close < LinReg(11)  AND  UT Bot == red
# Returns (bull_ce, bear_pe) bool tensors of shape (N_DAYS, T_BARS).
# ═══════════════════════════════════════════════════════════════════════════
_TREND_CACHE = {}

def build_trend_filter(htf, key=1.0, period=10, linlen=11):
    if htf in (0, None):
        return None, None
    assert htf in (5, 15), "trend filter HTF must be 5 or 15"
    ck = (int(htf), round(float(key), 4), int(period), int(linlen))
    if ck in _TREND_CACHE:
        return _TREND_CACHE[ck]
    n_h = T_BARS // htf
    o = d_open.view(N_DAYS, n_h, htf)
    h = d_high.view(N_DAYS, n_h, htf)
    l = d_low.view(N_DAYS, n_h, htf)
    c = d_close.view(N_DAYS, n_h, htf)
    htf_open = o[:, :, 0]
    htf_high = h.amax(dim=2)
    htf_low = l.amin(dim=2)
    htf_close = c[:, :, -1]

    # Heikin-Ashi (causal over htf bars, vectorized across days)
    ha_open = torch.zeros_like(htf_close)
    ha_close = torch.zeros_like(htf_close)
    ha_open[:, 0] = (htf_open[:, 0] + htf_close[:, 0]) / 2.0
    ha_close[:, 0] = (htf_open[:, 0] + htf_high[:, 0] + htf_low[:, 0] + htf_close[:, 0]) / 4.0
    for i in range(1, n_h):
        ha_open[:, i] = (ha_open[:, i - 1] + ha_close[:, i - 1]) / 2.0
        ha_close[:, i] = (htf_open[:, i] + htf_high[:, i] + htf_low[:, i] + htf_close[:, i]) / 4.0

    # UT Bot: Wilder ATR(period) on htf bars, trailing stop distance = key * ATR
    prev_close = torch.roll(htf_close, 1, dims=1)
    prev_close[:, 0] = htf_close[:, 0]
    tr = torch.maximum(htf_high - htf_low,
                        torch.maximum((htf_high - prev_close).abs(), (htf_low - prev_close).abs()))
    ub_period = int(period)
    atr = torch.zeros_like(tr)
    run = tr[:, 0].clone()
    atr[:, 0] = run
    for i in range(1, n_h):
        if i < ub_period:
            run = (run * i + tr[:, i]) / (i + 1)
        else:
            run = (run * (ub_period - 1) + tr[:, i]) / ub_period
        atr[:, i] = run

    # UT Bot trailing stop + persistent color state (+1 green / -1 red / 0 blue)
    trailing = torch.zeros((N_DAYS,), device=device)
    prev_src = torch.zeros((N_DAYS,), device=device)
    position = torch.zeros((N_DAYS,), dtype=torch.long, device=device)
    color = torch.zeros((N_DAYS, n_h), dtype=torch.long, device=device)
    for i in range(n_h):
        src = htf_close[:, i]
        loss = float(key) * atr[:, i]
        pstop, psrc = trailing, prev_src
        new_stop = torch.where(
            (src > pstop) & (psrc > pstop), torch.maximum(pstop, src - loss),
            torch.where((src < pstop) & (psrc < pstop), torch.minimum(pstop, src + loss),
                        torch.where(src > pstop, src - loss, src + loss)))
        up = (psrc < pstop) & (src > pstop)
        dn = (psrc > pstop) & (src < pstop)
        pos = torch.where(up, torch.ones_like(position),
                          torch.where(dn, -torch.ones_like(position), position))
        color[:, i] = torch.where(pos == 1, 1, torch.where(pos == -1, -1, 0))
        trailing, prev_src, position = new_stop, src, pos

    # LinReg(linlen): linlen-period SMA of ha_close (left-padded replicate)
    ha_r = ha_close.unsqueeze(1)
    linreg = F.avg_pool1d(F.pad(ha_r, (int(linlen) - 1, 0), mode="replicate"),
                          kernel_size=int(linlen), stride=1).squeeze(1)

    bull = (ha_close > linreg) & (color == 1)
    bear = (ha_close < linreg) & (color == -1)
    out = (bull.repeat_interleave(htf, 1), bear.repeat_interleave(htf, 1))
    _TREND_CACHE[ck] = out
    return out


_VOL_CACHE = {}


def build_vol_filter(atr_p=29, lookback=60, lo=20.0, hi=80.0):
    """Volatility-regime gate (web-validated whipsaw guard).

    ATR percentile rank of the current 1-min ATR within the last `lookback` bars.
    Entry is allowed only when the rank (0-100) lies inside [lo, hi]. Excluding the
    low-vol tail (chop / false breakouts) and the high-vol tail (exhaustion) has been
    shown to cut false positives materially. Bars before `lookback` get rank=50 (neutral).
    """
    ck = (int(atr_p), int(lookback), round(float(lo), 2), round(float(hi), 2))
    if ck in _VOL_CACHE:
        return _VOL_CACHE[ck]
    a = get_atr(1, int(atr_p))                       # (N, T) 1-min ATR, cached
    L = int(lookback)
    if L >= a.shape[1]:
        pctl = torch.full_like(a, 50.0)
    else:
        win = a.unfold(1, L, 1)                      # (N, T-L+1, L)
        cur = a[:, L - 1:].unsqueeze(-1)             # (N, T-L+1, 1)
        pctl = (win <= cur).float().mean(dim=2) * 100.0
        full = torch.full_like(a, 50.0)
        full[:, L - 1:] = pctl
        pctl = full
    gate = (pctl >= float(lo)) & (pctl <= float(hi))
    _VOL_CACHE[ck] = gate
    return gate


# ═══════════════════════════════════════════════════════════════════════════
# RE-ENTRY-CAPABLE SIMULATE  (forward greedy scan over T, multi-entry per day)
# ═══════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def simulate_direction_locked_batch_reentry(entries_mask, sl_tensor, tp_tensor, direction,
                                            daily_loss_rs, sess_end, day_mask=None,
                                            daily_profit_rs=None, reentry=None):
    """Like base.simulate_direction_locked_batch but allows >1 entry/day.

    `reentry` is a (B,) bool tensor. When True, a new entry may fire the bar
    after the previous trade exits; when False, only the first entry per day is
    taken (== base engine behaviour, for parity).
    """
    B = entries_mask.shape[0]
    if day_mask is not None:
        entries_mask = entries_mask & day_mask.unsqueeze(0).unsqueeze(-1)
    if not entries_mask.any():
        return {i: dict(EMPTY) for i in range(B)}

    if reentry is None:
        reentry = torch.zeros(B, dtype=torch.bool, device=device)
    reentry_b = reentry.view(B).bool()

    avail = torch.zeros((B, N_DAYS), dtype=torch.long, device=device)  # earliest allowable bar
    se_long = sess_end.long()

    E_b, E_n, E_t, E_pnl = [], [], [], []
    for t in range(T_BARS):
        bar_mask = entries_mask[:, :, t]
        enter = bar_mask & (avail <= t)
        if not enter.any():
            continue
        idx = torch.nonzero(enter, as_tuple=False)            # (K,2) [b,n]
        eb = idx[:, 0]; en = idx[:, 1]
        exit_off, pnl = _sim_exits(eb, en, t, sl_tensor, tp_tensor, direction, se_long[eb])
        E_b.append(eb); E_n.append(en)
        E_t.append(torch.full((eb.shape[0],), t, device=device, dtype=torch.long))
        E_pnl.append(pnl)
        # block-until-exit (re-entry) or block-rest-of-day (1 trade/day)
        new_avail = torch.where(reentry_b[eb], t + 1 + exit_off, se_long[eb])
        avail[eb, en] = new_avail

    if not E_b:
        return {i: dict(EMPTY) for i in range(B)}

    b_idx = torch.cat(E_b); d_idx = torch.cat(E_n); bar_idx = torch.cat(E_t)
    pnl = torch.cat(E_pnl)
    has_future = (bar_idx + 1) < se_long[b_idx]
    b_idx = b_idx[has_future]; d_idx = d_idx[has_future]; bar_idx = bar_idx[has_future]
    pnl = pnl[has_future]
    if pnl.shape[0] == 0:
        return {i: dict(EMPTY) for i in range(B)}

    b_np = b_idx.cpu().numpy(); d_np = d_idx.cpu().numpy()
    bar_np = bar_idx.cpu().numpy(); r_np = (pnl * LOT_SIZE - base.FEE).cpu().numpy()

    out = {i: dict(EMPTY) for i in range(B)}
    for bi in np.unique(b_np):
        m = b_np == bi
        out[int(bi)] = base._finalize(r_np[m], d_np[m], bar_np[m],
                                      daily_loss_rs[int(bi)], daily_profit_rs[int(bi)])
    return out


def _sim_exits(eb, en, t, sl_tensor, tp_tensor, direction, se_per):
    """Per-entering-position exit offset (bars) + raw points P&L."""
    K = eb.shape[0]
    ci = (t + 1) + torch.arange(340, device=device)
    valid = (ci < se_per.unsqueeze(1)) & (ci < 375)
    cs = ci.clamp(max=374)
    fh = d_high[en.unsqueeze(1), cs]
    fl = d_low[en.unsqueeze(1), cs]
    fh = torch.where(valid, fh, base.T(-1e9))
    fl = torch.where(valid, fl, base.T(1e9))
    sl_p = sl_tensor[eb, en, t]
    tp_p = tp_tensor[eb, en, t]
    if direction == "CE":
        hit_sl = fl <= sl_p.unsqueeze(1)
        hit_tp = fh >= tp_p.unsqueeze(1)
        entry_eff = d_close[en, t] + SLIPPAGE_PTS * 0.5
        exit_sl = sl_p - SLIPPAGE_PTS * 0.5
        exit_tp = tp_p - SLIPPAGE_PTS * 0.5
        eod_bar = (se_per - 1).clamp(max=374).long()
        exit_eod = d_close[en, eod_bar] - SLIPPAGE_PTS * 0.5
    else:
        hit_sl = fh >= sl_p.unsqueeze(1)
        hit_tp = fl <= tp_p.unsqueeze(1)
        entry_eff = d_close[en, t] - SLIPPAGE_PTS * 0.5
        exit_sl = sl_p + SLIPPAGE_PTS * 0.5
        exit_tp = tp_p + SLIPPAGE_PTS * 0.5
        eod_bar = (se_per - 1).clamp(max=374).long()
        exit_eod = d_close[en, eod_bar] + SLIPPAGE_PTS * 0.5
    sl_any = hit_sl.any(1); tp_any = hit_tp.any(1)
    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), 1), base.T(BIG))
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), 1), base.T(BIG))
    sl_ex = sl_any & (sl_first <= tp_first)
    tp_ex = tp_any & (~sl_ex)
    eod_off = (se_per - 1 - t).clamp(min=0)
    exit_off = torch.where(sl_ex, sl_first, torch.where(tp_ex, tp_first, eod_off))
    exit_px = torch.where(sl_ex, exit_sl, torch.where(tp_ex, exit_tp, exit_eod))
    if direction == "CE":
        raw_pts = (exit_px - entry_eff) * 0.50
    else:
        raw_pts = (entry_eff - exit_px) * 0.50
    return exit_off.long(), raw_pts


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_ensemble_batch(pairs, day_mask=None):
    """pairs: list of (components, ens_params); components = list of MAX_COMP dicts."""
    B = len(pairs)
    n_comp = len(pairs[0][0])
    confirm_k = base.T([int(p[1]["confirm_k"]) for p in pairs]).view(B, 1, 1)

    ce_vote = torch.zeros(B, N_DAYS, T_BARS, device=device)
    pe_vote = torch.zeros_like(ce_vote)
    for c in range(n_comp):
        comps = [p[0][c] for p in pairs]
        _, _, _, _, _, _, _, _, _, _, _, ce_m, pe_m = _build_indicators_and_masks(comps, 0.0)
        ce_vote += ce_m.float()
        pe_vote += pe_m.float()
    ce_ens = ce_vote >= confirm_k
    pe_ens = pe_vote >= confirm_k

    # MARNI trend-confirmation gate (global for the run): CE only with uptrend,
    # PE only with downtrend. trend_filter in {5, 15, None/0}.
    tf = pairs[0][1].get("trend_filter", None)
    if tf:
        m = pairs[0][1].get("marni") or {}
        bull, bear = build_trend_filter(tf, m.get("key", 1.0), m.get("period", 10), m.get("linlen", 11))
        ce_ens = ce_ens & bull.unsqueeze(0)
        pe_ens = pe_ens & bear.unsqueeze(0)

    vf = pairs[0][1].get("vol_filter", None)
    if vf:
        vgate = build_vol_filter(vf.get("atr_p", 29), vf.get("lookback", 60),
                                 vf.get("lo", 20.0), vf.get("hi", 80.0))
        ce_ens = ce_ens & vgate.unsqueeze(0)
        pe_ens = pe_ens & vgate.unsqueeze(0)

    # ensemble trade management uses ens_params (its own SL/TP/risk)
    eps = [p[1] for p in pairs]
    _, _, ATRe, _, _, sl_m, tp_m, sess_end, moneyness, vw, trade_ok, _, _ = \
        _build_indicators_and_masks(eps, 0.0, need_masks=False)
    offset = (moneyness - 0.5) * 2.0 * ATRe
    ce_sl = d_close - offset - ATRe * sl_m
    ce_tp = d_close - offset + ATRe * tp_m
    pe_sl = d_close + offset + ATRe * sl_m
    pe_tp = d_close + offset - ATRe * tp_m
    daily_loss_rs = torch.tensor([ep["daily_loss_pts"] * LOT_SIZE for ep in eps], device=device)
    daily_profit_rs = torch.tensor([ep["daily_profit_pts"] * LOT_SIZE for ep in eps], device=device)
    reentry = torch.tensor([bool(ep.get("reentry", False)) for ep in eps], device=device)

    ce_dict = simulate_direction_locked_batch_reentry(ce_ens, ce_sl, ce_tp, "CE",
                                                      daily_loss_rs, sess_end, day_mask,
                                                      daily_profit_rs, reentry)
    pe_dict = simulate_direction_locked_batch_reentry(pe_ens, pe_sl, pe_tp, "PE",
                                                      daily_loss_rs, sess_end, day_mask,
                                                      daily_profit_rs, reentry)
    return [merge_results(ce_dict.get(i, EMPTY), pe_dict.get(i, EMPTY)) for i in range(B)]


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER SUGGESTION + SCORING
# ═══════════════════════════════════════════════════════════════════════════
def _component(trial, tag):
    tf = trial.suggest_categorical(f"{tag}_tf", [1, 2, 3, 5])
    return {"strat_id": "B07", "timeframe": tf,
            "s1_k": trial.suggest_int(f"{tag}_s1k", 5, 30),
            "s4_k": trial.suggest_int(f"{tag}_s4k", 20, 120, step=5),
            "s1_os": trial.suggest_float(f"{tag}_s1os", 10.0, 40.0, step=2.5),
            "s4_ob": trial.suggest_float(f"{tag}_s4ob", 65.0, 90.0, step=2.5),
            "atr_p": trial.suggest_int(f"{tag}_atrp", 8, 35),
            "sl_m": 2.0, "tp_m": 4.0,            # unused for voting; kept for schema
            "daily_loss_pts": 10, "daily_profit_pts": 50,
            "moneyness": 0.5, "max_trade_loss_rs": 9999,
            "sess_start_off": 5, "sess_end_off": 45, "sess_end": 300}


def suggest_ensemble(trial):
    components = [_component(trial, f"c{i}") for i in range(MAX_COMP)]
    ens = {"timeframe": trial.suggest_categorical("ens_tf", [1, 2, 3, 5]),
           "s1_k": 13, "s4_k": 70, "s1_os": 15.0, "s4_ob": 77.5,
           "atr_p": trial.suggest_int("ens_atrp", 8, 35),
           "sl_m": trial.suggest_float("ens_sl", 1.0, 5.0, step=0.1),
           "tp_m": trial.suggest_float("ens_tp", 2.0, 12.0, step=0.25),
           "daily_loss_pts": trial.suggest_int("ens_dl", 10, 50, step=5),
           "daily_profit_pts": trial.suggest_int("ens_dp", 30, 80, step=10),
           "moneyness": trial.suggest_categorical("ens_money", [0.5, 0.6, 0.7]),
           "max_trade_loss_rs": 9999,
           "sess_start_off": trial.suggest_int("ens_sso", 0, 30, step=5),
           "sess_end_off": trial.suggest_int("ens_seo", 30, 75, step=15),
           "sess_end": 345 - trial.suggest_int("ens_seo", 30, 75, step=15),
           "confirm_k": trial.suggest_int("confirm_k", 1, MAX_COMP),
           "band_relax": trial.suggest_float("band_relax", 0.0, 15.0, step=1.0),
           "reentry": trial.suggest_categorical("reentry", [False, True])}
    if ens["tp_m"] < 1.5 * ens["sl_m"]:
        raise optuna.TrialPruned()
    return components, ens


def score_ensemble(res):
    return base._qp_score(res)


def search_ensemble(day_mask, n_trials, seed, bs=100):
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=seed, constant_liar=True, multivariate=True))
    n_batches = max(1, n_trials // bs)
    for _ in range(n_batches):
        batch = [study.ask() for _ in range(bs)]
        pds, keep = [], []
        for t in batch:
            try:
                pds.append(suggest_ensemble(t)); keep.append(t)
            except optuna.TrialPruned:
                study.tell(t, -999.0)
        if not pds:
            continue
        res_list = evaluate_ensemble_batch(pds, day_mask)
        for t, res in zip(keep, res_list):
            sc = score_ensemble(res)
            for k, v in res.items():
                t.set_user_attr(k, v)
            study.tell(t, sc)
    return study


# ═══════════════════════════════════════════════════════════════════════════
# PARITY: ensemble (4 identical components, confirm_k=1, relax=0, reentry=False)
# must equal base.evaluate_batch for the same single-strategy params.
# ═══════════════════════════════════════════════════════════════════════════
def ensemble_parity_check():
    print("=== ENSEMBLE PARITY (4 identical comps, confirm_k=1, relax=0, no reentry) ===", flush=True)
    single = dict(timeframe=3, s1_k=7, s4_k=50, s1_os=25.0, s4_ob=70.0, atr_p=10,
                  sl_m=1.5, tp_m=5.0, daily_loss_pts=8, daily_profit_pts=50, moneyness=0.5,
                  max_trade_loss_rs=1500, sess_start_off=0, sess_end_off=30, sess_end=315)
    comps = [dict(single) for _ in range(MAX_COMP)]
    ens = dict(single); ens["confirm_k"] = 1; ens["band_relax"] = 0.0; ens["reentry"] = False
    masks = [("NW", None), ("WF", base.d_is_mask), ("OOS", base.d_oos_mask)]
    ok = True
    for tag, dm in masks:
        bat = base.evaluate_batch("B07", [single], dm)[0]
        ens_r = evaluate_ensemble_batch([(comps, ens)], dm)[0]
        for k in ["trades", "win_rate", "net_rs", "pf", "max_dd"]:
            a, b = bat.get(k, 0), ens_r.get(k, 0)
            if abs(float(a) - float(b)) > 1e-2:
                ok = False
                print(f"  MISMATCH {tag}/{k}: single={a} ens={b}", flush=True)
        print(f"  {tag}: single net={bat['net_rs']:+,.1f} T={bat['trades']} | ens net={ens_r['net_rs']:+,.1f} T={ens_r['trades']}", flush=True)
    print("ENSEMBLE PARITY:", "PASS" if ok else "FAIL", flush=True)
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    if os.environ.get("PARITY") == "1":
        ensemble_parity_check()
        return
    n_trials = int(os.environ.get("TRIALS", "300"))
    print(f"\n=== CROSS-STRATEGY ENSEMBLE STUDY: {n_trials} trials (NW) ===", flush=True)
    t0 = time.time()
    study_nw = search_ensemble(None, n_trials, 42, bs=100)
    nw_params = study_nw.best_trial.params
    nw = study_nw.best_trial.user_attrs

    study_wf = search_ensemble(base.d_is_mask, n_trials, 42, bs=100)
    wf_pair = _pair_from_best(study_wf.best_trial)
    oos = evaluate_ensemble_batch([wf_pair], base.d_oos_mask)[0]

    print(f"\nNW Rs {nw.get('net_rs',0):+,.0f} WR={nw.get('win_rate',0):.1f}% PF={nw.get('pf',0):.2f} "
          f"T={nw.get('trades',0)} DD=Rs {nw.get('max_dd',0):,.0f}", flush=True)
    print(f"OOS Rs {oos['net_rs']:+,.0f} WR={oos['win_rate']:.1f}% PF={oos['pf']:.2f} "
          f"T={oos['trades']} DD=Rs {oos['max_dd']:,.0f}", flush=True)
    print(f"best NW params: {nw_params}", flush=True)
    out = HERE / "ensemble_candidates.json"
    with open(out, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "n_trials": n_trials,
                   "nw": nw_params, "nw_attrs": nw,
                   "wf_pair": [wf_pair[0], wf_pair[1]],
                   "oos": {k: oos.get(k, 0) for k in ["win_rate", "pf", "net_rs", "max_dd", "trades", "ce_trades", "pe_trades"]}},
                  f, indent=2)
    print(f"Saved: {out}  ({time.time()-t0:.0f}s)", flush=True)


def _pair_from_best(trial):
    """Rebuild a (components, ens) pair from a winning trial's sampled params."""
    components = []
    for i in range(MAX_COMP):
        tag = f"c{i}"
        components.append({"strat_id": "B07",
                           "timeframe": trial.params[f"{tag}_tf"],
                           "s1_k": trial.params[f"{tag}_s1k"], "s4_k": trial.params[f"{tag}_s4k"],
                           "s1_os": trial.params[f"{tag}_s1os"], "s4_ob": trial.params[f"{tag}_s4ob"],
                           "atr_p": trial.params[f"{tag}_atrp"],
                           "sl_m": 2.0, "tp_m": 4.0, "daily_loss_pts": 10, "daily_profit_pts": 50,
                           "moneyness": 0.5, "max_trade_loss_rs": 9999,
                           "sess_start_off": 5, "sess_end_off": 45, "sess_end": 300})
    ens = {"timeframe": trial.params["ens_tf"], "s1_k": 13, "s4_k": 70,
           "s1_os": 15.0, "s4_ob": 77.5, "atr_p": trial.params["ens_atrp"],
           "sl_m": trial.params["ens_sl"], "tp_m": trial.params["ens_tp"],
           "daily_loss_pts": trial.params["ens_dl"], "daily_profit_pts": trial.params["ens_dp"],
           "moneyness": trial.params["ens_money"], "max_trade_loss_rs": 9999,
           "sess_start_off": trial.params["ens_sso"], "sess_end_off": trial.params["ens_seo"],
           "sess_end": 345 - trial.params["ens_seo"],
           "confirm_k": trial.params["confirm_k"], "band_relax": trial.params["band_relax"],
           "reentry": trial.params["reentry"]}
    return components, ens


if __name__ == "__main__":
    main()

