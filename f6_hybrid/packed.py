"""Numba-packed F6 execution mirroring ``simulate_day_signal_state`` exactly.

The packed kernel keeps candidate state in flat arrays and reproduces the
reference position loop (entry, SL/TP, shutdown, bearish-peak reversal, EOD)
trade-for-trade. It is a benchmark vehicle: parity with the Python path is
asserted before any speedup number is trusted. Nothing in the reference
engine is modified by this module.
"""

import numpy as np

try:
    from numba import njit, prange

    _NUMBA_OK = True
except ImportError:  # pragma: no cover
    njit = None
    prange = None
    _NUMBA_OK = False

import grid_optimize_f6_atr as grid
from f6_hybrid.raw_features import (
    BaseStateCache,
    base_key_for,
    build_day_base_state,
    materialize_signal_state,
)

LOT_SIZE = grid.LOT_SIZE
DAILY_LOSS_RS = grid.DAILY_LOSS_RS
SESSION_START = grid.SESSION_START
SESSION_END = grid.SESSION_END
DAY_LAST = grid.DAY_LAST
CE_SIDE = 0
PE_SIDE = 1

DIV_CAP = 40
TRADE_CAP = 64
SIG_CAP = 768
MAX_SYMS = 64
MAX_MINUTES = 400

TF_TO_CODE = {name: index for index, name in enumerate(grid.TF_SPECS)}
TF_NAMES = list(grid.TF_SPECS)
REASON_NAMES = {
    0: "SHUTDOWN_LOSS",
    1: "SL",
    2: "TP",
    3: "BEARISH_PEAK_REVERSAL",
    4: "EOD",
}
SIDE_NAMES = {CE_SIDE: "CE", PE_SIDE: "PE"}


@njit(cache=True)
def _has_bearish_peak_divergence(prices, s1s, n):
    """Mirror ``DivergenceEngine.has_bearish_peak_divergence`` on a prefix."""
    if n < 6:
        return False
    recent = 10 if n > 10 else n
    p2_idx = n - recent
    for i in range(n - recent + 1, n):
        if prices[i] > prices[p2_idx]:
            p2_idx = i
    end_idx = p2_idx - 2
    if end_idx < 0:
        end_idx = 0
    start_idx = p2_idx - 30
    if start_idx < 0:
        start_idx = 0
    if end_idx <= start_idx:
        return False
    p1_idx = start_idx
    for i in range(start_idx + 1, end_idx):
        if prices[i] > prices[p1_idx]:
            p1_idx = i
    return prices[p2_idx] > prices[p1_idx] and s1s[p2_idx] < s1s[p1_idx]


@njit(cache=True)
def _store_trade(
    index,
    entry_min,
    exit_min,
    side,
    slot,
    entry,
    exit,
    pts,
    sl_pts,
    tp_pts,
    reason,
    dur,
    tf,
    t_entry_min,
    t_exit_min,
    t_side,
    t_slot,
    t_entry,
    t_exit,
    t_pts,
    t_rs,
    t_sl_pts,
    t_tp_pts,
    t_reason,
    t_dur,
    t_tf,
):
    t_entry_min[index] = entry_min
    t_exit_min[index] = exit_min
    t_side[index] = side
    t_slot[index] = slot
    t_entry[index] = entry
    t_exit[index] = exit
    t_pts[index] = pts
    t_rs[index] = int(round(pts * float(LOT_SIZE)))
    t_sl_pts[index] = sl_pts
    t_tp_pts[index] = tp_pts
    t_reason[index] = reason
    t_dur[index] = dur
    t_tf[index] = tf


@njit(cache=True)
def _run_candidate_body(
    cand,
    ns,
    sym_min,
    sym_o,
    sym_h,
    sym_l,
    sym_c,
    sym_n,
    slot_map,
    base_strike,
    n_strikes,
    pre_c,
    pre_s,
    pre_n,
    prev_s1,
    spot_min,
    spot_close,
    spot_n,
    sl_mult,
    tp_mult,
    consec_loss,
    sig_min,
    sig_side,
    sig_strike,
    sig_rev,
    sig_tf,
    sig_sl,
    sig_tp,
    sig_atr,
    n_sig,
    t_entry_min,
    t_exit_min,
    t_side,
    t_slot,
    t_entry,
    t_exit,
    t_pts,
    t_rs,
    t_sl_pts,
    t_tp_pts,
    t_reason,
    t_dur,
    t_tf,
    n_trades,
    div_c,
    div_s,
    div_len,
):
    for s in range(ns):
        ln = pre_n[s]
        for i in range(ln):
            div_c[s, i] = pre_c[s, i]
            div_s[s, i] = pre_s[s, i]
        div_len[s] = ln

    sp = 0
    dpnl = 0.0
    closs = 0
    shut = False
    active = False
    sym = -1
    cur = 0
    entry = 0.0
    sl = 0.0
    tgt = 0.0
    last_px = 0.0
    dur = 0
    entry_min = 0
    t_sl_pts_v = 0.0
    t_tp_pts_v = 0.0
    t_tf_v = 0
    t_side_v = 0
    nt = 0
    dl_pts = DAILY_LOSS_RS / LOT_SIZE

    for minute in range(SESSION_START, DAY_LAST + 1):
        if active:
            while cur < sym_n[sym] and sym_min[sym, cur] < minute:
                cur += 1
            if cur < sym_n[sym] and sym_min[sym, cur] == minute:
                c = sym_c[sym, cur]
                last_px = c
                dur += 1
                if dpnl * LOT_SIZE + (c - entry) * LOT_SIZE <= DAILY_LOSS_RS:
                    pts = round(c - entry, 2)
                    _store_trade(
                        nt,
                        entry_min,
                        minute,
                        t_side_v,
                        sym,
                        entry,
                        c,
                        pts,
                        t_sl_pts_v,
                        t_tp_pts_v,
                        0,
                        dur,
                        t_tf_v,
                        t_entry_min[cand],
                        t_exit_min[cand],
                        t_side[cand],
                        t_slot[cand],
                        t_entry[cand],
                        t_exit[cand],
                        t_pts[cand],
                        t_rs[cand],
                        t_sl_pts[cand],
                        t_tp_pts[cand],
                        t_reason[cand],
                        t_dur[cand],
                        t_tf[cand],
                    )
                    nt += 1
                    dpnl += pts
                    active = False
                    shut = True
                    continue
                ex = -1.0
                reason = 0
                hc = sym_h[sym, cur]
                lc = sym_l[sym, cur]
                if hc >= tgt and lc <= sl:
                    ex = sl
                    reason = 1
                elif hc >= tgt:
                    ex = tgt
                    reason = 2
                elif lc <= sl:
                    ex = sl
                    reason = 1
                if ex < 0.0:
                    ps = prev_s1[sym]
                    if ps == ps:
                        if div_len[sym] == DIV_CAP:
                            for i in range(DIV_CAP - 1):
                                div_c[sym, i] = div_c[sym, i + 1]
                                div_s[sym, i] = div_s[sym, i + 1]
                            div_c[sym, DIV_CAP - 1] = c
                            div_s[sym, DIV_CAP - 1] = ps
                        else:
                            div_c[sym, div_len[sym]] = c
                            div_s[sym, div_len[sym]] = ps
                            div_len[sym] += 1
                        if _has_bearish_peak_divergence(
                            div_c[sym], div_s[sym], div_len[sym]
                        ):
                            ex = c
                            reason = 3
                if ex >= 0.0:
                    pts = round(ex - entry, 2)
                    _store_trade(
                        nt,
                        entry_min,
                        minute,
                        t_side_v,
                        sym,
                        entry,
                        ex,
                        pts,
                        t_sl_pts_v,
                        t_tp_pts_v,
                        reason,
                        dur,
                        t_tf_v,
                        t_entry_min[cand],
                        t_exit_min[cand],
                        t_side[cand],
                        t_slot[cand],
                        t_entry[cand],
                        t_exit[cand],
                        t_pts[cand],
                        t_rs[cand],
                        t_sl_pts[cand],
                        t_tp_pts[cand],
                        t_reason[cand],
                        t_dur[cand],
                        t_tf[cand],
                    )
                    nt += 1
                    dpnl += pts
                    if pts <= 0.0:
                        closs += 1
                    else:
                        closs = 0
                    if closs >= consec_loss[cand] or dpnl <= dl_pts:
                        shut = True
                    active = False
        if minute >= SESSION_END and active:
            pts = round(last_px - entry, 2)
            _store_trade(
                nt,
                entry_min,
                minute,
                t_side_v,
                sym,
                entry,
                last_px,
                pts,
                t_sl_pts_v,
                t_tp_pts_v,
                4,
                dur,
                t_tf_v,
                t_entry_min[cand],
                t_exit_min[cand],
                t_side[cand],
                t_slot[cand],
                t_entry[cand],
                t_exit[cand],
                t_pts[cand],
                t_rs[cand],
                t_sl_pts[cand],
                t_tp_pts[cand],
                t_reason[cand],
                t_dur[cand],
                t_tf[cand],
            )
            nt += 1
            dpnl += pts
            break
        if active or shut or minute >= SESSION_END:
            continue

        for k in range(n_sig[cand]):
            if sig_min[cand, k] != minute:
                continue
            while sp < spot_n and spot_min[sp] <= minute:
                sp += 1
            if sp == 0:
                continue
            s_close = spot_close[sp - 1]
            atm = int(round(s_close / 50.0) * 50.0)
            side_code = sig_side[cand, k]
            atk = atm + (-100 if side_code == CE_SIDE else 100)
            if atk != sig_strike[cand, k]:
                continue
            if atk < base_strike:
                continue
            rel = (atk - base_strike) // 50
            if rel >= n_strikes:
                continue
            slot = slot_map[rel * 2 + side_code]
            if slot < 0:
                continue
            entry_sym = slot
            if sig_rev[cand, k]:
                opp_side = 1 - side_code
                atk_opp = atm + (-100 if opp_side == CE_SIDE else 100)
                if atk_opp < base_strike:
                    continue
                rel_opp = (atk_opp - base_strike) // 50
                if rel_opp >= n_strikes:
                    continue
                entry_sym = slot_map[rel_opp * 2 + opp_side]
                if entry_sym < 0:
                    continue
            ecur = 0
            while (
                ecur < sym_n[entry_sym]
                and sym_min[entry_sym, ecur] < minute
            ):
                ecur += 1
            if ecur >= sym_n[entry_sym] or sym_min[entry_sym, ecur] != minute:
                continue
            atr = sig_atr[cand, k]
            if atr > 0.5:
                sl_use = atr * sl_mult[cand]
                tp_use = atr * tp_mult[cand]
            else:
                sl_use = sig_sl[cand, k]
                tp_use = sig_tp[cand, k]
            entry = sym_c[entry_sym, ecur]
            sl = entry - sl_use
            tgt = entry + tp_use
            active = True
            sym = entry_sym
            cur = ecur
            entry_min = minute
            dur = 0
            last_px = entry
            t_side_v = 1 - side_code if sig_rev[cand, k] else side_code
            t_tf_v = sig_tf[cand, k]
            t_sl_pts_v = round(sl_use, 2)
            t_tp_pts_v = round(tp_use, 2)
            break
    n_trades[cand] = nt




@njit(cache=True)
def _run_candidates_serial(
    nc,
    ns,
    sym_min,
    sym_o,
    sym_h,
    sym_l,
    sym_c,
    sym_n,
    slot_map,
    base_strike,
    n_strikes,
    pre_c,
    pre_s,
    pre_n,
    prev_s1,
    spot_min,
    spot_close,
    spot_n,
    sl_mult,
    tp_mult,
    consec_loss,
    sig_min,
    sig_side,
    sig_strike,
    sig_rev,
    sig_tf,
    sig_sl,
    sig_tp,
    sig_atr,
    n_sig,
    t_entry_min,
    t_exit_min,
    t_side,
    t_slot,
    t_entry,
    t_exit,
    t_pts,
    t_rs,
    t_sl_pts,
    t_tp_pts,
    t_reason,
    t_dur,
    t_tf,
    n_trades,
):
    div_c = np.empty((nc, ns, DIV_CAP))
    div_s = np.empty((nc, ns, DIV_CAP))
    div_len = np.empty((nc, ns), np.int64)
    for cand in range(nc):
        _run_candidate_body(
            cand,
            ns,
            sym_min,
            sym_o,
            sym_h,
            sym_l,
            sym_c,
            sym_n,
            slot_map,
            base_strike,
            n_strikes,
            pre_c,
            pre_s,
            pre_n,
            prev_s1,
            spot_min,
            spot_close,
            spot_n,
            sl_mult,
            tp_mult,
            consec_loss,
            sig_min,
            sig_side,
            sig_strike,
            sig_rev,
            sig_tf,
            sig_sl,
            sig_tp,
            sig_atr,
            n_sig,
            t_entry_min,
            t_exit_min,
            t_side,
            t_slot,
            t_entry,
            t_exit,
            t_pts,
            t_rs,
            t_sl_pts,
            t_tp_pts,
            t_reason,
            t_dur,
            t_tf,
            n_trades,
            div_c[cand],
            div_s[cand],
            div_len[cand],
        )


@njit(cache=True, parallel=True)
def _run_candidates_parallel(
    nc,
    ns,
    sym_min,
    sym_o,
    sym_h,
    sym_l,
    sym_c,
    sym_n,
    slot_map,
    base_strike,
    n_strikes,
    pre_c,
    pre_s,
    pre_n,
    prev_s1,
    spot_min,
    spot_close,
    spot_n,
    sl_mult,
    tp_mult,
    consec_loss,
    sig_min,
    sig_side,
    sig_strike,
    sig_rev,
    sig_tf,
    sig_sl,
    sig_tp,
    sig_atr,
    n_sig,
    t_entry_min,
    t_exit_min,
    t_side,
    t_slot,
    t_entry,
    t_exit,
    t_pts,
    t_rs,
    t_sl_pts,
    t_tp_pts,
    t_reason,
    t_dur,
    t_tf,
    n_trades,
):
    div_c = np.empty((nc, ns, DIV_CAP))
    div_s = np.empty((nc, ns, DIV_CAP))
    div_len = np.empty((nc, ns), np.int64)
    for cand in prange(nc):
        _run_candidate_body(
            cand,
            ns,
            sym_min,
            sym_o,
            sym_h,
            sym_l,
            sym_c,
            sym_n,
            slot_map,
            base_strike,
            n_strikes,
            pre_c,
            pre_s,
            pre_n,
            prev_s1,
            spot_min,
            spot_close,
            spot_n,
            sl_mult,
            tp_mult,
            consec_loss,
            sig_min,
            sig_side,
            sig_strike,
            sig_rev,
            sig_tf,
            sig_sl,
            sig_tp,
            sig_atr,
            n_sig,
            t_entry_min,
            t_exit_min,
            t_side,
            t_slot,
            t_entry,
            t_exit,
            t_pts,
            t_rs,
            t_sl_pts,
            t_tp_pts,
            t_reason,
            t_dur,
            t_tf,
            n_trades,
            div_c[cand],
            div_s[cand],
            div_len[cand],
        )


class _DayBundle:
    """Shared per-day arrays plus kernels already loaded for this process."""

    def __init__(self, state, base):
        sym_keys = list(state.slices.keys())
        self.prefix = state.prefix
        self.day = state.day
        self.sym_keys = sym_keys
        self.n_syms = len(sym_keys)
        self.strike_of = []
        self.side_of = []
        min_strike = None
        max_strike = None
        for key in sym_keys:
            match = grid.SYM_RE.match(key)
            strike = int(match.group(2))
            side = 0 if match.group(3) == "CE" else 1
            self.strike_of.append(strike)
            self.side_of.append(side)
            if min_strike is None or strike < min_strike:
                min_strike = strike
            if max_strike is None or strike > max_strike:
                max_strike = strike
        self.base_strike = int(min_strike)
        self.n_strikes = (int(max_strike) - self.base_strike) // 50 + 1
        slot_map = np.full(self.n_strikes * 2, -1, dtype=np.int64)
        for index in range(self.n_syms):
            rel = (self.strike_of[index] - self.base_strike) // 50
            slot_map[rel * 2 + self.side_of[index]] = index
        self.slot_map = slot_map

        self.sym_min = np.zeros((self.n_syms, MAX_MINUTES), dtype=np.int64)
        self.sym_o = np.zeros((self.n_syms, MAX_MINUTES))
        self.sym_h = np.zeros((self.n_syms, MAX_MINUTES))
        self.sym_l = np.zeros((self.n_syms, MAX_MINUTES))
        self.sym_c = np.zeros((self.n_syms, MAX_MINUTES))
        self.sym_n = np.zeros(self.n_syms, dtype=np.int64)
        for index, key in enumerate(sym_keys):
            values = state.slices[key]
            minutes = np.asarray(values["min"], dtype=np.int64)
            count = len(minutes)
            self.sym_min[index, :count] = minutes
            self.sym_o[index, :count] = np.asarray(values["open"], dtype=np.float64)
            self.sym_h[index, :count] = np.asarray(values["high"], dtype=np.float64)
            self.sym_l[index, :count] = np.asarray(values["low"], dtype=np.float64)
            self.sym_c[index, :count] = np.asarray(values["close"], dtype=np.float64)
            self.sym_n[index] = count

        self.pre_c = np.zeros((self.n_syms, DIV_CAP))
        self.pre_s = np.zeros((self.n_syms, DIV_CAP))
        self.pre_n = np.zeros(self.n_syms, dtype=np.int64)
        self.prev_s1 = np.full(self.n_syms, np.nan)
        for index, key in enumerate(sym_keys):
            close_c, s1_c = [], []
            for bar in (
                list(base.warmup_features.get(key, {}).get("1m", ()))
                + list(base.features.get(key, {}).get("1m", ()))
            ):
                if bar.s1 is not None:
                    close_c.append(bar.close)
                    s1_c.append(bar.s1)
            if len(close_c) > DIV_CAP:
                close_c, s1_c = close_c[-DIV_CAP:], s1_c[-DIV_CAP:]
            count = len(close_c)
            self.pre_c[index, :count] = np.asarray(close_c, dtype=np.float64)
            self.pre_s[index, :count] = np.asarray(s1_c, dtype=np.float64)
            self.pre_n[index] = count
            current_1m = base.features.get(key, {}).get("1m", ())
            if current_1m and current_1m[-1].s1 is not None:
                self.prev_s1[index] = float(current_1m[-1].s1)

        self.spot_min = np.asarray(state.spot["min"], dtype=np.int64)
        self.spot_close = np.asarray(state.spot["close"], dtype=np.float64)
        self.spot_n = len(self.spot_min)


def _pack_signals(state):
    entries = []
    for minute, signals in state.pmtrig.items():
        for ordinal, signal in enumerate(signals):
            side, strike, _symbol, c_px, is_rev, tf, sl_pts, tp_pts, atr_val = signal
            entries.append(
                (
                    minute,
                    ordinal,
                    0 if side == "CE" else 1,
                    int(strike),
                    1 if is_rev else 0,
                    TF_TO_CODE[tf],
                    float(sl_pts),
                    float(tp_pts),
                    float("nan") if atr_val is None else float(atr_val),
                )
            )
    entries.sort(key=lambda item: (item[0], item[1]))
    minutes = np.array([item[0] for item in entries], dtype=np.int64)
    sides = np.array([item[2] for item in entries], dtype=np.int64)
    strikes = np.array([item[3] for item in entries], dtype=np.int64)
    revs = np.array([item[4] for item in entries], dtype=np.int64)
    tfs = np.array([item[5] for item in entries], dtype=np.int64)
    sls = np.array([item[6] for item in entries], dtype=np.float64)
    tps = np.array([item[7] for item in entries], dtype=np.float64)
    atrs = np.array([item[8] for item in entries], dtype=np.float64)
    return minutes, sides, strikes, revs, tfs, sls, tps, atrs


def _pad_signal(minutes, values, cap):
    padded = np.zeros(cap, dtype=np.int64 if minutes.dtype == np.int64 else np.float64)
    padded[: len(minutes)] = values
    return padded


def run_bundle_candidates(bundle, states, params_list, parallel=False):
    """Run many candidates against a shared day bundle in one kernel call."""
    nc = len(params_list)
    sig_min = np.zeros((nc, SIG_CAP), dtype=np.int64)
    sig_side = np.zeros((nc, SIG_CAP), dtype=np.int64)
    sig_strike = np.zeros((nc, SIG_CAP), dtype=np.int64)
    sig_rev = np.zeros((nc, SIG_CAP), dtype=np.int64)
    sig_tf = np.zeros((nc, SIG_CAP), dtype=np.int64)
    sig_sl = np.zeros((nc, SIG_CAP))
    sig_tp = np.zeros((nc, SIG_CAP))
    sig_atr = np.zeros((nc, SIG_CAP))
    n_sig = np.zeros(nc, dtype=np.int64)
    for cand in range(nc):
        minutes, sides, strikes, revs, tfs, sls, tps, atrs = _pack_signals(
            states[cand]
        )
        count = len(minutes)
        if count > SIG_CAP:
            raise ValueError(f"signal cap exceeded: {count} > {SIG_CAP}")
        n_sig[cand] = count
        if count:
            sig_min[cand, :count] = minutes
            sig_side[cand, :count] = sides
            sig_strike[cand, :count] = strikes
            sig_rev[cand, :count] = revs
            sig_tf[cand, :count] = tfs
            sig_sl[cand, :count] = sls
            sig_tp[cand, :count] = tps
            sig_atr[cand, :count] = atrs

    sl_mult = np.array([p["atr_sl_mult"] for p in params_list], dtype=np.float64)
    tp_mult = np.array([p["atr_tp_mult"] for p in params_list], dtype=np.float64)
    consec = np.array([p["consec_loss"] for p in params_list], dtype=np.int64)

    t_entry_min = np.zeros((nc, TRADE_CAP), dtype=np.int64)
    t_exit_min = np.zeros((nc, TRADE_CAP), dtype=np.int64)
    t_side = np.zeros((nc, TRADE_CAP), dtype=np.int64)
    t_slot = np.zeros((nc, TRADE_CAP), dtype=np.int64)
    t_entry = np.zeros((nc, TRADE_CAP))
    t_exit = np.zeros((nc, TRADE_CAP))
    t_pts = np.zeros((nc, TRADE_CAP))
    t_rs = np.zeros((nc, TRADE_CAP), dtype=np.int64)
    t_sl_pts = np.zeros((nc, TRADE_CAP))
    t_tp_pts = np.zeros((nc, TRADE_CAP))
    t_reason = np.zeros((nc, TRADE_CAP), dtype=np.int64)
    t_dur = np.zeros((nc, TRADE_CAP), dtype=np.int64)
    t_tf = np.zeros((nc, TRADE_CAP), dtype=np.int64)
    n_trades = np.zeros(nc, dtype=np.int64)

    if parallel:
        _run_candidates_parallel(
                nc,
                bundle.n_syms,
                bundle.sym_min,
                bundle.sym_o,
                bundle.sym_h,
                bundle.sym_l,
                bundle.sym_c,
                bundle.sym_n,
                bundle.slot_map,
                bundle.base_strike,
                bundle.n_strikes,
                bundle.pre_c,
                bundle.pre_s,
                bundle.pre_n,
                bundle.prev_s1,
                bundle.spot_min,
                bundle.spot_close,
                bundle.spot_n,
                sl_mult,
                tp_mult,
                consec,
                sig_min,
                sig_side,
                sig_strike,
                sig_rev,
                sig_tf,
                sig_sl,
                sig_tp,
                sig_atr,
                n_sig,
                t_entry_min,
                t_exit_min,
                t_side,
                t_slot,
                t_entry,
                t_exit,
                t_pts,
                t_rs,
                t_sl_pts,
                t_tp_pts,
                t_reason,
                t_dur,
                t_tf,
                n_trades,
        )
    else:
        _run_candidates_serial(
                nc,
                bundle.n_syms,
                bundle.sym_min,
                bundle.sym_o,
                bundle.sym_h,
                bundle.sym_l,
                bundle.sym_c,
                bundle.sym_n,
                bundle.slot_map,
                bundle.base_strike,
                bundle.n_strikes,
                bundle.pre_c,
                bundle.pre_s,
                bundle.pre_n,
                bundle.prev_s1,
                bundle.spot_min,
                bundle.spot_close,
                bundle.spot_n,
                sl_mult,
                tp_mult,
                consec,
                sig_min,
                sig_side,
                sig_strike,
                sig_rev,
                sig_tf,
                sig_sl,
                sig_tp,
                sig_atr,
                n_sig,
                t_entry_min,
                t_exit_min,
                t_side,
                t_slot,
                t_entry,
                t_exit,
                t_pts,
                t_rs,
                t_sl_pts,
                t_tp_pts,
                t_reason,
                t_dur,
                t_tf,
                n_trades,
        )

    trades = [[] for _ in range(nc)]
    for cand in range(nc):
        count_trades = int(n_trades[cand])
        for index in range(count_trades):
            slot = int(t_slot[cand, index])
            strike = bundle.strike_of[slot]
            side_name = SIDE_NAMES[int(t_side[cand, index])]
            trades[cand].append(
                {
                    "date": bundle.day,
                    "entry_min": int(t_entry_min[cand, index]),
                    "exit_min": int(t_exit_min[cand, index]),
                    "side": SIDE_NAMES[int(t_side[cand, index])],
                    "symbol": f"{bundle.prefix}{strike}{side_name}",
                    "entry": float(t_entry[cand, index]),
                    "exit": float(t_exit[cand, index]),
                    "pts": float(t_pts[cand, index]),
                    "rs": int(t_rs[cand, index]),
                    "sl_pts": float(t_sl_pts[cand, index]),
                    "tp_pts": float(t_tp_pts[cand, index]),
                    "reason": REASON_NAMES[int(t_reason[cand, index])],
                    "duration_min": int(t_dur[cand, index]),
                    "tf": TF_NAMES[int(t_tf[cand, index])],
                }
            )
    return trades


_PACKED_CANDIDATES: list[dict] = []
_PACKED_PARALLEL = False


def _init_packed_worker(spot_all: dict, candidates: list[dict], parallel: bool) -> None:
    global _PACKED_CANDIDATES, _PACKED_PARALLEL
    grid.init_worker_local(spot_all)
    _PACKED_CANDIDATES = candidates
    _PACKED_PARALLEL = bool(parallel)


def _process_packed_day(args):
    day, fpath, fprev = args
    spot = grid.GLOBAL_SPOT.get(day)
    base_cache = BaseStateCache()
    entries = {}
    for params in _PACKED_CANDIDATES:
        key = base_key_for(day, fprev, params)
        base = base_cache.get_or_build(
            key,
            lambda params=params: build_day_base_state(
                day, fpath, fprev, params, spot
            ),
        )
        threshold_key = (key, params["f6_s4_thresh"], params["f6_s1_thresh"])
        if threshold_key not in entries:
            if base is None:
                entries[threshold_key] = None
            else:
                state = materialize_signal_state(
                    base, params["f6_s4_thresh"], params["f6_s1_thresh"]
                )
                entries[threshold_key] = (state, base)

    results = []
    bundles = {}
    state_bases = []
    for params in _PACKED_CANDIDATES:
        key = base_key_for(day, fprev, params)
        threshold_key = (key, params["f6_s4_thresh"], params["f6_s1_thresh"])
        entry = entries[threshold_key]
        results.append([])
        if entry is None:
            state_bases.append(None)
            continue
        state, base = entry
        if key not in bundles:
            bundles[key] = _DayBundle(state, base)
        state_bases.append((state, base))
    for base_key, bundle in bundles.items():
        group = [
            (index, params)
            for index, params in enumerate(_PACKED_CANDIDATES)
            if state_bases[index] is not None
            and base_key_for(day, fprev, params) == base_key
        ]
        if not group:
            continue
        states = [state_bases[index][0] for index, _ in group]
        params_run = [params for _, params in group]
        packed = run_bundle_candidates(bundle, states, params_run, _PACKED_PARALLEL)
        for (index, _), trades in zip(group, packed):
            results[index] = trades
    return day, results, base_cache.build_count, len(entries)


def run_packed_candidates(
    candidates: list[dict],
    days,
    files,
    spot_all,
    workers: int = 8,
    parallel: bool = False,
):
    """Evaluate candidates through the numba-packed execution path."""
    workers = max(1, min(8, int(workers)))
    if not _NUMBA_OK:
        raise ImportError("numba is required for the packed execution path")
    if not candidates or not days:
        return [[] for _ in candidates], 0, 0
    days = list(days)
    tasks = [
        (
            day,
            str(files[day]),
            str(files[days[index - 1]]) if index else "",
        )
        for index, day in enumerate(days)
    ]
    results = [[] for _ in candidates]
    base_builds = 0
    signal_builds = 0
    with grid.Pool(
        processes=workers,
        initializer=_init_packed_worker,
        initargs=(spot_all, candidates, parallel),
    ) as pool:
        for _, day_results, day_base_builds, day_signal_builds in pool.imap(
            _process_packed_day, tasks
        ):
            base_builds += day_base_builds
            signal_builds += day_signal_builds
            for index, trades in enumerate(day_results):
                results[index].extend(trades)
    return results, base_builds, signal_builds
