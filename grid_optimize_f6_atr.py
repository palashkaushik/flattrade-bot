"""Combinatorial Optimization — ATR×2/×4 Unlimited + F6 (Champion Strategy).

Phase 1: Optuna TPE intelligent search (2020-2022, 200 trials).
Phase 2: validate_top_candidates.py runs full 5Y on the top 15.

Engine: parameterized port of test_f6_combinations.py (atr_unlimited config).
All 8 optimization axes are threaded through P (params dict) per task:

  s1_k          fast stoch %K period          [7, 9, 12, 14]
  s4_k          slow stoch %K period          [50, 60, 75]
  atr_period    ATR lookback                  [10, 14, 20]
  atr_sl_mult   ATR SL multiplier             [1.5, 2.0, 2.5, 3.0]
  atr_tp_mult   ATR TP multiplier             [3.0, 4.0, 5.0, 6.0]
  f6_s4_thresh  F6 flag S4 >= threshold       [75.0, 79.5, 85.0]
  f6_s1_thresh  F6 flag S1 <= threshold       [15.0, 20.5, 25.0]
  consec_loss   consecutive-loss shutdown     [4, 6, 8]

Speed architecture:
  - ONE persistent multiprocessing Pool lives for the whole study.
  - Each worker caches parsed day files in a worker-local dict keyed by path
    (bounded at MAX_CACHE_ENTRIES). The first trial pays the I/O; all later
    trials reuse cached numpy arrays -> per-trial I/O ~ 0 ms.
  - Pruning: 2020 (year 1) is evaluated first; MedianPruner kills trials whose
    year-1 net profit is below the running median before 2021-22 runs.

Usage:
  python grid_optimize_f6_atr.py --smoke          # 5-day sanity on champion params
  python grid_optimize_f6_atr.py --verify         # reproduce champion on full 5Y
  python grid_optimize_f6_atr.py --trials 200     # full Phase 1 study (default 200)
"""

import csv
import json
import os
import sys
import time
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError:
    optuna = None

from backtest_5y_optimized import (
    load_spot, option_files, SYM_RE, to_minutes,
    latest_spot, summarize, print_yearly_breakdown,
)
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from flattrade_bot.indicators.divergence import DivergenceEngine
from flattrade_bot.indicators.ema import IncrementalEMA

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_LOSS_RS = -2000.0
DAILY_PROFIT_PTS = float("inf")

TF_SPECS = {
    "1m": (1, 10, 6.0, 30.0),
    "2m": (2, 5, 10.0, 15.0),
    "3m": (3, 4, 8.0, 25.0),
    "5m": (5, 3, 10.0, 35.0),
}

CHAMPION = {
    "s1_k": 9, "s1_d": 3, "s4_k": 60,
    "atr_period": 14, "atr_sl_mult": 2.0, "atr_tp_mult": 4.0,
    "f6_s4_thresh": 79.5, "f6_s1_thresh": 20.5, "consec_loss": 6,
}
CHAMPION_EXPECTED = {"trades": 7843, "wr": 48.0, "rs": 1030642, "pf": 1.45}

SEARCH_SPACE = {
    "s1_k": [7, 9, 12, 14],
    "s4_k": [50, 60, 75],
    "atr_period": [10, 14, 20],
    "atr_sl_mult": [1.5, 2.0, 2.5, 3.0],
    "atr_tp_mult": [3.0, 4.0, 5.0, 6.0],
    "f6_s4_thresh": [75.0, 79.5, 85.0],
    "f6_s1_thresh": [15.0, 20.5, 25.0],
    "consec_loss": [4, 6, 8],
}

MAX_CACHE_ENTRIES = 1600
# User preference (2026-08-10): fixed 8 workers instead of auto 85% CPU.
WORKERS = 8
GLOBAL_SPOT = {}
GLOBAL_CACHE = {}


def init_worker_local(spot_dict):
    global GLOBAL_SPOT, GLOBAL_CACHE
    GLOBAL_SPOT = spot_dict
    GLOBAL_CACHE = {}


def cached_day(path_str):
    """Worker-local parse cache keyed by file path (persists across trials)."""
    c = GLOBAL_CACHE.get(path_str)
    if c is not None:
        return c
    columnar_root = os.environ.get("F6_COLUMNAR_CACHE_DIR")
    if columnar_root:
        from f6_hybrid.columnar import cache_path_for, load_parquet_day

        packed_path = cache_path_for(Path(path_str), Path(columnar_root))
        if packed_path.exists():
            data = load_parquet_day(packed_path)
            GLOBAL_CACHE[path_str] = data
            return data
    try:
        df = pd.read_csv(path_str, usecols=["time", "symbol", "open", "high", "low", "close"], engine="c")
        if df.empty:
            GLOBAL_CACHE[path_str] = None
            return None
        df["min"] = np.array([to_minutes(t) for t in df["time"]])
        # Blind archives can contain the same symbol/minute block twice.
        # Keep the last source row and restore the cursor's strict ordering.
        df = df.drop_duplicates(subset=["symbol", "min"], keep="last")
        df = df.sort_values(["symbol", "min"], kind="stable")
        data = {}
        for sym, g in df.groupby("symbol"):
            data[sym] = {
                "min": g["min"].to_numpy(),
                "open": g["open"].to_numpy(),
                "high": g["high"].to_numpy(),
                "low": g["low"].to_numpy(),
                "close": g["close"].to_numpy(),
            }
        if len(GLOBAL_CACHE) > MAX_CACHE_ENTRIES:
            GLOBAL_CACHE.clear()
        GLOBAL_CACHE[path_str] = data
        return data
    except Exception:
        GLOBAL_CACHE[path_str] = None
        return None


class IncrementalATR:
    def __init__(self, period=14):
        self.period = period
        self._buf = deque(maxlen=period)
        self.atr = None
        self.prev_close = None
        self._n = 0

    def update(self, h, l, c):
        tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close)) if self.prev_close else h - l
        self._buf.append(tr)
        self._n += 1
        self.prev_close = c
        if self._n < self.period:
            self.atr = None
        elif self._n == self.period:
            self.atr = sum(self._buf) / self.period
        else:
            self.atr = (self.atr * (self.period - 1) + tr) / self.period
        return self.atr


class ParamStoch:
    def __init__(self, s1_k, s1_d, s4_k):
        self.s1 = IncrementalStochastic(s1_k, s1_d)
        self.s2 = IncrementalStochastic(14, 3)
        self.s3 = IncrementalStochastic(40, 4)
        self.s4 = IncrementalStochastic(s4_k, 10)

    def push(self, h, l, c):
        return {"s1d": self.s1.push(h, l, c), "s2d": self.s2.push(h, l, c),
                "s3d": self.s3.push(h, l, c), "s4d": self.s4.push(h, l, c)}


class TFTracker:
    def __init__(self, lb, p):
        self.lb = lb
        self.stoch = ParamStoch(p["s1_k"], p["s1_d"], p["s4_k"])
        self.div = DivergenceEngine()
        self.hist = []
        self.setup = False
        self.stype = ""
        self._armed_bullish_divergence = None
        self.prev_s1 = None
        self.s4_emb = 0
        self.atr = IncrementalATR(p["atr_period"])
        self.ema20 = IncrementalEMA(20)
        self.ema20_value = None
        self.p_f6_s4 = p["f6_s4_thresh"]
        self.p_f6_s1 = p["f6_s1_thresh"]
        self.use_divergence = p.get("use_divergence", True)
        self.require_ema20 = p.get("require_ema20", False)

    def reset_session_state(self):
        """Drop pending setups without discarding indicator warm-up values."""
        self.hist.clear()
        self.setup = False
        self.stype = ""
        self._armed_bullish_divergence = None
        self._fired = False
        # NOTE: self.s4_emb is intentionally preserved across the session roll so
        # the embedded-S4 count carries the prior session's warm-up (otherwise
        # early-session supers never satisfy the >=25 embedded gate).

    def push(self, c: Candle):
        self.hist.append(c)
        if len(self.hist) > 40:
            self.hist.pop(0)
        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
        atr_val = self.atr.update(c.high, c.low, c.close)
        self.ema20_value = self.ema20.update(c.close)
        ema_ok = (
            not self.require_ema20
            or (self.ema20_value is not None and c.close > self.ema20_value)
        )
        prev_s1 = self.prev_s1
        self.prev_s1 = s1
        # S1 turn up = current S1 %D rising vs previous bar (the only entry trigger
        # for the normal bullish F6 setup; no pinbar required).
        s1_turn_up = prev_s1 is not None and s1 is not None and s1 > prev_s1
        if s4 is not None:
            self.s4_emb = self.s4_emb + 1 if s4 <= 20 else 0
        emb = self.s4_emb >= 25
        self.div.update(c.close, s1, low_price=c.low, high_price=c.high)
        bull_div = self.div.has_bullish_trough_divergence()
        bullish_divergence_id = self.div.bullish_divergence_id()
        # Lag setup: S4 embedded high (>=80) and S1 touches the lower limit (<=20).
        is_flag = s4 is not None and s1 is not None and s4 >= self.p_f6_s4 and s1 <= self.p_f6_s1
        # Normal bullish setup: ALL of S1,S2,S3,S4 <= 20 AND S1 turning up.
        is_super = (
            all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
            and s1_turn_up
        )
        cond = (is_flag or is_super) and (
            not self.use_divergence
            or (
                bull_div
                and bullish_divergence_id != self._armed_bullish_divergence
            )
        )
        if cond:
            self.stype = "super" if is_super else "flag"
        triggered = False
        if cond and not self._fired and ema_ok:
            triggered = True
            self._fired = True
            if self.use_divergence:
                self._armed_bullish_divergence = bullish_divergence_id
        if not cond:
            self._fired = False
        is_rev = emb and self.stype == "super"
        return triggered, is_rev, self.stype, c.close, atr_val


class FlagNoDivScanner:
    def __init__(self, s1_k, s1_d, s4_k, f6_s4_thresh, f6_s1_thresh):
        self.s1 = IncrementalStochastic(s1_k, s1_d)
        self.s4 = IncrementalStochastic(s4_k, 10)
        self.f6_s4_thresh = f6_s4_thresh
        self.f6_s1_thresh = f6_s1_thresh
        self._fired = False

    def reset_session_state(self):
        """Allow a new session to emit its first qualifying F6 bar."""
        self._fired = False

    def push(self, h, l, c):
        s1v = self.s1.push(h, l, c)
        s4v = self.s4.push(h, l, c)
        if s1v is None or s4v is None:
            return False
        flag = s4v >= self.f6_s4_thresh and s1v <= self.f6_s1_thresh
        if flag and not self._fired:
            self._fired = True
            return True
        if not flag:
            self._fired = False
        return False


class MTFTracker:
    def __init__(self, p):
        self.trackers = {tf: TFTracker(spec[1], p) for tf, spec in TF_SPECS.items()}
        self.f6scans = {tf: FlagNoDivScanner(p["s1_k"], p["s1_d"], p["s4_k"],
                                             p["f6_s4_thresh"], p["f6_s1_thresh"]) for tf in TF_SPECS}
        self.bufs = {tf: [] for tf in TF_SPECS}
        self._last_minute = None
        self.reverse_regime_active = False
        self.require_ema20 = any(
            tracker.require_ema20 for tracker in self.trackers.values()
        )

    def _reset_timeframe_buffers_if_session_rolled(self, minute):
        if self._last_minute is not None and minute > 0 and minute < self._last_minute:
            self.bufs = {tf: [] for tf in TF_SPECS}
            for tf in TF_SPECS:
                self.trackers[tf].reset_session_state()
                self.f6scans[tf].reset_session_state()
            self.reverse_regime_active = False
        self._last_minute = minute

    def push_1m(self, c1m: Candle):
        out = []
        self._reset_timeframe_buffers_if_session_rolled(c1m.minute)
        for tf, spec in TF_SPECS.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf]) == spec[0]:
                buf = self.bufs[tf]
                self.bufs[tf] = []
                ctf = Candle(open=buf[0].open, high=max(x.high for x in buf),
                             low=min(x.low for x in buf), close=buf[-1].close,
                             minute=buf[-1].minute)
                trig, is_rev, stype, px, atr_val = self.trackers[tf].push(ctf)
                if trig:
                    out.append((tf, is_rev, stype, px, atr_val))
        self.reverse_regime_active = any(
            tracker.s4_emb >= 25
            for tracker in self.trackers.values()
        )
        return [
            (
                tf,
                is_reverse or (self.reverse_regime_active and signal_type == "super"),
                signal_type,
                entry,
                atr_value,
            )
            for tf, is_reverse, signal_type, entry, atr_value in out
        ]


def process_day(args):
    day, fpath, fprev, p = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not fpath:
        return []
    gc = cached_day(fpath)
    if not gc:
        return []
    fsym = next(iter(gc))
    mm = SYM_RE.match(fsym)
    if not mm:
        return []
    prefix = mm.group(1)
    sp0 = latest_spot(spot, 555) or latest_spot(spot, 560)
    if sp0 is None:
        return []
    atm0 = int(round(sp0 / 50) * 50)
    target_strikes = set(range(atm0 - 250, atm0 + 300, 50))

    def filtered(data):
        return {sym: g for sym, g in data.items()
                if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}

    gu = filtered(gc)
    gp = {}
    if fprev:
        dp = cached_day(fprev)
        if dp:
            gp = filtered(dp)

    trk = {}
    for sym, g in gp.items():
        trk[sym] = MTFTracker(p)
        for i in range(len(g["min"])):
            trk[sym].push_1m(Candle(open=g["open"][i], high=g["high"][i],
                                    low=g["low"][i], close=g["close"][i], minute=g["min"][i]))

    pmtrig = {}
    slices = {}
    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = MTFTracker(p)
        t = trk[sym]
        slices[sym] = g
        mm2 = SYM_RE.match(sym)
        if not mm2:
            continue
        sv, side = int(mm2.group(2)), mm2.group(3)
        for i in range(len(g["min"])):
            m = g["min"][i]
            for (tf, is_rev, stype, px, atr_val) in t.push_1m(
                    Candle(open=g["open"][i], high=g["high"][i],
                           low=g["low"][i], close=g["close"][i], minute=m)):
                pmtrig.setdefault(m, []).append(
                    (side, sv, sym, px, is_rev, tf, TF_SPECS[tf][2], TF_SPECS[tf][3], atr_val))

    daily_loss_pts = DAILY_LOSS_RS / LOT_SIZE
    consec_loss = p["consec_loss"]
    sl_mult, tp_mult = p["atr_sl_mult"], p["atr_tp_mult"]

    def bslice(sl, m):
        idx = np.searchsorted(sl["min"], m)
        if idx < len(sl["min"]) and sl["min"][idx] == m:
            return sl["open"][idx], sl["high"][idx], sl["low"][idx], sl["close"][idx]
        return None

    def ainfo(side, m):
        spx = latest_spot(spot, m)
        if spx is None:
            return None
        atm = int(round(spx / 50) * 50)
        stk = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        sym = f"{prefix}{stk}{side}"
        sl = slices.get(sym)
        return (sym, sl, stk) if sl is not None else None

    trades = []
    pos = None
    dpnl = 0.0
    closs = 0
    shut = False
    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c)
                pos["duration_min"] += 1
                if dpnl * LOT_SIZE + (c - pos["entry"]) * LOT_SIZE <= DAILY_LOSS_RS:
                    pts = round(c - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                                   "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                                   "exit": c, "pts": pts, "rs": round(pts * LOT_SIZE),
                                   "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                                   "reason": "SHUTDOWN_LOSS", "duration_min": pos["duration_min"], "tf": pos["tf"]})
                    dpnl += pts
                    pos = None
                    shut = True
                    continue
                ex, rsn = None, ""
                has_tgt = pos.get("tgt") is not None
                if has_tgt and h >= pos["tgt"] and l <= pos["sl"]:
                    ex, rsn = pos["sl"], "SL"
                elif has_tgt and h >= pos["tgt"]:
                    ex, rsn = pos["tgt"], "TP"
                elif l <= pos["sl"]:
                    ex, rsn = pos["sl"], "SL"
                if ex is None:
                    t1 = trk.get(pos["symbol"])
                    if t1:
                        t1m = t1.trackers["1m"]
                        t1m.div.update(c, t1m.prev_s1, low_price=l, high_price=h)
                        if t1m.div.has_bearish_peak_divergence():
                            ex, rsn = c, "BEARISH_PEAK_REVERSAL"
                if ex is not None:
                    pts = round(ex - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                                   "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                                   "exit": ex, "pts": pts, "rs": round(pts * LOT_SIZE), "reason": rsn,
                                   "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                                   "duration_min": pos["duration_min"], "tf": pos["tf"]})
                    dpnl += pts
                    closs = closs + 1 if pts <= 0 else 0
                    if closs >= consec_loss or dpnl <= daily_loss_pts:
                        shut = True
                    pos = None
        if minute >= SESSION_END and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                           "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                           "exit": pos["last_px"], "pts": pts, "rs": round(pts * LOT_SIZE),
                           "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                           "reason": "EOD", "duration_min": pos["duration_min"], "tf": pos["tf"]})
            dpnl += pts
            pos = None
            break
        if pos is not None or shut or minute >= SESSION_END:
            continue

        for (sig_side, sig_stk, sig_sym, c_px, is_rev, tf, sl_pts, tp_pts, atr_val) in pmtrig.get(minute, []):
            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                if is_rev:
                    as2 = "PE" if sig_side == "CE" else "CE"
                    ai2 = ainfo(as2, minute)
                    if ai2 is None:
                        continue
                    asym, asl, _ = ai2
                else:
                    as2 = sig_side
                    asym = sig_sym
                    asl = ai[1]
                bar = bslice(asl, minute)
                if bar:
                    ep = float(bar[3])
                    if atr_val and atr_val > 0.5:
                        sl_use = atr_val * sl_mult
                        tp_use = atr_val * tp_mult
                    else:
                        sl_use = sl_pts
                        tp_use = tp_pts
                    pos = {"side": as2, "symbol": asym, "slice": asl, "entry": ep,
                           "sl": ep - sl_use, "tgt": ep + tp_use, "sl_pts": round(sl_use, 2), "tp_pts": round(tp_use, 2),
                           "entry_min": minute, "last_px": ep, "duration_min": 0, "tf": tf}
                    break
    return trades


def run_days(pool, params, days, files, spot_all):
    tasks = [(day, str(files[day]), str(files[days[i - 1]]) if i > 0 else "", params)
             for i, day in enumerate(days)]
    all_trades = []
    for res in pool.map(process_day, tasks):
        all_trades.extend(res)
    return all_trades


def stats_for(trades, start_year=None, end_year=None):
    if start_year is not None or end_year is not None:
        trades = [t for t in trades
                  if (start_year is None or int(t["date"][:4]) >= start_year)
                  and (end_year is None or int(t["date"][:4]) <= end_year)]
    return summarize(trades), trades


def composite_score(st):
    if st["trades"] == 0 or st["pf"] == 0:
        return -1e9
    return st["rs"] * (st["wr"] / 100.0) * st["pf"]


RESULTS_CSV = "optuna_results.csv"
LEADERBOARD = []


def csv_append(row):
    write_header = not Path(RESULTS_CSV).exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def print_leaderboard():
    if not LEADERBOARD:
        return
    print("\n" + "-" * 100)
    print("LIVE LEADERBOARD (top 5 by composite score)")
    print(f"{'#':>3} | {'SCORE':>12} | {'RS':>12} | {'WR%':>6} | {'PF':>5} | {'TR':>6} | params")
    for i, e in enumerate(LEADERBOARD[:5], 1):
        st, p = e["st"], e["params"]
        print(f"{i:3d} | {e['score']:12.0f} | {st['rs']:12,d} | {st['wr']:6.1f} | "
              f"{st['pf']:5.2f} | {st['trades']:6,d} | "
              f"s1={p['s1_k']} s4={p['s4_k']} atr={p['atr_period']} sl={p['atr_sl_mult']} "
              f"tp={p['atr_tp_mult']} f6s4={p['f6_s4_thresh']} f6s1={p['f6_s1_thresh']} cl={p['consec_loss']}")
    print("-" * 100, flush=True)


def objective(trial, pool, days_all, days_2020, days_rest, files, spot_all):
    params = {}
    for name, values in SEARCH_SPACE.items():
        params[name] = trial.suggest_categorical(name, values)
    params["s1_d"] = 3

    t0 = time.time()
    trades_y1 = run_days(pool, params, days_2020, files, spot_all)
    st_y1, _ = stats_for(trades_y1)
    trial.report(st_y1["rs"], step=1)
    if trial.should_prune():
        elapsed = time.time() - t0
        row = {"trial": trial.number, "score": -1e9, "s1_k": params["s1_k"],
               "s4_k": params["s4_k"], "atr_period": params["atr_period"],
               "atr_sl_mult": params["atr_sl_mult"], "atr_tp_mult": params["atr_tp_mult"],
               "f6_s4_thresh": params["f6_s4_thresh"], "f6_s1_thresh": params["f6_s1_thresh"],
               "consec_loss": params["consec_loss"], "trades": st_y1["trades"],
               "wr": round(st_y1["wr"], 2), "net_rs": st_y1["rs"], "pf": round(st_y1["pf"], 4),
               "year1_rs": st_y1["rs"], "elapsed_s": round(elapsed, 1), "pruned": 1}
        csv_append(row)
        print(f"Trial {trial.number:3d} | PRUNED (year1 rs={st_y1['rs']:+10,d}) | {elapsed:5.0f}s", flush=True)
        raise optuna.TrialPruned()

    trades_rest = run_days(pool, params, days_rest, files, spot_all)
    all_trades = trades_y1 + trades_rest
    st, _ = stats_for(all_trades)
    score = composite_score(st)
    elapsed = time.time() - t0

    row = {"trial": trial.number, "score": round(score), "s1_k": params["s1_k"],
           "s4_k": params["s4_k"], "atr_period": params["atr_period"],
           "atr_sl_mult": params["atr_sl_mult"], "atr_tp_mult": params["atr_tp_mult"],
           "f6_s4_thresh": params["f6_s4_thresh"], "f6_s1_thresh": params["f6_s1_thresh"],
           "consec_loss": params["consec_loss"], "trades": st["trades"],
           "wr": round(st["wr"], 2), "net_rs": st["rs"], "pf": round(st["pf"], 4),
           "year1_rs": st_y1["rs"], "elapsed_s": round(elapsed, 1), "pruned": 0}
    csv_append(row)
    LEADERBOARD.append({"score": score, "st": st, "params": params})
    LEADERBOARD.sort(key=lambda x: x["score"], reverse=True)
    LEADERBOARD[:] = LEADERBOARD[:20]
    print(f"Trial {trial.number:3d} | score={score:12.0f} | rs={st['rs']:+12,d} | wr={st['wr']:5.1f}% | "
          f"pf={st['pf']:5.2f} | trades={st['trades']:6,d} | {elapsed:5.0f}s", flush=True)
    print_leaderboard()
    return score


def run_optuna(n_trials, days_all, days_2020, days_rest, files, spot_all):
    if optuna is None:
        sys.exit("optuna not installed — run: pip install optuna")

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0, n_min_trials=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t_start = time.time()
    with Pool(processes=WORKERS, initializer=init_worker_local,
              initargs=(spot_all,)) as pool:
        study.optimize(lambda trial: objective(trial, pool, days_all, days_2020,
                                               days_rest, files, spot_all),
                       n_trials=n_trials)

    total = time.time() - t_start
    print(f"\n=== PHASE 1 COMPLETE: {n_trials} trials in {total/60:.1f} min ===")
    print(f"Results: {RESULTS_CSV}")


def smoke_test(spot_all, files, days):
    days5 = days[:5]
    print(f"=== SMOKE TEST — {len(days5)} DAYS ({days5[0]}..{days5[-1]}) — CHAMPION PARAMS ===")
    with Pool(processes=WORKERS, initializer=init_worker_local,
              initargs=(spot_all,)) as pool:
        trades = run_days(pool, CHAMPION, days5, files, spot_all)
    st, _ = stats_for(trades)
    print(f"Trades: {st['trades']} | WR: {st['wr']:.1f}% | Net Rs: {st['rs']:+,d} | PF: {st['pf']:.2f}")
    if st["trades"] < 15 or st["trades"] > 40:
        print(f"WARNING: expected 15-40 trades on 5 days, got {st['trades']}")
    print("SMOKE TEST OK" if 15 <= st["trades"] <= 40 else "SMOKE TEST SUSPICIOUS")


def verify_champion(spot_all, files, days):
    print(f"=== VERIFY — champion params on full {len(days)} days (expect 7,843 / 48.0% / +1,030,642 / 1.45) ===")
    with Pool(processes=WORKERS, initializer=init_worker_local,
              initargs=(spot_all,)) as pool:
        trades = run_days(pool, CHAMPION, days, files, spot_all)
    st, _ = stats_for(trades)
    print(f"GOT   — Trades: {st['trades']} | WR: {st['wr']:.1f}% | Net Rs: {st['rs']:+,d} | PF: {st['pf']:.2f}")
    exp = CHAMPION_EXPECTED
    ok = (abs(st["trades"] - exp["trades"]) <= 5 and abs(st["wr"] - exp["wr"]) <= 0.3
          and abs(st["rs"] - exp["rs"]) <= 20000)
    print("VERIFY PASS — engine reproduces champion" if ok else "VERIFY FAIL — engine diverges from champion!")
    print_yearly_breakdown(trades)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2022-12-31")
    args = ap.parse_args()

    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    if args.smoke:
        smoke_test(spot_all, files, days)
        return
    if args.verify:
        verify_champion(spot_all, files, days)
        return

    files3 = option_files(args.start, args.end)
    days_all = sorted(set(files3.keys()) & set(spot_all.keys()))
    days_2020 = [d for d in days_all if d.startswith("2020")]
    days_rest = [d for d in days_all if not d.startswith("2020")]
    print(f"Phase 1 window: {args.start}..{args.end} = {len(days_all)} days (2020 subset: {len(days_2020)})")
    print(f"Search space: {len(SEARCH_SPACE)} axes, "
          f"{np.prod([len(v) for v in SEARCH_SPACE.values()]):,} total combinations")
    print(f"Trials: {args.trials} | sampler: TPE multivariate | pruner: MedianPruner | workers: {WORKERS}")
    if Path(RESULTS_CSV).exists():
        print(f"NOTE: {RESULTS_CSV} exists — appending to it")
    run_optuna(args.trials, days_all, days_2020, days_rest, files3, spot_all)


if __name__ == "__main__":
    main()
