"""Combinatorial Optimization — Quad Futures-Signal -> ITM2 Options Strategy.

Signal source: NIFTY FUTURES 1-minute chart (aggregated concurrently to
1m/2m/3m/5m like the champion engine). Execution: 2nd-ITM weekly option
(CE = ATM-100, PE = ATM+100) on its own 1-minute bars. One position at a time.

FIXED signal rules (do not change, no VWAP / no divergence confirmation):
  bear_flag        S4 <= 20.5 embedded N bars, S1 was neutral -> crosses >= 79.5  -> buy PE
  bull_flag        S4 >= 79.5 embedded N bars, S1 was neutral -> crosses <= 20.5  -> buy CE
  supersignal_bear all S1..S4 >= 79.5 AND S1 turns down                            -> buy PE
  supersignal_bull all S1..S4 <= 20.5 AND S1 turns up                              -> buy CE
Each trigger re-arms only after S1 returns to neutral / leaves the extreme.

Rest copied from the champion (grid_optimize_f6_atr.py):
  - S1(9,3) S2(14,3) S3(40,4) S4(60,10) stochastic params (fixed)
  - ATR-based SL/TP on the option (ATR(period) x sl_mult / tp_mult), per-TF
    fallback SL/TP when ATR is not available
  - Daily max loss = 30 pts (user override, replaces Rs 2,000), profit UNLIMITED
  - Consecutive-loss shutdown = 6

Search space (4 axes, 144 combos):
  atr_period   [10, 14, 20]
  atr_sl_mult  [1.5, 2.0, 2.5, 3.0]
  atr_tp_mult  [3.0, 4.0, 5.0, 6.0]
  embed        [10, 14, 20]   (S4-embed bars before flag)

Speed: ONE persistent Pool; worker-local caches for futures day files and
option day files; pruning = 2020 first, MedianPruner.

Usage:
  python opt_futures_quad.py --smoke
  python opt_futures_quad.py --trials 200
"""

import csv
import re
import sys
import time
from collections import deque
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
except ImportError:
    optuna = None

AMMU = Path(r"C:\Websites\ammu")
FUT_DIR = AMMU / "nifty_fut"
OPTS_DIR = AMMU / "nifty_options"
SPOT_CSV = AMMU / "index" / "NIFTY 50_minute.csv"
LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_LOSS_PTS = -30.0            # user override: daily max loss 30 pts (~Rs 1,950)
DAILY_PROFIT_PTS = float("inf")   # profit unlimited
LIMIT, LOW_ZONE = 79.5, 20.5
CONSEC_LOSS = 6

TF_SPECS = {
    "1m": (1, 10, 6.0, 30.0),
    "2m": (2, 5, 10.0, 15.0),
    "3m": (3, 4, 8.0, 25.0),
    "5m": (5, 3, 10.0, 35.0),
}

CHAMPION = {"atr_period": 14, "atr_sl_mult": 2.0, "atr_tp_mult": 4.0, "embed": 14}

SEARCH_SPACE = {
    "atr_period": [10, 14, 20],
    "atr_sl_mult": [1.5, 2.0, 2.5, 3.0],
    "atr_tp_mult": [3.0, 4.0, 5.0, 6.0],
    "embed": [10, 14, 20],
}

MAX_CACHE_ENTRIES = 1600
WORKERS = 8
GLOBAL_SPOT = {}
GLOBAL_CACHE = {}


def init_worker_local(spot_dict):
    global GLOBAL_SPOT, GLOBAL_CACHE
    GLOBAL_SPOT = spot_dict
    GLOBAL_CACHE = {}


def cached_futures(path_str):
    c = GLOBAL_CACHE.get(path_str)
    if c is not None:
        return c
    try:
        df = pd.read_csv(path_str, usecols=["time", "open", "high", "low", "close"])
        if df.empty:
            GLOBAL_CACHE[path_str] = None
            return None
        out = {
            "min": np.array([int(t.split(":")[0]) * 60 + int(t.split(":")[1]) for t in df["time"]]),
            "open": df["open"].to_numpy(),
            "high": df["high"].to_numpy(),
            "low": df["low"].to_numpy(),
            "close": df["close"].to_numpy(),
        }
        if len(GLOBAL_CACHE) > MAX_CACHE_ENTRIES:
            GLOBAL_CACHE.clear()
        GLOBAL_CACHE[path_str] = out
        return out
    except Exception:
        GLOBAL_CACHE[path_str] = None
        return None


def cached_option(path_str):
    c = GLOBAL_CACHE.get("opt:" + path_str)
    if c is not None:
        return c
    try:
        df = pd.read_csv(path_str, usecols=["symbol", "time", "open", "high", "low", "close"])
        g = df.groupby("symbol", sort=False).indices
        prefixes = {}
        for sym in g:
            m = re.match(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$", sym)
            if m:
                prefixes[m.group(1)] = prefixes.get(m.group(1), 0) + 1
        prefix = max(prefixes, key=prefixes.get) if prefixes else None
        rec = (df, g, prefix)
        if len(GLOBAL_CACHE) > MAX_CACHE_ENTRIES:
            GLOBAL_CACHE.clear()
        GLOBAL_CACHE["opt:" + path_str] = rec
        return rec
    except Exception:
        GLOBAL_CACHE["opt:" + path_str] = None
        return None


def make_slice(df, g, sym):
    idx = g.get(sym)
    if idx is None:
        return None
    rows = df.iloc[idx]
    return {
        "times": np.array([int(t.split(":")[0]) * 60 + int(t.split(":")[1]) for t in rows["time"]]),
        "open": rows["open"].to_numpy(),
        "high": rows["high"].to_numpy(),
        "low": rows["low"].to_numpy(),
        "close": rows["close"].to_numpy(),
        "ptr": 0,
    }


def bar_at(sl, minute):
    i = sl.get("ptr", 0)
    times = sl["times"]
    while i < len(times) and times[i] < minute:
        i += 1
    sl["ptr"] = i
    if i >= len(times) or times[i] != minute:
        return None
    return (sl["open"][i], sl["high"][i], sl["low"][i], sl["close"][i])


def option_atr(sl, upto_minute, period):
    """Wilder ATR of the option slice up to (and including) minute."""
    idx = int(np.searchsorted(sl["times"], upto_minute, side="right"))
    if idx < period:
        return None
    h = sl["high"][:idx]
    l = sl["low"][:idx]
    c = sl["close"][:idx]
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = float(np.mean(tr[:period]))
    for i in range(period, idx):
        atr = (atr * (period - 1) + tr[i]) / period
    return atr


class QuadStoch:
    """Mirror of bt_quad_5y.StochState: 4-band stochastic with 8% proxy-tick
    clipping, %D = SMA of valid %K values (reported as soon as k bars exist)."""

    def __init__(self):
        self.specs = [(9, 3), (14, 3), (40, 4), (60, 10)]
        self.hl = [deque(maxlen=k) for k, d in self.specs]
        self.cl = [deque(maxlen=k) for k, d in self.specs]
        self.k = [deque(maxlen=d) for k, d in self.specs]

    @staticmethod
    def _clipped_extremes(hk, lk):
        m1 = m2 = -1e18
        for v in hk:
            if v > m1:
                m2, m1 = m1, v
            elif v > m2:
                m2 = v
        n1 = n2 = 1e18
        for v in lk:
            if v < n1:
                n2, n1 = n1, v
            elif v < n2:
                n2 = v
        hh = m2 if m1 > m2 * 1.08 else m1
        ll = n2 if n1 < n2 * 0.92 else n1
        return hh, ll

    def push(self, h, l, c):
        out = {}
        for i, (k, d) in enumerate(self.specs):
            self.hl[i].append((h, l))
            self.cl[i].append(c)
            if len(self.cl[i]) >= k:
                hh, ll = self._clipped_extremes(
                    [x[0] for x in self.hl[i]], [x[1] for x in self.hl[i]])
                kk = 50.0 if hh == ll else 100.0 * (c - ll) / (hh - ll)
                self.k[i].append(kk)
            out[f"s{i+1}d"] = sum(self.k[i]) / len(self.k[i]) if self.k[i] else None
        return out


class FuturesQuadTriggers:
    """Stateful mirrored super/flag detectors, one trigger per oscillation."""

    def __init__(self, embed):
        self.embed_n = embed
        self.prev_s1 = None
        self.low_embed = 0
        self.high_embed = 0
        self.bear_flag_armed = True
        self.bull_flag_armed = True
        self.super_bear_armed = True
        self.super_bull_armed = True

    def update_embed(self, s4):
        if s4 is None:
            self.low_embed = 0
            self.high_embed = 0
            return
        self.low_embed = self.low_embed + 1 if s4 <= LOW_ZONE else 0
        self.high_embed = self.high_embed + 1 if s4 >= LIMIT else 0

    def evaluate(self, values, signal_mode="all"):
        s1 = values.get("s1d")
        s2 = values.get("s2d")
        s3 = values.get("s3d")
        s4 = values.get("s4d")
        if any(v is None for v in (s1, s2, s3, s4)):
            self.prev_s1 = s1
            return None

        prev = self.prev_s1
        neutral_prev = prev is not None and LOW_ZONE < prev < LIMIT
        all_high = all(v >= LIMIT for v in (s1, s2, s3, s4))
        all_low = all(v <= LOW_ZONE for v in (s1, s2, s3, s4))

        if LOW_ZONE < s1 < LIMIT:
            self.bear_flag_armed = True
            self.bull_flag_armed = True
        if s1 < LIMIT:
            self.super_bear_armed = True
        if s1 > LOW_ZONE:
            self.super_bull_armed = True

        signal = None
        if self.low_embed >= self.embed_n and neutral_prev and s1 >= LIMIT and self.bear_flag_armed:
            self.bear_flag_armed = False
            signal = "bear_flag"
        elif self.high_embed >= self.embed_n and neutral_prev and s1 <= LOW_ZONE and self.bull_flag_armed:
            self.bull_flag_armed = False
            signal = "bull_flag"
        elif all_high and prev is not None and s1 < prev and self.super_bear_armed:
            self.super_bear_armed = False
            signal = "supersignal_bear"
        elif all_low and prev is not None and s1 > prev and self.super_bull_armed:
            self.super_bull_armed = False
            signal = "supersignal_bull"

        self.prev_s1 = s1
        if signal is None:
            return None
        if signal_mode == "flags" and not signal.endswith("flag"):
            return None
        if signal_mode == "super" and not signal.startswith("supersignal"):
            return None
        return signal


class MTFSignalFeed:
    """Feeds futures 1m bars into per-TF quad detectors (1m/2m/3m/5m concurrent)."""

    def __init__(self, p):
        self.trackers = {tf: FuturesQuadTriggers(p["embed"]) for tf in TF_SPECS}
        self.stochs = {tf: QuadStoch() for tf in TF_SPECS}
        self.bufs = {tf: [] for tf in TF_SPECS}
        self.last_values = {}

    def push_1m(self, h, l, c, minute):
        """Advance every TF buffer/stochastic unconditionally (matching baseline:
        embed/S4 state moves every minute, even while a position is open).
        Stores the latest values per TF for evaluate_now()."""
        for tf, spec in TF_SPECS.items():
            self.bufs[tf].append((h, l, c, minute))
            if len(self.bufs[tf]) == spec[0]:
                buf = self.bufs[tf]
                self.bufs[tf] = []
                bh = max(x[0] for x in buf)
                bl = min(x[1] for x in buf)
                bc = buf[-1][2]
                bm = buf[-1][3]
                self.last_values[tf] = self.stochs[tf].push(bh, bl, bc)
                self.trackers[tf].update_embed(self.last_values[tf].get("s4d"))

    def evaluate_now(self):
        """Evaluate each TF's signal detector using the latest pushed values."""
        out = []
        for tf in TF_SPECS:
            values = self.last_values.get(tf)
            if values is None:
                continue
            sig = self.trackers[tf].evaluate(values, signal_mode="all")
            if sig is not None:
                out.append((tf, sig))
        return out


def futures_day_files(start, end):
    files = sorted(FUT_DIR.rglob("*.csv"))
    result = {}
    for path in files:
        parts = path.stem.split("_")
        day = f"{parts[4]}-{parts[3]}-{parts[2]}"
        if start <= day <= end:
            result[day] = path
    return result


def option_day_files(start, end):
    files = sorted(
        OPTS_DIR.rglob("*.csv"),
        key=lambda p: (int(p.parent.parent.name), int(p.parent.name),
                       int(p.stem.split("_")[2]), int(p.stem.split("_")[3])),
    )
    result = {}
    for path in files:
        parts = path.stem.split("_")
        day = f"{parts[4]}-{parts[3]}-{parts[2]}"
        if start <= day <= end:
            result[day] = path
    return result


def latest_value(series, minute):
    idx = np.searchsorted(series["min"], minute, side="right") - 1
    if idx < 0:
        return None
    return float(series["close"][idx])


def process_day(args):
    day, fut_path, opt_path, p = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not fut_path or not opt_path:
        return []
    fut = cached_futures(str(fut_path))
    if fut is None:
        return []
    rec = cached_option(str(opt_path))
    if rec is None:
        return []
    df, groups, prefix = rec
    if prefix is None:
        return []

    feed = MTFSignalFeed(p)
    trades = []
    pos = None
    dpnl = 0.0
    closs = 0
    shut = False
    slices = {}
    atr_period = p["atr_period"]
    sl_mult, tp_mult = p["atr_sl_mult"], p["atr_tp_mult"]

    def get_slice(side, minute):
        spot_px = latest_value(spot, minute)
        if spot_px is None:
            return None
        atm = int(round(spot_px / 50) * 50)
        strike = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        symbol = f"{prefix}{strike}{side}"
        sl = slices.get(symbol)
        if sl is None:
            sl = make_slice(df, groups, symbol)
            if sl is not None:
                slices[symbol] = sl
        return (symbol, sl) if sl is not None else None

    for i in range(len(fut["min"])):
        minute = int(fut["min"][i])
        if minute < SESSION_START or minute > DAY_LAST:
            continue
        h, l, c = float(fut["high"][i]), float(fut["low"][i]), float(fut["close"][i])

        # Match the live Quad behavior: stochastic/S4-embedding state is advanced
        # EVERY minute (even while a position is open); only the signal detector
        # (position entry) is gated below.
        feed.push_1m(h, l, c, minute)

        if pos is not None:
            active = get_slice(pos["side"], minute)
            if active is not None:
                symbol, sl = active
                bar = bar_at(sl, minute)
                if bar is not None:
                    pos["last_px"] = float(bar[3])
                    if dpnl * LOT_SIZE + (bar[3] - pos["entry"]) * LOT_SIZE <= DAILY_LOSS_PTS * LOT_SIZE:
                        pts = round(bar[3] - pos["entry"], 2)
                        trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                                       "side": pos["side"], "signal": pos["signal"], "symbol": pos["symbol"],
                                       "entry": pos["entry"], "exit": bar[3], "pts": pts,
                                       "rs": round(pts * LOT_SIZE), "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                                       "reason": "SHUTDOWN", "tf": pos["tf"]})
                        dpnl += pts
                        pos = None
                        shut = True
                        continue
                    high, low = float(bar[1]), float(bar[2])
                    ex, rsn = None, ""
                    if high >= pos["tgt"] and low <= pos["sl"]:
                        ex, rsn = pos["sl"], "SL"
                    elif high >= pos["tgt"]:
                        ex, rsn = pos["tgt"], "TP"
                    elif low <= pos["sl"]:
                        ex, rsn = pos["sl"], "SL"
                    if ex is not None:
                        pts = round(ex - pos["entry"], 2)
                        trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                                       "side": pos["side"], "signal": pos["signal"], "symbol": pos["symbol"],
                                       "entry": pos["entry"], "exit": ex, "pts": pts,
                                       "rs": round(pts * LOT_SIZE), "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                                       "reason": rsn, "tf": pos["tf"]})
                        dpnl += pts
                        closs = closs + 1 if pts <= 0 else 0
                        if closs >= CONSEC_LOSS or dpnl <= DAILY_LOSS_PTS:
                            shut = True
                        pos = None
            if minute >= SESSION_END and pos is not None:
                pts = round(pos["last_px"] - pos["entry"], 2)
                trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                               "side": pos["side"], "signal": pos["signal"], "symbol": pos["symbol"],
                               "entry": pos["entry"], "exit": pos["last_px"], "pts": pts,
                               "rs": round(pts * LOT_SIZE), "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                               "reason": "EOD", "tf": pos["tf"]})
                dpnl += pts
                pos = None
                break

        if pos is not None or shut or minute >= SESSION_END:
            # Signal detector state is not advanced here (matches baseline: the
            # stochastic/S4-embedding state above is updated every minute, but
            # no new position is opened while one is open / after shutdown / EOD.
            continue

        fired = feed.evaluate_now()
        if not fired:
            continue
        tf, signal = fired[0]
        side = "PE" if signal in ("bear_flag", "supersignal_bear") else "CE"
        active = get_slice(side, minute)
        if active is None:
            continue
        symbol, sl = active
        bar = bar_at(sl, minute)
        if bar is None:
            continue
        entry = float(bar[3])
        if entry <= 0:
            continue
        atr_val = option_atr(sl, minute, atr_period)
        if atr_val and atr_val > 0.5:
            sl_use = atr_val * sl_mult
            tp_use = atr_val * tp_mult
        else:
            sl_use, tp_use = TF_SPECS[tf][2], TF_SPECS[tf][3]
        pos = {"side": side, "signal": signal, "symbol": symbol, "slice": sl,
               "entry": entry, "sl": entry - sl_use, "tgt": entry + tp_use,
               "sl_pts": round(sl_use, 2), "tp_pts": round(tp_use, 2),
               "entry_min": minute, "last_px": entry, "tf": tf}

    if pos is not None:
        pts = round(pos["last_px"] - pos["entry"], 2)
        trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": SESSION_END,
                       "side": pos["side"], "signal": pos["signal"], "symbol": pos["symbol"],
                       "entry": pos["entry"], "exit": pos["last_px"], "pts": pts,
                       "rs": round(pts * LOT_SIZE), "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                       "reason": "EOD", "tf": pos["tf"]})
    return trades


def run_days(pool, params, days, fut_files, opt_files, spot_all):
    tasks = [(day, str(fut_files[day]), str(opt_files[day]), params)
             for day in days]
    all_trades = []
    for res in pool.map(process_day, tasks):
        all_trades.extend(res)
    return all_trades


def summarize(trades):
    wins = [t for t in trades if t["pts"] > 0]
    losses = [t for t in trades if t["pts"] <= 0]
    gross_w = sum(t["pts"] for t in wins)
    gross_l = abs(sum(t["pts"] for t in losses))
    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100 if trades else 0.0,
        "pts": round(sum(t["pts"] for t in trades), 2),
        "rs": round(sum(t["pts"] for t in trades) * LOT_SIZE),
        "pf": gross_w / gross_l if gross_l else float("inf"),
    }


def composite_score(st):
    if st["trades"] == 0 or st["pf"] == 0:
        return -1e9
    return st["pts"] * (st["wr"] / 100.0) * st["pf"]


def full_stats(trades):
    """Max drawdown, avg trades/day, avg SL/TP + win/loss distribution."""
    import numpy as np
    if not trades:
        return None
    pts = np.array([t["pts"] for t in trades], dtype=float)
    equity = np.cumsum(pts)
    peak = np.maximum.accumulate(equity)
    dd = float((equity - peak).min())
    days = {}
    for t in trades:
        days[t["date"]] = days.get(t["date"], 0.0) + t["pts"]
    wins = pts > 0
    sls = np.array([t.get("sl_pts", 0.0) for t in trades], dtype=float)
    tps = np.array([t.get("tp_pts", 0.0) for t in trades], dtype=float)
    return {
        "max_dd_pts": round(dd, 1),
        "max_dd_rs": round(dd * LOT_SIZE),
        "avg_trades_per_day": round(len(trades) / max(len(days), 1), 2),
        "avg_sl_pts": round(float(sls.mean()), 2),
        "avg_tp_pts": round(float(tps.mean()), 2),
        "avg_win_pts": round(float(pts[wins].mean()), 2) if wins.any() else 0.0,
        "avg_loss_pts": round(float(pts[~wins].mean()), 2) if (~wins).any() else 0.0,
        "max_win": round(float(pts.max()), 2),
        "max_loss": round(float(pts.min()), 2),
    }


RESULTS_CSV = "optuna_futures_results.csv"
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
    print(f"{'#':>3} | {'SCORE':>12} | {'PTS':>10} | {'WR%':>6} | {'PF':>5} | {'TR':>6} | params")
    for i, e in enumerate(LEADERBOARD[:5], 1):
        st, p = e["st"], e["params"]
        print(f"{i:3d} | {e['score']:12.0f} | {st['pts']:10,.0f} | {st['wr']:6.1f} | "
              f"{st['pf']:5.2f} | {st['trades']:6,d} | "
              f"atr={p['atr_period']} sl={p['atr_sl_mult']} tp={p['atr_tp_mult']} embed={p['embed']}")
    print("-" * 100, flush=True)


def objective(trial, pool, days_all, days_2020, days_rest, fut_files, opt_files, spot_all):
    params = {}
    for name, values in SEARCH_SPACE.items():
        params[name] = trial.suggest_categorical(name, values)

    t0 = time.time()
    trades_y1 = run_days(pool, params, days_2020, fut_files, opt_files, spot_all)
    st_y1 = summarize(trades_y1)
    trial.report(st_y1["pts"], step=1)
    if trial.should_prune():
        elapsed = time.time() - t0
        row = {"trial": trial.number, "score": -1e9,
               "atr_period": params["atr_period"], "atr_sl_mult": params["atr_sl_mult"],
               "atr_tp_mult": params["atr_tp_mult"], "embed": params["embed"],
               "trades": st_y1["trades"], "wr": round(st_y1["wr"], 2),
               "net_pts": st_y1["pts"], "net_rs": st_y1["rs"], "pf": round(st_y1["pf"], 4),
               "year1_pts": st_y1["pts"], "elapsed_s": round(elapsed, 1), "pruned": 1}
        csv_append(row)
        print(f"Trial {trial.number:3d} | PRUNED (year1 pts={st_y1['pts']:+10,.0f}) | {elapsed:5.0f}s", flush=True)
        raise optuna.TrialPruned()

    trades_rest = run_days(pool, params, days_rest, fut_files, opt_files, spot_all)
    all_trades = trades_y1 + trades_rest
    st = summarize(all_trades)
    score = composite_score(st)
    elapsed = time.time() - t0

    row = {"trial": trial.number, "score": round(score),
           "atr_period": params["atr_period"], "atr_sl_mult": params["atr_sl_mult"],
           "atr_tp_mult": params["atr_tp_mult"], "embed": params["embed"],
           "trades": st["trades"], "wr": round(st["wr"], 2),
           "net_pts": st["pts"], "net_rs": st["rs"], "pf": round(st["pf"], 4),
           "year1_pts": st_y1["pts"], "elapsed_s": round(elapsed, 1), "pruned": 0}
    csv_append(row)
    LEADERBOARD.append({"score": score, "st": st, "params": params})
    LEADERBOARD.sort(key=lambda x: x["score"], reverse=True)
    LEADERBOARD[:] = LEADERBOARD[:20]
    print(f"Trial {trial.number:3d} | score={score:12.0f} | pts={st['pts']:+10,.0f} | "
          f"wr={st['wr']:5.1f}% | pf={st['pf']:5.2f} | trades={st['trades']:6,d} | {elapsed:5.0f}s", flush=True)
    print_leaderboard()
    return score


def run_optuna(n_trials, days_all, days_2020, days_rest, fut_files, opt_files, spot_all):
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
                                               days_rest, fut_files, opt_files, spot_all),
                       n_trials=n_trials)
        best = study.best_trial
        bp = {k: best.params[k] for k in SEARCH_SPACE}
        trades_champ = run_days(pool, CHAMPION, days_all, fut_files, opt_files, spot_all)
        trades_best = run_days(pool, bp, days_all, fut_files, opt_files, spot_all)
    total = time.time() - t_start
    print(f"\n=== PHASE 1 COMPLETE: {n_trials} trials in {total/60:.1f} min ===")
    print(f"Results: {RESULTS_CSV}")

    print("\n=== CHAMPION (full window) ===")
    sc = summarize(trades_champ)
    fc = full_stats(trades_champ)
    print(f"Trades: {sc['trades']} | WR: {sc['wr']:.1f}% | P&L: {sc['pts']:+,.1f} pts "
          f"({sc['rs']:+,} Rs) | PF: {sc['pf']:.2f}")
    for k, v in fc.items():
        print(f"  {k}: {v}")

    print("\n=== OPTUNA BEST (full window re-run) ===")
    print(f"score: {best.value:,.0f} | params: {bp}")
    sb = summarize(trades_best)
    fb = full_stats(trades_best)
    print(f"Trades: {sb['trades']} | WR: {sb['wr']:.1f}% | P&L: {sb['pts']:+,.1f} pts "
          f"({sb['rs']:+,} Rs) | PF: {sb['pf']:.2f}")
    for k, v in fb.items():
        print(f"  {k}: {v}")


def load_spot():
    spot = pd.read_csv(SPOT_CSV, parse_dates=["date"])
    spot = spot.sort_values("date").reset_index(drop=True)
    spot["day"] = spot["date"].dt.strftime("%Y-%m-%d")
    spot["min"] = spot["date"].dt.hour * 60 + spot["date"].dt.minute
    out = {}
    for day, g in spot.groupby("day"):
        out[day] = {
            "min": g["min"].to_numpy(),
            "open": g["open"].to_numpy(),
            "high": g["high"].to_numpy(),
            "low": g["low"].to_numpy(),
            "close": g["close"].to_numpy(),
        }
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2024-10-31")
    args = ap.parse_args()

    spot_all = load_spot()
    fut_files = futures_day_files("2020-01-01", "2024-12-31")
    opt_files = option_day_files("2020-01-01", "2024-12-31")
    days = sorted(set(fut_files) & set(opt_files) & set(spot_all))
    print(f"Futures days: {len(fut_files)} | option days: {len(opt_files)} | overlap: {len(days)}")

    if args.smoke:
        days5 = days[:5]
        print(f"=== SMOKE TEST — {len(days5)} DAYS ({days5[0]}..{days5[-1]}) — CHAMPION PARAMS ===")
        with Pool(processes=WORKERS, initializer=init_worker_local,
                  initargs=(spot_all,)) as pool:
            trades = run_days(pool, CHAMPION, days5, fut_files, opt_files, spot_all)
        st = summarize(trades)
        print(f"Trades: {st['trades']} | WR: {st['wr']:.1f}% | P&L: {st['pts']:+,.2f} pts | PF: {st['pf']:.2f}")
        print("SMOKE TEST OK" if 5 <= st["trades"] <= 40 else "SMOKE TEST SUSPICIOUS")
        return

    days_all = [d for d in days if args.start <= d <= args.end]
    days_2020 = [d for d in days_all if d.startswith("2020")]
    days_rest = [d for d in days_all if not d.startswith("2020")]
    print(f"Phase 1 window: {args.start}..{args.end} = {len(days_all)} days (2020 subset: {len(days_2020)})")
    print(f"Search space: {len(SEARCH_SPACE)} axes, "
          f"{np.prod([len(v) for v in SEARCH_SPACE.values()]):,} total combinations")
    print(f"Trials: {args.trials} | sampler: TPE multivariate | pruner: MedianPruner | workers: {WORKERS}")
    if Path(RESULTS_CSV).exists():
        print(f"NOTE: {RESULTS_CSV} exists — appending to it")
    run_optuna(args.trials, days_all, days_2020, days_rest, fut_files, opt_files, spot_all)


if __name__ == "__main__":
    main()
