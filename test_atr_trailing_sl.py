"""ATR-Based SL & TP + Trailing SL Backtest Evaluation Engine.

Backtest 1: ATR-Based SL & TP
  - SL = Entry - (ATR(14) * SL_MULT)
  - TP = Entry + (ATR(14) * TP_MULT)
  - Grid sweeps SL_MULT: 1.0, 1.5, 2.0 and TP_MULT: 2.0, 3.0, 4.0

Backtest 2: Trailing Stop Loss
  - Initial SL = Entry - 6.0 pts (1m default)
  - Trail Rule: For every +10.0 pts gain above entry, trail SL up by +5.0 pts
  - E.g. price at +10 -> SL moves to entry+0; price at +20 -> SL moves to entry+10
"""

import re
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count
from collections import deque

import numpy as np
import pandas as pd

from backtest_5y_optimized import load_spot, option_files, SYM_RE, to_minutes, latest_spot, TimeframeTracker, summarize, print_yearly_breakdown
from flattrade_bot.indicators.patterns import Candle

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_MAX_PROFIT_PTS = 30.0
DAILY_MAX_LOSS_PTS = -30.0
CONSECUTIVE_LOSS_LIMIT = 6

TF_SPECS = {
    "1m": (1, 10, 6.0, 30.0),
    "2m": (2, 5, 10.0, 15.0),
    "3m": (3, 4, 8.0, 25.0),
    "5m": (5, 3, 10.0, 35.0),
}

# ATR Grid params
ATR_PERIOD = 14
ATR_GRID = [(1.0, 2.0), (1.0, 3.0), (1.5, 2.0), (1.5, 3.0), (2.0, 3.0), (2.0, 4.0)]

# Trailing SL params
TRAIL_STEP_PTS = 10.0   # for every +10 pts gain
TRAIL_AMOUNT_PTS = 5.0  # SL moves up by +5 pts

GLOBAL_SPOT = {}

def init_worker_local(spot_dict):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot_dict


class IncrementalATR:
    def __init__(self, period: int = 14):
        self.period = period
        self._tr_buf = deque(maxlen=period)
        self.atr = None
        self.prev_close = None
        self._count = 0

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self._tr_buf.append(tr)
        self._count += 1
        if self._count < self.period:
            self.atr = None
        elif self._count == self.period:
            self.atr = sum(self._tr_buf) / self.period
        else:
            self.atr = (self.atr * (self.period - 1) + tr) / self.period
        self.prev_close = close
        return self.atr


class MultiTimeframeTracker:
    def __init__(self):
        self.trackers = {
            tf: TimeframeTracker(tf, max_lookback=spec[1]) for tf, spec in TF_SPECS.items()
        }
        self.atrs = {tf: IncrementalATR(ATR_PERIOD) for tf in TF_SPECS}
        self.buffers = {tf: [] for tf in TF_SPECS}

    def push_1m(self, c1m: Candle) -> List[Tuple[str, bool, str, float, Optional[float]]]:
        triggers = []
        for tf, spec in TF_SPECS.items():
            tf_m = spec[0]
            buf = self.buffers[tf]
            buf.append(c1m)
            if len(buf) == tf_m:
                c_tf = Candle(
                    open=buf[0].open, high=max(c.high for c in buf),
                    low=min(c.low for c in buf), close=buf[-1].close, minute=buf[-1].minute
                )
                self.buffers[tf] = []
                atr_val = self.atrs[tf].update(c_tf.high, c_tf.low, c_tf.close)
                trig, is_rev, stype, px = self.trackers[tf].push(c_tf)
                if trig:
                    triggers.append((tf, is_rev, stype, px, atr_val))
        return triggers


def process_single_day(args):
    day, file_path_str, prev_file_path_str, mode, atr_sl_mult, atr_tp_mult = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not file_path_str:
        return []

    file_path = Path(file_path_str)
    if not file_path.exists():
        return []

    spot_0915 = latest_spot(spot, 555) or latest_spot(spot, 560)
    if spot_0915 is None:
        return []

    atm_approx = int(round(spot_0915 / 50.0) * 50)
    target_strikes = set(range(atm_approx - 250, atm_approx + 300, 50))

    try:
        df_curr = pd.read_csv(file_path, usecols=["time", "symbol", "open", "high", "low", "close"], engine="c")
    except Exception:
        return []

    if df_curr.empty:
        return []

    first_sym = df_curr["symbol"].iloc[0]
    m_match = SYM_RE.match(first_sym)
    if not m_match:
        return []
    prefix = m_match.group(1)

    df_curr["min"] = np.array([to_minutes(t) for t in df_curr["time"]])

    groups_curr = {}
    for sym, g in df_curr.groupby("symbol"):
        m = SYM_RE.match(sym)
        if m and int(m.group(2)) in target_strikes:
            groups_curr[sym] = g

    groups_prev = {}
    if prev_file_path_str and Path(prev_file_path_str).exists():
        try:
            df_prev = pd.read_csv(prev_file_path_str, usecols=["time", "symbol", "open", "high", "low", "close"], engine="c")
            if not df_prev.empty:
                df_prev["min"] = np.array([to_minutes(t) for t in df_prev["time"]])
                for sym, g in df_prev.groupby("symbol"):
                    m = SYM_RE.match(sym)
                    if m and int(m.group(2)) in target_strikes:
                        groups_prev[sym] = g
        except Exception:
            pass

    mtf_trackers = {}

    for sym, g in groups_prev.items():
        mtf_trackers[sym] = MultiTimeframeTracker()
        mins = g["min"].to_numpy(); opens = g["open"].to_numpy()
        highs = g["high"].to_numpy(); lows = g["low"].to_numpy(); closes = g["close"].to_numpy()
        for i in range(len(mins)):
            mtf_trackers[sym].push_1m(Candle(open=opens[i], high=highs[i], low=lows[i], close=closes[i], minute=mins[i]))

    per_minute_triggers = {}
    slices = {}

    for sym, g in groups_curr.items():
        if sym not in mtf_trackers:
            mtf_trackers[sym] = MultiTimeframeTracker()
        tracker = mtf_trackers[sym]
        mins = g["min"].to_numpy(); opens = g["open"].to_numpy()
        highs = g["high"].to_numpy(); lows = g["low"].to_numpy(); closes = g["close"].to_numpy()
        slices[sym] = {"min": mins, "open": opens, "high": highs, "low": lows, "close": closes}
        m_match = SYM_RE.match(sym)
        if not m_match: continue
        strike_val = int(m_match.group(2)); side_val = m_match.group(3)
        for i in range(len(mins)):
            m = mins[i]
            trig_list = tracker.push_1m(Candle(open=opens[i], high=highs[i], low=lows[i], close=closes[i], minute=m))
            for (tf_label, is_rev, stype, px, atr_val) in trig_list:
                sl_pts = TF_SPECS[tf_label][2]; tp_pts = TF_SPECS[tf_label][3]
                per_minute_triggers.setdefault(m, []).append(
                    (side_val, strike_val, sym, px, is_rev, tf_label, sl_pts, tp_pts, atr_val))

    trades = []
    pos = None
    daily_pnl_pts = 0.0
    consecutive_losses = 0
    shutdown = False

    def bar_at_slice(sl, minute):
        idx = np.searchsorted(sl["min"], minute)
        if idx < len(sl["min"]) and sl["min"][idx] == minute:
            return sl["open"][idx], sl["high"][idx], sl["low"][idx], sl["close"][idx]
        return None

    def get_active_info(side, minute):
        spot_px = latest_spot(spot, minute)
        if spot_px is None: return None
        atm = int(round(spot_px / 50.0) * 50)
        strike = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        symbol = f"{prefix}{strike}{side}"
        sl = slices.get(symbol)
        return (symbol, sl, strike) if sl is not None else None

    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None:
            held = bar_at_slice(pos["slice"], minute)
            if held is not None:
                o_px, h_px, l_px, c_px = held
                pos["last_px"] = float(c_px)
                pos["duration_min"] += 1

                # Update trailing SL if in trailing mode
                if mode == "trailing":
                    gain = c_px - pos["entry"]
                    trail_steps = int(gain / TRAIL_STEP_PTS)
                    if trail_steps > pos["trail_steps_done"]:
                        sl_bump = (trail_steps - pos["trail_steps_done"]) * TRAIL_AMOUNT_PTS
                        pos["sl"] += sl_bump
                        pos["trail_steps_done"] = trail_steps

                # Daily max loss shutdown
                if daily_pnl_pts + (c_px - pos["entry"]) <= DAILY_MAX_LOSS_PTS:
                    pts = round(c_px - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": c_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": "SHUTDOWN_LOSS", "duration_min": pos["duration_min"],
                        "is_rev": pos["is_rev"], "tf": pos["tf"]})
                    daily_pnl_pts += pts; pos = None; shutdown = True; continue

                exit_px, reason = None, ""
                has_tgt = pos.get("tgt") is not None
                if has_tgt and h_px >= pos["tgt"] and l_px <= pos["sl"]:
                    exit_px, reason = pos["sl"], "SL"
                elif has_tgt and h_px >= pos["tgt"]:
                    exit_px, reason = pos["tgt"], "TP"
                elif l_px <= pos["sl"]:
                    exit_px, reason = pos["sl"], "SL"

                if exit_px is None:
                    tr = mtf_trackers.get(pos["symbol"])
                    if tr:
                        t1 = tr.trackers["1m"]
                        t1.divergence.update(c_px, t1.prev_s1)
                        if t1.divergence.has_bearish_peak_divergence():
                            exit_px, reason = c_px, "BEARISH_PEAK_REVERSAL"

                if exit_px is not None:
                    pts = round(exit_px - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": exit_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": reason, "duration_min": pos["duration_min"],
                        "is_rev": pos["is_rev"], "tf": pos["tf"]})
                    daily_pnl_pts += pts
                    consecutive_losses = consecutive_losses + 1 if pts <= 0 else 0
                    if daily_pnl_pts >= DAILY_MAX_PROFIT_PTS: shutdown = True
                    elif consecutive_losses >= CONSECUTIVE_LOSS_LIMIT or daily_pnl_pts <= DAILY_MAX_LOSS_PTS: shutdown = True
                    pos = None

            if minute >= SESSION_END and pos is not None:
                pts = round(pos["last_px"] - pos["entry"], 2)
                trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                    "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                    "exit": pos["last_px"], "pts": pts, "rs": round(pts * LOT_SIZE),
                    "reason": "EOD", "duration_min": pos["duration_min"],
                    "is_rev": pos["is_rev"], "tf": pos["tf"]})
                daily_pnl_pts += pts; pos = None; break

        if pos is not None or shutdown or minute >= SESSION_END:
            continue

        trigs = per_minute_triggers.get(minute, [])
        for (signal_side, signal_strike, signal_symbol, c_px, is_rev, tf_label, sl_pts, tp_pts, atr_val) in trigs:
            active_info = get_active_info(signal_side, minute)
            if active_info and active_info[2] == signal_strike and pos is None:
                if is_rev:
                    actual_side = "PE" if signal_side == "CE" else "CE"
                    actual_info = get_active_info(actual_side, minute)
                    if actual_info is None: continue
                    actual_symbol, actual_slice, _ = actual_info
                else:
                    actual_side = signal_side; actual_symbol = signal_symbol; actual_slice = active_info[1]

                bar = bar_at_slice(actual_slice, minute)
                if bar is None: continue
                entry_px = float(bar[3])

                if mode == "atr":
                    if atr_val is None or atr_val <= 0.5:
                        # Fallback to fixed if ATR not ready
                        sl = entry_px - sl_pts
                        tgt = entry_px + tp_pts
                    else:
                        sl = entry_px - (atr_val * atr_sl_mult)
                        tgt = entry_px + (atr_val * atr_tp_mult)
                    pos = {"side": actual_side, "symbol": actual_symbol, "slice": actual_slice,
                        "entry": entry_px, "sl": sl, "tgt": tgt,
                        "entry_min": minute, "last_px": entry_px, "duration_min": 0,
                        "is_rev": is_rev, "tf": tf_label}

                elif mode == "trailing":
                    pos = {"side": actual_side, "symbol": actual_symbol, "slice": actual_slice,
                        "entry": entry_px, "sl": entry_px - sl_pts, "tgt": None,
                        "entry_min": minute, "last_px": entry_px, "duration_min": 0,
                        "is_rev": is_rev, "tf": tf_label, "trail_steps_done": 0}

                else:  # baseline
                    pos = {"side": actual_side, "symbol": actual_symbol, "slice": actual_slice,
                        "entry": entry_px, "sl": entry_px - sl_pts, "tgt": entry_px + tp_pts,
                        "entry_min": minute, "last_px": entry_px, "duration_min": 0,
                        "is_rev": is_rev, "tf": tf_label}
                break

    return trades


def run_mode(mode: str, label: str, atr_sl_mult=1.0, atr_tp_mult=2.0):
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))
    tasks = [(day, str(files[day]), str(files[days[i-1]]) if i > 0 else "", mode, atr_sl_mult, atr_tp_mult)
             for i, day in enumerate(days)]

    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,)) as pool:
        results = pool.map(process_single_day, tasks)
        for res in results:
            all_trades.extend(res)

    st = summarize(all_trades)
    print(f"\n{'='*115}")
    print(f"5-YEAR RESULTS FOR: {label.upper()}")
    print(f"{'='*115}")
    print(f"Total Trades : {st['trades']}")
    print(f"Win Rate     : {st['wr']:.1f}%")
    print(f"Net Points   : {st['pts']:+.2f} pts")
    print(f"Net Profit   : Rs {st['rs']:+,d}")
    pf_str = f"{st['pf']:.2f}" if st['pf'] != float("inf") else "INF"
    print(f"Profit Factor: {pf_str}")
    print_yearly_breakdown(all_trades)
    return st


def main():
    print("BACKTEST 1: ATR-Based SL & TP (Grid Search over Multipliers)", flush=True)
    st_base = run_mode("baseline", "Baseline (Fixed SL/TP)")

    atr_results = []
    for sl_mult, tp_mult in ATR_GRID:
        label = f"ATR(14) SL x{sl_mult} | TP x{tp_mult}"
        print(f"\nTesting {label}...", flush=True)
        st = run_mode("atr", label, sl_mult, tp_mult)
        atr_results.append((sl_mult, tp_mult, st))

    print(f"\n{'='*115}")
    print("BACKTEST 2: Trailing SL (Trail +5 pts per +10 pts gain, No Fixed TP)", flush=True)
    st_trail = run_mode("trailing", f"Trailing SL (+{TRAIL_AMOUNT_PTS}pts per +{TRAIL_STEP_PTS}pts gain)")

    # Final comparison table
    print(f"\n{'='*125}")
    print("FULL SL/TP METHOD COMPARISON SUMMARY (5 Years, 2020-2024)")
    print(f"{'='*125}")
    print(f"{'STRATEGY':40s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'NET PROFIT (Rs)':14s} | {'PF'}")
    print(f"{'-'*125}")

    def row(name, st):
        pf_s = f"{st['pf']:.2f}" if st['pf'] != float("inf") else "INF"
        print(f"{name:40s} | {st['trades']:7d} | {st['wr']:8.1f}% | {st['pts']:+10.2f} | Rs {st['rs']:+12,d} | {pf_s}")

    row("Baseline (Fixed SL/TP per TF)", st_base)
    for sl_mult, tp_mult, st in sorted(atr_results, key=lambda x: x[2]["rs"], reverse=True):
        row(f"ATR(14) SL x{sl_mult} / TP x{tp_mult}", st)
    row(f"Trailing SL (+5pts per +10pts gain)", st_trail)


if __name__ == "__main__":
    main()
