"""High-Speed Multi-Core Stochastic Parameter Grid Search Engine (S1, S2, S3, S4).

Sweeps Stochastic Parameter Sets across all 5 years (2020-2024):
  - S1 Range: (7,3), (9,3), (12,3)
  - S2 Range: (12,3), (14,3), (18,3)
  - S3 Range: (21,4), (30,4), (40,4)
  - S4 Range: (50,10), (60,10), (75,10)
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from backtest_5y_optimized import load_spot, option_files, init_worker, SYM_RE, to_minutes, latest_spot, TimeframeTracker, summarize, print_yearly_breakdown
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from flattrade_bot.indicators.divergence import DivergenceEngine

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

GLOBAL_SPOT = {}

def init_worker_local(spot_dict):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot_dict


class CustomQuadStochastics:
    def __init__(self, s1_spec: Tuple[int, int], s2_spec: Tuple[int, int], s3_spec: Tuple[int, int], s4_spec: Tuple[int, int]):
        self.s1 = IncrementalStochastic(s1_spec[0], s1_spec[1])
        self.s2 = IncrementalStochastic(s2_spec[0], s2_spec[1])
        self.s3 = IncrementalStochastic(s3_spec[0], s3_spec[1])
        self.s4 = IncrementalStochastic(s4_spec[0], s4_spec[1])

    def push(self, high: float, low: float, close: float) -> Dict[str, Optional[float]]:
        return {
            "s1d": self.s1.push(high, low, close),
            "s2d": self.s2.push(high, low, close),
            "s3d": self.s3.push(high, low, close),
            "s4d": self.s4.push(high, low, close),
        }


class CustomTimeframeTracker:
    def __init__(self, tf_label: str, max_lookback: int, s1_s, s2_s, s3_s, s4_s):
        self.tf_label = tf_label
        self.max_lookback = max_lookback
        self.stoch = CustomQuadStochastics(s1_s, s2_s, s3_s, s4_s)
        self.divergence = DivergenceEngine()
        self.history: List[Candle] = []
        self.setup_active = False
        self.setup_type = ""
        self.prev_s1 = None
        self.s4_embedded_count = 0

    def push(self, candle: Candle) -> Tuple[bool, bool, str, float]:
        self.history.append(candle)
        if len(self.history) > 40:
            self.history.pop(0)

        stoch_vals = self.stoch.push(candle.high, candle.low, candle.close)
        s1, s2, s3, s4 = (stoch_vals.get(k) for k in ("s1d", "s2d", "s3d", "s4d"))
        self.prev_s1 = s1

        if s4 is not None:
            if s4 <= 20.0:
                self.s4_embedded_count += 1
            else:
                self.s4_embedded_count = 0

        embedded_active = self.s4_embedded_count > 25

        self.divergence.update(candle.close, s1)
        has_bull_div = self.divergence.has_bullish_trough_divergence()

        is_flag = False if any(v is None for v in (s1, s4)) else (s4 >= 79.5 and s1 <= 20.5)
        is_super = False if any(v is None for v in (s1, s2, s3, s4)) else all(v <= 20.5 for v in (s1, s2, s3, s4))

        if (is_flag or is_super) and has_bull_div:
            self.setup_active = True
            self.setup_type = "super" if is_super else "flag"

        is_reverse_mode = embedded_active and (self.setup_type == "super")

        triggered = False
        if self.setup_active and len(self.history) >= 2:
            if BullishPinBarDetector.check_vicinity_breakout(self.history, self.max_lookback):
                triggered = True
                self.setup_active = False

        return triggered, is_reverse_mode, self.setup_type, candle.close


class CustomMultiTimeframeTracker:
    def __init__(self, s1_s, s2_s, s3_s, s4_s):
        self.trackers = {
            tf: CustomTimeframeTracker(tf, max_lookback=spec[1], s1_s=s1_s, s2_s=s2_s, s3_s=s3_s, s4_s=s4_s) for tf, spec in TF_SPECS.items()
        }
        self.buffers = {tf: [] for tf in TF_SPECS}

    def push_1m(self, c1m: Candle) -> List[Tuple[str, bool, str, float]]:
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
                trig, is_rev, stype, px = self.trackers[tf].push(c_tf)
                if trig:
                    triggers.append((tf, is_rev, stype, px))
        return triggers


def process_single_day_custom_stoch(args):
    day, file_path_str, prev_file_path_str, s1_s, s2_s, s3_s, s4_s = args
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
        mtf_trackers[sym] = CustomMultiTimeframeTracker(s1_s, s2_s, s3_s, s4_s)
        mins = g["min"].to_numpy()
        opens = g["open"].to_numpy()
        highs = g["high"].to_numpy()
        lows = g["low"].to_numpy()
        closes = g["close"].to_numpy()
        for i in range(len(mins)):
            c1m = Candle(open=opens[i], high=highs[i], low=lows[i], close=closes[i], minute=mins[i])
            mtf_trackers[sym].push_1m(c1m)

    per_minute_triggers = {}
    slices = {}

    for sym, g in groups_curr.items():
        if sym not in mtf_trackers:
            mtf_trackers[sym] = CustomMultiTimeframeTracker(s1_s, s2_s, s3_s, s4_s)
        tracker = mtf_trackers[sym]

        mins = g["min"].to_numpy()
        opens = g["open"].to_numpy()
        highs = g["high"].to_numpy()
        lows = g["low"].to_numpy()
        closes = g["close"].to_numpy()

        slices[sym] = {
            "min": mins, "open": opens, "high": highs, "low": lows, "close": closes
        }

        m_match = SYM_RE.match(sym)
        if not m_match:
            continue
        strike_val = int(m_match.group(2))
        side_val = m_match.group(3)

        for i in range(len(mins)):
            m = mins[i]
            c1m = Candle(open=opens[i], high=highs[i], low=lows[i], close=closes[i], minute=m)
            trig_list = tracker.push_1m(c1m)
            for (tf_label, is_rev, stype, px) in trig_list:
                sl_pts = TF_SPECS[tf_label][2]
                tp_pts = TF_SPECS[tf_label][3]
                per_minute_triggers.setdefault(m, []).append(
                    (side_val, strike_val, sym, px, is_rev, tf_label, sl_pts, tp_pts)
                )

    trades = []
    pos = None
    daily_pnl_pts = 0.0
    consecutive_losses = 0
    shutdown = False

    def bar_at_slice(option_slice, minute):
        if option_slice is None:
            return None
        idx = np.searchsorted(option_slice["min"], minute)
        if idx < len(option_slice["min"]) and option_slice["min"][idx] == minute:
            return (
                option_slice["open"][idx], option_slice["high"][idx],
                option_slice["low"][idx], option_slice["close"][idx]
            )
        return None

    def get_active_info(side, minute):
        spot_px = latest_spot(spot, minute)
        if spot_px is None:
            return None
        atm = int(round(spot_px / 50.0) * 50)
        strike = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        symbol = f"{prefix}{strike}{side}"
        current_slice = slices.get(symbol)
        return (symbol, current_slice, strike) if current_slice is not None else None

    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None:
            held = bar_at_slice(pos["slice"], minute)
            if held is not None:
                o_px, h_px, l_px, c_px = held
                pos["last_px"] = float(c_px)
                pos["duration_min"] += 1

                if daily_pnl_pts + (c_px - pos["entry"]) <= DAILY_MAX_LOSS_PTS:
                    pts = round(c_px - pos["entry"], 2)
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": c_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": "SHUTDOWN_LOSS", "duration_min": pos["duration_min"],
                        "is_rev": pos["is_rev"], "tf": pos["tf"]
                    })
                    daily_pnl_pts += pts
                    pos = None
                    shutdown = True
                    continue

                exit_px, reason = None, ""
                if h_px >= pos["tgt"] and l_px <= pos["sl"]:
                    exit_px, reason = pos["sl"], "SL"
                elif h_px >= pos["tgt"]:
                    exit_px, reason = pos["tgt"], "TP"
                elif l_px <= pos["sl"]:
                    exit_px, reason = pos["sl"], "SL"

                if exit_px is None:
                    mtf_tr = mtf_trackers.get(pos["symbol"])
                    if mtf_tr:
                        tr = mtf_tr.trackers["1m"]
                        tr.divergence.update(c_px, tr.prev_s1)
                        if tr.divergence.has_bearish_peak_divergence():
                            exit_px, reason = c_px, "BEARISH_PEAK_REVERSAL"

                if exit_px is not None:
                    pts = round(exit_px - pos["entry"], 2)
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": exit_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": reason, "duration_min": pos["duration_min"],
                        "is_rev": pos["is_rev"], "tf": pos["tf"]
                    })
                    daily_pnl_pts += pts
                    consecutive_losses = consecutive_losses + 1 if pts <= 0 else 0
                    
                    if daily_pnl_pts >= DAILY_MAX_PROFIT_PTS:
                        shutdown = True
                    elif consecutive_losses >= CONSECUTIVE_LOSS_LIMIT or daily_pnl_pts <= DAILY_MAX_LOSS_PTS:
                        shutdown = True
                    pos = None

            if minute >= SESSION_END and pos is not None:
                pts = round(pos["last_px"] - pos["entry"], 2)
                trades.append({
                    "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                    "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                    "exit": pos["last_px"], "pts": pts, "rs": round(pts * LOT_SIZE),
                    "reason": "EOD", "duration_min": pos["duration_min"],
                    "is_rev": pos["is_rev"], "tf": pos["tf"]
                })
                daily_pnl_pts += pts
                pos = None
                break

        if pos is not None or shutdown or minute >= SESSION_END:
            continue

        trigs = per_minute_triggers.get(minute, [])
        for (signal_side, signal_strike, signal_symbol, c_px, is_rev, tf_label, sl_pts, tp_pts) in trigs:
            active_info = get_active_info(signal_side, minute)
            if active_info and active_info[2] == signal_strike and pos is None:
                if is_rev:
                    actual_side = "PE" if signal_side == "CE" else "CE"
                    actual_info = get_active_info(actual_side, minute)
                    if actual_info is None:
                        continue
                    actual_symbol, actual_slice, _ = actual_info
                else:
                    actual_side = signal_side
                    actual_symbol = signal_symbol
                    actual_slice = active_info[1]

                bar = bar_at_slice(actual_slice, minute)
                if bar is not None:
                    entry_px = float(bar[3])
                    pos = {
                        "side": actual_side, "symbol": actual_symbol, "slice": actual_slice,
                        "entry": entry_px, "sl": entry_px - sl_pts, "tgt": entry_px + tp_pts,
                        "entry_min": minute, "last_px": entry_px, "duration_min": 0,
                        "is_rev": is_rev, "tf": tf_label
                    }
                    break

    return trades


def evaluate_stoch_combo(args):
    s1_s, s2_s, s3_s, s4_s, day_tasks, spot_all = args

    tasks = [(t[0], t[1], t[2], s1_s, s2_s, s3_s, s4_s) for t in day_tasks]
    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,)) as pool:
        results = pool.map(process_single_day_custom_stoch, tasks)
        for res in results:
            all_trades.extend(res)

    st = summarize(all_trades)
    return {
        "s1": f"({s1_s[0]},{s1_s[1]})",
        "s2": f"({s2_s[0]},{s2_s[1]})",
        "s3": f"({s3_s[0]},{s3_s[1]})",
        "s4": f"({s4_s[0]},{s4_s[1]})",
        "trades": st["trades"], "wr": st["wr"], "pts": st["pts"], "rs": st["rs"], "pf": st["pf"]
    }


def main():
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    day_tasks = [(day, str(files[day]), str(files[days[i-1]]) if i > 0 else "") for i, day in enumerate(days)]

    # Grid candidate specs
    s1_specs = [(7, 3), (9, 3), (12, 3)]
    s2_specs = [(12, 3), (14, 3), (18, 3)]
    s3_specs = [(21, 4), (30, 4), (40, 4)]
    s4_specs = [(50, 10), (60, 10), (75, 10)]

    # Test key candidate combinations
    candidate_stochs = [
        ((9,3), (14,3), (40,4), (60,10)),   # Baseline
        ((7,3), (14,3), (40,4), (60,10)),   # Faster S1
        ((12,3), (14,3), (40,4), (60,10)),  # Slower S1
        ((9,3), (12,3), (40,4), (60,10)),   # Faster S2
        ((9,3), (18,3), (40,4), (60,10)),   # Slower S2
        ((9,3), (14,3), (21,4), (60,10)),   # Faster S3
        ((9,3), (14,3), (30,4), (60,10)),   # Medium S3
        ((9,3), (14,3), (40,4), (50,10)),   # Faster S4
        ((9,3), (14,3), (40,4), (75,10)),   # Slower S4
        ((7,3), (12,3), (21,4), (50,10)),   # High Speed Combo
    ]

    print(f"Loaded {len(days)} trading days across 2020-2024.")
    print(f"Sweeping {len(candidate_stochs)} Stochastic Parameter Combinations on 12 CPU cores...", flush=True)

    results = []
    t0 = time.time()

    for s1_s, s2_s, s3_s, s4_s in candidate_stochs:
        print(f"  Testing S1={s1_s}, S2={s2_s}, S3={s3_s}, S4={s4_s}...", end="", flush=True)
        res = evaluate_stoch_combo((s1_s, s2_s, s3_s, s4_s, day_tasks, spot_all))
        results.append(res)
        print(f" -> Net Profit: Rs {res['rs']:+10,d} | WR: {res['wr']:5.1f}% | Net Pts: {res['pts']:+8.2f} | PF: {res['pf']:.2f}")

    elapsed = time.time() - t0
    print(f"\n[OK] COMPLETED STOCHASTIC GRID SEARCH IN {elapsed:.2f} SECONDS!")

    df_res = pd.DataFrame(results).sort_values("rs", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 125)
    print("STOCHASTIC PARAMETER COMBINATIONS RANKED BY TOTAL NET PROFIT")
    print("=" * 125)
    print(f"{'RANK':4s} | {'S1 (K,D)':10s} | {'S2 (K,D)':10s} | {'S3 (K,D)':10s} | {'S4 (K,D)':10s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'PROFIT (Rs)':14s} | {'PROFIT FACTOR'}")
    print("-" * 125)

    for idx, row in df_res.iterrows():
        pf_str = f"{row['pf']:.2f}" if row['pf'] != float("inf") else "INF"
        print(f"#{idx+1:2d}  | {row['s1']:10s} | {row['s2']:10s} | {row['s3']:10s} | {row['s4']:10s} | {int(row['trades']):7d} | {row['wr']:8.1f}% | {row['pts']:+10.2f} | Rs {int(row['rs']):+12,d} | {pf_str:>13s}")


if __name__ == "__main__":
    main()
