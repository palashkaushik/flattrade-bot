"""5-Year Macro Filters Evaluation (India VIX & Dow Jones) on Overlapping Trading Days.

Filter Definitions:
  1. India VIX Filter:
     - Calming VIX (VIX_close < VIX_prev_close) -> Bullish Nifty -> CE trades allowed
     - Expanding VIX (VIX_close > VIX_prev_close) -> Bearish Nifty -> PE trades allowed

  2. Dow Jones (DJI) Filter:
     - Positive Dow return (DJI_close > DJI_prev_close) -> Bullish Nifty -> CE trades allowed
     - Negative Dow return (DJI_close < DJI_prev_close) -> Bearish Nifty -> PE trades allowed

Evaluated strictly across the 1,154 overlapping trading days (2020-2024).
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

GLOBAL_SPOT = {}

def init_worker_local(spot_dict):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot_dict


def load_macro_data():
    vix_df = pd.read_csv("c:/Websites/ammu/macro_data/INDIAVIX_day.csv")
    vix_df["prev_close"] = vix_df["close"].shift(1)
    vix_map = {}
    for _, row in vix_df.iterrows():
        d = str(row["date"])
        c, p = float(row["close"]), float(row["prev_close"]) if pd.notnull(row["prev_close"]) else float(row["close"])
        vix_map[d] = "FALLING" if c < p else "RISING"

    dji_df = pd.read_csv("c:/Websites/ammu/macro_data/DJI_day.csv")
    dji_df["prev_close"] = dji_df["close"].shift(1)
    dji_map = {}
    for _, row in dji_df.iterrows():
        d = str(row["date"])
        c, p = float(row["close"]), float(row["prev_close"]) if pd.notnull(row["prev_close"]) else float(row["close"])
        dji_map[d] = "BULLISH" if c >= p else "BEARISH"

    return vix_map, dji_map


class MultiTimeframeSymbolTrackerMacro:
    def __init__(self):
        self.trackers = {
            tf: TimeframeTracker(tf, max_lookback=spec[1]) for tf, spec in TF_SPECS.items()
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


def process_single_day_macro(args):
    day, file_path_str, prev_file_path_str, vix_sent, dji_sent, filter_mode = args
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
        mtf_trackers[sym] = MultiTimeframeSymbolTrackerMacro()
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
            mtf_trackers[sym] = MultiTimeframeSymbolTrackerMacro()
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

                # Apply Macro Filters
                if filter_mode == "vix":
                    if actual_side == "CE" and vix_sent != "FALLING":
                        continue
                    if actual_side == "PE" and vix_sent != "RISING":
                        continue
                elif filter_mode == "dji":
                    if actual_side == "CE" and dji_sent != "BULLISH":
                        continue
                    if actual_side == "PE" and dji_sent != "BEARISH":
                        continue
                elif filter_mode == "combined":
                    # Both VIX and DJI must align
                    if actual_side == "CE" and not (vix_sent == "FALLING" and dji_sent == "BULLISH"):
                        continue
                    if actual_side == "PE" and not (vix_sent == "RISING" and dji_sent == "BEARISH"):
                        continue

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


def run_macro_mode(filter_mode: str, label: str):
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    nifty_days = set(files.keys()) & set(spot_all.keys())

    vix_map, dji_map = load_macro_data()

    # Exact overlapping trading days only
    overlap_days = sorted(nifty_days & set(vix_map.keys()) & set(dji_map.keys()))
    all_nifty_sorted = sorted(nifty_days)

    tasks = []
    for day in overlap_days:
        idx = all_nifty_sorted.index(day)
        prev_day_str = str(files[all_nifty_sorted[idx-1]]) if idx > 0 else ""
        v_sent = vix_map.get(day, "NEUTRAL")
        d_sent = dji_map.get(day, "NEUTRAL")
        tasks.append((day, str(files[day]), prev_day_str, v_sent, d_sent, filter_mode))

    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,)) as pool:
        results = pool.map(process_single_day_macro, tasks)
        for res in results:
            all_trades.extend(res)

    st = summarize(all_trades)
    print("\n" + "=" * 115)
    print(f"OVERLAPPING DATASET RESULTS ({len(overlap_days)} DAYS) FOR: {label.upper()}")
    print("=" * 115)
    print(f"Total Trades : {st['trades']}")
    print(f"Win Rate     : {st['wr']:.1f}%")
    print(f"Net Points   : {st['pts']:+.2f} pts")
    print(f"Net Profit   : Rs {st['rs']:+,d}")
    pf_str = f"{st['pf']:.2f}" if st['pf'] != float("inf") else "INF"
    print(f"Profit Factor: {pf_str}")
    print_yearly_breakdown(all_trades)
    return st


def main():
    print("Evaluating India VIX & Dow Jones Macro Filters across 1,154 Overlapping Days...", flush=True)

    st_none = run_macro_mode("none", "1. Baseline Engine (No Macro Filter)")
    st_vix  = run_macro_mode("vix", "2. India VIX Filter (CE on Calming VIX, PE on Expanding VIX)")
    st_dji  = run_macro_mode("dji", "3. Dow Jones Filter (CE on Green Dow, PE on Red Dow)")
    st_comb = run_macro_mode("combined", "4. Combined Macro Filter (VIX + Dow Alignment)")

    print("\n" + "=" * 115)
    print("MACRO FILTERS COMPARISON SUMMARY (1,154 EXACT OVERLAPPING TRADING DAYS)")
    print("=" * 115)
    print(f"{'FILTER CONFIGURATION':40s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'NET PROFIT (Rs)':16s} | {'PROFIT FACTOR'}")
    print("-" * 115)
    for name, st in [
        ("Baseline (No Macro Filter)", st_none),
        ("India VIX Filter", st_vix),
        ("Dow Jones Filter", st_dji),
        ("Combined Macro Filter (VIX + Dow)", st_comb)
    ]:
        pf_s = f"{st['pf']:.2f}" if st['pf'] != float("inf") else "INF"
        print(f"{name:40s} | {st['trades']:7d} | {st['wr']:8.1f}% | {st['pts']:+10.2f} | Rs {st['rs']:+14,d} | {pf_s:>13s}")


if __name__ == "__main__":
    main()
