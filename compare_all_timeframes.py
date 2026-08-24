"""Multi-Timeframe 5-Year Backtest Comparison Engine: 1m vs 2m vs 3m vs 5m vs Dual (1m+2m)."""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from backtest_5y_optimized import load_spot, option_files, init_worker, SYM_RE, to_minutes, latest_spot, TimeframeTracker, summarize, print_yearly_breakdown, CE_OFFSET, PE_OFFSET, SL_POINTS, TP_POINTS, SESSION_START, SESSION_END, DAY_LAST, DAILY_SHUTDOWN_LOSS_RS, DAILY_SHUTDOWN_PROFIT_RS, CONSECUTIVE_LOSS_LIMIT, LOT_SIZE
from flattrade_bot.indicators.patterns import Candle

GLOBAL_SPOT = {}

def init_worker_local(spot_dict):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot_dict

def process_single_day_tf(args):
    day, file_path_str, prev_file_path_str, tf_minutes, max_lookback = args
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

    trackers = {}
    buffers = {}

    def push_1m_candle(sym, c1m):
        if sym not in trackers:
            trackers[sym] = TimeframeTracker(f"{tf_minutes}m", max_lookback=max_lookback)
            buffers[sym] = []
        
        buf = buffers[sym]
        buf.append(c1m)
        if len(buf) == tf_minutes:
            c_tf = Candle(
                open=buf[0].open,
                high=max(c.high for c in buf),
                low=min(c.low for c in buf),
                close=buf[-1].close,
                minute=buf[-1].minute
            )
            buffers[sym] = []
            return trackers[sym].push(c_tf)
        return False, False, "", 0.0

    # Warmup prev day
    for sym, g in groups_prev.items():
        mins = g["min"].to_numpy()
        opens = g["open"].to_numpy()
        highs = g["high"].to_numpy()
        lows = g["low"].to_numpy()
        closes = g["close"].to_numpy()
        for i in range(len(mins)):
            m = mins[i]
            c1m = Candle(open=opens[i], high=highs[i], low=lows[i], close=closes[i], minute=m)
            push_1m_candle(sym, c1m)

    per_minute_triggers = {}
    slices = {}

    for sym, g in groups_curr.items():
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
            trig, is_rev, stype, px = push_1m_candle(sym, c1m)
            if trig:
                per_minute_triggers.setdefault(m, []).append((side_val, strike_val, sym, px, is_rev))

    trades = []
    pos = None
    daily_pnl = 0.0
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

                if daily_pnl + (c_px - pos["entry"]) * LOT_SIZE <= -DAILY_SHUTDOWN_LOSS_RS:
                    pts = round(c_px - pos["entry"], 2)
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": c_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": "SHUTDOWN_LOSS", "duration_min": pos["duration_min"], "is_rev": pos["is_rev"]
                    })
                    daily_pnl += pts * LOT_SIZE
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
                    tr = trackers.get(pos["symbol"])
                    if tr:
                        tr.divergence.update(c_px, tr.prev_s1)
                        if tr.divergence.has_bearish_peak_divergence():
                            exit_px, reason = c_px, "BEARISH_PEAK_REVERSAL"

                if exit_px is not None:
                    pts = round(exit_px - pos["entry"], 2)
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": exit_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": reason, "duration_min": pos["duration_min"], "is_rev": pos["is_rev"]
                    })
                    daily_pnl += pts * LOT_SIZE
                    consecutive_losses = consecutive_losses + 1 if pts <= 0 else 0
                    
                    if daily_pnl >= DAILY_SHUTDOWN_PROFIT_RS:
                        shutdown = True
                    elif consecutive_losses >= CONSECUTIVE_LOSS_LIMIT or daily_pnl <= -DAILY_SHUTDOWN_LOSS_RS:
                        shutdown = True
                    pos = None

            if minute >= SESSION_END and pos is not None:
                pts = round(pos["last_px"] - pos["entry"], 2)
                trades.append({
                    "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                    "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                    "exit": pos["last_px"], "pts": pts, "rs": round(pts * LOT_SIZE),
                    "reason": "EOD", "duration_min": pos["duration_min"], "is_rev": pos["is_rev"]
                })
                daily_pnl += pts * LOT_SIZE
                pos = None
                break

        if pos is not None or shutdown or minute >= SESSION_END:
            continue

        trigs = per_minute_triggers.get(minute, [])
        for (signal_side, signal_strike, signal_symbol, c_px, is_rev) in trigs:
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
                        "entry": entry_px, "sl": entry_px - SL_POINTS, "tgt": entry_px + TP_POINTS,
                        "entry_min": minute, "last_px": entry_px, "duration_min": 0, "is_rev": is_rev
                    }
                    break

    return trades


def run_tf_backtest(tf_minutes: int, max_lookback: int):
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    tasks = []
    for i in range(len(days)):
        day = days[i]
        curr_file = str(files[day])
        prev_file = str(files[days[i-1]]) if i > 0 else ""
        tasks.append((day, curr_file, prev_file, tf_minutes, max_lookback))

    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,)) as pool:
        results = pool.map(process_single_day_tf, tasks)
        for res in results:
            all_trades.extend(res)

    return all_trades


def main():
    print("Running Multi-Timeframe Comparison (1m vs 2m vs 3m vs 5m) for 2020-2024...", flush=True)

    tf_configs = [
        ("1-Minute Timeframe", 1, 10),
        ("2-Minute Timeframe", 2, 5),
        ("3-Minute Timeframe", 3, 4),
        ("5-Minute Timeframe", 5, 3),
    ]

    results_map = {}
    for label, tf_m, lookback in tf_configs:
        t_list = run_tf_backtest(tf_m, lookback)
        results_map[label] = summarize(t_list)

    print("\n" + "=" * 115)
    print("5-YEAR TIMEFRAME COMPARISON SUMMARY (2020 - 2024)")
    print("=" * 115)
    print(f"{'TIMEFRAME':25s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'PROFIT (Rs)':14s} | {'PROFIT FACTOR'}")
    print("-" * 115)
    for label, st in results_map.items():
        pf_str = f"{st['pf']:.2f}" if st['pf'] != float("inf") else "INF"
        print(f"{label:25s} | {st['trades']:7d} | {st['wr']:8.1f}% | {st['pts']:+10.2f} | Rs {st['rs']:+12,d} | {pf_str:>13s}")


if __name__ == "__main__":
    main()
