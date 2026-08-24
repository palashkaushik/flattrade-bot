"""Multi-Timeframe SL/TP Grid Search Engine (1m, 2m, 3m, 5m, Dual 1m+2m).

Sweeps SL/TP parameters:
  SL: [6.0, 8.0, 10.0, 12.0]
  TP: [12.0, 15.0, 18.0, 20.0, 25.0, 30.0]
Across 1m, 2m, 3m, 5m, and Dual (1m+2m) scanning engines over 2020-2024.
"""

import sys
import time
from pathlib import Path
from itertools import product
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np

from backtest_5y_optimized import load_spot, option_files, init_worker, SYM_RE, to_minutes, latest_spot, TimeframeTracker, summarize, CE_OFFSET, PE_OFFSET, SESSION_START, SESSION_END, DAY_LAST, DAILY_SHUTDOWN_LOSS_RS, DAILY_SHUTDOWN_PROFIT_RS, CONSECUTIVE_LOSS_LIMIT, LOT_SIZE
from flattrade_bot.indicators.patterns import Candle

GLOBAL_SPOT = {}

def init_worker_local(spot_dict):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot_dict

def process_single_day_custom(args):
    day, file_path_str, prev_file_path_str, tf_minutes, max_lookback, sl_pts, tp_pts = args
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
                        "entry": entry_px, "sl": entry_px - sl_pts, "tgt": entry_px + tp_pts,
                        "entry_min": minute, "last_px": entry_px, "duration_min": 0, "is_rev": is_rev
                    }
                    break

    return trades


def run_sltp_grid_for_tf(tf_name, tf_minutes, max_lookback, day_tasks, spot_all, sl_range, tp_range):
    results = []
    print(f"\n--- GRID SEARCH FOR {tf_name.upper()} ---")

    for sl, tp in product(sl_range, tp_range):
        tasks = [(t[0], t[1], t[2], tf_minutes, max_lookback, sl, tp) for t in day_tasks]
        all_trades = []
        with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,)) as pool:
            res_list = pool.map(process_single_day_custom, tasks)
            for r in res_list:
                all_trades.extend(r)

        st = summarize(all_trades)
        rr = round(tp / sl, 2)
        results.append({
            "tf": tf_name, "sl": sl, "tp": tp, "rr": rr,
            "trades": st["trades"], "wr": st["wr"], "pts": st["pts"], "rs": st["rs"], "pf": st["pf"]
        })
        print(f"  {tf_name:10s} | SL: {sl:4.1f} | TP: {tp:4.1f} (R:R 1:{rr:<4.2f}) -> Net Profit: Rs {st['rs']:+9,d} | WR: {st['wr']:5.1f}% | PF: {st['pf']:.2f}")

    return results


def main():
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    day_tasks = []
    for i in range(len(days)):
        day = days[i]
        curr_file = str(files[day])
        prev_file = str(files[days[i-1]]) if i > 0 else ""
        day_tasks.append((day, curr_file, prev_file))

    sl_range = [6.0, 8.0, 10.0, 12.0]
    tp_range = [12.0, 15.0, 18.0, 20.0, 25.0, 30.0]

    tfs = [
        ("1-Minute", 1, 10),
        ("2-Minute", 2, 5),
        ("3-Minute", 3, 4),
        ("5-Minute", 5, 3),
    ]

    all_results = []
    t0 = time.time()

    for name, tf_m, lookback in tfs:
        res = run_sltp_grid_for_tf(name, tf_m, lookback, day_tasks, spot_all, sl_range, tp_range)
        all_results.extend(res)

    elapsed = time.time() - t0
    print(f"\n[OK] COMPLETED MULTI-TIMEFRAME GRID SEARCH IN {elapsed:.2f} SECONDS!")

    df_all = pd.DataFrame(all_results)
    
    print("\n" + "=" * 125)
    print("BEST SL / TP COMBINATION FOR EACH TIMEFRAME RANKED BY NET PROFIT")
    print("=" * 125)
    print(f"{'TIMEFRAME':12s} | {'SL (pts)':8s} | {'TP (pts)':8s} | {'R:R RATIO':9s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'PROFIT (Rs)':14s} | {'PROFIT FACTOR'}")
    print("-" * 125)

    for tf_name, g in df_all.groupby("tf"):
        best_row = g.sort_values("rs", ascending=False).iloc[0]
        pf_str = f"{best_row['pf']:.2f}" if best_row['pf'] != float("inf") else "INF"
        print(f"{best_row['tf']:12s} | {best_row['sl']:8.1f} | {best_row['tp']:8.1f} | 1:{best_row['rr']:<7.2f} | {int(best_row['trades']):7d} | {best_row['wr']:8.1f}% | {best_row['pts']:+10.2f} | Rs {int(best_row['rs']):+12,d} | {pf_str:>13s}")

if __name__ == "__main__":
    main()
