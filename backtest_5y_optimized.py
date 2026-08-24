"""Ultra-Optimized 5-Year Backtest Engine for Quad Rotation Options Strategy.

Rules:
  - Dual-Timeframe (1m + 2m) Independent Tracking.
  - Setup: Quad Flag (S4 >= 80, S1 <= 20) OR SuperSignal (all 4 <= 20) + Bullish Trough Divergence.
  - Trigger: Vicinity Pin Bar Breakout (10 1m candles lookback / 5 2m candles lookback).
  - Embedded S4 Reversal Rule:
      * When S4 <= 20.0 for > 25 bars:
      * SuperSignal setups ARE REVERSED (buy opposite ITM2 side).
      * Quad Flag setups REMAIN NORMAL (do not reverse).
  - Daily Max Profit Switch: Stop trading for the day at +Rs 1,950 (+30 pts / 2 winning trades).
  - Daily Max Loss Switch: Stop trading for the day at -Rs 2,000 (-30.7 pts) or 6 consecutive losses.
  - Exits: SL: 10 pts | TP: 15 pts | Bearish Peak Reversal Exit | EOD 15:00.
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

from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.divergence import DivergenceEngine

DATA_DIR = Path("C:/Websites/ammu")
OPTS_DIR = DATA_DIR / "nifty_options"
SPOT_PATH = DATA_DIR / "index" / "NIFTY 50_minute.csv"

SYM_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")
LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SL_POINTS, TP_POINTS = 10.0, 15.0
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_SHUTDOWN_LOSS_RS = 2000.0
DAILY_SHUTDOWN_PROFIT_RS = 1950.0  # 2 winning trades at 1 lot
CONSECUTIVE_LOSS_LIMIT = 6

GLOBAL_SPOT = {}


def to_minutes(t: str) -> int:
    h, m, s = t.split(":")
    return int(h) * 60 + int(m)


def load_spot():
    df = pd.read_csv(SPOT_PATH, parse_dates=["date"], engine="c")
    df = df.sort_values("date").reset_index(drop=True)
    df["day"] = df["date"].dt.strftime("%Y-%m-%d")
    df["min"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    out = {}
    for day, g in df.groupby("day"):
        out[day] = {
            "min": g["min"].to_numpy(),
            "close": g["close"].to_numpy(),
        }
    return out


def init_worker(spot_dict):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot_dict


def option_files(start_date: str, end_date: str):
    files = sorted(
        OPTS_DIR.rglob("*.csv"),
        key=lambda p: (
            int(p.parent.parent.name), int(p.parent.name),
            int(p.stem.split("_")[2]), int(p.stem.split("_")[3]),
        ),
    )
    result = {}
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    for path in files:
        parts = path.stem.split("_")
        day = f"{parts[4]}-{parts[3]}-{parts[2]}"
        if start_ts <= pd.Timestamp(day) <= end_ts:
            result[str(day)] = str(path)
    return result


def latest_spot(spot, minute):
    if spot is None:
        return None
    idx = np.searchsorted(spot["min"], minute, side="right") - 1
    return None if idx < 0 else float(spot["close"][idx])


class TimeframeTracker:
    def __init__(self, tf_label: str, max_lookback: int):
        self.tf_label = tf_label
        self.max_lookback = max_lookback
        self.stoch = QuadStochastics()
        self.divergence = DivergenceEngine()
        self.history: List[Candle] = []
        self.setup_active = False
        self.setup_type = ""  # "super" or "flag"
        self.prev_s1 = None
        self.s4_embedded_count = 0

    def push(self, candle: Candle) -> Tuple[bool, bool, str, float]:
        """Pushes candle, returns (triggered, is_reverse, setup_type, trigger_close)."""
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

        # Reverse trade ONLY if setup_type is super AND S4 is embedded (>25 bars)
        is_reverse_mode = embedded_active and (self.setup_type == "super")

        triggered = False
        if self.setup_active:
            if BullishPinBarDetector.check_vicinity_breakout(self.history, self.max_lookback):
                triggered = True
                self.setup_active = False

        return triggered, is_reverse_mode, self.setup_type, candle.close


class DualTimeframeSymbolTracker:
    def __init__(self):
        self.tf1m = TimeframeTracker("1m", max_lookback=10)
        self.tf2m = TimeframeTracker("2m", max_lookback=5)
        self.buf_1m = []


def process_single_day(args):
    day, file_path_str, prev_file_path_str = args
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

    dtf_trackers = {}
    
    # Warmup prev day
    for sym, g in groups_prev.items():
        dtf_trackers[sym] = DualTimeframeSymbolTracker()
        mins = g["min"].to_numpy()
        opens = g["open"].to_numpy()
        highs = g["high"].to_numpy()
        lows = g["low"].to_numpy()
        closes = g["close"].to_numpy()

        for i in range(len(mins)):
            m = mins[i]
            c1m = Candle(open=opens[i], high=highs[i], low=lows[i], close=closes[i], minute=m)
            dtf_trackers[sym].tf1m.push(c1m)

            dtf_trackers[sym].buf_1m.append(c1m)
            if len(dtf_trackers[sym].buf_1m) == 2:
                c1, c2 = dtf_trackers[sym].buf_1m
                c2m = Candle(open=c1.open, high=max(c1.high, c2.high), low=min(c1.low, c2.low), close=c2.close, minute=c2.minute)
                dtf_trackers[sym].buf_1m = []
                dtf_trackers[sym].tf2m.push(c2m)

    per_minute_triggers = {}
    slices = {}

    for sym, g in groups_curr.items():
        if sym not in dtf_trackers:
            dtf_trackers[sym] = DualTimeframeSymbolTracker()
        tracker = dtf_trackers[sym]

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

            # 1m evaluation
            t1, is_rev1, stype1, px1 = tracker.tf1m.push(c1m)
            if t1:
                per_minute_triggers.setdefault(m, []).append((side_val, strike_val, sym, px1, is_rev1, "1m"))

            # 2m evaluation
            tracker.buf_1m.append(c1m)
            if len(tracker.buf_1m) == 2:
                c1, c2 = tracker.buf_1m
                c2m = Candle(open=c1.open, high=max(c1.high, c2.high), low=min(c1.low, c2.low), close=c2.close, minute=c2.minute)
                tracker.buf_1m = []
                t2, is_rev2, stype2, px2 = tracker.tf2m.push(c2m)
                if t2:
                    per_minute_triggers.setdefault(m, []).append((side_val, strike_val, sym, px2, is_rev2, "2m"))

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
                        "reason": "SHUTDOWN_LOSS", "duration_min": pos["duration_min"], "is_rev": pos["is_rev"], "tf": pos["tf"]
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
                    dtf_tr = dtf_trackers.get(pos["symbol"])
                    if dtf_tr:
                        tracker = dtf_tr.tf1m
                        tracker.divergence.update(c_px, tracker.prev_s1)
                        if tracker.divergence.has_bearish_peak_divergence():
                            exit_px, reason = c_px, "BEARISH_PEAK_REVERSAL"

                if exit_px is not None:
                    pts = round(exit_px - pos["entry"], 2)
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": exit_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": reason, "duration_min": pos["duration_min"], "is_rev": pos["is_rev"], "tf": pos["tf"]
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
                    "reason": "EOD", "duration_min": pos["duration_min"], "is_rev": pos["is_rev"], "tf": pos["tf"]
                })
                daily_pnl += pts * LOT_SIZE
                pos = None
                break

        if pos is not None or shutdown or minute >= SESSION_END:
            continue

        trigs = per_minute_triggers.get(minute, [])
        for (signal_side, signal_strike, signal_symbol, c_px, is_rev, tf_label) in trigs:
            active_info = get_active_info(signal_side, minute)
            if active_info and active_info[2] == signal_strike and pos is None:
                # Reverse ONLY if SuperSignal + Embedded S4 (>25 bars)
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
                        "entry_min": minute, "last_px": entry_px, "duration_min": 0, "is_rev": is_rev, "tf": tf_label
                    }
                    break

    return trades


def summarize(trades):
    if not trades:
        return {"trades": 0, "wr": 0.0, "pts": 0.0, "rs": 0, "pf": 0.0}
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


def print_yearly_breakdown(trades):
    if not trades:
        print("\nNo trades executed.")
        return

    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["date"]).dt.year

    print("\n" + "=" * 115)
    print(f"{'YEAR':6s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'PROFIT (Rs)':14s} | {'PROFIT FACTOR':13s}")
    print("=" * 115)

    for year, g in df.groupby("year"):
        t_list = g.to_dict("records")
        st = summarize(t_list)
        pf_str = f"{st['pf']:.2f}" if st['pf'] != float("inf") else "INF"
        print(f"{year:6d} | {st['trades']:7d} | {st['wr']:8.1f}% | {st['pts']:+10.2f} | Rs {st['rs']:+12,d} | {pf_str:>13s}")

    st_all = summarize(trades)
    pf_all = f"{st_all['pf']:.2f}" if st_all['pf'] != float("inf") else "INF"
    print("=" * 115)
    print(f"{'TOTAL':6s} | {st_all['trades']:7d} | {st_all['wr']:8.1f}% | {st_all['pts']:+10.2f} | Rs {st_all['rs']:+12,d} | {pf_all:>13s}")


def main():
    parser = argparse.ArgumentParser(description="Ultra-Optimized 5-Year Backtest Engine")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    spot_all = load_spot()
    files = option_files(args.start, args.end)
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    print(f"Loaded {len(days)} trading days across {args.start} to {args.end}.")
    print(f"Launching Multi-Core Parallel Execution on {min(cpu_count(), 8)} worker processes...", flush=True)

    tasks = []
    for i in range(len(days)):
        day = days[i]
        curr_file = str(files[day])
        prev_file = str(files[days[i-1]]) if i > 0 else ""
        tasks.append((day, curr_file, prev_file))

    t0 = time.time()
    all_trades = []

    with Pool(processes=min(cpu_count(), 8), initializer=init_worker, initargs=(spot_all,)) as pool:
        results = pool.map(process_single_day, tasks)
        for res in results:
            all_trades.extend(res)

    elapsed = time.time() - t0
    print(f"\n[OK] COMPLETED 5-YEAR BACKTEST IN {elapsed:.2f} SECONDS ACROSS {len(days)} DAYS!")

    print_yearly_breakdown(all_trades)


if __name__ == "__main__":
    main()
