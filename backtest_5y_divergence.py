"""5-Year & Multi-Year Backtest Engine for Quad Rotation Options Strategy.

Rules:
  - Chart: 1m CE & PE Option Charts.
  - Strikes: Nifty Spot resolves ITM2 options (CE = ATM-100, PE = ATM+100).
  - Setup: Quad Flag (S4 >= 80, S1 <= 20) or SuperSignal (all 4 <= 20)
           AND Bullish Trough Divergence (Price T2 < T1, S1 T2 > T1).
  - Vicinity Pin Bar Breakout: Any Bullish Pin Bar candle in the last 10 candles (1m) or 5 candles (2m)
                               broken and closed above by a subsequent candle.
  - Embedded S4 Trade Reversal Rule:
      - If S4 <= 20.0 for >25 consecutive bars, Reverse Mode activates.
      - Instead of buying signaled side (e.g. CE), buy the OPPOSITE side ITM2 option contract (e.g. PE).
      - Resets when S4 moves above 20.0.
  - Exits: SL: 10 pts | TP: 15 pts | Bearish Peak Reversal Exit | EOD 15:00.
  - Risk Control: Session 09:20 to 15:00 | Shutdown at -2,000 Rs or 6 consecutive losses.
"""

import argparse
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930  # 09:20 to 15:00
DAILY_SHUTDOWN_RS = 2000.0
CONSECUTIVE_LOSS_LIMIT = 6


def to_minutes(t: str) -> int:
    h, m, s = t.split(":")
    return int(h) * 60 + int(m)


def load_spot():
    print("Loading Nifty 50 spot data...", flush=True)
    df = pd.read_csv(SPOT_PATH, parse_dates=["date"])
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
            result[day] = path
    return result


def load_day_options(path: Path):
    df = pd.read_csv(path, usecols=["date", "time", "symbol", "open", "high", "low", "close"])
    if df.empty:
        return None, {}, None

    df["min"] = np.array([to_minutes(t) for t in df["time"]])
    groups = {sym: g for sym, g in df.groupby("symbol")}
    first_sym = next(iter(groups.keys()))
    m = SYM_RE.match(first_sym)
    prefix = m.group(1) if m else None
    return df, groups, prefix


def make_slice(groups, symbol):
    if symbol not in groups:
        return None
    g = groups[symbol]
    return {
        "min": g["min"].to_numpy(),
        "open": g["open"].to_numpy(),
        "high": g["high"].to_numpy(),
        "low": g["low"].to_numpy(),
        "close": g["close"].to_numpy(),
    }


def latest_spot(spot, minute):
    idx = np.searchsorted(spot["min"], minute, side="right") - 1
    return None if idx < 0 else float(spot["close"][idx])


def bar_at(option_slice, minute):
    if option_slice is None:
        return None
    idx = np.searchsorted(option_slice["min"], minute)
    if idx < len(option_slice["min"]) and option_slice["min"][idx] == minute:
        return (
            option_slice["open"][idx],
            option_slice["high"][idx],
            option_slice["low"][idx],
            option_slice["close"][idx],
        )
    return None


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


class SymbolTracker1M:
    def __init__(self):
        self.stoch = QuadStochastics()
        self.divergence = DivergenceEngine()
        self.history_1m: List[Candle] = []
        self.history_2m: List[Candle] = []
        self.buf_1m = []
        self.setup_active = False
        self.prev_s1 = None
        self.s4_embedded_count = 0

    def push_1m(self, candle_1m: Candle) -> Tuple[bool, bool, float]:
        self.history_1m.append(candle_1m)
        if len(self.history_1m) > 50:
            self.history_1m.pop(0)

        stoch_vals = self.stoch.push(candle_1m.high, candle_1m.low, candle_1m.close)
        s1, s2, s3, s4 = (stoch_vals.get(k) for k in ("s1d", "s2d", "s3d", "s4d"))
        self.prev_s1 = s1

        # Embedded S4 counter (<= 20.0)
        if s4 is not None:
            if s4 <= 20.0:
                self.s4_embedded_count += 1
            else:
                self.s4_embedded_count = 0

        is_reverse_mode = self.s4_embedded_count > 25

        self.divergence.update(candle_1m.close, s1)
        has_bull_div = self.divergence.has_bullish_trough_divergence()

        is_flag = False if any(v is None for v in (s1, s4)) else (s4 >= 79.5 and s1 <= 20.5)
        is_super = False if any(v is None for v in (s1, s2, s3, s4)) else all(v <= 20.5 for v in (s1, s2, s3, s4))

        if (is_flag or is_super) and has_bull_div:
            self.setup_active = True

        trig_1m = False
        trig_2m = False
        px_2m = 0.0

        if self.setup_active:
            if BullishPinBarDetector.check_vicinity_breakout(self.history_1m, max_lookback=10):
                trig_1m = True
                self.setup_active = False

        self.buf_1m.append(candle_1m)
        if len(self.buf_1m) == 2:
            c1, c2 = self.buf_1m
            candle_2m = Candle(
                open=c1.open, high=max(c1.high, c2.high), low=min(c1.low, c2.low),
                close=c2.close, minute=c2.minute
            )
            self.buf_1m = []
            self.history_2m.append(candle_2m)
            if len(self.history_2m) > 30:
                self.history_2m.pop(0)

            if self.setup_active:
                if BullishPinBarDetector.check_vicinity_breakout(self.history_2m, max_lookback=5):
                    trig_2m = True
                    px_2m = c2.close
                    self.setup_active = False

        return trig_1m, is_reverse_mode, px_2m


def run_day(day, spot, option_record, sym_trackers=None):
    df, groups, prefix = option_record
    if prefix is None:
        return [], sym_trackers or {}

    if sym_trackers is None:
        sym_trackers = {}

    slices = {}
    trades = []
    pos = None
    daily_pnl = 0.0
    consecutive_losses = 0
    shutdown = False

    def get_slice(side, minute):
        spot_px = latest_spot(spot, minute)
        if spot_px is None:
            return None
        atm = int(round(spot_px / 50) * 50)
        strike = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        symbol = f"{prefix}{strike}{side}"
        if symbol not in slices:
            slices[symbol] = make_slice(groups, symbol)
        current = slices[symbol]
        return (symbol, current, strike) if current is not None else None

    # Pre-evaluate triggers for the day
    per_minute_triggers = {}
    for symbol in groups:
        if symbol not in sym_trackers:
            sym_trackers[symbol] = SymbolTracker1M()
        tracker = sym_trackers[symbol]
        g = groups[symbol]

        mins = g["min"].to_numpy()
        opens = g["open"].to_numpy()
        highs = g["high"].to_numpy()
        lows = g["low"].to_numpy()
        closes = g["close"].to_numpy()

        for i in range(len(mins)):
            m = mins[i]
            c1m = Candle(open=opens[i], high=highs[i], low=lows[i], close=closes[i], minute=m)
            t1, is_rev, _ = tracker.push_1m(c1m)

            m_match = SYM_RE.match(symbol)
            if m_match:
                strike_val = int(m_match.group(2))
                side_val = m_match.group(3)
                if t1:
                    per_minute_triggers.setdefault(m, []).append((side_val, strike_val, symbol, closes[i], is_rev))

    for minute in range(SESSION_START, DAY_LAST + 1):
        # 1. Update position exits
        if pos is not None:
            held = bar_at(pos["slice"], minute)
            if held is not None:
                o_px, h_px, l_px, c_px = held
                pos["last_px"] = float(c_px)
                pos["duration_min"] += 1

                if daily_pnl + (c_px - pos["entry"]) * LOT_SIZE <= -DAILY_SHUTDOWN_RS:
                    pts = round(c_px - pos["entry"], 2)
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": c_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": "SHUTDOWN", "duration_min": pos["duration_min"], "is_rev": pos["is_rev"]
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
                    tracker = sym_trackers.get(pos["symbol"])
                    if tracker:
                        tracker.divergence.update(c_px, tracker.prev_s1)
                        if tracker.divergence.has_bearish_peak_divergence():
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
                    if consecutive_losses >= CONSECUTIVE_LOSS_LIMIT or daily_pnl <= -DAILY_SHUTDOWN_RS:
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

        # 2. Evaluate Signals
        trigs = per_minute_triggers.get(minute, [])
        for (signal_side, signal_strike, signal_symbol, c_px, is_rev) in trigs:
            active_info = get_slice(signal_side, minute)
            if active_info and active_info[2] == signal_strike and pos is None:
                if is_rev:
                    actual_side = "PE" if signal_side == "CE" else "CE"
                    actual_info = get_slice(actual_side, minute)
                    if actual_info is None:
                        continue
                    actual_symbol, actual_slice, _ = actual_info
                else:
                    actual_side = signal_side
                    actual_symbol = signal_symbol
                    actual_slice = active_info[1]

                bar = bar_at(actual_slice, minute)
                if bar is not None:
                    entry_px = float(bar[3])
                    pos = {
                        "side": actual_side, "symbol": actual_symbol, "slice": actual_slice,
                        "entry": entry_px, "sl": entry_px - SL_POINTS, "tgt": entry_px + TP_POINTS,
                        "entry_min": minute, "last_px": entry_px, "duration_min": 0, "is_rev": is_rev
                    }
                    break

    return trades, sym_trackers


def run_backtest(start_date: str, end_date: str):
    spot_all = load_spot()
    files = option_files(start_date, end_date)
    days = sorted(set(files.keys()) & set(spot_all.keys()))
    print(f"Running backtest across {len(days)} trading days ({start_date} to {end_date})...", flush=True)

    all_trades = []
    sym_trackers = {}
    t0 = time.time()

    for idx, day in enumerate(days, 1):
        spot = spot_all[day]
        option_record = load_day_options(files[day])
        trades, sym_trackers = run_day(day, spot, option_record, sym_trackers=sym_trackers)
        all_trades.extend(trades)

        if idx % 100 == 0 or idx == len(days):
            elapsed = time.time() - t0
            stats = summarize(all_trades)
            print(f"[{idx:4d}/{len(days)}] Trades: {stats['trades']:4d} | Win Rate: {stats['wr']:5.1f}% | Net P&L: {stats['pts']:+8.2f} pts (Rs {stats['rs']:+10,d}) | Elapsed: {elapsed:.1f}s", flush=True)

    return all_trades, len(days)


def print_yearly_breakdown(trades):
    if not trades:
        print("\nNo trades executed.")
        return

    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["date"]).dt.year

    print("\n" + "=" * 110)
    print(f"{'YEAR':6s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'PROFIT (Rs)':14s} | {'PROFIT FACTOR':13s}")
    print("=" * 110)

    for year, g in df.groupby("year"):
        t_list = g.to_dict("records")
        st = summarize(t_list)
        pf_str = f"{st['pf']:.2f}" if st['pf'] != float("inf") else "INF"
        print(f"{year:6d} | {st['trades']:7d} | {st['wr']:8.1f}% | {st['pts']:+10.2f} | Rs {st['rs']:+12,d} | {pf_str:>13s}")

    st_all = summarize(trades)
    pf_all = f"{st_all['pf']:.2f}" if st_all['pf'] != float("inf") else "INF"
    print("=" * 110)
    print(f"{'TOTAL':6s} | {st_all['trades']:7d} | {st_all['wr']:8.1f}% | {st_all['pts']:+10.2f} | Rs {st_all['rs']:+12,d} | {pf_all:>13s}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Year Backtest Engine for Quad Rotation Strategy")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    trades, total_days = run_backtest(args.start, args.end)
    print_yearly_breakdown(trades)


if __name__ == "__main__":
    main()
