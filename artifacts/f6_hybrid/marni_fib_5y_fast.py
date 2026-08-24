"""High-Performance Causal Marni Fib / UT Bot 2020-2026 Backtest Engine."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.causal_live_parity_research import IncrementalATR

LOT_SIZE = 65
SESSION_START = 560
SESSION_END = 900
DAY_LAST = 930
CONSECUTIVE_LOSS_LIMIT = 4
UT_KEY = 1.0
UT_ATR_PERIOD = 10
ENTRY_LEVEL = 0.786
TARGET_LEVELS = (0.0, 0.29)
STOP_LEVELS = (1.079, 1.155, 1.25)
TIMEFRAME_PERIODS = {"1m": 1, "2m": 2, "3m": 3, "5m": 5}
BIAS_PERIODS = {tf: 15 for tf in TIMEFRAME_PERIODS}
SYMBOL_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")
GLOBAL_SPOT = {}


def init_worker(spot):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot
    source.GLOBAL_CACHE = {}


class UTBotState:
    """Causal translation of UT Bot Pine logic."""

    def __init__(self, key: float = 1.0, period: int = 10):
        self.key = key
        self.atr = IncrementalATR(period)
        self.trailing_stop = 0.0
        self.previous_source = None
        self.position = 0

    def update(self, candle: Candle, source_close: float | None = None) -> str:
        source_price = candle.close if source_close is None else source_close
        atr = self.atr.update(candle.high, candle.low, candle.close)
        previous_source = self.previous_source
        previous_stop = self.trailing_stop
        self.previous_source = source_price
        if atr is None or previous_source is None:
            return "blue"

        loss = self.key * atr
        if source_price > previous_stop and previous_source > previous_stop:
            self.trailing_stop = max(previous_stop, source_price - loss)
        elif source_price < previous_stop and previous_source < previous_stop:
            self.trailing_stop = min(previous_stop, source_price + loss)
        elif source_price > previous_stop:
            self.trailing_stop = source_price - loss
        else:
            self.trailing_stop = source_price + loss

        if previous_source < previous_stop and source_price > previous_stop:
            self.position = 1
        elif previous_source > previous_stop and source_price < previous_stop:
            self.position = -1
        return "green" if self.position == 1 else "red" if self.position == -1 else "blue"


class HeikinAshiState:
    def __init__(self):
        self.open = None
        self.close = None

    def update(self, candle: Candle) -> Candle:
        ha_close = (candle.open + candle.high + candle.low + candle.close) / 4.0
        ha_open = (
            (candle.open + candle.close) / 2.0
            if self.open is None
            else (self.open + self.close) / 2.0
        )
        ha_high = max(candle.high, ha_open, ha_close)
        ha_low = min(candle.low, ha_open, ha_close)
        self.open = ha_open
        self.close = ha_close
        return Candle(ha_open, ha_high, ha_low, ha_close, minute=candle.minute)


def linreg_value(values: deque[float]) -> float | None:
    if len(values) < 11:
        return None
    n = len(values)
    x_sum = n * (n - 1) / 2.0
    x2_sum = (n - 1) * n * (2 * n - 1) / 6.0
    y_sum = sum(values)
    xy_sum = sum(index * value for index, value in enumerate(values))
    denominator = n * x2_sum - x_sum * x_sum
    slope = (n * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / n
    return intercept + slope * (n - 1)


class StrictHTFBiasState:
    """Calculates 5x Higher-Timeframe Heikin-Ashi + UT Bot + LinReg(11) Bias."""

    def __init__(self, period: int):
        self.period = period
        self.buffer = []
        self.ha = HeikinAshiState()
        self.ut = UTBotState()
        self.closes = deque(maxlen=11)
        self.ha_candle = None
        self.ut_color = "blue"
        self.linreg_plot = None
        self.confirmed_minute = None

    def update_1m(self, candle: Candle):
        self.buffer.append(candle)
        if candle.minute % self.period != 0 or not self.buffer:
            return
        buf = self.buffer
        self.buffer = []
        aggregate = Candle(
            open=buf[0].open,
            high=max(i.high for i in buf),
            low=min(i.low for i in buf),
            close=buf[-1].close,
            minute=candle.minute,
        )
        self.confirmed_minute = aggregate.minute
        ha = self.ha.update(aggregate)
        self.ha_candle = ha
        self.ut_color = self.ut.update(ha)
        self.closes.append(ha.close)
        self.linreg_plot = sum(self.closes) / len(self.closes) if len(self.closes) >= 11 else None

    def snapshot(self) -> dict:
        if self.ha_candle is None or self.linreg_plot is None:
            return {"bullish": False, "bearish": False, "confirmed_minute": None}
        close = self.ha_candle.close
        open_p = self.ha_candle.open
        plot = self.linreg_plot

        bullish = (
            close > plot
            and self.ut_color == "green"
        )
        bearish = (
            close < plot
            and self.ut_color == "red"
        )
        return {
            "bullish": bullish,
            "bearish": bearish,
            "ha_open": open_p,
            "ha_close": close,
            "linreg_plot": plot,
            "ut_color": self.ut_color,
            "confirmed_minute": self.confirmed_minute,
        }


class BiasFeed:
    def __init__(self):
        self.states = {tf: StrictHTFBiasState(BIAS_PERIODS[tf]) for tf in TIMEFRAME_PERIODS}

    def warmup(self, rows):
        for r in rows:
            self.push(r)

    def push(self, row):
        c = Candle(float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), minute=int(row["minute"]))
        for state in self.states.values():
            state.update_1m(c)

    def snapshot(self, tf: str) -> dict:
        return self.states[tf].snapshot()


class FibTimeframe:
    """
    Exact Marni 3-Phase Pattern Tracker using 1m UT Bot Candle Colors:
    - Bearish (PE): 1 GREEN UT -> >= 5 Consecutive RED UT (middle) -> 1 GREEN UT (end)
    - Bullish (CE): 1 RED UT   -> >= 5 Consecutive GREEN UT (middle) -> 1 RED UT (end)
    Supports both intraday swings (anchored at 09:15 session open) and multi-day overnight swings.
    Filters out micro-chops with configurable min_span.
    """

    def __init__(self, period: int, pattern_specs=None, min_candles: int = 5, min_span: float = 15.0):
        self.period = period
        self.min_candles = min_candles
        self.min_span = min_span
        self.ut = UTBotState()
        self.buffer = []
        self.history = []  # list of (candle, ut_color, day_index)
        self.setups = []
        self.curr_day = 0

    def reset_session(self):
        self.buffer = []

    def push(self, candle: Candle, current_bias: dict | None = None, htf_eval: str = "touch"):
        self.buffer.append(candle)
        if candle.minute % self.period != 0 or not self.buffer:
            return []

        buf = self.buffer
        self.buffer = []
        aggregate = Candle(
            open=buf[0].open,
            high=max(i.high for i in buf),
            low=min(i.low for i in buf),
            close=buf[-1].close,
            minute=candle.minute,
        )
        if aggregate.minute == 555:
            self.curr_day += 1

        col = self.ut.update(aggregate)
        self.history.append((aggregate, col, self.curr_day))

        # --- Check Bearish Setup ---
        if col == "green" and len(self.history) >= self.min_candles + 2:
            red_count = 0
            k = len(self.history) - 2
            while k >= 0 and self.history[k][1] == "red":
                red_count += 1
                k -= 1
            if red_count >= self.min_candles:
                # 1. Multi-day setup if k was green on previous day
                if k >= 0 and self.history[k][1] == "green":
                    pattern_candles = [self.history[i][0] for i in range(k, len(self.history))]
                    origin_high = max(c.high for c in pattern_candles)
                    trough_low = min(c.low for c in pattern_candles)
                    span = origin_high - trough_low
                    if span >= self.min_span:
                        self.setups.append(("bearish", origin_high, trough_low, "low_to_high", current_bias or {}))

                # 2. Intraday setup starting from today's open if drop started at 09:15
                today_start_k = next((i for i in range(len(self.history) - 1, -1, -1) if self.history[i][2] == self.curr_day and self.history[i][0].minute == 555), None)
                if today_start_k is not None and today_start_k > k:
                    today_reds = [self.history[i] for i in range(today_start_k, len(self.history) - 1) if self.history[i][1] == "red"]
                    if len(today_reds) >= self.min_candles:
                        today_pattern = [self.history[i][0] for i in range(today_start_k, len(self.history))]
                        origin_high = max(c.high for c in today_pattern)
                        trough_low = min(c.low for c in today_pattern)
                        span = origin_high - trough_low
                        if span >= self.min_span:
                            self.setups.append(("bearish", origin_high, trough_low, "low_to_high", current_bias or {}))

        # --- Check Bullish Setup ---
        if col == "red" and len(self.history) >= self.min_candles + 2:
            green_count = 0
            k = len(self.history) - 2
            while k >= 0 and self.history[k][1] == "green":
                green_count += 1
                k -= 1
            if green_count >= self.min_candles:
                if k >= 0 and self.history[k][1] == "red":
                    pattern_candles = [self.history[i][0] for i in range(k, len(self.history))]
                    peak_high = max(c.high for c in pattern_candles)
                    origin_low = min(c.low for c in pattern_candles)
                    span = peak_high - origin_low
                    if span >= self.min_span:
                        self.setups.append(("bullish", peak_high, origin_low, "high_to_low", current_bias or {}))

                today_start_k = next((i for i in range(len(self.history) - 1, -1, -1) if self.history[i][2] == self.curr_day and self.history[i][0].minute == 555), None)
                if today_start_k is not None and today_start_k > k:
                    today_greens = [self.history[i] for i in range(today_start_k, len(self.history) - 1) if self.history[i][1] == "green"]
                    if len(today_greens) >= self.min_candles:
                        today_pattern = [self.history[i][0] for i in range(today_start_k, len(self.history))]
                        peak_high = max(c.high for c in today_pattern)
                        origin_low = min(c.low for c in today_pattern)
                        span = peak_high - origin_low
                        if span >= self.min_span:
                            self.setups.append(("bullish", peak_high, origin_low, "high_to_low", current_bias or {}))

        # --- Check for 0.786 Touches on Active Setups ---
        events = []
        valid_setups = []
        for direction, high, low, orientation, bias_creation in self.setups:
            span = high - low
            if span < 5.0:
                continue
            if orientation == "high_to_low" and aggregate.low < (low - 0.25 * span):
                continue
            if orientation == "low_to_high" and aggregate.high > (high + 0.25 * span):
                continue
            valid_setups.append((direction, high, low, orientation, bias_creation))

        rem = []
        for direction, high, low, orientation, bias_creation in valid_setups:
            span = high - low
            entry_level = (
                high - ENTRY_LEVEL * span
                if orientation == "high_to_low"
                else low + ENTRY_LEVEL * span
            )
            if aggregate.high >= entry_level - 1.0 and aggregate.low <= entry_level + 1.0:
                side = "CE" if direction == "bullish" else "PE"
                allowed = (current_bias or {}).get("bullish", False) if side == "CE" else (current_bias or {}).get("bearish", False)
                if allowed:
                    events.append({
                        "minute": aggregate.minute,
                        "entry_level": entry_level,
                        "entry_price": aggregate.close,
                        "fib_high": high,
                        "fib_low": low,
                        "direction": direction,
                        "orientation": orientation,
                        "timeframe": f"{self.period}m",
                    })
                    continue
            rem.append((direction, high, low, orientation, bias_creation))

        self.setups = rem
        return events


def fib_price(high: float, low: float, level: float, orientation: str) -> float:
    span = high - low
    if orientation == "high_to_low":
        return high - level * span
    return low + level * span


def spot_row(spot, index):
    return {
        "open": float(spot["open"][index]),
        "high": float(spot["high"][index]),
        "low": float(spot["low"][index]),
        "close": float(spot["close"][index]),
        "minute": int(spot["min"][index]),
    }


def active_strikes(spot, minute: int, side: str) -> int:
    idx = list(spot["min"]).index(minute) if minute in spot["min"] else 0
    ref_price = spot["open"][idx]
    atm = int(round(ref_price / 50.0) * 50)
    return atm - 50 if side == "CE" else atm + 50


def simulate(events, bars, index_bars, spot, timeframe_mode, target_level, stop_level, include_fees=True, mode="index"):
    allowed = None if timeframe_mode == "combined" else timeframe_mode
    events_by_minute = defaultdict(list)
    for event in events:
        if allowed is None or event["timeframe"] == allowed:
            events_by_minute[event["minute"]].append(event)

    position = None
    stopped = False
    consecutive_losses = 0
    trades = []
    timeline = [int(m) for m in spot["min"] if SESSION_START <= m <= DAY_LAST]

    for minute in timeline:
        if position is not None and minute > position["entry_min"]:
            row = bars[position["key"]].get(minute)
            if row:
                stop = fib_price(position["fib_high"], position["fib_low"], stop_level, position["orientation"])
                target = fib_price(position["fib_high"], position["fib_low"], target_level, position["orientation"])

                if mode == "index":
                    price_row = index_bars[minute]
                    if position["side"] == "CE":
                        hit_stop = price_row["low"] <= stop
                        hit_target = price_row["high"] >= target
                    else:
                        hit_stop = price_row["high"] >= stop
                        hit_target = price_row["low"] <= target
                else:
                    price_row = row
                    hit_stop = price_row["low"] <= stop
                    hit_target = price_row["high"] >= target

                reason = "SL" if hit_stop else "TP" if hit_target else None
                if minute >= SESSION_END and reason is None:
                    reason = "EOD"

                if reason:
                    slip = SLIPPAGE_PTS if include_fees else 0.0
                    brokerage = BROKERAGE_PER_ORDER if include_fees else 0.0
                    entry_fill = position["option_entry"] + slip
                    exit_fill = row["close"] - slip
                    points = round(exit_fill - entry_fill, 2)
                    fee = trade_cost(entry_fill, exit_fill, brokerage) if include_fees else 0.0
                    net_rs = round(points * LOT_SIZE - fee, 2)

                    trades.append({
                        "entry_min": position["entry_min"],
                        "exit_min": minute,
                        "side": position["side"],
                        "symbol": position["symbol"],
                        "timeframe": position["timeframe"],
                        "entry": entry_fill,
                        "exit": exit_fill,
                        "reason": reason,
                        "points": points,
                        "rs_net": net_rs,
                        "fee": fee,
                    })
                    consecutive_losses = consecutive_losses + 1 if net_rs <= 0 else 0
                    stopped = consecutive_losses >= CONSECUTIVE_LOSS_LIMIT
                    position = None

        if position is not None or stopped or minute >= SESSION_END:
            continue
        for event in events_by_minute.get(minute, []):
            if position is not None:
                break
            position = {
                **event,
                "entry_min": minute,
                "key": (event["side"], event["strike"]),
            }
    return trades


def option_rows(frame, groups, symbol):
    indexes = groups.get(symbol)
    if indexes is None:
        return []
    rows = frame.iloc[indexes].sort_values("time")
    return [
        {
            "time": row["time"],
            "minute": int(str(row["time"]).split(":")[0]) * 60 + int(str(row["time"]).split(":")[1]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for _, row in rows.iterrows()
    ]


def load_day_symbols(day, current_path, previous_path, spot):
    current = source.cached_option(str(current_path))
    previous = source.cached_option(str(previous_path)) if previous_path else None
    if current is None:
        return [], {}
    frame, groups, prefix = current
    previous_rows = {}
    if previous is not None:
        previous_rows = {
            symbol: option_rows(previous[0], previous[1], symbol)
            for symbol in previous[1]
        }
    current_rows = {
        symbol: option_rows(frame, groups, symbol)
        for symbol in groups
    }
    active_keys = set()
    for index, minute in enumerate(spot["min"]):
        if SESSION_START <= minute <= DAY_LAST:
            atm = int(round(float(spot["close"][index]) / 50.0) * 50)
            active_keys.add(("CE", atm - 50))
            active_keys.add(("PE", atm + 50))

    symbols = []
    rows_by_symbol = {}
    for symbol in current_rows:
        match = SYMBOL_RE.match(symbol)
        if not match:
            continue
        key = (match.group(3), int(match.group(2)))
        if key not in active_keys:
            continue
        symbols.append(key)
        rows_by_symbol[key] = {
            "symbol": symbol,
            "previous": previous_rows.get(symbol, []),
            "current": current_rows[symbol],
        }
    return sorted(symbols), rows_by_symbol


def process_day(args):
    day, current_path, previous_path, timeframe_modes, target_levels, stop_levels, bias_mode, include_fees, htf_eval, min_span = args
    spot = GLOBAL_SPOT[day]
    previous_days = sorted(key for key in GLOBAL_SPOT if key < day)
    symbols, symbol_rows = load_day_symbols(day, current_path, previous_path, spot)
    events = []
    bars = {}
    symbol_by_key = {}
    for side, strike in symbols:
        record = symbol_rows[(side, strike)]
        symbol_by_key[(side, strike)] = record["symbol"]
        bars[(side, strike)] = {row["minute"]: row for row in record["current"]}

    if bias_mode == "index":
        index_feed = {
            tf: FibTimeframe(p, min_span=min_span)
            for tf, p in TIMEFRAME_PERIODS.items()
        }
        index_bias = BiasFeed()
        if previous_days:
            previous_spot = GLOBAL_SPOT[previous_days[-1]]
            prows = [spot_row(previous_spot, i) for i in range(len(previous_spot["min"]))]
            index_bias.warmup(prows)
            for prow in prows:
                c = Candle(prow["open"], prow["high"], prow["low"], prow["close"], minute=prow["minute"])
                for tf, tf_inst in index_feed.items():
                    bias = index_bias.snapshot(tf)
                    tf_inst.push(c, bias, htf_eval)

        for idx in range(len(spot["min"])):
            row = spot_row(spot, idx)
            index_bias.push(row)
            c = Candle(row["open"], row["high"], row["low"], row["close"], minute=row["minute"])
            m = row["minute"]
            for tf, tf_inst in index_feed.items():
                bias = index_bias.snapshot(tf)
                for ev in tf_inst.push(c, bias, htf_eval):
                    side = "CE" if ev["direction"] == "bullish" else "PE"
                    strike = active_strikes(spot, m, side)
                    key = (side, strike)
                    if key not in bars or m not in bars[key]:
                        continue
                    events.append({
                        **ev,
                        "timeframe": tf,
                        "side": side,
                        "strike": strike,
                        "symbol": symbol_by_key[key],
                        "minute": m,
                        "option_entry": bars[key][m]["close"],
                    })
    else:
        for key in symbols:
            side, strike = key
            record = symbol_rows[key]
            opt_feed = {
                tf: FibTimeframe(p, min_span=min_span)
                for tf, p in TIMEFRAME_PERIODS.items()
            }
            opt_bias = BiasFeed()
            opt_bias.warmup(record["previous"])
            for prow in record["previous"]:
                c = Candle(prow["open"], prow["high"], prow["low"], prow["close"], minute=prow["minute"])
                for tf, tf_inst in opt_feed.items():
                    bias = opt_bias.snapshot(tf)
                    tf_inst.push(c, bias, htf_eval)

            for row in record["current"]:
                opt_bias.push(row)
                c = Candle(row["open"], row["high"], row["low"], row["close"], minute=row["minute"])
                m = row["minute"]
                if active_strikes(spot, m, side) != strike:
                    continue
                for tf, tf_inst in opt_feed.items():
                    bias = opt_bias.snapshot(tf)
                    for ev in tf_inst.push(c, bias, htf_eval):
                        events.append({
                            **ev,
                            "timeframe": tf,
                            "side": side,
                            "strike": strike,
                            "symbol": symbol_by_key[key],
                            "minute": m,
                            "option_entry": row["close"],
                        })

    index_bars = {int(spot["min"][i]): spot_row(spot, i) for i in range(len(spot["min"]))}
    output = {}
    for tf in timeframe_modes:
        for tp in target_levels:
            for sl in stop_levels:
                key = f"{tf}|tp{tp}|sl{sl}"
                trades = simulate(events, bars, index_bars, spot, tf, tp, sl, include_fees=include_fees, mode=bias_mode)
                for t in trades:
                    t["date"] = day
                output[key] = trades
    return output


def compute_stats(trades: list[dict], days_count: int) -> dict:
    wins = [t for t in trades if t["rs_net"] > 0]
    losses = [t for t in trades if t["rs_net"] <= 0]
    loss_total = abs(sum(t["rs_net"] for t in losses))
    win_total = sum(t["rs_net"] for t in wins)
    net_rs = sum(t["rs_net"] for t in trades)
    net_pts = sum(t["points"] for t in trades)
    wr = round(len(wins) / len(trades) * 100, 2) if trades else 0.0
    pf = round(win_total / loss_total, 4) if loss_total else (float("inf") if win_total else 0.0)
    fees = round(sum(t["fee"] for t in trades), 2)
    avg_trades = round(len(trades) / days_count, 3) if days_count else 0.0

    # Max Drawdown calculation
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: (x["date"], x["entry_min"])):
        equity += t["rs_net"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "net_rs": round(net_rs, 2),
        "net_points": round(net_pts, 2),
        "profit_factor": pf,
        "max_drawdown_rs": round(max_dd, 2),
        "fees_rs": fees,
        "avg_trades_per_day": avg_trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Year Causal Marni Fib Backtest")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers (default: 8)")
    parser.add_argument("--smoke", action="store_true", help="Run 5-day smoke test only")
    parser.add_argument("--no-fees", action="store_true", help="Disable fees & slippage")
    parser.add_argument("--bias-mode", default="index", choices=["index", "option"])
    parser.add_argument("--htf-eval", default="touch", choices=["touch", "prior"], help="Evaluate HTF at 0.786 touch (default: touch)")
    parser.add_argument("--min-span", type=float, default=15.0, help="Minimum impulse span in points (default: 15.0)")
    parser.add_argument("--output", default="artifacts/f6_hybrid/marni_fib_5y_fast.json")
    args = parser.parse_args()

    include_fees = not args.no_fees
    print(f"=== MARNI FIB 2020-2026 BACKTEST ENGINE ===")
    print(f"Date Range: {args.start} to {args.end} | Min Span: {args.min_span} pts")
    print(f"Mode: {args.bias_mode.upper()} | HTF Eval: {args.htf_eval.upper()} | Include Fees: {include_fees} | Workers: {args.workers}")

    spot_all = source.load_spot()
    opt_map = source.option_day_files(args.start, args.end)
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days[:5] if args.smoke else all_days
    print(f"Running on {len(days)} trading days {'(SMOKE TEST - 5 DAYS ONLY)' if args.smoke else ''}...")

    previous = {day: max((c for c in all_days if c < day), default="") for day in days}
    timeframe_modes = ("1m", "2m", "3m", "5m", "combined")
    target_levels = TARGET_LEVELS
    stop_levels = STOP_LEVELS

    tasks = [
        (
            day,
            opt_map[day],
            opt_map.get(previous[day], ""),
            timeframe_modes,
            target_levels,
            stop_levels,
            args.bias_mode,
            include_fees,
            args.htf_eval,
            args.min_span,
        )
        for day in days
    ]

    t0 = time.time()
    aggregated = defaultdict(list)

    if args.smoke or args.workers == 1:
        init_worker(spot_all)
        for task in tasks:
            res = process_day(task)
            for k, v in res.items():
                aggregated[k].extend(v)
    else:
        with Pool(processes=args.workers, initializer=init_worker, initargs=(spot_all,)) as pool:
            for res in pool.imap_unordered(process_day, tasks, chunksize=1):
                for k, v in res.items():
                    aggregated[k].extend(v)

    elapsed = time.time() - t0
    print(f"\nExecution finished in {elapsed:.2f} seconds ({len(days)/elapsed:.1f} days/sec).")

    summary = {}
    print(f"\n{'='*95}")
    print(f"SUMMARY RESULTS (Mode: {args.bias_mode.upper()} | Fees: {'YES' if include_fees else 'NO'})")
    print(f"{'='*95}")
    print(f"{'Configuration':25s} | {'Trades':6s} | {'WinRate':7s} | {'Net Rs':12s} | {'Net Pts':8s} | {'PF':6s} | {'MaxDD Rs':10s} | {'Fees Rs':8s}")
    print(f"{'-'*95}")

    for k in sorted(aggregated.keys()):
        trades = aggregated[k]
        st = compute_stats(trades, len(days))
        summary[k] = {"stats": st, "trades": trades}
        print(f"{k:25s} | {st['trades']:6d} | {st['win_rate']:6.1f}% | {st['net_rs']:+12,.2f} | {st['net_points']:+8.2f} | {st['profit_factor']:6.2f} | {st['max_drawdown_rs']:10,.2f} | {st['fees_rs']:8,.2f}")

    # Year-by-Year breakdown for combined tp0.0 sl1.079
    best_key = "combined|tp0.0|sl1.079" if "combined|tp0.0|sl1.079" in aggregated else list(aggregated.keys())[0]
    best_trades = aggregated[best_key]
    by_year = defaultdict(list)
    for t in best_trades:
        by_year[t["date"][:4]].append(t)

    print(f"\n{'='*75}")
    print(f"YEAR-BY-YEAR BREAKDOWN FOR: {best_key}")
    print(f"{'='*75}")
    for y in sorted(by_year.keys()):
        y_trades = by_year[y]
        y_days = len(set(t["date"] for t in y_trades))
        st = compute_stats(y_trades, y_days)
        print(f"Year {y}: Trades={st['trades']:3d} | WR={st['win_rate']:5.1f}% | Net Rs={st['net_rs']:+10,.2f} | PF={st['profit_factor']:5.2f} | MaxDD=Rs {st['max_drawdown_rs']:8,.2f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: {"stats": v["stats"], "trades": v["trades"]} for k, v in summary.items()}, f, indent=2)
    print(f"\nDetailed JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
