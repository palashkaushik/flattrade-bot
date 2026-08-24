"""Causal Marni Fib / UT Bot backtest on option charts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from backtest_5y_optimized import option_files
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
TARGET_LEVELS = (0.29, 0.0)
STOP_LEVELS = (1.079, 1.155, 1.25)
TIMEFRAME_PERIODS = {"1m": 1, "2m": 2, "3m": 3, "5m": 5}
BIAS_PERIODS = {tf: period * 5 for tf, period in TIMEFRAME_PERIODS.items()}
SYMBOL_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")
GLOBAL_SPOT = {}


def init_worker(spot):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot
    source.GLOBAL_CACHE = {}


class UTBotState:
    """Causal translation of the supplied UT Bot Pine v4 logic."""

    def __init__(self):
        self.atr = IncrementalATR(UT_ATR_PERIOD)
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

        loss = UT_KEY * atr
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


class BiasState:
    """Causal HA + UT Bot + LinReg Candle bias state."""

    def __init__(self, period: int):
        self.period = period
        self.buffer = []
        self.ha = HeikinAshiState()
        self.ut = UTBotState()
        self.opens = deque(maxlen=11)
        self.highs = deque(maxlen=11)
        self.lows = deque(maxlen=11)
        self.closes = deque(maxlen=11)
        self.signal_values = deque(maxlen=11)
        self.ha_candle = None
        self.ut_color = "blue"
        self.linreg_signal = None
        self.confirmed_minute = None

    def update_1m(self, candle: Candle):
        self.buffer.append(candle)
        if len(self.buffer) != self.period:
            return
        buf = self.buffer
        self.buffer = []
        aggregate = Candle(
            open=buf[0].open,
            high=max(item.high for item in buf),
            low=min(item.low for item in buf),
            close=buf[-1].close,
            minute=buf[-1].minute,
        )
        self.confirmed_minute = aggregate.minute
        ha = self.ha.update(aggregate)
        self.ha_candle = ha
        # TradingView's "Signals from Heikin Ashi Candles" input is disabled.
        self.ut_color = self.ut.update(aggregate)
        self.opens.append(ha.open)
        self.highs.append(ha.high)
        self.lows.append(ha.low)
        self.closes.append(ha.close)
        bclose = linreg_value(self.closes)
        if bclose is not None:
            self.signal_values.append(bclose)
            self.linreg_signal = (
                sum(self.signal_values) / len(self.signal_values)
                if len(self.signal_values) == 11
                else None
            )

    def snapshot(self) -> dict:
        close = self.ha_candle.close if self.ha_candle is not None else None
        open_price = self.ha_candle.open if self.ha_candle is not None else None
        bullish = (
            close is not None
            and open_price is not None
            and self.linreg_signal is not None
            and close > open_price
            and close > self.linreg_signal
            and self.ut_color == "green"
        )
        bearish = (
            close is not None
            and open_price is not None
            and self.linreg_signal is not None
            and close < open_price
            and close < self.linreg_signal
            and self.ut_color == "red"
        )
        return {
            "bullish": bullish,
            "bearish": bearish,
            "ha_open": open_price,
            "ha_close": close,
            "linreg_signal": self.linreg_signal,
            "ut_color": self.ut_color,
            "confirmed_minute": self.confirmed_minute,
        }


class BiasFeed:
    def __init__(self):
        self.states = {period: BiasState(period) for period in sorted(set(BIAS_PERIODS.values()))}

    def warmup(self, rows):
        for row in rows:
            self.push(row)

    def push(self, row):
        candle = row_to_candle(row)
        for state in self.states.values():
            state.update_1m(candle)

    def snapshot(self, timeframe: str) -> dict:
        return self.states[BIAS_PERIODS[timeframe]].snapshot()


class FuturesBiasFeed:
    """Bias feed driven by Nifty futures 5-minute bars (HA + UT Bot + LinReg)."""

    def __init__(self):
        self.state = BiasState(period=1)

    def warmup(self, rows):
        for row in rows:
            self.push(row)

    def push(self, row):
        self.state.update_1m(row_to_candle(row))

    def snapshot(self) -> dict:
        return self.state.snapshot()


class FibPattern:
    """Detect one UT-color sequence and retain its completed Fib range."""

    def __init__(self, direction="bullish", first="red", middle="green", final="red", orientation="high_to_low"):
        self.direction = direction
        self.first_color = first
        self.middle_color = middle
        self.final_color = final
        self.orientation = orientation
        self.previous_color = None
        self.previous_candle = None
        self.phase = "idle"
        self.range_high = None
        self.range_low = None
        self.setup = None
        self.middle_count = 0

    def reset_session(self):
        self.previous_color = None
        self.previous_candle = None
        self.phase = "idle"
        self.range_high = None
        self.range_low = None
        self.setup = None
        self.middle_count = 0

    def update(self, candle: Candle, color: str):
        completed_setup = None
        if self.phase == "green":
            if color == self.middle_color:
                self.middle_count += 1
                self.range_high = max(self.range_high, candle.high)
                self.range_low = min(self.range_low, candle.low)
            elif color == self.final_color:
                if self.middle_count >= 5:
                    self.range_high = max(self.range_high, candle.high)
                    self.range_low = min(self.range_low, candle.low)
                    completed_setup = (
                        self.direction,
                        self.range_high,
                        self.range_low,
                        self.orientation,
                    )
                self.phase = "idle"
                self.range_high = None
                self.range_low = None
                self.middle_count = 0
            else:
                self.phase = "idle"
                self.range_high = None
                self.range_low = None
                self.middle_count = 0
        elif (
            color == self.middle_color
            and self.previous_color == self.first_color
            and self.previous_candle is not None
        ):
            self.phase = "green"
            self.middle_count = 1
            self.range_high = max(self.previous_candle.high, candle.high)
            self.range_low = min(self.previous_candle.low, candle.low)

        self.previous_color = color
        self.previous_candle = candle
        if completed_setup is not None:
            self.setup = completed_setup
        return completed_setup


class FibTimeframe:
    def __init__(self, period: int, pattern_specs=None):
        self.period = period
        self.buffer = []
        self.ut = UTBotState()
        specs = pattern_specs or [
            ("bullish", "red", "green", "red", "high_to_low")
        ]
        self.patterns = [FibPattern(*spec) for spec in specs]
        self.setups = []

    def reset_session(self):
        self.buffer = []
        for pattern in self.patterns:
            pattern.reset_session()
        self.setups = []

    def push(self, candle: Candle):
        self.buffer.append(candle)
        if len(self.buffer) != self.period:
            return []
        buf = self.buffer
        self.buffer = []
        aggregate = Candle(
            open=buf[0].open,
            high=max(item.high for item in buf),
            low=min(item.low for item in buf),
            close=buf[-1].close,
            minute=buf[-1].minute,
        )
        color = self.ut.update(aggregate)
        active_setups = list(self.setups)
        new_setups = []
        for pattern in self.patterns:
            completed = pattern.update(aggregate, color)
            if completed is not None:
                new_setups.append(completed)

        events = []
        remaining_setups = []
        for direction, high, low, orientation in active_setups:
            entry_level = (
                high - ENTRY_LEVEL * (high - low)
                if orientation == "high_to_low"
                else low + ENTRY_LEVEL * (high - low)
            )
            if aggregate.high >= entry_level - 1.0 and aggregate.low <= entry_level + 1.0:
                events.append({
                    "minute": aggregate.minute,
                    "entry": aggregate.close,
                    "fib_high": high,
                    "fib_low": low,
                    "entry_level": entry_level,
                    "direction": direction,
                    "orientation": orientation,
                    "timeframe": f"{self.period}m",
                })
            else:
                remaining_setups.append((direction, high, low, orientation))

        self.setups = remaining_setups + new_setups
        return events


class SymbolFibFeed:
    def __init__(self, pattern_mode="option", strict_confirmation=True):
        if pattern_mode == "index":
            pattern_specs = [
                ("bullish", "red", "green", "red", "high_to_low"),
                ("bearish", "green", "red", "green", "low_to_high"),
            ]
        elif pattern_mode == "option5":
            # Mirror of the canonical "Fibonacci 5 Candles" color pattern for BOTH
            # directions. Bullish (CE): red -> 5+ green -> red on the option chart,
            # entry at high_to_low 0.786. Bearish (PE): green -> 5+ red -> green on
            # the option chart, entry at high_to_low 0.786 of the down-swing (the
            # level price retraces UP to after falling) -- this is the PE buy setup
            # the index-only color pattern was missing. Keeps the color pattern.
            pattern_specs = [
                ("bullish", "red", "green", "red", "high_to_low"),
                ("bearish", "green", "red", "green", "high_to_low"),
            ]
        else:
            pattern_specs = [("bullish", "red", "green", "red", "high_to_low")]
        self.timeframes = {
            tf: FibTimeframe(period, pattern_specs)
            for tf, period in TIMEFRAME_PERIODS.items()
        }
        self.bias = BiasFeed()
        self.strict_confirmation = strict_confirmation
        self.pending = []

    def warmup(self, rows, reset_session=True):
        for row in rows:
            self.bias.push(row)
            candle = row_to_candle(row)
            for timeframe in self.timeframes.values():
                timeframe.push(candle)
        if reset_session:
            for timeframe in self.timeframes.values():
                timeframe.reset_session()
        for state in self.bias.states.values():
            state.buffer = []
            state.confirmed_minute = None
        self.pending = []

    def push(self, row):
        self.bias.push(row)
        candle = row_to_candle(row)
        events = []
        if self.strict_confirmation:
            remaining = []
            for pending in self.pending:
                confirmed = confirm_event(
                    pending,
                    self.bias.snapshot(pending["timeframe"]),
                    row["minute"],
                )
                if confirmed is None:
                    remaining.append(pending)
                else:
                    events.append(confirmed)
            self.pending = remaining
        for tf, timeframe in self.timeframes.items():
            for event in timeframe.push(candle):
                event["timeframe"] = tf
                event["bias"] = self.bias.snapshot(tf)
                if self.strict_confirmation:
                    confirmed = confirm_event(event, event["bias"], row["minute"])
                    if confirmed is None:
                        self.pending.append(event)
                    else:
                        events.append(confirmed)
                else:
                    events.append(event)
        return events


def row_to_candle(row):
    return Candle(
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        minute=int(row["minute"]),
    )


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


def active_strikes(spot, minute, side, ce_offset=None, pe_offset=None):
    spot_price = source.latest_value(spot, minute)
    if spot_price is None:
        return None
    atm = int(round(spot_price / 50.0) * 50)
    if ce_offset is None:
        ce_offset = source.CE_OFFSET
    if pe_offset is None:
        pe_offset = source.PE_OFFSET
    return atm + (ce_offset if side == "CE" else pe_offset)


def fib_price(high, low, level, orientation="high_to_low"):
    return (
        high - level * (high - low)
        if orientation == "high_to_low"
        else low + level * (high - low)
    )


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
            active_keys.add(("CE", atm + source.CE_OFFSET))
            active_keys.add(("PE", atm + source.PE_OFFSET))

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
    return symbols, rows_by_symbol


def spot_row(spot, index):
    return {
        "minute": int(spot["min"][index]),
        "open": float(spot["open"][index]),
        "high": float(spot["high"][index]),
        "low": float(spot["low"][index]),
        "close": float(spot["close"][index]),
    }


def bias_allows(snapshot, side):
    return snapshot["bullish"] if side == "CE" else snapshot["bearish"]


def bias_confirmed_for_event(event, snapshot):
    confirmed_minute = snapshot.get("confirmed_minute")
    return confirmed_minute is not None and confirmed_minute >= event["minute"]


def confirm_event(event, snapshot, confirmation_minute):
    if not bias_confirmed_for_event(event, snapshot):
        return None
    confirmed = {**event, "bias": snapshot}
    if confirmation_minute > event["minute"]:
        confirmed["signal_minute"] = event["minute"]
        confirmed["minute"] = confirmation_minute
    return confirmed


def combined_bias_allows(index_snapshot, option_snapshot, side):
    return bias_allows(index_snapshot, side) and bias_allows(option_snapshot, side)


def simulate(
    events,
    bars,
    index_bars,
    spot,
    timeframe_mode,
    target_level,
    stop_level,
    concurrent=False,
    option_point_threshold=15.0,
    fallback_target_level=0.0,
    brokerage_per_order=BROKERAGE_PER_ORDER,
    fixed_cost_per_trade=None,
    max_trades_per_day=None,
    daily_loss_limit_rs=None,
):
    allowed = None if timeframe_mode == "combined" else timeframe_mode
    events_by_minute = defaultdict(list)
    for event in events:
        if allowed is None or event["timeframe"] == allowed:
            events_by_minute[event["minute"]].append(event)
    # positions: dict keyed by (side,strike) when concurrent, else a single
    # "global" slot. Single-slot preserves legacy (regression) behaviour.
    positions = {}
    stopped = False
    consecutive_losses = 0
    day_trade_count = 0
    day_net_rs = 0.0
    trades = []
    timeline = [
        int(minute)
        for minute in spot["min"]
        if SESSION_START <= minute <= DAY_LAST
    ]
    for minute in timeline:
        for slot in list(positions.keys()):
            position = positions[slot]
            if minute > position["entry_min"]:
                row = bars[position["key"]].get(minute)
                if row:
                    stop = fib_price(
                        position["fib_high"],
                        position["fib_low"],
                        stop_level,
                        position["orientation"],
                    )
                    active_target_level = position.get("active_target_level", target_level)
                    target = fib_price(
                        position["fib_high"],
                        position["fib_low"],
                        active_target_level,
                        position["orientation"],
                    )
                    price_row = index_bars[minute] if position["fib_source"] == "index" else row
                    price_rise = position.get(
                        "price_profit_on_rise",
                        position.get("profit_on_rise", position["side"] == "CE"),
                    )
                    premium_rise = position.get(
                        "profit_on_rise", position["side"] == "CE"
                    )
                    if price_rise:
                        hit_stop = price_row["low"] <= stop
                        hit_target = price_row["high"] >= target
                    else:
                        hit_stop = price_row["high"] >= stop
                        hit_target = price_row["low"] <= target
                    reason = "SL" if hit_stop else "TP" if hit_target else None
                    if minute >= SESSION_END and reason is None:
                        reason = "EOD"
                    if (
                        reason == "TP"
                        and active_target_level == target_level
                        and abs(target_level - 0.29) < 1e-9
                        and position.get("dynamic_target", False)
                        and minute < SESSION_END
                    ):
                        option_points = (
                            row["close"] - position["option_entry"]
                            if premium_rise
                            else position["option_entry"] - row["close"]
                        )
                        if option_points < option_point_threshold:
                            position["active_target_level"] = fallback_target_level
                            continue

                    if reason:
                        if premium_rise:
                            # buy to profit as the premium RISES
                            entry_fill = position["option_entry"] - SLIPPAGE_PTS
                            exit_fill = row["close"] + SLIPPAGE_PTS
                            points = round(exit_fill - entry_fill, 2)
                        else:
                            # buy to profit as the premium FALLS
                            entry_fill = position["option_entry"] + SLIPPAGE_PTS
                            exit_fill = row["close"] - SLIPPAGE_PTS
                            points = round(entry_fill - exit_fill, 2)
                        fee = (
                            round(float(fixed_cost_per_trade), 2)
                            if fixed_cost_per_trade is not None
                            else trade_cost(entry_fill, exit_fill, brokerage_per_order)
                        )
                        net_rs = round(points * LOT_SIZE - fee, 2)
                        trades.append({
                            "entry_min": position["entry_min"],
                            "exit_min": minute,
                            "side": position["side"],
                            "symbol": position["symbol"],
                            "timeframe": position["timeframe"],
                            "entry": entry_fill,
                            "exit": exit_fill,
                            "target_level": active_target_level,
                            "stop_level": stop_level,
                            "reason": reason,
                            "points": points,
                            "rs_net": net_rs,
                            "fee": fee,
                        })
                        consecutive_losses = consecutive_losses + 1 if net_rs <= 0 else 0
                        stopped = consecutive_losses >= CONSECUTIVE_LOSS_LIMIT
                        day_trade_count += 1
                        day_net_rs += net_rs
                        positions.pop(slot)
        if not concurrent:
            # legacy: only one position at a time
            if positions:
                continue
        if (
            stopped
            or minute >= SESSION_END
            or (max_trades_per_day is not None and day_trade_count >= max_trades_per_day)
            or (daily_loss_limit_rs is not None and day_net_rs <= -daily_loss_limit_rs)
        ):
            continue
        for event in events_by_minute.get(minute, []):
            slot = (event["side"], event["strike"]) if concurrent else "global"
            if slot in positions:
                continue
            if not concurrent and positions:
                break
            positions[slot] = {
                **event,
                "entry_min": minute,
                "key": (event["side"], event["strike"]),
            }
    return trades


def stats(trades, days):
    wins = [trade for trade in trades if trade["rs_net"] > 0]
    losses = [trade for trade in trades if trade["rs_net"] <= 0]
    loss_total = abs(sum(trade["rs_net"] for trade in losses))
    return {
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "net_rs": round(sum(trade["rs_net"] for trade in trades)),
        "net_points": round(sum(trade["rs_net"] for trade in trades) / LOT_SIZE, 2),
        "profit_factor": round(sum(trade["rs_net"] for trade in wins) / loss_total, 4) if loss_total else float("inf"),
        "avg_trades_per_day": round(len(trades) / days, 3) if days else 0.0,
        "fees_rs": round(sum(trade["fee"] for trade in trades), 2),
    }


def process_day(args):
    day, current_path, previous_path, timeframe_modes, target_levels, stop_levels, bias_mode = args
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
        feed = SymbolFibFeed("index")
        option_bias_feeds = {}
        for key in symbols:
            option_feed = SymbolFibFeed("option")
            option_feed.warmup(symbol_rows[key]["previous"])
            option_bias_feeds[key] = option_feed
        if previous_days:
            previous_spot = GLOBAL_SPOT[previous_days[-1]]
            feed.warmup(
                [
                    spot_row(previous_spot, index)
                    for index in range(len(previous_spot["min"]))
                ]
            )
        for index in range(len(spot["min"])):
            row = spot_row(spot, index)
            for key, option_feed in option_bias_feeds.items():
                option_row = bars[key].get(row["minute"])
                if option_row is not None:
                    option_feed.push(option_row)
            for event in feed.push(row):
                minute = row["minute"]
                side = "CE" if event["direction"] == "bullish" else "PE"
                strike = active_strikes(spot, minute, side)
                key = (side, strike)
                option_feed = option_bias_feeds.get(key)
                if key not in bars or option_feed is None:
                    continue
                option_bias = option_feed.bias.snapshot(event["timeframe"])
                if not combined_bias_allows(event["bias"], option_bias, side):
                    continue
                option_row = bars[key].get(minute)
                if option_row is None:
                    continue
                events.append({
                    **event,
                    "side": side,
                    "strike": strike,
                    "symbol": symbol_by_key[key],
                    "minute": minute,
                    "option_entry": option_row["close"],
                    "fib_source": "index",
                    "bias_mode": bias_mode,
                })
    else:
        for side, strike in symbols:
            record = symbol_rows[(side, strike)]
            feed = SymbolFibFeed("option")
            feed.warmup(record["previous"])
            for row in record["current"]:
                for event in feed.push(row):
                    minute = row["minute"]
                    if active_strikes(spot, minute, side) != strike:
                        continue
                    if not bias_allows(event["bias"], side):
                        continue
                    events.append({
                        **event,
                        "side": side,
                        "strike": strike,
                        "symbol": record["symbol"],
                        "minute": minute,
                        "option_entry": row["close"],
                        "fib_source": "option",
                        "orientation": "low_to_high" if side == "PE" else event["orientation"],
                        "bias_mode": bias_mode,
                    })

    index_bars = {
        int(spot["min"][index]): spot_row(spot, index)
        for index in range(len(spot["min"]))
    }

    output = {}
    for timeframe in timeframe_modes:
        for target_level in target_levels:
            for stop_level in stop_levels:
                key = f"{timeframe}|tp{target_level}|sl{stop_level}"
                output[key] = simulate(
                    events,
                    bars,
                    index_bars,
                    spot,
                    timeframe,
                    target_level,
                    stop_level,
                )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/marni_fib_backtest.json")
    args = parser.parse_args()

    spot_all = source.load_spot()
    option_map = source.option_day_files(args.start, args.end)
    all_days = sorted(set(option_map) & set(spot_all))
    days = all_days[:5] if args.smoke else all_days
    previous = {day: max((candidate for candidate in all_days if candidate < day), default="") for day in days}
    timeframe_modes = ("1m", "2m", "3m", "5m", "combined")
    target_levels = TARGET_LEVELS
    stop_levels = STOP_LEVELS
    bias_modes = ("index", "option")
    tasks = [
        (
            day,
            str(option_map[day]),
            str(option_map[previous[day]]) if previous[day] else "",
            timeframe_modes,
            target_levels,
            stop_levels,
            bias_mode,
        )
        for day in days
        for bias_mode in bias_modes
    ]
    aggregate = defaultdict(list)
    with Pool(max(1, min(8, args.workers)), initializer=init_worker, initargs=(spot_all,)) as pool:
        for task, day_result in zip(tasks, pool.imap(process_day, tasks)):
            bias_mode = task[6]
            for key, trades in day_result.items():
                aggregate[f"{bias_mode}|{key}"].extend(trades)
    results = {
        key: stats(trades, len(days))
        for key, trades in sorted(aggregate.items())
    }
    output_data = {
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "smoke": args.smoke,
        "ut_bot": {
            "key_value": UT_KEY,
            "atr_period": UT_ATR_PERIOD,
            "fib_source": "regular_candles",
            "bias_source": "heikin_ashi",
            "ut_source": "regular_candles",
        },
        "pattern": "red -> green+ -> red; full sequence high-to-low Fib range",
        "fib_setups": "multiple concurrent unfinished setups are retained",
        "entry": "0.786 touch zone +/-1, enter at confirming candle close",
        "bias_confirmation": "completed higher-timeframe candle; deferred events fill on confirmation close",
        "bias_requires_ha_body_color": True,
        "index_mode_requires_selected_option_bias": True,
        "target_levels": TARGET_LEVELS,
        "stop_levels": STOP_LEVELS,
        "timeframes": timeframe_modes,
        "bias_modes": bias_modes,
        "consecutive_loss_limit": CONSECUTIVE_LOSS_LIMIT,
        "daily_caps": False,
        "results": results,
    }
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(json.dumps(output_data, indent=2))
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
