"""Causal, clock-aligned, cost-aware research replay for the live F6 profile.

This is research-only. It deliberately uses the legacy rolling-close divergence
requested for the comparison, while matching live clock boundaries and the
live SL/TP/EOD exit set. No live bot code is changed by this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import date
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_5y_optimized import SYM_RE, latest_spot, load_spot, option_files, to_minutes
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost
from flattrade_bot.indicators.patterns import BullishPinBarDetector, Candle
from flattrade_bot.indicators.divergence import DivergenceEngine
from flattrade_bot.indicators.ema import IncrementalEMA
from flattrade_bot.indicators.elder import elder_allows, IncrementalElderImpulse
from flattrade_bot.indicators.stochastic import IncrementalStochastic


TF_SPECS = {
    "1m": (1, 10, 6.0, 30.0),
    "2m": (2, 5, 10.0, 15.0),
    "3m": (3, 4, 8.0, 25.0),
    "5m": (5, 3, 10.0, 35.0),
}
EXIT_POLICIES = (
    "dynamic_both",
    "ratchet_sl_dynamic_tp",
    "chandelier_sl_dynamic_tp",
)
BREAKOUT_MODES = ("legacy_high_break", "first_break_high")
SIGNAL_MODES = ("pinbar", "ordered_stoch")
ENTRY_FILTERS = ("none", "ema20", "elder_permissive")
SESSION_START = 560
SESSION_END = 900
DAY_LAST = 930
LOT_SIZE = 65
DAILY_LOSS_RS = -2000.0
GLOBAL_SPOT = {}
DAY_CACHE = {}


class LegacyDivergence:
    """Former rolling-close extrema divergence used by the old champion."""

    def __init__(self, max_history=40, min_lookback=3, max_lookback=30):
        self.max_history = max_history
        self.min_lookback = min_lookback
        self.max_lookback = max_lookback
        self.price = deque(maxlen=max_history)
        self.s1 = deque(maxlen=max_history)

    def update(self, close, s1, **_kwargs):
        if s1 is not None:
            self.price.append(close)
            self.s1.append(s1)

    def _pair(self, peak=False):
        prices = list(self.price)
        values = list(self.s1)
        n = len(prices)
        if n < self.min_lookback + 3:
            return None, None
        window = min(10, n)
        index2 = (max if peak else min)(range(n - window, n), key=prices.__getitem__)
        end = max(0, index2 - self.min_lookback + 1)
        start = max(0, index2 - self.max_lookback)
        if end <= start:
            return None, None
        prior = prices[start:end]
        index1 = start + (max if peak else min)(range(len(prior)), key=prior.__getitem__)
        return (prices[index1], values[index1]), (prices[index2], values[index2])

    def has_bullish(self):
        first, second = self._pair()
        return first is not None and second[0] < first[0] and second[1] > first[1]


def legacy_high_break(candle_history, max_lookback=10):
    """Legacy rule: reuse any nearby pinbar whose high was already broken."""
    if len(candle_history) < 2:
        return False
    current = candle_history[-1]
    lookback = min(len(candle_history) - 1, max_lookback)
    for index in range(1, lookback + 1):
        past = candle_history[-1 - index]
        if BullishPinBarDetector.is_bullish_pin_bar(past) and current.high > past.high:
            return True
    return False


def first_break_high(candle_history, max_lookback=10):
    """New rule: only the first candle trading above a pinbar high qualifies."""
    if len(candle_history) < 2:
        return False
    current = candle_history[-1]
    lookback = min(len(candle_history) - 1, max_lookback)
    for index in range(1, lookback + 1):
        past = candle_history[-1 - index]
        intervening = candle_history[-index:-1]
        if (
            BullishPinBarDetector.is_bullish_pin_bar(past)
            and current.high > past.high
            and all(candle.high <= past.high for candle in intervening)
        ):
            return True
    return False


def breakout_triggered(mode, candle_history, max_lookback):
    if mode == "legacy_high_break":
        return legacy_high_break(candle_history, max_lookback)
    if mode == "first_break_high":
        return first_break_high(candle_history, max_lookback)
    raise ValueError(f"Unknown breakout mode: {mode}")


def ordered_stochastics(values):
    """Return True only for strict S1 > S2 > S3 > S4 ordering."""
    if any(value is None for value in values):
        return False
    s1, s2, s3, s4 = values
    return s1 > s2 > s3 > s4


class IncrementalATR:
    def __init__(self, period=10):
        self.period = period
        self.buffer = deque(maxlen=period)
        self.prev_close = None
        self.value = None
        self.count = 0

    def update(self, high, low, close):
        tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close)) if self.prev_close else high - low
        self.buffer.append(tr)
        self.count += 1
        self.prev_close = close
        if self.count == 1:
            self.value = tr
        elif self.count <= self.period:
            self.value = sum(self.buffer) / len(self.buffer)
        else:
            self.value = (self.value * (self.period - 1) + tr) / self.period
        return self.value

    def peek(self, high, low, close):
        """ATR as-if the given (forming) bar were appended — no mutation."""
        if self.prev_close is None:
            return None
        tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        n = self.count + 1
        if n == 1:
            return tr
        if n <= self.period:
            return (sum(self.buffer) + tr) / n
        return (self.value * (self.period - 1) + tr) / self.period


class CausalTFTracker:
    def __init__(
        self,
        lookback,
        params,
        divergence_mode,
        breakout_mode,
        signal_mode,
        require_ema20=False,
        chart_side=None,
        entry_filter="none",
    ):
        self.lookback = lookback
        self.s1 = IncrementalStochastic(params["s1_k"], params["s1_d"])
        self.s2 = IncrementalStochastic(14, 3)
        self.s3 = IncrementalStochastic(40, 4)
        self.s4 = IncrementalStochastic(params["s4_k"], 10)
        self.atr = IncrementalATR(params["atr_period"])
        self.ema20 = IncrementalEMA(20)
        self.ema20_value = None
        self.elder = IncrementalElderImpulse()
        self.elder_color = "blue"
        self.divergence_mode = divergence_mode
        self.breakout_mode = breakout_mode
        self.signal_mode = signal_mode
        self.chart_side = chart_side
        self.entry_filter = "ema20" if require_ema20 and entry_filter == "none" else entry_filter
        self.div = LegacyDivergence() if divergence_mode == "previous_divergence" else DivergenceEngine()
        self.history = []
        self.setup = False
        self.setup_type = ""
        self.order_active = False
        self.s4_embedded = 0
        self.f6_s4 = params["f6_s4_thresh"]
        self.f6_s1 = params["f6_s1_thresh"]

    def reset_session_state(self):
        """Clear pending setups while retaining indicator warm-up state."""
        self.history.clear()
        self.setup = False
        self.setup_type = ""
        self.order_active = False
        self.s4_embedded = 0

    def push(self, candle):
        self.history.append(candle)
        if len(self.history) > 60:
            self.history.pop(0)
        s1 = self.s1.push(candle.high, candle.low, candle.close)
        s2 = self.s2.push(candle.high, candle.low, candle.close)
        s3 = self.s3.push(candle.high, candle.low, candle.close)
        s4 = self.s4.push(candle.high, candle.low, candle.close)
        atr = self.atr.update(candle.high, candle.low, candle.close)
        self.ema20_value = self.ema20.update(candle.close)
        self.elder_color = self.elder.update(candle.close)
        if s4 is not None:
            self.s4_embedded = self.s4_embedded + 1 if s4 <= 20.0 else 0
        self.div.update(candle.close, s1, low_price=candle.low, high_price=candle.high)
        is_flag = s4 is not None and s1 is not None and s4 >= self.f6_s4 and s1 <= self.f6_s1
        is_super = all(value is not None and value <= 20.5 for value in (s1, s2, s3, s4))
        if self.divergence_mode == "no_divergence":
            has_bullish = True
        elif self.divergence_mode == "previous_divergence":
            has_bullish = self.div.has_bullish()
        else:
            has_bullish = self.div.has_bullish_trough_divergence()
        filter_ok = self.entry_filter_allows(candle.close)
        if self.signal_mode == "ordered_stoch":
            ordered = ordered_stochastics((s1, s2, s3, s4))
            triggered = ordered and not self.order_active and has_bullish and filter_ok
            self.order_active = ordered
            return triggered, False, "ordered_stoch" if triggered else "", candle.close, atr
        if (is_flag or is_super) and has_bullish:
            self.setup = True
            self.setup_type = "super" if is_super else "flag"
        triggered = False
        if self.setup and len(self.history) >= 2 and breakout_triggered(self.breakout_mode, self.history, self.lookback):
            triggered = filter_ok
            self.setup = False
        return triggered, self.s4_embedded >= 25 and self.setup_type == "super", self.setup_type, candle.close, atr

    def entry_filter_allows(self, close: float) -> bool:
        if self.entry_filter == "none":
            return True
        if self.entry_filter == "ema20":
            return self.ema20_value is not None and close > self.ema20_value
        if self.entry_filter == "elder_permissive":
            return self.chart_side is not None and elder_allows(
                self.elder_color, self.chart_side, "permissive"
            )
        raise ValueError(f"Unknown entry filter: {self.entry_filter}")


class CausalMTF:
    def __init__(
        self,
        params,
        divergence_mode,
        breakout_mode,
        signal_mode,
        reverse_regime_enabled,
        require_ema20=False,
        chart_side=None,
        entry_filter="none",
    ):
        self.trackers = {
            tf: CausalTFTracker(
                spec[1],
                params,
                divergence_mode,
                breakout_mode,
                signal_mode,
                require_ema20,
                chart_side,
                entry_filter,
            )
            for tf, spec in TF_SPECS.items()
        }
        self.s1 = {tf: IncrementalStochastic(params["s1_k"], params["s1_d"]) for tf in TF_SPECS}
        self.s4 = {tf: IncrementalStochastic(params["s4_k"], 10) for tf in TF_SPECS}
        self.buffers = {tf: [] for tf in TF_SPECS}
        self._last_minute = None
        self.reverse_regime_active = False
        self.reverse_regime_enabled = reverse_regime_enabled
        self.signal_mode = signal_mode
        self.entry_filter = "ema20" if require_ema20 and entry_filter == "none" else entry_filter
        self.last_atr_updates = {}
        self.f6_s4 = params["f6_s4_thresh"]
        self.f6_s1 = params["f6_s1_thresh"]
        self.s4_k = params["s4_k"]
        self.s1_k = params["s1_k"]
        self.s1_d = params["s1_d"]

    def push(self, candle):
        signals = []
        minute = candle.minute
        self.last_atr_updates = {}
        if self._last_minute is not None and minute > 0 and minute < self._last_minute:
            self.buffers = {tf: [] for tf in TF_SPECS}
            for tracker in self.trackers.values():
                tracker.reset_session_state()
            self.reverse_regime_active = False
        self._last_minute = minute
        for tf, spec in TF_SPECS.items():
            self.buffers[tf].append(candle)
            period = spec[0]
            boundary = (minute % period == 0 and len(self.buffers[tf]) >= 1) if minute > 0 else len(self.buffers[tf]) >= period
            if not boundary:
                continue
            buffer = self.buffers[tf]
            self.buffers[tf] = []
            aggregate = Candle(
                open=buffer[0].open,
                high=max(item.high for item in buffer),
                low=min(item.low for item in buffer),
                close=buffer[-1].close,
                minute=buffer[-1].minute,
            )
            triggered, reverse, setup, price, atr = self.trackers[tf].push(aggregate)
            self.last_atr_updates[tf] = self.trackers[tf].atr.value
            s1 = self.s1[tf].push(aggregate.high, aggregate.low, aggregate.close)
            s4 = self.s4[tf].push(aggregate.high, aggregate.low, aggregate.close)
            if triggered:
                signals.append((tf, reverse, setup, price, atr))
            if (
                s1 is not None
                and s4 is not None
                and s4 >= self.f6_s4
                and s1 <= self.f6_s1
                and self.trackers[tf].entry_filter_allows(aggregate.close)
            ):
                # Preserve the old F6 immediate no-divergence branch.
                signals.append((tf, False, "flag_nodiv", aggregate.close, atr))
        self.reverse_regime_active = self.reverse_regime_enabled and any(
            tracker.s4_embedded >= 25 for tracker in self.trackers.values()
        )
        return [
            (
                tf,
                reverse or (self.reverse_regime_active and setup == "super"),
                setup,
                price,
                atr,
            )
            for tf, reverse, setup, price, atr in signals
        ]


def load_groups(path):
    cached = DAY_CACHE.get(path)
    if cached is not None:
        return cached
    frame = pd.read_csv(path, usecols=["time", "symbol", "open", "high", "low", "close"], engine="c")
    if frame.empty:
        DAY_CACHE[path] = {}
        return DAY_CACHE[path]
    frame["min"] = np.array([to_minutes(value) for value in frame["time"]])
    frame = frame.drop_duplicates(subset=["symbol", "min"], keep="last")
    frame = frame.sort_values(["symbol", "min"], kind="stable")
    output = {}
    for symbol, group in frame.groupby("symbol"):
        output[symbol] = {
            key: group[key].to_numpy() for key in ("min", "open", "high", "low", "close")
        }
    DAY_CACHE[path] = output
    return output


def init_worker(spot):
    global GLOBAL_SPOT, DAY_CACHE
    GLOBAL_SPOT = spot
    DAY_CACHE = {}


def is_daily_profit_cap_reached(daily_net_rs: float, max_daily_profit_points: Optional[float]) -> bool:
    """Returns whether realized net profit has reached the configured point cap."""
    return max_daily_profit_points is not None and daily_net_rs >= max_daily_profit_points * LOT_SIZE


def is_daily_loss_cap_reached(daily_net_rs: float, max_daily_loss_points: Optional[float]) -> bool:
    """Returns whether realized net loss has reached the configured point cap."""
    return max_daily_loss_points is not None and daily_net_rs <= -max_daily_loss_points * LOT_SIZE


def resolve_sl_points(
    atr_value: Optional[float],
    atr_sl_mult: float,
    fallback_points: float,
    fixed_sl_points: Optional[float] = None,
) -> float:
    """Resolve a fixed SL override without changing the ATR calculation."""
    if fixed_sl_points is not None:
        return fixed_sl_points
    return atr_value * atr_sl_mult if atr_value and atr_value > 0.5 else fallback_points


def resolve_daily_loss_limit(
    max_daily_loss_points: Optional[float],
    no_daily_caps: bool = False,
) -> float:
    """Resolve the daily loss guard without silently re-enabling the default cap."""
    if no_daily_caps:
        return float("-inf")
    return (
        -max_daily_loss_points * LOT_SIZE
        if max_daily_loss_points is not None
        else DAILY_LOSS_RS
    )


def resolve_tp_points(
    atr_value: Optional[float],
    atr_tp_mult: float,
    fallback_points: float,
    fixed_tp_points: Optional[float] = None,
) -> float:
    """Resolves a fixed TP override without changing the ATR-based SL path."""
    if fixed_tp_points is not None:
        return fixed_tp_points
    return atr_value * atr_tp_mult if atr_value and atr_value > 0.5 else fallback_points


def resolve_dynamic_exit_levels(position, atr_value, params, policy):
    """Return updated long-position levels for a dynamic ATR policy."""
    if policy == "static" or atr_value is None or atr_value <= 0.5:
        return position["sl"], position["target"]

    entry = position["entry"]
    sl_distance = atr_value * params["atr_sl_mult"]
    tp_distance = atr_value * params["atr_tp_mult"]
    dynamic_sl = entry - sl_distance
    dynamic_tp = entry + tp_distance

    if policy == "dynamic_both":
        return dynamic_sl, dynamic_tp
    if policy == "ratchet_sl_dynamic_tp":
        return max(position["sl"], dynamic_sl), dynamic_tp
    if policy == "chandelier_sl_dynamic_tp":
        chandelier_sl = position.get("high_watermark", entry) - sl_distance
        return max(position["sl"], chandelier_sl), dynamic_tp
    raise ValueError(f"Unknown exit policy: {policy}")


def process_day(args):
    day, path, previous_path, params, divergence_mode, *cost_args = args
    costs_enabled = cost_args[0] if cost_args else True
    max_daily_profit_points = cost_args[1] if len(cost_args) > 1 else None
    max_daily_loss_points = cost_args[2] if len(cost_args) > 2 else None
    fixed_tp_points = cost_args[3] if len(cost_args) > 3 else None
    exit_policy = cost_args[4] if len(cost_args) > 4 else "static"
    breakout_mode = cost_args[5] if len(cost_args) > 5 else "first_break_high"
    fixed_sl_points = cost_args[6] if len(cost_args) > 6 else None
    signal_mode = cost_args[7] if len(cost_args) > 7 else "pinbar"
    reverse_regime_enabled = cost_args[8] if len(cost_args) > 8 else True
    no_daily_caps = cost_args[9] if len(cost_args) > 9 else False
    require_ema20 = cost_args[10] if len(cost_args) > 10 else False
    entry_filter = cost_args[11] if len(cost_args) > 11 else ("ema20" if require_ema20 else "none")
    slippage_points = SLIPPAGE_PTS if costs_enabled else 0.0
    daily_loss_limit_rs = resolve_daily_loss_limit(max_daily_loss_points, no_daily_caps)
    spot = GLOBAL_SPOT.get(day)
    groups = load_groups(path)
    previous = load_groups(previous_path) if previous_path else {}
    if spot is None or not groups:
        return []
    first = next(iter(groups), "")
    match = SYM_RE.match(first)
    start_spot = latest_spot(spot, 555) or latest_spot(spot, SESSION_START)
    if not match or start_spot is None:
        return []
    base = int(round(start_spot / 50.0) * 50)
    target_strikes = set(range(base - 250, base + 300, 50))
    groups = {symbol: group for symbol, group in groups.items() if (m := SYM_RE.match(symbol)) and int(m.group(2)) in target_strikes}
    previous = {symbol: group for symbol, group in previous.items() if (m := SYM_RE.match(symbol)) and int(m.group(2)) in target_strikes}
    trackers = {}
    for symbol, group in previous.items():
        tracker = CausalMTF(
            params,
            divergence_mode,
            breakout_mode,
            signal_mode,
            reverse_regime_enabled,
            require_ema20,
            chart_side=SYM_RE.match(symbol).group(3),
            entry_filter=entry_filter,
        )
        for index in range(len(group["min"])):
            tracker.push(Candle(group["open"][index], group["high"][index], group["low"][index], group["close"][index], minute=group["min"][index]))
        trackers[symbol] = tracker
    current_trackers = {}
    events = {}
    atr_updates = {}
    slices = {}
    for symbol, group in groups.items():
        tracker = trackers.get(symbol)
        if tracker is None:
            tracker = CausalMTF(
                params,
                divergence_mode,
                breakout_mode,
                signal_mode,
                reverse_regime_enabled,
                require_ema20,
                chart_side=SYM_RE.match(symbol).group(3),
                entry_filter=entry_filter,
            )
        current_trackers[symbol] = tracker
        slices[symbol] = group
        for index in range(len(group["min"])):
            minute = group["min"][index]
            signals = tracker.push(Candle(group["open"][index], group["high"][index], group["low"][index], group["close"][index], minute=minute))
            for tf, atr_value in tracker.last_atr_updates.items():
                atr_updates.setdefault(minute, {}).setdefault(symbol, {})[tf] = atr_value
            for tf, reverse, setup, price, atr in signals:
                events.setdefault(minute, []).append((symbol, int(SYM_RE.match(symbol).group(2)), SYM_RE.match(symbol).group(3), tf, reverse, setup, price, atr))

    def bar_at(group, minute):
        index = np.searchsorted(group["min"], minute)
        if index < len(group["min"]) and group["min"][index] == minute:
            return group["open"][index], group["high"][index], group["low"][index], group["close"][index]
        return None

    def active(side, minute):
        price = latest_spot(spot, minute)
        if price is None:
            return None
        atm = int(round(price / 50.0) * 50)
        strike = atm + (-100 if side == "CE" else 100)
        symbol = f"{match.group(1)}{strike}{side}"
        group = slices.get(symbol)
        if group is None:
            return None
        return (symbol, group, strike)

    position = None
    daily_net = 0.0
    consecutive_losses = 0
    stopped = False
    trades = []
    for minute in range(SESSION_START, DAY_LAST + 1):
        if position is not None:
            held = bar_at(position["slice"], minute)
            if held:
                _, high, low, close = held
                position["last_px"] = close
                position["duration"] += 1
                exit_price = None
                reason = ""
                if high >= position["target"] and low <= position["sl"]:
                    exit_price, reason = position["sl"], "SL"
                elif high >= position["target"]:
                    exit_price, reason = position["target"], "TP"
                elif low <= position["sl"]:
                    exit_price, reason = position["sl"], "SL"
                if minute >= SESSION_END and exit_price is None:
                    exit_price, reason = close, "EOD"
                if exit_price is not None:
                    exit_fill = exit_price - slippage_points
                    gross_pts = round(exit_fill - position["entry"], 2)
                    fee = trade_cost(position["entry"], exit_fill, BROKERAGE_PER_ORDER) if costs_enabled else 0.0
                    net_rs = round(gross_pts * LOT_SIZE - fee, 2)
                    trades.append({"date": day, "entry_min": position["entry_min"], "exit_min": minute, "symbol": position["symbol"], "side": position["side"], "entry": position["entry"], "exit": exit_fill, "pts": gross_pts, "rs": round(gross_pts * LOT_SIZE), "rs_net": net_rs, "fee": fee, "reason": reason, "tf": position["tf"], "duration_min": position["duration"], "sl_points": position["sl_points"], "tp_points": position["tp_points"]})
                    daily_net += net_rs
                    consecutive_losses = consecutive_losses + 1 if net_rs <= 0 else 0
                    stopped = (
                        daily_net <= daily_loss_limit_rs
                        or consecutive_losses >= params["consec_loss"]
                        or is_daily_profit_cap_reached(daily_net, max_daily_profit_points)
                        or is_daily_loss_cap_reached(daily_net, max_daily_loss_points)
                    )
                    position = None
                elif exit_policy != "static":
                    position["high_watermark"] = max(position.get("high_watermark", position["entry"]), high)
                    current_atr = (
                        atr_updates.get(minute, {})
                        .get(position["symbol"], {})
                        .get(position["tf"])
                    )
                    new_sl, new_target = resolve_dynamic_exit_levels(
                        position,
                        current_atr,
                        params,
                        exit_policy,
                    )
                    position["sl"] = new_sl
                    position["target"] = new_target
                    position["sl_points"] = position["entry"] - new_sl
                    position["tp_points"] = new_target - position["entry"]
        if position is not None or stopped or minute >= SESSION_END:
            continue
        for symbol, strike, side, tf, reverse, setup, signal_price, atr in events.get(minute, []):
            source = active(side, minute)
            if source is None or source[2] != strike:
                continue
            actual_side = ("PE" if side == "CE" else "CE") if reverse else side
            actual = active(actual_side, minute)
            if actual is None:
                continue
            entry_bar = bar_at(actual[1], minute)
            if entry_bar is None:
                continue
            entry = entry_bar[3] + slippage_points
            atr_value = atr if atr and atr > 0.5 else None
            sl_points = resolve_sl_points(
                atr_value,
                params["atr_sl_mult"],
                TF_SPECS[tf][2],
                fixed_sl_points,
            )
            tp_points = resolve_tp_points(
                atr_value,
                params["atr_tp_mult"],
                TF_SPECS[tf][3],
                fixed_tp_points,
            )
            position = {"symbol": actual[0], "side": actual_side, "slice": actual[1], "entry": entry, "sl": entry - sl_points, "target": entry + tp_points, "entry_min": minute, "last_px": entry, "duration": 0, "tf": tf, "sl_points": sl_points, "tp_points": tp_points, "high_watermark": entry}
            break
    return trades


def stats(trades, day_count: Optional[int] = None):
    wins = [trade for trade in trades if trade["rs_net"] > 0]
    losses = [trade for trade in trades if trade["rs_net"] <= 0]
    gross_wins = sum(trade["rs_net"] for trade in wins)
    gross_losses = abs(sum(trade["rs_net"] for trade in losses))
    observed_days = len({trade["date"] for trade in trades})
    total_days = day_count if day_count is not None else observed_days
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "net_rs": round(sum(trade["rs_net"] for trade in trades)),
        "net_points": round(sum(trade["rs_net"] / LOT_SIZE for trade in trades), 2),
        "fees": round(sum(trade["fee"] for trade in trades), 2),
        "pf": round(gross_wins / gross_losses, 4) if gross_losses else float("inf"),
        "avg_trades_per_day": round(len(trades) / total_days, 3) if total_days else 0.0,
        "avg_sl_points": round(sum(trade["sl_points"] for trade in trades) / len(trades), 3) if trades else 0.0,
        "avg_tp_points": round(sum(trade["tp_points"] for trade in trades) / len(trades), 3) if trades else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mode", choices=("no_divergence", "new_divergence", "previous_divergence"), default="previous_divergence")
    parser.add_argument("--consec-loss", type=int, default=None, help="Override maximum consecutive losses")
    parser.add_argument("--no-costs", action="store_true", help="Disable slippage, brokerage, and statutory fees")
    parser.add_argument("--max-daily-profit-points", type=float, default=None, help="Stop opening new trades after this realized net point profit")
    parser.add_argument("--max-daily-loss-points", type=float, default=None, help="Stop opening new trades after this realized net point loss")
    parser.add_argument("--no-daily-caps", action="store_true", help="Disable daily profit and loss caps")
    parser.add_argument("--fixed-tp-points", type=float, default=None, help="Use an absolute TP value instead of ATR-derived TP")
    parser.add_argument("--fixed-sl-points", type=float, default=None, help="Use an absolute SL value instead of ATR-derived SL")
    parser.add_argument("--exit-policy", choices=("static", *EXIT_POLICIES), default="static")
    parser.add_argument("--breakout-mode", choices=BREAKOUT_MODES, default="first_break_high")
    parser.add_argument("--signal-mode", choices=SIGNAL_MODES, default="pinbar")
    parser.add_argument("--disable-reverse-regime", action="store_true")
    parser.add_argument("--require-ema20", action="store_true")
    parser.add_argument("--entry-filter", choices=ENTRY_FILTERS, default="none")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/causal_live_parity_research.json")
    args = parser.parse_args()
    params = json.loads(Path(args.params_file).read_text(encoding="utf-8"))["params"]
    if args.consec_loss is not None:
        params["consec_loss"] = args.consec_loss
    if args.require_ema20 and args.entry_filter == "none":
        args.entry_filter = "ema20"
    spot = load_spot()
    files = option_files(args.start, args.end)
    days = sorted(set(files) & set(spot))
    if args.smoke:
        days = days[:5]
    tasks = [
        (
            day,
            str(files[day]),
            str(files[days[index - 1]]) if index else "",
            params,
            args.mode,
            not args.no_costs,
            args.max_daily_profit_points,
            args.max_daily_loss_points,
            args.fixed_tp_points,
            args.exit_policy,
            args.breakout_mode,
            args.fixed_sl_points,
            args.signal_mode,
            not args.disable_reverse_regime,
            args.no_daily_caps,
            args.require_ema20,
            args.entry_filter,
        )
        for index, day in enumerate(days)
    ]
    trades = []
    with Pool(max(1, min(8, args.workers)), initializer=init_worker, initargs=(spot,)) as pool:
        for result in pool.imap(process_day, tasks):
            trades.extend(result)
    yearly = {}
    for trade in trades:
        yearly.setdefault(trade["date"][:4], []).append(trade)
    result = {
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "smoke": args.smoke,
        "mode": args.mode,
        "params": params,
        "engine": "causal clock-aligned fixed-parameter replay",
        "walk_forward": False,
        "max_daily_profit_points": args.max_daily_profit_points,
        "max_daily_loss_points": args.max_daily_loss_points,
        "fixed_tp_points": args.fixed_tp_points,
        "fixed_sl_points": args.fixed_sl_points,
        "exit_policy": args.exit_policy,
        "breakout_mode": args.breakout_mode,
        "signal_mode": args.signal_mode,
        "reverse_regime_enabled": not args.disable_reverse_regime,
        "daily_caps_enabled": not args.no_daily_caps,
        "ema20_filter_enabled": args.require_ema20,
        "entry_filter": args.entry_filter,
        "costs": {
            "enabled": not args.no_costs,
            "slippage_points_per_side": SLIPPAGE_PTS if not args.no_costs else 0.0,
            "brokerage_per_order": BROKERAGE_PER_ORDER if not args.no_costs else 0.0,
            "statutory_fees": not args.no_costs,
        },
        "stats": stats(trades, day_count=len(days)),
        "yearly": {year: stats(items) for year, items in sorted(yearly.items())},
    }
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
