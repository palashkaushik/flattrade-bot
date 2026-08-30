"""Smart Fib strategy engine fed entirely by the Flattrade day cache.

Signal construction is causal and follows the user's chart rules:

* Index 1m UT patterns use both ``red -> green -> red`` and
  ``green -> red -> green`` swings.
* Option 1m UT patterns use ``red -> green -> red`` for both CE and PE.
* Entry direction is confirmed only by the index 5m chart: Heikin-Ashi
  close versus the Humble LinReg signal plot, together with UT bar color.

All events are replayed through a single first-come-first-serve position. No
network access is used; all data comes from ``artifacts/flattrade_day_cache``.
"""

from __future__ import annotations

import copy
import math
import sys
from collections import deque
from datetime import date, datetime, timedelta
from operator import index as _index
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.flattrade_day_cache import load_day_cache
from artifacts.f6_hybrid.marni_fib_5y_fast import (
    ENTRY_LEVEL,
    Candle,
    HeikinAshiState,
    UTBotState,
    linreg_value,
)
from artifacts.f6_hybrid.marni_fib_backtest import (
    BROKERAGE_PER_ORDER,
    LOT_SIZE,
    fib_price,
    simulate,
    spot_row,
)
from flattrade_bot.indicators.stochastic import IncrementalStochastic

GLOBAL_CACHE_DIR = Path("artifacts/flattrade_day_cache")
STRATEGY_NAME = "Smart Fib"
MIN_SPAN = 15.0
MIN_PATTERN_CANDLES = 5
FILTER_PERIOD = 5
LINREG_LENGTH = 11
LINREG_SIGNAL_LENGTH = 11
OPTION_DELTA = 0.5
OPTION_POINT_THRESHOLD = 10.0
FILTER_SESSION_START = 555
S1_K_PERIOD = 12
S1_D_PERIOD = 3
S1_ZONE_START = 0.618
S1_ZONE_END = 1.0
ZONE_PAIRS = (
    (0.618, 1.0),
    (0.618, 0.786),
    (0.786, 1.0),
    (0.5, 0.786),
    (0.705, 0.886),
)
SETUP_MAX_AGE = 45


def _normalize_s1_periods(k_period, d_period):
    """Validate S1 periods once while preserving the historical defaults."""
    try:
        normalized_k = _index(k_period)
        normalized_d = _index(d_period)
    except TypeError as exc:
        raise ValueError("S1 k_period and d_period must be integer-like") from exc
    if normalized_k <= 0 or normalized_d <= 0:
        raise ValueError("S1 k_period and d_period must be positive")
    return normalized_k, normalized_d


def parse_row(row):
    parsed = datetime.strptime(row["time"], "%d-%m-%Y %H:%M:%S")
    return {
        "time": row["time"],
        "minute": parsed.hour * 60 + parsed.minute,
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
    }


def clean_terminal_quotes(rows):
    """Drop an obvious stale terminal quote without altering normal bars.

    Some cached sessions end with several identical flat quotes followed by a
    final flat quote that jumps far away from that plateau. That quote is not a
    tradable candle and can corrupt an overnight Fib high/low.
    """
    cleaned = sorted(rows, key=lambda row: (row["time"], row["minute"]))
    while len(cleaned) >= 5:
        tail = cleaned[-1]
        previous = cleaned[-2]
        prior_tail = cleaned[-4:-1]

        def is_flat(row):
            return (
                abs(row["high"] - row["low"]) <= 1e-9
                and abs(row["open"] - row["close"]) <= 1e-9
                and abs(row["close"] - row["high"]) <= 1e-9
            )

        recent_moves = [
            abs(cleaned[index]["close"] - cleaned[index - 1]["close"])
            for index in range(max(1, len(cleaned) - 20), len(cleaned) - 1)
        ]
        baseline_move = max(1.0, float(np.median(recent_moves))) if recent_moves else 1.0
        terminal_gap = abs(tail["close"] - previous["close"])
        stale_tail = (
            is_flat(tail)
            and all(is_flat(row) and abs(row["close"] - previous["close"]) <= 1e-9 for row in prior_tail)
            and terminal_gap > max(20.0, 10.0 * baseline_move)
        )
        if not stale_tail:
            break
        cleaned.pop()
    return cleaned


def normalize_spot(rows):
    parsed = sorted((parse_row(row) for row in rows), key=lambda row: row["minute"])
    return {
        "min": np.array([r["minute"] for r in parsed], dtype=int),
        "open": np.array([r["open"] for r in parsed], dtype=float),
        "high": np.array([r["high"] for r in parsed], dtype=float),
        "low": np.array([r["low"] for r in parsed], dtype=float),
        "close": np.array([r["close"] for r in parsed], dtype=float),
    }


class S1DState:
    """Repository-standard S1 stochastic: %D from a (12,3) oscillator."""

    def __init__(self, k_period=S1_K_PERIOD, d_period=S1_D_PERIOD):
        k_period, d_period = _normalize_s1_periods(k_period, d_period)
        self.k_period = k_period
        self.d_period = d_period
        self.stochastic = IncrementalStochastic(k_period, d_period)
        self.before_previous = None
        self.previous = None

    def update(self, row):
        value = self.stochastic.push(row["high"], row["low"], row["close"])
        turn = None
        if (
            value is not None
            and self.previous is not None
            and self.before_previous is not None
        ):
            if value > self.previous and self.previous <= self.before_previous:
                turn = "up"
            elif value < self.previous and self.previous >= self.before_previous:
                turn = "down"
        self.before_previous = self.previous
        self.previous = value
        return {"value": value, "turn": turn}


def build_s1_snapshots(
    previous_rows,
    current_rows,
    k_period=S1_K_PERIOD,
    d_period=S1_D_PERIOD,
):
    k_period, d_period = _normalize_s1_periods(k_period, d_period)
    state = S1DState(k_period, d_period)
    for row in sorted(previous_rows, key=lambda item: (item["time"], item["minute"])):
        state.update(row)

    snapshots = {}
    for row in sorted(current_rows, key=lambda item: item["minute"]):
        snapshots[row["minute"]] = state.update(row)
    return snapshots


def _resample_rows(rows, bar_minutes=1):
    """Aggregate consecutive 1m rows into complete ``bar_minutes`` candles.

    Candle-count grouping anchored at the first row, matching the legacy
    multi-timeframe engines: a group of ``bar_minutes`` rows closes at the
    last row's minute. Trailing partial groups are dropped. Identity for
    ``bar_minutes <= 1`` (same objects, same order).
    """
    if bar_minutes <= 1:
        return rows
    resampled = []
    group = None
    count = 0
    for row in rows:
        if group is None:
            group = dict(row)
            count = 1
        else:
            group["high"] = max(group["high"], row["high"])
            group["low"] = min(group["low"], row["low"])
            group["close"] = row["close"]
            group["minute"] = row["minute"]
            count += 1
        if count == bar_minutes:
            resampled.append(group)
            group = None
    return resampled


def _normalize_zone_bounds(zone_start, zone_end):
    try:
        normalized = (float(zone_start), float(zone_end))
    except (TypeError, ValueError) as exc:
        raise ValueError("zone_start and zone_end must be numeric") from exc
    if not any(
        math.isclose(normalized[0], start)
        and math.isclose(normalized[1], end)
        for start, end in ZONE_PAIRS
    ):
        raise ValueError(f"unsupported Fibonacci zone pair: {normalized}")
    return normalized


def fib_zone(setup, zone_start=S1_ZONE_START, zone_end=S1_ZONE_END):
    zone_start, zone_end = _normalize_zone_bounds(zone_start, zone_end)
    level_start = fib_price(
        setup["fib_high"], setup["fib_low"], zone_start, setup["orientation"]
    )
    level_end = fib_price(
        setup["fib_high"], setup["fib_low"], zone_end, setup["orientation"]
    )
    return min(level_start, level_end), max(level_start, level_end)


def setup_in_zone(
    setup,
    price,
    buffer=0.0,
    zone_start=S1_ZONE_START,
    zone_end=S1_ZONE_END,
):
    low, high = fib_zone(setup, zone_start, zone_end)
    return low - buffer <= price <= high + buffer


def active_strikes_50(spot, minute, side):
    """Second ITM strike in the tracked ITM scalping candidate set."""
    return active_strike_candidates(spot, minute, side)[1]


def active_strike_candidates(spot, minute, side):
    """Return first, second, and third ITM candidates (no ATM) in stable order."""
    i = list(spot["min"]).index(minute)
    p = spot["close"][i]
    atm = int(round(p / 50.0) * 50)
    offsets = (-50, -100, -150) if side == "CE" else (50, 100, 150)
    return tuple(atm + offset for offset in offsets)


def _candle(spot, minute, field):
    """Return one field (open/high/low/close) of a spot 1m bar by minute."""
    if spot is None:
        return None
    for i in range(len(spot["min"])):
        if spot["min"][i] == minute:
            return float(spot[field][i])
    return None


class UTColorState(UTBotState):
    """UT Bot state with Pine's always-colored bar semantics.

    The supplied script colors bars from ``src`` versus the trailing stop, not
    from the internal flip position. The 5m filter uses Heikin-Ashi close as
    ``src``; the 1m Fib feeds use the regular close.
    """

    def __init__(self, use_heikin_ashi=False):
        super().__init__()
        self.use_heikin_ashi = use_heikin_ashi

    def update(self, candle):
        source = (
            (candle.open + candle.high + candle.low + candle.close) / 4.0
            if self.use_heikin_ashi
            else candle.close
        )
        atr = self.atr.update(candle.high, candle.low, candle.close)
        previous_source = self.previous_source
        previous_stop = self.trailing_stop
        self.previous_source = source

        if atr is None or previous_source is None:
            return "green" if source > previous_stop else "red"

        loss = self.key * atr
        if source > previous_stop and previous_source > previous_stop:
            self.trailing_stop = max(previous_stop, source - loss)
        elif source < previous_stop and previous_source < previous_stop:
            self.trailing_stop = min(previous_stop, source + loss)
        elif source > previous_stop:
            self.trailing_stop = source - loss
        else:
            self.trailing_stop = source + loss

        return "green" if source > self.trailing_stop else "red"


class UTSwingPattern:
    """Detect one first/middle/final UT color sequence."""

    def __init__(self, name, first, middle, final, orientation, min_middle):
        self.name = name
        self.first = first
        self.middle = middle
        self.final = final
        self.orientation = orientation
        self.min_middle = min_middle
        self.previous_color = None
        self.previous_candle = None
        self.middle_candles = []
        self.pattern_candles = []

    def _reset_middle(self):
        self.middle_candles = []
        self.pattern_candles = []

    def update(self, candle, color):
        completed = None

        if self.middle_candles:
            if color == self.middle:
                self.middle_candles.append(candle)
                self.pattern_candles.append(candle)
            elif color == self.final:
                if len(self.middle_candles) >= self.min_middle:
                    pattern_candles = self.pattern_candles + [candle]
                    completed = {
                        "pattern": self.name,
                        "start_minute": self.pattern_candles[0].minute,
                        "fib_high": max(item.high for item in pattern_candles),
                        "fib_low": min(item.low for item in pattern_candles),
                        "orientation": self.orientation,
                        "completion_minute": candle.minute,
                    }
                self._reset_middle()
            else:
                self._reset_middle()
        elif (
            color == self.middle
            and self.previous_color == self.first
            and self.previous_candle is not None
        ):
            self.pattern_candles = [self.previous_candle, candle]
            self.middle_candles = [candle]

        self.previous_color = color
        self.previous_candle = candle
        return completed


class UTSwingFeed:
    """Causal 1m UT feed used to create and consume Fib setups."""

    def __init__(
        self,
        patterns,
        min_middle=MIN_PATTERN_CANDLES,
        touch_buffer=0.0,
        emit_touches=True,
        allowed_patterns=None,
        block_overlapping=False,
        replace_setups=True,
        max_setup_age=SETUP_MAX_AGE,
        min_span=MIN_SPAN,
    ):
        self.ut = UTColorState(use_heikin_ashi=False)
        self.touch_buffer = touch_buffer
        self.emit_touches = emit_touches
        self.allowed_patterns = set(allowed_patterns) if allowed_patterns is not None else None
        self.block_overlapping = block_overlapping
        self.replace_setups = replace_setups
        self.max_setup_age = max_setup_age
        self.min_span = min_span
        self.blocked_through = -1
        self.patterns = [
            UTSwingPattern(*pattern, min_middle=min_middle) for pattern in patterns
        ]
        self.setups = []

    def clear_setups(self):
        """Discard completed ranges without resetting the causal UT state."""
        self.setups = []

    def push(self, row):
        candle = Candle(
            row["open"], row["high"], row["low"], row["close"], minute=row["minute"]
        )
        color = self.ut.update(candle)

        completed = []
        for pattern in self.patterns:
            setup = pattern.update(candle, color)
            if setup is not None:
                completed.append(setup)

        events = []
        remaining = []
        for setup in self.setups:
            if (
                self.max_setup_age is not None
                and candle.minute - setup["completion_minute"] > self.max_setup_age
            ):
                continue
            high = setup["fib_high"]
            low = setup["fib_low"]
            if setup["orientation"] == "high_to_low" and candle.low < low:
                # The retracement has broken the swing base; this Fib is stale.
                continue
            if setup["orientation"] == "low_to_high" and candle.high > high:
                continue
            if not self.emit_touches:
                remaining.append(setup)
                continue
            span = high - low
            if span < self.min_span:
                continue
            entry_level = fib_price(high, low, ENTRY_LEVEL, setup["orientation"])
            if setup["orientation"] == "high_to_low":
                touched = candle.low <= entry_level + self.touch_buffer
            else:
                touched = candle.high >= entry_level - self.touch_buffer
            if touched:
                events.append({
                    **setup,
                    "minute": candle.minute,
                    "entry_level": entry_level,
                    "entry_price": candle.close,
                })
            else:
                remaining.append(setup)

        # Keep the newest pending Fib for each pattern. This prevents an old
        # overnight setup from competing with the newly completed swing.
        for setup in completed:
            allowed = self.allowed_patterns is None or setup["pattern"] in self.allowed_patterns
            if self.block_overlapping and not allowed:
                self.blocked_through = max(
                    self.blocked_through, setup["completion_minute"]
                )
                continue
            if (
                self.block_overlapping
                and setup["start_minute"] <= self.blocked_through
            ):
                continue
            if self.replace_setups:
                remaining = [
                    item for item in remaining if item["pattern"] != setup["pattern"]
                ]
            remaining.append(setup)
        self.setups = remaining
        return events


class Index5mFilter:
    """Index-only 5m Heikin-Ashi, UT, and LinReg bias filter."""

    def __init__(self, period=FILTER_PERIOD):
        self.period = period
        self.ha = HeikinAshiState()
        # UT is applied after the raw index candle has been converted to HA.
        self.ut = UTColorState(use_heikin_ashi=False)
        self.raw_closes = deque(maxlen=LINREG_LENGTH)
        self.linreg_values = deque(maxlen=LINREG_SIGNAL_LENGTH)
        self.buffer = []
        self.current_date = None
        self.current_bucket = None
        self.latest = None

    @staticmethod
    def _aggregate(rows, minute):
        return Candle(
            rows[0]["open"],
            max(item["high"] for item in rows),
            min(item["low"] for item in rows),
            rows[-1]["close"],
            minute=minute,
        )

    @staticmethod
    def _calculate(aggregate, ha_state, ut, raw_closes, linreg_values, forming):
        ha_candle = ha_state.update(aggregate)
        ha_close = ha_candle.close
        ut_color = ut.update(ha_candle)

        # The LinReg script is applied to the Heikin-Ashi index chart, so its
        # close series is the HA close rather than the underlying raw close.
        raw_closes.append(ha_close)
        bclose = linreg_value(raw_closes)
        if bclose is not None:
            linreg_values.append(bclose)
        plot = (
            sum(linreg_values) / len(linreg_values)
            if len(linreg_values) >= LINREG_SIGNAL_LENGTH
            else None
        )
        return {
            "confirmed_minute": aggregate.minute,
            "ha_open": ha_candle.open,
            "ha_high": ha_candle.high,
            "ha_low": ha_candle.low,
            "ha_close": ha_close,
            "linreg_plot": plot,
            "ut_color": ut_color,
            "forming": forming,
        }

    def _commit_buffer(self):
        if not self.buffer:
            return
        aggregate = self._aggregate(self.buffer, self.current_bucket)
        self.latest = self._calculate(
            aggregate,
            self.ha,
            self.ut,
            self.raw_closes,
            self.linreg_values,
            forming=False,
        )
        self.buffer = []

    def push(self, row):
        row_date = row["time"].split(" ")[0]
        if self.current_date != row_date:
            self._commit_buffer()
            self.buffer = []
            self.current_bucket = None
            self.current_date = row_date

        if row["minute"] < FILTER_SESSION_START:
            return self.latest

        bucket = row["minute"] - (row["minute"] % self.period)
        if self.current_bucket is None:
            self.current_bucket = bucket
        elif bucket != self.current_bucket:
            self._commit_buffer()
            self.current_bucket = bucket

        self.buffer.append(row)
        # Recompute a copy of the indicator state so the current forming
        # candle participates without being committed twice at its close.
        live_ha = copy.deepcopy(self.ha)
        live_ut = copy.deepcopy(self.ut)
        live_closes = deque(self.raw_closes, maxlen=LINREG_LENGTH)
        live_values = deque(self.linreg_values, maxlen=LINREG_SIGNAL_LENGTH)
        aggregate = self._aggregate(self.buffer, row["minute"])
        snapshot = self._calculate(
            aggregate,
            live_ha,
            live_ut,
            live_closes,
            live_values,
            forming=True,
        )
        if row["minute"] % FILTER_PERIOD == FILTER_PERIOD - 1:
            self.ut = live_ut
            self.raw_closes = live_closes
            self.linreg_values = live_values
            self.latest = {**snapshot, "forming": False}
            self.buffer = []
            return self.latest
        return snapshot


def build_index_filter(previous_rows, current_rows, period=FILTER_PERIOD):
    feed = Index5mFilter(period=period)
    for row in sorted(previous_rows, key=lambda item: (item["time"], item["minute"])):
        feed.push(row)

    snapshots = {}
    for row in sorted(current_rows, key=lambda item: item["minute"]):
        feed.push(row)
        snapshot = feed.latest
        snapshots[row["minute"]] = dict(snapshot) if snapshot is not None else None
    return snapshots


def index_filter_allows(snapshot, side):
    if snapshot is None or snapshot["linreg_plot"] is None:
        return False
    above = snapshot["ha_close"] > snapshot["linreg_plot"]
    below = snapshot["ha_close"] < snapshot["linreg_plot"]
    if side == "CE":
        return above and snapshot["ut_color"] == "green"
    return below and snapshot["ut_color"] == "red"


def extract_day_events(
    day,
    *,
    cache_loader=None,
    cache_dir=None,
    min_span=MIN_SPAN,
    touch_buffer=0.0,
    setup_max_age=SETUP_MAX_AGE,
    zone_start=S1_ZONE_START,
    zone_end=S1_ZONE_END,
    s1_k_period=S1_K_PERIOD,
    s1_d_period=S1_D_PERIOD,
    bar_minutes=1,
    filter_period=FILTER_PERIOD,
    debug=True,
):
    """Build the causal Smart Fib event stream and source bars for one day.

    ``cache_loader`` is an adapter seam for historical datasets that expose the
    same day-cache payload without requiring the Flattrade replay cache. The
    default path remains unchanged for live-cache regression runs. The returned
    records intentionally retain actual option symbols and current OHLC rows so
    downstream evaluators can replay exits without rebuilding signals.

    ``bar_minutes`` resamples the index and option feeds into 1m/2m/3m/5m
    candles (identity for 1); ``filter_period`` sets the index bias filter
    bucket width (the x5 rule: 1m->5, 2m->10, 3m->15, 5m->25). The 1m
    default reproduces the champion signal stream exactly.
    """
    zone_start, zone_end = _normalize_zone_bounds(zone_start, zone_end)
    s1_k_period, s1_d_period = _normalize_s1_periods(s1_k_period, s1_d_period)
    load_cache = load_day_cache if cache_loader is None else cache_loader
    cache_root = GLOBAL_CACHE_DIR if cache_dir is None else cache_dir
    cache = load_cache(cache_root, date.fromisoformat(day))
    if cache is None:
        return {}
    target_date = date.fromisoformat(day)
    target_text = target_date.strftime("%d-%m-%Y")
    current_spot_rows = clean_terminal_quotes(
        [parse_row(row) for row in cache["spot_rows"]]
    )
    spot = normalize_spot(current_spot_rows)

    records = {}
    for key, info in cache["contracts"].items():
        side, strike_text = key.split(":", 1)
        strike = int(strike_text)
        rows = [parse_row(row) for row in info["rows"]]
        records[(side, strike)] = {
            "symbol": info.get("tsym") or f"{side}:{strike}",
            "previous": sorted(
                [r for r in rows if r["time"].split(" ")[0] != target_text],
                key=lambda row: (row["time"].split(" ")[0], row["minute"]),
            ),
            "current": sorted(
                [r for r in rows if r["time"].split(" ")[0] == target_text],
                key=lambda row: row["minute"],
            ),
        }

    bars = {
        key: {row["minute"]: row for row in rec["current"]}
        for key, rec in records.items()
    }

    events = []
    prev_date = target_date - timedelta(days=1)
    prev_cache = load_cache(cache_root, prev_date)
    previous_spot_rows = (
        clean_terminal_quotes([parse_row(row) for row in prev_cache["spot_rows"]])
        if prev_cache
        else []
    )
    index_filter = build_index_filter(
        previous_spot_rows, current_spot_rows, period=filter_period
    )
    tf_current_rows = _resample_rows(current_spot_rows, bar_minutes)
    tf_previous_rows = _resample_rows(previous_spot_rows, bar_minutes)
    tf_rows_by_minute = {row["minute"]: row for row in tf_current_rows}
    tf_close_minutes = set(tf_rows_by_minute)

    filtered = []
    signals = []          # list of dicts (timeframe-agnostic)
    index_retraced = False
    retraced_at = 9999

    # ===================== CASE 1: INDEX 1m UT Fib =====================
    index_feed = UTSwingFeed(
        [
            ("bullish", "red", "green", "red", "high_to_low"),
            ("bearish", "green", "red", "green", "low_to_high"),
        ],
        emit_touches=False,
        replace_setups=False,
        min_span=min_span,
        max_setup_age=setup_max_age,
        touch_buffer=touch_buffer,
    )
    for row in tf_previous_rows:
        index_feed.push(row)
    index_feed.clear_setups()

    index_s1 = build_s1_snapshots(
        tf_previous_rows,
        tf_current_rows,
        k_period=s1_k_period,
        d_period=s1_d_period,
    )
    consumed_index_setups = set()
    pending_index = []
    for row in current_spot_rows:
        m = row["minute"]
        if bar_minutes > 1 and m not in tf_close_minutes:
            continue
        index_feed.push(tf_rows_by_minute[m])
        snapshot = index_filter.get(m)

        active_pending = []
        for pending in pending_index:
            if m - pending["setup"]["completion_minute"] > setup_max_age:
                continue
            if index_filter_allows(snapshot, pending["side"]):
                candidates = [
                    (key, strike)
                    for key, strike in pending["candidates"]
                    if key in bars and m in bars[key]
                ]
                if not candidates:
                    active_pending.append(pending)
                    continue
                setup = pending["setup"]
                index_retraced = True
                retraced_at = m
                zone_low, zone_high = fib_zone(setup, zone_start, zone_end)
                for key, strike in candidates:
                    signals.append({
                        "side": pending["side"],
                        "strike": strike,
                        "symbol": records[key]["symbol"],
                        "minute": m,
                        "signal_minute": pending["trigger_minute"],
                        "option_entry": bars[key][m]["close"],
                        "fib_source": "index",
                        "trigger": "index",
                        "fib_high": setup["fib_high"],
                        "fib_low": setup["fib_low"],
                        "orientation": setup["orientation"],
                        "profit_on_rise": True,
                        "price_profit_on_rise": pending["side"] == "CE",
                        "dynamic_target": True,
                        "option_delta": OPTION_DELTA,
                        "s1_value": pending["s1"]["value"],
                        "s1_turn": pending["s1"]["turn"],
                        "zone_low": zone_low,
                        "zone_high": zone_high,
                        "zone_start": zone_start,
                        "zone_end": zone_end,
                    })
                continue
            active_pending.append(pending)
        pending_index = active_pending

        s1_snapshot = index_s1.get(m)
        if s1_snapshot is None or s1_snapshot["turn"] is None:
            continue

        latest_index_setups = {}
        for candidate in index_feed.setups:
            current = latest_index_setups.get(candidate["pattern"])
            if current is None or candidate["completion_minute"] > current["completion_minute"]:
                latest_index_setups[candidate["pattern"]] = candidate

        for setup in sorted(
            latest_index_setups.values(),
            key=lambda item: item["completion_minute"],
        ):
            setup_id = (setup["pattern"], setup["completion_minute"])
            # Multi-touch: same fib leg can fire multiple entries
            # as price oscillates through the zone. One-position-at-a-time
            # is enforced downstream by simulate(concurrent=False).
            if setup["completion_minute"] >= m:
                continue
            if setup["fib_high"] - setup["fib_low"] < min_span:
                continue
            if not setup_in_zone(
                setup,
                row["close"],
                touch_buffer,
                zone_start,
                zone_end,
            ):
                continue

            side = "CE" if setup["pattern"] == "bullish" else "PE"
            expected_turn = "up" if side == "CE" else "down"
            if s1_snapshot["turn"] != expected_turn:
                continue
            consumed_index_setups.add(setup_id)

            candidates = [
                ((side, strike), strike)
                for strike in active_strike_candidates(spot, m, side)
                if (side, strike) in bars and m in bars[(side, strike)]
            ]
            if not candidates:
                continue

            if not index_filter_allows(snapshot, side):
                pending_index.append({
                    "setup": setup,
                    "side": side,
                    "candidates": candidates,
                    "trigger_minute": m,
                    "s1": s1_snapshot,
                })
                continue

            index_retraced = True
            retraced_at = m
            zone_low, zone_high = fib_zone(setup, zone_start, zone_end)
            for key, strike in candidates:
                signals.append({
                    "side": side,
                    "strike": strike,
                    "symbol": records[key]["symbol"],
                    "minute": m,
                    "signal_minute": m,
                    "option_entry": bars[key][m]["close"],
                    "fib_source": "index",
                    "trigger": "index",
                    "fib_high": setup["fib_high"],
                    "fib_low": setup["fib_low"],
                    "orientation": setup["orientation"],
                    "profit_on_rise": True,
                    "price_profit_on_rise": side == "CE",
                    "dynamic_target": True,
                    "option_delta": OPTION_DELTA,
                    "s1_value": s1_snapshot["value"],
                    "s1_turn": s1_snapshot["turn"],
                    "zone_low": zone_low,
                    "zone_high": zone_high,
                    "zone_start": zone_start,
                    "zone_end": zone_end,
                })

    # ===================== CASE 2: OPTION 1m RGR Fib =====================
    # Both CE and PE charts must independently form RGR. The common index bias
    # selects the side, while the selected option's own Fib zone and bullish
    # S1 turn-up provide the trigger.
    option_candidates = []
    for (side, strike), record in records.items():
        feed = UTSwingFeed(
            [("option_rgr", "red", "green", "red", "high_to_low")],
            emit_touches=False,
            min_span=min_span,
            max_setup_age=setup_max_age,
            touch_buffer=touch_buffer,
        )
        option_prev_rows = _resample_rows(record["previous"], bar_minutes)
        option_current_rows = _resample_rows(record["current"], bar_minutes)
        for row in option_prev_rows:
            feed.push(row)
        option_s1 = build_s1_snapshots(
            option_prev_rows,
            option_current_rows,
            k_period=s1_k_period,
            d_period=s1_d_period,
        )
        consumed_setups = set()
        for row in option_current_rows:
            minute = row["minute"]
            feed.push(row)
            s1_snapshot = option_s1.get(minute)
            index_snapshot = index_filter.get(minute)
            if s1_snapshot is None or s1_snapshot["turn"] is None:
                continue
            if index_snapshot is None:
                continue
            bias_side = (
                "CE" if index_filter_allows(index_snapshot, "CE") else
                "PE" if index_filter_allows(index_snapshot, "PE") else
                None
            )
            if bias_side != side:
                continue
            if s1_snapshot["turn"] != "up":
                continue
            if minute not in bars[(side, strike)]:
                continue
            if strike not in active_strike_candidates(spot, minute, side):
                continue

            for setup in sorted(
                feed.setups,
                key=lambda item: item["completion_minute"],
            ):
                setup_id = (setup["pattern"], setup["completion_minute"])
                if setup_id in consumed_setups:
                    continue
                if setup["completion_minute"] >= minute:
                    continue
                if setup["fib_high"] - setup["fib_low"] < min_span:
                    continue
                if not setup_in_zone(
                    setup,
                    row["close"],
                    touch_buffer,
                    zone_start,
                    zone_end,
                ):
                    continue
                consumed_setups.add(setup_id)
                option_candidates.append((minute, side, strike, setup, s1_snapshot))

    option_candidates.sort(key=lambda item: (item[0], 0 if item[1] == "CE" else 1))
    touched = set()
    for m, side, strike, option_event, s1_snapshot in option_candidates:
        key = (side, strike)
        candidate_id = (key, option_event["completion_minute"])
        if candidate_id in touched or m not in bars[key]:
            continue
        touched.add(candidate_id)
        snapshot = index_filter.get(m)
        if not index_filter_allows(snapshot, side):
            filtered.append({
                "minute": m,
                "side": side,
                "strike": strike,
                "trigger": "option",
                "zone_start": zone_start,
                "zone_end": zone_end,
                "snapshot": snapshot,
            })
            continue
        row = bars[key][m]
        zone_low, zone_high = fib_zone(option_event, zone_start, zone_end)
        signals.append({
            "side": side,
            "strike": strike,
            "symbol": records[key]["symbol"],
            "minute": m,
            "option_entry": row["close"],
            "fib_source": "option",
            "trigger": "option",
            "fib_high": option_event["fib_high"],
            "fib_low": option_event["fib_low"],
            "orientation": option_event["orientation"],
            "profit_on_rise": True,
            "price_profit_on_rise": True,
            "dynamic_target": True,
            "option_delta": OPTION_DELTA,
            "s1_value": s1_snapshot["value"],
            "s1_turn": s1_snapshot["turn"],
            "zone_low": zone_low,
            "zone_high": zone_high,
            "zone_start": zone_start,
            "zone_end": zone_end,
        })

    index_bars = {int(spot["min"][i]): spot_row(spot, i) for i in range(len(spot["min"]))}

    n_idx = sum(1 for s in signals if s["trigger"] == "index")
    n_opt = sum(1 for s in signals if s["trigger"] == "option")
    if debug:
        print(
            f"[debug {day}] signals={len(signals)} idx={n_idx} opt={n_opt} "
            f"index_retraced={index_retraced} retraced_at={retraced_at}",
            flush=True,
        )
        for s in signals:
            signal_fib = ""
            if "signal_fib_high" in s:
                signal_fib = f" signal_fib=({s['signal_fib_high']:.2f},{s['signal_fib_low']:.2f})"
            print(
                f"  sig {s['minute']:04d} {s['side']} {s['strike']} "
                f"trig={s['trigger']} entry={s['option_entry']:.2f} "
                f"fib=({s['fib_high']:.2f},{s['fib_low']:.2f}) "
                f"zone=({s.get('zone_low')},{s.get('zone_high')}) "
                f"s1={s.get('s1_value')} turn={s.get('s1_turn')}{signal_fib}",
                flush=True,
            )
        for item in filtered:
            snapshot = item["snapshot"] or {}
            print(
                f"  filtered {item['minute']:04d} {item['side']} {item['strike']} "
                f"trig={item['trigger']} ut={snapshot.get('ut_color')} "
                f"ha={snapshot.get('ha_close')} plot={snapshot.get('linreg_plot')}",
                flush=True,
            )

    return {
        "day": day,
        "signals": signals,
        "events": list(signals),
        "bars": bars,
        "index_bars": index_bars,
        "spot": spot,
        "records": records,
        "filtered": filtered,
        "current_spot_rows": current_spot_rows,
        "previous_spot_rows": previous_spot_rows,
        "zone_start": zone_start,
        "zone_end": zone_end,
    }


def process_day(
    day,
    timeframe_modes,
    target_levels,
    stop_levels,
    *,
    cache_loader=None,
    cache_dir=None,
    min_span=MIN_SPAN,
    touch_buffer=0.0,
    setup_max_age=SETUP_MAX_AGE,
    zone_start=S1_ZONE_START,
    zone_end=S1_ZONE_END,
    option_point_threshold=OPTION_POINT_THRESHOLD,
    fallback_target_level=0.0,
    s1_k_period=S1_K_PERIOD,
    s1_d_period=S1_D_PERIOD,
    debug=True,
    brokerage_per_order=BROKERAGE_PER_ORDER,
    fixed_cost_per_trade=None,
):
    """Replay one day using the Smart Fib signal path.

    Signal extraction is shared with GPU-first evaluators through
    :func:`extract_day_events`; this wrapper preserves the historical output
    shape and the default CPU execution behavior.
    """
    s1_k_period, s1_d_period = _normalize_s1_periods(s1_k_period, s1_d_period)
    prepared = extract_day_events(
        day,
        cache_loader=cache_loader,
        cache_dir=cache_dir,
        min_span=min_span,
        touch_buffer=touch_buffer,
        setup_max_age=setup_max_age,
        zone_start=zone_start,
        zone_end=zone_end,
        s1_k_period=s1_k_period,
        s1_d_period=s1_d_period,
        debug=debug,
    )
    events = []
    for tf in timeframe_modes:
        for signal in prepared["signals"]:
            events.append({**signal, "timeframe": tf})

    output = {}
    for tf in timeframe_modes:
        for tp in target_levels:
            for sl in stop_levels:
                key = f"smart-fib|{tf}|tp{tp}|sl{sl}"
                trades = simulate(
                    events, prepared["bars"], prepared["index_bars"], prepared["spot"],
                    tf, tp, sl,
                    concurrent=False,
                    option_point_threshold=option_point_threshold,
                    fallback_target_level=fallback_target_level,
                    brokerage_per_order=brokerage_per_order,
                    fixed_cost_per_trade=fixed_cost_per_trade,
                )
                for t in trades:
                    t["date"] = day
                    t["zone_start"] = zone_start
                    t["zone_end"] = zone_end
                output[key] = trades
    return output


def compute_stats(trades, days_count):
    wins = [t for t in trades if t["rs_net"] > 0]
    losses = [t for t in trades if t["rs_net"] <= 0]
    loss_total = abs(sum(t["rs_net"] for t in losses))
    win_total = sum(t["rs_net"] for t in wins)
    net_rs = sum(t["rs_net"] for t in trades)
    net_pts = sum(t["points"] for t in trades)
    wr = round(len(wins) / len(trades) * 100, 2) if trades else 0.0
    pf = round(win_total / loss_total, 4) if loss_total else (float("inf") if win_total else 0.0)
    return {
        "trades": len(trades),
        "win_rate": wr,
        "net_rs": round(net_rs, 2),
        "net_points": round(net_pts, 2),
        "profit_factor": pf,
    }


def main():
    days = ["2026-08-12", "2026-08-13", "2026-08-14"]
    timeframe_modes = ("1m", "2m", "3m", "5m", "combined")
    target_levels = (0.29,)
    stop_levels = (1.155, 1.25)

    aggregated = {}
    for day in days:
        res = process_day(day, timeframe_modes, target_levels, stop_levels)
        for k, v in res.items():
            aggregated.setdefault(k, []).extend(v)

    print(f"{'Configuration':28s} | {'Trades':6s} | {'WinRate':7s} | {'Net Rs':12s} | {'Net Pts':8s} | {'PF':6s}")
    print("-" * 80)
    for k in sorted(aggregated.keys()):
        st = compute_stats(aggregated[k], len(days))
        print(f"{k:28s} | {st['trades']:6d} | {st['win_rate']:6.1f}% | {st['net_rs']:+12,.2f} | {st['net_points']:+8.2f} | {st['profit_factor']:6.2f}")

    # Detailed trade list for 2026-08-14
    print(f"\n=== {STRATEGY_NAME}: 2026-08-14 trade list (1m, tp0.29, sl1.155) ===")
    key = "smart-fib|1m|tp0.29|sl1.155"
    trades = sorted(aggregated.get(key, []), key=lambda t: (t["date"], t["entry_min"]))
    for t in trades:
        if t["date"] == "2026-08-14":
            print("  %02d:%02d %s %s entry=%.2f exit=%.2f %s pts=%+.2f net=%+.2f src=%s" % (
                t["entry_min"] // 60, t["entry_min"] % 60, t["side"], t["symbol"],
                t["entry"], t["exit"], t["reason"], t["points"], t["rs_net"],
                t.get("fib_source", "?"),
            ))


if __name__ == "__main__":
    main()
