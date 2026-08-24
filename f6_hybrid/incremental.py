"""Incremental F6 signal-state caching and pointer primitives."""

import copy
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Sequence, TypeVar

import numpy as np

import grid_optimize_f6_atr as grid


@dataclass(frozen=True)
class SignalKey:
    """All values that can change the generated F6 signal stream."""

    day: str
    previous_day: str
    s1_k: int
    s1_d: int
    s4_k: int
    atr_period: int
    f6_s4_thresh: float
    f6_s1_thresh: float

    def as_tuple(self) -> tuple[Hashable, ...]:
        return (
            self.day,
            self.previous_day,
            self.s1_k,
            self.s1_d,
            self.s4_k,
            self.atr_period,
            self.f6_s4_thresh,
            self.f6_s1_thresh,
        )


@dataclass(frozen=True)
class ExecutionKey:
    """Values that change only the stateful position/exit simulation."""

    atr_sl_mult: float
    atr_tp_mult: float
    consec_loss: int

    def as_tuple(self) -> tuple[float, float, int]:
        return self.atr_sl_mult, self.atr_tp_mult, self.consec_loss


@dataclass
class DaySignalState:
    """All pre-trade state produced by the reference signal scan for one day."""

    day: str
    spot: object
    prefix: str
    trackers: dict[str, object]
    slices: dict[str, dict[str, np.ndarray]]
    pmtrig: dict[int, list[tuple]]


T = TypeVar("T")


class SignalStateCache(Generic[T]):
    """Memoize signal state while keeping construction observable for tests."""

    def __init__(self) -> None:
        self._states: dict[SignalKey, T] = {}
        self.build_count = 0

    def get_or_build(self, key: SignalKey, builder: Callable[[], T]) -> T:
        state = self._states.get(key)
        if state is None:
            state = builder()
            self._states[key] = state
            self.build_count += 1
        return state

    def clear(self) -> None:
        self._states.clear()
        self.build_count = 0


class MinuteCursor(Generic[T]):
    """Read a sorted minute/value series with a forward-only cursor."""

    def __init__(self, minutes: Sequence[int], values: Sequence[T]) -> None:
        if len(minutes) != len(values):
            raise ValueError("minutes and values must have the same length")
        if any(left >= right for left, right in zip(minutes, minutes[1:])):
            raise ValueError("minutes must be strictly increasing")
        self.minutes = minutes
        self.values = values
        self.index = 0
        self._last_query: int | None = None

    def at(self, minute: int) -> T | None:
        if self._last_query is not None and minute < self._last_query:
            raise ValueError("MinuteCursor cannot move backwards")
        self._last_query = minute
        while self.index < len(self.minutes) and self.minutes[self.index] < minute:
            self.index += 1
        if self.index >= len(self.minutes) or self.minutes[self.index] != minute:
            return None
        return self.values[self.index]

    def latest_at_or_before(self, minute: int) -> T | None:
        """Return the latest value whose timestamp is no greater than minute."""
        if self._last_query is not None and minute < self._last_query:
            raise ValueError("MinuteCursor cannot move backwards")
        self._last_query = minute
        while self.index < len(self.minutes) and self.minutes[self.index] <= minute:
            self.index += 1
        return None if self.index == 0 else self.values[self.index - 1]


def signal_key_for(day: str, previous_day: str, params: dict) -> SignalKey:
    """Build the cache key from every parameter used by signal generation."""
    return SignalKey(
        day=day,
        previous_day=previous_day,
        s1_k=params["s1_k"],
        s1_d=params["s1_d"],
        s4_k=params["s4_k"],
        atr_period=params["atr_period"],
        f6_s4_thresh=params["f6_s4_thresh"],
        f6_s1_thresh=params["f6_s1_thresh"],
    )


def build_day_signal_state(
    day: str,
    fpath: str,
    fprev: str,
    params: dict,
    spot: object,
) -> DaySignalState | None:
    """Extract the reference engine's pre-trade scan for one day.

    This deliberately mirrors ``grid_optimize_f6_atr.process_day`` through its
    ``pmtrig`` construction. It does not apply execution parameters.
    """
    if spot is None or not fpath:
        return None
    gc = grid.cached_day(fpath)
    if not gc:
        return None
    fsym = next(iter(gc))
    match = grid.SYM_RE.match(fsym)
    if not match:
        return None
    prefix = match.group(1)
    sp0 = grid.latest_spot(spot, 555) or grid.latest_spot(spot, 560)
    if sp0 is None:
        return None
    atm0 = int(round(sp0 / 50) * 50)
    target_strikes = set(range(atm0 - 250, atm0 + 300, 50))

    def filtered(data):
        return {
            sym: values
            for sym, values in data.items()
            if (symbol_match := grid.SYM_RE.match(sym))
            and int(symbol_match.group(2)) in target_strikes
        }

    gu = filtered(gc)
    gp = {}
    if fprev:
        previous = grid.cached_day(fprev)
        if previous:
            gp = filtered(previous)

    trackers = {}
    for sym, values in gp.items():
        trackers[sym] = grid.MTFTracker(params)
        for i in range(len(values["min"])):
            trackers[sym].push_1m(
                grid.Candle(
                    open=values["open"][i],
                    high=values["high"][i],
                    low=values["low"][i],
                    close=values["close"][i],
                    minute=values["min"][i],
                )
            )

    pmtrig = {}
    slices = {}
    for sym, values in gu.items():
        if sym not in trackers:
            trackers[sym] = grid.MTFTracker(params)
        tracker = trackers[sym]
        slices[sym] = values
        symbol_match = grid.SYM_RE.match(sym)
        if not symbol_match:
            continue
        strike, side = int(symbol_match.group(2)), symbol_match.group(3)
        for i in range(len(values["min"])):
            minute = values["min"][i]
            candle = grid.Candle(
                open=values["open"][i],
                high=values["high"][i],
                low=values["low"][i],
                close=values["close"][i],
                minute=minute,
            )
            for tf, is_rev, stype, px, atr_val in tracker.push_1m(candle):
                pmtrig.setdefault(minute, []).append(
                    (
                        side,
                        strike,
                        sym,
                        px,
                        is_rev,
                        tf,
                        grid.TF_SPECS[tf][2],
                        grid.TF_SPECS[tf][3],
                        atr_val,
                    )
                )

    return DaySignalState(
        day=day,
        spot=spot,
        prefix=prefix,
        trackers=trackers,
        slices=slices,
        pmtrig=pmtrig,
    )


def simulate_day_signal_state(state: DaySignalState, params: dict) -> list[dict]:
    """Run the reference position loop against a cached signal state."""
    if not state.pmtrig:
        return []

    # The reference exit path mutates divergence state. Each execution variant
    # therefore receives an isolated copy of the final tracker state.
    trackers = copy.deepcopy(state.trackers)
    spot_cursor = MinuteCursor(state.spot["min"], state.spot["close"])
    daily_loss_pts = grid.DAILY_LOSS_RS / grid.LOT_SIZE
    consec_loss = params["consec_loss"]
    sl_mult, tp_mult = params["atr_sl_mult"], params["atr_tp_mult"]

    def bar_cursor(values):
        bars = tuple(
            zip(values["open"], values["high"], values["low"], values["close"])
        )
        return MinuteCursor(values["min"], bars)

    def ainfo(side, minute):
        spot_px = spot_cursor.latest_at_or_before(minute)
        if spot_px is None:
            return None
        atm = int(round(spot_px / 50) * 50)
        strike = atm + (grid.CE_OFFSET if side == "CE" else grid.PE_OFFSET)
        symbol = f"{state.prefix}{strike}{side}"
        values = state.slices.get(symbol)
        return (symbol, values, strike) if values is not None else None

    trades = []
    pos = None
    dpnl = 0.0
    closs = 0
    shut = False
    for minute in range(grid.SESSION_START, grid.DAY_LAST + 1):
        if pos is not None:
            held = pos["cursor"].at(minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c)
                pos["duration_min"] += 1
                if dpnl * grid.LOT_SIZE + (c - pos["entry"]) * grid.LOT_SIZE <= grid.DAILY_LOSS_RS:
                    pts = round(c - pos["entry"], 2)
                    trades.append(
                        {
                            "date": state.day,
                            "entry_min": pos["entry_min"],
                            "exit_min": minute,
                            "side": pos["side"],
                            "symbol": pos["symbol"],
                            "entry": pos["entry"],
                            "exit": c,
                            "pts": pts,
                            "rs": round(pts * grid.LOT_SIZE),
                            "sl_pts": pos["sl_pts"],
                            "tp_pts": pos["tp_pts"],
                            "reason": "SHUTDOWN_LOSS",
                            "duration_min": pos["duration_min"],
                            "tf": pos["tf"],
                        }
                    )
                    dpnl += pts
                    pos = None
                    shut = True
                    continue
                ex, reason = None, ""
                has_target = pos.get("tgt") is not None
                if has_target and h >= pos["tgt"] and l <= pos["sl"]:
                    ex, reason = pos["sl"], "SL"
                elif has_target and h >= pos["tgt"]:
                    ex, reason = pos["tgt"], "TP"
                elif l <= pos["sl"]:
                    ex, reason = pos["sl"], "SL"
                if ex is None:
                    tracker = trackers.get(pos["symbol"])
                    if tracker:
                        tracker_1m = tracker.trackers["1m"]
                        tracker_1m.div.update(c, tracker_1m.prev_s1, low_price=l, high_price=h)
                        if tracker_1m.div.has_bearish_peak_divergence():
                            ex, reason = c, "BEARISH_PEAK_REVERSAL"
                if ex is not None:
                    pts = round(ex - pos["entry"], 2)
                    trades.append(
                        {
                            "date": state.day,
                            "entry_min": pos["entry_min"],
                            "exit_min": minute,
                            "side": pos["side"],
                            "symbol": pos["symbol"],
                            "entry": pos["entry"],
                            "exit": ex,
                            "pts": pts,
                            "rs": round(pts * grid.LOT_SIZE),
                            "reason": reason,
                            "sl_pts": pos["sl_pts"],
                            "tp_pts": pos["tp_pts"],
                            "duration_min": pos["duration_min"],
                            "tf": pos["tf"],
                        }
                    )
                    dpnl += pts
                    closs = closs + 1 if pts <= 0 else 0
                    if closs >= consec_loss or dpnl <= daily_loss_pts:
                        shut = True
                    pos = None
        if minute >= grid.SESSION_END and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            trades.append(
                {
                    "date": state.day,
                    "entry_min": pos["entry_min"],
                    "exit_min": minute,
                    "side": pos["side"],
                    "symbol": pos["symbol"],
                    "entry": pos["entry"],
                    "exit": pos["last_px"],
                    "pts": pts,
                    "rs": round(pts * grid.LOT_SIZE),
                    "sl_pts": pos["sl_pts"],
                    "tp_pts": pos["tp_pts"],
                    "reason": "EOD",
                    "duration_min": pos["duration_min"],
                    "tf": pos["tf"],
                }
            )
            dpnl += pts
            pos = None
            break
        if pos is not None or shut or minute >= grid.SESSION_END:
            continue

        for signal in state.pmtrig.get(minute, []):
            sig_side, sig_stk, sig_sym, c_px, is_rev, tf, sl_pts, tp_pts, atr_val = signal
            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                if is_rev:
                    opposite = "PE" if sig_side == "CE" else "CE"
                    ai2 = ainfo(opposite, minute)
                    if ai2 is None:
                        continue
                    symbol, values, _ = ai2
                else:
                    symbol, values = sig_sym, ai[1]
                cursor = bar_cursor(values)
                bar = cursor.at(minute)
                if bar:
                    entry = float(bar[3])
                    if atr_val and atr_val > 0.5:
                        sl_use = atr_val * sl_mult
                        tp_use = atr_val * tp_mult
                    else:
                        sl_use = sl_pts
                        tp_use = tp_pts
                    pos = {
                        "side": opposite if is_rev else sig_side,
                        "symbol": symbol,
                        "slice": values,
                        "cursor": cursor,
                        "entry": entry,
                        "sl": entry - sl_use,
                        "tgt": entry + tp_use,
                        "sl_pts": round(sl_use, 2),
                        "tp_pts": round(tp_use, 2),
                        "entry_min": minute,
                        "last_px": entry,
                        "duration_min": 0,
                        "tf": tf,
                    }
                    break
    return trades


def build_or_get_day_signal_state(
    cache: SignalStateCache[DaySignalState],
    day: str,
    fpath: str,
    fprev: str,
    params: dict,
    spot: object,
) -> DaySignalState | None:
    """Build one signal state and make reuse explicit to callers."""
    key = signal_key_for(day, fprev, params)
    return cache.get_or_build(
        key,
        lambda: build_day_signal_state(day, fpath, fprev, params, spot),
    )


_WORKER_CANDIDATES: list[dict] = []


def _init_incremental_worker(spot_all: dict, candidates: list[dict]) -> None:
    global _WORKER_CANDIDATES
    grid.init_worker_local(spot_all)
    _WORKER_CANDIDATES = candidates


def _process_candidate_day(args):
    day, fpath, fprev = args
    spot = grid.GLOBAL_SPOT.get(day)
    cache: SignalStateCache[DaySignalState] = SignalStateCache()
    day_results = []
    for params in _WORKER_CANDIDATES:
        state = build_or_get_day_signal_state(
            cache, day, fpath, fprev, params, spot
        )
        day_results.append([] if state is None else simulate_day_signal_state(state, params))
    return day, day_results, cache.build_count


def run_incremental_candidates(
    candidates: list[dict],
    days: Sequence[str],
    files: dict[str, str],
    spot_all: dict,
    workers: int = 8,
) -> tuple[list[list[dict]], int]:
    """Evaluate candidates while reusing signal state within each day.

    The returned trade lists are aligned with ``candidates``. ``signal_builds``
    counts actual signal-state constructions and is used as a benchmark guard.
    """
    workers = max(1, min(8, int(workers)))
    if not candidates or not days:
        return [[] for _ in candidates], 0

    days = list(days)
    tasks = [
        (
            day,
            str(files[day]),
            str(files[days[index - 1]]) if index else "",
        )
        for index, day in enumerate(days)
    ]
    results = [[] for _ in candidates]
    signal_builds = 0
    with grid.Pool(
        processes=workers,
        initializer=_init_incremental_worker,
        initargs=(spot_all, candidates),
    ) as pool:
        for _, day_results, day_builds in pool.imap(_process_candidate_day, tasks):
            signal_builds += day_builds
            for index, trades in enumerate(day_results):
                results[index].extend(trades)
    return results, signal_builds
