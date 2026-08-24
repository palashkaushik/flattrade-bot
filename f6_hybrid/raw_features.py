"""Factorized F6 feature generation for signal-parameter sweeps."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Sequence

import grid_optimize_f6_atr as grid
from f6_hybrid.incremental import DaySignalState, simulate_day_signal_state


@dataclass(frozen=True)
class FeatureBar:
    minute: int
    close: float
    atr: float | None
    s1: float | None
    s2: float | None
    s3: float | None
    s4: float | None
    bullish_divergence: bool
    pin_break: bool
    embedded: bool
    tf: str
    tf_sl: float
    tf_tp: float
    bullish_divergence_id: tuple[int, int] | None = None


@dataclass(frozen=True)
class BaseKey:
    day: str
    previous_day: str
    s1_k: int
    s1_d: int
    s4_k: int
    atr_period: int
    use_divergence: bool = True


@dataclass
class BaseDayState:
    day: str
    spot: object
    prefix: str
    trackers: dict[str, object]
    slices: dict[str, dict]
    features: dict[str, dict[str, tuple[FeatureBar, ...]]]
    warmup_features: dict[str, dict[str, tuple[FeatureBar, ...]]] = field(
        default_factory=dict
    )


class _BaseTFTracker:
    def __init__(self, lookback: int, params: dict):
        self.lookback = lookback
        self.stoch = grid.ParamStoch(params["s1_k"], params["s1_d"], params["s4_k"])
        self.div = grid.DivergenceEngine()
        self.hist = []
        self.prev_s1 = None
        self.s4_emb = 0
        self.atr = grid.IncrementalATR(params["atr_period"])
        self.use_divergence = params.get("use_divergence", True)

    def push(self, candle: grid.Candle, tf: str) -> FeatureBar:
        self.hist.append(candle)
        if len(self.hist) > 40:
            self.hist.pop(0)
        values = self.stoch.push(candle.high, candle.low, candle.close)
        s1, s2, s3, s4 = (
            values["s1d"], values["s2d"], values["s3d"], values["s4d"]
        )
        atr_value = self.atr.update(candle.high, candle.low, candle.close)
        self.prev_s1 = s1
        if s4 is not None:
            self.s4_emb = self.s4_emb + 1 if s4 <= 20 else 0
        embedded = self.s4_emb > 25
        self.div.update(candle.close, s1, low_price=candle.low, high_price=candle.high)
        bullish_divergence_id = self.div.bullish_divergence_id()
        bullish_divergence = (
            True if not self.use_divergence else bullish_divergence_id is not None
        )
        if not self.use_divergence:
            bullish_divergence_id = None
        pin_break = (
            len(self.hist) >= 2
            and grid.BullishPinBarDetector.check_vicinity_breakout(
                self.hist, self.lookback
            )
        )
        tf_sl, tf_tp = grid.TF_SPECS[tf][2], grid.TF_SPECS[tf][3]
        return FeatureBar(
            minute=candle.minute,
            close=candle.close,
            atr=atr_value,
            s1=s1,
            s2=s2,
            s3=s3,
            s4=s4,
            bullish_divergence=bullish_divergence,
            pin_break=pin_break,
            embedded=embedded,
            tf=tf,
            tf_sl=tf_sl,
            tf_tp=tf_tp,
            bullish_divergence_id=bullish_divergence_id,
        )


class _BaseMTFTracker:
    def __init__(self, params: dict):
        self.trackers = {
            tf: _BaseTFTracker(spec[1], params)
            for tf, spec in grid.TF_SPECS.items()
        }
        self.bufs = {tf: [] for tf in grid.TF_SPECS}

    def push_1m(self, candle: grid.Candle):
        out = []
        for tf, spec in grid.TF_SPECS.items():
            self.bufs[tf].append(candle)
            if len(self.bufs[tf]) != spec[0]:
                continue
            buf = self.bufs[tf]
            self.bufs[tf] = []
            aggregate = grid.Candle(
                open=buf[0].open,
                high=max(item.high for item in buf),
                low=min(item.low for item in buf),
                close=buf[-1].close,
                minute=buf[-1].minute,
            )
            out.append((tf, self.trackers[tf].push(aggregate, tf)))
        return out


def base_key_for(day: str, previous_day: str, params: dict) -> BaseKey:
    return BaseKey(
        day=day,
        previous_day=previous_day,
        s1_k=params["s1_k"],
        s1_d=params["s1_d"],
        s4_k=params["s4_k"],
        atr_period=params["atr_period"],
        use_divergence=params.get("use_divergence", True),
    )


def _filtered(data, target_strikes):
    return {
        symbol: values
        for symbol, values in data.items()
        if (match := grid.SYM_RE.match(symbol))
        and int(match.group(2)) in target_strikes
    }


def build_day_base_state(
    day: str,
    fpath: str,
    fprev: str,
    params: dict,
    spot: object,
) -> BaseDayState | None:
    if spot is None or not fpath:
        return None
    current = grid.cached_day(fpath)
    if not current:
        return None
    first_match = grid.SYM_RE.match(next(iter(current)))
    if not first_match:
        return None
    prefix = first_match.group(1)
    spot_at_open = grid.latest_spot(spot, 555) or grid.latest_spot(spot, 560)
    if spot_at_open is None:
        return None
    atm = int(round(spot_at_open / 50) * 50)
    target_strikes = set(range(atm - 250, atm + 300, 50))
    current = _filtered(current, target_strikes)
    previous = {}
    if fprev:
        previous_data = grid.cached_day(fprev)
        if previous_data:
            previous = _filtered(previous_data, target_strikes)

    trackers = {}
    warmup_features = {}
    for symbol, values in previous.items():
        tracker = _BaseMTFTracker(params)
        trackers[symbol] = tracker
        per_tf = {tf: [] for tf in grid.TF_SPECS}
        for i in range(len(values["min"])):
            candle = grid.Candle(
                open=values["open"][i],
                high=values["high"][i],
                low=values["low"][i],
                close=values["close"][i],
                minute=values["min"][i],
            )
            for tf, feature in tracker.push_1m(candle):
                per_tf[tf].append(feature)
        warmup_features[symbol] = {
            tf: tuple(bars) for tf, bars in per_tf.items()
        }

    features = {}
    slices = {}
    for symbol, values in current.items():
        tracker = trackers.setdefault(symbol, _BaseMTFTracker(params))
        slices[symbol] = values
        per_tf = {tf: [] for tf in grid.TF_SPECS}
        for i in range(len(values["min"])):
            candle = grid.Candle(
                open=values["open"][i],
                high=values["high"][i],
                low=values["low"][i],
                close=values["close"][i],
                minute=values["min"][i],
            )
            for tf, feature in tracker.push_1m(candle):
                per_tf[tf].append(feature)
        features[symbol] = {
            tf: tuple(bars) for tf, bars in per_tf.items()
        }

    # The reference exit path only reads the final 1m divergence state.
    final_trackers = {
        symbol: SimpleNamespace(trackers={"1m": tracker.trackers["1m"]})
        for symbol, tracker in trackers.items()
    }
    return BaseDayState(
        day=day,
        spot=spot,
        prefix=prefix,
        trackers=final_trackers,
        slices=slices,
        features=features,
        warmup_features=warmup_features,
    )


def materialize_signal_state(
    base: BaseDayState,
    f6_s4_thresh: float,
    f6_s1_thresh: float,
) -> DaySignalState:
    """Apply threshold state machines to already-computed feature bars."""
    pmtrig = {}
    for symbol, per_tf in base.features.items():
        match = grid.SYM_RE.match(symbol)
        if not match:
            continue
        strike, side = int(match.group(2)), match.group(3)
        for tf in grid.TF_SPECS:
            setup = False
            setup_type = ""
            fired = False
            armed_divergence = None
            warmup = base.warmup_features.get(symbol, {}).get(tf, ())
            current = per_tf.get(tf, ())
            for is_warmup, bar in (
                [(True, item) for item in warmup]
                + [(False, item) for item in current]
            ):
                is_flag = (
                    bar.s4 is not None
                    and bar.s1 is not None
                    and bar.s4 >= f6_s4_thresh
                    and bar.s1 <= f6_s1_thresh
                )
                is_super = all(
                    value is not None and value <= 20.5
                    for value in (bar.s1, bar.s2, bar.s3, bar.s4)
                )
                if (
                    (is_flag or is_super)
                    and bar.bullish_divergence
                    and (
                        bar.bullish_divergence_id is None
                        or bar.bullish_divergence_id != armed_divergence
                    )
                ):
                    setup = True
                    setup_type = "super" if is_super else "flag"
                    armed_divergence = bar.bullish_divergence_id
                is_reverse = bar.embedded and setup_type == "super"
                if setup and bar.pin_break:
                    if not is_warmup:
                        pmtrig.setdefault(bar.minute, []).append(
                            (
                                side,
                                strike,
                                symbol,
                                bar.close,
                                is_reverse,
                                tf,
                                bar.tf_sl,
                                bar.tf_tp,
                                bar.atr,
                            )
                        )
                    setup = False
                if is_flag and not fired:
                    if not is_warmup:
                        pmtrig.setdefault(bar.minute, []).append(
                            (
                                side,
                                strike,
                                symbol,
                                bar.close,
                                False,
                                tf,
                                bar.tf_sl,
                                bar.tf_tp,
                                bar.atr,
                            )
                        )
                    fired = True
                elif not is_flag:
                    fired = False
    return DaySignalState(
        day=base.day,
        spot=base.spot,
        prefix=base.prefix,
        trackers=base.trackers,
        slices=base.slices,
        pmtrig=pmtrig,
    )


class BaseStateCache:
    def __init__(self):
        self.states = {}
        self.build_count = 0

    def get_or_build(self, key: BaseKey, builder):
        if key not in self.states:
            self.states[key] = builder()
            self.build_count += 1
        return self.states[key]


_FACTOR_CANDIDATES = []


def _init_factor_worker(spot_all: dict, candidates: list[dict]):
    global _FACTOR_CANDIDATES
    grid.init_worker_local(spot_all)
    _FACTOR_CANDIDATES = candidates


def _set_factor_candidates(candidates: list[dict]) -> None:
    global _FACTOR_CANDIDATES
    _FACTOR_CANDIDATES = candidates


def _process_factor_day(args):
    day, fpath, fprev = args
    spot = grid.GLOBAL_SPOT.get(day)
    base_cache = BaseStateCache()
    signal_states = {}
    results = []
    for params in _FACTOR_CANDIDATES:
        key = base_key_for(day, fprev, params)
        base = base_cache.get_or_build(
            key,
            lambda params=params: build_day_base_state(
                day, fpath, fprev, params, spot
            ),
        )
        threshold_key = (key, params["f6_s4_thresh"], params["f6_s1_thresh"])
        if threshold_key not in signal_states:
            signal_states[threshold_key] = (
                None
                if base is None
                else materialize_signal_state(
                    base, params["f6_s4_thresh"], params["f6_s1_thresh"]
                )
            )
        state = signal_states[threshold_key]
        results.append([] if state is None else simulate_day_signal_state(state, params))
    return day, results, base_cache.build_count, len(signal_states)


class FactorizedCandidatePool:
    """Keep one Windows worker pool alive across candidate batches."""

    def __init__(self, spot_all: dict, workers: int = 8):
        self.workers = max(1, min(8, int(workers)))
        self.pool = grid.Pool(
            processes=self.workers,
            initializer=_init_factor_worker,
            initargs=(spot_all, []),
        )
        self._closed = False

    def run(
        self,
        candidates: list[dict],
        days: Sequence[str],
        files: dict[str, str],
    ):
        if not candidates or not days:
            return [[] for _ in candidates], 0, 0
        candidates = list(candidates)
        days = list(days)
        self.pool.map(_set_factor_candidates, [candidates] * self.workers)
        tasks = [
            (
                day,
                str(files[day]),
                str(files[days[index - 1]]) if index else "",
            )
            for index, day in enumerate(days)
        ]
        results = [[] for _ in candidates]
        base_builds = 0
        signal_builds = 0
        for _, day_results, day_base_builds, day_signal_builds in self.pool.imap(
            _process_factor_day, tasks
        ):
            base_builds += day_base_builds
            signal_builds += day_signal_builds
            for index, trades in enumerate(day_results):
                results[index].extend(trades)
        return results, base_builds, signal_builds

    def close(self) -> None:
        if not self._closed:
            self.pool.close()
            self.pool.join()
            self._closed = True

    def terminate(self) -> None:
        if not self._closed:
            self.pool.terminate()
            self.pool.join()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.terminate()


def run_factorized_candidates(
    candidates: list[dict],
    days: Sequence[str],
    files: dict[str, str],
    spot_all: dict,
    workers: int = 8,
):
    """Evaluate candidates with one raw feature build per base key/day."""
    workers = max(1, min(8, int(workers)))
    if not candidates or not days:
        return [[] for _ in candidates], 0, 0
    with FactorizedCandidatePool(spot_all, workers=workers) as pool:
        return pool.run(candidates, days, files)
