"""Bounded Smart Fib signal-variant grid evaluated by the CUDA exit engine.

The CPU side is deliberately an oracle, not an optimizer: each selected signal
variant is extracted once per day through the Polars historical adapter and is
then frozen into fixed ``(N, T, C)`` option tensors. The GPU side evaluates the
threshold/stop grid from those tensors with the matrix-first engine in
``smart_fib_optimus_gpu``. No trial calls ``process_day`` or rebuilds a signal
stream.

The default is a five-variant staged probe. The full 540-variant zone-aware
signal catalog is available only through explicit ``--variants`` selection or
a bounded ``--max-variants`` request, and non-smoke runs require
``--allow-expensive``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

# Must be set before importing torch, including when the module is imported by
# a spawned Windows preparation worker.
CUDA_ALLOCATOR = "backend:cudaMallocAsync"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", CUDA_ALLOCATOR)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.smart_fib_cuda_bootstrap import configure_cuda_toolkit

configure_cuda_toolkit()

import numpy as np
import torch

from artifacts.f6_hybrid import marni_fib_core_combo_cache as smart_core
from artifacts.f6_hybrid import smart_fib_optimizer as historical
from artifacts.f6_hybrid import smart_fib_optimus_gpu as optimus


T_BARS = optimus.T_BARS
LOT_SIZE = optimus.LOT_SIZE
PRIMARY_TARGET_LEVEL = optimus.PRIMARY_TARGET_LEVEL
FALLBACK_TARGET_LEVEL = optimus.FALLBACK_TARGET_LEVEL
CONTRACT_SLOT_LIMIT = optimus.CONTRACT_SLOT_LIMIT
NET_POINTS_TOLERANCE = optimus.NET_POINTS_TOLERANCE
TRADES_TOLERANCE = optimus.TRADES_TOLERANCE
NET_RS_TOLERANCE = 0.05
DRAWDOWN_PENALTY = 0.20
GRID_CACHE_VERSION = 3

ZONE_PAIRS = tuple(smart_core.ZONE_PAIRS)
TARGET_LEVELS = (0.0, 0.236, 0.29, 0.382, 0.5, 0.618, 0.786, 1.0)
FALLBACK_TARGET_LEVELS = (0.0, 0.236)
DEFAULT_TARGET_LEVELS = (PRIMARY_TARGET_LEVEL,)
DEFAULT_FALLBACK_TARGET_LEVELS = (FALLBACK_TARGET_LEVEL,)
S1_VARIANTS = ((9, 3), (12, 3), (14, 3), (12, 4))
MIN_SPANS = (10.0, 15.0, 20.0)
SETUP_MAX_AGES = (30, 45, 60)
TOUCH_BUFFERS = (0.0, 0.5, 1.0)
EXIT_THRESHOLDS = (5.0, 10.0, 15.0)
STOP_LEVELS = (1.05, 1.13, 1.155, 1.25, 1.272, 1.382, 1.618)
DEFAULT_STOP_LEVELS = (1.155, 1.25)


class GridParityError(RuntimeError):
    """Raised when a selected signal variant fails CPU/GPU parity."""


def _number_token(value: float | int) -> str:
    text = format(float(value), "g")
    return text.replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class SignalVariant:
    """One CPU-generated Smart Fib signal stream."""

    s1_k_period: int
    s1_d_period: int
    min_span: float
    setup_max_age: int
    touch_buffer: float
    zone_start: float = smart_core.S1_ZONE_START
    zone_end: float = smart_core.S1_ZONE_END
    bar_minutes: int = 1
    filter_period: int = smart_core.FILTER_PERIOD

    def validate(self) -> "SignalVariant":
        if (self.s1_k_period, self.s1_d_period) not in S1_VARIANTS:
            raise ValueError(
                f"unsupported S1 variant: {(self.s1_k_period, self.s1_d_period)}"
            )
        if not any(math.isclose(self.min_span, value) for value in MIN_SPANS):
            raise ValueError(f"unsupported min_span: {self.min_span}")
        if self.setup_max_age not in SETUP_MAX_AGES:
            raise ValueError(f"unsupported setup_max_age: {self.setup_max_age}")
        if not any(math.isclose(self.touch_buffer, value) for value in TOUCH_BUFFERS):
            raise ValueError(f"unsupported touch_buffer: {self.touch_buffer}")
        if not any(
            math.isclose(self.zone_start, start)
            and math.isclose(self.zone_end, end)
            for start, end in ZONE_PAIRS
        ):
            raise ValueError(
                f"unsupported Fibonacci zone pair: {(self.zone_start, self.zone_end)}"
            )
        if self.bar_minutes not in (1, 2, 3, 5):
            raise ValueError(f"unsupported bar_minutes: {self.bar_minutes}")
        if self.filter_period != 5 * self.bar_minutes:
            raise ValueError(
                f"filter_period must follow the x5 bias rule "
                f"(5*bar_minutes): got {self.filter_period}"
            )
        return self

    @property
    def variant_id(self) -> str:
        tf_token = "" if self.bar_minutes == 1 else f"_tf{self.bar_minutes}"
        return (
            f"s1k{self.s1_k_period}d{self.s1_d_period}"
            f"_span{_number_token(self.min_span)}"
            f"_age{self.setup_max_age}"
            f"_buf{_number_token(self.touch_buffer)}"
            f"_z{_number_token(self.zone_start)}-{_number_token(self.zone_end)}"
            f"{tf_token}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "s1_k_period": self.s1_k_period,
            "s1_d_period": self.s1_d_period,
            "min_span": self.min_span,
            "setup_max_age": self.setup_max_age,
            "touch_buffer": self.touch_buffer,
            "zone_start": self.zone_start,
            "zone_end": self.zone_end,
            "bar_minutes": self.bar_minutes,
            "filter_period": self.filter_period,
            "variant_id": self.variant_id,
        }


BASELINE_VARIANT = SignalVariant(12, 3, 15.0, 45, 0.0).validate()

# This is intentionally an orthogonal probe rather than a guessed optimum:
# it covers every requested zone pair while retaining bounded S1/filter
# variation. The complete catalog remains available explicitly.
STAGED_VARIANTS = (
    BASELINE_VARIANT,
    SignalVariant(9, 3, 10.0, 30, 0.5, 0.618, 0.786).validate(),
    SignalVariant(14, 3, 20.0, 60, 1.0, 0.786, 1.0).validate(),
    SignalVariant(12, 4, 15.0, 45, 0.5, 0.5, 0.786).validate(),
    SignalVariant(12, 3, 15.0, 45, 0.0, 0.705, 0.886).validate(),
)


def _all_variants() -> tuple[SignalVariant, ...]:
    values = [
        SignalVariant(
            s1_k,
            s1_d,
            min_span,
            setup_age,
            touch_buffer,
            zone_start,
            zone_end,
        ).validate()
        for (zone_start, zone_end), (s1_k, s1_d), min_span, setup_age, touch_buffer in product(
            ZONE_PAIRS, S1_VARIANTS, MIN_SPANS, SETUP_MAX_AGES, TOUCH_BUFFERS
        )
    ]
    values.remove(BASELINE_VARIANT)
    return (BASELINE_VARIANT, *values)


ALL_VARIANTS = _all_variants()


class PolarsHistoricalDataAdapter(historical.CsvHistoricalDataAdapter):
    """Historical adapter that keeps both index and option loading in Polars."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        try:
            import polars  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "smart_fib_optimus_grid_gpu requires Polars for historical loading"
            ) from exc
        super().__init__(*args, **kwargs)

    def _load_spot_csv(self) -> dict[str, dict[str, Any]]:
        import polars as pl

        if not self.index_path.exists():
            raise FileNotFoundError(f"index CSV not found: {self.index_path}")
        frame = pl.read_csv(
            self.index_path,
            columns=["date", "open", "high", "low", "close"],
            try_parse_dates=True,
        )
        date_dtype = str(frame.schema["date"])
        if date_dtype in {"String", "Utf8"}:
            frame = frame.with_columns(
                pl.col("date").str.to_datetime(strict=False).alias("date")
            )
        else:
            frame = frame.with_columns(
                pl.col("date").cast(pl.Datetime, strict=False).alias("date")
            )
        frame = frame.with_columns(
            [pl.col(name).cast(pl.Float64, strict=False).alias(name)
             for name in ("open", "high", "low", "close")]
        ).drop_nulls(["date", "open", "high", "low", "close"])
        frame = frame.with_columns(
            pl.col("date").dt.strftime("%Y-%m-%d").alias("day")
        )
        lower = min(self._option_files, default=self.start)
        frame = frame.filter(
            (pl.col("day") >= lower) & (pl.col("day") <= self.end)
        )
        output: dict[str, dict[str, Any]] = {}
        for group in frame.partition_by("day", maintain_order=True):
            day = str(group["day"][0])
            output[day] = {
                "time": group["date"].dt.strftime("%Y-%m-%d %H:%M:%S").to_list(),
                "open": group["open"].to_numpy(),
                "high": group["high"].to_numpy(),
                "low": group["low"].to_numpy(),
                "close": group["close"].to_numpy(),
            }
        return output


@dataclass
class VariantDayRaw:
    variant: SignalVariant
    day: str
    prepared: dict[str, Any]
    selected_events: list[dict[str, Any]]
    raw_event_count: int
    contract_keys: list[tuple[str, int]]


@dataclass
class VariantCpuDataset:
    """Frozen logical ``(N,T,C)`` tensors plus sparse event metadata."""

    days: list[str]
    index_open: np.ndarray
    index_high: np.ndarray
    index_low: np.ndarray
    index_close: np.ndarray
    index_valid: np.ndarray
    contract_open_ntc: np.ndarray
    contract_high_ntc: np.ndarray
    contract_low_ntc: np.ndarray
    contract_close_ntc: np.ndarray
    contract_valid_ntc: np.ndarray
    event_mask: np.ndarray
    event_contract: np.ndarray
    event_entry: np.ndarray
    event_fib_high: np.ndarray
    event_fib_low: np.ndarray
    event_index_source: np.ndarray
    event_price_rise: np.ndarray
    event_premium_rise: np.ndarray
    event_high_to_low: np.ndarray
    event_side_ce: np.ndarray
    event_symbols: list[list[str | None]]
    contract_symbols: list[list[str | None]]
    raw_event_counts: list[int]
    selected_event_counts: list[int]

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def contract_slots(self) -> int:
        return int(self.contract_close_ntc.shape[2])


@dataclass
class VariantCpuBundle:
    variant: SignalVariant
    dataset: VariantCpuDataset
    parity_payloads: dict[str, dict[str, Any]]
    cache_source: str
    cache_path: str | None
    prep_seconds: float


@dataclass
class VariantGpuDataset:
    """Resident NTC tensors and the NCT indexing view required by the engine."""

    engine: optimus.GpuDataset
    contract_open_ntc: torch.Tensor
    contract_high_ntc: torch.Tensor
    contract_low_ntc: torch.Tensor
    contract_close_ntc: torch.Tensor
    contract_valid_ntc: torch.Tensor


_GRID_WORKER_ADAPTER: PolarsHistoricalDataAdapter | None = None


def _init_grid_worker(data_root: str, start: str, end: str) -> None:
    global _GRID_WORKER_ADAPTER
    _GRID_WORKER_ADAPTER = PolarsHistoricalDataAdapter(
        data_root,
        start=start,
        end=end,
        cache_days=8,
    )


def _extract_variant_day(
    task: tuple[
        str, str, str, str,
        tuple[int, int, float, int, float, float, float, int, int],
    ]
) -> VariantDayRaw:
    """Worker entry point; this is the only CPU signal extraction call."""
    data_root, start, end, day, values = task
    variant = SignalVariant(*values).validate()
    adapter = _GRID_WORKER_ADAPTER
    if adapter is None:
        adapter = PolarsHistoricalDataAdapter(
            data_root,
            start=start,
            end=end,
            cache_days=8,
        )
    prepared = smart_core.extract_day_events(
        day,
        cache_loader=adapter.load_day_cache,
        min_span=variant.min_span,
        touch_buffer=variant.touch_buffer,
        setup_max_age=variant.setup_max_age,
        zone_start=variant.zone_start,
        zone_end=variant.zone_end,
        s1_k_period=variant.s1_k_period,
        s1_d_period=variant.s1_d_period,
        bar_minutes=variant.bar_minutes,
        filter_period=variant.filter_period,
        debug=False,
    )
    if not prepared:
        raise RuntimeError(f"no Smart Fib cache payload for {day}")
    selected, raw_count = optimus._select_day_events(prepared)
    return VariantDayRaw(
        variant=variant,
        day=day,
        prepared=dict(prepared),
        selected_events=selected,
        raw_event_count=raw_count,
        contract_keys=list(prepared["records"]),
    )


def _build_cpu_dataset(days: Sequence[str], raw_days: Sequence[VariantDayRaw]) -> VariantCpuDataset:
    raw_by_day = {raw.day: raw for raw in raw_days}
    missing = [day for day in days if day not in raw_by_day]
    if missing:
        raise RuntimeError(f"CPU preparation missed days: {missing}")
    max_contracts = max((len(raw_by_day[day].contract_keys) for day in days), default=0)
    if max_contracts > CONTRACT_SLOT_LIMIT:
        raise RuntimeError(
            f"{max_contracts} contracts exceed fixed contract-slot limit "
            f"C={CONTRACT_SLOT_LIMIT}; refusing to truncate candidates"
        )

    n_days = len(days)
    contract_slots = max(1, max_contracts)
    shape = (n_days, T_BARS)
    contract_shape = (n_days, T_BARS, contract_slots)
    index_open = np.zeros(shape, dtype=np.float64)
    index_high = np.zeros(shape, dtype=np.float64)
    index_low = np.zeros(shape, dtype=np.float64)
    index_close = np.zeros(shape, dtype=np.float64)
    index_valid = np.zeros(shape, dtype=np.bool_)
    contract_open = np.zeros(contract_shape, dtype=np.float64)
    contract_high = np.zeros(contract_shape, dtype=np.float64)
    contract_low = np.zeros(contract_shape, dtype=np.float64)
    contract_close = np.zeros(contract_shape, dtype=np.float64)
    contract_valid = np.zeros(contract_shape, dtype=np.bool_)
    event_mask = np.zeros(shape, dtype=np.bool_)
    event_contract = np.zeros(shape, dtype=np.int64)
    event_entry = np.zeros(shape, dtype=np.float64)
    event_fib_high = np.zeros(shape, dtype=np.float64)
    event_fib_low = np.zeros(shape, dtype=np.float64)
    event_index_source = np.zeros(shape, dtype=np.bool_)
    event_price_rise = np.zeros(shape, dtype=np.bool_)
    event_premium_rise = np.zeros(shape, dtype=np.bool_)
    event_high_to_low = np.zeros(shape, dtype=np.bool_)
    event_side_ce = np.zeros(shape, dtype=np.bool_)
    event_symbols: list[list[str | None]] = [[None] * T_BARS for _ in range(n_days)]
    contract_symbols: list[list[str | None]] = [
        [None] * contract_slots for _ in range(n_days)
    ]
    raw_event_counts: list[int] = []
    selected_event_counts: list[int] = []

    for day_index, day in enumerate(days):
        raw = raw_by_day[day]
        prepared = raw.prepared
        spot = prepared["spot"]
        for minute, open_, high, low, close in zip(
            spot["min"], spot["open"], spot["high"], spot["low"], spot["close"]
        ):
            position = optimus._slot(int(minute))
            if position is None:
                continue
            index_open[day_index, position] = float(open_)
            index_high[day_index, position] = float(high)
            index_low[day_index, position] = float(low)
            index_close[day_index, position] = float(close)
            index_valid[day_index, position] = True

        contract_index = {
            key: index for index, key in enumerate(raw.contract_keys)
        }
        for key, slot_index in contract_index.items():
            record = prepared["records"][key]
            contract_symbols[day_index][slot_index] = str(record["symbol"])
            arrays = (
                contract_open[day_index, :, slot_index],
                contract_high[day_index, :, slot_index],
                contract_low[day_index, :, slot_index],
                contract_close[day_index, :, slot_index],
            )
            for row in record["current"]:
                minute = int(row["minute"])
                position = optimus._slot(minute)
                if position is None:
                    continue
                optimus._put_ohlc(arrays, minute, row)
                contract_valid[day_index, position, slot_index] = True

        for event in raw.selected_events:
            minute = int(event["minute"])
            position = optimus._slot(minute)
            if position is None or event_mask[day_index, position]:
                continue
            key = (str(event["side"]), int(event["strike"]))
            slot_index = contract_index[key]
            event_mask[day_index, position] = True
            event_contract[day_index, position] = slot_index
            event_entry[day_index, position] = float(event["option_entry"])
            event_fib_high[day_index, position] = float(event["fib_high"])
            event_fib_low[day_index, position] = float(event["fib_low"])
            event_index_source[day_index, position] = event["fib_source"] == "index"
            event_price_rise[day_index, position] = bool(
                event.get("price_profit_on_rise", True)
            )
            event_premium_rise[day_index, position] = bool(
                event.get("profit_on_rise", True)
            )
            event_high_to_low[day_index, position] = optimus._event_orientation(event)
            event_side_ce[day_index, position] = str(event["side"]) == "CE"
            event_symbols[day_index][position] = contract_symbols[day_index][slot_index]

        raw_event_counts.append(raw.raw_event_count)
        selected_event_counts.append(len(raw.selected_events))

    return VariantCpuDataset(
        days=list(days),
        index_open=index_open,
        index_high=index_high,
        index_low=index_low,
        index_close=index_close,
        index_valid=index_valid,
        contract_open_ntc=contract_open,
        contract_high_ntc=contract_high,
        contract_low_ntc=contract_low,
        contract_close_ntc=contract_close,
        contract_valid_ntc=contract_valid,
        event_mask=event_mask,
        event_contract=event_contract,
        event_entry=event_entry,
        event_fib_high=event_fib_high,
        event_fib_low=event_fib_low,
        event_index_source=event_index_source,
        event_price_rise=event_price_rise,
        event_premium_rise=event_premium_rise,
        event_high_to_low=event_high_to_low,
        event_side_ce=event_side_ce,
        event_symbols=event_symbols,
        contract_symbols=contract_symbols,
        raw_event_counts=raw_event_counts,
        selected_event_counts=selected_event_counts,
    )


_CACHE_ARRAY_NAMES = (
    "index_open", "index_high", "index_low", "index_close", "index_valid",
    "contract_open_ntc", "contract_high_ntc", "contract_low_ntc",
    "contract_close_ntc", "contract_valid_ntc", "event_mask", "event_contract",
    "event_entry", "event_fib_high", "event_fib_low", "event_index_source",
    "event_price_rise", "event_premium_rise", "event_high_to_low", "event_side_ce",
)


def _variant_cache_path(cache_dir: Path, days: Sequence[str], variant: SignalVariant) -> Path:
    return cache_dir / (
        f"smart_fib_grid_tensor_cache_{days[0]}_{days[-1]}_{variant.variant_id}.npz"
    )


def _save_variant_cache(
    dataset: VariantCpuDataset,
    path: Path,
    data_root: str | Path,
    variant: SignalVariant,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite variant tensor cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache_version": GRID_CACHE_VERSION,
        "cache_key": variant.variant_id,
        "variant": variant.as_dict(),
        "data_root": str(Path(data_root).resolve()),
        "days": dataset.days,
        "contract_slots": dataset.contract_slots,
        "raw_event_counts": dataset.raw_event_counts,
        "selected_event_counts": dataset.selected_event_counts,
        "event_symbols": dataset.event_symbols,
        "contract_symbols": dataset.contract_symbols,
    }
    arrays = {name: getattr(dataset, name) for name in _CACHE_ARRAY_NAMES}
    np.savez_compressed(path, **arrays, metadata=np.array(json.dumps(metadata)))


def _load_variant_cache(
    path: Path,
    data_root: str | Path,
    days: Sequence[str],
    variant: SignalVariant,
) -> VariantCpuDataset | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("cache_version") != GRID_CACHE_VERSION:
                return None
            if metadata.get("cache_key") != variant.variant_id:
                return None
            if metadata.get("variant", {}).get("variant_id") != variant.variant_id:
                return None
            if metadata.get("data_root") != str(Path(data_root).resolve()):
                return None
            if metadata.get("days") != list(days):
                return None
            arrays = {name: archive[name].copy() for name in _CACHE_ARRAY_NAMES}
        return VariantCpuDataset(
            days=list(days),
            **arrays,
            event_symbols=metadata["event_symbols"],
            contract_symbols=metadata["contract_symbols"],
            raw_event_counts=[int(value) for value in metadata["raw_event_counts"]],
            selected_event_counts=[
                int(value) for value in metadata["selected_event_counts"]
            ],
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _from_legacy_cache(dataset: optimus.CpuDataset) -> VariantCpuDataset:
    """Adapt the existing baseline NCT cache without redoing CPU extraction."""
    return VariantCpuDataset(
        days=list(dataset.days),
        index_open=dataset.index_open,
        index_high=dataset.index_high,
        index_low=dataset.index_low,
        index_close=dataset.index_close,
        index_valid=dataset.index_valid,
        contract_open_ntc=np.ascontiguousarray(
            np.transpose(dataset.contract_open, (0, 2, 1))
        ),
        contract_high_ntc=np.ascontiguousarray(
            np.transpose(dataset.contract_high, (0, 2, 1))
        ),
        contract_low_ntc=np.ascontiguousarray(
            np.transpose(dataset.contract_low, (0, 2, 1))
        ),
        contract_close_ntc=np.ascontiguousarray(
            np.transpose(dataset.contract_close, (0, 2, 1))
        ),
        contract_valid_ntc=np.ascontiguousarray(
            np.transpose(dataset.contract_valid, (0, 2, 1))
        ),
        event_mask=dataset.event_mask,
        event_contract=dataset.event_contract,
        event_entry=dataset.event_entry,
        event_fib_high=dataset.event_fib_high,
        event_fib_low=dataset.event_fib_low,
        event_index_source=dataset.event_index_source,
        event_price_rise=dataset.event_price_rise,
        event_premium_rise=dataset.event_premium_rise,
        event_high_to_low=dataset.event_high_to_low,
        event_side_ce=dataset.event_side_ce,
        event_symbols=dataset.event_symbols,
        contract_symbols=dataset.contract_symbols,
        raw_event_counts=dataset.raw_event_counts,
        selected_event_counts=dataset.selected_event_counts,
    )


def _legacy_cache_path(days: Sequence[str]) -> Path:
    return ROOT / "artifacts" / "f6_hybrid" / (
        f"smart_fib_gpu_tensor_cache_{days[0]}_{days[-1]}.npz"
    )


def _cached_variant_dataset(
    data_root: str | Path,
    days: Sequence[str],
    variant: SignalVariant,
    tensor_cache_dir: Path | None,
) -> tuple[VariantCpuDataset | None, str, str | None]:
    if tensor_cache_dir is not None:
        path = _variant_cache_path(tensor_cache_dir, days, variant)
        cached = _load_variant_cache(path, data_root, days, variant)
        if cached is not None:
            return cached, "variant_cache", str(path)

    # The old full cache is a safe compatibility source only for the unchanged
    # baseline. Any newly written cache carries the variant identity in its key.
    if variant == BASELINE_VARIANT:
        legacy_path = _legacy_cache_path(days)
        cached = optimus.load_cpu_dataset(legacy_path, str(data_root), days)
        if cached is not None:
            return _from_legacy_cache(cached), "legacy_full_cache_baseline", str(legacy_path)
    return None, "cpu_extraction", None


def collect_variant_cpu_dataset(
    adapter: PolarsHistoricalDataAdapter,
    days: Sequence[str],
    variant: SignalVariant,
    *,
    workers: int,
    parity_days: Sequence[str],
    tensor_cache_dir: Path | None,
) -> VariantCpuBundle:
    """Extract one variant once per day, then freeze its CPU oracle tensors."""
    variant.validate()
    cached, source, cache_path = _cached_variant_dataset(
        adapter.data_root, days, variant, tensor_cache_dir
    )
    if cached is not None:
        print(
            f"[CPU PREP] {variant.variant_id} cache={source} "
            f"N={cached.n_days} T={T_BARS} C={cached.contract_slots}",
            flush=True,
        )
        return VariantCpuBundle(
            variant=variant,
            dataset=cached,
            parity_payloads={},
            cache_source=source,
            cache_path=cache_path,
            prep_seconds=0.0,
        )

    started = time.perf_counter()
    task_values = (
        variant.s1_k_period,
        variant.s1_d_period,
        variant.min_span,
        variant.setup_max_age,
        variant.touch_buffer,
        variant.zone_start,
        variant.zone_end,
        variant.bar_minutes,
        variant.filter_period,
    )
    tasks = [
        (str(adapter.data_root), adapter.start, adapter.end, day, task_values)
        for day in days
    ]
    raw_days: list[VariantDayRaw] = []
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_grid_worker,
            initargs=(str(adapter.data_root), adapter.start, adapter.end),
        ) as pool:
            iterator = pool.map(_extract_variant_day, tasks, chunksize=1)
            for count, raw in enumerate(iterator, start=1):
                raw_days.append(raw)
                if count == 1 or count % 25 == 0 or count == len(tasks):
                    print(
                        f"[CPU PREP] {variant.variant_id} {count}/{len(tasks)} "
                        f"last={raw.day}",
                        flush=True,
                    )
    else:
        for count, task in enumerate(tasks, start=1):
            raw = _extract_variant_day(task)
            raw_days.append(raw)
            if count == 1 or count % 25 == 0 or count == len(tasks):
                print(
                    f"[CPU PREP] {variant.variant_id} {count}/{len(tasks)} "
                    f"last={raw.day}",
                    flush=True,
                )

    dataset = _build_cpu_dataset(days, raw_days)
    parity_payloads = {
        raw.day: raw.prepared for raw in raw_days if raw.day in set(parity_days)
    }
    if tensor_cache_dir is not None:
        path = _variant_cache_path(tensor_cache_dir, days, variant)
        _save_variant_cache(dataset, path, adapter.data_root, variant)
        cache_path = str(path)
        source = "cpu_extraction_saved_variant_cache"
    elapsed = time.perf_counter() - started
    print(
        f"[CPU PREP] {variant.variant_id} complete {elapsed:.3f}s "
        f"N={dataset.n_days} T={T_BARS} C={dataset.contract_slots} "
        f"events={sum(dataset.selected_event_counts)}",
        flush=True,
    )
    return VariantCpuBundle(
        variant=variant,
        dataset=dataset,
        parity_payloads=parity_payloads,
        cache_source=source,
        cache_path=cache_path,
        prep_seconds=elapsed,
    )


def _copy_to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    host = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        host = host.pin_memory()
        return host.to(device, non_blocking=True)
    return host.to(device)


def to_gpu_dataset(
    cpu: VariantCpuDataset,
    device: torch.device,
) -> VariantGpuDataset:
    """Transfer NTC tensors once and expose NCT views to the existing engine."""
    contract_open_ntc = _copy_to_device(cpu.contract_open_ntc, device)
    contract_high_ntc = _copy_to_device(cpu.contract_high_ntc, device)
    contract_low_ntc = _copy_to_device(cpu.contract_low_ntc, device)
    contract_close_ntc = _copy_to_device(cpu.contract_close_ntc, device)
    contract_valid_ntc = _copy_to_device(cpu.contract_valid_ntc, device)
    engine = optimus.GpuDataset(
        days=cpu.days,
        index_open=_copy_to_device(cpu.index_open, device),
        index_high=_copy_to_device(cpu.index_high, device),
        index_low=_copy_to_device(cpu.index_low, device),
        index_close=_copy_to_device(cpu.index_close, device),
        index_valid=_copy_to_device(cpu.index_valid, device),
        contract_open=contract_open_ntc.permute(0, 2, 1),
        contract_high=contract_high_ntc.permute(0, 2, 1),
        contract_low=contract_low_ntc.permute(0, 2, 1),
        contract_close=contract_close_ntc.permute(0, 2, 1),
        contract_valid=contract_valid_ntc.permute(0, 2, 1),
        event_mask=_copy_to_device(cpu.event_mask, device),
        event_contract=_copy_to_device(cpu.event_contract, device),
        event_entry=_copy_to_device(cpu.event_entry, device),
        event_fib_high=_copy_to_device(cpu.event_fib_high, device),
        event_fib_low=_copy_to_device(cpu.event_fib_low, device),
        event_index_source=_copy_to_device(cpu.event_index_source, device),
        event_price_rise=_copy_to_device(cpu.event_price_rise, device),
        event_premium_rise=_copy_to_device(cpu.event_premium_rise, device),
        event_high_to_low=_copy_to_device(cpu.event_high_to_low, device),
        event_side_ce=_copy_to_device(cpu.event_side_ce, device),
        event_symbols=cpu.event_symbols,
        contract_symbols=cpu.contract_symbols,
        raw_event_counts=cpu.raw_event_counts,
        selected_event_counts=cpu.selected_event_counts,
        device=device,
    )
    return VariantGpuDataset(
        engine=engine,
        contract_open_ntc=contract_open_ntc,
        contract_high_ntc=contract_high_ntc,
        contract_low_ntc=contract_low_ntc,
        contract_close_ntc=contract_close_ntc,
        contract_valid_ntc=contract_valid_ntc,
    )


def _mask_for_days(data: optimus.GpuDataset, days: Sequence[str]) -> torch.Tensor:
    return optimus._mask_for_days(data, days)


def _grid_params(
    targets: Sequence[float],
    fallback_targets: Sequence[float],
    thresholds: Sequence[float],
    stops: Sequence[float],
) -> list[dict[str, Any]]:
    targets = _validate_axis(targets, TARGET_LEVELS, "target")
    fallback_targets = _validate_axis(
        fallback_targets, FALLBACK_TARGET_LEVELS, "fallback-target"
    )
    thresholds = _validate_axis(thresholds, EXIT_THRESHOLDS, "threshold")
    stops = _validate_axis(stops, STOP_LEVELS, "stop")
    params = []
    for target in targets:
        for fallback in fallback_targets:
            if not 0.0 <= float(fallback) <= float(target) <= 1.0:
                continue
            for threshold in thresholds:
                for stop in stops:
                    params.append({
                        "stop_level": float(stop),
                        "target_level": float(target),
                        "fallback_target_level": float(fallback),
                        "option_point_threshold": float(threshold),
                    })
    if not params:
        raise ValueError("no valid target/fallback combinations remain")
    return params


def _zero_result(data: optimus.GpuDataset) -> dict[str, Any]:
    zeros = [0] * data.n_days
    zero_floats = [0.0] * data.n_days
    return {
        "trades": 0,
        "wins": 0,
        "win_rate": 0.0,
        "fees_rs": 0.0,
        "net_points": 0.0,
        "net_rs": 0.0,
        "max_drawdown_rs": 0.0,
        "max_drawdown_points": 0.0,
        "profit_factor": 0.0,
        "daily_trades": zeros,
        "daily_net_points": zero_floats,
        "daily_net_rs": zero_floats,
        "daily_drawdown_rs": zero_floats,
    }


def _evaluate_gpu(
    evaluator: optimus.GpuEvaluator,
    params: Sequence[Mapping[str, Any]],
    day_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    """Run a real matrix-first GPU batch, including the empty-event guard."""
    if evaluator.events.time_index.numel() == 0:
        return [_zero_result(evaluator.data) for _ in params]
    with torch.inference_mode():
        return evaluator.evaluate(params, day_mask)


def _cpu_stats(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        trades,
        key=lambda trade: (
            int(trade.get("entry_min", 0)),
            int(trade.get("exit_min", 0)),
        ),
    )
    wins = [trade for trade in ordered if float(trade.get("rs_net", 0.0)) > 0.0]
    losses = [trade for trade in ordered if float(trade.get("rs_net", 0.0)) <= 0.0]
    net_points = sum(float(trade.get("points", 0.0)) for trade in ordered)
    net_rs = sum(float(trade.get("rs_net", 0.0)) for trade in ordered)
    fees_rs = sum(float(trade.get("fee", 0.0)) for trade in ordered)
    equity = 0.0
    peak = 0.0
    max_dd_rs = 0.0
    for trade in ordered:
        equity += float(trade.get("rs_net", 0.0))
        peak = max(peak, equity)
        max_dd_rs = max(max_dd_rs, peak - equity)
    gross_wins = sum(float(trade.get("rs_net", 0.0)) for trade in wins)
    gross_losses = abs(sum(float(trade.get("rs_net", 0.0)) for trade in losses))
    return {
        "trades": len(ordered),
        "wins": len(wins),
        "win_rate": round(len(wins) / len(ordered) * 100.0, 2) if ordered else 0.0,
        "fees_rs": round(fees_rs, 2),
        "net_points": round(net_points, 2),
        "net_rs": round(net_rs, 2),
        "max_drawdown_rs": round(max_dd_rs, 2),
        "max_drawdown_points": round(max_dd_rs / LOT_SIZE, 2),
        "profit_factor": (
            round(gross_wins / gross_losses, 4)
            if gross_losses
            else (float("inf") if gross_wins else 0.0)
        ),
    }


def _cpu_variant_trades(
    prepared: Mapping[str, Any],
    target: float,
    fallback_target: float,
    threshold: float,
    stop: float,
    brokerage_per_order: float = optimus.BROKERAGE_PER_ORDER,
    fixed_cost_per_trade: float | None = None,
    max_trades_per_day: int | None = None,
    daily_loss_limit_rs: float | None = None,
) -> list[dict[str, Any]]:
    """Replay frozen CPU events with the canonical CPU exit simulator."""
    selected, _ = optimus._select_day_events(prepared)
    events = [{**signal, "timeframe": "combined"} for signal in selected]
    return smart_core.simulate(
        events,
        prepared["bars"],
        prepared["index_bars"],
        prepared["spot"],
        "combined",
        target,
        stop,
        concurrent=False,
        option_point_threshold=threshold,
        fallback_target_level=fallback_target,
        brokerage_per_order=brokerage_per_order,
        fixed_cost_per_trade=fixed_cost_per_trade,
        max_trades_per_day=max_trades_per_day,
        daily_loss_limit_rs=daily_loss_limit_rs,
    )


def _parity_payloads(
    adapter: PolarsHistoricalDataAdapter,
    bundle: VariantCpuBundle,
    parity_days: Sequence[str],
) -> dict[str, dict[str, Any]]:
    payloads = dict(bundle.parity_payloads)
    for day in parity_days:
        if day in payloads:
            continue
        # Cache files contain tensors, not the full CPU oracle payload. Rehydrate
        # only the three required parity dates, outside any GPU trial batch.
        payload = smart_core.extract_day_events(
            day,
            cache_loader=adapter.load_day_cache,
            min_span=bundle.variant.min_span,
            touch_buffer=bundle.variant.touch_buffer,
            setup_max_age=bundle.variant.setup_max_age,
            zone_start=bundle.variant.zone_start,
            zone_end=bundle.variant.zone_end,
            s1_k_period=bundle.variant.s1_k_period,
            s1_d_period=bundle.variant.s1_d_period,
            bar_minutes=bundle.variant.bar_minutes,
            filter_period=bundle.variant.filter_period,
            debug=False,
        )
        if not payload:
            raise RuntimeError(f"no parity payload for {bundle.variant.variant_id} {day}")
        payloads[day] = payload
    return payloads


def validate_variant_parity(
    adapter: PolarsHistoricalDataAdapter,
    bundle: VariantCpuBundle,
    gpu_data: VariantGpuDataset,
    evaluator: optimus.GpuEvaluator,
    parity_days: Sequence[str],
    targets: Sequence[float],
    fallback_targets: Sequence[float],
    thresholds: Sequence[float],
    stops: Sequence[float],
) -> dict[str, Any]:
    """Require exact three-date CPU/GPU parity for every exit combination."""
    if len(parity_days) != 3:
        raise ValueError("parity validation requires exactly three dates")
    payloads = _parity_payloads(adapter, bundle, parity_days)
    params = _grid_params(targets, fallback_targets, thresholds, stops)
    cpu_by_combo: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    for param in params:
        daily = {}
        for day in parity_days:
            daily[day] = _cpu_stats(
                _cpu_variant_trades(
                    payloads[day],
                    float(param["target_level"]),
                    float(param["fallback_target_level"]),
                    float(param["option_point_threshold"]),
                    float(param["stop_level"]),
                    evaluator.brokerage_per_order,
                    evaluator.fixed_cost_per_trade,
                )
            )
        aggregate_trades = sum(item["trades"] for item in daily.values())
        aggregate_wins = sum(item["wins"] for item in daily.values())
        aggregate_points = sum(item["net_points"] for item in daily.values())
        aggregate_rs = sum(item["net_rs"] for item in daily.values())
        aggregate_fees = sum(item["fees_rs"] for item in daily.values())
        cpu_by_combo[
            (
                float(param["target_level"]),
                float(param["fallback_target_level"]),
                float(param["option_point_threshold"]),
                float(param["stop_level"]),
            )
        ] = {
            "trades": aggregate_trades,
            "wins": aggregate_wins,
            "net_points": round(aggregate_points, 2),
            "net_rs": round(aggregate_rs, 2),
            "fees_rs": round(aggregate_fees, 2),
            "daily": daily,
        }

    mask = _mask_for_days(gpu_data.engine, parity_days)
    started = time.perf_counter()
    gpu_results = _evaluate_gpu(evaluator, params, mask)
    gpu_wall = time.perf_counter() - started
    selected_indices = [gpu_data.engine.days.index(day) for day in parity_days]
    comparisons: list[dict[str, Any]] = []
    passed = True
    for param, gpu_result in zip(params, gpu_results):
        threshold = float(param["option_point_threshold"])
        stop = float(param["stop_level"])
        target = float(param["target_level"])
        fallback_target = float(param["fallback_target_level"])
        cpu = cpu_by_combo[(target, fallback_target, threshold, stop)]
        gpu_daily = {}
        for day, index in zip(parity_days, selected_indices):
            gpu_daily[day] = {
                "trades": int(gpu_result["daily_trades"][index]),
                "net_points": round(float(gpu_result["daily_net_points"][index]), 2),
                "net_rs": round(float(gpu_result["daily_net_rs"][index]), 2),
                "max_drawdown_points": round(
                    float(gpu_result["daily_drawdown_rs"][index]) / LOT_SIZE, 2
                ),
            }
        gpu = {
            "trades": sum(item["trades"] for item in gpu_daily.values()),
            "wins": int(gpu_result["wins"]),
            "net_points": round(
                sum(item["net_points"] for item in gpu_daily.values()), 2
            ),
            "net_rs": round(
                sum(item["net_rs"] for item in gpu_daily.values()), 2
            ),
            "fees_rs": round(float(gpu_result["fees_rs"]), 2),
            "daily": gpu_daily,
        }
        daily_failures = []
        for day in parity_days:
            cpu_day = cpu["daily"][day]
            gpu_day = gpu_daily[day]
            if gpu_day["trades"] != cpu_day["trades"]:
                daily_failures.append(f"{day}: trades {gpu_day['trades']} != {cpu_day['trades']}")
            if abs(gpu_day["net_points"] - cpu_day["net_points"]) > NET_POINTS_TOLERANCE:
                daily_failures.append(
                    f"{day}: points {gpu_day['net_points']} != {cpu_day['net_points']}"
                )
            if abs(
                gpu_day["max_drawdown_points"] - cpu_day["max_drawdown_points"]
            ) > NET_POINTS_TOLERANCE:
                daily_failures.append(
                    f"{day}: dd {gpu_day['max_drawdown_points']} != "
                    f"{cpu_day['max_drawdown_points']}"
                )
        aggregate_failures = []
        if abs(gpu["trades"] - cpu["trades"]) > TRADES_TOLERANCE:
            aggregate_failures.append("aggregate trades")
        if abs(gpu["wins"] - cpu["wins"]) > TRADES_TOLERANCE:
            aggregate_failures.append("aggregate wins")
        if abs(gpu["net_points"] - cpu["net_points"]) > NET_POINTS_TOLERANCE:
            aggregate_failures.append("aggregate net points")
        if abs(gpu["net_rs"] - cpu["net_rs"]) > NET_RS_TOLERANCE:
            aggregate_failures.append("aggregate net Rs")
        if abs(gpu["fees_rs"] - cpu["fees_rs"]) > NET_RS_TOLERANCE:
            aggregate_failures.append("aggregate fees")
        item_passed = not daily_failures and not aggregate_failures
        passed = passed and item_passed
        comparisons.append({
            "target_level": target,
            "fallback_target_level": fallback_target,
            "threshold": threshold,
            "stop_level": stop,
            "cpu": cpu,
            "gpu": gpu,
            "tolerances": {
                "trades": TRADES_TOLERANCE,
                "net_points": NET_POINTS_TOLERANCE,
                "net_rs": NET_RS_TOLERANCE,
            },
            "failures": daily_failures + aggregate_failures,
            "passed": item_passed,
        })
    report = {
        "variant": bundle.variant.as_dict(),
        "dates": list(parity_days),
        "passed": passed,
        "comparisons": comparisons,
        "gpu_wall_seconds": round(gpu_wall, 6),
        "gpu_cuda_ms": round(float(evaluator.last_cuda_ms), 3),
    }
    if not passed:
        raise GridParityError(json.dumps(report, indent=2, default=str))
    return report


def _score(result: Mapping[str, Any]) -> float:
    return float(result["net_points"]) - DRAWDOWN_PENALTY * float(
        result["max_drawdown_points"]
    )


def evaluate_variant_grid(
    variant: SignalVariant,
    gpu_data: VariantGpuDataset,
    evaluator: optimus.GpuEvaluator,
    days: Sequence[str],
    targets: Sequence[float],
    fallback_targets: Sequence[float],
    thresholds: Sequence[float],
    stops: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = _grid_params(targets, fallback_targets, thresholds, stops)
    mask = _mask_for_days(gpu_data.engine, days)
    before_evaluations = evaluator.evaluations
    before_cuda_ms = evaluator.cuda_ms
    before_wall = evaluator.wall_seconds
    started = time.perf_counter()
    results = _evaluate_gpu(evaluator, params, mask)
    wall = time.perf_counter() - started
    configs = []
    for param, stats in zip(params, results):
        config_id = (
            f"{variant.variant_id}|threshold{_number_token(param['option_point_threshold'])}"
            f"|target{_number_token(param['target_level'])}"
            f"|fallback{_number_token(param['fallback_target_level'])}"
            f"|stop{_number_token(param['stop_level'])}"
        )
        record = {
            "config_id": config_id,
            "variant": variant.as_dict(),
            "stop_level": float(param["stop_level"]),
            "target_level": float(param["target_level"]),
            "fallback_target_level": float(param["fallback_target_level"]),
            "option_point_threshold": float(param["option_point_threshold"]),
            "trades": int(stats["trades"]),
            "wins": int(stats["wins"]),
            "win_rate": float(stats["win_rate"]),
            "net_points": float(stats["net_points"]),
            "net_rs": float(stats["net_rs"]),
            "max_drawdown_points": float(stats["max_drawdown_points"]),
            "max_drawdown_rs": float(stats["max_drawdown_rs"]),
            "profit_factor": stats["profit_factor"],
            "fees_rs": float(stats["fees_rs"]),
            "composite_score": 0.0,
            "daily_trades": stats["daily_trades"],
            "daily_net_points": stats["daily_net_points"],
            "daily_net_rs": stats["daily_net_rs"],
            "daily_drawdown_rs": stats["daily_drawdown_rs"],
        }
        record["composite_score"] = _score(record)
        configs.append(record)
    return configs, {
        "variant_id": variant.variant_id,
        "grid_evaluations": evaluator.evaluations - before_evaluations,
        "cuda_ms": round(evaluator.cuda_ms - before_cuda_ms, 3),
        "wall_seconds": round(max(wall, evaluator.wall_seconds - before_wall), 6),
        "batch_size": evaluator.batch_size,
        "event_count": int(evaluator.events.time_index.numel()),
    }


def _parse_variant(token: str) -> SignalVariant:
    if token.lower() == "baseline":
        return BASELINE_VARIANT
    pieces = [piece for piece in re.split(r"[:,/]+", token.strip()) if piece]
    if len(pieces) not in (5, 7):
        raise ValueError(
            "variant must be baseline or "
            "s1_k:s1_d:min_span:setup_max_age:touch_buffer[:zone_start:zone_end]"
        )
    zone_start = smart_core.S1_ZONE_START if len(pieces) == 5 else float(pieces[5])
    zone_end = smart_core.S1_ZONE_END if len(pieces) == 5 else float(pieces[6])
    return SignalVariant(
        int(pieces[0]),
        int(pieces[1]),
        float(pieces[2]),
        int(pieces[3]),
        float(pieces[4]),
        zone_start,
        zone_end,
    ).validate()


def _parse_float_tokens(tokens: Sequence[str] | None, default: Sequence[float]) -> list[float]:
    if not tokens:
        return [float(value) for value in default]
    values: list[float] = []
    for token in tokens:
        values.extend(float(piece) for piece in token.split(",") if piece)
    return values


def _validate_axis(values: Sequence[float], allowed: Sequence[float], name: str) -> list[float]:
    output: list[float] = []
    for value in values:
        if not any(math.isclose(value, candidate) for candidate in allowed):
            raise ValueError(f"{name} value {value} is outside the explicit grid {allowed}")
        canonical = next(candidate for candidate in allowed if math.isclose(value, candidate))
        if canonical not in output:
            output.append(canonical)
    if not output:
        raise ValueError(f"{name} cannot be empty")
    return output


def _validate_exit_axes(
    targets: Sequence[float],
    fallback_targets: Sequence[float],
    stops: Sequence[float],
) -> tuple[list[float], list[float], list[float]]:
    normalized_targets = _validate_axis(targets, TARGET_LEVELS, "target")
    normalized_fallbacks = _validate_axis(
        fallback_targets, FALLBACK_TARGET_LEVELS, "fallback-target"
    )
    normalized_stops = _validate_axis(stops, STOP_LEVELS, "stop")
    valid_pairs = [
        (target, fallback)
        for target in normalized_targets
        for fallback in normalized_fallbacks
        if 0.0 <= fallback <= target <= 1.0
    ]
    if not valid_pairs:
        raise ValueError(
            "no valid target/fallback pair: require 0 <= fallback <= target <= 1"
        )
    if any(stop <= 1.0 for stop in normalized_stops):
        raise ValueError("stop must be greater than 1.0")
    return normalized_targets, normalized_fallbacks, normalized_stops


def select_variants(
    explicit: Sequence[str] | None,
    max_variants: int,
    allow_expensive: bool,
) -> list[SignalVariant]:
    if max_variants <= 0:
        raise ValueError("--max-variants must be positive")
    if max_variants > len(ALL_VARIANTS):
        raise ValueError(
            f"--max-variants cannot exceed the explicit catalog size {len(ALL_VARIANTS)}"
        )
    if explicit:
        selected = [BASELINE_VARIANT]
        selected.extend(_parse_variant(token) for token in explicit)
    else:
        if max_variants > len(STAGED_VARIANTS) and not allow_expensive:
            raise ValueError(
                "--max-variants above the five-variant staged probe requires "
                "--allow-expensive"
            )
        catalog = STAGED_VARIANTS if max_variants <= len(STAGED_VARIANTS) else ALL_VARIANTS
        selected = list(catalog[:max_variants])
    unique: list[SignalVariant] = []
    seen: set[str] = set()
    for variant in selected:
        variant.validate()
        if variant.variant_id not in seen:
            unique.append(variant)
            seen.add(variant.variant_id)
    if len(unique) > len(ALL_VARIANTS):
        raise ValueError(
            f"the explicit Smart Fib signal grid contains at most {len(ALL_VARIANTS)} variants"
        )
    return unique


def _write_output(path_text: str, output: Mapping[str, Any]) -> None:
    if path_text == "-":
        print(json.dumps(output, indent=2, default=str))
        return
    path = Path(path_text)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")


def _device_report(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device": "cpu",
            "gpu_used": False,
            "gpu_name": None,
            "vram_gb": None,
            "allocator": None,
        }
    props = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "gpu_used": True,
        "gpu_name": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=historical.DEFAULT_DATA_ROOT)
    parser.add_argument("--start", default=historical.DEFAULT_START)
    parser.add_argument("--end", default=historical.DEFAULT_END)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=(
            "baseline or s1k:s1d:min_span:setup_max_age:touch_buffer[:zone_start:zone_end]; "
            "baseline is automatic"
        ),
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=5,
        help="bounded staged-variant count when --variants is omitted (default: 5)",
    )
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--fallback-targets", nargs="+", default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--stops", nargs="+", default=None)
    parser.add_argument("--prep-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--brokerage-per-order", type=float, default=0.0)
    parser.add_argument("--fixed-cost-per-trade", type=float, default=None)
    parser.add_argument(
        "--timeframes",
        default=None,
        help="comma list of bar minutes (1,2,3,5) to run each selected variant on; "
        "bias filter period follows the x5 rule (filter_period = 5 * bar_minutes)",
    )
    parser.add_argument("--smoke", action="store_true", help="use exactly the first five available days")
    parser.add_argument("--allow-expensive", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="smoke/parity debug only")
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument(
        "--tensor-cache-dir",
        default=None,
        help="optional directory for variant-keyed NTC tensor caches",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts/f6_hybrid/smart_fib_optimus_grid_gpu.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.prep_workers <= 0:
        raise SystemExit("--prep-workers must be positive")
    if args.min_trades < 1:
        raise SystemExit("--min-trades must be at least one")
    if not args.smoke and not args.allow_expensive:
        raise SystemExit(
            "refusing a non-smoke run; use --smoke or --allow-expensive"
        )
    if args.allow_cpu and not args.smoke:
        raise SystemExit("--allow-cpu is restricted to smoke/parity debugging")
    if args.output != "-" and Path(args.output).exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    try:
        variants = select_variants(args.variants, args.max_variants, args.allow_expensive)
        if args.timeframes:
            timeframes = [int(piece) for piece in args.timeframes.split(",") if piece]
            if not timeframes or any(tf not in (1, 2, 3, 5) for tf in timeframes):
                raise ValueError("--timeframes must be a comma list of 1,2,3,5")
            expanded = []
            for variant in variants:
                for tf in timeframes:
                    expanded.append(
                        replace(
                            variant,
                            bar_minutes=tf,
                            filter_period=5 * tf,
                        ).validate()
                    )
            variants = expanded
        targets, fallback_targets, stops = _validate_exit_axes(
            _parse_float_tokens(args.targets, DEFAULT_TARGET_LEVELS),
            _parse_float_tokens(args.fallback_targets, DEFAULT_FALLBACK_TARGET_LEVELS),
            _parse_float_tokens(args.stops, DEFAULT_STOP_LEVELS),
        )
        thresholds = _validate_axis(
            _parse_float_tokens(args.thresholds, EXIT_THRESHOLDS),
            EXIT_THRESHOLDS,
            "threshold",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    exit_param_count = len(_grid_params(targets, fallback_targets, thresholds, stops))
    config_count = len(variants) * exit_param_count
    if config_count < 5:
        raise SystemExit(
            "the selected grid must contain at least five unique final configurations"
        )

    device = optimus.require_device(args.allow_cpu)
    print(f"[GPU INIT] device={device} allocator={CUDA_ALLOCATOR}", flush=True)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        torch.cuda.reset_peak_memory_stats(device)
        print(
            f"[GPU INIT] name={props.name} vram_gb={props.total_memory / (1024 ** 3):.2f} "
            f"cuda={torch.version.cuda}",
            flush=True,
        )

    adapter = PolarsHistoricalDataAdapter(
        args.data_root,
        start=args.start,
        end=args.end,
    )
    available = adapter.available_days(args.start, args.end)
    if not available:
        raise SystemExit(f"no overlapping index/options days in {args.start}..{args.end}")
    days = available[:5] if args.smoke else available
    if args.smoke and len(days) != 5:
        raise SystemExit("--smoke requires exactly five available days")
    if len(days) < 3:
        raise SystemExit("at least three available days are required for parity")
    parity_days = days[:3]
    tensor_cache_dir = (
        Path(args.tensor_cache_dir) if args.tensor_cache_dir is not None else None
    )

    all_configs: list[dict[str, Any]] = []
    variant_reports: list[dict[str, Any]] = []
    total_cpu_seconds = 0.0
    total_parity_cuda_ms = 0.0
    total_grid_cuda_ms = 0.0
    total_grid_wall = 0.0
    run_started = time.perf_counter()

    for variant_index, variant in enumerate(variants, start=1):
        print(
            f"[VARIANT] {variant_index}/{len(variants)} {variant.variant_id} "
            f"S1=({variant.s1_k_period},{variant.s1_d_period}) "
            f"span={variant.min_span:g} age={variant.setup_max_age} "
            f"buffer={variant.touch_buffer:g} "
            f"zone=({variant.zone_start:g},{variant.zone_end:g}) "
            f"tf={variant.bar_minutes}m bias={variant.filter_period}m",
            flush=True,
        )
        bundle = collect_variant_cpu_dataset(
            adapter,
            days,
            variant,
            workers=args.prep_workers,
            parity_days=parity_days,
            tensor_cache_dir=tensor_cache_dir,
        )
        total_cpu_seconds += bundle.prep_seconds
        gpu_variant = to_gpu_dataset(bundle.dataset, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluator = optimus.GpuEvaluator(
            gpu_variant.engine,
            args.batch_size,
            brokerage_per_order=args.brokerage_per_order,
            fixed_cost_per_trade=args.fixed_cost_per_trade,
        )
        print(
            f"[PARITY] {variant.variant_id} exactly three dates "
            f"{parity_days}",
            flush=True,
        )
        parity = validate_variant_parity(
            adapter,
            bundle,
            gpu_variant,
            evaluator,
            parity_days,
            targets,
            fallback_targets,
            thresholds,
            stops,
        )
        total_parity_cuda_ms += float(parity["gpu_cuda_ms"])
        print(f"[PARITY] {variant.variant_id} PASS", flush=True)
        configs, grid_timing = evaluate_variant_grid(
            variant,
            gpu_variant,
            evaluator,
            days,
            targets,
            fallback_targets,
            thresholds,
            stops,
        )
        total_grid_cuda_ms += float(grid_timing["cuda_ms"])
        total_grid_wall += float(grid_timing["wall_seconds"])
        all_configs.extend(configs)
        variant_reports.append({
            "variant": variant.as_dict(),
            "cache": {
                "source": bundle.cache_source,
                "path": bundle.cache_path,
                "identity_in_key": True,
            },
            "tensor_contract": {
                "logical_shape": [bundle.dataset.n_days, T_BARS, bundle.dataset.contract_slots],
                "logical_format": "(N,T,C)",
                "engine_view": "(N,C,T)",
            },
            "events": {
                "raw": dict(zip(days, bundle.dataset.raw_event_counts)),
                "selected": dict(zip(days, bundle.dataset.selected_event_counts)),
                "total_selected": sum(bundle.dataset.selected_event_counts),
            },
            "timing": {
                "cpu_prep_seconds": round(bundle.prep_seconds, 6),
                "parity_gpu_cuda_ms": parity["gpu_cuda_ms"],
                "parity_gpu_wall_seconds": parity["gpu_wall_seconds"],
                "grid_gpu_cuda_ms": grid_timing["cuda_ms"],
                "grid_gpu_wall_seconds": grid_timing["wall_seconds"],
                "grid_evaluations": grid_timing["grid_evaluations"],
                "batch_size": args.batch_size,
                "prep_workers": args.prep_workers,
            },
            "parity": parity,
        })
        del evaluator, gpu_variant
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    config_ids = [config["config_id"] for config in all_configs]
    if len(config_ids) != len(set(config_ids)):
        raise RuntimeError("GPU grid produced duplicate final configuration identities")
    eligible = [config for config in all_configs if config["trades"] >= args.min_trades]
    eligible.sort(
        key=lambda config: (
            float(config["composite_score"]),
            float(config["net_points"]),
            -float(config["max_drawdown_points"]),
            -float(config["fees_rs"]),
        ),
        reverse=True,
    )
    top_five = eligible[:5]
    for rank, config in enumerate(top_five, start=1):
        variant = config["variant"]
        print(
            f"[TOP {rank}] {variant['variant_id']} "
            f"target={config['target_level']:g} fallback={config['fallback_target_level']:g} "
            f"thr={config['option_point_threshold']:g} stop={config['stop_level']:g} "
            f"trades={config['trades']} WR={config['win_rate']:.2f}% "
            f"net={config['net_points']:+.2f}pts/Rs {config['net_rs']:+,.2f} "
            f"DD={config['max_drawdown_points']:.2f} PF={config['profit_factor']} "
            f"fees=Rs {config['fees_rs']:,.2f} score={config['composite_score']:+.2f}",
            flush=True,
        )
    if len(top_five) < 5:
        print(
            f"[RANK] only {len(top_five)} configurations passed min-trades="
            f"{args.min_trades}; no zero-trade configuration is promoted",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory = {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2),
        }
    else:
        memory = {"peak_allocated_mb": None, "peak_reserved_mb": None}
    output = {
        "engine": "smart_fib_optimus_grid_gpu",
        "mode": "smoke" if args.smoke else "bounded_full_window",
        "data_root": str(Path(args.data_root)),
        "date_range": {"start": args.start, "end": args.end},
        "days": days,
        "selection": {
            "variant_count": len(variants),
            "source": "explicit_variants" if args.variants else "staged_shortlist",
            "max_variants": args.max_variants,
            "variants": [variant.as_dict() for variant in variants],
        },
        "signal_grid_axes": {
            "zone_pairs": [list(value) for value in ZONE_PAIRS],
            "s1_variants": [list(value) for value in S1_VARIANTS],
            "min_span": list(MIN_SPANS),
            "setup_max_age": list(SETUP_MAX_AGES),
            "touch_buffer": list(TOUCH_BUFFERS),
        },
        "exit_grid_axes": {
            "target_level": list(targets),
            "fallback_target_level": list(fallback_targets),
            "target_levels_allowed": list(TARGET_LEVELS),
            "fallback_target_levels_allowed": list(FALLBACK_TARGET_LEVELS),
            "valid_target_fallback_pairs": [
                [target, fallback]
                for target in targets
                for fallback in fallback_targets
                if fallback <= target
            ],
            "option_point_threshold": list(thresholds),
            "stop_level": list(stops),
            "stop_levels_allowed": list(STOP_LEVELS),
        },
        "execution_contract": {
            "entry_stream": "CPU extract_day_events once per selected variant/day",
            "cpu_process_day_in_trial_loop": False,
            "actual_option_ohlc": True,
            "lot_size": LOT_SIZE,
            "brokerage_per_order": args.brokerage_per_order,
            "fixed_cost_per_trade": args.fixed_cost_per_trade,
            "fixed_bars": T_BARS,
            "one_global_position_per_day": True,
            "matrix_engine": "smart_fib_optimus_gpu.simulate_event_batch_matrix",
            "grid_parameters": "target/fallback/threshold/stop evaluated in GPU batches",
        },
        "parity": {
            "required_dates": parity_days,
            "exactly_three_dates": True,
            "all_variants_passed": all(report["parity"]["passed"] for report in variant_reports),
            "variants": [report["parity"] for report in variant_reports],
        },
        "hardware": _device_report(device),
        "gpu_evidence": {
            "torch_inference_mode": True,
            "pinned_host_transfer": device.type == "cuda",
            "cuda_allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "batch_size": args.batch_size,
            "matrix_first": True,
            "batched_exit_axes": [
                "target_level",
                "fallback_target_level",
                "option_point_threshold",
                "stop_level",
            ],
            "variant_timings": [report["timing"] for report in variant_reports],
        },
        "timing": {
            "total_wall_seconds": round(time.perf_counter() - run_started, 3),
            "cpu_prep_seconds": round(total_cpu_seconds, 3),
            "parity_gpu_cuda_ms": round(total_parity_cuda_ms, 3),
            "grid_gpu_cuda_ms": round(total_grid_cuda_ms, 3),
            "grid_gpu_wall_seconds": round(total_grid_wall, 3),
            "memory": memory,
        },
        "min_trade_guard": args.min_trades,
        "configs_evaluated": len(all_configs),
        "unique_configurations": len(set(config_ids)),
        "eligible_configs": len(eligible),
        "variant_reports": variant_reports,
        "results": all_configs,
        "top_five": top_five,
    }
    _write_output(args.output, output)
    print(
        json.dumps(
            {
                "engine": output["engine"],
                "gpu_used": output["hardware"]["gpu_used"],
                "days": len(days),
                "variants": len(variants),
                "configs": len(all_configs),
                "parity": output["parity"]["all_variants_passed"],
                "top_five": len(top_five),
                "output": args.output,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
