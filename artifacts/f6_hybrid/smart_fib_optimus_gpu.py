"""GPU-first Smart Fib Optimus evaluator.

The CPU phase adapts the historical CSV archive and extracts the canonical Smart
Fib event stream once per day. Trial evaluation never calls ``process_day`` or
rebuilds indicators. It replays fixed event metadata and actual option OHLC
arrays on CUDA in batched ``(B, N, T)`` form, where ``T`` is always 375 bars.

Signal parameters are intentionally fixed to the current Smart Fib contract:
combined event stream, minimum span 15, zero touch buffer, and 45-minute setup
age. The only optimized field is the valid Fib stop extension (1.155 or 1.25).
The primary target, 10-point premium threshold, and fallback target remain
0.29, 10, and 0.0 respectively so the GPU evaluator cannot silently change the
entry stream or exit semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.smart_fib_cuda_bootstrap import configure_cuda_toolkit

configure_cuda_toolkit()

import numpy as np
import optuna
import torch

from artifacts.f6_hybrid import marni_fib_core_combo_cache as smart_core
from artifacts.f6_hybrid import smart_fib_optimizer as historical
from backtest_walkforward_fees import (
    BROKERAGE_PER_ORDER,
    EXCHANGE_PCT,
    GST_PCT,
    SEBI_PCT,
    SLIPPAGE_PTS,
    STAMP_PCT,
    STT_PCT,
)


T_BARS = 375
BAR_START_MINUTE = 555
SESSION_START_MINUTE = 560
SESSION_END_MINUTE = 900
SESSION_START_SLOT = SESSION_START_MINUTE - BAR_START_MINUTE
SESSION_END_SLOT = SESSION_END_MINUTE - BAR_START_MINUTE
LOT_SIZE = int(smart_core.LOT_SIZE)
ENTRY_LEVEL = 0.786
PRIMARY_TARGET_LEVEL = 0.29
FALLBACK_TARGET_LEVEL = 0.0
OPTION_POINT_THRESHOLD = 10.0
SIGNAL_MIN_SPAN = 15.0
SIGNAL_TOUCH_BUFFER = 0.0
SIGNAL_SETUP_MAX_AGE = 45
STOP_LEVELS = (1.155, 1.25)
CONSECUTIVE_LOSS_LIMIT = 4
# Historical spot movement can expose more than 64 candidate strikes over a
# long window. Keep a generous safety ceiling while allocating only the actual
# per-dataset maximum contract axis.
CONTRACT_SLOT_LIMIT = 256
NET_POINTS_TOLERANCE = 0.05
TRADES_TOLERANCE = 0
CACHE_VERSION = 1


class ParityError(RuntimeError):
    """Raised when the GPU replay does not match the CPU reference."""


@dataclass
class DayRaw:
    day: str
    prepared: dict[str, Any]
    selected_events: list[dict[str, Any]]
    raw_event_count: int
    contract_keys: list[tuple[str, int]]


_CPU_WORKER_ADAPTER: historical.CsvHistoricalDataAdapter | None = None


def _init_cpu_worker(data_root: str, start: str, end: str) -> None:
    global _CPU_WORKER_ADAPTER
    _CPU_WORKER_ADAPTER = historical.CsvHistoricalDataAdapter(
        data_root,
        start=start,
        end=end,
        cache_days=8,
    )


def _extract_day_raw(task: tuple[str, str, str, str]) -> DayRaw:
    """Worker entry point for one-time CPU preparation; never touches CUDA."""
    data_root, start, end, day = task
    adapter = _CPU_WORKER_ADAPTER
    if adapter is None:
        adapter = historical.CsvHistoricalDataAdapter(
            data_root,
            start=start,
            end=end,
            cache_days=8,
        )
    prepared = smart_core.extract_day_events(
        day,
        cache_loader=adapter.load_day_cache,
        min_span=SIGNAL_MIN_SPAN,
        touch_buffer=SIGNAL_TOUCH_BUFFER,
        setup_max_age=SIGNAL_SETUP_MAX_AGE,
        debug=False,
    )
    if not prepared:
        raise RuntimeError(f"no Smart Fib cache payload for {day}")
    selected, raw_count = _select_day_events(prepared)
    return DayRaw(
        day,
        dict(prepared),
        selected,
        raw_count,
        list(prepared["records"]),
    )


@dataclass
class CpuDataset:
    days: list[str]
    index_open: np.ndarray
    index_high: np.ndarray
    index_low: np.ndarray
    index_close: np.ndarray
    index_valid: np.ndarray
    contract_open: np.ndarray
    contract_high: np.ndarray
    contract_low: np.ndarray
    contract_close: np.ndarray
    contract_valid: np.ndarray
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
        return int(self.contract_close.shape[1])


@dataclass
class GpuDataset:
    days: list[str]
    index_open: torch.Tensor
    index_high: torch.Tensor
    index_low: torch.Tensor
    index_close: torch.Tensor
    index_valid: torch.Tensor
    contract_open: torch.Tensor
    contract_high: torch.Tensor
    contract_low: torch.Tensor
    contract_close: torch.Tensor
    contract_valid: torch.Tensor
    event_mask: torch.Tensor
    event_contract: torch.Tensor
    event_entry: torch.Tensor
    event_fib_high: torch.Tensor
    event_fib_low: torch.Tensor
    event_index_source: torch.Tensor
    event_price_rise: torch.Tensor
    event_premium_rise: torch.Tensor
    event_high_to_low: torch.Tensor
    event_side_ce: torch.Tensor
    event_symbols: list[list[str | None]]
    contract_symbols: list[list[str | None]]
    raw_event_counts: list[int]
    selected_event_counts: list[int]
    device: torch.device

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def contract_slots(self) -> int:
        return int(self.contract_close.shape[1])


@dataclass
class PackedEvents:
    """Flattened Smart Fib events used by the matrix exit kernel."""

    day_index: torch.Tensor
    time_index: torch.Tensor
    contract_slot: torch.Tensor
    entry_price: torch.Tensor
    fib_high: torch.Tensor
    fib_low: torch.Tensor
    index_source: torch.Tensor
    price_rise: torch.Tensor
    premium_rise: torch.Tensor
    high_to_low: torch.Tensor
    event_grid: torch.Tensor


def pack_events(data: GpuDataset) -> PackedEvents:
    """Flatten sparse events once; no trial touches the CPU event extractor."""
    coordinates = torch.nonzero(data.event_mask, as_tuple=False)
    day_index = coordinates[:, 0].long()
    time_index = coordinates[:, 1].long()
    event_grid = torch.full(
        data.event_mask.shape,
        -1,
        dtype=torch.long,
        device=data.device,
    )
    event_grid[day_index, time_index] = torch.arange(
        coordinates.shape[0], device=data.device, dtype=torch.long
    )
    return PackedEvents(
        day_index=day_index,
        time_index=time_index,
        contract_slot=data.event_contract[day_index, time_index].long(),
        entry_price=data.event_entry[day_index, time_index],
        fib_high=data.event_fib_high[day_index, time_index],
        fib_low=data.event_fib_low[day_index, time_index],
        index_source=data.event_index_source[day_index, time_index],
        price_rise=data.event_price_rise[day_index, time_index],
        premium_rise=data.event_premium_rise[day_index, time_index],
        high_to_low=data.event_high_to_low[day_index, time_index],
        event_grid=event_grid,
    )


def require_device(allow_cpu: bool) -> torch.device:
    """Fail explicitly unless CUDA exists; ``--allow-cpu`` is opt-in only."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        "CUDA is required for Smart Fib Optimus evaluation. "
        "torch.cuda.is_available() is false; pass --allow-cpu only for parity/debug."
    )


def _slot(minute: int) -> int | None:
    value = int(minute) - BAR_START_MINUTE
    return value if 0 <= value < T_BARS else None


def _put_ohlc(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    minute: int,
    row: Mapping[str, Any],
) -> None:
    position = _slot(minute)
    if position is None:
        return
    for array, field in zip(arrays, ("open", "high", "low", "close")):
        array[position] = float(row[field])


def _event_orientation(event: Mapping[str, Any]) -> bool:
    return event.get("orientation") == "high_to_low"


def _select_day_events(prepared: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Apply the CPU global-slot ordering once, retaining one reachable event/bar."""
    selected: list[dict[str, Any]] = []
    seen_minutes: set[int] = set()
    signals = list(prepared["signals"])
    bars = prepared["bars"]
    records = prepared["records"]
    for signal in signals:
        minute = int(signal["minute"])
        if minute in seen_minutes:
            continue
        if not SESSION_START_MINUTE <= minute < SESSION_END_MINUTE:
            continue
        key = (str(signal["side"]), int(signal["strike"]))
        if key not in bars or minute not in bars[key] or key not in records:
            raise ParityError(
                f"Smart Fib event has no actual option OHLC row: {key} at {minute}"
            )
        seen_minutes.add(minute)
        selected.append(dict(signal))
    return selected, len(signals)


def collect_cpu_dataset(
    adapter: historical.CsvHistoricalDataAdapter,
    days: Sequence[str],
    *,
    workers: int = 1,
) -> CpuDataset:
    """Load CSVs and extract the canonical event stream exactly once per day.

    ``workers`` applies only to this CPU preparation phase. CUDA evaluation
    remains a single process with one resident dataset, as required by the
    Optimus runbook.
    """
    raw_days: list[DayRaw] = []
    max_contracts = 0
    total_days = len(days)
    prep_started = time.perf_counter()
    if workers > 1:
        task_args = [
            (str(adapter.data_root), adapter.start, adapter.end, day)
            for day in days
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_cpu_worker,
            initargs=(str(adapter.data_root), adapter.start, adapter.end),
        ) as pool:
            raw_iterator = pool.map(_extract_day_raw, task_args, chunksize=1)
            extracted = enumerate(raw_iterator, start=1)
    else:
        def serial_iterator():
            for day in days:
                prepared = smart_core.extract_day_events(
                    day,
                    cache_loader=adapter.load_day_cache,
                    min_span=SIGNAL_MIN_SPAN,
                    touch_buffer=SIGNAL_TOUCH_BUFFER,
                    setup_max_age=SIGNAL_SETUP_MAX_AGE,
                    debug=False,
                )
                if not prepared:
                    raise RuntimeError(f"no Smart Fib cache payload for {day}")
                selected, raw_count = _select_day_events(prepared)
                yield DayRaw(
                    day,
                    dict(prepared),
                    selected,
                    raw_count,
                    list(prepared["records"]),
                )
        extracted = enumerate(serial_iterator(), start=1)

    for day_index, raw in extracted:
        day = raw.day
        contract_keys = raw.contract_keys
        max_contracts = max(max_contracts, len(contract_keys))
        raw_days.append(raw)
        if day_index == 1 or day_index % 25 == 0 or day_index == total_days:
            elapsed = time.perf_counter() - prep_started
            rate = day_index / elapsed if elapsed else 0.0
            print(
                f"[CPU PREP] {day_index}/{total_days} days | "
                f"{rate:.2f} days/s | last={day}",
                flush=True,
            )

    if max_contracts > CONTRACT_SLOT_LIMIT:
        raise RuntimeError(
            f"{max_contracts} contracts are needed, above fixed contract-slot limit "
            f"C={CONTRACT_SLOT_LIMIT}; refusing to truncate actual candidates"
        )
    n_days = len(raw_days)
    contract_slots = max(1, max_contracts)
    shape = (n_days, T_BARS)
    index_open = np.zeros(shape, dtype=np.float32)
    index_high = np.zeros(shape, dtype=np.float32)
    index_low = np.zeros(shape, dtype=np.float32)
    index_close = np.zeros(shape, dtype=np.float32)
    index_valid = np.zeros(shape, dtype=np.bool_)
    contract_shape = (n_days, contract_slots, T_BARS)
    contract_open = np.zeros(contract_shape, dtype=np.float32)
    contract_high = np.zeros(contract_shape, dtype=np.float32)
    contract_low = np.zeros(contract_shape, dtype=np.float32)
    contract_close = np.zeros(contract_shape, dtype=np.float32)
    contract_valid = np.zeros(contract_shape, dtype=np.bool_)
    event_mask = np.zeros(shape, dtype=np.bool_)
    event_contract = np.zeros(shape, dtype=np.int64)
    event_entry = np.zeros(shape, dtype=np.float32)
    event_fib_high = np.zeros(shape, dtype=np.float32)
    event_fib_low = np.zeros(shape, dtype=np.float32)
    event_index_source = np.zeros(shape, dtype=np.bool_)
    event_price_rise = np.zeros(shape, dtype=np.bool_)
    event_premium_rise = np.zeros(shape, dtype=np.bool_)
    event_high_to_low = np.zeros(shape, dtype=np.bool_)
    event_side_ce = np.zeros(shape, dtype=np.bool_)
    event_symbols: list[list[str | None]] = [
        [None] * T_BARS for _ in range(n_days)
    ]
    contract_symbols: list[list[str | None]] = [
        [None] * contract_slots for _ in range(n_days)
    ]
    raw_event_counts: list[int] = []
    selected_event_counts: list[int] = []

    for day_index, raw in enumerate(raw_days):
        prepared = raw.prepared
        spot = prepared["spot"]
        for minute, open_, high, low, close in zip(
            spot["min"], spot["open"], spot["high"], spot["low"], spot["close"]
        ):
            position = _slot(int(minute))
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
                contract_open[day_index, slot_index],
                contract_high[day_index, slot_index],
                contract_low[day_index, slot_index],
                contract_close[day_index, slot_index],
            )
            for row in record["current"]:
                minute = int(row["minute"])
                position = _slot(minute)
                if position is None:
                    continue
                _put_ohlc(arrays, minute, row)
                contract_valid[day_index, slot_index, position] = True

        for event in raw.selected_events:
            minute = int(event["minute"])
            position = _slot(minute)
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
            event_premium_rise[day_index, position] = bool(event.get("profit_on_rise", True))
            event_high_to_low[day_index, position] = _event_orientation(event)
            event_side_ce[day_index, position] = str(event["side"]) == "CE"
            event_symbols[day_index][position] = contract_symbols[day_index][slot_index]

        raw_event_counts.append(raw.raw_event_count)
        selected_event_counts.append(len(raw.selected_events))

    return CpuDataset(
        days=list(days),
        index_open=index_open,
        index_high=index_high,
        index_low=index_low,
        index_close=index_close,
        index_valid=index_valid,
        contract_open=contract_open,
        contract_high=contract_high,
        contract_low=contract_low,
        contract_close=contract_close,
        contract_valid=contract_valid,
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


def _initial_device_copy(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Copy a preprocessed array once; CUDA copies use pinned async host memory."""
    host = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        host = host.pin_memory()
        return host.to(device, non_blocking=True)
    return host.to(device)


def to_gpu_dataset(cpu: CpuDataset, device: torch.device) -> GpuDataset:
    """Transfer all candidate data once and leave it resident on ``device``."""
    return GpuDataset(
        days=cpu.days,
        index_open=_initial_device_copy(cpu.index_open, device),
        index_high=_initial_device_copy(cpu.index_high, device),
        index_low=_initial_device_copy(cpu.index_low, device),
        index_close=_initial_device_copy(cpu.index_close, device),
        index_valid=_initial_device_copy(cpu.index_valid, device),
        contract_open=_initial_device_copy(cpu.contract_open, device),
        contract_high=_initial_device_copy(cpu.contract_high, device),
        contract_low=_initial_device_copy(cpu.contract_low, device),
        contract_close=_initial_device_copy(cpu.contract_close, device),
        contract_valid=_initial_device_copy(cpu.contract_valid, device),
        event_mask=_initial_device_copy(cpu.event_mask, device),
        event_contract=_initial_device_copy(cpu.event_contract, device),
        event_entry=_initial_device_copy(cpu.event_entry, device),
        event_fib_high=_initial_device_copy(cpu.event_fib_high, device),
        event_fib_low=_initial_device_copy(cpu.event_fib_low, device),
        event_index_source=_initial_device_copy(cpu.event_index_source, device),
        event_price_rise=_initial_device_copy(cpu.event_price_rise, device),
        event_premium_rise=_initial_device_copy(cpu.event_premium_rise, device),
        event_high_to_low=_initial_device_copy(cpu.event_high_to_low, device),
        event_side_ce=_initial_device_copy(cpu.event_side_ce, device),
        event_symbols=cpu.event_symbols,
        contract_symbols=cpu.contract_symbols,
        raw_event_counts=cpu.raw_event_counts,
        selected_event_counts=cpu.selected_event_counts,
        device=device,
    )


def save_cpu_dataset(cpu: CpuDataset, path: Path, data_root: str) -> None:
    """Persist the expensive CPU event extraction for later GPU reruns."""
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing tensor cache: {path}; "
            "choose a new --tensor-cache path"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache_version": CACHE_VERSION,
        "data_root": str(Path(data_root).resolve()),
        "days": cpu.days,
        "contract_slots": cpu.contract_slots,
        "raw_event_counts": cpu.raw_event_counts,
        "selected_event_counts": cpu.selected_event_counts,
        "event_symbols": cpu.event_symbols,
        "contract_symbols": cpu.contract_symbols,
    }
    names = (
        "index_open", "index_high", "index_low", "index_close", "index_valid",
        "contract_open", "contract_high", "contract_low", "contract_close",
        "contract_valid", "event_mask", "event_contract", "event_entry",
        "event_fib_high", "event_fib_low", "event_index_source",
        "event_price_rise", "event_premium_rise", "event_high_to_low",
        "event_side_ce",
    )
    arrays = {name: getattr(cpu, name) for name in names}
    np.savez_compressed(path, **arrays, metadata=np.array(json.dumps(metadata)))


def load_cpu_dataset(path: Path, data_root: str, days: Sequence[str]) -> CpuDataset | None:
    """Load a matching event cache; return ``None`` when it is stale/missing."""
    if not path.exists():
        return None
    names = (
        "index_open", "index_high", "index_low", "index_close", "index_valid",
        "contract_open", "contract_high", "contract_low", "contract_close",
        "contract_valid", "event_mask", "event_contract", "event_entry",
        "event_fib_high", "event_fib_low", "event_index_source",
        "event_price_rise", "event_premium_rise", "event_high_to_low",
        "event_side_ce",
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("cache_version") != CACHE_VERSION:
                return None
            if metadata.get("data_root") != str(Path(data_root).resolve()):
                return None
            if metadata.get("days") != list(days):
                return None
            arrays = {name: archive[name].copy() for name in names}
        return CpuDataset(
            days=list(metadata["days"]),
            **arrays,
            event_symbols=metadata["event_symbols"],
            contract_symbols=metadata["contract_symbols"],
            raw_event_counts=[int(value) for value in metadata["raw_event_counts"]],
            selected_event_counts=[int(value) for value in metadata["selected_event_counts"]],
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _round2(value: torch.Tensor) -> torch.Tensor:
    return torch.round(value * 100.0) / 100.0


def _trade_fee(
    entry_fill: torch.Tensor,
    exit_fill: torch.Tensor,
    brokerage_per_order: float = BROKERAGE_PER_ORDER,
    fixed_cost_per_trade: float | None = None,
) -> torch.Tensor:
    """Vectorized copy of backtest_walkforward_fees.trade_cost()."""
    if fixed_cost_per_trade is not None:
        return torch.full_like(entry_fill, float(fixed_cost_per_trade))
    prem_buy = entry_fill * LOT_SIZE
    prem_sell = exit_fill * LOT_SIZE
    stt = STT_PCT / 100.0 * prem_sell
    exchange = EXCHANGE_PCT / 100.0 * (prem_buy + prem_sell)
    sebi = SEBI_PCT / 100.0 * (prem_buy + prem_sell)
    stamp = STAMP_PCT / 100.0 * prem_buy
    brokerage = float(brokerage_per_order) * 2.0
    gst = GST_PCT / 100.0 * (brokerage + exchange + sebi)
    return _round2(stt + exchange + sebi + stamp + gst + brokerage)


@torch.inference_mode()
def _vectorized_event_outcomes(
    data: GpuDataset,
    events: PackedEvents,
    controls: Mapping[str, torch.Tensor],
    brokerage_per_order: float = BROKERAGE_PER_ORDER,
    fixed_cost_per_trade: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute every event's first exit with one GPU future-bar matrix."""
    batch_size = int(controls["stop_level"].shape[0])
    event_count = int(events.time_index.shape[0])
    max_future = SESSION_END_SLOT - SESSION_START_SLOT
    offsets = torch.arange(max_future, device=data.device, dtype=torch.long)
    columns = events.time_index.unsqueeze(1) + 1 + offsets.unsqueeze(0)
    valid = (columns < SESSION_END_SLOT) & (columns < T_BARS)
    safe_columns = columns.clamp(min=0, max=T_BARS - 1)
    day_columns = events.day_index.unsqueeze(1).expand(event_count, max_future)
    contract_columns = events.contract_slot.unsqueeze(1).expand(event_count, max_future)

    index_high = data.index_high[day_columns, safe_columns]
    index_low = data.index_low[day_columns, safe_columns]
    option_high = data.contract_high[day_columns, contract_columns, safe_columns]
    option_low = data.contract_low[day_columns, contract_columns, safe_columns]
    option_close = data.contract_close[day_columns, contract_columns, safe_columns]
    index_valid = data.index_valid[day_columns, safe_columns]
    option_valid = data.contract_valid[day_columns, contract_columns, safe_columns]
    valid = valid & index_valid & option_valid

    source_high = torch.where(events.index_source.unsqueeze(1), index_high, option_high)
    source_low = torch.where(events.index_source.unsqueeze(1), index_low, option_low)
    source_high = torch.where(valid, source_high, torch.full_like(source_high, -1e9))
    source_low = torch.where(valid, source_low, torch.full_like(source_low, 1e9))
    # The expanded grid keeps broker OHLC/Fib values in float64, so preserve
    # Python CPU comparison semantics there. The legacy float32 path gets the
    # two-decimal normalization needed to avoid a representation-only miss.
    if data.index_high.dtype != torch.float64:
        source_high = _round2(source_high)
        source_low = _round2(source_low)

    span = (events.fib_high - events.fib_low).unsqueeze(0)
    high_to_low = events.high_to_low.unsqueeze(0)
    stop_level = controls["stop_level"].view(batch_size, 1)
    primary_level = controls["target_level"].view(batch_size, 1)
    fallback_level = controls["fallback_target_level"].view(batch_size, 1)
    threshold = controls["option_point_threshold"].view(batch_size, 1)
    stop_price = torch.where(
        high_to_low,
        events.fib_high.unsqueeze(0) - stop_level * span,
        events.fib_low.unsqueeze(0) + stop_level * span,
    )
    primary_price = torch.where(
        high_to_low,
        events.fib_high.unsqueeze(0) - primary_level * span,
        events.fib_low.unsqueeze(0) + primary_level * span,
    )
    fallback_price = torch.where(
        high_to_low,
        events.fib_high.unsqueeze(0) - fallback_level * span,
        events.fib_low.unsqueeze(0) + fallback_level * span,
    )
    if data.index_high.dtype != torch.float64:
        stop_price = _round2(stop_price)
        primary_price = _round2(primary_price)
        fallback_price = _round2(fallback_price)
    price_rise = events.price_rise.view(1, event_count, 1)
    hit_stop = torch.where(
        price_rise,
        source_low.unsqueeze(0) <= stop_price.unsqueeze(-1),
        source_high.unsqueeze(0) >= stop_price.unsqueeze(-1),
    )
    hit_primary = torch.where(
        price_rise,
        source_high.unsqueeze(0) >= primary_price.unsqueeze(-1),
        source_low.unsqueeze(0) <= primary_price.unsqueeze(-1),
    )
    hit_fallback = torch.where(
        price_rise,
        source_high.unsqueeze(0) >= fallback_price.unsqueeze(-1),
        source_low.unsqueeze(0) <= fallback_price.unsqueeze(-1),
    )

    big = max_future + 1
    stop_any = hit_stop.any(dim=2)
    primary_any = hit_primary.any(dim=2)
    stop_first = torch.where(
        stop_any,
        torch.argmax(hit_stop.to(torch.int8), dim=2),
        torch.full((batch_size, event_count), big, device=data.device, dtype=torch.long),
    )
    primary_first = torch.where(
        primary_any,
        torch.argmax(hit_primary.to(torch.int8), dim=2),
        torch.full((batch_size, event_count), big, device=data.device, dtype=torch.long),
    )
    primary_before_stop = primary_any & (primary_first < stop_first)
    option_close_batch = option_close.unsqueeze(0).expand(batch_size, -1, -1)
    primary_close = option_close_batch.gather(
        2, primary_first.clamp(max=max_future - 1).unsqueeze(-1)
    ).squeeze(-1)
    premium_points = torch.where(
        events.premium_rise.view(1, event_count),
        primary_close - events.entry_price.view(1, event_count),
        events.entry_price.view(1, event_count) - primary_close,
    )
    fallback = (
        primary_before_stop
        & (primary_level == PRIMARY_TARGET_LEVEL)
        & (events.time_index.view(1, event_count) < SESSION_END_SLOT)
        & (premium_points < threshold)
    )

    primary_absolute = events.time_index.view(1, event_count) + 1 + primary_first
    after_primary = columns.unsqueeze(0) > primary_absolute.unsqueeze(-1)
    hit_stop_after = hit_stop & after_primary
    hit_fallback_after = hit_fallback & after_primary
    stop_after_any = hit_stop_after.any(dim=2)
    fallback_after_any = hit_fallback_after.any(dim=2)
    stop_after_first = torch.where(
        stop_after_any,
        torch.argmax(hit_stop_after.to(torch.int8), dim=2),
        torch.full((batch_size, event_count), big, device=data.device, dtype=torch.long),
    )
    fallback_after_first = torch.where(
        fallback_after_any,
        torch.argmax(hit_fallback_after.to(torch.int8), dim=2),
        torch.full((batch_size, event_count), big, device=data.device, dtype=torch.long),
    )
    initial_first = torch.minimum(stop_first, primary_first)
    fallback_first = torch.minimum(stop_after_first, fallback_after_first)
    exit_relative = torch.where(fallback, fallback_first, initial_first)
    timed_exit = exit_relative < big
    exit_absolute = torch.where(
        timed_exit,
        events.time_index.view(1, event_count) + 1 + exit_relative,
        torch.full_like(exit_relative, SESSION_END_SLOT),
    )
    exit_close = option_close_batch.gather(
        2, exit_relative.clamp(max=max_future - 1).unsqueeze(-1)
    ).squeeze(-1)
    eod_close = data.contract_close[
        events.day_index,
        events.contract_slot,
        torch.full_like(events.day_index, SESSION_END_SLOT),
    ].view(1, event_count).expand(batch_size, -1)
    exit_close = torch.where(timed_exit, exit_close, eod_close)
    entry_price = events.entry_price.view(1, event_count)
    premium_rise = events.premium_rise.view(1, event_count)
    entry_fill = torch.where(
        premium_rise,
        entry_price - SLIPPAGE_PTS,
        entry_price + SLIPPAGE_PTS,
    )
    exit_fill = torch.where(
        premium_rise,
        exit_close + SLIPPAGE_PTS,
        exit_close - SLIPPAGE_PTS,
    )
    raw_points = torch.where(
        premium_rise,
        exit_fill - entry_fill,
        entry_fill - exit_fill,
    )
    points = _round2(raw_points)
    fees = _trade_fee(entry_fill, exit_fill, brokerage_per_order, fixed_cost_per_trade)
    net_rs = _round2(points * LOT_SIZE - fees)
    return exit_absolute.long(), points, fees, net_rs


def _parameter_tensors(
    params: Sequence[Mapping[str, Any]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    def values(name: str) -> torch.Tensor:
        return torch.tensor(
            [float(param[name]) for param in params],
            dtype=dtype,
            device=device,
        )

    return {
        "stop_level": values("stop_level"),
        "target_level": values("target_level"),
        "fallback_target_level": values("fallback_target_level"),
        "option_point_threshold": values("option_point_threshold"),
        "max_trades_per_day": torch.tensor(
            [int(param.get("max_trades_per_day", -1)) for param in params],
            dtype=torch.int32,
            device=device,
        ),
        "daily_loss_limit_rs": torch.tensor(
            [float(param.get("daily_loss_limit_rs", float("inf"))) for param in params],
            dtype=dtype,
            device=device,
        ),
    }


@torch.inference_mode()
def simulate_event_batch(
    data: GpuDataset,
    params: Sequence[Mapping[str, Any]],
    day_mask: torch.Tensor | None = None,
    brokerage_per_order: float = BROKERAGE_PER_ORDER,
    fixed_cost_per_trade: float | None = None,
) -> dict[str, torch.Tensor]:
    """Replay a batch with one global position per day and unlimited re-entry.

    The entry stream is fixed by ``data.event_mask``. At every time slot the
    current position is exited before the first event for that minute is
    considered, matching the CPU simulator. Contract OHLC is selected from the
    fixed C-slot tensor by the open position's actual contract index. No scalar
    GPU readback occurs in this loop.
    """
    batch_size = len(params)
    device = data.device
    n_days = data.n_days
    dtype = data.contract_close.dtype
    controls = _parameter_tensors(params, device, dtype=dtype)
    primary_target = controls["target_level"].view(batch_size, 1)
    fallback_target = controls["fallback_target_level"].view(batch_size, 1)
    threshold = controls["option_point_threshold"].view(batch_size, 1)
    stop_level = controls["stop_level"].view(batch_size, 1)

    open_position = torch.zeros((batch_size, n_days), dtype=torch.bool, device=device)
    entry_bar = torch.zeros((batch_size, n_days), dtype=torch.long, device=device)
    contract_slot = torch.zeros((batch_size, n_days), dtype=torch.long, device=device)
    entry_price = torch.zeros((batch_size, n_days), dtype=dtype, device=device)
    fib_high = torch.zeros((batch_size, n_days), dtype=dtype, device=device)
    fib_low = torch.zeros((batch_size, n_days), dtype=dtype, device=device)
    active_target = primary_target.expand(-1, n_days).clone()
    source_index = torch.zeros((batch_size, n_days), dtype=torch.bool, device=device)
    side_ce = torch.zeros((batch_size, n_days), dtype=torch.bool, device=device)
    price_rise = torch.zeros((batch_size, n_days), dtype=torch.bool, device=device)
    premium_rise = torch.zeros((batch_size, n_days), dtype=torch.bool, device=device)
    high_to_low = torch.zeros((batch_size, n_days), dtype=torch.bool, device=device)

    trade_count = torch.zeros((batch_size, n_days), dtype=torch.int32, device=device)
    win_count = torch.zeros_like(trade_count)
    fee_sum = torch.zeros((batch_size, n_days), dtype=dtype, device=device)
    points_sum = torch.zeros_like(fee_sum)
    net_sum = torch.zeros_like(fee_sum)
    day_equity = torch.zeros_like(fee_sum)
    day_peak = torch.zeros_like(fee_sum)
    day_trough = torch.zeros_like(fee_sum)
    day_drawdown = torch.zeros_like(fee_sum)

    if day_mask is None:
        active_days = torch.ones((1, n_days), dtype=torch.bool, device=device)
    else:
        active_days = day_mask.view(1, n_days).bool()

    for time_slot in range(T_BARS):
        slot_index = contract_slot.clamp(min=0, max=data.contract_slots - 1)
        contract_high_t = data.contract_high[:, :, time_slot].unsqueeze(0).expand(batch_size, -1, -1)
        contract_low_t = data.contract_low[:, :, time_slot].unsqueeze(0).expand(batch_size, -1, -1)
        contract_close_t = data.contract_close[:, :, time_slot].unsqueeze(0).expand(batch_size, -1, -1)
        contract_valid_t = data.contract_valid[:, :, time_slot].unsqueeze(0).expand(batch_size, -1, -1)
        option_high = contract_high_t.gather(2, slot_index.unsqueeze(-1)).squeeze(-1)
        option_low = contract_low_t.gather(2, slot_index.unsqueeze(-1)).squeeze(-1)
        option_close = contract_close_t.gather(2, slot_index.unsqueeze(-1)).squeeze(-1)
        option_valid = contract_valid_t.gather(2, slot_index.unsqueeze(-1)).squeeze(-1)

        index_high = data.index_high[:, time_slot].view(1, n_days).expand(batch_size, -1)
        index_low = data.index_low[:, time_slot].view(1, n_days).expand(batch_size, -1)
        index_valid = data.index_valid[:, time_slot].view(1, n_days).expand(batch_size, -1)
        checkable = (
            open_position
            & (time_slot > entry_bar)
            & index_valid
            & option_valid
        )

        span = fib_high - fib_low
        stop_price = torch.where(
            high_to_low,
            fib_high - stop_level * span,
            fib_low + stop_level * span,
        )
        target_price = torch.where(
            high_to_low,
            fib_high - active_target * span,
            fib_low + active_target * span,
        )
        price_high = torch.where(source_index, index_high, option_high)
        price_low = torch.where(source_index, index_low, option_low)
        hit_stop = torch.where(price_rise, price_low <= stop_price, price_high >= stop_price)
        hit_target = torch.where(price_rise, price_high >= target_price, price_low <= target_price)
        target_reason = checkable & (~hit_stop) & hit_target
        stop_reason = checkable & hit_stop

        premium_points = torch.where(
            premium_rise,
            option_close - entry_price,
            entry_price - option_close,
        )
        fallback = (
            target_reason
            & (active_target == primary_target)
            & (primary_target == PRIMARY_TARGET_LEVEL)
            & (time_slot < SESSION_END_SLOT)
            & (premium_points < threshold)
        )
        close_reason = stop_reason | (target_reason & ~fallback) | (
            checkable & ~hit_stop & ~hit_target & (time_slot >= SESSION_END_SLOT)
        )
        close_reason = close_reason & open_position

        entry_fill = torch.where(premium_rise, entry_price - SLIPPAGE_PTS, entry_price + SLIPPAGE_PTS)
        exit_fill = torch.where(premium_rise, option_close + SLIPPAGE_PTS, option_close - SLIPPAGE_PTS)
        raw_points = torch.where(premium_rise, exit_fill - entry_fill, entry_fill - exit_fill)
        points = _round2(raw_points)
        fee = _trade_fee(entry_fill, exit_fill, brokerage_per_order, fixed_cost_per_trade)
        net_rs = _round2(points * LOT_SIZE - fee)
        exit_points = torch.where(close_reason, points, torch.zeros_like(points))
        exit_fee = torch.where(close_reason, fee, torch.zeros_like(fee))
        exit_net = torch.where(close_reason, net_rs, torch.zeros_like(net_rs))
        exit_wins = close_reason & (net_rs > 0.0)

        trade_count = trade_count + close_reason.to(torch.int32)
        win_count = win_count + exit_wins.to(torch.int32)
        fee_sum = fee_sum + exit_fee
        points_sum = points_sum + exit_points
        net_sum = net_sum + exit_net
        day_equity = day_equity + exit_net
        day_peak = torch.maximum(day_peak, day_equity)
        day_trough = torch.minimum(day_trough, day_equity)
        day_drawdown = torch.maximum(day_drawdown, day_peak - day_equity)
        active_target = torch.where(fallback, fallback_target.expand(-1, n_days), active_target)
        open_position = open_position & ~close_reason

        event_here = data.event_mask[:, time_slot].view(1, n_days).expand(batch_size, -1)
        enter = event_here & active_days & ~open_position
        event_contract = data.event_contract[:, time_slot].view(1, n_days).expand(batch_size, -1)
        event_entry = data.event_entry[:, time_slot].view(1, n_days).expand(batch_size, -1)
        event_high = data.event_fib_high[:, time_slot].view(1, n_days).expand(batch_size, -1)
        event_low = data.event_fib_low[:, time_slot].view(1, n_days).expand(batch_size, -1)
        event_index = data.event_index_source[:, time_slot].view(1, n_days).expand(batch_size, -1)
        event_side = data.event_side_ce[:, time_slot].view(1, n_days).expand(batch_size, -1)
        event_rise = data.event_price_rise[:, time_slot].view(1, n_days).expand(batch_size, -1)
        event_premium = data.event_premium_rise[:, time_slot].view(1, n_days).expand(batch_size, -1)
        event_orientation = data.event_high_to_low[:, time_slot].view(1, n_days).expand(batch_size, -1)
        open_position = open_position | enter
        entry_bar = torch.where(enter, torch.full_like(entry_bar, time_slot), entry_bar)
        contract_slot = torch.where(enter, event_contract, contract_slot)
        entry_price = torch.where(enter, event_entry, entry_price)
        fib_high = torch.where(enter, event_high, fib_high)
        fib_low = torch.where(enter, event_low, fib_low)
        source_index = torch.where(enter, event_index, source_index)
        side_ce = torch.where(enter, event_side, side_ce)
        price_rise = torch.where(enter, event_rise, price_rise)
        premium_rise = torch.where(enter, event_premium, premium_rise)
        high_to_low = torch.where(enter, event_orientation, high_to_low)
        active_target = torch.where(enter, primary_target.expand(-1, n_days), active_target)

    trades = trade_count.sum(dim=1)
    wins = win_count.sum(dim=1)
    fees = fee_sum.sum(dim=1)
    points = points_sum.sum(dim=1)
    net = net_sum.sum(dim=1)
    # Reconstruct chronological cross-day drawdown from per-day local paths.
    day_starts = torch.cat(
        [torch.zeros((batch_size, 1), dtype=dtype, device=device), net_sum[:, :-1]],
        dim=1,
    ).cumsum(dim=1)
    day_peak_candidates = day_starts + day_peak
    prior_peak = torch.cat(
        [torch.zeros((batch_size, 1), dtype=dtype, device=device), day_peak_candidates[:, :-1]],
        dim=1,
    )
    prior_peak = torch.cummax(prior_peak, dim=1).values
    global_peak = torch.maximum(prior_peak, day_peak_candidates)
    global_drawdown = global_peak - (day_starts + day_trough)
    max_dd_rs = global_drawdown.amax(dim=1)
    positives = torch.where(net_sum > 0.0, net_sum, torch.zeros_like(net_sum)).sum(dim=1)
    negatives = torch.where(net_sum <= 0.0, -net_sum, torch.zeros_like(net_sum)).sum(dim=1)
    return {
        "trades": trades,
        "wins": wins,
        "fees_rs": fees,
        "net_points": points,
        "net_rs": net,
        "max_drawdown_rs": max_dd_rs,
        "profit_positive_rs": positives,
        "profit_negative_rs": negatives,
        "daily_trades": trade_count,
        "daily_net_points": points_sum,
        "daily_net_rs": net_sum,
        "daily_drawdown_rs": day_drawdown,
        "daily_peak_rs": day_peak,
        "daily_trough_rs": day_trough,
    }


@torch.inference_mode()
def simulate_event_batch_matrix(
    data: GpuDataset,
    events: PackedEvents,
    params: Sequence[Mapping[str, Any]],
    day_mask: torch.Tensor | None = None,
    brokerage_per_order: float = BROKERAGE_PER_ORDER,
    fixed_cost_per_trade: float | None = None,
) -> dict[str, torch.Tensor]:
    """Matrix-first CUDA replay with a lightweight global-lock scan.

    First-hit exits for every sparse event are computed in parallel over the
    future-bar matrix. The remaining time scan only applies the one-position
    lock and re-entry state, so it no longer performs OHLC work or gathers a
    contract on every minute.
    """
    batch_size = len(params)
    device = data.device
    n_days = data.n_days
    dtype = data.contract_close.dtype
    controls = _parameter_tensors(params, device, dtype=dtype)
    exit_absolute, exit_points, exit_fees, exit_net = _vectorized_event_outcomes(
        data, events, controls, brokerage_per_order, fixed_cost_per_trade
    )

    open_position = torch.zeros((batch_size, n_days), dtype=torch.bool, device=device)
    current_exit = torch.full(
        (batch_size, n_days), SESSION_END_SLOT, dtype=torch.long, device=device
    )
    current_points = torch.zeros((batch_size, n_days), dtype=dtype, device=device)
    current_fees = torch.zeros_like(current_points)
    current_net = torch.zeros_like(current_points)
    trade_count = torch.zeros((batch_size, n_days), dtype=torch.int32, device=device)
    win_count = torch.zeros_like(trade_count)
    fee_sum = torch.zeros_like(current_points)
    points_sum = torch.zeros_like(current_points)
    net_sum = torch.zeros_like(current_points)
    day_equity = torch.zeros_like(current_points)
    day_peak = torch.zeros_like(current_points)
    day_trough = torch.zeros_like(current_points)
    day_drawdown = torch.zeros_like(current_points)
    consecutive_losses = torch.zeros(
        (batch_size, n_days), dtype=torch.int32, device=device
    )
    stopped = torch.zeros((batch_size, n_days), dtype=torch.bool, device=device)
    active_days = (
        torch.ones((1, n_days), dtype=torch.bool, device=device)
        if day_mask is None
        else day_mask.view(1, n_days).bool()
    )

    for time_slot in range(SESSION_END_SLOT + 1):
        close_now = open_position & (current_exit == time_slot)
        close_net = torch.where(close_now, current_net, torch.zeros_like(current_net))
        close_points = torch.where(close_now, current_points, torch.zeros_like(current_points))
        close_fees = torch.where(close_now, current_fees, torch.zeros_like(current_fees))
        close_wins = close_now & (current_net > 0.0)
        loss_streak = torch.where(
            close_now & ~close_wins,
            consecutive_losses + 1,
            torch.zeros_like(consecutive_losses),
        )
        consecutive_losses = torch.where(close_now, loss_streak, consecutive_losses)
        stopped = stopped | (close_now & (loss_streak >= CONSECUTIVE_LOSS_LIMIT))
        trade_count = trade_count + close_now.to(torch.int32)
        win_count = win_count + close_wins.to(torch.int32)
        fee_sum = fee_sum + close_fees
        points_sum = points_sum + close_points
        net_sum = net_sum + close_net
        day_equity = day_equity + close_net
        day_peak = torch.maximum(day_peak, day_equity)
        day_trough = torch.minimum(day_trough, day_equity)
        day_drawdown = torch.maximum(day_drawdown, day_peak - day_equity)
        open_position = open_position & ~close_now

        event_indices = events.event_grid[:, time_slot]
        has_event = event_indices >= 0
        safe_event_indices = event_indices.clamp(min=0)
        event_exit = exit_absolute[:, safe_event_indices]
        event_points = exit_points[:, safe_event_indices]
        event_fees = exit_fees[:, safe_event_indices]
        event_net = exit_net[:, safe_event_indices]
        enter = (
            has_event.view(1, n_days)
            & active_days
            & ~open_position
            & ~stopped
            & ((controls["max_trades_per_day"].view(batch_size, 1) < 0)
               | (trade_count < controls["max_trades_per_day"].view(batch_size, 1)))
            & (day_equity > -controls["daily_loss_limit_rs"].view(batch_size, 1))
            & (event_exit > time_slot)
        )
        open_position = open_position | enter
        current_exit = torch.where(enter, event_exit, current_exit)
        current_points = torch.where(enter, event_points, current_points)
        current_fees = torch.where(enter, event_fees, current_fees)
        current_net = torch.where(enter, event_net, current_net)

    trades = trade_count.sum(dim=1)
    wins = win_count.sum(dim=1)
    fees = fee_sum.sum(dim=1)
    points = points_sum.sum(dim=1)
    net = net_sum.sum(dim=1)
    day_starts = torch.cat(
        [torch.zeros((batch_size, 1), dtype=dtype, device=device), net_sum[:, :-1]],
        dim=1,
    ).cumsum(dim=1)
    day_peak_candidates = day_starts + day_peak
    prior_peak = torch.cat(
        [torch.zeros((batch_size, 1), dtype=dtype, device=device), day_peak_candidates[:, :-1]],
        dim=1,
    )
    prior_peak = torch.cummax(prior_peak, dim=1).values
    global_peak = torch.maximum(prior_peak, day_peak_candidates)
    global_drawdown = global_peak - (day_starts + day_trough)
    max_dd_rs = global_drawdown.amax(dim=1)
    positives = torch.where(net_sum > 0.0, net_sum, torch.zeros_like(net_sum)).sum(dim=1)
    negatives = torch.where(net_sum <= 0.0, -net_sum, torch.zeros_like(net_sum)).sum(dim=1)
    return {
        "trades": trades,
        "wins": wins,
        "fees_rs": fees,
        "net_points": points,
        "net_rs": net,
        "max_drawdown_rs": max_dd_rs,
        "profit_positive_rs": positives,
        "profit_negative_rs": negatives,
        "daily_trades": trade_count,
        "daily_net_points": points_sum,
        "daily_net_rs": net_sum,
        "daily_drawdown_rs": day_drawdown,
        "daily_peak_rs": day_peak,
        "daily_trough_rs": day_trough,
    }


def _d2h_summary(
    gpu_result: Mapping[str, torch.Tensor],
    data: GpuDataset,
) -> list[dict[str, Any]]:
    """Perform the single batched D2H summary readback after a GPU evaluation."""
    values = {
        key: value.detach().cpu().numpy()
        for key, value in gpu_result.items()
    }
    output: list[dict[str, Any]] = []
    for batch_index in range(len(values["trades"])):
        trades = int(values["trades"][batch_index])
        net_rs = float(values["net_rs"][batch_index])
        max_dd_rs = float(values["max_drawdown_rs"][batch_index])
        positive = float(values["profit_positive_rs"][batch_index])
        negative = float(values["profit_negative_rs"][batch_index])
        output.append({
            "trades": trades,
            "wins": int(values["wins"][batch_index]),
            "win_rate": round(float(values["wins"][batch_index]) / trades * 100.0, 2) if trades else 0.0,
            "fees_rs": round(float(values["fees_rs"][batch_index]), 2),
            "net_points": round(float(values["net_points"][batch_index]), 2),
            "net_rs": round(net_rs, 2),
            "max_drawdown_rs": round(max_dd_rs, 2),
            "max_drawdown_points": round(max_dd_rs / LOT_SIZE, 2),
            "profit_factor": round(positive / negative, 4) if negative else (float("inf") if positive else 0.0),
            "daily_trades": values["daily_trades"][batch_index].astype(int).tolist(),
            "daily_net_points": values["daily_net_points"][batch_index].astype(float).round(2).tolist(),
            "daily_net_rs": values["daily_net_rs"][batch_index].astype(float).round(2).tolist(),
            "daily_drawdown_rs": values["daily_drawdown_rs"][batch_index].astype(float).round(2).tolist(),
            "daily_peak_rs": values["daily_peak_rs"][batch_index].astype(float).round(2).tolist(),
            "daily_trough_rs": values["daily_trough_rs"][batch_index].astype(float).round(2).tolist(),
        })
    return output


def _score(result: Mapping[str, Any], drawdown_penalty: float = 0.20) -> float:
    return float(result["net_points"]) - drawdown_penalty * float(result["max_drawdown_points"])


def _fixed_params(stop_level: float) -> dict[str, Any]:
    return {
        "stop_level": float(stop_level),
        "target_level": PRIMARY_TARGET_LEVEL,
        "fallback_target_level": FALLBACK_TARGET_LEVEL,
        "option_point_threshold": OPTION_POINT_THRESHOLD,
    }


def suggest_exit_parameters(trial: optuna.Trial) -> dict[str, Any]:
    """CPU-side Optuna suggestions for the GPU-representable exit risk axis."""
    return _fixed_params(trial.suggest_categorical("stop_level", list(STOP_LEVELS)))


class GpuEvaluator:
    def __init__(
        self,
        data: GpuDataset,
        batch_size: int,
        brokerage_per_order: float = BROKERAGE_PER_ORDER,
        fixed_cost_per_trade: float | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.data = data
        self.events = pack_events(data)
        self.batch_size = int(batch_size)
        self.brokerage_per_order = float(brokerage_per_order)
        self.fixed_cost_per_trade = (
            None if fixed_cost_per_trade is None else float(fixed_cost_per_trade)
        )
        self.cuda_ms = 0.0
        self.wall_seconds = 0.0
        self.evaluations = 0
        self.last_cuda_ms = 0.0

    def evaluate(
        self,
        params: Sequence[Mapping[str, Any]],
        day_mask: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        if not params:
            return []
        actual_count = len(params)
        padded_params = list(params)
        while len(padded_params) < self.batch_size:
            padded_params.append(dict(params[0]))
        start_wall = time.perf_counter()
        if self.data.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = simulate_event_batch_matrix(
                self.data,
                self.events,
                padded_params,
                day_mask,
                brokerage_per_order=self.brokerage_per_order,
                fixed_cost_per_trade=self.fixed_cost_per_trade,
            )
            end_event.record()
            end_event.synchronize()
            self.last_cuda_ms = float(start_event.elapsed_time(end_event))
            self.cuda_ms += self.last_cuda_ms
        else:
            self.last_cuda_ms = 0.0
            result = simulate_event_batch_matrix(
                self.data,
                self.events,
                padded_params,
                day_mask,
                brokerage_per_order=self.brokerage_per_order,
                fixed_cost_per_trade=self.fixed_cost_per_trade,
            )
        self.wall_seconds += time.perf_counter() - start_wall
        self.evaluations += actual_count
        return _d2h_summary(result, self.data)[:actual_count]


def _mask_for_days(data: GpuDataset, allowed_days: Iterable[str]) -> torch.Tensor:
    allowed = set(allowed_days)
    return torch.tensor(
        [day in allowed for day in data.days],
        dtype=torch.bool,
        device=data.device,
    )


def _cpu_stats(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        trades,
        key=lambda trade: (
            int(trade.get("entry_min", 0)),
            int(trade.get("exit_min", 0)),
        ),
    )
    net_points = sum(float(trade.get("points", 0.0)) for trade in ordered)
    net_rs = sum(float(trade.get("rs_net", 0.0)) for trade in ordered)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for trade in ordered:
        equity += float(trade.get("rs_net", 0.0))
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "trades": len(ordered),
        "net_points": round(net_points, 2),
        "net_rs": round(net_rs, 2),
        "max_drawdown_points": round(max_dd / LOT_SIZE, 2),
    }


def validate_parity(
    adapter: historical.CsvHistoricalDataAdapter,
    data: GpuDataset,
    evaluator: GpuEvaluator,
    parity_days: Sequence[str],
) -> dict[str, Any]:
    """Compare exactly three dates against the current CPU Smart Fib process."""
    if len(parity_days) != 3:
        raise ValueError("parity validation requires exactly three dates")
    cpu_by_stop: dict[str, dict[str, Any]] = {
        str(stop): {"trades": 0, "net_points": 0.0, "net_rs": 0.0, "days": {}}
        for stop in STOP_LEVELS
    }
    for day in parity_days:
        cpu_output = smart_core.process_day(
            day,
            ("combined",),
            (PRIMARY_TARGET_LEVEL,),
            STOP_LEVELS,
            cache_loader=adapter.load_day_cache,
            min_span=SIGNAL_MIN_SPAN,
            touch_buffer=SIGNAL_TOUCH_BUFFER,
            setup_max_age=SIGNAL_SETUP_MAX_AGE,
            option_point_threshold=OPTION_POINT_THRESHOLD,
            fallback_target_level=FALLBACK_TARGET_LEVEL,
            debug=False,
        )
        for stop in STOP_LEVELS:
            key = f"smart-fib|combined|tp{PRIMARY_TARGET_LEVEL}|sl{stop}"
            stats = _cpu_stats(cpu_output.get(key, []))
            bucket = cpu_by_stop[str(stop)]
            bucket["trades"] += stats["trades"]
            bucket["net_points"] += stats["net_points"]
            bucket["net_rs"] += stats["net_rs"]
            bucket["days"][day] = stats

    mask = _mask_for_days(data, parity_days)
    gpu_results = evaluator.evaluate([_fixed_params(stop) for stop in STOP_LEVELS], mask)
    comparisons: list[dict[str, Any]] = []
    passed = True
    selected_indices = [data.days.index(day) for day in parity_days]
    for result, stop in zip(gpu_results, STOP_LEVELS):
        cpu = cpu_by_stop[str(stop)]
        gpu_trades = sum(result["daily_trades"][index] for index in selected_indices)
        gpu_points = sum(result["daily_net_points"][index] for index in selected_indices)
        trade_ok = abs(gpu_trades - int(cpu["trades"])) <= TRADES_TOLERANCE
        points_ok = abs(gpu_points - float(cpu["net_points"])) <= NET_POINTS_TOLERANCE
        item = {
            "stop_level": stop,
            "dates": list(parity_days),
            "cpu": cpu,
            "gpu": {
                "trades": gpu_trades,
                "net_points": round(gpu_points, 2),
                "net_rs": round(sum(result["daily_net_rs"][index] for index in selected_indices), 2),
            },
            "tolerances": {
                "trades": TRADES_TOLERANCE,
                "net_points": NET_POINTS_TOLERANCE,
            },
            "passed": trade_ok and points_ok,
        }
        comparisons.append(item)
        passed = passed and item["passed"]
    if not passed:
        raise ParityError(json.dumps({"comparisons": comparisons}, indent=2))
    return {"passed": True, "dates": list(parity_days), "comparisons": comparisons}


def _study_top_five(study: optuna.Study) -> list[dict[str, Any]]:
    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    completed.sort(
        key=lambda trial: float(trial.user_attrs.get("net_points", float("-inf"))),
        reverse=True,
    )
    output = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for trial in completed:
        identity = tuple(sorted(trial.params.items()))
        if identity in seen:
            continue
        seen.add(identity)
        output.append({
            "trial": trial.number,
            "objective": round(float(trial.value), 4),
            "params": dict(trial.params),
            "stats": {
                key: trial.user_attrs[key]
                for key in (
                    "trades", "wins", "win_rate", "net_points", "net_rs",
                    "fees_rs", "max_drawdown_points", "profit_factor",
                )
                if key in trial.user_attrs
            },
        })
        if len(output) == 5:
            break
    return output


def run_study(
    evaluator: GpuEvaluator,
    data: GpuDataset,
    n_trials: int,
    day_mask: torch.Tensor | None,
    *,
    seed: int,
) -> tuple[optuna.Study, dict[str, Any]]:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        constant_liar=True,
        multivariate=True,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for offset in range(0, n_trials, evaluator.batch_size):
        count = min(evaluator.batch_size, n_trials - offset)
        asked = [study.ask() for _ in range(count)]
        params = [suggest_exit_parameters(trial) for trial in asked]
        results = evaluator.evaluate(params, day_mask)
        for trial, result in zip(asked, results):
            score = _score(result)
            for key in (
                "trades", "wins", "win_rate", "net_points", "net_rs", "fees_rs",
                "max_drawdown_points", "profit_factor",
            ):
                trial.set_user_attr(key, result[key])
            study.tell(trial, score)
        completed = offset + count
        if offset == 0 or completed % (evaluator.batch_size * 5) == 0 or completed == n_trials:
            batches_done = (completed + evaluator.batch_size - 1) // evaluator.batch_size
            print(
                f"[CUDA] trials {completed}/{n_trials} | batch={count} | "
                f"last_batch_ms={evaluator.last_cuda_ms:.2f} | "
                f"avg_batch_ms={evaluator.cuda_ms / max(1, batches_done):.2f}",
                flush=True,
            )
    best = study.best_trial
    best_result = {
        key: best.user_attrs[key]
        for key in (
            "trades", "wins", "win_rate", "net_points", "net_rs", "fees_rs",
            "max_drawdown_points", "profit_factor",
        )
        if key in best.user_attrs
    }
    return study, {
        "best_trial": best.number,
        "best_params": dict(best.params),
        "best_stats": best_result,
        "top_five": _study_top_five(study),
    }


def _wfo_folds(end: str) -> list[dict[str, str]]:
    upper = str(end)[:10]
    folds: list[dict[str, str]] = []
    for year in range(2021, 2027):
        validation_start = f"{year}-01-01"
        validation_end = f"{year}-12-31"
        if validation_start > upper:
            continue
        folds.append({
            "name": str(year),
            "train_start": "2020-01-01",
            "train_end": f"{year - 1}-12-31",
            "validation_start": validation_start,
            "validation_end": min(validation_end, upper),
        })
    return folds


def run_wfo(
    evaluator: GpuEvaluator,
    data: GpuDataset,
    n_trials: int,
    start: str,
    end: str,
) -> dict[str, Any]:
    stitched_days: list[str] = []
    stitched_net_points: list[float] = []
    stitched_net_rs: list[float] = []
    fold_results: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(_wfo_folds(end)):
        train_days = [
            day for day in data.days
            if max(start, fold["train_start"]) <= day <= min(end, fold["train_end"])
        ]
        validation_days = [
            day for day in data.days
            if max(start, fold["validation_start"]) <= day <= min(end, fold["validation_end"])
        ]
        if not train_days or not validation_days:
            continue
        train_mask = _mask_for_days(data, train_days)
        study, train = run_study(
            evaluator,
            data,
            n_trials,
            train_mask,
            seed=42 + fold_index,
        )
        selected = _fixed_params(float(train["best_params"]["stop_level"]))
        validation_mask = _mask_for_days(data, validation_days)
        validation = evaluator.evaluate([selected], validation_mask)[0]
        fold_results.append({
            "fold": fold,
            "train_days": [train_days[0], train_days[-1]],
            "validation_days": [validation_days[0], validation_days[-1]],
            "selected_params": selected,
            "train": train,
            "validation": {
                key: validation[key]
                for key in (
                    "trades", "wins", "win_rate", "net_points", "net_rs", "fees_rs",
                    "max_drawdown_points", "profit_factor",
                )
            },
            "top_five_train": _study_top_five(study),
        })
        stitched_days.extend(validation_days)
        stitched_net_points.append(float(validation["net_points"]))
        stitched_net_rs.append(float(validation["net_rs"]))
    return {
        "folds": fold_results,
        "stitched_oos": {
            "days": stitched_days,
            "net_points": round(sum(stitched_net_points), 2),
            "net_rs": round(sum(stitched_net_rs), 2),
        },
    }


def _device_report(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device": "cpu",
            "gpu_used": False,
            "gpu_name": None,
            "vram_gb": None,
        }
    props = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "gpu_used": True,
        "gpu_name": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
    }


def _write_output(path: str, output: Mapping[str, Any]) -> None:
    if path == "-":
        print(json.dumps(output, indent=2, default=str))
        return
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output/cache: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=historical.DEFAULT_DATA_ROOT)
    parser.add_argument("--start", default=historical.DEFAULT_START)
    parser.add_argument("--end", default=historical.DEFAULT_END)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--prep-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="CPU workers for one-time event extraction; CUDA remains single-process",
    )
    parser.add_argument("--smoke", action="store_true", help="use exactly the first five available days")
    parser.add_argument("--wfo", action="store_true", help="run annual train-only 2021-2026 WFO folds")
    parser.add_argument("--allow-expensive", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "artifacts/f6_hybrid/smart_fib_optimus_gpu.json"))
    parser.add_argument(
        "--tensor-cache",
        default=None,
        help="optional .npz cache for the one-time CPU event extraction",
    )
    parser.add_argument("--allow-cpu", action="store_true", help="debug/parity only; never a silent fallback")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.prep_workers <= 0:
        raise SystemExit("--prep-workers must be positive")
    if args.trials < 0:
        raise SystemExit("--trials cannot be negative")
    if args.smoke and args.wfo:
        raise SystemExit("--smoke and --wfo are mutually exclusive")
    if args.smoke and args.trials > 0:
        raise SystemExit("--smoke and --trials are mutually exclusive")
    if args.allow_cpu and (args.trials > 0 or args.wfo):
        raise SystemExit("--allow-cpu is restricted to bounded smoke/parity debugging")
    if args.wfo and args.trials <= 0:
        raise SystemExit("--wfo requires --trials N --allow-expensive")
    if not args.smoke and args.trials <= 0:
        raise SystemExit("refusing an unbounded run; use --smoke or --trials N --allow-expensive")
    if (args.trials > 0 or args.wfo) and not args.allow_expensive:
        raise SystemExit("--trials/--wfo require --allow-expensive")
    if args.output != "-" and Path(args.output).exists():
        raise SystemExit(f"refusing to overwrite existing output/cache: {args.output}")

    device = require_device(args.allow_cpu)
    print(f"[GPU INIT] device={device}", flush=True)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(
            f"[GPU INIT] name={props.name} vram_gb={props.total_memory / (1024 ** 3):.2f} "
            f"cuda={torch.version.cuda}",
            flush=True,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    adapter = historical.CsvHistoricalDataAdapter(
        args.data_root,
        start=args.start,
        end=args.end,
    )
    available = adapter.available_days(args.start, args.end)
    if not available:
        raise SystemExit(f"no overlapping index/options days in {args.start}..{args.end}")
    days = available[:5] if args.smoke else available
    if len(days) < 3:
        raise SystemExit("at least three available days are required for parity validation")
    if args.smoke and len(days) != 5:
        raise SystemExit("--smoke requires exactly five available days")

    load_start = time.perf_counter()
    tensor_cache = (
        Path(args.tensor_cache)
        if args.tensor_cache
        else ROOT / "artifacts" / "f6_hybrid" / (
            f"smart_fib_gpu_tensor_cache_{days[0]}_{days[-1]}.npz"
        )
    )
    cpu_data = load_cpu_dataset(tensor_cache, args.data_root, days)
    if cpu_data is None:
        print(
            f"[CPU PREP] cache miss; extracting {len(days)} days to {tensor_cache}",
            flush=True,
        )
        cpu_data = collect_cpu_dataset(adapter, days, workers=args.prep_workers)
        save_cpu_dataset(cpu_data, tensor_cache, args.data_root)
        print(f"[CPU PREP] saved tensor cache: {tensor_cache}", flush=True)
    else:
        print(f"[CPU PREP] loaded tensor cache: {tensor_cache}", flush=True)
    gpu_data = to_gpu_dataset(cpu_data, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start
    evaluator = GpuEvaluator(gpu_data, args.batch_size)
    print(
        f"[VRAM] resident tensors N={gpu_data.n_days} T={T_BARS} C={gpu_data.contract_slots} "
        f"batch={args.batch_size}",
        flush=True,
    )
    print("[PARITY] running exactly three CPU/GPU dates", flush=True)
    parity = validate_parity(adapter, gpu_data, evaluator, days[:3])
    print("[PARITY] PASS", flush=True)

    result: dict[str, Any] = {
        "engine": "smart_fib_optimus_gpu",
        "mode": "smoke" if args.smoke else "walk_forward" if args.wfo else "non_walk_forward",
        "data_root": str(Path(args.data_root)),
        "date_range": {"start": args.start, "end": args.end},
        "days": days,
        "tensor_contract": {
            "shape": "(B,N,T)",
            "T": T_BARS,
            "C": gpu_data.contract_slots,
            "bar_start_minute": BAR_START_MINUTE,
            "contract_slot_limit": CONTRACT_SLOT_LIMIT,
        },
        "fixed_signal_contract": {
            "timeframe": "combined",
            "min_span": SIGNAL_MIN_SPAN,
            "touch_buffer": SIGNAL_TOUCH_BUFFER,
            "setup_max_age": SIGNAL_SETUP_MAX_AGE,
            "fib_entry_level": ENTRY_LEVEL,
            "fib_zone": [0.618, 1.0],
            "entry_stream": "CPU Smart Fib extract_day_events, once per day",
            "signal_optimization": "not tensorized; entry events are fixed",
        },
        "fixed_execution_contract": {
            "dynamic_contract_candidates": {
                "CE": ["ATM", "ATM-50", "ATM-100"],
                "PE": ["ATM", "ATM+50", "ATM+100"],
            },
            "primary_target_level": PRIMARY_TARGET_LEVEL,
            "option_point_threshold": OPTION_POINT_THRESHOLD,
            "fallback_target_level": FALLBACK_TARGET_LEVEL,
            "stop_levels": list(STOP_LEVELS),
            "one_global_position_per_day": True,
            "sequential_reentry": "unlimited after each exit until session end",
            "actual_option_ohlc": True,
            "slippage_points_per_leg": SLIPPAGE_PTS,
            "lot_size": LOT_SIZE,
            "fee_model": "backtest_walkforward_fees.trade_cost, brokerage included",
        },
        "preprocessing": {
            "index_transfer": "one-time pinned host to device copy",
            "option_transfer": "one-time pinned host to device copy",
            "raw_event_counts": dict(zip(days, cpu_data.raw_event_counts)),
            "selected_event_counts": dict(zip(days, cpu_data.selected_event_counts)),
            "actual_symbols_sample": sorted({
                symbol
                for symbols in cpu_data.event_symbols
                for symbol in symbols
                if symbol
            })[:50],
        },
        "parity": parity,
        "hardware": _device_report(device),
        "timing": {
            "preprocess_and_initial_transfer_s": round(load_seconds, 3),
            "cuda_event_eval_ms": None,
            "wall_eval_s": None,
            "batch_size": args.batch_size,
            "prep_workers": args.prep_workers,
            "tensor_cache": str(tensor_cache),
            "evaluations": 0,
            "memory_peak_allocated_mb": None,
            "memory_peak_reserved_mb": None,
        },
    }

    if args.smoke:
        smoke_mask = _mask_for_days(gpu_data, days)
        smoke_results = evaluator.evaluate(
            [_fixed_params(stop) for stop in STOP_LEVELS],
            smoke_mask,
        )
        result["smoke_results"] = [
            {
                "params": _fixed_params(stop),
                "stats": {
                    key: value
                    for key, value in smoke_result.items()
                    if not key.startswith("daily_")
                },
            }
            for stop, smoke_result in zip(STOP_LEVELS, smoke_results)
        ]
    elif args.wfo:
        print(f"[CUDA] starting WFO: {args.trials} trials/fold", flush=True)
        result["wfo"] = run_wfo(evaluator, gpu_data, args.trials, args.start, args.end)
    else:
        print(f"[CUDA] starting non-WF study: {args.trials} trials", flush=True)
        full_mask = _mask_for_days(gpu_data, days)
        study, study_result = run_study(
            evaluator,
            gpu_data,
            args.trials,
            full_mask,
            seed=42,
        )
        result["study"] = {
            "trials": args.trials,
            **study_result,
        }

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result["timing"].update({
            "cuda_event_eval_ms": round(evaluator.cuda_ms, 3),
            "wall_eval_s": round(evaluator.wall_seconds, 3),
            "evaluations": evaluator.evaluations,
            "memory_peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
            "memory_peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2),
        })
    else:
        result["timing"].update({
            "cuda_event_eval_ms": None,
            "wall_eval_s": round(evaluator.wall_seconds, 3),
            "evaluations": evaluator.evaluations,
        })

    _write_output(args.output, result)
    print(json.dumps({
        "engine": result["engine"],
        "gpu_used": result["hardware"]["gpu_used"],
        "gpu_name": result["hardware"]["gpu_name"],
        "days": len(days),
        "parity": result["parity"]["passed"],
        "output": args.output,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
