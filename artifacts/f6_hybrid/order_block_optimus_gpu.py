"""Order-block (Fibb Block Strategy) Optimus backtest — GPU-first.

Signal stream: the order-block engine (``order_block_engine.py``) replaces the
Smart Fib fib-zone entry condition with the video's order-block condition —
nearest completed consolidation block at the S1-turn setup, two flavors run as
separate variants:

  * ``flip`` — breakout + retest + flip back through the block edge (80% rule)
  * ``turn`` — price comes straight down/up into the block and turns inside it

All four timeframes (1m/2m/3m/5m, bias filter = 5x TF) are merged into ONE
event stream per day (union, dedup by (minute, side, symbol), lowest TF wins)
and evaluated on CUDA via the Optimus matrix engine. The CPU simulator is used
ONLY as a three-date parity oracle, never for the production grid.

Exit grid is the video's geometry, mapped onto the standard setup-swing fib
levels (target 0 = swing top, target -0.055 = beyond it; stops 1.079 / 1.155 /
1.25 = 0.079 / 0.155 / 0.25 of span beyond the swing low). The option-point
fallback layer is INERT here by construction (both the CPU and GPU fallback
branches gate on target == 0.29), so it is excluded from the grid axes.

Runbook compliance (OPTIMIZED_GPU_BACKTEST.md): no per-trade scalar
readbacks, no CUDA graphs, resident VRAM tensors, pinned non-blocking copies,
cudaMallocAsync, parity gate before every grid, one padded matrix pass per
variant (--batch-size 16 covers the default 12 exit configs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.smart_fib_cuda_bootstrap import configure_cuda_toolkit

configure_cuda_toolkit()

import numpy as np
import torch

from artifacts.f6_hybrid import marni_fib_core_combo_cache as smart_core
from artifacts.f6_hybrid import order_block_engine as ob_core
from artifacts.f6_hybrid import smart_fib_optimus_gpu as optimus
from artifacts.f6_hybrid import smart_fib_optimus_grid_gpu as grid

LOT_SIZE = int(smart_core.LOT_SIZE)
FIXED_COST_PER_TRADE = 40.0
OB_CACHE_VERSION = 2

# Video exit levels (SL 1.079 / 1.155 / 1.25, TP zero / -0.055).
VIDEO_TARGETS = (0.0, -0.055)
VIDEO_FALLBACK_TARGETS = (0.0,)
VIDEO_THRESHOLDS = (5.0,)
VIDEO_STOPS = (1.079, 1.155, 1.25)

FLAVORS = ("flip", "turn")


def _number_token(value: float | int) -> str:
    text = format(float(value), "g")
    return text.replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class ObVariant:
    """One order-block signal stream (setup params + block params + flavor)."""

    flavor: str
    s1_k_period: int = smart_core.S1_K_PERIOD
    s1_d_period: int = smart_core.S1_D_PERIOD
    min_span: float = smart_core.MIN_SPAN
    setup_max_age: int = smart_core.SETUP_MAX_AGE
    touch_buffer: float = 0.0
    block_window: int = ob_core.BLOCK_WINDOW
    squeeze_ratio: float = ob_core.SQUEEZE_RATIO
    lookback: int = ob_core.BLOCK_LOOKBACK
    min_block_span: float = ob_core.MIN_BLOCK_SPAN
    breakout_window: int = ob_core.BREAKOUT_WINDOW
    max_block_age: int = ob_core.MAX_BLOCK_AGE
    straight_lookback: int = ob_core.STRAIGHT_LOOKBACK

    @property
    def variant_id(self) -> str:
        return (
            f"ob_{self.flavor}"
            f"_s1k{self.s1_k_period}d{self.s1_d_period}"
            f"_span{_number_token(self.min_span)}"
            f"_age{self.setup_max_age}"
            f"_buf{_number_token(self.touch_buffer)}"
            f"_bw{self.block_window}"
            f"_sq{_number_token(self.squeeze_ratio)}"
            f"_lb{self.lookback}"
            f"_msp{_number_token(self.min_block_span)}"
            f"_bkw{self.breakout_window}"
            f"_ageb{self.max_block_age}"
            f"_stl{self.straight_lookback}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "flavor": self.flavor,
            "s1_k_period": self.s1_k_period,
            "s1_d_period": self.s1_d_period,
            "min_span": self.min_span,
            "setup_max_age": self.setup_max_age,
            "touch_buffer": self.touch_buffer,
            "block_window": self.block_window,
            "squeeze_ratio": self.squeeze_ratio,
            "lookback": self.lookback,
            "min_block_span": self.min_block_span,
            "breakout_window": self.breakout_window,
            "max_block_age": self.max_block_age,
            "straight_lookback": self.straight_lookback,
            "variant_id": self.variant_id,
        }


DEFAULT_VARIANTS = tuple(
    ObVariant(flavor=flavor) for flavor in FLAVORS
)


@dataclass
class ObDayRaw(grid.VariantDayRaw):
    """VariantDayRaw plus per-day order-block attribution."""

    attribution: dict[str, Any] = field(default_factory=dict)


def _extract_ob_day(
    task: tuple[str, str, str, str, tuple[int, ...], tuple[Any, ...]],
) -> ObDayRaw:
    """Worker: extract every TF for one day, merge into the union stream."""
    data_root, start, end, day, timeframes, values = task
    variant = ObVariant(*values)
    adapter = grid._GRID_WORKER_ADAPTER
    if adapter is None:
        adapter = grid.PolarsHistoricalDataAdapter(
            data_root,
            start=start,
            end=end,
            cache_days=8,
        )
    merged: list[tuple[int, int, dict[str, Any]]] = []
    payload = None
    raw_total = 0
    per_tf_selected: dict[int, int] = {}
    block_stats: dict[str, Any] = {}
    for tf in timeframes:
        prepared = ob_core.extract_order_block_events(
            day,
            cache_loader=adapter.load_day_cache,
            bar_minutes=tf,
            filter_period=5 * tf,
            min_span=variant.min_span,
            touch_buffer=variant.touch_buffer,
            setup_max_age=variant.setup_max_age,
            s1_k_period=variant.s1_k_period,
            s1_d_period=variant.s1_d_period,
            flavor=variant.flavor,
            block_window=variant.block_window,
            squeeze_ratio=variant.squeeze_ratio,
            lookback=variant.lookback,
            min_block_span=variant.min_block_span,
            breakout_window=variant.breakout_window,
            max_block_age=variant.max_block_age,
            straight_lookback=variant.straight_lookback,
            debug=False,
        )
        if not prepared:
            raise RuntimeError(f"no order-block cache payload for {day} tf={tf}")
        selected, raw_count = optimus._select_day_events(prepared)
        raw_total += raw_count
        per_tf_selected[tf] = len(selected)
        for signal in selected:
            merged.append((int(signal["minute"]), tf, signal))
        if tf == 1:
            payload = prepared
        blocks = prepared.get("ob_blocks") or []
        if blocks:
            breakout_blocks = [b for b in blocks if b["breakout_minute"] is not None]
            block_stats[f"tf{tf}"] = {
                "blocks": len(blocks),
                "breakouts": len(breakout_blocks),
                "mean_span": round(
                    sum(float(b["high"]) - float(b["low"]) for b in blocks)
                    / max(1, len(blocks)),
                    2,
                ),
            }
    merged.sort(key=lambda item: (item[0], item[1]))
    events: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for minute, tf, signal in merged:
        key = (minute, signal["side"], signal["symbol"])
        if key in seen:
            continue
        seen.add(key)
        events.append({**signal, "timeframe": "combined"})
    attribution = {
        "selected": len(events),
        "raw_total": raw_total,
        "by_source_tf": per_tf_selected,
        "block_stats": block_stats,
        "blocks_total": sum(
            item.get("blocks", 0) for item in block_stats.values()
        ),
    }
    return ObDayRaw(
        variant=variant,
        day=day,
        prepared=dict(payload),
        selected_events=events,
        raw_event_count=raw_total,
        contract_keys=list(payload["records"]),
        attribution=attribution,
    )


def _ob_cache_path(
    cache_dir: Path, days: Sequence[str], variant: ObVariant, timeframes: tuple[int, ...]
) -> Path:
    token = "x".join(str(tf) for tf in timeframes)
    return cache_dir / (
        f"order_block_tensor_cache_{days[0]}_{days[-1]}_{variant.variant_id}_u{token}.npz"
    )


def _save_ob_cache(
    dataset: grid.VariantCpuDataset,
    path: Path,
    data_root: str | Path,
    variant: ObVariant,
    timeframes: tuple[int, ...],
    events_by_day: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    attribution_by_day: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite order-block tensor cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache_version": OB_CACHE_VERSION,
        "cache_key": variant.variant_id,
        "union_timeframes": list(timeframes),
        "variant": variant.as_dict(),
        "data_root": str(Path(data_root).resolve()),
        "days": dataset.days,
        "contract_slots": dataset.contract_slots,
        "raw_event_counts": dataset.raw_event_counts,
        "selected_event_counts": dataset.selected_event_counts,
        "event_symbols": dataset.event_symbols,
        "contract_symbols": dataset.contract_symbols,
        "events_by_day": json.dumps(
            {day: list(events) for day, events in (events_by_day or {}).items()}
        ),
        "attribution_by_day": json.dumps(
            dict(attribution_by_day or {})
        ),
    }
    arrays = {name: getattr(dataset, name) for name in grid._CACHE_ARRAY_NAMES}
    np.savez_compressed(path, **arrays, metadata=np.array(json.dumps(metadata)))


@dataclass
class ObCacheLoad:
    dataset: grid.VariantCpuDataset
    events_by_day: dict[str, list[dict[str, Any]]]
    attribution_by_day: dict[str, dict[str, Any]]


def _load_ob_cache(path: Path, data_root: str | Path, days: Sequence[str],
                   variant: ObVariant,
                   timeframes: tuple[int, ...]) -> ObCacheLoad | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("cache_version") != OB_CACHE_VERSION:
                return None
            if metadata.get("cache_key") != variant.variant_id:
                return None
            if metadata.get("union_timeframes") != list(timeframes):
                return None
            if metadata.get("data_root") != str(Path(data_root).resolve()):
                return None
            if metadata.get("days") != list(days):
                return None
            arrays = {name: archive[name].copy() for name in grid._CACHE_ARRAY_NAMES}
            events_by_day = json.loads(metadata.get("events_by_day", "{}"))
            attribution_by_day = json.loads(metadata.get("attribution_by_day", "{}"))
        return ObCacheLoad(
            dataset=grid.VariantCpuDataset(
                days=list(days),
                **arrays,
                event_symbols=metadata["event_symbols"],
                contract_symbols=metadata["contract_symbols"],
                raw_event_counts=[int(value) for value in metadata["raw_event_counts"]],
                selected_event_counts=[int(value) for value in metadata["selected_event_counts"]],
            ),
            events_by_day=events_by_day,
            attribution_by_day=attribution_by_day,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


@dataclass
class ObBundle:
    variant: ObVariant
    dataset: grid.VariantCpuDataset
    parity_payloads: dict[str, dict[str, Any]]
    union_events_by_day: dict[str, list[dict[str, Any]]]
    attribution_by_day: dict[str, dict[str, Any]]
    cache_source: str
    cache_path: str | None
    prep_seconds: float


def collect_ob_cpu_dataset(
    adapter: grid.PolarsHistoricalDataAdapter,
    days: Sequence[str],
    variant: ObVariant,
    timeframes: tuple[int, ...],
    *,
    workers: int,
    parity_days: Sequence[str],
    tensor_cache_dir: Path | None,
) -> ObBundle:
    """Extract the union stream once per day, freeze its CPU oracle tensors."""
    cached: grid.VariantCpuDataset | None = None
    cache_path: str | None = None
    if tensor_cache_dir is not None:
        path = _ob_cache_path(tensor_cache_dir, days, variant, timeframes)
        cached = _load_ob_cache(path, adapter.data_root, days, variant, timeframes)
        if cached is not None:
            cache_path = str(path)
    if cached is not None:
        print(
            f"[OB CPU PREP] {variant.variant_id} cache N={cached.dataset.n_days} "
            f"T={optimus.T_BARS} C={cached.dataset.contract_slots}",
            flush=True,
        )
        return ObBundle(
            variant=variant,
            dataset=cached.dataset,
            parity_payloads={},
            union_events_by_day=cached.events_by_day,
            attribution_by_day=cached.attribution_by_day,
            cache_source="ob_variant_cache",
            cache_path=str(path),
            prep_seconds=0.0,
        )

    started = time.perf_counter()
    task_values = (
        variant.flavor,
        variant.s1_k_period,
        variant.s1_d_period,
        variant.min_span,
        variant.setup_max_age,
        variant.touch_buffer,
        variant.block_window,
        variant.squeeze_ratio,
        variant.lookback,
        variant.min_block_span,
        variant.breakout_window,
        variant.max_block_age,
        variant.straight_lookback,
    )
    tasks = [
        (str(adapter.data_root), adapter.start, adapter.end, day, timeframes, task_values)
        for day in days
    ]
    raw_days: list[ObDayRaw] = []
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=grid._init_grid_worker,
            initargs=(str(adapter.data_root), adapter.start, adapter.end),
        ) as pool:
            iterator = pool.map(_extract_ob_day, tasks, chunksize=1)
            for count, raw in enumerate(iterator, start=1):
                raw_days.append(raw)
                if count == 1 or count % 25 == 0 or count == len(tasks):
                    print(
                        f"[OB CPU PREP] {variant.variant_id} {count}/{len(tasks)} "
                        f"last={raw.day}",
                        flush=True,
                    )
    else:
        for count, task in enumerate(tasks, start=1):
            raw = _extract_ob_day(task)
            raw_days.append(raw)
            if count == 1 or count % 25 == 0 or count == len(tasks):
                print(
                    f"[OB CPU PREP] {variant.variant_id} {count}/{len(tasks)} "
                    f"last={raw.day}",
                    flush=True,
                )

    dataset = grid._build_cpu_dataset(days, raw_days)
    parity_payloads = {raw.day: raw.prepared for raw in raw_days if raw.day in set(parity_days)}
    union_events_by_day = {raw.day: raw.selected_events for raw in raw_days if raw.day in set(parity_days)}
    attribution_by_day = {raw.day: raw.attribution for raw in raw_days}
    source = "cpu_extraction"
    if tensor_cache_dir is not None:
        path = _ob_cache_path(tensor_cache_dir, days, variant, timeframes)
        _save_ob_cache(
            dataset, path, adapter.data_root, variant, timeframes,
            events_by_day=union_events_by_day, attribution_by_day=attribution_by_day,
        )
        cache_path = str(path)
        source = "cpu_extraction_saved_ob_cache"
    elapsed = time.perf_counter() - started
    total_events = sum(dataset.selected_event_counts)
    print(
        f"[OB CPU PREP] {variant.variant_id} complete {elapsed:.3f}s "
        f"N={dataset.n_days} T={optimus.T_BARS} C={dataset.contract_slots} "
        f"events={total_events}",
        flush=True,
    )
    return ObBundle(
        variant=variant,
        dataset=dataset,
        parity_payloads=parity_payloads,
        union_events_by_day=union_events_by_day,
        attribution_by_day=attribution_by_day,
        cache_source=source,
        cache_path=cache_path,
        prep_seconds=elapsed,
    )


def _video_grid_params(
    targets: Sequence[float],
    fallback_targets: Sequence[float],
    thresholds: Sequence[float],
    stops: Sequence[float],
) -> list[dict[str, Any]]:
    """Plain cross-product exit grid (no axis whitelist).

    The fallback layer is inert by construction (CPU and GPU fallback branches
    both gate on target == 0.29), so the usual fallback<=target guard is not
    needed; the video's negative target (-0.055) is kept verbatim.
    """
    params = []
    for target in targets:
        for fallback in fallback_targets:
            for threshold in thresholds:
                for stop in stops:
                    params.append({
                        "stop_level": float(stop),
                        "target_level": float(target),
                        "fallback_target_level": float(fallback),
                        "option_point_threshold": float(threshold),
                    })
    if not params:
        raise ValueError("the exit grid cannot be empty")
    return params


def _ob_cpu_trades(
    events: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Any],
    target: float,
    fallback_target: float,
    threshold: float,
    stop: float,
    fixed_cost_per_trade: float | None,
) -> list[dict[str, Any]]:
    """CPU oracle: replay the frozen union events with the canonical simulator."""
    return smart_core.simulate(
        [{**event, "timeframe": "combined"} for event in events],
        prepared["bars"],
        prepared["index_bars"],
        prepared["spot"],
        "combined",
        target,
        stop,
        concurrent=False,
        option_point_threshold=threshold,
        fallback_target_level=fallback_target,
        fixed_cost_per_trade=fixed_cost_per_trade,
    )


def validate_ob_parity(
    bundle: ObBundle,
    gpu_data: grid.VariantGpuDataset,
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
    params = _video_grid_params(targets, fallback_targets, thresholds, stops)
    cpu_by_combo: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    for param in params:
        daily = {}
        for day in parity_days:
            daily[day] = grid._cpu_stats(
                _ob_cpu_trades(
                    bundle.union_events_by_day[day],
                    bundle.parity_payloads[day],
                    float(param["target_level"]),
                    float(param["fallback_target_level"]),
                    float(param["option_point_threshold"]),
                    float(param["stop_level"]),
                    evaluator.fixed_cost_per_trade,
                )
            )
        cpu_by_combo[
            (
                float(param["target_level"]),
                float(param["fallback_target_level"]),
                float(param["option_point_threshold"]),
                float(param["stop_level"]),
            )
        ] = {
            "trades": sum(item["trades"] for item in daily.values()),
            "wins": sum(item["wins"] for item in daily.values()),
            "net_points": round(sum(item["net_points"] for item in daily.values()), 2),
            "net_rs": round(sum(item["net_rs"] for item in daily.values()), 2),
            "fees_rs": round(sum(item["fees_rs"] for item in daily.values()), 2),
            "daily": daily,
        }

    mask = grid._mask_for_days(gpu_data.engine, parity_days)
    started = time.perf_counter()
    gpu_results = grid._evaluate_gpu(evaluator, params, mask)
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
            "net_points": round(sum(item["net_points"] for item in gpu_daily.values()), 2),
            "net_rs": round(sum(item["net_rs"] for item in gpu_daily.values()), 2),
            "fees_rs": round(float(gpu_result["fees_rs"]), 2),
            "daily": gpu_daily,
        }
        daily_failures = []
        for day in parity_days:
            cpu_day = cpu["daily"][day]
            gpu_day = gpu_daily[day]
            if gpu_day["trades"] != cpu_day["trades"]:
                daily_failures.append(f"{day}: trades {gpu_day['trades']} != {cpu_day['trades']}")
            if abs(gpu_day["net_points"] - cpu_day["net_points"]) > grid.NET_POINTS_TOLERANCE:
                daily_failures.append(
                    f"{day}: points {gpu_day['net_points']} != {cpu_day['net_points']}"
                )
            if abs(gpu_day["max_drawdown_points"] - cpu_day["max_drawdown_points"]) > grid.NET_POINTS_TOLERANCE:
                daily_failures.append(
                    f"{day}: dd {gpu_day['max_drawdown_points']} != {cpu_day['max_drawdown_points']}"
                )
        aggregate_failures = []
        if abs(gpu["trades"] - cpu["trades"]) > grid.TRADES_TOLERANCE:
            aggregate_failures.append("aggregate trades")
        if abs(gpu["wins"] - cpu["wins"]) > grid.TRADES_TOLERANCE:
            aggregate_failures.append("aggregate wins")
        if abs(gpu["net_points"] - cpu["net_points"]) > grid.NET_POINTS_TOLERANCE:
            aggregate_failures.append("aggregate net points")
        if abs(gpu["net_rs"] - cpu["net_rs"]) > grid.NET_RS_TOLERANCE:
            aggregate_failures.append("aggregate net Rs")
        if abs(gpu["fees_rs"] - cpu["fees_rs"]) > grid.NET_RS_TOLERANCE:
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
                "trades": grid.TRADES_TOLERANCE,
                "net_points": grid.NET_POINTS_TOLERANCE,
                "net_rs": grid.NET_RS_TOLERANCE,
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
        raise grid.GridParityError(json.dumps(report, indent=2, default=str))
    return report


def evaluate_video_grid(
    variant: ObVariant,
    gpu_data: grid.VariantGpuDataset,
    evaluator: optimus.GpuEvaluator,
    days: Sequence[str],
    targets: Sequence[float],
    fallback_targets: Sequence[float],
    thresholds: Sequence[float],
    stops: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = _video_grid_params(targets, fallback_targets, thresholds, stops)
    mask = grid._mask_for_days(gpu_data.engine, days)
    before_evaluations = evaluator.evaluations
    before_cuda_ms = evaluator.cuda_ms
    before_wall = evaluator.wall_seconds
    started = time.perf_counter()
    results = grid._evaluate_gpu(evaluator, params, mask)
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
        record["composite_score"] = grid._score(record)
        configs.append(record)
    return configs, {
        "variant_id": variant.variant_id,
        "grid_evaluations": evaluator.evaluations - before_evaluations,
        "cuda_ms": round(evaluator.cuda_ms - before_cuda_ms, 3),
        "wall_seconds": round(max(wall, evaluator.wall_seconds - before_wall), 6),
        "batch_size": evaluator.batch_size,
        "event_count": int(evaluator.events.time_index.numel()),
    }


def _event_attribution(bundle: ObBundle) -> dict[str, Any]:
    """Aggregate union event + block attribution (fresh extraction only)."""
    tfs: dict[int, int] = {}
    blocks_total = 0
    breakouts_total = 0
    span_sum = 0.0
    span_count = 0
    days = 0
    for day, attr in bundle.attribution_by_day.items():
        days += 1
        for tf, count in attr["by_source_tf"].items():
            tfs[int(tf)] = tfs.get(int(tf), 0) + int(count)
        blocks_total += int(attr.get("blocks_total", 0))
        for tf_key, stats in attr.get("block_stats", {}).items():
            breakouts_total += int(stats.get("breakouts", 0))
            span_sum += float(stats.get("mean_span", 0.0)) * int(stats.get("blocks", 0))
            span_count += int(stats.get("blocks", 0))
    return {
        "days": days,
        "events_by_source_tf": tfs,
        "blocks_total": blocks_total,
        "block_breakouts_total": breakouts_total,
        "mean_block_span": round(span_sum / max(1, span_count), 2),
    }


def _dataset_geometry_by_day(
    dataset: grid.VariantCpuDataset,
    target: float,
    stop: float,
) -> tuple[dict[int, tuple[list[float], list[float], list[float]]], int]:
    """Per-day SL/TP distances in points, read straight from the frozen
    dataset tensors (exact sim reachability: contract_valid at entry slot).

    CE (low_to_high) geometry:
        sl_price = fib_high - stop*span, tp_price = fib_high - target*span
    PE (high_to_low) is mirrored below the swing:
        sl_price = fib_low + stop*span,  tp_price = fib_low + target*span
    """
    per_day: dict[int, tuple[list[float], list[float], list[float]]] = {}
    tradable = 0
    for day_index in range(dataset.n_days):
        sl_distances: list[float] = []
        tp_distances: list[float] = []
        spans: list[float] = []
        positions = np.nonzero(dataset.event_mask[day_index])[0]
        for position in positions:
            contract = int(dataset.event_contract[day_index, position])
            if not bool(dataset.contract_valid_ntc[day_index, position, contract]):
                continue
            entry = float(dataset.event_entry[day_index, position])
            high = float(dataset.event_fib_high[day_index, position])
            low = float(dataset.event_fib_low[day_index, position])
            span = high - low
            if bool(dataset.event_high_to_low[day_index, position]):
                sl_price = low + stop * span
                tp_price = low + target * span
            else:
                sl_price = high - stop * span
                tp_price = high - target * span
            tradable += 1
            sl_distances.append(abs(entry - sl_price))
            tp_distances.append(abs(tp_price - entry))
            spans.append(span)
        per_day[day_index] = (sl_distances, tp_distances, spans)
    return per_day, tradable


def _stats(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0,
                "min": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "p10": round(float(np.percentile(arr, 10)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
    }


def _event_geometry(
    bundle: ObBundle,
    target: float,
    stop: float,
) -> dict[str, Any]:
    per_day, tradable = _dataset_geometry_by_day(bundle.dataset, target, stop)
    sl_distances: list[float] = []
    tp_distances: list[float] = []
    spans: list[float] = []
    for sl_day, tp_day, span_day in per_day.values():
        sl_distances.extend(sl_day)
        tp_distances.extend(tp_day)
        spans.extend(span_day)
    return {
        "target_level": float(target),
        "stop_level": float(stop),
        "tradable_events": tradable,
        "span_points": _stats(spans),
        "sl_points": _stats(sl_distances),
        "tp_points": _stats(tp_distances),
    }


def _config_geometry_report(
    bundle: ObBundle,
    configs: Sequence[Mapping[str, Any]],
    days: Sequence[str],
) -> list[dict[str, Any]]:
    """Per-config geometry + avg trades/day for one variant's grid."""
    report = []
    for config in configs:
        geometry = _event_geometry(
            bundle,
            float(config["target_level"]),
            float(config["stop_level"]),
        )
        report.append({
            "config_id": config["config_id"],
            "target_level": config["target_level"],
            "stop_level": config["stop_level"],
            "trades": int(config["trades"]),
            "avg_trades_per_day": round(float(config["trades"]) / max(1, len(days)), 3),
            **geometry,
        })
    return report


WF_FOLDS = (
    ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2020-01-01", "2024-12-31", "2025-01-01", "2026-05-05"),
)

WF_DD_PENALTY = 0.5


def run_walk_forward(
    variant: ObVariant,
    bundle: ObBundle,
    gpu_data: grid.VariantGpuDataset,
    evaluator: optimus.GpuEvaluator,
    days: Sequence[str],
    targets: Sequence[float],
    fallback_targets: Sequence[float],
    thresholds: Sequence[float],
    stops: Sequence[float],
) -> dict[str, Any]:
    """Expanding walk-forward over the video exit grid.

    Per fold: in-sample grid search over the train window (best by
    net - DD_PENALTY*DD), then the chosen config is replayed out-of-sample.
    The video's fixed levels (target 0 / stop 1.155) are ALSO replayed OOS
    every fold (no selection) for an unbiased read. OOS trades accumulate
    across folds; avg SL/TP come from the event geometry of the OOS days
    using the per-fold levels.
    """
    engine = gpu_data.engine
    params = _video_grid_params(targets, fallback_targets, thresholds, stops)
    video_param = next(
        param for param in params
        if abs(float(param["target_level"])) < 1e-9
        and abs(float(param["stop_level"]) - 1.155) < 1e-9
    )
    day_index = {day: index for index, day in enumerate(engine.days)}
    oos_video_days: list[str] = []
    oos_chosen_days: list[str] = []
    oos_video_sums = {"trades": 0, "wins": 0, "net_points": 0.0, "net_rs": 0.0,
                      "fees_rs": 0.0, "dd_points": 0.0}
    oos_chosen_sums = dict(oos_video_sums)
    oos_video_dd_peak = 0.0
    oos_chosen_dd_peak = 0.0
    geometry_video: dict[str, list[float]] = {"sl": [], "tp": [], "span": []}
    geometry_chosen: dict[str, list[float]] = {"sl": [], "tp": [], "span": []}
    folds = []
    before_evaluations = evaluator.evaluations
    before_cuda_ms = evaluator.cuda_ms
    started = time.perf_counter()
    for fold_index, (train_start, train_end, test_start, test_end) in enumerate(WF_FOLDS, start=1):
        train_days = [day for day in days if train_start <= day <= train_end]
        test_days = [day for day in days if test_start <= day <= test_end]
        if not train_days or not test_days:
            continue
        train_mask = grid._mask_for_days(engine, train_days)
        test_mask = grid._mask_for_days(engine, test_days)
        train_results = grid._evaluate_gpu(evaluator, params, train_mask)
        train_records = []
        for param, stats in zip(params, train_results):
            record = {
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
            }
            record["composite_score"] = record["net_points"] - WF_DD_PENALTY * record["max_drawdown_points"]
            train_records.append(record)
        best_train = max(
            train_records,
            key=lambda record: (record["composite_score"], record["net_points"]),
        )
        chosen_param = next(
            param for param in params
            if abs(float(param["target_level"]) - float(best_train["target_level"])) < 1e-9
            and abs(float(param["stop_level"]) - float(best_train["stop_level"])) < 1e-9
        )
        chosen_result = grid._evaluate_gpu(evaluator, [chosen_param], test_mask)[0]
        video_result = grid._evaluate_gpu(evaluator, [video_param], test_mask)[0]

        test_indices = [day_index[day] for day in test_days]
        for index in test_indices:
            oos_video_dd_peak = max(
                oos_video_dd_peak, float(video_result["daily_drawdown_rs"][index])
            )
            oos_chosen_dd_peak = max(
                oos_chosen_dd_peak, float(chosen_result["daily_drawdown_rs"][index])
            )

        def _accumulate(target: dict[str, Any], result: Mapping[str, Any]):
            target["trades"] += int(result["trades"])
            target["wins"] += int(result["wins"])
            target["net_points"] += float(result["net_points"])
            target["net_rs"] += float(result["net_rs"])
            target["fees_rs"] += float(result["fees_rs"])

        def _gather_geometry(levels: Mapping[str, Any], bucket: dict[str, list[float]],
                             test_days: Sequence[str]):
            test_index_set = set(day_index[day] for day in test_days)
            per_day, _ = _dataset_geometry_by_day(
                bundle.dataset,
                float(levels["target_level"]),
                float(levels["stop_level"]),
            )
            for day_index_flat in test_index_set:
                sl_day, tp_day, span_day = per_day[day_index_flat]
                bucket["sl"].extend(sl_day)
                bucket["tp"].extend(tp_day)
                bucket["span"].extend(span_day)

        _accumulate(oos_video_sums, video_result)
        _accumulate(oos_chosen_sums, chosen_result)
        oos_video_days.extend(test_days)
        oos_chosen_days.extend(test_days)
        _gather_geometry(video_param, geometry_video, test_days)
        _gather_geometry(chosen_param, geometry_chosen, test_days)
        folds.append({
            "fold": fold_index,
            "train": {"start": train_start, "end": train_end, "days": len(train_days)},
            "test": {"start": test_start, "end": test_end, "days": len(test_days)},
            "in_sample_best": best_train,
            "out_of_sample_chosen": {
                "target_level": float(chosen_param["target_level"]),
                "stop_level": float(chosen_param["stop_level"]),
                "trades": int(chosen_result["trades"]),
                "wins": int(chosen_result["wins"]),
                "win_rate": float(chosen_result["win_rate"]),
                "net_points": float(chosen_result["net_points"]),
                "net_rs": float(chosen_result["net_rs"]),
                "max_drawdown_points": float(chosen_result["max_drawdown_points"]),
                "max_drawdown_rs": float(chosen_result["max_drawdown_rs"]),
                "profit_factor": chosen_result["profit_factor"],
                "fees_rs": float(chosen_result["fees_rs"]),
            },
            "out_of_sample_video": {
                "target_level": float(video_param["target_level"]),
                "stop_level": float(video_param["stop_level"]),
                "trades": int(video_result["trades"]),
                "wins": int(video_result["wins"]),
                "win_rate": float(video_result["win_rate"]),
                "net_points": float(video_result["net_points"]),
                "net_rs": float(video_result["net_rs"]),
                "max_drawdown_points": float(video_result["max_drawdown_points"]),
                "max_drawdown_rs": float(video_result["max_drawdown_rs"]),
                "profit_factor": video_result["profit_factor"],
                "fees_rs": float(video_result["fees_rs"]),
            },
            "test_day_indices": test_indices,
        })
        print(
            f"[WF fold {fold_index}] train={train_start}..{train_end} "
            f"test={test_start}..{test_end} "
            f"chosen tgt={float(chosen_param['target_level']):g} "
            f"stop={float(chosen_param['stop_level']):g} "
            f"OOS trades={int(chosen_result['trades'])} "
            f"net={float(chosen_result['net_points']):+.2f}pts "
            f"| video-fixed OOS trades={int(video_result['trades'])} "
            f"net={float(video_result['net_points']):+.2f}pts",
            flush=True,
        )

    def _summary(sums: dict[str, Any], dd_peak: float, geometry: dict[str, list[float]],
                 oos_days: Sequence[str]) -> dict[str, Any]:
        def _stats(values: Sequence[float]) -> dict[str, float]:
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
            return {
                "count": int(arr.size),
                "mean": round(float(arr.mean()), 2),
                "median": round(float(np.median(arr)), 2),
                "min": round(float(arr.min()), 2),
                "max": round(float(arr.max()), 2),
            }
        trades = int(sums["trades"])
        wins = int(sums["wins"])
        return {
            "trades": trades,
            "avg_trades_per_day": round(trades / max(1, len(oos_days)), 3),
            "days": len(oos_days),
            "wins": wins,
            "win_rate": round(100.0 * wins / max(1, trades), 2),
            "net_points": round(float(sums["net_points"]), 2),
            "net_rs": round(float(sums["net_rs"]), 2),
            "max_drawdown_points": round(dd_peak / LOT_SIZE, 2),
            "profit_factor": None,
            "fees_rs": round(float(sums["fees_rs"]), 2),
            "avg_sl_points": _stats(geometry["sl"]),
            "avg_tp_points": _stats(geometry["tp"]),
            "avg_span_points": _stats(geometry["span"]),
        }

    oos_video_summary = _summary(oos_video_sums, oos_video_dd_peak, geometry_video, oos_video_days)
    oos_chosen_summary = _summary(oos_chosen_sums, oos_chosen_dd_peak, geometry_chosen, oos_chosen_days)
    return {
        "variant_id": variant.variant_id,
        "variant": variant.as_dict(),
        "folds": folds,
        "out_of_sample_video_fixed": oos_video_summary,
        "out_of_sample_chosen_per_fold": oos_chosen_summary,
        "grid_evaluations": evaluator.evaluations - before_evaluations,
        "cuda_ms": round(evaluator.cuda_ms - before_cuda_ms, 3),
        "wall_seconds": round(time.perf_counter() - started, 6),
    }


def _select_flavors(explicit: Sequence[str] | None) -> list[ObVariant]:
    if explicit:
        selected = []
        for token in explicit:
            flavor = token.strip().lower()
            if flavor not in FLAVORS:
                raise ValueError(
                    f"unsupported order-block flavor: {flavor} (use flip or turn)"
                )
            selected.append(ObVariant(flavor=flavor))
        return selected
    return list(DEFAULT_VARIANTS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=grid.historical.DEFAULT_DATA_ROOT)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--timeframes", default="1,2,3,5",
                        help="union of these bar minutes (1,2,3,5); merged into ONE stream")
    parser.add_argument("--flavors", nargs="+", default=None,
                        help="explicit flavors (flip, turn), or omit for both")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--fallback-targets", nargs="+", default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--stops", nargs="+", default=None)
    parser.add_argument("--prep-workers", type=int,
                        default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--batch-size", type=int, default=16,
                        help="padded matrix batch (default 16 covers the 12 video configs)")
    parser.add_argument("--fixed-cost-per-trade", type=float, default=FIXED_COST_PER_TRADE)
    parser.add_argument("--smoke", action="store_true", help="use exactly the first five available days")
    parser.add_argument("--wfo", action="store_true",
                        help="expanding walk-forward over the video exit grid "
                             "(full data + tensor cache required; GPU only)")
    parser.add_argument("--allow-expensive", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="smoke/parity debug only")
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--tensor-cache-dir", default=None)
    parser.add_argument("--output",
                        default=str(ROOT / "artifacts/f6_hybrid/order_block_optimus_full.json"))
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.prep_workers <= 0:
        raise SystemExit("--prep-workers must be positive")
    if not args.smoke and not args.allow_expensive and not args.wfo:
        raise SystemExit("refusing a non-smoke run; use --smoke or --allow-expensive")
    if args.wfo and args.smoke:
        raise SystemExit("--wfo cannot be combined with --smoke")
    if args.wfo and not args.allow_expensive:
        raise SystemExit("--wfo requires --allow-expensive")
    if args.allow_cpu and not args.smoke:
        raise SystemExit("--allow-cpu is restricted to smoke/parity debugging")
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")

    timeframes = tuple(int(piece) for piece in args.timeframes.split(",") if piece)
    if not timeframes or any(tf not in (1, 2, 3, 5) for tf in timeframes):
        raise SystemExit(f"--timeframes must be a comma list of 1,2,3,5: {args.timeframes}")

    try:
        variants = _select_flavors(args.flavors)
        targets = grid._parse_float_tokens(args.targets, VIDEO_TARGETS)
        fallback_targets = grid._parse_float_tokens(args.fallback_targets, VIDEO_FALLBACK_TARGETS)
        thresholds = grid._parse_float_tokens(args.thresholds, VIDEO_THRESHOLDS)
        stops = grid._parse_float_tokens(args.stops, VIDEO_STOPS)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    exit_param_count = len(_video_grid_params(targets, fallback_targets, thresholds, stops))
    config_count = len(variants) * exit_param_count
    if config_count < 5:
        raise SystemExit("the selected grid must contain at least five unique configurations")
    if args.batch_size < exit_param_count:
        print(f"[GRID] padding matrix batch {args.batch_size} -> {exit_param_count} "
              f"(one pass per variant)", flush=True)
        args.batch_size = exit_param_count

    device = optimus.require_device(args.allow_cpu)
    print(f"[GPU INIT] device={device} allocator={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        torch.cuda.reset_peak_memory_stats(device)
        print(f"[GPU INIT] name={props.name} vram_gb={props.total_memory / (1024 ** 3):.2f} "
              f"cuda={torch.version.cuda}", flush=True)

    adapter = grid.PolarsHistoricalDataAdapter(args.data_root, start=args.start, end=args.end)
    available = adapter.available_days(args.start, args.end)
    if not available:
        raise SystemExit(f"no overlapping index/options days in {args.start}..{args.end}")
    days = available[:5] if args.smoke else available
    if len(days) < 3:
        raise SystemExit("at least three available days are required for parity")
    if args.wfo and len(days) < 500:
        raise SystemExit("full 2020-2026 data is required for walk-forward")
    parity_days = days[:3]
    tensor_cache_dir = Path(args.tensor_cache_dir) if args.tensor_cache_dir is not None else None

    all_configs: list[dict[str, Any]] = []
    variant_reports: list[dict[str, Any]] = []
    wf_reports: list[dict[str, Any]] = []
    total_cpu_seconds = 0.0
    total_grid_cuda_ms = 0.0
    run_started = time.perf_counter()
    for variant_index, variant in enumerate(variants, start=1):
        print(f"[VARIANT] {variant_index}/{len(variants)} {variant.variant_id} "
              f"S1=({variant.s1_k_period},{variant.s1_d_period}) span={variant.min_span:g} "
              f"age={variant.setup_max_age} buffer={variant.touch_buffer:g} "
              f"block_w={variant.block_window} sq={variant.squeeze_ratio:g} "
              f"lb={variant.lookback} msp={variant.min_block_span:g} "
              f"bkw={variant.breakout_window} ageb={variant.max_block_age} "
              f"stl={variant.straight_lookback} "
              f"union_tfs={list(timeframes)}", flush=True)
        bundle = collect_ob_cpu_dataset(
            adapter, days, variant, timeframes,
            workers=args.prep_workers,
            parity_days=parity_days,
            tensor_cache_dir=tensor_cache_dir,
        )
        total_cpu_seconds += bundle.prep_seconds
        gpu_variant = grid.to_gpu_dataset(bundle.dataset, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluator = optimus.GpuEvaluator(
            gpu_variant.engine,
            args.batch_size,
            brokerage_per_order=0.0,
            fixed_cost_per_trade=args.fixed_cost_per_trade,
        )
        if args.wfo:
            wf_report = run_walk_forward(
                variant, bundle, gpu_variant, evaluator, days,
                targets, fallback_targets, thresholds, stops,
            )
            wf_reports.append(wf_report)
            print(
                f"[WF VIDEO-FIXED] {variant.variant_id} "
                f"trades={wf_report['out_of_sample_video_fixed']['trades']} "
                f"avg/day={wf_report['out_of_sample_video_fixed']['avg_trades_per_day']} "
                f"net={wf_report['out_of_sample_video_fixed']['net_points']:+.2f}pts "
                f"DD={wf_report['out_of_sample_video_fixed']['max_drawdown_points']:.2f}",
                flush=True,
            )
            del evaluator, gpu_variant
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            continue
        print(f"[PARITY] {variant.variant_id} exactly three dates {parity_days}", flush=True)
        parity = validate_ob_parity(
            bundle, gpu_variant, evaluator, parity_days,
            targets, fallback_targets, thresholds, stops,
        )
        print(f"[PARITY] {variant.variant_id} PASS", flush=True)
        configs, grid_timing = evaluate_video_grid(
            variant, gpu_variant, evaluator, days,
            targets, fallback_targets, thresholds, stops,
        )
        total_grid_cuda_ms += float(grid_timing["cuda_ms"])
        all_configs.extend(configs)
        attribution = _event_attribution(bundle)
        variant_reports.append({
            "variant": variant.as_dict(),
            "union_timeframes": list(timeframes),
            "cache": {"source": bundle.cache_source, "path": bundle.cache_path},
            "event_attribution": attribution,
            "exit_geometry": _config_geometry_report(bundle, configs, days),
            "timing": {
                "cpu_prep_seconds": round(bundle.prep_seconds, 6),
                "parity_gpu_cuda_ms": parity["gpu_cuda_ms"],
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

    if args.wfo:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            memory = {
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2),
            }
        else:
            memory = {"peak_allocated_mb": None, "peak_reserved_mb": None}
        result = {
            "engine": "order_block_optimus_gpu",
            "mode": "expanding_walk_forward",
            "entry_source": "order_block (Fibb Block Strategy video, "
                            "nearest completed block at S1-turn setup, no time window)",
            "data_root": str(Path(args.data_root)),
            "date_range": {"start": args.start, "end": args.end},
            "days": days,
            "timeframes": list(timeframes),
            "selection": {
                "flavor_count": len(variants),
                "source": "explicit_flavors" if args.flavors else "both_flavors",
                "variants": [variant.as_dict() for variant in variants],
            },
            "exit_grid_axes": {
                "target_level": list(targets),
                "fallback_target_level": list(fallback_targets),
                "option_point_threshold": list(thresholds),
                "stop_level": list(stops),
                "note": "fallback layer inert: CPU/GPU fallback branches gate on "
                        "target == 0.29; video targets/stops kept verbatim",
            },
            "walk_forward": {
                "folds": [
                    {"train": {"start": fold[0], "end": fold[1]},
                     "test": {"start": fold[2], "end": fold[3]}}
                    for fold in WF_FOLDS
                ],
                "selection": "per-fold in-sample best by net - %.2f*DD" % WF_DD_PENALTY,
                "video_fixed": "target 0 / stop 1.155 replayed OOS every fold (no selection)",
            },
            "execution_contract": {
                "entry_stream": "union of 4 TF order-block streams merged per day "
                                "(dedup minute/side/symbol)",
                "signal_flavors": {
                    "flip": "breakout + retest + flip back through block edge (80% rule)",
                    "turn": "price enters block from beyond the edge and turns inside",
                },
                "cpu_process_day_in_trial_loop": False,
                "actual_option_ohlc": True,
                "lot_size": LOT_SIZE,
                "fixed_cost_per_trade": args.fixed_cost_per_trade,
                "fixed_bars": optimus.T_BARS,
                "one_global_position_per_day": True,
                "matrix_engine": "smart_fib_optimus_gpu.simulate_event_batch_matrix",
            },
            "hardware": grid._device_report(device),
            "gpu_memory_peak": memory,
            "variant_reports": wf_reports,
            "configs_evaluated": len(variants) * len(_video_grid_params(
                targets, fallback_targets, thresholds, stops,
            )) * len(WF_FOLDS),
            "prep_seconds": round(total_cpu_seconds, 3),
            "wall_seconds": round(time.perf_counter() - run_started, 3),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"JSON: {output}")
        return 0

    config_ids = [config["config_id"] for config in all_configs]
    if len(config_ids) != len(set(config_ids)):
        raise RuntimeError("GPU grid produced duplicate final configuration identities")

    eligible = [config for config in all_configs if config["trades"] >= args.min_trades]
    ranked = sorted(
        eligible,
        key=lambda config: (
            float(config["net_points"]),
            -float(config["max_drawdown_points"]),
            float(config["net_rs"]),
        ),
        reverse=True,
    )
    top_net = ranked[:5]
    for rank, config in enumerate(top_net, start=1):
        print(
            f"[TOP-NET {rank}] {config['variant']['variant_id']} "
            f"target={config['target_level']:g} stop={config['stop_level']:g} "
            f"trades={config['trades']} WR={config['win_rate']:.2f}% "
            f"net={config['net_points']:+.2f}pts/Rs {config['net_rs']:+,.2f} "
            f"DD={config['max_drawdown_points']:.2f} PF={config['profit_factor']} "
            f"fees=Rs {config['fees_rs']:,.2f} score={config['composite_score']:+.2f}",
            flush=True,
        )

    video_requested = [
        config for config in all_configs
        if abs(float(config["target_level"]) - 0.0) < 1e-9
        and abs(float(config["stop_level"]) - 1.155) < 1e-9
    ]
    for config in video_requested:
        print(
            f"[VIDEO target0/stop1.155] {config['variant']['variant_id']} "
            f"trades={config['trades']} WR={config['win_rate']:.2f}% "
            f"net={config['net_points']:+.2f}pts/Rs {config['net_rs']:+,.2f} "
            f"DD={config['max_drawdown_points']:.2f} PF={config['profit_factor']} "
            f"fees=Rs {config['fees_rs']:,.2f}",
            flush=True,
        )

    composite = sorted(
        eligible,
        key=lambda config: (
            float(config["composite_score"]),
            float(config["net_points"]),
            -float(config["max_drawdown_points"]),
        ),
        reverse=True,
    )[:5]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory = {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2),
        }
    else:
        memory = {"peak_allocated_mb": None, "peak_reserved_mb": None}

    result = {
        "engine": "order_block_optimus_gpu",
        "mode": "smoke" if args.smoke else "union_of_tf_streams",
        "entry_source": "order_block (Fibb Block Strategy video, "
                        "nearest completed block at S1-turn setup, no time window)",
        "data_root": str(Path(args.data_root)),
        "date_range": {"start": args.start, "end": args.end},
        "days": days,
        "timeframes": list(timeframes),
        "selection": {
            "flavor_count": len(variants),
            "source": "explicit_flavors" if args.flavors else "both_flavors",
            "variants": [variant.as_dict() for variant in variants],
        },
        "exit_grid_axes": {
            "target_level": list(targets),
            "fallback_target_level": list(fallback_targets),
            "option_point_threshold": list(thresholds),
            "stop_level": list(stops),
            "note": "fallback layer inert: CPU/GPU fallback branches gate on "
                    "target == 0.29; video targets/stops kept verbatim",
        },
        "execution_contract": {
            "entry_stream": "union of 4 TF order-block streams merged per day "
                            "(dedup minute/side/symbol)",
            "signal_flavors": {
                "flip": "breakout + retest + flip back through block edge (80% rule)",
                "turn": "price enters block from beyond the edge and turns inside",
            },
            "cpu_process_day_in_trial_loop": False,
            "actual_option_ohlc": True,
            "lot_size": LOT_SIZE,
            "fixed_cost_per_trade": args.fixed_cost_per_trade,
            "fixed_bars": optimus.T_BARS,
            "one_global_position_per_day": True,
            "matrix_engine": "smart_fib_optimus_gpu.simulate_event_batch_matrix",
        },
        "hardware": grid._device_report(device),
        "gpu_memory_peak": memory,
        "variant_reports": variant_reports,
        "results": all_configs,
        "top_five_by_net_points": top_net,
        "top_five_by_composite": composite,
        "video_target0_stop1p155": video_requested,
        "configs_evaluated": len(all_configs),
        "prep_seconds": round(total_cpu_seconds, 3),
        "grid_gpu_cuda_ms": round(total_grid_cuda_ms, 3),
        "wall_seconds": round(time.perf_counter() - run_started, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
