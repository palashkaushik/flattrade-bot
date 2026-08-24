"""Union-of-TF Smart Fib Optimus backtest — GPU-first.

The four timeframes (1m/2m/3m/5m, bias filter = 5x TF) are merged into ONE
event stream per day (union, dedup by (minute, side, symbol), lowest TF wins
at a shared minute) and evaluated on CUDA via the Optimus matrix engine
(``simulate_event_batch_matrix``) — the CPU simulator is used ONLY as a
three-date parity oracle, never for the production grid.

Both signal-source kinds are exercised by construction (same extractor as the
live strategy and the champion combined backtest):
  * index-chart events (Case 1): S1 turn UP for CE / turn DOWN for PE, inside
    the fib zone, with the 5x-TF index bias filter;
  * option-chart events (Case 2): S1 turn UP on the option chart for both
    sides (option-monitored exits).
Event attribution by source kind and source TF is reported per run so the
index-trade inclusion is auditable.

Exit grids: target/fallback/threshold/stop. Every axis pair keeps the entry
inside the zone BETWEEN the TP level and the SL level (max target 0.5 <= min
zone start 0.5; min stop 1.13 > max zone end 1.0).

Ranking goal: variant with the best net points and the least drawdown.
Leaderboard sorts by (net_points DESC, max_drawdown_points ASC); the
composite score (net - 0.20 * DD) is kept for reference.

Runbook compliance (OPTIMIZED_GPU_BACKTEST.md): no per-trade scalar
readbacks, no CUDA graphs (variable-size gathers), resident VRAM tensors,
pinned non-blocking copies, cudaMallocAsync, parity gate before every grid,
one padded matrix pass per variant (--batch-size 135 covers the default 135
exit configs without padding).
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
from artifacts.f6_hybrid import smart_fib_optimus_gpu as optimus
from artifacts.f6_hybrid import smart_fib_optimus_grid_gpu as grid

LOT_SIZE = int(smart_core.LOT_SIZE)
FIXED_COST_PER_TRADE = 40.0
UNION_CACHE_VERSION = 1

CHAMPION_VARIANT_ID = "s1k12d4_span15_age45_buf0p5_z0p5-0p786"
CHAMPION_EXITS = dict(
    target_level=0.786,
    fallback_target_level=0.0,
    option_point_threshold=5.0,
    stop_level=1.13,
)
REQUESTED_EXITS = dict(
    target_level=0.29,
    fallback_target_level=0.0,
    option_point_threshold=5.0,
    stop_level=1.155,
)
KNOWN_CHAMPION_COMBINED = dict(
    trades=17005,
    net_points=35122.4,
    max_drawdown_points=136.62,
    profit_factor=4.62,
)


@dataclass
class UnionDayRaw(grid.VariantDayRaw):
    """VariantDayRaw plus per-day union attribution."""

    attribution: dict[str, Any] = field(default_factory=dict)


def _extract_union_day(
    task: tuple[str, str, str, str, tuple[int, ...], tuple[Any, ...]],
) -> UnionDayRaw:
    """Worker: extract every TF for one day, merge into the union stream."""
    data_root, start, end, day, timeframes, values = task
    variant = grid.SignalVariant(*values).validate()
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
    for tf in timeframes:
        prepared = smart_core.extract_day_events(
            day,
            cache_loader=adapter.load_day_cache,
            bar_minutes=tf,
            filter_period=5 * tf,
            min_span=variant.min_span,
            touch_buffer=variant.touch_buffer,
            setup_max_age=variant.setup_max_age,
            zone_start=variant.zone_start,
            zone_end=variant.zone_end,
            s1_k_period=variant.s1_k_period,
            s1_d_period=variant.s1_d_period,
            debug=False,
        )
        if not prepared:
            raise RuntimeError(f"no Smart Fib cache payload for {day} tf={tf}")
        selected, raw_count = optimus._select_day_events(prepared)
        raw_total += raw_count
        per_tf_selected[tf] = len(selected)
        for signal in selected:
            merged.append((int(signal["minute"]), tf, signal))
        if tf == 1:
            payload = prepared
    merged.sort(key=lambda item: (item[0], item[1]))
    events: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    kinds = {"index": 0, "option": 0}
    for minute, tf, signal in merged:
        key = (minute, signal["side"], signal["symbol"])
        if key in seen:
            continue
        seen.add(key)
        events.append({**signal, "timeframe": "combined"})
        kind = str(signal["fib_source"])
        kinds[kind] = kinds.get(kind, 0) + 1
    attribution = {
        "selected": len(events),
        "raw_total": raw_total,
        "by_source_kind": kinds,
        "by_source_tf": per_tf_selected,
    }
    return UnionDayRaw(
        variant=variant,
        day=day,
        prepared=dict(payload),
        selected_events=events,
        raw_event_count=raw_total,
        contract_keys=list(payload["records"]),
        attribution=attribution,
    )


def _union_cache_path(
    cache_dir: Path, days: Sequence[str], variant: grid.SignalVariant, timeframes: tuple[int, ...]
) -> Path:
    token = "x".join(str(tf) for tf in timeframes)
    return cache_dir / (
        f"smart_fib_union_tensor_cache_{days[0]}_{days[-1]}_{variant.variant_id}_u{token}.npz"
    )


def _save_union_cache(dataset: grid.VariantCpuDataset, path: Path, data_root: str | Path,
                      variant: grid.SignalVariant, timeframes: tuple[int, ...]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite union tensor cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache_version": UNION_CACHE_VERSION,
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
    }
    arrays = {name: getattr(dataset, name) for name in grid._CACHE_ARRAY_NAMES}
    np.savez_compressed(path, **arrays, metadata=np.array(json.dumps(metadata)))


def _load_union_cache(path: Path, data_root: str | Path, days: Sequence[str],
                      variant: grid.SignalVariant,
                      timeframes: tuple[int, ...]) -> grid.VariantCpuDataset | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("cache_version") != UNION_CACHE_VERSION:
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
        return grid.VariantCpuDataset(
            days=list(days),
            **arrays,
            event_symbols=metadata["event_symbols"],
            contract_symbols=metadata["contract_symbols"],
            raw_event_counts=[int(value) for value in metadata["raw_event_counts"]],
            selected_event_counts=[int(value) for value in metadata["selected_event_counts"]],
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


@dataclass
class UnionBundle:
    variant: grid.SignalVariant
    dataset: grid.VariantCpuDataset
    parity_payloads: dict[str, dict[str, Any]]
    union_events_by_day: dict[str, list[dict[str, Any]]]
    attribution_by_day: dict[str, dict[str, Any]]
    cache_source: str
    cache_path: str | None
    prep_seconds: float


def collect_union_cpu_dataset(
    adapter: grid.PolarsHistoricalDataAdapter,
    days: Sequence[str],
    variant: grid.SignalVariant,
    timeframes: tuple[int, ...],
    *,
    workers: int,
    parity_days: Sequence[str],
    tensor_cache_dir: Path | None,
) -> UnionBundle:
    """Extract the union stream once per day, freeze its CPU oracle tensors."""
    cached: grid.VariantCpuDataset | None = None
    cache_path: str | None = None
    if tensor_cache_dir is not None:
        path = _union_cache_path(tensor_cache_dir, days, variant, timeframes)
        cached = _load_union_cache(path, adapter.data_root, days, variant, timeframes)
        if cached is not None:
            cache_path = str(path)
    if cached is not None:
        print(
            f"[UNION CPU PREP] {variant.variant_id} cache N={cached.n_days} "
            f"T={optimus.T_BARS} C={cached.contract_slots}",
            flush=True,
        )
        return UnionBundle(
            variant=variant,
            dataset=cached,
            parity_payloads={},
            union_events_by_day={},
            attribution_by_day={},
            cache_source="union_variant_cache",
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
    )
    tasks = [
        (str(adapter.data_root), adapter.start, adapter.end, day, timeframes, task_values)
        for day in days
    ]
    raw_days: list[UnionDayRaw] = []
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=grid._init_grid_worker,
            initargs=(str(adapter.data_root), adapter.start, adapter.end),
        ) as pool:
            iterator = pool.map(_extract_union_day, tasks, chunksize=1)
            for count, raw in enumerate(iterator, start=1):
                raw_days.append(raw)
                if count == 1 or count % 25 == 0 or count == len(tasks):
                    print(
                        f"[UNION CPU PREP] {variant.variant_id} {count}/{len(tasks)} "
                        f"last={raw.day}",
                        flush=True,
                    )
    else:
        for count, task in enumerate(tasks, start=1):
            raw = _extract_union_day(task)
            raw_days.append(raw)
            if count == 1 or count % 25 == 0 or count == len(tasks):
                print(
                    f"[UNION CPU PREP] {variant.variant_id} {count}/{len(tasks)} "
                    f"last={raw.day}",
                    flush=True,
                )

    dataset = grid._build_cpu_dataset(days, raw_days)
    parity_payloads = {raw.day: raw.prepared for raw in raw_days if raw.day in set(parity_days)}
    union_events_by_day = {raw.day: raw.selected_events for raw in raw_days if raw.day in set(parity_days)}
    attribution_by_day = {raw.day: raw.attribution for raw in raw_days}
    source = "cpu_extraction"
    if tensor_cache_dir is not None:
        path = _union_cache_path(tensor_cache_dir, days, variant, timeframes)
        _save_union_cache(dataset, path, adapter.data_root, variant, timeframes)
        cache_path = str(path)
        source = "cpu_extraction_saved_union_cache"
    elapsed = time.perf_counter() - started
    total_events = sum(dataset.selected_event_counts)
    print(
        f"[UNION CPU PREP] {variant.variant_id} complete {elapsed:.3f}s "
        f"N={dataset.n_days} T={optimus.T_BARS} C={dataset.contract_slots} "
        f"events={total_events}",
        flush=True,
    )
    return UnionBundle(
        variant=variant,
        dataset=dataset,
        parity_payloads=parity_payloads,
        union_events_by_day=union_events_by_day,
        attribution_by_day=attribution_by_day,
        cache_source=source,
        cache_path=cache_path,
        prep_seconds=elapsed,
    )


def _union_cpu_trades(
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


def validate_union_parity(
    bundle: UnionBundle,
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
    params = grid._grid_params(targets, fallback_targets, thresholds, stops)
    cpu_by_combo: dict[tuple[float, float, float, float], dict[str, Any]] = {}
    for param in params:
        daily = {}
        for day in parity_days:
            daily[day] = grid._cpu_stats(
                _union_cpu_trades(
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


def _event_attribution(bundle: UnionBundle) -> dict[str, Any]:
    """Aggregate union event attribution (fresh extraction only)."""
    kinds = {"index": 0, "option": 0}
    tfs: dict[int, int] = {}
    days = 0
    for day, attr in bundle.attribution_by_day.items():
        days += 1
        for kind, count in attr["by_source_kind"].items():
            kinds[kind] = kinds.get(kind, 0) + int(count)
        for tf, count in attr["by_source_tf"].items():
            tfs[int(tf)] = tfs.get(int(tf), 0) + int(count)
    return {
        "days": days,
        "events_by_source_kind": kinds,
        "events_by_source_tf": tfs,
        "index_share_pct": round(100.0 * kinds["index"] / max(1, sum(kinds.values())), 2),
    }


def _champion_cross_check(all_configs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """If the champion variant+exits were evaluated, compare to the known CPU run."""
    for config in all_configs:
        if config["variant"]["variant_id"] != CHAMPION_VARIANT_ID:
            continue
        if not all(
            abs(float(config[key]) - float(value)) < 1e-9
            for key, value in CHAMPION_EXITS.items()
        ):
            continue
        return {
            "config_id": config["config_id"],
            "known": KNOWN_CHAMPION_COMBINED,
            "measured": {
                "trades": config["trades"],
                "net_points": config["net_points"],
                "max_drawdown_points": config["max_drawdown_points"],
                "profit_factor": config["profit_factor"],
            },
            "deltas": {
                "trades": config["trades"] - KNOWN_CHAMPION_COMBINED["trades"],
                "net_points": round(config["net_points"] - KNOWN_CHAMPION_COMBINED["net_points"], 2),
                "max_drawdown_points": round(
                    config["max_drawdown_points"] - KNOWN_CHAMPION_COMBINED["max_drawdown_points"], 2
                ),
                "profit_factor": round(config["profit_factor"] - KNOWN_CHAMPION_COMBINED["profit_factor"], 4),
            },
        }
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=grid.historical.DEFAULT_DATA_ROOT)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--timeframes", default="1,2,3,5",
                        help="union of these bar minutes (1,2,3,5); merged into ONE stream")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="explicit variant ids, or omit for the staged shortlist (max 5)")
    parser.add_argument("--max-variants", type=int, default=5)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--fallback-targets", nargs="+", default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--stops", nargs="+", default=None)
    parser.add_argument("--prep-workers", type=int,
                        default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--batch-size", type=int, default=135,
                        help="padded matrix batch (default 135 = one pass for the default exit grid)")
    parser.add_argument("--fixed-cost-per-trade", type=float, default=FIXED_COST_PER_TRADE)
    parser.add_argument("--smoke", action="store_true", help="use exactly the first five available days")
    parser.add_argument("--allow-expensive", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="smoke/parity debug only")
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--tensor-cache-dir", default=None)
    parser.add_argument("--output",
                        default=str(ROOT / "artifacts/f6_hybrid/smart_fib_combined_tf_optimus_full.json"))
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.prep_workers <= 0:
        raise SystemExit("--prep-workers must be positive")
    if not args.smoke and not args.allow_expensive:
        raise SystemExit("refusing a non-smoke run; use --smoke or --allow-expensive")
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
        variants = grid.select_variants(args.variants, args.max_variants, args.allow_expensive)
        targets, fallback_targets, stops = grid._validate_exit_axes(
            grid._parse_float_tokens(args.targets, grid.DEFAULT_TARGET_LEVELS),
            grid._parse_float_tokens(args.fallback_targets, grid.DEFAULT_FALLBACK_TARGET_LEVELS),
            grid._parse_float_tokens(args.stops, grid.DEFAULT_STOP_LEVELS),
        )
        thresholds = grid._validate_axis(
            grid._parse_float_tokens(args.thresholds, grid.EXIT_THRESHOLDS),
            grid.EXIT_THRESHOLDS,
            "threshold",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    exit_param_count = len(grid._grid_params(targets, fallback_targets, thresholds, stops))
    config_count = len(variants) * exit_param_count
    if config_count < 5:
        raise SystemExit("the selected grid must contain at least five unique configurations")

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
    parity_days = days[:3]
    tensor_cache_dir = Path(args.tensor_cache_dir) if args.tensor_cache_dir is not None else None

    all_configs: list[dict[str, Any]] = []
    variant_reports: list[dict[str, Any]] = []
    total_cpu_seconds = 0.0
    total_grid_cuda_ms = 0.0
    run_started = time.perf_counter()
    for variant_index, variant in enumerate(variants, start=1):
        print(f"[VARIANT] {variant_index}/{len(variants)} {variant.variant_id} "
              f"S1=({variant.s1_k_period},{variant.s1_d_period}) span={variant.min_span:g} "
              f"age={variant.setup_max_age} buffer={variant.touch_buffer:g} "
              f"zone=({variant.zone_start:g},{variant.zone_end:g}) union_tfs={list(timeframes)}",
              flush=True)
        bundle = collect_union_cpu_dataset(
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
        print(f"[PARITY] {variant.variant_id} exactly three dates {parity_days}", flush=True)
        parity = validate_union_parity(
            bundle, gpu_variant, evaluator, parity_days,
            targets, fallback_targets, thresholds, stops,
        )
        print(f"[PARITY] {variant.variant_id} PASS", flush=True)
        configs, grid_timing = grid.evaluate_variant_grid(
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
            f"target={config['target_level']:g} fallback={config['fallback_target_level']:g} "
            f"thr={config['option_point_threshold']:g} stop={config['stop_level']:g} "
            f"trades={config['trades']} WR={config['win_rate']:.2f}% "
            f"net={config['net_points']:+.2f}pts/Rs {config['net_rs']:+,.2f} "
            f"DD={config['max_drawdown_points']:.2f} PF={config['profit_factor']} "
            f"fees=Rs {config['fees_rs']:,.2f} score={config['composite_score']:+.2f}",
            flush=True,
        )

    requested = next(
        (
            config for config in all_configs
            if config["variant"]["variant_id"] == grid.BASELINE_VARIANT.variant_id
            and all(
                abs(float(config[key]) - float(value)) < 1e-9
                for key, value in REQUESTED_EXITS.items()
            )
        ),
        None,
    )
    if requested is not None:
        print(
            f"[REQUESTED thr5/1.155] {requested['variant']['variant_id']} "
            f"trades={requested['trades']} WR={requested['win_rate']:.2f}% "
            f"net={requested['net_points']:+.2f}pts/Rs {requested['net_rs']:+,.2f} "
            f"DD={requested['max_drawdown_points']:.2f} PF={requested['profit_factor']} "
            f"fees=Rs {requested['fees_rs']:,.2f}",
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
    champion_check = _champion_cross_check(all_configs)
    if champion_check is not None:
        print(f"[CHAMPION CROSS-CHECK] deltas {champion_check['deltas']}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory = {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2),
        }
    else:
        memory = {"peak_allocated_mb": None, "peak_reserved_mb": None}

    result = {
        "engine": "smart_fib_combined_tf_optimus_gpu",
        "mode": "smoke" if args.smoke else "union_of_tf_streams",
        "data_root": str(Path(args.data_root)),
        "date_range": {"start": args.start, "end": args.end},
        "days": days,
        "timeframes": list(timeframes),
        "selection": {
            "variant_count": len(variants),
            "source": "explicit_variants" if args.variants else "staged_shortlist",
            "max_variants": args.max_variants,
            "variants": [variant.as_dict() for variant in variants],
        },
        "exit_grid_axes": {
            "target_level": list(targets),
            "fallback_target_level": list(fallback_targets),
            "valid_target_fallback_pairs": [
                [target, fallback]
                for target in targets
                for fallback in fallback_targets
                if fallback <= target
            ],
            "option_point_threshold": list(thresholds),
            "stop_level": list(stops),
        },
        "execution_contract": {
            "entry_stream": "union of 4 TF streams merged per day (dedup minute/side/symbol)",
            "signal_cases": {
                "index": "Case 1: S1 turn up (CE) / turn down (PE) inside zone, 5x-TF bias filter",
                "option": "Case 2: S1 turn up on option chart for both sides",
            },
            "cpu_process_day_in_trial_loop": False,
            "actual_option_ohlc": True,
            "lot_size": LOT_SIZE,
            "fixed_cost_per_trade": args.fixed_cost_per_trade,
            "fixed_bars": optimus.T_BARS,
            "one_global_position_per_day": True,
            "matrix_engine": "smart_fib_optimus_gpu.simulate_event_batch_matrix",
            "entry_between_tp_and_sl": True,
        },
        "hardware": grid._device_report(device),
        "gpu_memory_peak": memory,
        "variant_reports": variant_reports,
        "results": all_configs,
        "top_five_by_net_points": top_net,
        "top_five_by_composite": composite,
        "requested_thr5_sl1p155": requested,
        "champion_cross_check": champion_check,
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
