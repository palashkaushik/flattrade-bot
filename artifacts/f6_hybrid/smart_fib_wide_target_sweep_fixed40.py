"""Wide-target sweep around the Smart Fib Optimus champion, fixed Rs40/trade cost.

Widens the TP axis to capture larger per-trade points: targets
(0.618, 0.786, 1.0, 1.272, 1.618, 2.0) x stops (1.05, 1.13, 1.272, 1.382, 1.618),
fallback 0.0, threshold 5.0, on the champion signal variant
(s1k12d4_span15_age45_buf0p5_z0p5-0p786). Runs both the full-window (non-WF)
evaluation and the expanding walk-forward used by the reference runner.
GPU-only; refuses CPU fallback and refuses to overwrite existing output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

from artifacts.f6_hybrid.smart_fib_cuda_bootstrap import configure_cuda_toolkit

configure_cuda_toolkit()

import numpy as np
import torch

from artifacts.f6_hybrid import smart_fib_optimus_gpu as optimus
from artifacts.f6_hybrid import smart_fib_optimus_grid_gpu as grid

CHAMPION_VARIANT = grid._parse_variant("12:4:15:45:0.5:0.5:0.786")

FOLDS = (
    ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2020-01-01", "2024-12-31", "2025-01-01", "2026-05-05"),
)


def _params():
    targets = (0.618, 0.786, 1.0, 1.272, 1.618, 2.0)
    stops = (1.05, 1.13, 1.272, 1.382, 1.618)
    out = []
    for target in targets:
        for stop in stops:
            out.append({
                "stop_level": float(stop),
                "target_level": float(target),
                "fallback_target_level": 0.0,
                "option_point_threshold": 5.0,
            })
    return out


def _score(result, penalty):
    return float(result["net_points"]) - penalty * float(result["max_drawdown_points"])


def _mask(days, start, end, device):
    return torch.tensor(
        [start <= day <= end for day in days],
        dtype=torch.bool,
        device=device,
    )


def _load(adapter, days, cache_dir, device, batch_size):
    dataset, source, path = grid._cached_variant_dataset(
        adapter.data_root, days, CHAMPION_VARIANT, cache_dir,
    )
    if dataset is None:
        raise RuntimeError(f"missing full GPU tensor cache for {CHAMPION_VARIANT.variant_id}")
    print(f"[CACHE] {CHAMPION_VARIANT.variant_id} source={source} path={path}", flush=True)
    resident = grid.to_gpu_dataset(dataset, device)
    evaluator = optimus.GpuEvaluator(
        resident.engine,
        batch_size,
        brokerage_per_order=0.0,
        fixed_cost_per_trade=40.0,
    )
    return dataset, resident, evaluator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=r"C:\Users\user\Desktop\nifty50 data")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dd-penalty", type=float, default=0.50)
    parser.add_argument("--tensor-cache-dir", default="artifacts/f6_hybrid/smart_fib_grid_cache_full_float64")
    parser.add_argument("--output", default="artifacts/f6_hybrid/smart_fib_wide_target_sweep_fixed40.json")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")

    device = torch.device("cuda")
    adapter = grid.PolarsHistoricalDataAdapter(args.data_root, start=args.start, end=args.end)
    days = adapter.available_days(args.start, args.end)
    if len(days) < 500:
        raise SystemExit("full 2020-2026 data is required")
    params = _params()
    cache_dir = Path(args.tensor_cache_dir)
    started = time.perf_counter()

    dataset, resident, evaluator = _load(adapter, days, cache_dir, device, args.batch_size)

    all_mask = _mask(days, args.start, args.end, device)
    full_results = [
        dict(result, params=params[index])
        for index, result in enumerate(evaluator.evaluate(params, all_mask))
    ]
    ranked = sorted(
        full_results,
        key=lambda result: (_score(result, args.dd_penalty), float(result["net_points"]), -float(result["max_drawdown_points"])),
        reverse=True,
    )
    print("=== FULL WINDOW (non-WF) TOP 10 ===", flush=True)
    for index, result in enumerate(ranked[:10]):
        print(
            f"  #{index + 1} target={result['params']['target_level']} stop={result['params']['stop_level']} "
            f"trades={result['trades']} WR={result['win_rate']:.2f} net={result['net_points']:+.2f} "
            f"net_rs={result['net_rs']:+,.2f} DD={result['max_drawdown_points']:.2f} PF={result['profit_factor']:.4f}",
            flush=True,
        )

    best_train = [None] * len(FOLDS)
    fold_results = []
    timings = []
    for fold_index, (_, train_end, val_start, val_end) in enumerate(FOLDS):
        train_mask = _mask(days, args.start, train_end, device)
        fold_started = time.perf_counter()
        results = evaluator.evaluate(params, train_mask)
        timings.append({
            "fold": fold_index + 1,
            "cuda_ms": evaluator.last_cuda_ms,
            "wall_seconds": time.perf_counter() - fold_started,
        })
        winner = max(
            (
                dict(result, params=params[index])
                for index, result in enumerate(results)
            ),
            key=lambda result: (_score(result, args.dd_penalty), float(result["net_points"]), -float(result["max_drawdown_points"])),
        )
        winner["train_score"] = round(_score(winner, args.dd_penalty), 4)
        winner["fold"] = fold_index + 1
        winner["variant"] = CHAMPION_VARIANT.as_dict()
        best_train[fold_index] = winner
        print(
            f"[WF TRAIN] fold={fold_index + 1} score={winner['train_score']:+.2f} "
            f"net={winner['net_points']:+.2f} DD={winner['max_drawdown_points']:.2f} "
            f"params={{target={winner['params']['target_level']}, stop={winner['params']['stop_level']}}}",
            flush=True,
        )
        validation_mask = _mask(days, val_start, val_end, device)
        raw_validation = evaluator.evaluate([winner["params"]], validation_mask)[0]
        validation = dict(raw_validation, params=winner["params"], variant=CHAMPION_VARIANT.as_dict())
        validation["fold"] = fold_index + 1
        validation["train_selection"] = winner
        validation["validation_start"] = val_start
        validation["validation_end"] = val_end
        fold_results.append(validation)
        print(
            f"[WF VALIDATE] fold={fold_index + 1} {val_start}..{val_end} "
            f"trades={validation['trades']} net={validation['net_points']:+.2f} "
            f"DD={validation['max_drawdown_points']:.2f}",
            flush=True,
        )

    del dataset, resident, evaluator
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stitched = {
        "trades": 0,
        "wins": 0,
        "fees_rs": 0.0,
        "net_points": 0.0,
        "net_rs": 0.0,
        "max_fold_drawdown_points": 0.0,
    }
    stitched_daily_net_rs = np.zeros(len(days), dtype=np.float64)
    stitched_daily_peak_rs = np.zeros(len(days), dtype=np.float64)
    stitched_daily_trough_rs = np.zeros(len(days), dtype=np.float64)

    dataset, resident, evaluator = _load(adapter, days, cache_dir, device, args.batch_size)
    for fold_index, (_, _, val_start, val_end) in enumerate(FOLDS):
        winner = best_train[fold_index]
        validation_mask = _mask(days, val_start, val_end, device)
        raw_validation = evaluator.evaluate([winner["params"]], validation_mask)[0]
        for day_index, allowed in enumerate(validation_mask.detach().cpu().numpy()):
            if allowed:
                stitched_daily_net_rs[day_index] = raw_validation["daily_net_rs"][day_index]
                stitched_daily_peak_rs[day_index] = raw_validation["daily_peak_rs"][day_index]
                stitched_daily_trough_rs[day_index] = raw_validation["daily_trough_rs"][day_index]
        for key in ("trades", "wins"):
            stitched[key] += int(raw_validation[key])
        for key in ("fees_rs", "net_points", "net_rs"):
            stitched[key] += float(raw_validation[key])
        stitched["max_fold_drawdown_points"] = max(
            stitched["max_fold_drawdown_points"],
            float(raw_validation["max_drawdown_points"]),
        )
    del dataset, resident, evaluator

    stitched["trades"] = int(stitched["trades"])
    stitched["wins"] = int(stitched["wins"])
    stitched["fees_rs"] = round(stitched["fees_rs"], 2)
    stitched["net_points"] = round(stitched["net_points"], 2)
    stitched["net_rs"] = round(stitched["net_rs"], 2)
    stitched["win_rate"] = round(100.0 * stitched["wins"] / stitched["trades"], 2) if stitched["trades"] else 0.0
    equity = 0.0
    global_peak = 0.0
    max_drawdown_rs = 0.0
    for day_index in range(len(days)):
        day_start = equity
        global_peak = max(global_peak, day_start + stitched_daily_peak_rs[day_index])
        max_drawdown_rs = max(
            max_drawdown_rs,
            global_peak - (day_start + stitched_daily_trough_rs[day_index]),
        )
        equity += stitched_daily_net_rs[day_index]
    stitched["max_drawdown_points"] = round(max_drawdown_rs / optimus.LOT_SIZE, 2)
    stitched["score"] = round(
        stitched["net_points"] - args.dd_penalty * stitched["max_drawdown_points"],
        4,
    )
    print(json.dumps({"stitched_oos": stitched, "wall_seconds": round(time.perf_counter() - started, 3)}, indent=2))

    result = {
        "engine": "smart_fib_wide_target_sweep_fixed40",
        "mode": "wide_target_sweep_fixed40",
        "data_root": str(Path(args.data_root).resolve()),
        "start": days[0],
        "end": days[-1],
        "days": len(days),
        "variant": CHAMPION_VARIANT.as_dict(),
        "configs": len(params),
        "batch_size": args.batch_size,
        "brokerage_per_order": 0.0,
        "fixed_cost_per_trade": 40.0,
        "dd_penalty": args.dd_penalty,
        "full_window_ranked": [
            dict(result)
            for result in ranked
        ],
        "folds": fold_results,
        "train_selections": best_train,
        "stitched_oos": stitched,
        "timings": timings,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())