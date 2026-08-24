"""GPU-only expanding walk-forward search for cached Smart Fib variants."""

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

from artifacts.f6_hybrid import smart_fib_optimus_grid_gpu as grid
from artifacts.f6_hybrid import smart_fib_optimus_gpu as optimus


FOLDS = (
    ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2020-01-01", "2024-12-31", "2025-01-01", "2026-05-05"),
)


def _score(result, penalty):
    return float(result["net_points"]) - penalty * float(result["max_drawdown_points"])


def _with_params(result, params, variant):
    output = dict(result)
    output.pop("daily_trades", None)
    output.pop("daily_net_points", None)
    output.pop("daily_net_rs", None)
    output.pop("daily_drawdown_rs", None)
    output.pop("daily_peak_rs", None)
    output.pop("daily_trough_rs", None)
    output["params"] = dict(params)
    output["variant"] = variant.as_dict()
    return output


def _load_gpu_variant(
    adapter,
    days,
    variant,
    cache_dir,
    device,
    batch_size,
    brokerage_per_order,
    fixed_cost_per_trade,
):
    dataset, source, path = grid._cached_variant_dataset(
        adapter.data_root, days, variant, cache_dir,
    )
    if dataset is None:
        raise RuntimeError(
            f"missing full GPU tensor cache for {variant.variant_id}; "
            "refusing to fall back to CPU extraction"
        )
    print(f"[WF CACHE] {variant.variant_id} source={source} path={path}", flush=True)
    resident = grid.to_gpu_dataset(dataset, device)
    evaluator = optimus.GpuEvaluator(
        resident.engine,
        batch_size,
        brokerage_per_order=brokerage_per_order,
        fixed_cost_per_trade=fixed_cost_per_trade,
    )
    return dataset, resident, evaluator


def _mask(days, start, end, device):
    return torch.tensor(
        [start <= day <= end for day in days],
        dtype=torch.bool,
        device=device,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=r"C:\Users\user\Desktop\nifty50 data")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--brokerage-per-order", type=float, default=20.0)
    parser.add_argument("--fixed-cost-per-trade", type=float, default=40.0)
    parser.add_argument("--dd-penalty", type=float, default=0.50)
    parser.add_argument("--tensor-cache-dir", default="artifacts/f6_hybrid/smart_fib_grid_cache_full_float64")
    parser.add_argument("--output", default="artifacts/f6_hybrid/smart_fib_optimus_wf_gpu_2020_2026.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")

    device = torch.device("cuda")
    adapter = grid.PolarsHistoricalDataAdapter(
        args.data_root,
        start=args.start,
        end=args.end,
    )
    days = adapter.available_days(args.start, args.end)
    if len(days) < 500:
        raise SystemExit("full 2020-2026 data is required for walk-forward")
    variants = list(grid.STAGED_VARIANTS)
    params = grid._grid_params(
        (0.0, 0.236, 0.29, 0.382, 0.5),
        (0.0, 0.236),
        (5.0, 10.0, 15.0),
        (1.13, 1.155, 1.25, 1.382, 1.618),
    )
    cache_dir = Path(args.tensor_cache_dir)
    best_train = [None] * len(FOLDS)
    timings = []
    started = time.perf_counter()

    # Each variant is loaded once, then evaluated against every train mask.
    for variant_index, variant in enumerate(variants, start=1):
        dataset, resident, evaluator = _load_gpu_variant(
            adapter, days, variant, cache_dir, device, args.batch_size,
            args.brokerage_per_order,
            args.fixed_cost_per_trade,
        )
        for fold_index, (_, train_end, val_start, val_end) in enumerate(FOLDS):
            train_mask = _mask(days, args.start, train_end, device)
            fold_started = time.perf_counter()
            results = evaluator.evaluate(params, train_mask)
            timings.append({
                "variant": variant.variant_id,
                "fold": fold_index + 1,
                "cuda_ms": evaluator.last_cuda_ms,
                "wall_seconds": time.perf_counter() - fold_started,
            })
            candidates = [
                _with_params(result, params[index], variant)
                for index, result in enumerate(results)
            ]
            winner = max(
                candidates,
                key=lambda result: (
                    _score(result, args.dd_penalty),
                    float(result["net_points"]),
                    -float(result["max_drawdown_points"]),
                ),
            )
            winner["train_score"] = round(_score(winner, args.dd_penalty), 4)
            winner["fold"] = fold_index + 1
            winner["validation_start"] = val_start
            winner["validation_end"] = val_end
            if best_train[fold_index] is None or winner["train_score"] > best_train[fold_index]["train_score"]:
                best_train[fold_index] = winner
            print(
                f"[WF TRAIN] variant={variant_index}/{len(variants)} fold={fold_index + 1} "
                f"score={winner['train_score']:+.2f} net={winner['net_points']:+.2f} "
                f"DD={winner['max_drawdown_points']:.2f} params={winner['params']}",
                flush=True,
            )
        del dataset, resident, evaluator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_results = []
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
    for fold_index, (_, _, val_start, val_end) in enumerate(FOLDS):
        winner = best_train[fold_index]
        variant = next(item for item in variants if item.variant_id == winner["variant"]["variant_id"])
        dataset, resident, evaluator = _load_gpu_variant(
            adapter, days, variant, cache_dir, device, args.batch_size,
            args.brokerage_per_order,
            args.fixed_cost_per_trade,
        )
        validation_mask = _mask(days, val_start, val_end, device)
        raw_validation = evaluator.evaluate([winner["params"]], validation_mask)[0]
        for day_index, allowed in enumerate(validation_mask.detach().cpu().numpy()):
            if allowed:
                stitched_daily_net_rs[day_index] = raw_validation["daily_net_rs"][day_index]
                stitched_daily_peak_rs[day_index] = raw_validation["daily_peak_rs"][day_index]
                stitched_daily_trough_rs[day_index] = raw_validation["daily_trough_rs"][day_index]
        validation = _with_params(raw_validation, winner["params"], variant)
        validation["fold"] = fold_index + 1
        validation["train_selection"] = winner
        validation["validation_start"] = val_start
        validation["validation_end"] = val_end
        fold_results.append(validation)
        for key in ("trades", "wins"):
            stitched[key] += int(validation[key])
        for key in ("fees_rs", "net_points", "net_rs"):
            stitched[key] += float(validation[key])
        stitched["max_fold_drawdown_points"] = max(
            stitched["max_fold_drawdown_points"],
            float(validation["max_drawdown_points"]),
        )
        print(
            f"[WF VALIDATE] fold={fold_index + 1} {val_start}..{val_end} "
            f"trades={validation['trades']} net={validation['net_points']:+.2f} "
            f"DD={validation['max_drawdown_points']:.2f} params={winner['params']}",
            flush=True,
        )
        del dataset, resident, evaluator
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
    result = {
        "engine": "smart_fib_optimus_grid_gpu",
        "mode": "expanding_walk_forward",
        "data_root": str(Path(args.data_root).resolve()),
        "start": days[0],
        "end": days[-1],
        "days": len(days),
        "variants": [variant.as_dict() for variant in variants],
        "configs_per_variant": len(params),
        "batch_size": args.batch_size,
        "brokerage_per_order": args.brokerage_per_order,
        "fixed_cost_per_trade": args.fixed_cost_per_trade,
        "dd_penalty": args.dd_penalty,
        "folds": fold_results,
        "train_selections": best_train,
        "stitched_oos": stitched,
        "timings": timings,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"stitched_oos": stitched, "wall_seconds": result["wall_seconds"]}, indent=2))
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
