"""Five-day reference vs incremental signal-cache benchmark."""

import json
from itertools import product
import os
import sys
import time
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files
from f6_hybrid.incremental import run_incremental_candidates
from f6_hybrid.raw_features import run_factorized_candidates


WORKERS = 8


def run_mode(candidates, days, files, spot_all, columnar_dir=None):
    previous = os.environ.get("F6_COLUMNAR_CACHE_DIR")
    if columnar_dir is None:
        os.environ.pop("F6_COLUMNAR_CACHE_DIR", None)
    else:
        os.environ["F6_COLUMNAR_CACHE_DIR"] = str(columnar_dir)
    reference_start = time.perf_counter()
    try:
        with grid.Pool(
            processes=WORKERS,
            initializer=grid.init_worker_local,
            initargs=(spot_all,),
        ) as pool:
            reference = [
                grid.run_days(pool, params, days, files, spot_all)
                for params in candidates
            ]
    finally:
        if previous is None:
            os.environ.pop("F6_COLUMNAR_CACHE_DIR", None)
        else:
            os.environ["F6_COLUMNAR_CACHE_DIR"] = previous
    reference_seconds = time.perf_counter() - reference_start

    previous = os.environ.get("F6_COLUMNAR_CACHE_DIR")
    if columnar_dir is None:
        os.environ.pop("F6_COLUMNAR_CACHE_DIR", None)
    else:
        os.environ["F6_COLUMNAR_CACHE_DIR"] = str(columnar_dir)
    incremental_start = time.perf_counter()
    try:
        optimized, signal_builds = run_incremental_candidates(
            candidates, days, files, spot_all, workers=WORKERS
        )
    finally:
        if previous is None:
            os.environ.pop("F6_COLUMNAR_CACHE_DIR", None)
        else:
            os.environ["F6_COLUMNAR_CACHE_DIR"] = previous
    incremental_seconds = time.perf_counter() - incremental_start

    previous = os.environ.get("F6_COLUMNAR_CACHE_DIR")
    if columnar_dir is None:
        os.environ.pop("F6_COLUMNAR_CACHE_DIR", None)
    else:
        os.environ["F6_COLUMNAR_CACHE_DIR"] = str(columnar_dir)
    factorized_start = time.perf_counter()
    try:
        factorized, base_builds, factorized_signal_builds = run_factorized_candidates(
            candidates, days, files, spot_all, workers=WORKERS
        )
    finally:
        if previous is None:
            os.environ.pop("F6_COLUMNAR_CACHE_DIR", None)
        else:
            os.environ["F6_COLUMNAR_CACHE_DIR"] = previous
    factorized_seconds = time.perf_counter() - factorized_start

    return {
        "reference_seconds": round(reference_seconds, 3),
        "incremental_seconds": round(incremental_seconds, 3),
        "speedup": round(reference_seconds / incremental_seconds, 3)
        if incremental_seconds
        else None,
        "signal_builds": signal_builds,
        "factorized": {
            "seconds": round(factorized_seconds, 3),
            "speedup": round(reference_seconds / factorized_seconds, 3)
            if factorized_seconds
            else None,
            "base_builds": base_builds,
            "signal_builds": factorized_signal_builds,
            "summaries_match": [
                grid.summarize(reference[index]) == grid.summarize(factorized[index])
                for index in range(len(candidates))
            ],
        },
        "summaries_match": [
            grid.summarize(reference[index]) == grid.summarize(optimized[index])
            for index in range(len(candidates))
        ],
    }


def main():
    spot_all = load_spot()
    files = option_files("2020-01-01", "2020-01-07")
    days = sorted(set(files) & set(spot_all))[:5]
    first = dict(grid.CHAMPION)
    candidates = []
    for sl_mult, tp_mult, consecutive_loss in product(
        (1.5, 2.0, 2.5, 3.0), (3.0, 4.0, 5.0, 6.0), (4, 6, 8)
    ):
        candidate = dict(first)
        candidate.update(
            atr_sl_mult=sl_mult,
            atr_tp_mult=tp_mult,
            consec_loss=consecutive_loss,
        )
        candidates.append(candidate)

    result = {
        "days": days,
        "workers": WORKERS,
        "candidates": len(candidates),
        "expected_signal_builds": len(days),
        "csv": run_mode(candidates, days, files, spot_all),
        "parquet": run_mode(
            candidates,
            days,
            files,
            spot_all,
            Path("artifacts/f6_hybrid/columnar_cache").resolve(),
        ),
    }
    output = Path("artifacts/f6_hybrid/benchmark_incremental.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
