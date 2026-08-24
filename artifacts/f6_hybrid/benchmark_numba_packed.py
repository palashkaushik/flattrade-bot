"""Five-day reference vs factorized vs numba-packed execution benchmark."""

import json
import sys
import time
from itertools import product
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files
from f6_hybrid.packed import _NUMBA_OK, run_packed_candidates
from f6_hybrid.raw_features import run_factorized_candidates

WORKERS = 8


def candidate_sweep():
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
    return candidates


def run_mode(candidates, days, files, spot_all):
    reference_start = time.perf_counter()
    factorized, base_builds, signal_builds = run_factorized_candidates(
        candidates, days, files, spot_all, workers=WORKERS
    )
    reference_seconds = time.perf_counter() - reference_start

    warmup_start = time.perf_counter()
    run_packed_candidates(candidates, days, files, spot_all, workers=WORKERS)
    warmup_seconds = time.perf_counter() - warmup_start

    serial_start = time.perf_counter()
    packed_serial, _, _ = run_packed_candidates(
        candidates, days, files, spot_all, workers=WORKERS, parallel=False
    )
    serial_seconds = time.perf_counter() - serial_start

    parallel_start = time.perf_counter()
    packed_parallel, _, _ = run_packed_candidates(
        candidates, days, files, spot_all, workers=WORKERS, parallel=True
    )
    parallel_seconds = time.perf_counter() - parallel_start

    exact_matches = [
        packed_serial[index] == factorized[index]
        for index in range(len(candidates))
    ]
    parallel_matches = [
        packed_parallel[index] == factorized[index]
        for index in range(len(candidates))
    ]

    return {
        "factorized_seconds": round(reference_seconds, 3),
        "factorized_speedup": round(37.604 / reference_seconds, 3)
        if reference_seconds
        else None,
        "packed_warmup_seconds": round(warmup_seconds, 3),
        "packed_serial_seconds": round(serial_seconds, 3),
        "packed_serial_speedup": round(reference_seconds / serial_seconds, 3)
        if serial_seconds
        else None,
        "packed_parallel_seconds": round(parallel_seconds, 3),
        "packed_parallel_speedup": round(reference_seconds / parallel_seconds, 3)
        if parallel_seconds
        else None,
        "parity_serial_ok": all(exact_matches),
        "parity_parallel_ok": all(parallel_matches),
        "parity_mismatch_indices": [
            index for index, ok in enumerate(exact_matches) if not ok
        ],
        "base_builds": base_builds,
        "signal_builds": signal_builds,
    }


def main():
    if not _NUMBA_OK:
        print("SKIPPED: numba is not installed")
        return
    spot_all = load_spot()
    files = option_files("2020-01-01", "2020-01-07")
    days = sorted(set(files) & set(spot_all))[:5]
    candidates = candidate_sweep()
    result = {
        "days": days,
        "workers": WORKERS,
        "candidates": len(candidates),
        "csv": run_mode(candidates, days, files, spot_all),
    }
    output = Path("artifacts/f6_hybrid/benchmark_numba_packed.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
