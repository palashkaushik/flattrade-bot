"""Smoke test for the numba packed path before any timed benchmark."""

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files
from f6_hybrid.packed import _NUMBA_OK, run_packed_candidates
from f6_hybrid.raw_features import run_factorized_candidates

WORKERS = 2


def main():
    if not _NUMBA_OK:
        print("SMOKE FAIL: numba not installed")
        return 1
    spot_all = load_spot()
    files = option_files("2020-01-01", "2020-01-07")
    days = sorted(set(files) & set(spot_all))[:5]
    candidates = [dict(grid.CHAMPION)]
    for sl_mult, tp_mult in ((1.5, 3.0), (3.0, 6.0)):
        candidate = dict(grid.CHAMPION)
        candidate.update(atr_sl_mult=sl_mult, atr_tp_mult=tp_mult)
        candidates.append(candidate)

    factorized, fb, fs = run_factorized_candidates(
        candidates, days, files, spot_all, workers=WORKERS
    )
    for index, trades in enumerate(factorized):
        count = len(trades)
        print(f"factorized[{index}] trades={count}")
        if not (1 <= count <= 50):
            print(f"SMOKE FAIL: suspicious trade count {count}")
            return 1

    packed, pb, ps = run_packed_candidates(
        candidates, days, files, spot_all, workers=WORKERS, parallel=False
    )
    exact = [packed[index] == factorized[index] for index in range(len(candidates))]
    print("packed base_builds", pb, "signal_builds", ps, "parity", exact)
    if not all(exact):
        for index, ok in enumerate(exact):
            if ok:
                continue
            a, b = factorized[index], packed[index]
            print(f"mismatch[{index}]: factorized={len(a)} packed={len(b)}")
            for ta, tb in zip(a, b):
                if ta != tb:
                    print("  ref:", ta)
                    print("  pck:", tb)
                    break
        print("SMOKE FAIL: parity mismatch")
        return 1
    print("SMOKE TEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())