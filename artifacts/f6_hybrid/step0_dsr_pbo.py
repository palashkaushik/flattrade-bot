"""Step 0 — validate the existing 200-trial search with DSR / PBO (sharpebench).

Re-runs every completed trial from optuna_results.csv on its ORIGINAL 2020-2022
window, groups trades into DAILY net-Rs series (zero-filled), and reports:
  - per-trial Sharpe (used for effective trials / E[max SR])
  - Deflated Sharpe Ratio of the CSV winner (N = actual, reported, worst-case)
  - Probability of Backtest Overfitting (CSCV) over the daily-return matrix
  - parity check: grouped per-day totals must match the logged net_rs

Usage:  python artifacts/f6_hybrid/step0_dsr_pbo.py
"""
import csv, json, sys, time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

WORKERS = 8  # fixed pool size (user preference: 8, not engine default ~10)

import numpy as np

from backtest_5y_optimized import load_spot, option_files
import grid_optimize_f6_atr as grid
from sharpebench.sharpebench_py import (
    deflated_sharpe_ratio, probability_of_backtest_overfitting, sharpe_ratio,
    expected_max_sharpe,
)
from f6_hybrid.raw_features import run_factorized_candidates

WINDOW_START, WINDOW_END = "2020-01-01", "2022-12-31"
OUT = Path("artifacts/f6_hybrid/step0_dsr_pbo.json")
PARAM_KEYS = ["s1_k", "s4_k", "atr_period", "atr_sl_mult", "atr_tp_mult",
              "f6_s4_thresh", "f6_s1_thresh", "consec_loss"]


def load_completed_trials():
    rows = []
    with open("optuna_results.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["pruned"] == "1":
                continue
            p = {k: float(r[k]) if "." in r[k] else int(r[k]) for k in PARAM_KEYS}
            p["s1_d"] = 3
            rows.append({"trial": int(r["trial"]), "score": float(r["score"]),
                         "params": p, "net_rs": float(r["net_rs"])})
    seen, uniq = set(), []
    for r in rows:
        key = tuple(sorted((k, v) for k, v in r["params"].items() if k != "s1_d"))
        if key in seen:
            r["dup_of"] = True
            continue
        seen.add(key)
        uniq.append(r)
    return rows, uniq


def daily_series_for(trades, days):
    by_day = defaultdict(float)
    for t in trades:
        by_day[t["date"]] += float(t["pts"])
    arr = np.array([by_day.get(d, 0.0) for d in days], dtype=np.float64)
    return arr, arr.sum()


def main():
    all_rows, distinct = load_completed_trials()
    print(f"completed trials: {len(all_rows)} | distinct param sets: {len(distinct)}")
    distinct.sort(key=lambda r: r["score"], reverse=True)

    spot_all = load_spot()
    files = option_files(WINDOW_START, WINDOW_END)
    days = sorted(set(files.keys()) & set(spot_all.keys()))
    days = [d for d in days if WINDOW_START <= d <= WINDOW_END]
    print(f"window {WINDOW_START}..{WINDOW_END}: {len(days)} days")

    t0 = time.time()
    candidate_params = [row["params"] for row in distinct]
    trade_lists, base_builds, signal_builds = run_factorized_candidates(
        candidate_params, days, files, spot_all, workers=WORKERS
    )
    series, parity_ok, parity_bad = [], 0, []
    for row, trades in zip(distinct, trade_lists):
        arr_pts, _ = daily_series_for(trades, days)
        series.append(arr_pts)
        expected_rs = grid.summarize(trades)["rs"]
        if abs(expected_rs - row["net_rs"]) <= 1.0:
            parity_ok += 1
        else:
            parity_bad.append((row["trial"], expected_rs, round(row["net_rs"], 0)))
    runtime_seconds = time.time() - t0
    print(f"incremental runtime {runtime_seconds:.1f}s | signal builds {signal_builds} | parity ok {parity_ok}")
    pnl = np.array(series) * grid.LOT_SIZE  # rupees per day, same rounding semantics as engine
    mat = pnl.T                     # CSCV matrix: rows=days, cols=configs

    daily_srs = np.array([sharpe_ratio(s) for s in pnl])
    srs_std = float(np.std(daily_srs, ddof=1)) if len(daily_srs) > 1 else 0.0

    winner = distinct[0]
    w = 0  # distinct is sorted by score desc; index 0 is the CSV winner
    win_returns = pnl[w]
    n_obs = len(win_returns)

    dsr = {}
    for label, n in (("actual_distinct", len(distinct)),
                     ("reported_200", len(all_rows)),
                     ("worst_case_15552", 15552)):
        dsr[label] = float(deflated_sharpe_ratio(win_returns, n, srs_std))

    pbo = {}
    for label, mm in (("completed", mat),):
        try:
            pbo[label] = float(probability_of_backtest_overfitting(mm))
        except Exception as e:
            pbo[label] = f"error: {e}"

    emax = {label: float(expected_max_sharpe(srs_std, n))
            for label, n in (("actual", len(distinct)), ("reported", len(all_rows)),
                             ("worst", 15552))}

    report = {
        "window": [WINDOW_START, WINDOW_END],
        "days": len(days),
        "n_configs_evaluated": len(distinct),
        "n_reported_trials": len(all_rows),
        "workers": WORKERS,
        "runtime_seconds": round(runtime_seconds, 1),
        "base_builds": base_builds,
        "signal_builds": signal_builds,
        "parity_ok": parity_ok, "parity_mismatches": parity_bad[:10],
        "winner": {"trial": winner["trial"], "score": winner["score"],
                   "params": winner["params"], "daily_sr": float(daily_srs[w])},
        "trials_sr_std": srs_std,
        "expected_max_sharpe": emax,
        "deflated_sharpe": dsr,
        "pbo_cscv": pbo,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
