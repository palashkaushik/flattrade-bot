"""Cost-aware Optuna refit per walk-forward window through 2026."""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files
from backtest_monthly_ramp import init_worker, process_day
from backtest_walkforward_fees import apply_costs, net_stats, summarize_net
from artifacts.f6_hybrid.backtest_champion_walkforward_2020_2026 import (
    FOLDS,
    select_fold_days,
)
from artifacts.f6_hybrid.reordered_search import run_search


def fixed_threshold_space() -> dict[str, list]:
    return {
        **grid.SEARCH_SPACE,
        "f6_s4_thresh": [79.5],
        "f6_s1_thresh": [20.5],
    }


def evaluate_oos(params, fold, days, files, spot_all, workers, smoke):
    selected = select_fold_days(days, fold)
    oos_days = selected["oos"][:5] if smoke else selected["oos"]
    index = {day: i for i, day in enumerate(days)}
    tasks = []
    for day in oos_days:
        previous = days[index[day] - 1] if index[day] else None
        tasks.append((day, str(files[day]), str(files[previous]) if previous else ""))
    trades = []
    with Pool(processes=workers, initializer=init_worker, initargs=(spot_all, params)) as pool:
        for result in pool.imap(process_day, tasks):
            trades.extend(result)
    raw = grid.summarize(trades)
    apply_costs(trades, 0.0)
    return {
        "oos_days": len(oos_days),
        "raw": raw,
        "net": net_stats(trades),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        default="artifacts/f6_hybrid/optuna_walkforward_net_2020_2026.json",
    )
    args = parser.parse_args()
    workers = max(1, min(8, args.workers))
    trials = 2 if args.smoke else args.trials

    spot_all = load_spot()
    files = option_files(args.start, args.end)
    days = sorted(set(files) & set(spot_all))
    if not days:
        raise SystemExit("no overlapping option and spot days")

    results = []
    started = time.time()
    for fold in FOLDS:
        selected = select_fold_days(days, fold)
        is_days = selected["is"]
        if not is_days or not selected["oos"]:
            results.append({"fold": fold, "skipped": True})
            continue
        search = run_search(
            is_days,
            files,
            spot_all,
            n_trials=trials,
            batch_size=args.batch_size,
            workers=workers,
            output_prefix=f"artifacts/f6_hybrid/wf_search_{fold['oos_year']}",
            search_space=fixed_threshold_space(),
            cost_aware=True,
            smoke=False,
        )
        best = search["best_full_fidelity"][0]["params"]
        oos = evaluate_oos(best, fold, days, files, spot_all, workers, args.smoke)
        row = {
            "fold": fold,
            "is_days": len(is_days),
            "best": best,
            "search": {
                "trials": search["trial_count"],
                "completed": search["completed_count"],
                "pruned": search["pruned_count"],
                "seconds": search["wall_seconds"],
            },
            "oos": oos,
            "smoke": args.smoke,
        }
        results.append(row)
        print(
            f"OOS {fold['oos_year']} | best={best} | "
            f"{summarize_net(oos['net'])}",
            flush=True,
        )

    stitched = [row["oos"]["net"] for row in results if not row.get("skipped")]
    stitched_summary = {
        "trades": sum(row["trades"] for row in stitched),
        "pts": round(sum(row["pts"] for row in stitched), 2),
        "rs": round(sum(row["rs"] for row in stitched)),
        "fees": round(sum(row["fees"] for row in stitched), 2),
    }
    summary = {
        "mode": "cost_aware_optuna_walk_forward",
        "start": args.start,
        "end": args.end,
        "workers": workers,
        "trials_per_fold": trials,
        "fixed_thresholds": {"f6_s4_thresh": 79.5, "f6_s1_thresh": 20.5},
        "folds": results,
        "stitched_net": stitched_summary,
        "seconds": round(time.time() - started, 3),
    }
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"STITCHED NET: {stitched_summary}", flush=True)
    print(f"JSON: {output}", flush=True)


if __name__ == "__main__":
    main()
