"""Fixed-champion, fee-adjusted walk-forward run through 2026.

The champion is frozen. Each OOS year uses only its preceding training window
for warmup context; no parameters are re-fit in this command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files
from backtest_monthly_ramp import (
    CHAMPION_PARAMS,
    MARGIN_PER_LOT,
    apply_monthly_ramp,
    init_worker,
    print_ramp_table,
    print_yearly_ramp,
    process_day,
)
from backtest_walkforward_fees import (
    BROKERAGE_PER_ORDER,
    apply_costs,
    net_stats,
    summarize_net,
)


def load_params(path):
    if not path:
        return dict(CHAMPION_PARAMS)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "best_full_fidelity" in data:
        data = data["best_full_fidelity"][0]["params"]
    elif "params" in data:
        data = data["params"]
    params = dict(CHAMPION_PARAMS)
    params.update(data)
    params["s1_d"] = 3
    return params


FOLDS = [
    {"is_start": "2020", "is_end": "2022", "oos_year": "2023"},
    {"is_start": "2021", "is_end": "2023", "oos_year": "2024"},
    {"is_start": "2022", "is_end": "2024", "oos_year": "2025"},
    {"is_start": "2023", "is_end": "2025", "oos_year": "2026"},
]


def select_fold_days(days: list[str], fold: dict) -> dict[str, list[str]]:
    oos = [day for day in days if day.startswith(fold["oos_year"])]
    is_days = [
        day for day in days
        if fold["is_start"] <= day[:4] <= fold["is_end"]
    ]
    warmup = [is_days[-1]] if is_days else []
    return {"is": is_days, "oos": oos, "warmup": warmup}


def run_fold(pool, fold: dict, days: list[str], files: dict[str, str], smoke: bool) -> dict:
    selected = select_fold_days(days, fold)
    oos = selected["oos"][:5] if smoke else selected["oos"]
    if not oos:
        return {"fold": fold, "skipped": True, "reason": "no OOS data"}

    all_days = days
    index = {day: i for i, day in enumerate(all_days)}
    tasks = []
    for day in oos:
        previous = all_days[index[day] - 1] if index[day] else None
        tasks.append((day, str(files[day]), str(files[previous]) if previous else ""))

    started = time.time()
    trades = []
    for result in pool.imap(process_day, tasks):
        trades.extend(result)
    raw_stats = grid.summarize(trades)
    apply_costs(trades, BROKERAGE_PER_ORDER)
    net = net_stats(trades)
    return {
        "fold": fold,
        "is_days": len(selected["is"]),
        "warmup": selected["warmup"],
        "oos_days": len(oos),
        "raw": raw_stats,
        "net": net,
        "seconds": round(time.time() - started, 3),
        "smoke": smoke,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--capital", type=float, default=20000.0)
    parser.add_argument("--increment", type=float, default=40000.0)
    parser.add_argument("--params-file")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        default="artifacts/f6_hybrid/backtest_champion_walkforward_2020_2026.json",
    )
    args = parser.parse_args()
    workers = max(1, min(8, args.workers))
    params = load_params(args.params_file)

    spot_all = load_spot()
    files = option_files(args.start, args.end)
    days = sorted(set(files) & set(spot_all))
    if not days:
        raise SystemExit("no overlapping option and spot days")

    print(
        f"CHAMPION WALK-FORWARD | {days[0]}..{days[-1]} | "
        f"workers={workers} | smoke={args.smoke} | params={params}",
        flush=True,
    )
    started = time.time()
    with Pool(processes=workers, initializer=init_worker, initargs=(spot_all, params)) as pool:
        folds = [run_fold(pool, fold, days, files, args.smoke) for fold in FOLDS]

    valid = [row for row in folds if not row.get("skipped")]
    stitched_net = {
        "trades": sum(row["net"]["trades"] for row in valid),
        "pts": round(sum(row["net"]["pts"] for row in valid), 2),
        "rs": round(sum(row["net"]["rs"] for row in valid)),
        "fees": round(sum(row["net"]["fees"] for row in valid), 2),
    }
    summary = {
        "mode": "fixed_champion_walk_forward",
        "start": args.start,
        "end": args.end,
        "workers": workers,
        "smoke": args.smoke,
        "params": params,
        "folds": folds,
        "stitched_net_partial": stitched_net,
        "seconds": round(time.time() - started, 3),
    }
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    for row in valid:
        print(
            f"OOS {row['fold']['oos_year']} | days={row['oos_days']} | "
            f"{summarize_net(row['net'])} | {row['seconds']:.1f}s",
            flush=True,
        )
    print(f"JSON: {output}", flush=True)

    if args.smoke:
        return

    # The fixed one-lot stitched result is the primary metric. The ramp is a
    # separate money-management scenario and must not replace it.
    print(f"STITCHED NET | trades={stitched_net['trades']} | "
          f"pts={stitched_net['pts']:+.2f} | Rs={stitched_net['rs']:+,} | "
          f"fees={stitched_net['fees']:+,.2f}")


if __name__ == "__main__":
    main()
