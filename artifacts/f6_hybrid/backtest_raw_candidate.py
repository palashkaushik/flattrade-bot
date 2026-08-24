"""Raw no-cost backtest for one explicit optimized F6 parameter set."""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files
from backtest_blind_2024_2026 import load_params
from backtest_monthly_ramp import init_worker, process_day


def run(params, start, end, workers, smoke):
    spot = load_spot()
    files = option_files(start, end)
    days = sorted(set(files) & set(spot))
    if smoke:
        days = days[:5]
    if not days:
        raise SystemExit("no overlapping option and spot days")
    tasks = [
        (
            day,
            str(files[day]),
            str(files[days[index - 1]]) if index else "",
        )
        for index, day in enumerate(days)
    ]
    with Pool(
        processes=max(1, min(8, workers)),
        initializer=init_worker,
        initargs=(spot, params),
    ) as pool:
        trades = [trade for result in pool.imap(process_day, tasks) for trade in result]
    return days, grid.summarize(trades)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-divergence", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/backtest_raw_candidate.json")
    args = parser.parse_args()

    params = load_params(args.params_file)
    params["use_divergence"] = not args.no_divergence
    days, stats = run(params, args.start, args.end, args.workers, args.smoke)
    result = {
        "costs_applied": False,
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "workers": max(1, min(8, args.workers)),
        "smoke": args.smoke,
        "params": params,
        "stats": stats,
    }
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"RAW BACKTEST | costs=False | days={len(days)} | workers={result['workers']} | params={params}")
    print(f"{stats}")
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
