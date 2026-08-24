"""Research comparison using the current grid runner instead of the ramp fork."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-divergence", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/grid_runner_comparison.json")
    args = parser.parse_args()
    params = json.loads(Path(args.params_file).read_text(encoding="utf-8"))["params"]
    params["use_divergence"] = not args.no_divergence
    spot = load_spot()
    files = option_files(args.start, args.end)
    days = sorted(set(files) & set(spot))
    if args.smoke:
        days = days[:5]
    with Pool(max(1, min(8, args.workers)), initializer=grid.init_worker_local, initargs=(spot,)) as pool:
        trades = grid.run_days(pool, params, days, files, spot)
    result = {
        "runner": "grid_optimize_f6_atr.run_days",
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "smoke": args.smoke,
        "params": params,
        "stats": grid.summarize(trades),
        "yearly": {
            year: grid.summarize([trade for trade in trades if trade["date"].startswith(year)])
            for year in sorted({trade["date"][:4] for trade in trades})
        },
    }
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
