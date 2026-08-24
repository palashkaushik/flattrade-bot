"""Sweep legacy ATR SL multipliers with fixed TP15 on the causal engine."""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import causal_live_parity_research as engine
from backtest_5y_optimized import load_spot, option_files


def max_drawdown_points(trades: list[dict]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for trade in sorted(trades, key=lambda item: (item["date"], item["exit_min"])):
        equity += trade["rs_net"] / engine.LOT_SIZE
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return round(drawdown, 2)


def run_factor(
    pool,
    factor: float,
    days: list[str],
    all_days: list[str],
    files: dict[str, str],
    base_params: dict,
    mode: str,
) -> dict:
    params = dict(base_params)
    params["atr_sl_mult"] = factor
    positions = {day: index for index, day in enumerate(all_days)}
    tasks = [
        (
            day,
            str(files[day]),
            str(files[all_days[positions[day] - 1]]) if positions[day] else "",
            params,
            mode,
            True,
            None,
            None,
            15.0,
            "static",
            "legacy_high_break",
        )
        for day in days
    ]
    trades = []
    for result in pool.imap(engine.process_day, tasks):
        trades.extend(result)
    summary = engine.stats(trades, day_count=len(days))
    summary.update({
        "atr_sl_mult": factor,
        "max_drawdown_points": max_drawdown_points(trades),
        "consec_loss": params["consec_loss"],
        "fixed_tp_points": 15.0,
        "breakout_mode": "legacy_high_break",
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params-file",
        default="artifacts/f6_hybrid/backtest_live_atr_3_6_params.json",
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mode", default="previous_divergence")
    parser.add_argument(
        "--factors",
        default="1.0,1.5,2.0,2.5,3.0",
        help="Comma-separated ATR SL multipliers",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        default="artifacts/f6_hybrid/legacy_atr_sl_fixed_tp15_sweep.json",
    )
    args = parser.parse_args()

    base_params = json.loads(Path(args.params_file).read_text(encoding="utf-8"))["params"]
    factors = [float(value) for value in args.factors.split(",") if value.strip()]
    spot = load_spot()
    files = option_files(args.start, args.end)
    all_days = sorted(set(files) & set(spot))
    days = all_days[:5] if args.smoke else all_days

    with Pool(max(1, min(8, args.workers)), initializer=engine.init_worker, initargs=(spot,)) as pool:
        results = [run_factor(pool, factor, days, all_days, files, base_params, args.mode) for factor in factors]

    results.sort(key=lambda item: item["atr_sl_mult"])
    ranked = sorted(results, key=lambda item: item["net_points"], reverse=True)
    output_data = {
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "smoke": args.smoke,
        "mode": args.mode,
        "params_base": base_params,
        "fixed_tp_points": 15.0,
        "consec_loss": base_params["consec_loss"],
        "breakout_mode": "legacy_high_break",
        "costs_enabled": True,
        "results": results,
        "best_by_net_points": ranked[0] if ranked else None,
    }
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(json.dumps(output_data, indent=2))
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
