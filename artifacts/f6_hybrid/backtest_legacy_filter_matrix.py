"""Run legacy EMA20 and permissive Elder filters across divergence modes."""

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


def run_variant(pool, days, all_days, files, params, divergence_mode, entry_filter):
    positions = {day: index for index, day in enumerate(all_days)}
    tasks = [
        (
            day,
            str(files[day]),
            str(files[all_days[positions[day] - 1]]) if positions[day] else "",
            params,
            divergence_mode,
            True,
            None,
            None,
            None,
            "static",
            "legacy_high_break",
            None,
            "pinbar",
            True,
            True,
            entry_filter,
        )
        for day in days
    ]
    trades = []
    for result in pool.imap(engine.process_day, tasks):
        trades.extend(result)
    summary = engine.stats(trades, day_count=len(days))
    summary.update({
        "divergence_mode": divergence_mode,
        "entry_filter": entry_filter,
        "signal_mode": "pinbar",
        "exit_mode": "atr_sl_tp",
        "reverse_regime_enabled": True,
        "daily_caps_enabled": False,
        "max_drawdown_points": max_drawdown_points(trades),
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-file", default="artifacts/f6_hybrid/backtest_live_atr_3_6_params.json")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/legacy_filter_matrix.json")
    args = parser.parse_args()

    params = json.loads(Path(args.params_file).read_text(encoding="utf-8"))["params"]
    params["consec_loss"] = 4
    spot = load_spot()
    files = option_files(args.start, args.end)
    all_days = sorted(set(files) & set(spot))
    days = all_days[:5] if args.smoke else all_days
    divergence_modes = ("no_divergence", "new_divergence", "previous_divergence")
    entry_filters = ("ema20", "elder_permissive")

    with Pool(max(1, min(8, args.workers)), initializer=engine.init_worker, initargs=(spot,)) as pool:
        results = [
            run_variant(pool, days, all_days, files, params, divergence_mode, entry_filter)
            for entry_filter in entry_filters
            for divergence_mode in divergence_modes
        ]

    output_data = {
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "smoke": args.smoke,
        "params": params,
        "entry_filters": entry_filters,
        "divergence_modes": divergence_modes,
        "exit_mode": "atr_sl_tp",
        "fixed_sl": None,
        "fixed_tp": None,
        "reverse_regime_enabled": True,
        "daily_caps_enabled": False,
        "results": results,
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
