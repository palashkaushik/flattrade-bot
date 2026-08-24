"""Gross and fee-adjusted year-wise portfolio report for one fixed champion."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files
from backtest_monthly_ramp import init_worker, process_day
from backtest_walkforward_fees import (
    BROKERAGE_PER_ORDER,
    SLIPPAGE_PTS,
    apply_costs,
    net_stats,
)


def collect_trades(params, start, end, workers, smoke):
    spot = load_spot()
    files = option_files(start, end)
    days = sorted(set(files) & set(spot))
    if smoke:
        days = days[:5]
    tasks = [
        (
            day,
            str(files[day]),
            str(files[days[index - 1]]) if index else "",
        )
        for index, day in enumerate(days)
    ]
    with Pool(max(1, min(8, workers)), initializer=init_worker, initargs=(spot, params)) as pool:
        trades = [trade for result in pool.imap(process_day, tasks) for trade in result]
    return days, trades


def yearly_rows(trades, starting_capital, net=False):
    by_year = {}
    for trade in trades:
        by_year.setdefault(trade["date"][:4], []).append(trade)
    equity = float(starting_capital)
    rows = []
    for year in sorted(by_year):
        group = by_year[year]
        stats = net_stats(group) if net else grid.summarize(group)
        pnl = stats["rs"]
        equity += pnl
        rows.append({
            "year": year,
            "trades": stats["trades"],
            "win_rate": round(stats["wr"], 2),
            "points": round(stats["pts"], 2),
            "pnl_rs": pnl,
            "profit_factor": round(stats["pf"], 4),
            "fees_rs": round(stats.get("fees", 0.0), 2),
            "end_one_lot_equity_rs": round(equity, 2),
        })
    return rows, equity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--capital", type=float, default=20000.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/champion_portfolio_report.json")
    args = parser.parse_args()

    params = json.loads(Path(args.params_file).read_text(encoding="utf-8"))["params"]
    days, gross_trades = collect_trades(params, args.start, args.end, args.workers, args.smoke)
    net_trades = copy.deepcopy(gross_trades)
    apply_costs(net_trades, BROKERAGE_PER_ORDER, slippage_pts=SLIPPAGE_PTS)

    gross_years, gross_equity = yearly_rows(gross_trades, args.capital, net=False)
    net_years, net_equity = yearly_rows(net_trades, args.capital, net=True)
    result = {
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "workers": max(1, min(8, args.workers)),
        "smoke": args.smoke,
        "starting_capital_rs": args.capital,
        "cost_model": {
            "slippage_points_per_side": SLIPPAGE_PTS,
            "brokerage_per_order": BROKERAGE_PER_ORDER,
        },
        "params": params,
        "gross": {
            "trades": len(gross_trades),
            "stats": grid.summarize(gross_trades),
            "yearly": gross_years,
            "final_one_lot_equity_rs": gross_equity,
        },
        "cost_adjusted": {
            "trades": len(net_trades),
            "stats": net_stats(net_trades),
            "yearly": net_years,
            "final_one_lot_equity_rs": net_equity,
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
