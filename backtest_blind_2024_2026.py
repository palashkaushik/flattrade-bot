"""Fee-adjusted champion run on the newly available 2024-2026 option data."""

import argparse
import json
import time
from multiprocessing import Pool

from backtest_5y_optimized import load_spot, option_files
from backtest_monthly_ramp import (
    CHAMPION_PARAMS,
    MARGIN_PER_LOT,
    WORKERS,
    apply_monthly_ramp,
    init_worker,
    print_ramp_table,
    print_yearly_ramp,
    run_days,
)
from backtest_walkforward_fees import (
    BROKERAGE_PER_ORDER,
    EXCHANGE_PCT,
    GST_PCT,
    SEBI_PCT,
    SLIPPAGE_PTS,
    STAMP_PCT,
    STT_PCT,
    apply_costs,
    net_stats,
    summarize_net,
)


def select_days(files, spot, start_date, end_date):
    """Return dates with both option and spot data inside the requested range."""
    return sorted(
        day
        for day in set(files) & set(spot)
        if start_date <= day <= end_date
    )


def print_period_stats(label, trades):
    print(f"\n--- {label} ({len(trades):,} trades)")
    print(summarize_net(net_stats(trades)))


def load_params(path):
    if not path:
        return dict(CHAMPION_PARAMS)
    data = json.loads(open(path, encoding="utf-8").read())
    if "best_full_fidelity" in data:
        data = data["best_full_fidelity"][0]["params"]
    elif "params" in data:
        data = data["params"]
    params = dict(CHAMPION_PARAMS)
    params.update(data)
    params["s1_d"] = 3
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Fee-adjusted blind backtest for the fixed champion"
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--blind-start", default="2025-01-01")
    parser.add_argument("--capital", type=float, default=20000.0)
    parser.add_argument("--increment", type=float, default=40000.0)
    parser.add_argument("--brokerage", type=float, default=BROKERAGE_PER_ORDER)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--params-file")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.workers = max(1, min(8, args.workers))
    params = load_params(args.params_file)

    spot = load_spot()
    files = option_files(args.start, args.end)
    days = select_days(files, spot, args.start, args.end)
    if args.smoke:
        days = days[:5]
    if not days:
        parser.error("No dates have both option and spot data in the requested range")

    label = "SMOKE TEST - 5 DAYS" if args.smoke else "FULL BLIND RUN"
    print(
        f"=== {label} ({days[0]}..{days[-1]}) | champion ATR F6 | "
        f"slippage {SLIPPAGE_PTS} pts/side | STT {STT_PCT}% | "
        f"exchange {EXCHANGE_PCT}% | SEBI {SEBI_PCT}% | stamp {STAMP_PCT}% | "
        f"GST {GST_PCT}% | brokerage {args.brokerage}/order ===",
        flush=True,
    )
    print(
        f"Capital Rs {args.capital:,.0f} | ramp Rs {args.increment:,.0f}/lot | "
        f"workers {args.workers} | params {params}",
        flush=True,
    )

    started = time.time()
    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(spot, params),
    ) as pool:
        trades = run_days(pool, days, files)
    elapsed = time.time() - started

    apply_costs(trades, args.brokerage)
    print(f"\nEngine completed in {elapsed:.1f}s")
    print_period_stats("REQUESTED PERIOD", trades)

    for year in sorted({trade["date"][:4] for trade in trades}):
        year_trades = [trade for trade in trades if trade["date"].startswith(year)]
        print_period_stats(year, year_trades)

    blind_trades = [trade for trade in trades if trade["date"] >= args.blind_start]
    print_period_stats(
        f"TRUE BLIND PERIOD ({args.blind_start} onward)", blind_trades
    )

    if args.smoke:
        return

    rows, _ = apply_monthly_ramp(
        trades,
        args.capital,
        args.increment,
        MARGIN_PER_LOT,
    )
    print_ramp_table(rows)
    print_yearly_ramp(rows, args.capital)


if __name__ == "__main__":
    main()
