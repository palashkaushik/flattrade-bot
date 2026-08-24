"""Walk-Forward Fee-Adjusted Backtest — Champion ATR F6, OOS-only stitch.

Goal: measure the champion WITHOUT curve-fit contamination, after penalizing
every trade with real exchange fees + slippage.

Walk-forward (no re-optimization per window — params stay the Optuna champion
that was fit on 2020-2022):
    IS 2020-22 -> OOS 2023   (true OOS, config never saw 2023)
    IS 2021-23 -> OOS 2024   (true OOS, config never saw 2024)
    IS 2020-21 -> OOS 2022   (pseudo-OOS: 2022 WAS in config's training set,
                              reported separately for completeness)

Only OOS years are stitched into the final equity/ramp report.

Per-trade cost model (deducted on EVERY trade, both legs):
  - Slippage:   1.0 pts adverse on entry + 1.0 pts adverse on exit
                (backtest assumption; live-order slippage remains separate)
  - STT:        0.0625% of premium (sell leg)        [govt, options]
  - Exchange:   0.035%  of premium (both legs)       [NSE transaction charge]
  - SEBI:       0.0001% of premium (both legs)       [₹10/crore turnover fee]
  - Stamp:      0.003%  of premium (buy leg)         [state stamp duty]
  - GST:        18% on (brokerage + exchange + SEBI) [tax on charges]
  - Brokerage:  0 per order by default (Flattrade zero-brokerage options);
                pass --brokerage 20 to test the flat-fee plan.

NOTE ON APPROXIMATION: the engine's daily 30-pt shutdown is triggered on RAW
points; slippage/fees are applied afterwards in the P&L pass. Per-trade costs
are constant (~2 pts + ~₹10), so this shifts shutdown breakpoints marginally
but preserves the signal/exit path exactly.

Monthly lot ramp (same money rules as backtest_monthly_ramp.py):
  lots constant per month; lots = max(1, floor(equity_month_start / 40,000));
   daily stop 30.77 pts (Rs 2,000 / 65); consecutive-loss stop 8; starting equity re-set to
  --capital at the START of the OOS stitch (IS years earn nothing toward it).

Usage:
  python backtest_walkforward_fees.py --smoke
  python backtest_walkforward_fees.py
  python backtest_walkforward_fees.py --brokerage 20 --capital 20000
"""

import argparse
import sys
import time
from math import floor
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files, summarize
from backtest_monthly_ramp import (
    CHAMPION_PARAMS, init_worker, process_day, MARGIN_PER_LOT, LOT_SIZE,
    apply_monthly_ramp, print_ramp_table, print_yearly_ramp,
)

WORKERS = 8

# ── fee model ────────────────────────────────────────────────────────────────
SLIPPAGE_PTS = 1.0          # per side (entry + exit), backtests only
BROKERAGE_PER_ORDER = 0.0   # default zero-brokerage; --brokerage overrides
STT_PCT = 0.0625
EXCHANGE_PCT = 0.035
SEBI_PCT = 0.0001
STAMP_PCT = 0.003
GST_PCT = 18.0

WF_WINDOWS = [
    ("2020", "2022", "2023"),
    ("2021", "2023", "2024"),
    ("2020", "2021", "2022"),
]


def trade_cost(entry_px, exit_px, brokerage_per_order):
    """Rupees deducted for one option trade (buy + sell legs), NIFTY lot = 65."""
    prem_buy = entry_px * LOT_SIZE
    prem_sell = exit_px * LOT_SIZE
    stt = STT_PCT / 100.0 * prem_sell
    exch = EXCHANGE_PCT / 100.0 * (prem_buy + prem_sell)
    sebi = SEBI_PCT / 100.0 * (prem_buy + prem_sell)
    stamp = STAMP_PCT / 100.0 * prem_buy
    brokerage = brokerage_per_order * 2
    gst = GST_PCT / 100.0 * (brokerage + exch + sebi)
    return round(stt + exch + sebi + stamp + gst + brokerage, 2)


def apply_costs(trades, brokerage_per_order, slippage_pts=None):
    """Deduct slippage (pts) + fees (Rs) from every trade. Mutates pts/rs."""
    slippage = SLIPPAGE_PTS if slippage_pts is None else float(slippage_pts)
    for t in trades:
        pts_net = t["pts"] - 2 * slippage
        fee = trade_cost(t["entry"], t["exit"], brokerage_per_order)
        t["pts_net"] = round(pts_net, 2)
        t["fee"] = fee
        t["rs_net"] = round(pts_net * LOT_SIZE - fee, 2)
    return trades


def run_oos_year(pool, year, files, spot_all, days):
    oos_days = [d for d in days if d.startswith(f"{year}-")]
    return run_days(pool, oos_days, files)


def run_days(pool, days, files):
    tasks = [(day, str(files[day]), str(files[days[i - 1]]) if i > 0 else "")
             for i, day in enumerate(days)]
    all_trades = []
    for res in pool.map(process_day, tasks):
        all_trades.extend(res)
    return all_trades


def net_stats(trades):
    """Summarize on NET (slippage+fee-adjusted) rupee values."""
    if not trades:
        return {"trades": 0, "wr": 0.0, "pts": 0.0, "rs": 0, "pf": 0.0, "fees": 0.0}
    wins = [t for t in trades if t["rs_net"] > 0]
    losses = [t for t in trades if t["rs_net"] <= 0]
    gross_w = sum(t["rs_net"] for t in wins)
    gross_l = abs(sum(t["rs_net"] for t in losses))
    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "pts": round(sum(t["pts_net"] for t in trades), 2),
        "rs": round(sum(t["rs_net"] for t in trades)),
        "pf": gross_w / gross_l if gross_l else float("inf"),
        "fees": round(sum(t["fee"] for t in trades), 2),
    }


def summarize_net(st):
    return (f"trades {st['trades']:5,d} | WR {st['wr']:5.1f}% | "
            f"pts_net {st['pts']:+10.2f} | Rs_net {st['rs']:+13,d} | PF {st['pf']:5.2f} | "
            f"fees {st['fees']:+,.2f}")


def main():
    ap = argparse.ArgumentParser(description="Walk-Forward Fee-Adjusted Backtest (Champion ATR F6)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--capital", type=float, default=20000.0)
    ap.add_argument("--increment", type=float, default=40000.0)
    ap.add_argument("--brokerage", type=float, default=BROKERAGE_PER_ORDER)
    args = ap.parse_args()

    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    print(f"=== WALK-FORWARD FEE-ADJUSTED | slippage {SLIPPAGE_PTS}pts/side | "
          f"STT {STT_PCT}% | exch {EXCHANGE_PCT}% | SEBI {SEBI_PCT}% | stamp {STAMP_PCT}% | "
          f"GST {GST_PCT}% | brokerage {args.brokerage}/order ===", flush=True)
    print(f"Capital {args.capital:,.0f} | ramp {args.increment:,.0f}/lot | workers {WORKERS}", flush=True)

    oos_collect = []
    fee_total = 0.0

    for is_start, is_end, oos_year in WF_WINDOWS:
        oos_days = [d for d in days if d.startswith(f"{oos_year}-")]
        if args.smoke:
            oos_days = oos_days[:5]
        t0 = time.time()
        with Pool(processes=WORKERS, initializer=init_worker,
                  initargs=(spot_all,)) as pool:
            trades = run_days(pool, oos_days, files)
        elapsed = time.time() - t0
        apply_costs(trades, args.brokerage)
        st = net_stats(trades)
        tag = "TRUE OOS " if oos_year in ("2023", "2024") else "pseudo-OOS (2022 IS for config)"
        print(f"\n--- OOS {oos_year} (IS {is_start}-{is_end}, {len(oos_days)}d, {elapsed:.1f}s) [{tag}]")
        print(f"    {summarize_net(st)}")
        if args.smoke:
            ok = 15 <= st["trades"] <= 40
            print(f"SMOKE {oos_year}: {st['trades']} trades (expect 15-40) -> {'OK' if ok else 'SUSPICIOUS'}")
            continue
        if oos_year in ("2023", "2024"):
            oos_collect.extend(trades)
        fee_total += st["fees"]

    if args.smoke:
        sys.exit(0)

    # stitched TRUE-OOS stream (2023 + 2024) → monthly ramp
    stitched = oos_collect
    print(f"\n{'='*104}")
    print(f"STITCHED TRUE-OOS STREAM (2023 + 2024) — {len(stitched)} trades, "
          f"total fees Rs {fee_total:,.2f}")
    print(f"{'='*104}")
    st = net_stats(stitched)
    print(summarize_net(st))

    rows, _ = apply_monthly_ramp(stitched, args.capital, args.increment, MARGIN_PER_LOT)
    print_ramp_table(rows)
    print_yearly_ramp(rows, args.capital)


if __name__ == "__main__":
    main()
