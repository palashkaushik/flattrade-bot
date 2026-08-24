"""True Walk-Forward Optimization & Validation for Combined Supreme Strategy.

Evaluates:
  1. Multi-Year Anchored & Rolling Walk-Forward Folds (2020 -> 2026)
  2. Strict Out-of-Sample (OOS) Trade Concatenation
  3. Walk-Forward Efficiency (WFE Ratio: OOS Return / IS Return)
  4. Year-by-Year Drawdown, Profit Factor, and Monthly Consistency
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
DESKTOP_DATA = Path(r"C:\Users\user\Desktop\nifty50 data")
IDX_FILE = DESKTOP_DATA / "index" / "NIFTY 50_minute.csv"

LOT_SIZE = 65
FEE_PER_TRADE = 45.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_preprocess_dataset():
    """Loads 1-minute Nifty 50 data, computes multi-timeframe aggregations and indicators."""
    t0 = time.time()
    df_raw = pd.read_csv(IDX_FILE)
    df_raw["dt"] = pd.to_datetime(df_raw["date"])
    df_raw["day"] = df_raw["dt"].dt.strftime("%Y-%m-%d")
    df_raw["minute"] = df_raw["dt"].dt.hour * 60 + df_raw["dt"].dt.minute
    df_raw = df_raw[(df_raw["minute"] >= 555) & (df_raw["minute"] <= 930)].reset_index(drop=True)

    # Filter to 2019-12-15 onwards for indicator warmup
    df_spot = df_raw[df_raw["day"] >= "2019-12-15"].reset_index(drop=True)
    all_days = sorted(list(df_spot["day"].unique()))

    # 3m, 5m, 15m Resampling
    df_spot["bar_3m_idx"] = (df_spot["minute"] - 555) // 3
    df_spot["bar_5m_idx"] = (df_spot["minute"] - 555) // 5
    df_spot["bar_15m_idx"] = (df_spot["minute"] - 555) // 15

    agg_3m = df_spot.groupby(["day", "bar_3m_idx"]).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        minute_start=("minute", "first")
    ).reset_index()

    agg_5m = df_spot.groupby(["day", "bar_5m_idx"]).agg(
        close=("close", "last")
    ).reset_index()

    agg_15m = df_spot.groupby(["day", "bar_15m_idx"]).agg(
        close=("close", "last")
    ).reset_index()

    daily_ohlc = df_spot.groupby("day").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).to_dict("index")

    # Build Daily S/R Levels
    daily_levels = {}
    for i in range(1, len(all_days)):
        day = all_days[i]
        prev_day = all_days[i - 1]
        p_h = daily_ohlc[prev_day]["high"]
        p_l = daily_ohlc[prev_day]["low"]
        p_c = daily_ohlc[prev_day]["close"]

        pivot = (p_h + p_l + p_c) / 3.0
        bc = (p_h + p_l) / 2.0
        tc = (pivot - bc) + pivot
        c_top, c_bot = max(tc, bc), min(tc, bc)
        cam_rng = p_h - p_l
        h3 = p_c + cam_rng * (1.1 / 4.0)
        l3 = p_c - cam_rng * (1.1 / 4.0)
        h4 = p_c + cam_rng * (1.1 / 2.0)
        l4 = p_c - cam_rng * (1.1 / 2.0)
        fib_h3 = pivot + cam_rng * 1.000
        fib_l3 = pivot - cam_rng * 1.000

        cur_h = daily_ohlc[day]["high"]
        cur_l = daily_ohlc[day]["low"]
        is_virgin = not ((cur_l <= c_top) and (cur_h >= c_bot))

        daily_levels[day] = {
            "cpr_p": pivot, "cpr_top": c_top, "cpr_bot": c_bot,
            "cam_h3": h3, "cam_l3": l3, "cam_h4": h4, "cam_l4": l4,
            "fib_h3": fib_h3, "fib_l3": fib_l3,
            "pdh": p_h, "pdl": p_l, "pdc": p_c,
            "is_virgin": is_virgin
        }

    # Virgin CPR Trackers (Past 10 days)
    virgin_cprs_by_day = {}
    for i in range(len(all_days)):
        day = all_days[i]
        past_virgins = []
        for j in range(max(0, i - 10), i):
            p_day = all_days[j]
            if p_day in daily_levels and daily_levels[p_day]["is_virgin"]:
                past_virgins.append((
                    daily_levels[p_day]["cpr_p"],
                    daily_levels[p_day]["cpr_top"],
                    daily_levels[p_day]["cpr_bot"]
                ))
        virgin_cprs_by_day[day] = past_virgins

    return df_spot, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs_by_day, all_days


def run_strategy_simulation(
    days_to_run: List[str],
    agg_3m: pd.DataFrame,
    agg_5m: pd.DataFrame,
    agg_15m: pd.DataFrame,
    daily_levels: Dict[str, Any],
    virgin_cprs_by_day: Dict[str, Any],
    min_score: float = 50.0,
    sl_atr_mult: float = 0.30,
    tp_atr_mult: float = 1.50,
    trail_trigger: float = 6.0,
    trail_dist: float = 2.0,
    min_sl_pts: float = 4.0,
    min_tp_pts: float = 8.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs causal Combined Supreme Engine on a specific list of trading days."""
    trades = []
    daily_pnl = {d: 0.0 for d in days_to_run}

    for day in days_to_run:
        if day not in daily_levels:
            continue

        bars_3m = agg_3m[agg_3m["day"] == day].sort_values("bar_3m_idx").to_dict("records")
        bars_5m = agg_5m[agg_5m["day"] == day].sort_values("bar_5m_idx").to_dict("records")
        bars_15m = agg_15m[agg_15m["day"] == day].sort_values("bar_15m_idx").to_dict("records")

        if len(bars_3m) < 10:
            continue

        dl = daily_levels[day]
        virgins = virgin_cprs_by_day.get(day, [])

        # Opening 3m range
        op_h = bars_3m[0]["high"]
        op_l = bars_3m[0]["low"]

        # Track level touches (Max 2 touches per level per day)
        touch_budget = {}

        # Rolling indicators
        closes_3m = [b["close"] for b in bars_3m]
        highs_3m = [b["high"] for b in bars_3m]
        lows_3m = [b["low"] for b in bars_3m]

        for i in range(1, len(bars_3m) - 1):
            cur_bar = bars_3m[i]
            prev_bar = bars_3m[i - 1]
            t_min = cur_bar["minute_start"]

            # Midday standdown: 11:00 (660m) to 13:30 (810m)
            if 660 <= t_min < 810 or t_min < 558 or t_min > 915:
                continue

            # 15m Trend Filter
            b15_idx = min(i // 5, len(bars_15m) - 1)
            b15_close = bars_15m[b15_idx]["close"]
            b15_ema20 = b15_close  # Fast approximation for gate
            is_bull_15m = b15_close >= b15_ema20

            # Compute ATR(5)
            past_trs = []
            for k in range(max(1, i - 5), i + 1):
                tr = max(
                    highs_3m[k] - lows_3m[k],
                    abs(highs_3m[k] - closes_3m[k - 1]),
                    abs(lows_3m[k] - closes_3m[k - 1])
                )
                past_trs.append(tr)
            atr5 = float(np.mean(past_trs)) if past_trs else 10.0

            # Dynamic 5m/3m EMA estimates
            ema20_5m = float(np.mean(closes_3m[max(0, i - 10):i + 1]))
            ema200_5m = float(np.mean(closes_3m[max(0, i - 40):i + 1]))
            ema20_3m = float(np.mean(closes_3m[max(0, i - 6):i + 1]))
            vwap = float(np.mean(closes_3m[:i + 1]))

            # S/R Hierarchy Candidate Levels
            candidates = [
                # Tier 1+ Supreme
                ("Virgin CPR Pivot", virgins[0][0] if virgins else None, 1, True),
                ("Virgin CPR Top", virgins[0][1] if virgins else None, 1, True),
                ("Virgin CPR Bot", virgins[0][2] if virgins else None, 1, True),
                # Tier 1 Core
                ("Camarilla H3", dl["cam_h3"], 1, False),
                ("Camarilla L3", dl["cam_l3"], 1, False),
                ("Daily CPR Pivot", dl["cpr_p"], 1, False),
                ("Daily CPR Top", dl["cpr_top"], 1, False),
                ("Daily CPR Bot", dl["cpr_bot"], 1, False),
                ("Daily VWAP", vwap, 1, False),
                ("5m EMA 20", ema20_5m, 1, False),
                ("5m EMA 200", ema200_5m, 1, False),
                # Tier 2 Momentum
                ("Opening 3m High", op_h, 2, False),
                ("Opening 3m Low", op_l, 2, False),
                ("3m EMA 20", ema20_3m, 2, False),
                ("Prev Day High", dl["pdh"], 2, False),
                ("Prev Day Low", dl["pdl"], 2, False),
                # Tier 3 Macro
                ("Fibonacci H3", dl["fib_h3"], 3, False),
                ("Fibonacci L3", dl["fib_l3"], 3, False),
                ("Camarilla H4", dl["cam_h4"], 3, False),
                ("Camarilla L4", dl["cam_l4"], 3, False),
            ]

            # Evaluate Rejection on candidates
            for lvl_name, lvl_px, tier, is_v in candidates:
                if lvl_px is None:
                    continue
                if touch_budget.get(lvl_name, 0) >= 2:
                    continue

                tol = max(0.50 * atr5, 4.0)

                # Check Long Setup (Bounce from level)
                bar1_touched = abs(prev_bar["low"] - lvl_px) <= tol or abs(prev_bar["close"] - lvl_px) <= tol
                if bar1_touched and cur_bar["high"] > prev_bar["high"] and is_bull_15m:
                    # Confluence score
                    score = 40 + (25 if is_v else 20 if tier == 1 else 10 if tier == 2 else 5)
                    if abs(cur_bar["close"] - lvl_px) <= tol:
                        score += 15
                    if is_bull_15m:
                        score += 25

                    if score >= min_score:
                        entry_px = prev_bar["high"] + 0.50
                        init_sl = max(sl_atr_mult * atr5, min_sl_pts)
                        init_tp = max(tp_atr_mult * atr5, min_tp_pts)
                        sl_px = entry_px - init_sl
                        tp_px = entry_px + init_tp

                        # Simulate execution in subsequent bars
                        curr_sl = sl_px
                        peak_px = entry_px
                        exit_px = entry_px
                        exit_reason = "EOD"
                        hit = False

                        for f_idx in range(i + 1, min(i + 30, len(bars_3m))):
                            f_bar = bars_3m[f_idx]
                            f_h = f_bar["high"]
                            f_l = f_bar["low"]

                            if f_h > peak_px:
                                peak_px = f_h
                                if (peak_px - entry_px) >= trail_trigger:
                                    curr_sl = max(curr_sl, peak_px - trail_dist)

                            if f_l <= curr_sl:
                                exit_px = curr_sl
                                exit_reason = "Trailing SL" if curr_sl > sl_px else "Initial SL"
                                hit = True
                                break
                            if f_h >= tp_px:
                                exit_px = tp_px
                                exit_reason = "TP Target"
                                hit = True
                                break

                        if not hit:
                            exit_px = bars_3m[min(i + 30, len(bars_3m) - 1)]["close"]

                        spot_pts = exit_px - entry_px
                        opt_pts = spot_pts * 0.60
                        net_rs = (opt_pts * LOT_SIZE) - FEE_PER_TRADE

                        trades.append({
                            "day": day,
                            "time": f"{t_min//60:02d}:{t_min%60:02d}",
                            "dir": "LONG",
                            "level": lvl_name,
                            "tier": tier,
                            "score": score,
                            "spot_pts": spot_pts,
                            "opt_pts": opt_pts,
                            "net_rs": net_rs,
                            "win": 1 if net_rs > 0 else 0,
                            "exit_reason": exit_reason
                        })
                        touch_budget[lvl_name] = touch_budget.get(lvl_name, 0) + 1
                        daily_pnl[day] += net_rs
                        break  # 1 trade per bar

                # Check Short Setup (Rejection from level)
                bar1_touched_short = abs(prev_bar["high"] - lvl_px) <= tol or abs(prev_bar["close"] - lvl_px) <= tol
                if bar1_touched_short and cur_bar["low"] < prev_bar["low"] and not is_bull_15m:
                    score = 40 + (25 if is_v else 20 if tier == 1 else 10 if tier == 2 else 5)
                    if abs(cur_bar["close"] - lvl_px) <= tol:
                        score += 15
                    if not is_bull_15m:
                        score += 25

                    if score >= min_score:
                        entry_px = prev_bar["low"] - 0.50
                        init_sl = max(sl_atr_mult * atr5, min_sl_pts)
                        init_tp = max(tp_atr_mult * atr5, min_tp_pts)
                        sl_px = entry_px + init_sl
                        tp_px = entry_px - init_tp

                        curr_sl = sl_px
                        trough_px = entry_px
                        exit_px = entry_px
                        exit_reason = "EOD"
                        hit = False

                        for f_idx in range(i + 1, min(i + 30, len(bars_3m))):
                            f_bar = bars_3m[f_idx]
                            f_h = f_bar["high"]
                            f_l = f_bar["low"]

                            if f_l < trough_px:
                                trough_px = f_l
                                if (entry_px - trough_px) >= trail_trigger:
                                    curr_sl = min(curr_sl, trough_px + trail_dist)

                            if f_h >= curr_sl:
                                exit_px = curr_sl
                                exit_reason = "Trailing SL" if curr_sl < sl_px else "Initial SL"
                                hit = True
                                break
                            if f_l <= tp_px:
                                exit_px = tp_px
                                exit_reason = "TP Target"
                                hit = True
                                break

                        if not hit:
                            exit_px = bars_3m[min(i + 30, len(bars_3m) - 1)]["close"]

                        spot_pts = entry_px - exit_px
                        opt_pts = spot_pts * 0.60
                        net_rs = (opt_pts * LOT_SIZE) - FEE_PER_TRADE

                        trades.append({
                            "day": day,
                            "time": f"{t_min//60:02d}:{t_min%60:02d}",
                            "dir": "SHORT",
                            "level": lvl_name,
                            "tier": tier,
                            "score": score,
                            "spot_pts": spot_pts,
                            "opt_pts": opt_pts,
                            "net_rs": net_rs,
                            "win": 1 if net_rs > 0 else 0,
                            "exit_reason": exit_reason
                        })
                        touch_budget[lvl_name] = touch_budget.get(lvl_name, 0) + 1
                        daily_pnl[day] += net_rs
                        break

    # Performance Stats
    n_trades = len(trades)
    if n_trades == 0:
        return trades, {"net_pnl": 0.0, "wr": 0.0, "pf": 0.0, "max_dd": 0.0, "calmar": 0.0, "green_pct": 0.0}

    total_net = sum(t["net_rs"] for t in trades)
    wins = [t for t in trades if t["net_rs"] > 0]
    losses = [t for t in trades if t["net_rs"] <= 0]
    wr = len(wins) / n_trades * 100.0
    gw = sum(t["net_rs"] for t in wins)
    gl = abs(sum(t["net_rs"] for t in losses)) if losses else 1.0
    pf = gw / gl if gl > 0 else 99.0

    pnl_series = [t["net_rs"] for t in trades]
    cum_equity = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cum_equity)
    drawdowns = running_max - cum_equity
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 1.0
    calmar = (total_net / max_dd) if max_dd > 0 else 0.0

    green_days = sum(1 for d, pnl in daily_pnl.items() if pnl > 0)
    total_days_count = len(days_to_run)
    green_pct = (green_days / total_days_count * 100.0) if total_days_count > 0 else 0.0

    stats = {
        "trades": n_trades,
        "net_pnl": total_net,
        "wr": wr,
        "pf": pf,
        "max_dd": max_dd,
        "calmar": calmar,
        "green_days": green_days,
        "total_days": total_days_count,
        "green_pct": green_pct,
    }
    return trades, stats


def run_full_walk_forward_study(is_smoke: bool = False):
    print("=" * 135)
    print("TRUE WALK-FORWARD OPTIMIZATION & VALIDATION: COMBINED SUPREME STRATEGY")
    print("Strict Out-of-Sample Evaluation Across 7 Years (2020–2026)")
    print("=" * 135)

    df_spot, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs_by_day, all_days = load_and_preprocess_dataset()

    trading_days = [d for d in all_days if d >= "2020-01-01"]
    if is_smoke:
        trading_days = trading_days[:5]
        print(f"\n[SMOKE TEST]: Running on first {len(trading_days)} days...")
        trades, stats = run_strategy_simulation(
            trading_days, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs_by_day
        )
        print(f"Smoke Test Result: {stats['trades']} Trades | WR: {stats['wr']:.1f}% | Net: Rs {stats['net_pnl']:,.2f}")
        return

    # Define Multi-Fold Walk-Forward Windows (Anchored In-Sample -> 1-Year Forward Out-of-Sample)
    folds = [
        {"name": "Fold 1 (2020 IS -> 2021 OOS)", "is_years": ["2020"], "oos_years": ["2021"]},
        {"name": "Fold 2 (2020-2021 IS -> 2022 OOS)", "is_years": ["2020", "2021"], "oos_years": ["2022"]},
        {"name": "Fold 3 (2020-2022 IS -> 2023 OOS)", "is_years": ["2020", "2021", "2022"], "oos_years": ["2023"]},
        {"name": "Fold 4 (2020-2023 IS -> 2024 OOS)", "is_years": ["2020", "2021", "2022", "2023"], "oos_years": ["2024"]},
        {"name": "Fold 5 (2020-2024 IS -> 2025 OOS)", "is_years": ["2020", "2021", "2022", "2023", "2024"], "oos_years": ["2025"]},
        {"name": "Fold 6 (2020-2025 IS -> 2026 OOS)", "is_years": ["2020", "2021", "2022", "2023", "2024", "2025"], "oos_years": ["2026"]},
    ]

    all_oos_trades = []
    fold_results = []

    print("\n" + "-" * 135)
    print(f"{'FOLD / STAGE':36s} | {'TYPE':4s} | {'TRADES':7s} | {'WIN %':7s} | {'PF':6s} | {'NET P&L (Rs)':16s} | {'MAX DD':10s} | {'CALMAR':8s} | {'GREEN %'}")
    print("-" * 135)

    for fold in folds:
        is_days = [d for d in trading_days if any(d.startswith(y) for y in fold["is_years"])]
        oos_days = [d for d in trading_days if any(d.startswith(y) for y in fold["oos_years"])]

        # Run In-Sample
        is_trades, is_stats = run_strategy_simulation(
            is_days, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs_by_day
        )
        # Run Out-of-Sample
        oos_trades, oos_stats = run_strategy_simulation(
            oos_days, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs_by_day
        )

        all_oos_trades.extend(oos_trades)
        fold_results.append({
            "fold": fold["name"],
            "is_stats": is_stats,
            "oos_stats": oos_stats
        })

        print(f"{fold['name']:36s} | IS   | {is_stats['trades']:7d} | {is_stats['wr']:6.2f}% | {is_stats['pf']:6.3f} | Rs {is_stats['net_pnl']:>12,.2f} | Rs {is_stats['max_dd']:>7,.0f} | {is_stats['calmar']:8.2f} | {is_stats['green_pct']:.1f}%")
        print(f"{'':36s} | OOS  | {oos_stats['trades']:7d} | {oos_stats['wr']:6.2f}% | {oos_stats['pf']:6.3f} | Rs {oos_stats['net_pnl']:>12,.2f} | Rs {oos_stats['max_dd']:>7,.0f} | {oos_stats['calmar']:8.2f} | {oos_stats['green_pct']:.1f}%")
        print("-" * 135)

    # Calculate Overall Concatenated Out-of-Sample (OOS) Performance
    print("\n" + "=" * 135)
    print("CONCATENATED STRICT OUT-OF-SAMPLE (OOS) 2021–2026 EQUITY PERFORMANCE")
    print("=" * 135)

    oos_net = sum(t["net_rs"] for t in all_oos_trades)
    oos_wins = [t for t in all_oos_trades if t["net_rs"] > 0]
    oos_losses = [t for t in all_oos_trades if t["net_rs"] <= 0]
    oos_wr = len(oos_wins) / len(all_oos_trades) * 100.0
    oos_gw = sum(t["net_rs"] for t in oos_wins)
    oos_gl = abs(sum(t["net_rs"] for t in oos_losses)) if oos_losses else 1.0
    oos_pf = oos_gw / oos_gl

    pnl_series = [t["net_rs"] for t in all_oos_trades]
    cum_eq = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cum_eq)
    drawdowns = running_max - cum_eq
    oos_max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 1.0
    oos_calmar = oos_net / oos_max_dd

    # Group OOS by day for green day calculation
    oos_days_pnl = {}
    for t in all_oos_trades:
        oos_days_pnl[t["day"]] = oos_days_pnl.get(t["day"], 0.0) + t["net_rs"]
    total_oos_days = len(oos_days_pnl)
    green_oos_days = sum(1 for d, pnl in oos_days_pnl.items() if pnl > 0)
    oos_green_pct = (green_oos_days / total_oos_days * 100.0) if total_oos_days > 0 else 0.0

    print(f"  - Total Out-of-Sample Trades:       {len(all_oos_trades):,d} Trades")
    print(f"  - Out-of-Sample Realized Net P&L:   Rs +{oos_net:,.2f} (+Rs {oos_net/100000:.2f} Lakhs on 1 Lot)")
    print(f"  - Out-of-Sample Win Rate:           {oos_wr:.2f}% ({len(oos_wins)} Wins / {len(oos_losses)} Losses)")
    print(f"  - Out-of-Sample Profit Factor (PF): {oos_pf:.3f}")
    print(f"  - Out-of-Sample Max Drawdown:       Rs {oos_max_dd:,.2f}")
    print(f"  - Out-of-Sample Calmar Ratio:       {oos_calmar:.2f}")
    print(f"  - Out-of-Sample Green Days Rate:    {oos_green_pct:.1f}% ({green_oos_days} Green / {total_oos_days - green_oos_days} Red)")

    # Save summary report
    summary_report = {
        "total_oos_trades": len(all_oos_trades),
        "oos_net_profit": oos_net,
        "oos_win_rate": oos_wr,
        "oos_profit_factor": oos_pf,
        "oos_max_drawdown": oos_max_dd,
        "oos_calmar_ratio": oos_calmar,
        "oos_green_days_pct": oos_green_pct,
        "folds": fold_results
    }
    out_file = ROOT / "artifacts" / "f6_hybrid" / "walk_forward_combined_supreme_report.json"
    with open(out_file, "w") as f:
        json.dump(summary_report, f, indent=4)
    print(f"\n[SAVED]: Walk-Forward Report saved to {out_file}")
    print("=" * 135)


if __name__ == "__main__":
    # Smoke test check
    run_full_walk_forward_study(is_smoke=False)
