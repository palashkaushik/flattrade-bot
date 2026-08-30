"""Detailed Year-by-Year Backtest: Master Combined Supreme vs. Supreme + 3m SuperTrend-VWAP Chop Filter.

Directly imports and tests CombinedSupremeEngine from flattrade_bot.strategies.undisputed_rejection!
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattrade_bot.strategies.undisputed_rejection import CombinedSupremeEngine

LOT_SIZE = 65
FEE_PER_TRADE = 45.0
DESKTOP_DATA = Path(r"C:\Users\user\Desktop\nifty50 data")
IDX_FILE = DESKTOP_DATA / "index" / "NIFTY 50_minute.csv"


def compute_supertrend(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 10, multiplier: float = 3.0) -> np.ndarray:
    n = len(closes)
    st = np.zeros(n)
    direction = np.ones(n)
    trs = np.zeros(n)
    trs[0] = highs[0] - lows[0]
    for i in range(1, n):
        trs[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
    atr = np.zeros(n)
    atr[period - 1] = np.mean(trs[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + trs[i]) / period

    hl2 = (highs + lows) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper_band = np.copy(upper_basic)
    lower_band = np.copy(lower_basic)

    for i in range(period, n):
        if lower_basic[i] > lower_band[i - 1] or closes[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i - 1]

        if upper_basic[i] < upper_band[i - 1] or closes[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i - 1]

        if closes[i] > upper_band[i - 1]:
            direction[i] = 1
        elif closes[i] < lower_band[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        st[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
    return st


def load_dataset():
    print("Loading 1-minute Nifty Spot Index dataset...")
    df_raw = pd.read_csv(IDX_FILE)
    df_raw["dt"] = pd.to_datetime(df_raw["date"])
    df_raw["day"] = df_raw["dt"].dt.strftime("%Y-%m-%d")
    df_raw["minute"] = df_raw["dt"].dt.hour * 60 + df_raw["dt"].dt.minute
    df_raw = df_raw[(df_raw["minute"] >= 555) & (df_raw["minute"] <= 930)].reset_index(drop=True)
    df_spot = df_raw[df_raw["day"] >= "2019-12-15"].reset_index(drop=True)
    all_days = sorted(list(df_spot["day"].unique()))

    df_spot["bar_3m_idx"] = (df_spot["minute"] - 555) // 3
    df_spot["bar_5m_idx"] = (df_spot["minute"] - 555) // 5
    df_spot["bar_15m_idx"] = (df_spot["minute"] - 555) // 15

    agg_3m = df_spot.groupby(["day", "bar_3m_idx"]).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        minute_start=("minute", "first")
    ).reset_index()

    agg_5m = df_spot.groupby(["day", "bar_5m_idx"]).agg(close=("close", "last")).reset_index()
    agg_15m = df_spot.groupby(["day", "bar_15m_idx"]).agg(close=("close", "last")).reset_index()

    daily_ohlc = df_spot.groupby("day").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).to_dict("index")

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
                    daily_levels[p_day]["cpr_bot"],
                    p_day
                ))
        virgin_cprs_by_day[day] = past_virgins

    return df_spot, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs_by_day, all_days


def run_strategy_backtest(
    days_to_run: List[str],
    agg_3m: pd.DataFrame,
    agg_5m: pd.DataFrame,
    agg_15m: pd.DataFrame,
    daily_levels: Dict[str, Any],
    virgin_cprs_by_day: Dict[str, Any],
    enable_chop_filter: bool = True,
    all_day_session: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs backtest directly using CombinedSupremeEngine."""
    trades = []
    daily_pnl = {d: 0.0 for d in days_to_run}

    for day in days_to_run:
        if day not in daily_levels:
            continue

        bars_3m = agg_3m[agg_3m["day"] == day].sort_values("bar_3m_idx").to_dict("records")
        bars_5m = agg_5m[agg_5m["day"] == day].sort_values("bar_5m_idx").to_dict("records")
        bars_15m = agg_15m[agg_15m["day"] == day].sort_values("bar_15m_idx").to_dict("records")

        if len(bars_3m) < 15:
            continue

        dl = daily_levels[day]
        virgins = virgin_cprs_by_day.get(day, [])

        closes_3m = np.array([b["close"] for b in bars_3m])
        highs_3m = np.array([b["high"] for b in bars_3m])
        lows_3m = np.array([b["low"] for b in bars_3m])

        st_values = compute_supertrend(highs_3m, lows_3m, closes_3m, period=10, multiplier=3.0)

        engine = CombinedSupremeEngine(
            enable_chop_filter=enable_chop_filter,
            all_day_session=all_day_session
        )

        op_h = bars_3m[0]["high"]
        op_l = bars_3m[0]["low"]

        engine.initialize_daily_levels(
            prev_high=dl["pdh"],
            prev_low=dl["pdl"],
            prev_close=dl["pdc"],
            initial_vwap=float(closes_3m[0]),
            ema200=float(closes_3m[0]),
            ema20=float(closes_3m[0]),
            opening_3m_high=op_h,
            opening_3m_low=op_l,
            virgin_cprs=virgins,
        )

        for i in range(1, len(bars_3m) - 1):
            cur_bar = bars_3m[i]
            prev_bar = bars_3m[i - 1]
            t_min = cur_bar["minute_start"]

            # Compute rolling indicators
            vwap = float(np.mean(closes_3m[:i + 1]))
            st_val = float(st_values[i])
            ema20 = float(np.mean(closes_3m[max(0, i - 6):i + 1]))
            ema200 = float(np.mean(closes_3m[max(0, i - 40):i + 1]))
            ema20_5m = float(np.mean(closes_3m[max(0, i - 10):i + 1]))
            ema200_5m = float(np.mean(closes_3m[max(0, i - 40):i + 1]))

            b15_idx = min(i // 5, len(bars_15m) - 1)
            b15_close = bars_15m[b15_idx]["close"]
            b15_ema20 = b15_close

            past_trs = []
            for k in range(max(1, i - 5), i + 1):
                tr = max(
                    highs_3m[k] - lows_3m[k],
                    abs(highs_3m[k] - closes_3m[k - 1]),
                    abs(lows_3m[k] - closes_3m[k - 1])
                )
                past_trs.append(tr)
            atr5 = float(np.mean(past_trs)) if past_trs else 14.0

            engine.update_indicators(
                spot_price=float(cur_bar["close"]),
                vwap=vwap,
                ema20=ema20,
                ema200=ema200,
                spot_15m_close=b15_close,
                spot_15m_ema20=b15_ema20,
                ema20_5m=ema20_5m,
                ema200_5m=ema200_5m,
                atr=atr5,
                supertrend=st_val,
            )

            # Evaluate trigger
            sim_dt = pd.to_datetime(f"{day} {t_min//60:02d}:{t_min%60:02d}:00")
            setup = engine.evaluate_rejection_trigger(prev_bar, cur_bar, now=sim_dt)

            if setup is not None and setup.confirmed:
                entry_px = setup.entry_price
                sl_px = entry_px - setup.initial_sl if setup.direction == "LONG" else entry_px + setup.initial_sl
                tp_px = entry_px + setup.target_price if setup.direction == "LONG" else entry_px - setup.target_price

                curr_sl = sl_px
                peak_px = entry_px
                exit_px = entry_px
                hit = False

                for f_idx in range(i + 1, min(i + 30, len(bars_3m))):
                    f_bar = bars_3m[f_idx]
                    f_h = f_bar["high"]
                    f_l = f_bar["low"]

                    if setup.direction == "LONG":
                        if f_h > peak_px:
                            peak_px = f_h
                            if (peak_px - entry_px) >= 6.0:
                                curr_sl = max(curr_sl, peak_px - 2.0)
                        if f_l <= curr_sl:
                            exit_px = curr_sl
                            hit = True
                            break
                        if f_h >= tp_px:
                            exit_px = tp_px
                            hit = True
                            break
                    else:
                        if f_l < peak_px:
                            peak_px = f_l
                            if (entry_px - peak_px) >= 6.0:
                                curr_sl = min(curr_sl, peak_px + 2.0)
                        if f_h >= curr_sl:
                            exit_px = curr_sl
                            hit = True
                            break
                        if f_l <= tp_px:
                            exit_px = tp_px
                            hit = True
                            break

                if not hit:
                    exit_px = bars_3m[min(i + 30, len(bars_3m) - 1)]["close"]

                spot_pts = (exit_px - entry_px) if setup.direction == "LONG" else (entry_px - exit_px)
                opt_pts = spot_pts * 0.60
                net_rs = (opt_pts * LOT_SIZE) - FEE_PER_TRADE

                trades.append({
                    "day": day,
                    "year": day[:4],
                    "time": f"{t_min//60:02d}:{t_min%60:02d}",
                    "dir": setup.direction,
                    "level": setup.level.name,
                    "tier": setup.level.priority,
                    "score": setup.score,
                    "spot_pts": spot_pts,
                    "opt_pts": opt_pts,
                    "net_rs": net_rs,
                    "win": 1 if net_rs > 0 else 0,
                })
                daily_pnl[day] += net_rs

    df_tr = pd.DataFrame(trades) if trades else pd.DataFrame()
    pnl_series = pd.Series(daily_pnl)
    cum = pnl_series.cumsum()
    max_dd = float((cum.cummax() - cum).max())
    net_profit = float(df_tr["net_rs"].sum()) if not df_tr.empty else 0.0

    summary = {
        "trades": len(df_tr),
        "net_profit": net_profit,
        "win_rate": float(df_tr["win"].mean() * 100) if not df_tr.empty else 0.0,
        "pf": round(float(df_tr[df_tr["net_rs"] > 0]["net_rs"].sum() / abs(df_tr[df_tr["net_rs"] < 0]["net_rs"].sum())), 2) if not df_tr.empty and abs(df_tr[df_tr["net_rs"] < 0]["net_rs"].sum()) > 0 else 999.0,
        "max_dd": max_dd,
        "calmar": round(net_profit / max_dd, 1) if max_dd > 0 else 999.0,
        "green_days": float((pnl_series > 0).sum() / max(1, (pnl_series != 0).sum()) * 100)
    }

    return trades, summary


def main():
    print("=" * 80)
    print(" 🏆 FULL 7-YEAR AUDIT: MASTER COMBINED SUPREME + SUPERTREND-VWAP CHOP FILTER")
    print("=" * 80)

    _, agg_3m, agg_5m, agg_15m, daily_levels, virgins, all_days = load_dataset()
    valid_days = [d for d in all_days if "2020-01-01" <= d <= "2026-08-20"]

    # 1. SMOKE TEST (5 Days)
    print("\n--- SMOKE TEST (5 Days) ---")
    smoke_days = valid_days[:5]
    _, smoke_sum = run_strategy_backtest(smoke_days, agg_3m, agg_5m, agg_15m, daily_levels, virgins, enable_chop_filter=True, all_day_session=True)
    print(f"Smoke Test: Trades={smoke_sum['trades']}, WinRate={smoke_sum['win_rate']:.1f}%, NetProfit=Rs {smoke_sum['net_profit']:,.2f}")
    assert smoke_sum['trades'] > 0, "Smoke test failed: 0 trades"
    print("✅ Smoke test passed!\n")

    # 2. RUN FULL BACKTEST (Baseline vs Supreme + Chop Filter)
    print("Running Full 7-Year Backtest (2020 - 2026)...")
    t0 = time.time()

    # Model 1: Baseline Dual Sessions
    tr_base, sum_base = run_strategy_backtest(valid_days, agg_3m, agg_5m, agg_15m, daily_levels, virgins, enable_chop_filter=False, all_day_session=False)

    # Model 2: Master Supreme + 3m SuperTrend-VWAP Chop Corridor Filter (All-Day Full Session)
    tr_chop, sum_chop = run_strategy_backtest(valid_days, agg_3m, agg_5m, agg_15m, daily_levels, virgins, enable_chop_filter=True, all_day_session=True)

    elapsed = time.time() - t0
    print(f"Backtest completed in {elapsed:.1f}s.\n")

    # Overall Comparison Table
    print("=" * 95)
    print(f"{'STRATEGY CONFIGURATION':<48} | {'TRADES':<7} | {'WIN RATE':<9} | {'NET PROFIT':<14} | {'PF':<5} | {'CALMAR':<7}")
    print("-" * 95)
    print(f"{'1. Baseline Supreme (Dual Session Standdown)':<48} | {sum_base['trades']:<7} | {sum_base['win_rate']:<8.1f}% | Rs {sum_base['net_profit']:>10,.2f} | {sum_base['pf']:<5.2f} | {sum_base['calmar']:<7.1f}")
    print(f"{'2. Master Supreme + SuperTrend-VWAP Chop (All-Day)':<48} | {sum_chop['trades']:<7} | {sum_chop['win_rate']:<8.1f}% | Rs {sum_chop['net_profit']:>10,.2f} | {sum_chop['pf']:<5.2f} | {sum_chop['calmar']:<7.1f}")
    print("=" * 95)

    # Year by Year Breakdown
    df_chop = pd.DataFrame(tr_chop)
    years = sorted(list(df_chop["year"].unique()))
    print("\n--- 📅 YEAR-BY-YEAR PERFORMANCE BREAKDOWN (MASTER SUPREME + CHOP FILTER) ---")
    print(f"{'YEAR':<6} | {'TRADES':<7} | {'WIN RATE':<9} | {'NET PROFIT (Rs)':<16} | {'PROFIT FACTOR':<13} | {'GREEN DAYS':<10}")
    print("-" * 75)
    for y in years:
        sub = df_chop[df_chop["year"] == y]
        n_tr = len(sub)
        wr = float(sub["win"].mean() * 100)
        net = float(sub["net_rs"].sum())
        gw = float(sub[sub["net_rs"] > 0]["net_rs"].sum())
        gl = float(abs(sub[sub["net_rs"] < 0]["net_rs"].sum()))
        pf = round(gw / gl, 2) if gl > 0 else 999.0
        dp = sub.groupby("day")["net_rs"].sum()
        gd = float((dp > 0).sum() / max(1, len(dp)) * 100)
        print(f"{y:<6} | {n_tr:<7} | {wr:<8.1f}% | Rs {net:>12,.2f} | {pf:<13.2f} | {gd:<9.1f}%")
    print("-" * 75)
    print(f"{'TOTAL':<6} | {sum_chop['trades']:<7} | {sum_chop['win_rate']:<8.1f}% | Rs {sum_chop['net_profit']:>12,.2f} | {sum_chop['pf']:<13.2f} | {sum_chop['green_days']:<9.1f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
