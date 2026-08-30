"""Backtest: Combined Supreme All-Day + 3m SuperTrend vs VWAP Chop Filter.

Compares:
  1. NIFTY Spot Index
  2. NIFTY Futures
Timeframe: 3-Minute Primary Candles
Rule: Cannot trade when Price is between SuperTrend(10, 3) and VWAP on 3-Minute Chart.
Operating Hours: Full Session (09:18 - 15:00) with No-Chop Filter.
Execution: 2nd ITM Nifty Options (Delta = 0.60, Lot Size = 65, Fee = Rs 45).
"""

from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

LOT_SIZE = 65
FEE_PER_TRADE = 45.0
DESKTOP_DATA = Path(r"C:\Users\user\Desktop\nifty50 data")
IDX_FILE = DESKTOP_DATA / "index" / "NIFTY 50_minute.csv"
FUT_DIR = DESKTOP_DATA / "nifty_futures"


def compute_supertrend(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Computes standard SuperTrend(period, multiplier) on 3m bars."""
    n = len(closes)
    st = np.zeros(n)
    direction = np.ones(n)  # 1 = Bullish, -1 = Bearish

    # 1. Compute ATR
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

    # 2. Upper and Lower Bands
    hl2 = (highs + lows) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper_band = np.copy(upper_basic)
    lower_band = np.copy(lower_basic)

    for i in range(period, n):
        # Lower band trailing
        if lower_basic[i] > lower_band[i - 1] or closes[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i - 1]

        # Upper band trailing
        if upper_basic[i] < upper_band[i - 1] or closes[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i - 1]

        # Direction switch
        if closes[i] > upper_band[i - 1]:
            direction[i] = 1
        elif closes[i] < lower_band[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        st[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    return st, direction


def load_spot_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], List[str]]:
    """Loads and aggregates Nifty Spot Index into 3m, 5m, 15m bars."""
    print("Loading Nifty Spot 1-minute data...")
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
                    daily_levels[p_day]["cpr_bot"]
                ))
        virgin_cprs_by_day[day] = past_virgins

    return df_spot, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs_by_day, all_days


def load_futures_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], List[str]]:
    """Loads and aggregates Nifty Futures 1-minute files into 3m, 5m, 15m bars."""
    print("Loading Nifty Futures 1-minute data across all years...")
    csv_files = glob.glob(str(FUT_DIR / "**" / "*.csv"), recursive=True)
    dfs = []
    for f in csv_files:
        try:
            d = pd.read_csv(f)
            if "date" in d.columns and "time" in d.columns:
                dfs.append(d)
        except Exception:
            continue

    df_fut = pd.concat(dfs, ignore_index=True)
    df_fut["dt"] = pd.to_datetime(df_fut["date"] + " " + df_fut["time"])
    df_fut["day"] = df_fut["dt"].dt.strftime("%Y-%m-%d")
    df_fut["minute"] = df_fut["dt"].dt.hour * 60 + df_fut["dt"].dt.minute
    df_fut = df_fut[(df_fut["minute"] >= 555) & (df_fut["minute"] <= 930)].sort_values("dt").reset_index(drop=True)
    all_days = sorted(list(df_fut["day"].unique()))

    df_fut["bar_3m_idx"] = (df_fut["minute"] - 555) // 3
    df_fut["bar_5m_idx"] = (df_fut["minute"] - 555) // 5
    df_fut["bar_15m_idx"] = (df_fut["minute"] - 555) // 15

    agg_3m = df_fut.groupby(["day", "bar_3m_idx"]).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
        minute_start=("minute", "first")
    ).reset_index()

    agg_5m = df_fut.groupby(["day", "bar_5m_idx"]).agg(close=("close", "last")).reset_index()
    agg_15m = df_fut.groupby(["day", "bar_15m_idx"]).agg(close=("close", "last")).reset_index()

    daily_ohlc = df_fut.groupby("day").agg(
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
                    daily_levels[p_day]["cpr_bot"]
                ))
        virgin_cprs_by_day[day] = past_virgins

    return df_fut, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs_by_day, all_days


def run_simulation(
    days_to_run: List[str],
    agg_3m: pd.DataFrame,
    agg_5m: pd.DataFrame,
    agg_15m: pd.DataFrame,
    daily_levels: Dict[str, Any],
    virgin_cprs_by_day: Dict[str, Any],
    enable_supertrend_vwap_filter: bool = True,
    standdown_disabled: bool = True,  # All-day trading
) -> Dict[str, Any]:
    """Runs simulation with the SuperTrend vs VWAP Chop Corridor Filter."""
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
        op_h = bars_3m[0]["high"]
        op_l = bars_3m[0]["low"]
        touch_budget = {}

        closes_3m = np.array([b["close"] for b in bars_3m])
        highs_3m = np.array([b["high"] for b in bars_3m])
        lows_3m = np.array([b["low"] for b in bars_3m])

        # Precompute 3m SuperTrend(10, 3)
        st_values, st_dir = compute_supertrend(highs_3m, lows_3m, closes_3m, period=10, multiplier=3.0)

        for i in range(1, len(bars_3m) - 1):
            cur_bar = bars_3m[i]
            prev_bar = bars_3m[i - 1]
            t_min = cur_bar["minute_start"]

            # Time check (09:18 to 15:00)
            if t_min < 558 or t_min > 915:
                continue

            if not standdown_disabled and (660 <= t_min < 810):
                continue  # Standdown active

            # Compute Session VWAP
            vwap = float(np.mean(closes_3m[:i + 1]))
            st_val = st_values[i]
            cur_close = cur_bar["close"]

            # --- SUPERTREND VS VWAP CHOP FILTER ---
            # Rule: Cannot trade when Price is between SuperTrend and VWAP on 3-Minute Chart
            if enable_supertrend_vwap_filter:
                chop_upper = max(st_val, vwap)
                chop_lower = min(st_val, vwap)
                if chop_lower <= cur_close <= chop_upper:
                    continue  # Trapped in chop corridor!

            # 15m Trend Gate
            b15_idx = min(i // 5, len(bars_15m) - 1)
            b15_close = bars_15m[b15_idx]["close"]
            b15_ema20 = b15_close
            is_bull_15m = b15_close >= b15_ema20

            # ATR(5)
            past_trs = []
            for k in range(max(1, i - 5), i + 1):
                tr = max(
                    highs_3m[k] - lows_3m[k],
                    abs(highs_3m[k] - closes_3m[k - 1]),
                    abs(lows_3m[k] - closes_3m[k - 1])
                )
                past_trs.append(tr)
            atr5 = float(np.mean(past_trs)) if past_trs else 10.0

            ema20_5m = float(np.mean(closes_3m[max(0, i - 10):i + 1]))
            ema200_5m = float(np.mean(closes_3m[max(0, i - 40):i + 1]))
            ema20_3m = float(np.mean(closes_3m[max(0, i - 6):i + 1]))

            candidates = [
                ("Virgin CPR Pivot", virgins[0][0] if virgins else None, 1, True),
                ("Virgin CPR Top", virgins[0][1] if virgins else None, 1, True),
                ("Virgin CPR Bot", virgins[0][2] if virgins else None, 1, True),
                ("Camarilla H3", dl["cam_h3"], 1, False),
                ("Camarilla L3", dl["cam_l3"], 1, False),
                ("Daily CPR Pivot", dl["cpr_p"], 1, False),
                ("Daily CPR Top", dl["cpr_top"], 1, False),
                ("Daily CPR Bot", dl["cpr_bot"], 1, False),
                ("Daily VWAP", vwap, 1, False),
                ("5m EMA 20", ema20_5m, 1, False),
                ("5m EMA 200", ema200_5m, 1, False),
                ("Opening 3m High", op_h, 2, False),
                ("Opening 3m Low", op_l, 2, False),
                ("3m EMA 20", ema20_3m, 2, False),
                ("Prev Day High", dl["pdh"], 2, False),
                ("Prev Day Low", dl["pdl"], 2, False),
                ("Fibonacci H3", dl["fib_h3"], 3, False),
                ("Fibonacci L3", dl["fib_l3"], 3, False),
                ("Camarilla H4", dl["cam_h4"], 3, False),
                ("Camarilla L4", dl["cam_l4"], 3, False),
            ]

            tol = max(0.50 * atr5, 4.0)

            for lvl_name, lvl_px, tier, is_v in candidates:
                if lvl_px is None:
                    continue
                if touch_budget.get(lvl_name, 0) >= 2:
                    continue

                # LONG SETUP (Bounce from support)
                bar1_touched = abs(prev_bar["low"] - lvl_px) <= tol or abs(prev_bar["close"] - lvl_px) <= tol
                # Additional check: Price must be above both or breaking out of chop for Long
                valid_long_zone = (cur_close > max(st_val, vwap)) if enable_supertrend_vwap_filter else True

                if bar1_touched and cur_bar["high"] > prev_bar["high"] and is_bull_15m and valid_long_zone:
                    score = 40 + (25 if is_v else 20 if tier == 1 else 10 if tier == 2 else 5)
                    if abs(cur_bar["close"] - lvl_px) <= tol:
                        score += 15
                    if is_bull_15m:
                        score += 25

                    if score >= 50:
                        entry_px = prev_bar["high"] + 0.50
                        init_sl = max(0.30 * atr5, 4.0)
                        init_tp = max(1.50 * atr5, 8.0)
                        sl_px = entry_px - init_sl
                        tp_px = entry_px + init_tp

                        curr_sl = sl_px
                        peak_px = entry_px
                        exit_px = entry_px
                        exit_reason = "EOD"
                        hit = False

                        for f_idx in range(i + 1, min(i + 30, len(bars_3m))):
                            f_bar = bars_3m[f_idx]
                            if f_bar["high"] > peak_px:
                                peak_px = f_bar["high"]
                                if (peak_px - entry_px) >= 6.0:
                                    curr_sl = max(curr_sl, peak_px - 2.0)
                            if f_bar["low"] <= curr_sl:
                                exit_px = curr_sl
                                exit_reason = "Trailing SL" if curr_sl > sl_px else "Initial SL"
                                hit = True
                                break
                            if f_bar["high"] >= tp_px:
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
                            "spot_pts": spot_pts,
                            "opt_pts": opt_pts,
                            "net_rs": net_rs,
                            "win": 1 if net_rs > 0 else 0,
                            "exit_reason": exit_reason
                        })
                        touch_budget[lvl_name] = touch_budget.get(lvl_name, 0) + 1
                        daily_pnl[day] += net_rs

    # Calculate Performance Stats
    if not trades:
        return {"trades": 0, "net_profit": 0.0, "win_rate": 0.0, "pf": 0.0, "green_days": 0.0, "max_dd": 0.0}

    df_tr = pd.DataFrame(trades)
    n_trades = len(df_tr)
    wr = float(df_tr["win"].mean() * 100)
    net_profit = float(df_tr["net_rs"].sum())
    gross_win = float(df_tr[df_tr["net_rs"] > 0]["net_rs"].sum())
    gross_loss = float(abs(df_tr[df_tr["net_rs"] < 0]["net_rs"].sum()))
    pf = round(gross_win / gross_loss, 3) if gross_loss > 0 else 999.0

    pnl_series = pd.Series(daily_pnl)
    green_days = float((pnl_series > 0).sum() / max(1, (pnl_series != 0).sum()) * 100)
    cum = pnl_series.cumsum()
    max_dd = float((cum.cummax() - cum).max())

    return {
        "trades": n_trades,
        "net_profit": net_profit,
        "win_rate": wr,
        "pf": pf,
        "green_days": green_days,
        "max_dd": max_dd,
        "calmar": round(net_profit / max_dd, 2) if max_dd > 0 else 999.0
    }


def main():
    print("=" * 70)
    print(" 🚀 COMBINED SUPREME: ALL-DAY + 3M SUPERTREND vs VWAP CHOP FILTER")
    print("=" * 70)

    # 1. Load Datasets
    _, spot_3m, spot_5m, spot_15m, spot_levels, spot_virgins, spot_days = load_spot_dataset()
    _, fut_3m, fut_5m, fut_15m, fut_levels, fut_virgins, fut_days = load_futures_dataset()

    # Filter to 2020-2026 common range
    valid_spot_days = [d for d in spot_days if "2020-01-01" <= d <= "2026-08-20"]
    valid_fut_days = [d for d in fut_days if "2020-01-01" <= d <= "2026-08-20"]

    # -------------------------------------------------------------
    # STEP 1: MANDATORY SMOKE TEST (First 5 Days)
    # -------------------------------------------------------------
    print("\n--- SMOKE TEST (5 Days Sanity Check) ---")
    smoke_days = valid_spot_days[:5]
    smoke_spot = run_simulation(smoke_days, spot_3m, spot_5m, spot_15m, spot_levels, spot_virgins, enable_supertrend_vwap_filter=True, standdown_disabled=True)
    print(f"Smoke Spot (5 days): Trades={smoke_spot['trades']}, WR={smoke_spot['win_rate']:.1f}%, Profit=Rs {smoke_spot['net_profit']:,.2f}")
    assert smoke_spot['trades'] > 0, "Smoke test failed: 0 trades"
    print("✅ Smoke test passed!\n")

    # -------------------------------------------------------------
    # STEP 2: FULL MULTI-YEAR BENCHMARK COMPARISON
    # -------------------------------------------------------------
    print("Running Full 7-Year Multi-Configuration Benchmark...")

    # Config A: Baseline Combined Supreme (Dual Sessions, No Standdown Trading)
    res_baseline = run_simulation(valid_spot_days, spot_3m, spot_5m, spot_15m, spot_levels, spot_virgins, enable_supertrend_vwap_filter=False, standdown_disabled=False)

    # Config B: SPOT INDEX — All-Day Trading + 3m SuperTrend vs VWAP Chop Corridor Filter
    res_spot_chop = run_simulation(valid_spot_days, spot_3m, spot_5m, spot_15m, spot_levels, spot_virgins, enable_supertrend_vwap_filter=True, standdown_disabled=True)

    # Config C: NIFTY FUTURES — All-Day Trading + 3m SuperTrend vs VWAP Chop Corridor Filter
    res_fut_chop = run_simulation(valid_fut_days, fut_3m, fut_5m, fut_15m, fut_levels, fut_virgins, enable_supertrend_vwap_filter=True, standdown_disabled=True)

    # Output Results Table
    print("\n" + "=" * 85)
    print(f"{'CONFIGURATION':<45} | {'TRADES':<7} | {'WIN RATE':<9} | {'PROFIT (Rs)':<14} | {'PF':<5} | {'CALMAR':<7}")
    print("-" * 85)
    print(f"{'1. Baseline Spot (Dual Sessions - 11:00 Chop Pause)':<45} | {res_baseline['trades']:<7} | {res_baseline['win_rate']:<8.1f}% | Rs {res_baseline['net_profit']:>10,.2f} | {res_baseline['pf']:<5.2f} | {res_baseline['calmar']:<7.1f}")
    print(f"{'2. Spot All-Day + 3m SuperTrend-VWAP Chop Filter':<45} | {res_spot_chop['trades']:<7} | {res_spot_chop['win_rate']:<8.1f}% | Rs {res_spot_chop['net_profit']:>10,.2f} | {res_spot_chop['pf']:<5.2f} | {res_spot_chop['calmar']:<7.1f}")
    print(f"{'3. Futures All-Day + 3m SuperTrend-VWAP Chop Filter':<45} | {res_fut_chop['trades']:<7} | {res_fut_chop['win_rate']:<8.1f}% | Rs {res_fut_chop['net_profit']:>10,.2f} | {res_fut_chop['pf']:<5.2f} | {res_fut_chop['calmar']:<7.1f}")
    print("=" * 85)


if __name__ == "__main__":
    main()

