"""7-Year Backtest: Master Combined Supreme Strategy + 3m Elder Impulse Filter.

PineScript Specification:
  - MACD Fast = 12, Slow = 26, Signal = 9
  - EMA Length = 13
  - Bulls (GREEN): ema[0] > ema[1] and macd_hist[0] > macd_hist[1] -> CE Trades Only
  - Bears (RED): ema[0] < ema[1] and macd_hist[0] < macd_hist[1] -> PE Trades Only
  - Neutral (BLUE): Neither -> Both CE & PE Trades Allowed

Dataset: Nifty Spot 1-Minute Aggregated to 3-Minute, 5-Minute, 15-Minute (2020 - 2026).
Execution: 2nd ITM Nifty Options (Delta = 0.60, Lot Size = 65, Cost = Rs 45/trade).
"""

from __future__ import annotations

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


def compute_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Computes Exponential Moving Average."""
    n = len(series)
    ema = np.zeros(n)
    if n == 0:
        return ema
    alpha = 2.0 / (span + 1.0)
    ema[0] = series[0]
    for i in range(1, n):
        ema[i] = alpha * series[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def compute_elder_impulse(closes: np.ndarray) -> np.ndarray:
    """
    Computes Elder Impulse Color on 3m bars:
      1 = GREEN (Bulls: EMA13 rising & MACD Hist rising)
     -1 = RED (Bears: EMA13 falling & MACD Hist falling)
      0 = BLUE (Neutral: Neither)
    """
    n = len(closes)
    colors = np.zeros(n, dtype=int)
    if n < 30:
        return colors

    # 1. EMA 13
    ema13 = compute_ema(closes, 13)

    # 2. MACD(12, 26, 9)
    fast_ema = compute_ema(closes, 12)
    slow_ema = compute_ema(closes, 26)
    macd_line = fast_ema - slow_ema
    macd_signal = compute_ema(macd_line, 9)
    macd_hist = macd_line - macd_signal

    for i in range(1, n):
        ema_rising = ema13[i] > ema13[i - 1]
        ema_falling = ema13[i] < ema13[i - 1]
        hist_rising = macd_hist[i] > macd_hist[i - 1]
        hist_falling = macd_hist[i] < macd_hist[i - 1]

        if ema_rising and hist_rising:
            colors[i] = 1   # GREEN
        elif ema_falling and hist_falling:
            colors[i] = -1  # RED
        else:
            colors[i] = 0   # BLUE

    return colors


def compute_supertrend(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    n = len(closes)
    st = np.zeros(n)
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

    direction = np.ones(n)
    for i in range(period, n):
        if direction[i - 1] == 1:
            if closes[i] < lower_band[i]:
                direction[i] = -1
                st[i] = upper_band[i]
            else:
                direction[i] = 1
                st[i] = lower_band[i]
        else:
            if closes[i] > upper_band[i]:
                direction[i] = 1
                st[i] = lower_band[i]
            else:
                direction[i] = -1
                st[i] = upper_band[i]

    return st, direction


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], List[str]]:
    print("Loading 1-minute Nifty Spot Index dataset...")
    df = pd.read_csv(IDX_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else "datetime"
    df["dt"] = pd.to_datetime(df[date_col])
    df = df.sort_values("dt").reset_index(drop=True)
    df["day"] = df["dt"].dt.strftime("%Y-%m-%d")
    df["minute_start"] = df["dt"].dt.hour * 60 + df["dt"].dt.minute

    # 3-Minute Aggregation
    df["bar_3m_idx"] = (df["minute_start"] - 555) // 3
    agg_3m = df.groupby(["day", "bar_3m_idx"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum") if "volume" in df.columns else ("close", "count"),
        minute_start=("minute_start", "first"),
    ).reset_index()

    # 5-Minute Aggregation
    df["bar_5m_idx"] = (df["minute_start"] - 555) // 5
    agg_5m = df.groupby(["day", "bar_5m_idx"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        minute_start=("minute_start", "first"),
    ).reset_index()

    # 15-Minute Aggregation
    df["bar_15m_idx"] = (df["minute_start"] - 555) // 15
    agg_15m = df.groupby(["day", "bar_15m_idx"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        minute_start=("minute_start", "first"),
    ).reset_index()

    # Daily Levels & Virgin CPRs
    daily_stats = df.groupby("day").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index()

    days = list(daily_stats["day"].unique())
    daily_levels = {}
    virgin_cprs = {}
    history = []

    for i in range(1, len(days)):
        prev_d = days[i - 1]
        cur_d = days[i]
        p_row = daily_stats[daily_stats["day"] == prev_d].iloc[0]
        ph, pl, pc = p_row["high"], p_row["low"], p_row["close"]

        p_pivot = (ph + pl + pc) / 3.0
        p_bc = (ph + pl) / 2.0
        p_tc = (p_pivot - p_bc) + p_pivot
        cpr_top = max(p_tc, p_bc)
        cpr_bot = min(p_tc, p_bc)

        rng = ph - pl
        h3 = pc + rng * (1.1 / 4.0)
        l3 = pc - rng * (1.1 / 4.0)
        h4 = pc + rng * (1.1 / 2.0)
        l4 = pc - rng * (1.1 / 2.0)
        fib_h3 = p_pivot + rng * 1.0
        fib_l3 = p_pivot - rng * 1.0

        daily_levels[cur_d] = {
            "pdh": ph, "pdl": pl, "pdc": pc,
            "cpr_p": p_pivot, "cpr_top": cpr_top, "cpr_bot": cpr_bot,
            "cam_h3": h3, "cam_l3": l3, "cam_h4": h4, "cam_l4": l4,
            "fib_h3": fib_h3, "fib_l3": fib_l3,
        }

        # Track Virgin CPRs
        history.append((p_pivot, cpr_top, cpr_bot, prev_d))
        active_virgins = []
        for vp, vtc, vbc, vday in history[:-1]:
            d_rows = df[df["day"] == cur_d]
            if len(d_rows) > 0:
                dh = d_rows["high"].max()
                dl = d_rows["low"].min()
                if not (dl <= vtc and dh >= vbc):
                    active_virgins.append((vp, vtc, vbc, vday))
        virgin_cprs[cur_d] = active_virgins

    return agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs, days[1:]


def run_simulation(
    days_to_run: List[str],
    agg_3m: pd.DataFrame,
    agg_5m: pd.DataFrame,
    agg_15m: pd.DataFrame,
    daily_levels: Dict[str, Any],
    virgin_cprs_by_day: Dict[str, Any],
    enable_supertrend_vwap_filter: bool = True,
    enable_elder_impulse_filter: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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

        # Precompute 3m Elder Impulse Colors: 1 = GREEN, -1 = RED, 0 = BLUE
        elder_colors = compute_elder_impulse(closes_3m)

        for i in range(1, len(bars_3m) - 1):
            cur_bar = bars_3m[i]
            prev_bar = bars_3m[i - 1]
            t_min = cur_bar["minute_start"]

            # Operating hours: 09:18 to 15:00
            if t_min < 558 or t_min > 915:
                continue

            # Compute Session VWAP
            vwap = float(np.mean(closes_3m[:i + 1]))
            st_val = st_values[i]
            cur_close = cur_bar["close"]
            elder_col = elder_colors[i]  # 1 = Green (CE), -1 = Red (PE), 0 = Blue (Both)

            # --- SUPERTREND VS VWAP CHOP FILTER ---
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

                # --- LONG SETUP (CE) ---
                # Elder Impulse Condition: Green (1) or Blue (0) allowed
                elder_allows_long = (elder_col >= 0) if enable_elder_impulse_filter else True
                bar1_touched = abs(prev_bar["low"] - lvl_px) <= tol or abs(prev_bar["close"] - lvl_px) <= tol
                valid_long_zone = (cur_close > max(st_val, vwap)) if enable_supertrend_vwap_filter else True

                if bar1_touched and cur_bar["high"] > prev_bar["high"] and is_bull_15m and valid_long_zone and elder_allows_long:
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

                        # Simulate Trade Outcome
                        trail_active = False
                        cur_sl = sl_px
                        peak = entry_px
                        won = False
                        exit_px = entry_px

                        for f in range(i, len(bars_3m)):
                            f_high = bars_3m[f]["high"]
                            f_low = bars_3m[f]["low"]

                            if f_high > peak:
                                peak = f_high
                                if (peak - entry_px) >= 6.0:
                                    trail_active = True

                            if trail_active:
                                cur_sl = max(cur_sl, peak - 2.0)

                            if f_high >= tp_px:
                                exit_px = tp_px
                                won = True
                                break
                            elif f_low <= cur_sl:
                                exit_px = cur_sl
                                won = (exit_px > entry_px)
                                break

                        opt_pnl_pts = (exit_px - entry_px) * 0.60
                        net_rs = (opt_pnl_pts * LOT_SIZE) - FEE_PER_TRADE
                        trades.append({
                            "day": day,
                            "year": day[:4],
                            "side": "CE",
                            "level": lvl_name,
                            "win": won,
                            "net_rs": net_rs,
                        })
                        daily_pnl[day] += net_rs
                        touch_budget[lvl_name] = touch_budget.get(lvl_name, 0) + 1
                        break

                # --- SHORT SETUP (PE) ---
                # Elder Impulse Condition: Red (-1) or Blue (0) allowed
                elder_allows_short = (elder_col <= 0) if enable_elder_impulse_filter else True
                bar1_touched_s = abs(prev_bar["high"] - lvl_px) <= tol or abs(prev_bar["close"] - lvl_px) <= tol
                valid_short_zone = (cur_close < min(st_val, vwap)) if enable_supertrend_vwap_filter else True

                if bar1_touched_s and cur_bar["low"] < prev_bar["low"] and (not is_bull_15m) and valid_short_zone and elder_allows_short:
                    score = 40 + (25 if is_v else 20 if tier == 1 else 10 if tier == 2 else 5)
                    if abs(cur_bar["close"] - lvl_px) <= tol:
                        score += 15
                    if not is_bull_15m:
                        score += 25

                    if score >= 50:
                        entry_px = prev_bar["low"] - 0.50
                        init_sl = max(0.30 * atr5, 4.0)
                        init_tp = max(1.50 * atr5, 8.0)
                        sl_px = entry_px + init_sl
                        tp_px = entry_px - init_tp

                        trail_active = False
                        cur_sl = sl_px
                        trough = entry_px
                        won = False
                        exit_px = entry_px

                        for f in range(i, len(bars_3m)):
                            f_high = bars_3m[f]["high"]
                            f_low = bars_3m[f]["low"]

                            if f_low < trough:
                                trough = f_low
                                if (entry_px - trough) >= 6.0:
                                    trail_active = True

                            if trail_active:
                                cur_sl = min(cur_sl, trough + 2.0)

                            if f_low <= tp_px:
                                exit_px = tp_px
                                won = True
                                break
                            elif f_high >= cur_sl:
                                exit_px = cur_sl
                                won = (entry_px > exit_px)
                                break

                        opt_pnl_pts = (entry_px - exit_px) * 0.60
                        net_rs = (opt_pnl_pts * LOT_SIZE) - FEE_PER_TRADE
                        trades.append({
                            "day": day,
                            "year": day[:4],
                            "side": "PE",
                            "level": lvl_name,
                            "win": won,
                            "net_rs": net_rs,
                        })
                        daily_pnl[day] += net_rs
                        touch_budget[lvl_name] = touch_budget.get(lvl_name, 0) + 1
                        break

    # Calculate Summary Metrics
    if not trades:
        return [], {"trades": 0, "win_rate": 0.0, "net_profit": 0.0, "pf": 0.0, "calmar": 0.0}

    df_tr = pd.DataFrame(trades)
    n_tr = len(df_tr)
    wr = float(df_tr["win"].mean() * 100)
    net_p = float(df_tr["net_rs"].sum())
    gw = float(df_tr[df_tr["net_rs"] > 0]["net_rs"].sum())
    gl = float(abs(df_tr[df_tr["net_rs"] < 0]["net_rs"].sum()))
    pf = round(gw / gl, 2) if gl > 0 else 999.0

    # Max Drawdown & Calmar
    cum = df_tr["net_rs"].cumsum()
    peak_cum = np.maximum.accumulate(cum)
    dd = peak_cum - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 1.0
    calmar = round((net_p / max(max_dd, 100.0)), 1)

    summary = {
        "trades": n_tr,
        "win_rate": wr,
        "net_profit": net_p,
        "pf": pf,
        "max_dd": max_dd,
        "calmar": calmar,
    }
    return trades, summary


def main():
    print("=" * 80)
    print(" 🏆 7-YEAR BENCHMARK: MASTER COMBINED SUPREME vs SUPREME + ELDER IMPULSE FILTER")
    print("=" * 80)

    agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs, valid_days = load_dataset()

    # 1. Smoke Test (5 Days)
    print("\n--- SMOKE TEST (5 Days Sanity Check) ---")
    smoke_days = valid_days[:5]
    _, s_sum_base = run_simulation(smoke_days, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs, enable_supertrend_vwap_filter=True, enable_elder_impulse_filter=False)
    _, s_sum_elder = run_simulation(smoke_days, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs, enable_supertrend_vwap_filter=True, enable_elder_impulse_filter=True)
    print(f"Smoke Base : Trades={s_sum_base['trades']}, WR={s_sum_base['win_rate']:.1f}%, Profit=Rs {s_sum_base['net_profit']:,.2f}")
    print(f"Smoke Elder: Trades={s_sum_elder['trades']}, WR={s_sum_elder['win_rate']:.1f}%, Profit=Rs {s_sum_elder['net_profit']:,.2f}")
    assert s_sum_elder['trades'] > 0, "Smoke test failed: 0 trades"
    print("✅ Smoke test passed!\n")

    # 2. Full 7-Year Benchmark (2020 - 2026)
    print("Running Full 7-Year Multi-Configuration Benchmark...")
    t0 = time.time()
    tr_base, sum_base = run_simulation(valid_days, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs, enable_supertrend_vwap_filter=True, enable_elder_impulse_filter=False)
    tr_elder, sum_elder = run_simulation(valid_days, agg_3m, agg_5m, agg_15m, daily_levels, virgin_cprs, enable_supertrend_vwap_filter=True, enable_elder_impulse_filter=True)
    elapsed = time.time() - t0
    print(f"Backtest completed in {elapsed:.1f}s.\n")

    # Summary Table
    print("=" * 105)
    print(f"{'CONFIGURATION':<52} | {'TRADES':<7} | {'WIN RATE':<9} | {'PROFIT (Rs)':<16} | {'PF':<5} | {'CALMAR':<7}")
    print("-" * 105)
    print(f"{'1. Master Supreme + Chop Filter (Current)':<52} | {sum_base['trades']:<7} | {sum_base['win_rate']:<8.1f}% | Rs {sum_base['net_profit']:>12,.2f} | {sum_base['pf']:<5.2f} | {sum_base['calmar']:<7.1f}")
    print(f"{'2. Master Supreme + Elder Impulse Filter (3m)':<52} | {sum_elder['trades']:<7} | {sum_elder['win_rate']:<8.1f}% | Rs {sum_elder['net_profit']:>12,.2f} | {sum_elder['pf']:<5.2f} | {sum_elder['calmar']:<7.1f}")
    print("=" * 105)

    # Year by Year Breakdown for Elder Impulse
    df_el = pd.DataFrame(tr_elder)
    years = sorted(list(df_el["year"].unique()))
    print("\n--- 📅 YEAR-BY-YEAR PERFORMANCE (SUPREME + ELDER IMPULSE FILTER) ---")
    print(f"{'YEAR':<6} | {'TRADES':<7} | {'WIN RATE':<9} | {'NET PROFIT (Rs)':<16} | {'PROFIT FACTOR':<13} | {'GREEN DAYS':<10}")
    print("-" * 75)
    for y in years:
        sub = df_el[df_el["year"] == y]
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
    print(f"{'TOTAL':<6} | {sum_elder['trades']:<7} | {sum_elder['win_rate']:<8.1f}% | Rs {sum_elder['net_profit']:>12,.2f} | {sum_elder['pf']:<13.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
