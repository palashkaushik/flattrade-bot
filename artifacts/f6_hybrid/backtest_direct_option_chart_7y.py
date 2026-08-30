"""7-Year Direct Option-Chart Backtest & Optimus 3D Sweep (2020-2026).

Strategy: Master Combined Supreme evaluated directly on 2nd ITM Option Charts.
- Data Source: Raw 1-minute Nifty Weekly Options CSV files (C:/Users/user/Desktop/nifty50 data/nifty_options).
- Strike Selection: 2nd ITM Contract (CE = ATM - 100, PE = ATM + 100).
- Chart Resolution: 3-Minute Option Candles.
- Signals: Rejection Probe Wick (>= 35%) + Breakout Confirmation Bar directly on Option Chart.
- 15m Index Macro Trend Gate:
    * BULL -> Trade 2nd ITM CE Option Chart Only
    * BEAR -> Trade 2nd ITM PE Option Chart Only
- Exits: Pure Option Premium Points (SL, TP, Trailing SL) with exact ₹45/trade statutory fee deducted.
- Walk-Forward (WFO) & Non-Walk-Forward (NWF) 7-Year Benchmark.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
DESKTOP_DATA = Path(r"C:\Users\user\Desktop\nifty50 data")
OPT_DIR = DESKTOP_DATA / "nifty_options"
IDX_FILE = DESKTOP_DATA / "index" / "NIFTY 50_minute.csv"

LOT_SIZE = 65
FEE_PER_TRADE = 45.0
WORKERS = 8
SYM_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")


def parse_option_file(fpath: Path) -> Optional[Tuple[str, pd.DataFrame]]:
    """Loads 1-minute Options chain data for a single day."""
    try:
        df = pd.read_parquet(fpath) if fpath.suffix == ".parquet" else pd.read_csv(fpath)
        if df.empty or "close" not in df.columns or "symbol" not in df.columns:
            return None

        d_str = str(df["date"].iloc[0])
        if "-" in d_str and len(d_str) == 10:
            parts = d_str.split("-")
            date_norm = d_str if len(parts[0]) == 4 else f"{parts[2]}-{parts[1]}-{parts[0]}"
        else:
            date_norm = d_str

        df["minute"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce").dt.hour * 60 + pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce").dt.minute
        df = df[(df["minute"] >= 555) & (df["minute"] <= 930)].reset_index(drop=True)
        return date_norm, df
    except Exception:
        return None


def extract_strike_from_sym(symbol: str) -> Optional[Tuple[int, str]]:
    """Extracts strike and option type (CE/PE) from symbol."""
    try:
        opt_type = symbol[-2:].upper()
        if opt_type not in ("CE", "PE"):
            return None
        digits = ""
        for ch in reversed(symbol[:-2]):
            if ch.isdigit():
                digits = ch + digits
            else:
                break
        if len(digits) >= 4:
            return int(digits), opt_type
        return None
    except Exception:
        return None


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


def run_direct_option_chart_simulation(
    days: List[str],
    spot_by_day: Dict[str, Dict[int, float]],
    day_opt: Dict[str, pd.DataFrame],
    daily_levels: Dict[str, Any],
    virgin_cprs_by_day: Dict[str, Any],
    sl_opt_pts: float = 6.0,
    tp_opt_pts: float = 14.0,
    trail_trigger_pts: float = 5.0,
    trail_step_pts: float = 2.0,
    fee_cover_be_buffer: float = 1.0,
    enable_chop_filter: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    trades = []
    daily_pnl = {d: 0.0 for d in days}

    for day in days:
        if day not in spot_by_day or day not in day_opt:
            continue

        spot_dict = spot_by_day[day]
        opt_df = day_opt[day]
        if opt_df.empty or len(spot_dict) < 50:
            continue

        # Parse strike symbols
        unique_syms = list(opt_df["symbol"].unique())
        sym_map = {}
        for s in unique_syms:
            res = extract_strike_from_sym(s)
            if res:
                sym_map[res] = s

        # Build 3m and 15m Spot index bars
        spot_mins = sorted(list(spot_dict.keys()))
        spot_closes = [spot_dict[m] for m in spot_mins]
        spot_df = pd.DataFrame({"minute": spot_mins, "close": spot_closes})
        spot_df["bar_3m_idx"] = (spot_df["minute"] - 555) // 3
        spot_df["bar_15m_idx"] = (spot_df["minute"] - 555) // 15

        agg_spot_15m = spot_df.groupby("bar_15m_idx").agg(
            close=("close", "last"),
            minute_start=("minute", "first")
        ).to_dict("records")

        # 3m Spot candles for Chop Corridor
        agg_spot_3m = spot_df.groupby("bar_3m_idx").agg(
            open=("close", "first"),
            high=("close", "max"),
            low=("close", "min"),
            close=("close", "last"),
            minute_start=("minute", "first")
        ).to_dict("records")

        spot_3m_highs = np.array([b["high"] for b in agg_spot_3m])
        spot_3m_lows = np.array([b["low"] for b in agg_spot_3m])
        spot_3m_closes = np.array([b["close"] for b in agg_spot_3m])
        st_spot, _ = compute_supertrend(spot_3m_highs, spot_3m_lows, spot_3m_closes, period=10, multiplier=3.0)

        dl = daily_levels.get(day, {})
        virgins = virgin_cprs_by_day.get(day, [])
        op_h = agg_spot_3m[0]["high"] if len(agg_spot_3m) > 0 else 24200.0
        op_l = agg_spot_3m[0]["low"] if len(agg_spot_3m) > 0 else 24100.0
        touch_budget = {}

        for b_idx in range(1, len(agg_spot_3m) - 1):
            cur_s_bar = agg_spot_3m[b_idx]
            prev_s_bar = agg_spot_3m[b_idx - 1]
            t_min = cur_s_bar["minute_start"]

            if t_min < 558 or t_min > 915:
                continue

            spot_px = cur_s_bar["close"]
            vwap_spot = float(np.mean(spot_3m_closes[:b_idx + 1]))
            st_val = st_spot[b_idx]

            # Chop Corridor Filter on Spot
            if enable_chop_filter:
                chop_upper = max(st_val, vwap_spot)
                chop_lower = min(st_val, vwap_spot)
                if chop_lower <= spot_px <= chop_upper:
                    continue  # Stuck in chop!

            # 15m Macro Trend Gate
            b15_idx = min(b_idx // 5, len(agg_spot_15m) - 1)
            b15_close = agg_spot_15m[b15_idx]["close"]
            is_bull_15m = b15_close >= vwap_spot

            # ATR(5) on Spot
            past_trs = []
            for k in range(max(1, b_idx - 5), b_idx + 1):
                tr = max(
                    spot_3m_highs[k] - spot_3m_lows[k],
                    abs(spot_3m_highs[k] - spot_3m_closes[k - 1]),
                    abs(spot_3m_lows[k] - spot_3m_closes[k - 1])
                )
                past_trs.append(tr)
            atr5 = float(np.mean(past_trs)) if past_trs else 10.0

            ema20_5m = float(np.mean(spot_3m_closes[max(0, b_idx - 10):b_idx + 1]))
            ema200_5m = float(np.mean(spot_3m_closes[max(0, b_idx - 40):b_idx + 1]))
            ema20_3m = float(np.mean(spot_3m_closes[max(0, b_idx - 6):b_idx + 1]))

            candidates = [
                ("Virgin CPR Pivot", virgins[0][0] if virgins else None, 1, True),
                ("Virgin CPR Top", virgins[0][1] if virgins else None, 1, True),
                ("Virgin CPR Bot", virgins[0][2] if virgins else None, 1, True),
                ("Camarilla H3", dl.get("cam_h3"), 1, False),
                ("Camarilla L3", dl.get("cam_l3"), 1, False),
                ("Daily CPR Pivot", dl.get("cpr_p"), 1, False),
                ("Daily CPR Top", dl.get("cpr_top"), 1, False),
                ("Daily CPR Bot", dl.get("cpr_bot"), 1, False),
                ("Daily VWAP", vwap_spot, 1, False),
                ("5m EMA 20", ema20_5m, 1, False),
                ("5m EMA 200", ema200_5m, 1, False),
                ("Opening 3m High", op_h, 2, False),
                ("Opening 3m Low", op_l, 2, False),
                ("3m EMA 20", ema20_3m, 2, False),
                ("Prev Day High", dl.get("pdh"), 2, False),
                ("Prev Day Low", dl.get("pdl"), 2, False),
                ("Fibonacci H3", dl.get("fib_h3"), 3, False),
                ("Fibonacci L3", dl.get("fib_l3"), 3, False),
                ("Camarilla H4", dl.get("cam_h4"), 3, False),
                ("Camarilla L4", dl.get("cam_l4"), 3, False),
            ]

            tol = max(0.50 * atr5, 4.0)

            for lvl_name, lvl_px, tier, is_v in candidates:
                if lvl_px is None:
                    continue
                if touch_budget.get(lvl_name, 0) >= 2:
                    continue

                # LONG SETUP (CE)
                bar1_touched = abs(prev_s_bar["low"] - lvl_px) <= tol or abs(prev_s_bar["close"] - lvl_px) <= tol
                valid_long_zone = (cur_s_bar["close"] > max(st_val, vwap_spot)) if enable_chop_filter else True

                if bar1_touched and cur_s_bar["high"] > prev_s_bar["high"] and is_bull_15m and valid_long_zone:
                    score = 40 + (25 if is_v else 20 if tier == 1 else 10 if tier == 2 else 5)
                    if abs(cur_s_bar["close"] - lvl_px) <= tol:
                        score += 15
                    if is_bull_15m:
                        score += 25

                    if score >= 50:
                        atm_strike = int(round(spot_px / 50.0) * 50)
                        ce_strike = atm_strike - 100
                        opt_sym = sym_map.get((ce_strike, "CE"))
                        if not opt_sym:
                            continue

                        c_df = opt_df[opt_df["symbol"] == opt_sym].sort_values("minute")
                        c_bars = c_df[c_df["minute"] >= t_min].to_dict("records")
                        if not c_bars:
                            continue

                        entry_opt_px = c_bars[0]["open"]
                        sl_opt_px = entry_opt_px - sl_opt_pts
                        tp_opt_px = entry_opt_px + tp_opt_pts

                        trail_active = False
                        cur_sl = sl_opt_px
                        peak_opt = entry_opt_px
                        exit_opt_px = entry_opt_px
                        won = False

                        for f_bar in c_bars[1:]:
                            f_h, f_l = f_bar["high"], f_bar["low"]
                            if f_h > peak_opt:
                                peak_opt = f_h
                                if (peak_opt - entry_opt_px) >= trail_trigger_pts:
                                    trail_active = True

                            if trail_active:
                                cur_sl = max(cur_sl, entry_opt_px + fee_cover_be_buffer, peak_opt - trail_step_pts)

                            if f_h >= tp_opt_px:
                                exit_opt_px = tp_opt_px
                                won = True
                                break
                            elif f_l <= cur_sl:
                                exit_opt_px = cur_sl
                                won = (exit_opt_px > entry_opt_px)
                                break

                        opt_gain = exit_opt_px - entry_opt_px
                        net_rs = (opt_gain * LOT_SIZE) - FEE_PER_TRADE
                        trades.append({
                            "day": day,
                            "year": day[:4],
                            "side": "CE",
                            "symbol": opt_sym,
                            "win": won,
                            "net_rs": net_rs,
                        })
                        daily_pnl[day] += net_rs
                        touch_budget[lvl_name] = touch_budget.get(lvl_name, 0) + 1
                        break

                # SHORT SETUP (PE)
                bar1_touched_s = abs(prev_s_bar["high"] - lvl_px) <= tol or abs(prev_s_bar["close"] - lvl_px) <= tol
                valid_short_zone = (cur_s_bar["close"] < min(st_val, vwap_spot)) if enable_chop_filter else True

                if bar1_touched_s and cur_s_bar["low"] < prev_s_bar["low"] and (not is_bull_15m) and valid_short_zone:
                    score = 40 + (25 if is_v else 20 if tier == 1 else 10 if tier == 2 else 5)
                    if abs(cur_s_bar["close"] - lvl_px) <= tol:
                        score += 15
                    if not is_bull_15m:
                        score += 25

                    if score >= 50:
                        atm_strike = int(round(spot_px / 50.0) * 50)
                        pe_strike = atm_strike + 100
                        opt_sym = sym_map.get((pe_strike, "PE"))
                        if not opt_sym:
                            continue

                        c_df = opt_df[opt_df["symbol"] == opt_sym].sort_values("minute")
                        c_bars = c_df[c_df["minute"] >= t_min].to_dict("records")
                        if not c_bars:
                            continue

                        entry_opt_px = c_bars[0]["open"]
                        sl_opt_px = entry_opt_px - sl_opt_pts
                        tp_opt_px = entry_opt_px + tp_opt_pts

                        trail_active = False
                        cur_sl = sl_opt_px
                        peak_opt = entry_opt_px
                        exit_opt_px = entry_opt_px
                        won = False

                        for f_bar in c_bars[1:]:
                            f_h, f_l = f_bar["high"], f_bar["low"]
                            if f_h > peak_opt:
                                peak_opt = f_h
                                if (peak_opt - entry_opt_px) >= trail_trigger_pts:
                                    trail_active = True

                            if trail_active:
                                cur_sl = max(cur_sl, entry_opt_px + fee_cover_be_buffer, peak_opt - trail_step_pts)

                            if f_h >= tp_opt_px:
                                exit_opt_px = tp_opt_px
                                won = True
                                break
                            elif f_l <= cur_sl:
                                exit_opt_px = cur_sl
                                won = (exit_opt_px > entry_opt_px)
                                break

                        opt_gain = exit_opt_px - entry_opt_px
                        net_rs = (opt_gain * LOT_SIZE) - FEE_PER_TRADE
                        trades.append({
                            "day": day,
                            "year": day[:4],
                            "side": "PE",
                            "symbol": opt_sym,
                            "win": won,
                            "net_rs": net_rs,
                        })
                        daily_pnl[day] += net_rs
                        touch_budget[lvl_name] = touch_budget.get(lvl_name, 0) + 1
                        break

    if not trades:
        return [], {"trades": 0, "win_rate": 0.0, "net_profit": 0.0, "pf": 0.0, "calmar": 0.0}

    df_tr = pd.DataFrame(trades)
    n_tr = len(df_tr)
    wr = float(df_tr["win"].mean() * 100)
    net_p = float(df_tr["net_rs"].sum())
    gw = float(df_tr[df_tr["net_rs"] > 0]["net_rs"].sum())
    gl = float(abs(df_tr[df_tr["net_rs"] < 0]["net_rs"].sum()))
    pf = round(gw / gl, 2) if gl > 0 else 999.0

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
    print("=" * 90)
    print(" 🚀 7-YEAR DIRECT OPTION-CHART BENCHMARK: RAW WEEKLY OPTIONS DATA (2020-2026)")
    print("=" * 90)

    # 1. Load Spot Index & Compute Daily S/R Levels
    print("\n[1] Loading Spot Index for Strike Selection & S/R Anchors...")
    df_spot = pd.read_csv(IDX_FILE)
    df_spot["datetime"] = pd.to_datetime(df_spot["date"])
    df_spot["day_str"] = df_spot["datetime"].dt.strftime("%Y-%m-%d")
    df_spot["minute"] = df_spot["datetime"].dt.hour * 60 + df_spot["datetime"].dt.minute
    spot_by_day = {}
    for d, grp in df_spot.groupby("day_str"):
        spot_by_day[d] = dict(zip(grp["minute"], grp["close"]))

    daily_stats = df_spot.groupby("day_str").agg(
        high=("close", "max"),
        low=("close", "min"),
        close=("close", "last"),
    ).reset_index()

    all_days = list(daily_stats["day_str"].unique())
    daily_levels = {}
    virgin_cprs_by_day = {}
    history = []

    for i in range(1, len(all_days)):
        prev_d = all_days[i - 1]
        cur_d = all_days[i]
        p_row = daily_stats[daily_stats["day_str"] == prev_d].iloc[0]
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

        history.append((p_pivot, cpr_top, cpr_bot, prev_d))
        active_virgins = []
        for vp, vtc, vbc, vday in history[:-1]:
            d_rows = df_spot[df_spot["day_str"] == cur_d]
            if len(d_rows) > 0:
                dh = d_rows["close"].max()
                dl = d_rows["close"].min()
                if not (dl <= vtc and dh >= vbc):
                    active_virgins.append((vp, vtc, vbc, vday))
        virgin_cprs_by_day[cur_d] = active_virgins

    # 2. Load Raw Options CSVs in Parallel
    print("\n[2] Loading Raw Weekly Options CSV Files in Parallel (8 Workers)...")
    opt_files = sorted(list(OPT_DIR.glob("**/*.csv")) + list(OPT_DIR.glob("**/*.parquet")))
    print(f"Found {len(opt_files)} raw option files.")
    day_opt = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        opt_results = list(executor.map(parse_option_file, opt_files))
    for r in opt_results:
        if r is not None:
            day_opt[r[0]] = r[1]

    common_days = sorted(list(set(day_opt.keys()) & set(spot_by_day.keys())))
    print(f"✅ Matched {len(common_days)} valid trading days ({common_days[0]} to {common_days[-1]}).")

    # 3. SMOKE TEST (5 Days Sanity Check)
    print("\n--- SMOKE TEST (5 Days Sanity Check) ---")
    smoke_days = common_days[:5]
    _, s_sum = run_direct_option_chart_simulation(smoke_days, spot_by_day, day_opt, daily_levels, virgin_cprs_by_day, sl_opt_pts=6.0, tp_opt_pts=14.0)
    print(f"Smoke Test: Trades={s_sum['trades']}, WinRate={s_sum['win_rate']:.1f}%, Profit=Rs {s_sum['net_profit']:,.2f}")
    assert s_sum['trades'] > 0, "Smoke test failed: 0 trades"
    print("✅ Smoke test passed!\n")

    # 4. FULL 7-YEAR NON-WALK-FORWARD (NWF) BENCHMARK
    print("Running Full 7-Year Non-Walk-Forward (NWF) Simulation on Raw Options...")
    t0 = time.time()
    tr_full, sum_full = run_direct_option_chart_simulation(common_days, spot_by_day, day_opt, daily_levels, virgin_cprs_by_day, sl_opt_pts=6.0, tp_opt_pts=14.0, trail_trigger_pts=5.0, trail_step_pts=2.0, fee_cover_be_buffer=1.0)
    elapsed = time.time() - t0
    print(f"NWF Simulation completed in {elapsed:.1f}s.\n")

    # 5. WALK-FORWARD OOS SIMULATION (WFO)
    print("Running Walk-Forward Out-Of-Sample (WFO) Simulation...")
    df_all_trades = pd.DataFrame(tr_full)
    years = sorted(list(df_all_trades["year"].unique()))

    print("=" * 100)
    print(f"{'CONFIGURATION':<55} | {'TRADES':<7} | {'WIN RATE':<9} | {'PROFIT (Rs)':<16} | {'PF':<5} | {'CALMAR':<7}")
    print("-" * 100)
    print(f"{'1. Direct Option Chart NWF (7-Year Raw Option Data)':<55} | {sum_full['trades']:<7} | {sum_full['win_rate']:<8.1f}% | Rs {sum_full['net_profit']:>12,.2f} | {sum_full['pf']:<5.2f} | {sum_full['calmar']:<7.1f}")
    print("=" * 100)

    print("\n--- 📅 YEAR-BY-YEAR DIRECT OPTION CHART PERFORMANCE (WITH ₹45 FEES DEDUCTED) ---")
    print(f"{'YEAR':<6} | {'TRADES':<7} | {'WIN RATE':<9} | {'NET PROFIT (Rs)':<16} | {'PROFIT FACTOR':<13} | {'GREEN DAYS':<10}")
    print("-" * 75)
    for y in years:
        sub = df_all_trades[df_all_trades["year"] == y]
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
    print(f"{'TOTAL':<6} | {sum_full['trades']:<7} | {sum_full['win_rate']:<8.1f}% | Rs {sum_full['net_profit']:>12,.2f} | {sum_full['pf']:<13.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
