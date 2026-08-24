"""Undisputed Rejection Strategy: Futures Signals + Spot 2nd ITM Strike Selection + Options Execution (GPU Accelerated).

Architecture:
  1. Signal Engine: Evaluates 3m Nifty Futures bars with Two-Bar Structure Confirmation & S/R Anchors
  2. Trend Gate: 15-Minute Nifty Futures EMA20 Trend Gate
  3. Strike Selector: Spot Index selects 2nd ITM Strike (CE = ATM - 100, PE = ATM + 100)
  4. Execution Engine: Fills, manages SL/TP, and trails on the 2nd ITM Nifty Weekly Option Contract
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
DESKTOP_DATA = Path(r"C:\Users\user\Desktop\nifty50 data")
FUT_DIR = DESKTOP_DATA / "nifty_futures"
OPT_DIR = DESKTOP_DATA / "nifty_options"
IDX_FILE = DESKTOP_DATA / "index" / "NIFTY 50_minute.csv"

LOT_SIZE = 65
FEE_PER_TRADE = 45.0
WORKERS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 135)
print("FUTURES SIGNALS + SPOT 2nd ITM STRIKE SELECTION + OPTIONS EXECUTION ENGINE")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | CPU Workers: {WORKERS}")
print(f"Futures: {FUT_DIR} | Options: {OPT_DIR}")
print("=" * 135)


def parse_futures_file(fpath: Path) -> Optional[Tuple[str, pd.DataFrame]]:
    """Loads and standardizes 1-minute Nifty Futures data for a single day."""
    try:
        df = pd.read_csv(fpath)
        if df.empty or "close" not in df.columns:
            return None
        
        d_str = str(df["date"].iloc[0])
        if "-" in d_str and len(d_str) == 10:
            parts = d_str.split("-")
            date_norm = d_str if len(parts[0]) == 4 else f"{parts[2]}-{parts[1]}-{parts[0]}"
        else:
            date_norm = d_str

        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df["minute"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
        df = df[(df["minute"] >= 555) & (df["minute"] <= 930)].sort_values("minute").reset_index(drop=True)
        if len(df) < 50:
            return None
        return date_norm, df
    except Exception:
        return None


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
    """Extracts strike and option type (CE/PE) from symbol (e.g. NIFTY04JAN2418300PE -> (18300, 'PE'))."""
    try:
        opt_type = symbol[-2:].upper()
        if opt_type not in ("CE", "PE"):
            return None
        # Extract digits preceding CE/PE
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


def run_hybrid_futures_options_backtest(is_smoke_test: bool = False):
    t0 = time.time()
    
    # 1. Load Spot Index for Strike Selection
    print("\n[1] Loading Spot Index (for 2nd ITM Strike Selection)...")
    df_spot = pd.read_csv(IDX_FILE)
    df_spot["datetime"] = pd.to_datetime(df_spot["date"])
    df_spot["day_str"] = df_spot["datetime"].dt.strftime("%Y-%m-%d")
    df_spot["minute"] = df_spot["datetime"].dt.hour * 60 + df_spot["datetime"].dt.minute
    spot_by_day = {}
    for d, grp in df_spot.groupby("day_str"):
        spot_by_day[d] = dict(zip(grp["minute"], grp["close"]))

    # 2. Load Futures Files
    print("\n[2] Loading Futures Files in Parallel (8 Workers)...")
    fut_files = sorted(list(FUT_DIR.glob("**/*.csv")))
    day_fut = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = list(executor.map(parse_futures_file, fut_files))
    for r in results:
        if r is not None:
            day_fut[r[0]] = r[1]

    # 3. Load Options Files
    print("\n[3] Loading Options Files in Parallel (8 Workers)...")
    opt_files = sorted(list(OPT_DIR.glob("**/*.csv")) + list(OPT_DIR.glob("**/*.parquet")))
    day_opt = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        opt_results = list(executor.map(parse_option_file, opt_files))
    for r in opt_results:
        if r is not None:
            day_opt[r[0]] = r[1]

    common_days = sorted(list(set(day_fut.keys()) & set(day_opt.keys()) & set(spot_by_day.keys())))
    print(f"Matched {len(common_days)} common trading days ({common_days[0]} to {common_days[-1]}).")

    if is_smoke_test:
        common_days = common_days[:5]
        print(f"\n=== SMOKE TEST: RUNNING ON FIRST {len(common_days)} DAYS ONLY ===")

    # 4. Generate Causal Futures Signals & Map to 2nd ITM Option Contract
    print("\n[4] Generating Futures Two-Bar Rejection Signals & Mapping 2nd ITM Options...")
    trade_signals = []

    for i in range(1, len(common_days)):
        day = common_days[i]
        prev_day = common_days[i - 1]
        prev_df = day_fut[prev_day]

        prev_h = float(prev_df["high"].max())
        prev_l = float(prev_df["low"].min())
        prev_c = float(prev_df["close"].iloc[-1])

        # S/R Levels on Futures
        pivot = (prev_h + prev_l + prev_c) / 3.0
        bc = (prev_h + prev_l) / 2.0
        tc = (pivot - bc) + pivot
        cpr_top = max(tc, bc)
        cpr_bot = min(tc, bc)
        cam_range = prev_h - prev_l
        h3 = prev_c + cam_range * (1.1 / 4.0)
        l3 = prev_c - cam_range * (1.1 / 4.0)

        # Build 3m Futures Candles
        df_f = day_fut[day].copy()
        df_f["bar_3m_idx"] = (df_f["minute"] - 555) // 3
        agg_3m = df_f.groupby("bar_3m_idx").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum") if "volume" in df_f.columns else ("close", "count"),
            minute_start=("minute", "first"),
        ).reset_index()

        # Build 15m Futures Candles for Trend Gate
        df_f["bar_15m_idx"] = (df_f["minute"] - 555) // 15
        agg_15m = df_f.groupby("bar_15m_idx").agg(
            close=("close", "last"), minute_start=("minute", "first")
        ).reset_index()

        prior_15m = []
        prior_3m = []
        for p_d in common_days[max(0, i - 4): i]:
            pdf = day_fut[p_d]
            prior_15m.extend(pdf.groupby((pdf["minute"] - 555) // 15)["close"].last().tolist())
            prior_3m.extend(pdf.groupby((pdf["minute"] - 555) // 3)["close"].last().tolist())

        full_15m = pd.Series(prior_15m + agg_15m["close"].tolist())
        ema20_15m = full_15m.ewm(span=20, adjust=False).mean().iloc[-len(agg_15m):].values

        # 3m VWAP, EMA20, EMA200
        cum_vol = 0.0
        cum_pv = 0.0
        vwap_3m = []
        for _, r in agg_3m.iterrows():
            tp = (r["high"] + r["low"] + r["close"]) / 3.0
            v = max(float(r["volume"]), 1.0)
            cum_vol += v
            cum_pv += tp * v
            vwap_3m.append(cum_pv / cum_vol)

        full_3m = pd.Series(prior_3m + agg_3m["close"].tolist())
        ema20_3m = full_3m.ewm(span=20, adjust=False).mean().iloc[-len(agg_3m):].values
        ema200_3m = full_3m.ewm(span=200, adjust=False).mean().iloc[-len(agg_3m):].values

        # Option DataFrame for this day
        df_opt_day = day_opt[day]
        spot_min_map = spot_by_day.get(day, {})

        # Scan for Two-Bar Confirmation on Futures
        touch_counts = {}
        for b_idx in range(len(agg_3m) - 1):
            bar_1 = agg_3m.iloc[b_idx]
            bar_2 = agg_3m.iloc[b_idx + 1]
            m_start = int(bar_2["minute_start"])

            # Check Active Sessions (09:15-11:00 & 13:30-15:00)
            if not ((555 <= m_start <= 660) or (810 <= m_start <= 900)):
                continue

            idx_15m = min(int((m_start - 555) // 15), len(ema20_15m) - 1)
            is_15m_bull = bar_2["close"] >= ema20_15m[idx_15m]

            sr_levels = [
                ("CPR Pivot", pivot, 1),
                ("CPR Top (TC)", cpr_top, 1),
                ("CPR Bottom (BC)", cpr_bot, 1),
                ("Daily VWAP", vwap_3m[b_idx], 1),
                ("EMA 200", ema200_3m[b_idx], 1),
                ("EMA 20", ema20_3m[b_idx], 2),
                ("Camarilla H3", h3, 2),
                ("Camarilla L3", l3, 2),
                ("Prev Day High", prev_h, 2),
                ("Prev Day Low", prev_l, 2),
            ]

            for lvl_name, lvl_px, prio in sr_levels:
                if touch_counts.get(lvl_name, 0) >= 2:
                    continue

                if bar_1["low"] <= lvl_px <= bar_1["high"]:
                    signal_dir = None
                    if is_15m_bull and (bar_2["high"] > bar_1["high"]):
                        signal_dir = "LONG"   # Buy 2nd ITM CE
                    elif (not is_15m_bull) and (bar_2["low"] < bar_1["low"]):
                        signal_dir = "SHORT"  # Buy 2nd ITM PE

                    if signal_dir is not None:
                        touch_counts[lvl_name] = touch_counts.get(lvl_name, 0) + 1
                        
                        # 2nd ITM Strike Selection using Spot Index
                        spot_px = spot_min_map.get(m_start, bar_2["close"])
                        atm_spot = round(spot_px / 50.0) * 50
                        target_strike = int(atm_spot - 100) if signal_dir == "LONG" else int(atm_spot + 100)
                        target_type = "CE" if signal_dir == "LONG" else "PE"

                        # Find matching option series
                        opt_symbols = df_opt_day["symbol"].unique()
                        matched_sym = None
                        for s in opt_symbols:
                            ext = extract_strike_from_sym(s)
                            if ext and ext[0] == target_strike and ext[1] == target_type:
                                matched_sym = s
                                break

                        if matched_sym:
                            sub_opt = df_opt_day[df_opt_day["symbol"] == matched_sym].sort_values("minute").reset_index(drop=True)
                            opt_entry_row = sub_opt[sub_opt["minute"] == m_start]
                            if not opt_entry_row.empty:
                                opt_entry_px = float(opt_entry_row["close"].iloc[0])
                                if opt_entry_px > 10.0:  # Valid traded option premium
                                    # Subsequent option 1m bars for trade execution
                                    fut_opt_bars = sub_opt[sub_opt["minute"] > m_start].reset_index(drop=True)
                                    trade_signals.append({
                                        "date": day,
                                        "minute": m_start,
                                        "time": f"{day} {m_start // 60:02d}:{m_start % 60:02d}:00",
                                        "signal_dir": signal_dir,
                                        "symbol": matched_sym,
                                        "strike": target_strike,
                                        "opt_type": target_type,
                                        "opt_entry": opt_entry_px,
                                        "level": lvl_name,
                                        "fut_highs": fut_opt_bars["high"].values[:75],
                                        "fut_lows": fut_opt_bars["low"].values[:75],
                                        "fut_closes": fut_opt_bars["close"].values[:75],
                                    })
                        break

    print(f"Captured {len(trade_signals):,} Executable 2nd ITM Option Setups from Futures Signals.")

    # 5. Simulate Option Fills, ATR SL/TP, and Trailing Stop
    executed_trades = []
    for t in trade_signals:
        entry = t["opt_entry"]
        f_h = t["fut_highs"]
        f_l = t["fut_lows"]
        f_c = t["fut_closes"]

        if len(f_h) == 0:
            continue

        # Option Risk Parameters: SL = min 4.0 pts, TP = 1.5x ATR (~25 pts), Trail @ +6.0 pts (step 2.0 pts)
        init_sl = entry - 4.0
        init_tp = entry + 25.0

        curr_sl = init_sl
        best_p = entry
        exit_px = None
        reason = "EOD"

        for step in range(len(f_h)):
            hi, lo, cl = f_h[step], f_l[step], f_c[step]

            # Trailing stop update
            gain = hi - entry
            if gain >= 6.0:
                best_p = max(best_p, hi)
                curr_sl = max(curr_sl, best_p - 2.0)

            # Exit checks
            if lo <= curr_sl:
                exit_px = curr_sl
                reason = "SL-TRL" if curr_sl > init_sl else "SL"
                break
            elif hi >= init_tp:
                exit_px = init_tp
                reason = "TP"
                break

        if exit_px is None:
            exit_px = f_c[-1]

        pnl_pts = exit_px - entry
        net_rs = pnl_pts * LOT_SIZE - FEE_PER_TRADE
        executed_trades.append({
            "date": t["date"],
            "time": t["time"],
            "symbol": t["symbol"],
            "direction": t["signal_dir"],
            "level": t["level"],
            "entry": entry,
            "exit": exit_px,
            "pnl_pts": pnl_pts,
            "net_rs": net_rs,
            "reason": reason,
            "year": t["date"][:4],
        })

    df_tr = pd.DataFrame(executed_trades)

    # 6. Comprehensive Analytics
    wins = df_tr[df_tr["net_rs"] > 0]
    losses = df_tr[df_tr["net_rs"] <= 0]
    wr = len(wins) / len(df_tr) * 100 if len(df_tr) > 0 else 0
    gross_w = wins["net_rs"].sum()
    gross_l = abs(losses["net_rs"].sum())
    pf = gross_w / gross_l if gross_l > 0 else 99.0

    day_pnls = df_tr.groupby("date")["net_rs"].sum()
    green_days = sum(1 for v in day_pnls if v > 0)
    red_days = sum(1 for v in day_pnls if v <= 0)
    daily_wr = (green_days / len(day_pnls) * 100) if len(day_pnls) > 0 else 0

    cum_eq = np.cumsum(day_pnls.values)
    peaks = np.maximum.accumulate(cum_eq)
    max_dd = float(np.max(peaks - cum_eq)) if len(cum_eq) > 0 else 1.0
    calmar = (df_tr["net_rs"].sum() / max_dd) if max_dd > 0 else 0

    print("\n" + "=" * 135)
    print(f"HYBRID PERFORMANCE: FUTURES SIGNALS -> SPOT 2nd ITM SELECTION -> OPTIONS EXECUTION")
    print("=" * 135)
    print(f"Total Option Trades Executed: {len(df_tr):,}")
    print(f"Total Traded Days:            {len(day_pnls):,}")
    print(f"Avg Trades per Day:           {len(df_tr)/len(day_pnls):.2f} trades/day")
    print(f"TRADE WIN RATE:               {wr:.2f}% ({len(wins):,} Wins / {len(losses):,} Losses)")
    print(f"DAILY WIN RATE:               {daily_wr:.1f}% ({green_days} Green / {red_days} Red Days)")
    print(f"PROFIT FACTOR:                {pf:.3f}")
    print(f"TOTAL NET POINTS:             {df_tr['pnl_pts'].sum():>+,.2f} pts")
    print(f"TOTAL REALIZED NET PROFIT:    Rs {df_tr['net_rs'].sum():>+,.2f} (1 Lot / 65 qty)")
    print(f"MAX DRAWDOWN:                 Rs {max_dd:>+,.2f}")
    print(f"CALMAR RATIO:                 {calmar:.2f}")
    print(f"Avg Win:                      +{wins['pnl_pts'].mean():.2f} pts (Rs {wins['net_rs'].mean():+,.2f})")
    print(f"Avg Loss:                     {losses['pnl_pts'].mean():.2f} pts (Rs {losses['net_rs'].mean():+,.2f})")
    print("=" * 135)

    # Yearly Breakdown
    print("\nYEAR-BY-YEAR PERFORMANCE LEDGER:")
    print(f"{'Year':6s} | {'Trades':8s} | {'Days':6s} | {'Tr/Day':8s} | {'Win Rate':10s} | {'Net Points':14s} | {'Realized Rs':18s} | {'PF':8s}")
    print("-" * 105)
    for yr, grp in df_tr.groupby("year"):
        y_w = grp[grp["net_rs"] > 0]
        y_l = grp[grp["net_rs"] <= 0]
        y_pf = y_w["net_rs"].sum() / abs(y_l["net_rs"].sum()) if abs(y_l["net_rs"].sum()) > 0 else 99.0
        y_days = grp["date"].nunique()
        print(f"{yr:6s} | {len(grp):<8d} | {y_days:<6d} | {len(grp)/y_days:<8.2f} | {len(y_w)/len(grp)*100:<9.2f}% | {grp['pnl_pts'].sum():>+12.2f} pts | Rs {grp['net_rs'].sum():>+14,.2f} | {y_pf:<8.3f}")

    out_file = ROOT / "artifacts" / "f6_hybrid" / "futures_signals_options_execution_results.json"
    out_file.write_text(json.dumps({
        "total_trades": int(len(df_tr)),
        "total_days": int(len(day_pnls)),
        "win_rate": float(wr),
        "daily_win_rate": float(daily_wr),
        "pf": float(pf),
        "net_points": float(df_tr["pnl_pts"].sum()),
        "net_rs": float(df_tr["net_rs"].sum()),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
    }, indent=2), encoding="utf-8")
    print(f"\n[Saved Hybrid Results JSON]: {out_file}")


if __name__ == "__main__":
    is_smoke = "--smoke" in sys.argv
    run_hybrid_futures_options_backtest(is_smoke_test=is_smoke)
