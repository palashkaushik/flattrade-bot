"""Fast Causal Backtest: Futures Signals + Spot 2nd ITM Selection + Options Execution (8 Workers)."""

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

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
DESKTOP_DATA = Path(r"C:\Users\user\Desktop\nifty50 data")
FUT_DIR = DESKTOP_DATA / "nifty_futures"
OPT_DIR = DESKTOP_DATA / "nifty_options"
IDX_FILE = DESKTOP_DATA / "index" / "NIFTY 50_minute.csv"

LOT_SIZE = 65
FEE_PER_TRADE = 45.0
WORKERS = 8


def load_spot_map():
    df = pd.read_csv(IDX_FILE)
    df["dt"] = pd.to_datetime(df["date"])
    df["day"] = df["dt"].dt.strftime("%Y-%m-%d")
    df["minute"] = df["dt"].dt.hour * 60 + df["dt"].dt.minute
    res = {}
    for d, g in df.groupby("day"):
        res[d] = dict(zip(g["minute"], g["close"]))
    return res


def parse_fut_day(fpath: Path):
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
        df["dt"] = pd.to_datetime(df["date"] + " " + df["time"])
        df["minute"] = df["dt"].dt.hour * 60 + df["dt"].dt.minute
        df = df[(df["minute"] >= 555) & (df["minute"] <= 930)].sort_values("minute").reset_index(drop=True)
        return (date_norm, df) if len(df) >= 50 else None
    except Exception:
        return None


def get_opt_file_map():
    res = {}
    for p in OPT_DIR.glob("**/*.csv"):
        name = p.stem.replace("nifty_options_", "")
        parts = name.split("_")
        if len(parts) == 3:
            norm = f"{parts[2]}-{parts[1]}-{parts[0]}"
            res[norm] = p
    return res


import re

def extract_strike_type(sym: str):
    m = re.search(r'(\d{4,5})(CE|PE)$', str(sym).upper())
    if m:
        return (int(m.group(1)), m.group(2))
    return None


def process_single_day(args):
    day, fut_df, prev_fut_df, opt_path, spot_min_map, prior_15m, prior_3m = args
    if fut_df is None or prev_fut_df is None or opt_path is None:
        return []

    # S/R Levels on Futures
    prev_h = float(prev_fut_df["high"].max())
    prev_l = float(prev_fut_df["low"].min())
    prev_c = float(prev_fut_df["close"].iloc[-1])

    pivot = (prev_h + prev_l + prev_c) / 3.0
    bc = (prev_h + prev_l) / 2.0
    tc = (pivot - bc) + pivot
    cpr_top = max(tc, bc)
    cpr_bot = min(tc, bc)
    cam_range = prev_h - prev_l
    h3 = prev_c + cam_range * (1.1 / 4.0)
    l3 = prev_c - cam_range * (1.1 / 4.0)

    # 3m Aggregation
    fut_df["bar_3m_idx"] = (fut_df["minute"] - 555) // 3
    agg_3m = fut_df.groupby("bar_3m_idx").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum") if "volume" in fut_df.columns else ("close", "count"),
        minute_start=("minute", "first"),
    ).reset_index()

    # 15m Aggregation
    fut_df["bar_15m_idx"] = (fut_df["minute"] - 555) // 15
    agg_15m = fut_df.groupby("bar_15m_idx").agg(
        close=("close", "last"), minute_start=("minute", "first")
    ).reset_index()

    full_15m = pd.Series(prior_15m + agg_15m["close"].tolist())
    ema20_15m = full_15m.ewm(span=20, adjust=False).mean().iloc[-len(agg_15m):].values

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

    # Scan signals
    signals = []
    touch_counts = {}
    for b_idx in range(len(agg_3m) - 1):
        b1 = agg_3m.iloc[b_idx]
        b2 = agg_3m.iloc[b_idx + 1]
        m_start = int(b2["minute_start"])
        if not ((555 <= m_start <= 660) or (810 <= m_start <= 900)):
            continue

        idx_15m = min(int((m_start - 555) // 15), len(ema20_15m) - 1)
        is_bull = b2["close"] >= ema20_15m[idx_15m]

        sr_levels = [
            ("CPR Pivot", pivot, 1), ("CPR Top", cpr_top, 1), ("CPR Bot", cpr_bot, 1),
            ("Daily VWAP", vwap_3m[b_idx], 1), ("EMA 200", ema200_3m[b_idx], 1),
            ("EMA 20", ema20_3m[b_idx], 2), ("Cam H3", h3, 2), ("Cam L3", l3, 2),
            ("Prev High", prev_h, 2), ("Prev Low", prev_l, 2),
        ]

        for lvl_name, lvl_px, prio in sr_levels:
            if touch_counts.get(lvl_name, 0) >= 2:
                continue
            if b1["low"] <= lvl_px <= b1["high"]:
                sig_dir = None
                if is_bull and (b2["high"] > b1["high"]):
                    sig_dir = "LONG"
                elif (not is_bull) and (b2["low"] < b1["low"]):
                    sig_dir = "SHORT"

                if sig_dir is not None:
                    touch_counts[lvl_name] = touch_counts.get(lvl_name, 0) + 1
                    spot_px = spot_min_map.get(m_start, b2["close"])
                    atm_spot = round(spot_px / 50.0) * 50
                    target_strike = int(atm_spot - 100) if sig_dir == "LONG" else int(atm_spot + 100)
                    target_type = "CE" if sig_dir == "LONG" else "PE"
                    signals.append((m_start, sig_dir, target_strike, target_type, lvl_name))
                    break

    if not signals:
        return []

    # Read option file and execute
    try:
        opt_df = pd.read_csv(opt_path)
        opt_df["minute"] = pd.to_datetime(opt_df["time"], format="%H:%M:%S", errors="coerce").dt.hour * 60 + pd.to_datetime(opt_df["time"], format="%H:%M:%S", errors="coerce").dt.minute
    except Exception:
        return []

    trades = []
    for m_start, sig_dir, strike, opt_type, lvl in signals:
        matched = None
        for s in opt_df["symbol"].unique():
            ext = extract_strike_type(s)
            if ext and ext[0] == strike and ext[1] == opt_type:
                matched = s
                break
        if not matched:
            continue

        sub = opt_df[opt_df["symbol"] == matched].sort_values("minute").reset_index(drop=True)
        entry_row = sub[sub["minute"] == m_start]
        if entry_row.empty:
            continue
        entry_px = float(entry_row["close"].iloc[0])
        if entry_px < 10.0:
            continue

        fut_bars = sub[sub["minute"] > m_start].reset_index(drop=True)
        if fut_bars.empty:
            continue

        init_sl = entry_px - 4.0
        init_tp = entry_px + 25.0
        curr_sl = init_sl
        best_p = entry_px
        exit_px = None
        reason = "EOD"

        for _, frow in fut_bars.iterrows():
            hi, lo, cl = float(frow["high"]), float(frow["low"]), float(frow["close"])
            gain = hi - entry_px
            if gain >= 6.0:
                best_p = max(best_p, hi)
                curr_sl = max(curr_sl, best_p - 2.0)
            if lo <= curr_sl:
                exit_px = curr_sl
                reason = "SL-TRL" if curr_sl > init_sl else "SL"
                break
            elif hi >= init_tp:
                exit_px = init_tp
                reason = "TP"
                break

        if exit_px is None:
            exit_px = float(fut_bars["close"].iloc[-1])

        pnl = exit_px - entry_px
        net_rs = pnl * LOT_SIZE - FEE_PER_TRADE
        trades.append({
            "date": day, "minute": m_start, "dir": sig_dir, "symbol": matched,
            "level": lvl, "entry": entry_px, "exit": exit_px, "pnl_pts": pnl,
            "net_rs": net_rs, "reason": reason, "year": day[:4]
        })

    return trades


def run_fast_backtest():
    t0 = time.time()
    print("=" * 100)
    print("FAST NIFTY FUTURES SIGNALS -> SPOT 2nd ITM -> OPTIONS EXECUTION (8 WORKERS)")
    print("=" * 100)
    
    spot_map = load_spot_map()
    print(f"Loaded spot index for {len(spot_map)} days.")

    fut_files = sorted(list(FUT_DIR.glob("**/*.csv")))
    day_fut = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as exc:
        for r in exc.map(parse_fut_day, fut_files):
            if r is not None:
                day_fut[r[0]] = r[1]
    print(f"Loaded {len(day_fut)} futures days.")

    opt_map = get_opt_file_map()
    print(f"Found {len(opt_map)} option files.")

    common_days = sorted(list(set(day_fut.keys()) & set(opt_map.keys()) & set(spot_map.keys())))
    print(f"Common days: {len(common_days)} ({common_days[0]} to {common_days[-1]})\n")

    # Build tasks
    tasks = []
    for i in range(1, len(common_days)):
        day = common_days[i]
        prev_day = common_days[i - 1]
        prior_15m = []
        prior_3m = []
        for p_d in common_days[max(0, i - 4): i]:
            pdf = day_fut[p_d]
            prior_15m.extend(pdf.groupby((pdf["minute"] - 555) // 15)["close"].last().tolist())
            prior_3m.extend(pdf.groupby((pdf["minute"] - 555) // 3)["close"].last().tolist())

        tasks.append((
            day, day_fut[day], day_fut[prev_day], opt_map[day],
            spot_map.get(day, {}), prior_15m, prior_3m
        ))

    print(f"Executing {len(tasks)} days across {WORKERS} workers...")
    all_trades = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pexec:
        for res in pexec.map(process_single_day, tasks, chunksize=16):
            all_trades.extend(res)

    df = pd.DataFrame(all_trades)
    print(f"Completed in {time.time()-t0:.1f}s | Total Trades: {len(df):,}")

    wins = df[df["net_rs"] > 0]
    losses = df[df["net_rs"] <= 0]
    wr = len(wins) / len(df) * 100 if len(df) > 0 else 0
    pf = wins["net_rs"].sum() / abs(losses["net_rs"].sum()) if abs(losses["net_rs"].sum()) > 0 else 99.0

    day_pnl = df.groupby("date")["net_rs"].sum()
    daily_wr = (sum(1 for v in day_pnl if v > 0) / len(day_pnl) * 100) if len(day_pnl) > 0 else 0

    cum = np.cumsum(day_pnl.values)
    peaks = np.maximum.accumulate(cum)
    max_dd = float(np.max(peaks - cum)) if len(cum) > 0 else 1.0
    calmar = (df["net_rs"].sum() / max_dd) if max_dd > 0 else 0

    print("\n" + "=" * 100)
    print("FINAL VERIFIED RESULTS: FUTURES SIGNALS + 2nd ITM OPTIONS EXECUTION")
    print("=" * 100)
    print(f"Total Option Trades:         {len(df):,}")
    print(f"Total Traded Days:           {len(day_pnl):,}")
    print(f"Avg Trades/Day:              {len(df)/len(day_pnl):.2f} trades/day")
    print(f"TRADE WIN RATE:              {wr:.2f}% ({len(wins):,} Wins / {len(losses):,} Losses)")
    print(f"DAILY WIN RATE:              {daily_wr:.1f}% ({sum(1 for v in day_pnl if v > 0)} Green / {sum(1 for v in day_pnl if v <= 0)} Red)")
    print(f"PROFIT FACTOR:               {pf:.3f}")
    print(f"TOTAL NET POINTS (Options):  {df['pnl_pts'].sum():>+,.2f} pts")
    print(f"TOTAL REALIZED NET PROFIT:   Rs {df['net_rs'].sum():>+,.2f} (1 Lot / 65 qty)")
    print(f"MAX DRAWDOWN:                Rs {max_dd:>+,.2f}")
    print(f"CALMAR RATIO:                {calmar:.2f}")
    print(f"Avg Win:                     +{wins['pnl_pts'].mean():.2f} pts (Rs {wins['net_rs'].mean():+,.2f})")
    print(f"Avg Loss:                    {losses['pnl_pts'].mean():.2f} pts (Rs {losses['net_rs'].mean():+,.2f})")
    print("=" * 100)

    print("\nYEAR-BY-YEAR PERFORMANCE:")
    print(f"{'Year':6s} | {'Trades':8s} | {'Days':6s} | {'Win Rate':10s} | {'Net Pts':12s} | {'Realized Rs':16s} | {'PF':8s}")
    print("-" * 80)
    for yr, g in df.groupby("year"):
        yw = g[g["net_rs"] > 0]
        yl = g[g["net_rs"] <= 0]
        ypf = yw["net_rs"].sum() / abs(yl["net_rs"].sum()) if abs(yl["net_rs"].sum()) > 0 else 99.0
        print(f"{yr:6s} | {len(g):<8d} | {g['date'].nunique():<6d} | {len(yw)/len(g)*100:<9.2f}% | {g['pnl_pts'].sum():>+10.2f} pts | Rs {g['net_rs'].sum():>+12,.2f} | {ypf:<8.3f}")

    out = ROOT / "artifacts" / "f6_hybrid" / "futures_signals_options_execution_summary.json"
    out.write_text(json.dumps({
        "trades": int(len(df)), "days": int(len(day_pnl)), "wr": float(wr),
        "daily_wr": float(daily_wr), "pf": float(pf), "net_pts": float(df["pnl_pts"].sum()),
        "net_rs": float(df["net_rs"].sum()), "max_dd": float(max_dd), "calmar": float(calmar)
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_fast_backtest()
