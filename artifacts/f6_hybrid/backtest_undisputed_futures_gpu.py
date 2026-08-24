"""Undisputed Rejection Champion Backtest on Nifty Futures (GPU Accelerated with 8 Workers).

Data Source: C:\\Users\\user\\Desktop\\nifty50 data\\nifty_futures
Strategy:
  - 3-Minute Price Action with Two-Bar Structure Confirmation (Break of Extreme)
  - S/R Anchors: Daily CPR (Pivot, TC, BC), Daily VWAP, 200 EMA, 20 EMA, Camarilla H3/L3, PDH/PDL
  - 15-Minute Trend Gate: 15m Close >= 20 EMA (Long) | 15m Close < 20 EMA (Short)
  - Sessions: 09:15-11:00 (Morning) & 13:30-15:00 (Afternoon)
  - Risk Geometry: Initial SL = 0.30x ATR (min 4.0 pts), TP = 1.50x ATR, Trail @ +6.0 pts (step 2.0 pts)
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
DATA_DIR = Path(r"C:\Users\user\Desktop\nifty50 data\nifty_futures")

LOT_SIZE = 65
FEE_PER_TRADE = 45.0
WORKERS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 135)
print("UNDISPUTED REJECTION CHAMPION — NIFTY FUTURES BACKTEST ENGINE")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | CPU Workers: {WORKERS}")
print(f"Data Source: {DATA_DIR}")
print("=" * 135)


def parse_futures_day_file(fpath: Path) -> Optional[Tuple[str, pd.DataFrame]]:
    """Loads and cleans 1-minute Nifty Futures data for a single day."""
    try:
        df = pd.read_csv(fpath)
        if df.empty or "close" not in df.columns:
            return None
        
        # Standardize date and time
        if "date" in df.columns and "time" in df.columns:
            d_str = str(df["date"].iloc[0])
            # Handle YYYY-MM-DD or DD-MM-YYYY
            if "-" in d_str and len(d_str) == 10:
                parts = d_str.split("-")
                if len(parts[0]) == 4:
                    date_norm = d_str
                else:
                    date_norm = f"{parts[2]}-{parts[1]}-{parts[0]}"
            else:
                date_norm = d_str

            df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        else:
            return None

        df = df.sort_values("datetime").reset_index(drop=True)
        # Filter trading hours 09:15 to 15:30
        df["minute"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
        df = df[(df["minute"] >= 555) & (df["minute"] <= 930)].reset_index(drop=True)
        if len(df) < 50:
            return None

        return date_norm, df
    except Exception:
        return None


def process_futures_day(
    day: str,
    df_day: pd.DataFrame,
    prev_h: float,
    prev_l: float,
    prev_c: float,
    prior_3m_closes: List[float],
    prior_15m_closes: List[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extracts 3m bars, indicators, S/R anchors, and causal Two-Bar Rejection signals."""
    # 1. S/R Levels from Previous Day
    pivot = (prev_h + prev_l + prev_c) / 3.0
    bc = (prev_h + prev_l) / 2.0
    tc = (pivot - bc) + pivot
    cpr_top = max(tc, bc)
    cpr_bot = min(tc, bc)

    cam_range = prev_h - prev_l
    h3 = prev_c + cam_range * (1.1 / 4.0)
    l3 = prev_c - cam_range * (1.1 / 4.0)

    # 2. Build 3m Aggregate Candles
    df_day["bar_3m_idx"] = (df_day["minute"] - 555) // 3
    agg_3m = df_day.groupby("bar_3m_idx").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum") if "volume" in df_day.columns else ("close", "count"),
        minute_start=("minute", "first"),
    ).reset_index()

    # 3. Build 15m Aggregate Candles for Trend Gate
    df_day["bar_15m_idx"] = (df_day["minute"] - 555) // 15
    agg_15m = df_day.groupby("bar_15m_idx").agg(
        close=("close", "last"), minute_start=("minute", "first")
    ).reset_index()

    full_15m = pd.Series(prior_15m_closes + agg_15m["close"].tolist())
    ema20_15m = full_15m.ewm(span=20, adjust=False).mean().iloc[-len(agg_15m):].values

    # 4. 3m VWAP, EMA20, EMA200
    cum_vol = 0.0
    cum_pv = 0.0
    vwap_3m = []
    for _, r in agg_3m.iterrows():
        tp = (r["high"] + r["low"] + r["close"]) / 3.0
        v = max(float(r["volume"]), 1.0)
        cum_vol += v
        cum_pv += tp * v
        vwap_3m.append(cum_pv / cum_vol)

    full_3m = pd.Series(prior_3m_closes + agg_3m["close"].tolist())
    ema20_3m = full_3m.ewm(span=20, adjust=False).mean().iloc[-len(agg_3m):].values
    ema200_3m = full_3m.ewm(span=200, adjust=False).mean().iloc[-len(agg_3m):].values

    # 5. Incremental ATR5
    tr_list = []
    for i in range(len(agg_3m)):
        h, l = agg_3m.iloc[i]["high"], agg_3m.iloc[i]["low"]
        c_prev = agg_3m.iloc[i - 1]["close"] if i > 0 else agg_3m.iloc[i]["open"]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        tr_list.append(tr)
    atr5 = pd.Series(tr_list).rolling(5, min_periods=1).mean().values

    # 6. Extract Two-Bar Confirmation Signals
    signals = []
    day_3m_records = agg_3m.to_dict("records")
    for r in day_3m_records:
        r["date"] = day

    for b_idx in range(len(agg_3m) - 1):
        bar_1 = agg_3m.iloc[b_idx]
        bar_2 = agg_3m.iloc[b_idx + 1]
        m_start = int(bar_2["minute_start"])

        # Check Active Sessions: 09:15-11:00 (555-660) or 13:30-15:00 (810-900)
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

        cur_atr = max(float(atr5[b_idx]), 8.0)
        sl_dist = max(cur_atr * 0.30, 4.0)
        tp_dist = max(cur_atr * 1.50, 8.0)

        for lvl_name, lvl_px, prio in sr_levels:
            if bar_1["low"] <= lvl_px <= bar_1["high"]:
                # --- SUPPORT BOUNCE (LONG) ---
                if is_15m_bull and (bar_2["high"] > bar_1["high"]):
                    entry_px = bar_1["high"] + 0.5
                    signals.append({
                        "date": day,
                        "minute": m_start,
                        "time": f"{day} {m_start // 60:02d}:{m_start % 60:02d}:00",
                        "direction": 1,
                        "entry": float(entry_px),
                        "sl_dist": float(sl_dist),
                        "tgt_dist": float(tp_dist),
                        "level": lvl_name,
                        "bar_idx": b_idx + 1,
                    })
                    break

                # --- RESISTANCE REJECTION (SHORT) ---
                elif (not is_15m_bull) and (bar_2["low"] < bar_1["low"]):
                    entry_px = bar_1["low"] - 0.5
                    signals.append({
                        "date": day,
                        "minute": m_start,
                        "time": f"{day} {m_start // 60:02d}:{m_start % 60:02d}:00",
                        "direction": -1,
                        "entry": float(entry_px),
                        "sl_dist": float(sl_dist),
                        "tgt_dist": float(tp_dist),
                        "level": lvl_name,
                        "bar_idx": b_idx + 1,
                    })
                    break

    return signals, day_3m_records


def run_futures_backtest(is_smoke_test: bool = False):
    t0 = time.time()
    all_files = sorted(list(DATA_DIR.glob("**/*.csv")))
    print(f"Found {len(all_files)} Futures CSV files. Loading with {WORKERS} workers...")

    # Load all daily files in parallel
    day_dfs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = list(executor.map(parse_futures_day_file, all_files))

    for r in results:
        if r is not None:
            day_str, df_d = r
            day_dfs[day_str] = df_d

    sorted_days = sorted(list(day_dfs.keys()))
    print(f"Loaded {len(sorted_days)} valid trading days ({sorted_days[0]} to {sorted_days[-1]}).")

    if is_smoke_test:
        sorted_days = sorted_days[:5]
        print(f"\n=== SMOKE TEST: RUNNING ON FIRST {len(sorted_days)} DAYS ONLY ===")

    all_signals = []
    all_3m_candles = []
    
    # Process sequentially for causal warmups
    for i in range(1, len(sorted_days)):
        day = sorted_days[i]
        prev_day = sorted_days[i - 1]
        prev_df = day_dfs[prev_day]

        prev_h = float(prev_df["high"].max())
        prev_l = float(prev_df["low"].min())
        prev_c = float(prev_df["close"].iloc[-1])

        # Warmup slices
        prior_3m = []
        prior_15m = []
        for p_d in sorted_days[max(0, i - 4): i]:
            pdf = day_dfs[p_d]
            prior_3m.extend(pdf.groupby((pdf["minute"] - 555) // 3)["close"].last().tolist())
            prior_15m.extend(pdf.groupby((pdf["minute"] - 555) // 15)["close"].last().tolist())

        sigs, candles_3m = process_futures_day(
            day, day_dfs[day], prev_h, prev_l, prev_c, prior_3m, prior_15m
        )
        for s in sigs:
            s["global_bar_idx"] = len(all_3m_candles) + s["bar_idx"]
        all_signals.extend(sigs)
        all_3m_candles.extend(candles_3m)

    df_signals = pd.DataFrame(all_signals)
    print(f"Extracted {len(df_signals)} Causal Two-Bar Rejection Signals across {len(all_3m_candles):,} 3m bars.")

    # 7. GPU Vectorized Parameter Execution
    N_SIG = len(df_signals)
    MAX_FUT = 75

    t_entries = torch.tensor(df_signals["entry"].values, device=device, dtype=torch.float32)
    t_dirs = torch.tensor(df_signals["direction"].values, device=device, dtype=torch.float32)
    t_sl_dists = torch.tensor(df_signals["sl_dist"].values, device=device, dtype=torch.float32)
    t_tgt_dists = torch.tensor(df_signals["tgt_dist"].values, device=device, dtype=torch.float32)

    # Build future price tensors
    fut_highs = np.zeros((N_SIG, MAX_FUT), dtype=np.float32)
    fut_lows = np.zeros((N_SIG, MAX_FUT), dtype=np.float32)
    fut_closes = np.zeros((N_SIG, MAX_FUT), dtype=np.float32)
    fut_valid = np.zeros((N_SIG, MAX_FUT), dtype=bool)

    for i, (_, row) in enumerate(df_signals.iterrows()):
        g_idx = int(row["global_bar_idx"])
        sig_day = str(row["date"])
        for step in range(1, MAX_FUT + 1):
            target_idx = g_idx + step
            if target_idx >= len(all_3m_candles):
                break
            c_fut = all_3m_candles[target_idx]
            if c_fut["date"] != sig_day:
                break
            fut_highs[i, step - 1] = c_fut["high"]
            fut_lows[i, step - 1] = c_fut["low"]
            fut_closes[i, step - 1] = c_fut["close"]
            fut_valid[i, step - 1] = True

    t_fut_h = torch.tensor(fut_highs, device=device)
    t_fut_l = torch.tensor(fut_lows, device=device)
    t_fut_c = torch.tensor(fut_closes, device=device)
    t_fut_v = torch.tensor(fut_valid, device=device)

    # Simulate SL / TP / Trailing Stop in GPU
    is_long = (t_dirs == 1).unsqueeze(1)
    entries = t_entries.unsqueeze(1)
    init_sl = torch.where(is_long, entries - t_sl_dists.unsqueeze(1), entries + t_sl_dists.unsqueeze(1))
    init_tp = torch.where(is_long, entries + t_tgt_dists.unsqueeze(1), entries - t_tgt_dists.unsqueeze(1))

    # Long Trailing SL
    run_peaks_long = torch.cummax(torch.where(t_fut_v, t_fut_h, entries), dim=1).values
    dyn_sl_long = torch.where((run_peaks_long - entries) >= 6.0, torch.maximum(init_sl, run_peaks_long - 2.0), init_sl)

    # Short Trailing SL
    run_peaks_short = torch.cummin(torch.where(t_fut_v, t_fut_l, entries), dim=1).values
    dyn_sl_short = torch.where((entries - run_peaks_short) >= 6.0, torch.minimum(init_sl, run_peaks_short + 2.0), init_sl)

    dyn_sl = torch.where(is_long, dyn_sl_long, dyn_sl_short)

    hit_sl = torch.where(is_long, t_fut_l <= dyn_sl, t_fut_h >= dyn_sl)
    hit_tp = torch.where(is_long, t_fut_h >= init_tp, t_fut_l <= init_tp)

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    sl_idx_clamp = sl_first.clamp(max=MAX_FUT - 1).unsqueeze(1)
    exit_sl_px = dyn_sl.gather(1, sl_idx_clamp).squeeze(1)
    exit_tp_px = init_tp.squeeze(1)

    last_valid_idx = (t_fut_v.sum(dim=1) - 1).clamp(min=0).unsqueeze(1)
    exit_eod_px = t_fut_c.gather(1, last_valid_idx).squeeze(1)

    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, exit_eod_px))
    pts_raw = torch.where(is_long.squeeze(1), exit_px - entries.squeeze(1), entries.squeeze(1) - exit_px)
    rs_net = pts_raw * LOT_SIZE - FEE_PER_TRADE

    df_signals["pnl_pts"] = pts_raw.cpu().numpy()
    df_signals["net_rs"] = rs_net.cpu().numpy()
    df_signals["year"] = df_signals["date"].str[:4]

    # Analytics
    wins = df_signals[df_signals["net_rs"] > 0]
    losses = df_signals[df_signals["net_rs"] <= 0]
    wr = len(wins) / len(df_signals) * 100 if len(df_signals) > 0 else 0
    gross_w = wins["net_rs"].sum()
    gross_l = abs(losses["net_rs"].sum())
    pf = gross_w / gross_l if gross_l > 0 else 99.0

    day_pnls = df_signals.groupby("date")["net_rs"].sum()
    green_days = sum(1 for v in day_pnls if v > 0)
    red_days = sum(1 for v in day_pnls if v <= 0)
    daily_wr = (green_days / len(day_pnls) * 100) if len(day_pnls) > 0 else 0

    cum_eq = np.cumsum(day_pnls.values)
    peaks = np.maximum.accumulate(cum_eq)
    max_dd = float(np.max(peaks - cum_eq)) if len(cum_eq) > 0 else 1.0
    calmar = (df_signals["net_rs"].sum() / max_dd) if max_dd > 0 else 0

    print("\n" + "=" * 135)
    print(f"NIFTY FUTURES BACKTEST RESULTS ({'SMOKE TEST' if is_smoke_test else 'FULL 7-YEAR RUN'})")
    print("=" * 135)
    print(f"Total Futures Trades:        {len(df_signals):,}")
    print(f"Total Traded Days:           {len(day_pnls):,}")
    print(f"Avg Trades per Day:          {len(df_signals)/len(day_pnls):.2f} trades/day")
    print(f"TRADE WIN RATE:              {wr:.2f}% ({len(wins):,} Wins / {len(losses):,} Losses)")
    print(f"DAILY WIN RATE:              {daily_wr:.1f}% ({green_days} Green / {red_days} Red Days)")
    print(f"PROFIT FACTOR:               {pf:.3f}")
    print(f"TOTAL NET POINTS:            {df_signals['pnl_pts'].sum():>+,.2f} pts")
    print(f"TOTAL REALIZED NET PROFIT:   Rs {df_signals['net_rs'].sum():>+,.2f} (1 Lot / 65 qty)")
    print(f"MAX DRAWDOWN:                Rs {max_dd:>+,.2f}")
    print(f"CALMAR RATIO:                {calmar:.2f}")
    print(f"Avg Stop Loss:               {df_signals['sl_dist'].mean():.2f} pts")
    print(f"Avg Target:                  {df_signals['tgt_dist'].mean():.2f} pts")
    print(f"Avg Win:                     +{wins['pnl_pts'].mean():.2f} pts (Rs {wins['net_rs'].mean():+,.2f})")
    print(f"Avg Loss:                    {losses['pnl_pts'].mean():.2f} pts (Rs {losses['net_rs'].mean():+,.2f})")
    print("=" * 135)

    # Yearly Breakdown
    print("\nYEAR-BY-YEAR BREAKDOWN ON NIFTY FUTURES:")
    print(f"{'Year':6s} | {'Trades':8s} | {'Days':6s} | {'Tr/Day':8s} | {'Win Rate':10s} | {'Net Points':14s} | {'Realized Rs':18s} | {'PF':8s}")
    print("-" * 105)
    for yr, grp in df_signals.groupby("year"):
        y_w = grp[grp["net_rs"] > 0]
        y_l = grp[grp["net_rs"] <= 0]
        y_pf = y_w["net_rs"].sum() / abs(y_l["net_rs"].sum()) if abs(y_l["net_rs"].sum()) > 0 else 99.0
        y_days = grp["date"].nunique()
        print(f"{yr:6s} | {len(grp):<8d} | {y_days:<6d} | {len(grp)/y_days:<8.2f} | {len(y_w)/len(grp)*100:<9.2f}% | {grp['pnl_pts'].sum():>+12.2f} pts | Rs {grp['net_rs'].sum():>+14,.2f} | {y_pf:<8.3f}")

    out_file = ROOT / "artifacts" / "f6_hybrid" / "futures_undisputed_results.json"
    out_file.write_text(json.dumps({
        "total_trades": int(len(df_signals)),
        "total_days": int(len(day_pnls)),
        "win_rate": float(wr),
        "daily_win_rate": float(daily_wr),
        "pf": float(pf),
        "net_points": float(df_signals["pnl_pts"].sum()),
        "net_rs": float(df_signals["net_rs"].sum()),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
    }, indent=2), encoding="utf-8")
    print(f"\n[Saved Futures Results JSON]: {out_file}")


if __name__ == "__main__":
    is_smoke = "--smoke" in sys.argv
    run_futures_backtest(is_smoke_test=is_smoke)
