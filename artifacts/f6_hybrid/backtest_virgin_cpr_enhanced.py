"""Virgin CPR Enhanced Backtest Engine (GPU Accelerated with 8 Workers).

Compares:
  1. Baseline: Current Daily CPR + Daily VWAP + EMA 200 + Camarilla H3/L3
  2. Enhanced: Current Daily CPR + Active Historical Virgin CPRs (Untouched CPRs) + VWAP + EMA 200 + Camarilla
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
print("VIRGIN CPR ENHANCED BACKTEST ENGINE — FULL 7-YEAR COMPARISON")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | CPU Workers: {WORKERS}")
print("=" * 135)


def parse_futures_file(fpath: Path) -> Optional[Tuple[str, pd.DataFrame]]:
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
        return (date_norm, df) if len(df) >= 50 else None
    except Exception:
        return None


def run_virgin_cpr_study():
    t0 = time.time()
    fut_files = sorted(list(FUT_DIR.glob("**/*.csv")))
    day_dfs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = list(executor.map(parse_futures_file, fut_files))

    for r in results:
        if r is not None:
            day_dfs[r[0]] = r[1]

    sorted_days = sorted(list(day_dfs.keys()))
    print(f"Loaded {len(sorted_days)} Futures Trading Days ({sorted_days[0]} to {sorted_days[-1]}).")

    # Step 1: Pre-calculate CPR for each day and track Virgin (Untouched) CPR status
    daily_cprs = {}
    for i in range(1, len(sorted_days)):
        day = sorted_days[i]
        prev_day = sorted_days[i - 1]
        pdf = day_dfs[prev_day]
        cur_df = day_dfs[day]

        prev_h = float(pdf["high"].max())
        prev_l = float(pdf["low"].min())
        prev_c = float(pdf["close"].iloc[-1])

        pivot = (prev_h + prev_l + prev_c) / 3.0
        bc = (prev_h + prev_l) / 2.0
        tc = (pivot - bc) + pivot
        cpr_top = max(tc, bc)
        cpr_bot = min(tc, bc)

        cur_h = float(cur_df["high"].max())
        cur_l = float(cur_df["low"].min())

        # Check if touched today
        touched = (cur_l <= cpr_top) and (cur_h >= cpr_bot)
        daily_cprs[day] = {
            "pivot": pivot, "tc": cpr_top, "bc": cpr_bot,
            "virgin": not touched, "prev_h": prev_h, "prev_l": prev_l, "prev_c": prev_c
        }

    # Step 2: Run Two Backtests: Baseline vs With Virgin CPR
    for include_virgin in [False, True]:
        label = "WITH VIRGIN CPR" if include_virgin else "BASELINE (CURRENT DAY CPR ONLY)"
        print(f"\nEvaluating: {label}...")

        all_signals = []
        all_3m_candles = []
        active_virgin_cprs = []  # Stores list of active virgin CPRs [(pivot, tc, bc, origin_day)]

        for i in range(1, len(sorted_days)):
            day = sorted_days[i]
            cur_df = day_dfs[day].copy()
            cpr_info = daily_cprs[day]

            # S/R Levels
            pivot = cpr_info["pivot"]
            cpr_top = cpr_info["tc"]
            cpr_bot = cpr_info["bc"]
            prev_h = cpr_info["prev_h"]
            prev_l = cpr_info["prev_l"]
            prev_c = cpr_info["prev_c"]

            cam_range = prev_h - prev_l
            h3 = prev_c + cam_range * (1.1 / 4.0)
            l3 = prev_c - cam_range * (1.1 / 4.0)

            # Build 3m & 15m Candles
            cur_df["bar_3m_idx"] = (cur_df["minute"] - 555) // 3
            agg_3m = cur_df.groupby("bar_3m_idx").agg(
                open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum") if "volume" in cur_df.columns else ("close", "count"),
                minute_start=("minute", "first"),
            ).reset_index()

            cur_df["bar_15m_idx"] = (cur_df["minute"] - 555) // 15
            agg_15m = cur_df.groupby("bar_15m_idx").agg(
                close=("close", "last"), minute_start=("minute", "first")
            ).reset_index()

            prior_3m = []
            prior_15m = []
            for p_d in sorted_days[max(0, i - 4): i]:
                pdf = day_dfs[p_d]
                prior_3m.extend(pdf.groupby((pdf["minute"] - 555) // 3)["close"].last().tolist())
                prior_15m.extend(pdf.groupby((pdf["minute"] - 555) // 15)["close"].last().tolist())

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

            # Incremental ATR5
            tr_list = []
            for k in range(len(agg_3m)):
                h, l = agg_3m.iloc[k]["high"], agg_3m.iloc[k]["low"]
                c_prev = agg_3m.iloc[k - 1]["close"] if k > 0 else agg_3m.iloc[k]["open"]
                tr_list.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
            atr5 = pd.Series(tr_list).rolling(5, min_periods=1).mean().values

            # Active S/R Levels for today
            sr_levels = [
                ("Daily CPR Pivot", pivot, 1),
                ("Daily CPR Top (TC)", cpr_top, 1),
                ("Daily CPR Bottom (BC)", cpr_bot, 1),
                ("Camarilla H3", h3, 2),
                ("Camarilla L3", l3, 2),
                ("Prev Day High", prev_h, 2),
                ("Prev Day Low", prev_l, 2),
            ]

            # Add Virgin CPRs if enabled
            if include_virgin:
                for v_p, v_tc, v_bc, o_day in active_virgin_cprs[-3:]:  # Keep most recent 3 virgin CPRs
                    sr_levels.append((f"Virgin CPR Pivot ({o_day[-5:]})", v_p, 1))
                    sr_levels.append((f"Virgin CPR Top ({o_day[-5:]})", v_tc, 1))
                    sr_levels.append((f"Virgin CPR Bot ({o_day[-5:]})", v_bc, 1))

            day_3m_records = agg_3m.to_dict("records")
            for r in day_3m_records:
                r["date"] = day

            # Two-Bar Scan
            touch_counts = {}
            for b_idx in range(len(agg_3m) - 1):
                bar_1 = agg_3m.iloc[b_idx]
                bar_2 = agg_3m.iloc[b_idx + 1]
                m_start = int(bar_2["minute_start"])

                if not ((555 <= m_start <= 660) or (810 <= m_start <= 900)):
                    continue

                idx_15m = min(int((m_start - 555) // 15), len(ema20_15m) - 1)
                is_15m_bull = bar_2["close"] >= ema20_15m[idx_15m]

                # Dynamic levels at this bar
                cur_sr = list(sr_levels) + [
                    ("Daily VWAP", vwap_3m[b_idx], 1),
                    ("EMA 200", ema200_3m[b_idx], 1),
                    ("EMA 20", ema20_3m[b_idx], 2),
                ]

                cur_atr = max(float(atr5[b_idx]), 8.0)
                sl_dist = max(cur_atr * 0.30, 4.0)
                tp_dist = max(cur_atr * 1.50, 8.0)

                for lvl_name, lvl_px, prio in cur_sr:
                    if touch_counts.get(lvl_name, 0) >= 2:
                        continue

                    if bar_1["low"] <= lvl_px <= bar_1["high"]:
                        if is_15m_bull and (bar_2["high"] > bar_1["high"]):
                            touch_counts[lvl_name] = touch_counts.get(lvl_name, 0) + 1
                            all_signals.append({
                                "date": day, "minute": m_start,
                                "direction": 1, "entry": float(bar_1["high"] + 0.5),
                                "sl_dist": float(sl_dist), "tgt_dist": float(tp_dist),
                                "level": lvl_name, "bar_idx": b_idx + 1,
                                "global_bar_idx": len(all_3m_candles) + b_idx + 1,
                            })
                            break
                        elif (not is_15m_bull) and (bar_2["low"] < bar_1["low"]):
                            touch_counts[lvl_name] = touch_counts.get(lvl_name, 0) + 1
                            all_signals.append({
                                "date": day, "minute": m_start,
                                "direction": -1, "entry": float(bar_1["low"] - 0.5),
                                "sl_dist": float(sl_dist), "tgt_dist": float(tp_dist),
                                "level": lvl_name, "bar_idx": b_idx + 1,
                                "global_bar_idx": len(all_3m_candles) + b_idx + 1,
                            })
                            break

            all_3m_candles.extend(day_3m_records)

            # Update active virgin CPR list for next days
            if include_virgin:
                # 1. Remove any virgin CPRs touched today
                surviving = []
                for v_p, v_tc, v_bc, o_day in active_virgin_cprs:
                    if not ((cur_df["low"].min() <= v_tc) and (cur_df["high"].max() >= v_bc)):
                        surviving.append((v_p, v_tc, v_bc, o_day))
                active_virgin_cprs = surviving

                # 2. Add today's CPR if it remained virgin
                if cpr_info["virgin"]:
                    active_virgin_cprs.append((pivot, cpr_top, cpr_bot, day))

        # GPU Vectorized Trade Simulation
        df_sig = pd.DataFrame(all_signals)
        N_SIG = len(df_sig)
        MAX_FUT = 75

        t_entries = torch.tensor(df_sig["entry"].values, device=device, dtype=torch.float32)
        t_dirs = torch.tensor(df_sig["direction"].values, device=device, dtype=torch.float32)
        t_sl_dists = torch.tensor(df_sig["sl_dist"].values, device=device, dtype=torch.float32)
        t_tgt_dists = torch.tensor(df_sig["tgt_dist"].values, device=device, dtype=torch.float32)

        fut_highs = np.zeros((N_SIG, MAX_FUT), dtype=np.float32)
        fut_lows = np.zeros((N_SIG, MAX_FUT), dtype=np.float32)
        fut_closes = np.zeros((N_SIG, MAX_FUT), dtype=np.float32)
        fut_valid = np.zeros((N_SIG, MAX_FUT), dtype=bool)

        for s_i, (_, s_row) in enumerate(df_sig.iterrows()):
            g_i = int(s_row["global_bar_idx"])
            s_day = str(s_row["date"])
            for step in range(1, MAX_FUT + 1):
                tgt_i = g_i + step
                if tgt_i >= len(all_3m_candles):
                    break
                c_f = all_3m_candles[tgt_i]
                if c_f["date"] != s_day:
                    break
                fut_highs[s_i, step - 1] = c_f["high"]
                fut_lows[s_i, step - 1] = c_f["low"]
                fut_closes[s_i, step - 1] = c_f["close"]
                fut_valid[s_i, step - 1] = True

        t_fut_h = torch.tensor(fut_highs, device=device)
        t_fut_l = torch.tensor(fut_lows, device=device)
        t_fut_c = torch.tensor(fut_closes, device=device)
        t_fut_v = torch.tensor(fut_valid, device=device)

        is_long = (t_dirs == 1).unsqueeze(1)
        entries = t_entries.unsqueeze(1)
        init_sl = torch.where(is_long, entries - t_sl_dists.unsqueeze(1), entries + t_sl_dists.unsqueeze(1))
        init_tp = torch.where(is_long, entries + t_tgt_dists.unsqueeze(1), entries - t_tgt_dists.unsqueeze(1))

        run_peaks_l = torch.cummax(torch.where(t_fut_v, t_fut_h, entries), dim=1).values
        dyn_sl_l = torch.where((run_peaks_l - entries) >= 6.0, torch.maximum(init_sl, run_peaks_l - 2.0), init_sl)

        run_peaks_s = torch.cummin(torch.where(t_fut_v, t_fut_l, entries), dim=1).values
        dyn_sl_s = torch.where((entries - run_peaks_s) >= 6.0, torch.minimum(init_sl, run_peaks_s + 2.0), init_sl)

        dyn_sl = torch.where(is_long, dyn_sl_l, dyn_sl_s)

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

        df_sig["pnl_pts"] = pts_raw.cpu().numpy()
        df_sig["net_rs"] = rs_net.cpu().numpy()

        wins = df_sig[df_sig["net_rs"] > 0]
        losses = df_sig[df_sig["net_rs"] <= 0]
        wr = len(wins) / len(df_sig) * 100
        pf = wins["net_rs"].sum() / abs(losses["net_rs"].sum()) if abs(losses["net_rs"].sum()) > 0 else 99.0

        day_pnls = df_sig.groupby("date")["net_rs"].sum()
        green_days = sum(1 for v in day_pnls if v > 0)
        daily_wr = green_days / len(day_pnls) * 100

        cum_eq = np.cumsum(day_pnls.values)
        peaks = np.maximum.accumulate(cum_eq)
        max_dd = float(np.max(peaks - cum_eq)) if len(cum_eq) > 0 else 1.0
        calmar = (df_sig["net_rs"].sum() / max_dd) if max_dd > 0 else 0

        # Virgin CPR specific trade count
        virgin_trades = df_sig[df_sig["level"].str.contains("Virgin", na=False)]

        print("=" * 100)
        print(f"RESULTS FOR: {label}")
        print("=" * 100)
        print(f"Total Trades:                {len(df_sig):,}")
        if include_virgin:
            v_w = len(virgin_trades[virgin_trades['net_rs'] > 0])
            v_wr = (v_w / len(virgin_trades) * 100) if len(virgin_trades) > 0 else 0
            print(f"  * Virgin CPR Setups:       {len(virgin_trades):,} trades | {v_wr:.2f}% Win Rate | Rs {virgin_trades['net_rs'].sum():>+,.2f}")
        print(f"TRADE WIN RATE:              {wr:.2f}% ({len(wins):,} Wins / {len(losses):,} Losses)")
        print(f"DAILY WIN RATE:              {daily_wr:.1f}% ({green_days} Green / {len(day_pnls)-green_days} Red Days)")
        print(f"PROFIT FACTOR:               {pf:.3f}")
        print(f"TOTAL NET POINTS:            {df_sig['pnl_pts'].sum():>+,.2f} pts")
        print(f"TOTAL REALIZED NET PROFIT:   Rs {df_sig['net_rs'].sum():>+,.2f} (1 Lot / 65 qty)")
        print(f"MAX DRAWDOWN:                Rs {max_dd:>+,.2f}")
        print(f"CALMAR RATIO:                {calmar:.2f}")
        print(f"Avg Win:                     +{wins['pnl_pts'].mean():.2f} pts (Rs {wins['net_rs'].mean():+,.2f})")
        print(f"Avg Loss:                    {losses['pnl_pts'].mean():.2f} pts (Rs {losses['net_rs'].mean():+,.2f})")
        print("=" * 100)


if __name__ == "__main__":
    run_virgin_cpr_study()
