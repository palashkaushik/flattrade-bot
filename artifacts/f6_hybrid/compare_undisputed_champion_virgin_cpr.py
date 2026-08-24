"""Impact of Virgin CPR on the Undisputed Rejection Champion Strategy (794+ Calmar)."""

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


def load_data():
    df_spot = pd.read_csv(IDX_FILE)
    df_spot["dt"] = pd.to_datetime(df_spot["date"])
    df_spot["day"] = df_spot["dt"].dt.strftime("%Y-%m-%d")
    df_spot["minute"] = df_spot["dt"].dt.hour * 60 + df_spot["dt"].dt.minute
    spot_map = {}
    for d, g in df_spot.groupby("day"):
        spot_map[d] = dict(zip(g["minute"], g["close"]))

    df_days = {}
    for d, g in df_spot.groupby("day"):
        df_days[d] = g.sort_values("minute").reset_index(drop=True)

    sorted_days = sorted(list(df_days.keys()))
    return df_days, sorted_days, spot_map


def run_comparison():
    df_days, sorted_days, spot_map = load_data()
    print("=" * 110)
    print("EXACT IMPACT OF VIRGIN CPR ON UNDISPUTED REJECTION CHAMPION (794+ CALMAR ENGINE)")
    print("=" * 110)

    # Pre-calculate Virgin CPR for each day
    daily_cprs = {}
    for i in range(1, len(sorted_days)):
        day = sorted_days[i]
        prev_df = df_days[sorted_days[i - 1]]
        cur_df = df_days[day]

        p_h, p_l, p_c = float(prev_df["high"].max()), float(prev_df["low"].min()), float(prev_df["close"].iloc[-1])
        pivot = (p_h + p_l + p_c) / 3.0
        bc = (p_h + p_l) / 2.0
        tc = (pivot - bc) + pivot
        c_top, c_bot = max(tc, bc), min(tc, bc)

        c_h, c_l = float(cur_df["high"].max()), float(cur_df["low"].min())
        touched = (c_l <= c_top) and (c_h >= c_bot)
        daily_cprs[day] = {
            "pivot": pivot, "tc": c_top, "bc": c_bot,
            "virgin": not touched, "p_h": p_h, "p_l": p_l, "p_c": p_c
        }

    results = {}

    for include_virgin in [False, True]:
        label = "UNDISPUTED CHAMPION + VIRGIN CPR" if include_virgin else "UNDISPUTED CHAMPION BASELINE"
        all_signals = []
        all_3m = []
        active_virgin = []

        for i in range(1, len(sorted_days)):
            day = sorted_days[i]
            cur_df = df_days[day].copy()
            cpr = daily_cprs[day]

            # 3m aggregation
            cur_df["bar_3m_idx"] = (cur_df["minute"] - 555) // 3
            agg_3m = cur_df.groupby("bar_3m_idx").agg(
                open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"),
                minute_start=("minute", "first")
            ).reset_index()

            # 15m aggregation
            cur_df["bar_15m_idx"] = (cur_df["minute"] - 555) // 15
            agg_15m = cur_df.groupby("bar_15m_idx").agg(
                close=("close", "last")
            ).reset_index()

            prior_15m = []
            prior_3m = []
            for p_d in sorted_days[max(0, i - 4): i]:
                pdf = df_days[p_d]
                prior_15m.extend(pdf.groupby((pdf["minute"] - 555) // 15)["close"].last().tolist())
                prior_3m.extend(pdf.groupby((pdf["minute"] - 555) // 3)["close"].last().tolist())

            full_15m = pd.Series(prior_15m + agg_15m["close"].tolist())
            ema20_15m = full_15m.ewm(span=20, adjust=False).mean().iloc[-len(agg_15m):].values

            # 3m VWAP, EMA20, EMA200
            cum_vol = np.arange(1, len(agg_3m) + 1)
            tp = (agg_3m["high"] + agg_3m["low"] + agg_3m["close"]) / 3.0
            vwap_3m = (np.cumsum(tp) / cum_vol).values

            full_3m = pd.Series(prior_3m + agg_3m["close"].tolist())
            ema20_3m = full_3m.ewm(span=20, adjust=False).mean().iloc[-len(agg_3m):].values
            ema200_3m = full_3m.ewm(span=200, adjust=False).mean().iloc[-len(agg_3m):].values

            # ATR5
            tr_l = []
            for k in range(len(agg_3m)):
                h, l = agg_3m.iloc[k]["high"], agg_3m.iloc[k]["low"]
                cp = agg_3m.iloc[k - 1]["close"] if k > 0 else agg_3m.iloc[k]["open"]
                tr_l.append(max(h - l, abs(h - cp), abs(l - cp)))
            atr5 = pd.Series(tr_l).rolling(5, min_periods=1).mean().values

            cam_rng = cpr["p_h"] - cpr["p_l"]
            h3 = cpr["p_c"] + cam_rng * (1.1 / 4.0)
            l3 = cpr["p_c"] - cam_rng * (1.1 / 4.0)

            base_sr = [
                ("CPR Pivot", cpr["pivot"], 1),
                ("CPR Top", cpr["tc"], 1),
                ("CPR Bot", cpr["bc"], 1),
                ("Cam H3", h3, 2),
                ("Cam L3", l3, 2),
                ("PDH", cpr["p_h"], 2),
                ("PDL", cpr["p_l"], 2),
            ]

            if include_virgin:
                for v_p, v_tc, v_bc, o_d in active_virgin[-3:]:
                    base_sr.append((f"Virgin CPR Pivot", v_p, 1))
                    base_sr.append((f"Virgin CPR Top", v_tc, 1))
                    base_sr.append((f"Virgin CPR Bot", v_bc, 1))

            day_records = agg_3m.to_dict("records")
            for r in day_records:
                r["date"] = day

            touch_counts = {}
            for b_idx in range(len(agg_3m) - 1):
                b1 = agg_3m.iloc[b_idx]
                b2 = agg_3m.iloc[b_idx + 1]
                m_start = int(b2["minute_start"])

                if not ((555 <= m_start <= 660) or (810 <= m_start <= 900)):
                    continue

                idx_15m = min(int((m_start - 555) // 15), len(ema20_15m) - 1)
                is_bull = b2["close"] >= ema20_15m[idx_15m]

                cur_sr = list(base_sr) + [
                    ("Daily VWAP", vwap_3m[b_idx], 1),
                    ("EMA 200", ema200_3m[b_idx], 1),
                    ("EMA 20", ema20_3m[b_idx], 2),
                ]

                cur_atr = max(float(atr5[b_idx]), 8.0)
                sl_pts = max(cur_atr * 0.30, 4.0)
                tp_pts = max(cur_atr * 1.50, 8.0)

                for lvl_name, lvl_px, prio in cur_sr:
                    if touch_counts.get(lvl_name, 0) >= 2:
                        continue
                    if b1["low"] <= lvl_px <= b1["high"]:
                        if is_bull and (b2["high"] > b1["high"]):
                            touch_counts[lvl_name] = touch_counts.get(lvl_name, 0) + 1
                            all_signals.append({
                                "date": day, "minute": m_start,
                                "direction": 1, "entry": float(b1["high"] + 0.5),
                                "sl_dist": float(sl_pts), "tgt_dist": float(tp_pts),
                                "level": lvl_name, "global_bar_idx": len(all_3m) + b_idx + 1,
                            })
                            break
                        elif (not is_bull) and (b2["low"] < b1["low"]):
                            touch_counts[lvl_name] = touch_counts.get(lvl_name, 0) + 1
                            all_signals.append({
                                "date": day, "minute": m_start,
                                "direction": -1, "entry": float(b1["low"] - 0.5),
                                "sl_dist": float(sl_pts), "tgt_dist": float(tp_pts),
                                "level": lvl_name, "global_bar_idx": len(all_3m) + b_idx + 1,
                            })
                            break

            all_3m.extend(day_records)

            if include_virgin:
                surviving = []
                for v_p, v_tc, v_bc, o_d in active_virgin:
                    if not ((cur_df["low"].min() <= v_tc) and (cur_df["high"].max() >= v_bc)):
                        surviving.append((v_p, v_tc, v_bc, o_d))
                active_virgin = surviving
                if cpr["virgin"]:
                    active_virgin.append((cpr["pivot"], cpr["tc"], cpr["bc"], day))

        # GPU Vectorized trade simulation
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
                if tgt_i >= len(all_3m):
                    break
                c_f = all_3m[tgt_i]
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

        # Dynamic Options Execution Model with Delta ~0.60
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
        
        # 2nd ITM Options Net Realization: delta 0.60 on points
        opt_pts = pts_raw * 0.60
        rs_net = opt_pts * LOT_SIZE - FEE_PER_TRADE

        df_sig["pnl_pts"] = opt_pts.cpu().numpy()
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

        virgin_trades = df_sig[df_sig["level"].str.contains("Virgin", na=False)]

        results[label] = {
            "total_trades": len(df_sig),
            "virgin_trades": len(virgin_trades),
            "win_rate": wr,
            "daily_win_rate": daily_wr,
            "green_days": green_days,
            "red_days": len(day_pnls) - green_days,
            "pf": pf,
            "net_points": float(df_sig["pnl_pts"].sum()),
            "net_rs": float(df_sig["net_rs"].sum()),
            "max_dd": max_dd,
            "calmar": calmar,
            "avg_win": float(wins["pnl_pts"].mean()),
            "avg_loss": float(losses["pnl_pts"].mean()),
        }

    print("\n" + "=" * 110)
    print(f"{'METRIC':35s} | {'UNDISPUTED BASELINE':25s} | {'WITH VIRGIN CPR':25s} | {'DELTA / CHANGE':18s}")
    print("-" * 110)
    b = results["UNDISPUTED CHAMPION BASELINE"]
    v = results["UNDISPUTED CHAMPION + VIRGIN CPR"]

    print(f"{'Total Trades (7 Years)':35s} | {b['total_trades']:<25,d} | {v['total_trades']:<25,d} | {v['total_trades']-b['total_trades']:>+8d} trades")
    print(f"{'Virgin CPR Specific Trades':35s} | {'0':<25s} | {v['virgin_trades']:<25d} | {v['virgin_trades']:>+8d} trades")
    print(f"{'Trade Win Rate (%)':35s} | {b['win_rate']:<24.2f}% | {v['win_rate']:<24.2f}% | {v['win_rate']-b['win_rate']:>+7.2f}%")
    print(f"{'Daily Win Rate (% Green Days)':35s} | {b['daily_win_rate']:<24.1f}% | {v['daily_win_rate']:<24.1f}% | {v['daily_win_rate']-b['daily_win_rate']:>+7.1f}%")
    print(f"{'Profit Factor (PF)':35s} | {b['pf']:<25.3f} | {v['pf']:<25.3f} | {v['pf']-b['pf']:>+8.3f}")
    print(f"{'Total Net Points (Options)':35s} | {b['net_points']:>+21,.2f} pts | {v['net_points']:>+21,.2f} pts | {v['net_points']-b['net_points']:>+8.2f} pts")
    print(f"{'Total Realized Net Profit (1 Lot)':35s} | Rs {b['net_rs']:>+18,.2f} | Rs {v['net_rs']:>+18,.2f} | Rs {v['net_rs']-b['net_rs']:>+11,.2f}")
    print(f"{'Max Drawdown':35s} | Rs {b['max_dd']:>+18,.2f} | Rs {v['max_dd']:>+18,.2f} | Rs {v['max_dd']-b['max_dd']:>+11,.2f}")
    print(f"{'Calmar Ratio':35s} | {b['calmar']:<25.2f} | {v['calmar']:<25.2f} | {v['calmar']-b['calmar']:>+8.2f}")
    print("=" * 110)


if __name__ == "__main__":
    run_comparison()
