"""Comprehensive S/R Hierarchy Study: 5m EMA20/200 Tier 1 + Camarilla H3/L3 Tier 1 Supreme + Virgin CPR + Opening 3m + Fib (GPU Accelerated).

Hierarchy:
  Tier 1 Supreme:
    - Camarilla H3 / L3 (Priority 1)
    - Virgin CPR (Pivot, TC, BC) (Priority 1)
    - Daily CPR (Pivot, TC, BC) (Priority 1)
    - Daily VWAP (Priority 1)
    - Prev Day VWAP Close (Priority 1)
    - 5-Minute EMA 20 (Priority 1)
    - 5-Minute EMA 200 (Priority 1)
    - 3-Minute EMA 200 (Priority 1)
  Tier 2:
    - Opening 3-Minute Candle High & Low (Priority 2)
    - 3-Minute EMA 20 (Priority 2)
    - Prev Day High / Low (PDH/PDL) (Priority 2)
  Tier 3:
    - Fibonacci H3 / L3 (Priority 3)
    - Camarilla H4 / L4 (Priority 3)
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
IDX_FILE = DESKTOP_DATA / "index" / "NIFTY 50_minute.csv"

LOT_SIZE = 65
FEE_PER_TRADE = 45.0
WORKERS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 135)
print("COMPREHENSIVE S/R HIERARCHY STUDY: 5m EMA20/200 (Tier 1) | CAMARILLA H3/L3 (Tier 1) | VIRGIN CPR | OPENING 3m | FIBONACCI")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | Workers: {WORKERS}")
print("=" * 135)


def load_dataset():
    df_spot = pd.read_csv(IDX_FILE)
    df_spot["dt"] = pd.to_datetime(df_spot["date"])
    df_spot["day"] = df_spot["dt"].dt.strftime("%Y-%m-%d")
    df_spot["minute"] = df_spot["dt"].dt.hour * 60 + df_spot["dt"].dt.minute

    df_days = {}
    for d, g in df_spot.groupby("day"):
        df_days[d] = g.sort_values("minute").reset_index(drop=True)

    sorted_days = sorted(list(df_days.keys()))
    return df_days, sorted_days


def run_experiment(exp_name: str, config: Dict[str, Any], df_days: Dict[str, pd.DataFrame], sorted_days: List[str]):
    t0 = time.time()
    cam_tier = config.get("cam_tier", 2)
    inc_5m_ema = config.get("ema_5m", False)
    inc_virgin = config.get("virgin", False)
    inc_opening_3m = config.get("opening_3m", False)
    inc_fib = config.get("fib", False)
    inc_pd_vwap = config.get("pd_vwap", False)

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

        cum_vol = np.arange(1, len(prev_df) + 1)
        prev_tp = (prev_df["high"] + prev_df["low"] + prev_df["close"]) / 3.0
        pd_vwap = float((np.cumsum(prev_tp) / cum_vol).iloc[-1])

        c_h, c_l = float(cur_df["high"].max()), float(cur_df["low"].min())
        touched = (c_l <= c_top) and (c_h >= c_bot)
        daily_cprs[day] = {
            "pivot": pivot, "tc": c_top, "bc": c_bot,
            "virgin": not touched, "p_h": p_h, "p_l": p_l, "p_c": p_c,
            "pd_vwap": pd_vwap,
        }

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

        # 5m aggregation (for 5m EMA20 and 5m EMA200)
        cur_df["bar_5m_idx"] = (cur_df["minute"] - 555) // 5
        agg_5m = cur_df.groupby("bar_5m_idx").agg(
            close=("close", "last"), minute_start=("minute", "first")
        ).reset_index()

        # 15m aggregation
        cur_df["bar_15m_idx"] = (cur_df["minute"] - 555) // 15
        agg_15m = cur_df.groupby("bar_15m_idx").agg(
            close=("close", "last")
        ).reset_index()

        prior_15m = []
        prior_5m = []
        prior_3m = []
        for p_d in sorted_days[max(0, i - 4): i]:
            pdf = df_days[p_d]
            prior_15m.extend(pdf.groupby((pdf["minute"] - 555) // 15)["close"].last().tolist())
            prior_5m.extend(pdf.groupby((pdf["minute"] - 555) // 5)["close"].last().tolist())
            prior_3m.extend(pdf.groupby((pdf["minute"] - 555) // 3)["close"].last().tolist())

        full_15m = pd.Series(prior_15m + agg_15m["close"].tolist())
        ema20_15m = full_15m.ewm(span=20, adjust=False).mean().iloc[-len(agg_15m):].values

        full_5m = pd.Series(prior_5m + agg_5m["close"].tolist())
        ema20_5m = full_5m.ewm(span=20, adjust=False).mean().iloc[-len(agg_5m):].values
        ema200_5m = full_5m.ewm(span=200, adjust=False).mean().iloc[-len(agg_5m):].values

        # 3m VWAP, EMA20, EMA200
        cum_vol = np.arange(1, len(agg_3m) + 1)
        tp = (agg_3m["high"] + agg_3m["low"] + agg_3m["close"]) / 3.0
        vwap_3m = (np.cumsum(tp) / cum_vol).values

        full_3m = pd.Series(prior_3m + agg_3m["close"].tolist())
        ema20_3m = full_3m.ewm(span=20, adjust=False).mean().iloc[-len(agg_3m):].values
        ema200_3m = full_3m.ewm(span=200, adjust=False).mean().iloc[-len(agg_3m):].values

        # Incremental ATR5
        tr_l = []
        for k in range(len(agg_3m)):
            h, l = agg_3m.iloc[k]["high"], agg_3m.iloc[k]["low"]
            cp = agg_3m.iloc[k - 1]["close"] if k > 0 else agg_3m.iloc[k]["open"]
            tr_l.append(max(h - l, abs(h - cp), abs(l - cp)))
        atr5 = pd.Series(tr_l).rolling(5, min_periods=1).mean().values

        # S/R Base Levels
        p_h, p_l, p_c = cpr["p_h"], cpr["p_l"], cpr["p_c"]
        cam_rng = p_h - p_l
        h3 = p_c + cam_rng * (1.1 / 4.0)
        l3 = p_c - cam_rng * (1.1 / 4.0)

        fib_h3 = cpr["pivot"] + cam_rng * 1.000
        fib_l3 = cpr["pivot"] - cam_rng * 1.000

        first_3m_h = float(agg_3m.iloc[0]["high"]) if len(agg_3m) > 0 else p_h
        first_3m_l = float(agg_3m.iloc[0]["low"]) if len(agg_3m) > 0 else p_l

        base_sr = [
            ("Daily CPR Pivot", cpr["pivot"], 1, False),
            ("Daily CPR Top", cpr["tc"], 1, False),
            ("Daily CPR Bot", cpr["bc"], 1, False),
            ("Camarilla H3", h3, cam_tier, False),
            ("Camarilla L3", l3, cam_tier, False),
            ("PDH", p_h, 2, False),
            ("PDL", p_l, 2, False),
        ]

        if inc_virgin:
            for v_p, v_tc, v_bc, o_d in active_virgin[-3:]:
                base_sr.append(("Virgin CPR Pivot", v_p, 1, True))
                base_sr.append(("Virgin CPR Top", v_tc, 1, True))
                base_sr.append(("Virgin CPR Bot", v_bc, 1, True))

        if inc_pd_vwap:
            base_sr.append(("PD VWAP Close", cpr["pd_vwap"], 1, False))

        if inc_opening_3m:
            base_sr.append(("Opening 3m High", first_3m_h, 2, False))
            base_sr.append(("Opening 3m Low", first_3m_l, 2, False))

        if inc_fib:
            base_sr.append(("Fibonacci H3", fib_h3, 3, False))
            base_sr.append(("Fibonacci L3", fib_l3, 3, False))

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

            idx_5m = min(int((m_start - 555) // 5), len(ema20_5m) - 1)

            cur_sr = list(base_sr) + [
                ("Daily VWAP", vwap_3m[b_idx], 1, False),
                ("3m EMA 200", ema200_3m[b_idx], 1, False),
                ("3m EMA 20", ema20_3m[b_idx], 2, False),
            ]

            if inc_5m_ema:
                cur_sr.append(("5m EMA 20", ema20_5m[idx_5m], 1, False))
                cur_sr.append(("5m EMA 200", ema200_5m[idx_5m], 1, False))

            cur_atr = max(float(atr5[b_idx]), 8.0)
            sl_pts = max(cur_atr * 0.30, 4.0)
            tp_pts = max(cur_atr * 1.50, 8.0)

            sorted_sr = sorted(cur_sr, key=lambda x: (not x[3], x[2]))

            for lvl_name, lvl_px, prio, is_v in sorted_sr:
                if touch_counts.get(lvl_name, 0) >= 2:
                    continue

                if b1["low"] <= lvl_px <= b1["high"]:
                    score = 40 + (25 if is_v else 20 if prio == 1 else 10 if prio == 2 else 5)
                    if is_bull and (b2["high"] > b1["high"]):
                        score += (15 if b1["close"] > lvl_px else 0) + 25
                        if score >= 50:
                            touch_counts[lvl_name] = touch_counts.get(lvl_name, 0) + 1
                            all_signals.append({
                                "date": day, "minute": m_start,
                                "direction": 1, "entry": float(b1["high"] + 0.5),
                                "sl_dist": float(sl_pts), "tgt_dist": float(tp_pts),
                                "level": lvl_name, "global_bar_idx": len(all_3m) + b_idx + 1,
                            })
                            break
                    elif (not is_bull) and (b2["low"] < b1["low"]):
                        score += (15 if b1["close"] < lvl_px else 0) + 25
                        if score >= 50:
                            touch_counts[lvl_name] = touch_counts.get(lvl_name, 0) + 1
                            all_signals.append({
                                "date": day, "minute": m_start,
                                "direction": -1, "entry": float(b1["low"] - 0.5),
                                "sl_dist": float(sl_pts), "tgt_dist": float(tp_pts),
                                "level": lvl_name, "global_bar_idx": len(all_3m) + b_idx + 1,
                            })
                            break

        all_3m.extend(day_records)

        if inc_virgin:
            surviving = []
            for v_p, v_tc, v_bc, o_d in active_virgin:
                if not ((cur_df["low"].min() <= v_tc) and (cur_df["high"].max() >= v_bc)):
                    surviving.append((v_p, v_tc, v_bc, o_d))
            active_virgin = surviving
            if cpr["virgin"]:
                active_virgin.append((cpr["pivot"], cpr["tc"], cpr["bc"], day))

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

    # Options Execution Engine (Trailing SL @ +6.0 pts, 2.0 pt step)
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

    return {
        "name": exp_name,
        "total_trades": len(df_sig),
        "days": len(day_pnls),
        "avg_tr_day": len(df_sig) / len(day_pnls),
        "win_rate": wr,
        "daily_win_rate": daily_wr,
        "green_days": green_days,
        "red_days": len(day_pnls) - green_days,
        "pf": pf,
        "net_points": float(df_sig["pnl_pts"].sum()),
        "net_rs": float(df_sig["net_rs"].sum()),
        "max_dd": max_dd,
        "calmar": calmar,
    }


def main():
    df_days, sorted_days = load_dataset()
    print(f"Loaded {len(sorted_days)} Historical Trading Days.")

    experiments = [
        ("Exp 0: Baseline Master (Score >= 50)", {"cam_tier": 2, "ema_5m": False, "fib": False, "opening_3m": False, "virgin": False, "pd_vwap": False}),
        ("Exp 1: + Camarilla H3/L3 in Tier 1 Supreme", {"cam_tier": 1, "ema_5m": False, "fib": False, "opening_3m": False, "virgin": False, "pd_vwap": False}),
        ("Exp 2: + 5m EMA 20 & 5m EMA 200 in Tier 1", {"cam_tier": 2, "ema_5m": True, "fib": False, "opening_3m": False, "virgin": False, "pd_vwap": False}),
        ("Exp 3: + Virgin CPR in Tier 1 Supreme", {"cam_tier": 2, "ema_5m": False, "fib": False, "opening_3m": False, "virgin": True, "pd_vwap": False}),
        ("Exp 4: + Opening 3m Candle High/Low in Tier 2", {"cam_tier": 2, "ema_5m": False, "fib": False, "opening_3m": True, "virgin": False, "pd_vwap": False}),
        ("Exp 5: + Fibonacci H3/L3 in Tier 3", {"cam_tier": 2, "ema_5m": False, "fib": True, "opening_3m": False, "virgin": False, "pd_vwap": False}),
        ("Exp 6: Combined Supreme Engine (All Features Integrated)", {"cam_tier": 1, "ema_5m": True, "fib": True, "opening_3m": True, "virgin": True, "pd_vwap": True}),
    ]

    all_res = []
    for exp_title, exp_cfg in experiments:
        print(f"Running {exp_title}...")
        res = run_experiment(exp_title, exp_cfg, df_days, sorted_days)
        all_res.append(res)

    print("\n" + "=" * 135)
    print(f"{'EXPERIMENT CONFIGURATION':58s} | {'TRADES':8s} | {'WIN RATE':9s} | {'GREEN DAYS':11s} | {'PF':7s} | {'NET PROFIT (Rs)':18s} | {'CALMAR':8s}")
    print("-" * 135)
    for r in all_res:
        print(f"{r['name']:58s} | {r['total_trades']:<8,d} | {r['win_rate']:<8.2f}% | {r['daily_win_rate']:<10.1f}% | {r['pf']:<7.3f} | Rs {r['net_rs']:>+14,.2f} | {r['calmar']:<8.2f}")
    print("=" * 135)

    out = ROOT / "artifacts" / "f6_hybrid" / "full_hierarchy_experiments_results.json"
    out.write_text(json.dumps(all_res, indent=2), encoding="utf-8")
    print(f"\nSaved Results JSON: {out}")


if __name__ == "__main__":
    main()
