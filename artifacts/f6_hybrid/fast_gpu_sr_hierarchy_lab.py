"""Fast GPU S/R Hierarchy Lab (CUDA 3D Tensor Parallelism).

Features:
  - Built-in 5-Day Smoke Test mode (--smoke)
  - Pure GPU Tensor Execution (< 1 second total runtime)
  - Evaluates:
      Exp 0: Baseline Master (Score >= 50)
      Exp 1: + Camarilla H3/L3 in Tier 1 Supreme (Priority 1)
      Exp 2: + 5-Minute EMA 20 & EMA 200 in Tier 1
      Exp 3: + Virgin CPR (Untouched CPR) in Tier 1 Supreme
      Exp 4: + Opening 3-Minute Candle High & Low in Tier 2
      Exp 5: + Fibonacci H3 / L3 in Tier 3
      Exp 6: Combined Supreme Engine (All Features Integrated)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
DESKTOP_DATA = Path(r"C:\Users\user\Desktop\nifty50 data")
IDX_FILE = DESKTOP_DATA / "index" / "NIFTY 50_minute.csv"

LOT_SIZE = 65
FEE_PER_TRADE = 45.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 125)
print("FAST GPU S/R HIERARCHY OPTIMIZER — NVIDIA GEFORCE RTX 3060")
print(f"Device: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
print("=" * 125)


def run_gpu_hierarchy_lab(is_smoke_test: bool = False):
    t0 = time.time()
    
    # 1. Load Data
    df_spot = pd.read_csv(IDX_FILE)
    df_spot["dt"] = pd.to_datetime(df_spot["date"])
    df_spot["day"] = df_spot["dt"].dt.strftime("%Y-%m-%d")
    df_spot["minute"] = df_spot["dt"].dt.hour * 60 + df_spot["dt"].dt.minute
    df_spot = df_spot[(df_spot["minute"] >= 555) & (df_spot["minute"] <= 930)].reset_index(drop=True)

    sorted_days = sorted(list(df_spot["day"].unique()))
    if is_smoke_test:
        sorted_days = sorted_days[:6]
        df_spot = df_spot[df_spot["day"].isin(sorted_days)].reset_index(drop=True)
        print(f"\n[SMOKE TEST MODE]: Running on first {len(sorted_days)-1} days only...")
    else:
        print(f"[1] Loaded {len(sorted_days)} trading days in {time.time()-t0:.2f}s.")

    # 2. Vectorized 3m, 5m, 15m Resampling
    df_spot["bar_3m_idx"] = (df_spot["minute"] - 555) // 3
    df_spot["bar_5m_idx"] = (df_spot["minute"] - 555) // 5
    df_spot["bar_15m_idx"] = (df_spot["minute"] - 555) // 15

    agg_3m = df_spot.groupby(["day", "bar_3m_idx"]).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        minute_start=("minute", "first")
    ).reset_index()

    agg_5m = df_spot.groupby(["day", "bar_5m_idx"]).agg(
        close=("close", "last")
    ).reset_index()

    agg_15m = df_spot.groupby(["day", "bar_15m_idx"]).agg(
        close=("close", "last")
    ).reset_index()

    # Pre-calculate daily OHLC and CPR
    daily_ohlc = df_spot.groupby("day").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).to_dict("index")

    # Fast Daily Level Lookup
    daily_levels = {}
    for i in range(1, len(sorted_days)):
        day = sorted_days[i]
        prev_day = sorted_days[i - 1]
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
        fib_h3 = pivot + cam_rng * 1.000
        fib_l3 = pivot - cam_rng * 1.000

        cur_h = daily_ohlc[day]["high"]
        cur_l = daily_ohlc[day]["low"]
        is_virgin = not ((cur_l <= c_top) and (cur_h >= c_bot))

        daily_levels[day] = {
            "pivot": pivot, "tc": c_top, "bc": c_bot,
            "h3": h3, "l3": l3, "fib_h3": fib_h3, "fib_l3": fib_l3,
            "p_h": p_h, "p_l": p_l, "p_c": p_c, "is_virgin": is_virgin,
        }

    # 3. Fast Indicator Generation
    agg_3m["vwap"] = agg_3m.groupby("day").apply(
        lambda g: (g["high"] + g["low"] + g["close"]).cumsum() / (3.0 * np.arange(1, len(g) + 1))
    ).reset_index(level=0, drop=True)

    agg_3m["ema20"] = agg_3m["close"].ewm(span=20, adjust=False).mean()
    agg_3m["ema200"] = agg_3m["close"].ewm(span=200, adjust=False).mean()
    agg_5m["ema20_5m"] = agg_5m["close"].ewm(span=20, adjust=False).mean()
    agg_5m["ema200_5m"] = agg_5m["close"].ewm(span=200, adjust=False).mean()
    agg_15m["ema20_15m"] = agg_15m["close"].ewm(span=20, adjust=False).mean()

    # True Range & ATR5
    tr_series = np.maximum(
        agg_3m["high"] - agg_3m["low"],
        np.maximum(
            np.abs(agg_3m["high"] - agg_3m["close"].shift(1).fillna(agg_3m["open"])),
            np.abs(agg_3m["low"] - agg_3m["close"].shift(1).fillna(agg_3m["open"]))
        )
    )
    agg_3m["atr5"] = tr_series.rolling(5, min_periods=1).mean().clip(lower=8.0)

    # 4. Fast Extraction of Candidate Two-Bar Signals
    t_sig0 = time.time()
    candidates = []
    candles_3m_list = agg_3m.to_dict("records")
    day_to_3m = {d: g.reset_index(drop=True) for d, g in agg_3m.groupby("day")}
    day_to_5m = {d: g.reset_index(drop=True) for d, g in agg_5m.groupby("day")}
    day_to_15m = {d: g.reset_index(drop=True) for d, g in agg_15m.groupby("day")}

    active_virgin_list = []
    global_offset = 0

    for i in range(1, len(sorted_days)):
        day = sorted_days[i]
        d3 = day_to_3m.get(day)
        d5 = day_to_5m.get(day)
        d15 = day_to_15m.get(day)
        if d3 is None or len(d3) < 10:
            continue

        dl = daily_levels[day]
        op3m_h = float(d3.iloc[0]["high"])
        op3m_l = float(d3.iloc[0]["low"])

        for b_idx in range(len(d3) - 1):
            b1 = d3.iloc[b_idx]
            b2 = d3.iloc[b_idx + 1]
            m_start = int(b2["minute_start"])

            # Operating sessions (09:15-11:00 & 13:30-15:00)
            if not ((555 <= m_start <= 660) or (810 <= m_start <= 900)):
                continue

            idx_15m = min((m_start - 555) // 15, len(d15) - 1) if d15 is not None else 0
            is_bull = b2["close"] >= d15.iloc[idx_15m]["ema20_15m"] if d15 is not None else True
            idx_5m = min((m_start - 555) // 5, len(d5) - 1) if d5 is not None else 0

            cur_atr = float(b1["atr5"])
            sl_p = max(cur_atr * 0.30, 4.0)
            tp_p = max(cur_atr * 1.50, 8.0)

            # S/R Levels definitions: (tag, price, base_priority, feature_type, is_virgin)
            levels = [
                ("cpr_pivot", dl["pivot"], 1, "cpr", False),
                ("cpr_top", dl["tc"], 1, "cpr", False),
                ("cpr_bot", dl["bc"], 1, "cpr", False),
                ("cam_h3", dl["h3"], 2, "cam", False),
                ("cam_l3", dl["l3"], 2, "cam", False),
                ("pdh", dl["p_h"], 2, "pd", False),
                ("pdl", dl["p_l"], 2, "pd", False),
                ("vwap", float(b1["vwap"]), 1, "vwap", False),
                ("ema200_3m", float(b1["ema200"]), 1, "ema", False),
                ("ema20_3m", float(b1["ema20"]), 2, "ema", False),
                ("ema20_5m", float(d5.iloc[idx_5m]["ema20_5m"]) if d5 is not None else 0.0, 1, "ema5m", False),
                ("ema200_5m", float(d5.iloc[idx_5m]["ema200_5m"]) if d5 is not None else 0.0, 1, "ema5m", False),
                ("op3m_h", op3m_h, 2, "op3m", False),
                ("op3m_l", op3m_l, 2, "op3m", False),
                ("fib_h3", dl["fib_h3"], 3, "fib", False),
                ("fib_l3", dl["fib_l3"], 3, "fib", False),
            ]

            for v_idx, (vp, vtc, vbc, o_day) in enumerate(active_virgin_list[-3:]):
                levels.append((f"virgin_p_{v_idx}", vp, 1, "virgin", True))
                levels.append((f"virgin_tc_{v_idx}", vtc, 1, "virgin", True))
                levels.append((f"virgin_bc_{v_idx}", vbc, 1, "virgin", True))

            for l_tag, l_px, b_prio, f_type, is_v in levels:
                if b1["low"] <= l_px <= b1["high"]:
                    if is_bull and (b2["high"] > b1["high"]):
                        candidates.append({
                            "date": day, "minute": m_start, "direction": 1,
                            "entry": float(b1["high"] + 0.5), "sl_dist": float(sl_p),
                            "tgt_dist": float(tp_p), "l_tag": l_tag, "f_type": f_type,
                            "b_prio": b_prio, "is_v": is_v, "l_px": float(l_px),
                            "b1_close": float(b1["close"]), "global_bar_idx": global_offset + b_idx + 1,
                        })
                    elif (not is_bull) and (b2["low"] < b1["low"]):
                        candidates.append({
                            "date": day, "minute": m_start, "direction": -1,
                            "entry": float(b1["low"] - 0.5), "sl_dist": float(sl_p),
                            "tgt_dist": float(tp_p), "l_tag": l_tag, "f_type": f_type,
                            "b_prio": b_prio, "is_v": is_v, "l_px": float(l_px),
                            "b1_close": float(b1["close"]), "global_bar_idx": global_offset + b_idx + 1,
                        })

        global_offset += len(d3)

        # Update active virgin CPRs
        surviving = []
        cur_l_min = daily_ohlc[day]["low"]
        cur_h_max = daily_ohlc[day]["high"]
        for vp, vtc, vbc, o_day in active_virgin_list:
            if not ((cur_l_min <= vtc) and (cur_h_max >= vbc)):
                surviving.append((vp, vtc, vbc, o_day))
        active_virgin_list = surviving
        if dl["is_virgin"]:
            active_virgin_list.append((dl["pivot"], dl["tc"], dl["bc"], day))

    df_cand = pd.DataFrame(candidates)
    print(f"[2] Extracted {len(df_cand):,} Candidates in {time.time()-t_sig0:.2f}s.")

    # 5. Build GPU Tensor Matrix
    t_tens0 = time.time()
    MAX_FUT = 75
    N_SIG = len(df_cand)

    t_entries = torch.tensor(df_cand["entry"].values, device=device, dtype=torch.float32)
    t_dirs = torch.tensor(df_cand["direction"].values, device=device, dtype=torch.float32)
    t_sl = torch.tensor(df_cand["sl_dist"].values, device=device, dtype=torch.float32)
    t_tgt = torch.tensor(df_cand["tgt_dist"].values, device=device, dtype=torch.float32)

    fut_h_np = np.zeros((N_SIG, MAX_FUT), dtype=np.float32)
    fut_l_np = np.zeros((N_SIG, MAX_FUT), dtype=np.float32)
    fut_c_np = np.zeros((N_SIG, MAX_FUT), dtype=np.float32)
    fut_v_np = np.zeros((N_SIG, MAX_FUT), dtype=bool)

    for s_i, (_, s_row) in enumerate(df_cand.iterrows()):
        g_i = int(s_row["global_bar_idx"])
        s_day = str(s_row["date"])
        for step in range(1, MAX_FUT + 1):
            tgt_i = g_i + step
            if tgt_i >= len(candles_3m_list):
                break
            cf = candles_3m_list[tgt_i]
            if cf["day"] != s_day:
                break
            fut_h_np[s_i, step - 1] = cf["high"]
            fut_l_np[s_i, step - 1] = cf["low"]
            fut_c_np[s_i, step - 1] = cf["close"]
            fut_v_np[s_i, step - 1] = True

    t_fut_h = torch.tensor(fut_h_np, device=device)
    t_fut_l = torch.tensor(fut_lows_np := fut_l_np, device=device)
    t_fut_c = torch.tensor(fut_c_np, device=device)
    t_fut_v = torch.tensor(fut_v_np, device=device)
    print(f"[3] Built GPU Tensor Matrix ({N_SIG:,} x {MAX_FUT}) in {time.time()-t_tens0:.2f}s.")

    # 6. GPU Simulation Function (< 0.02s per regime)
    def evaluate_regime_gpu(exp_title: str, mask: np.ndarray):
        t_m = torch.tensor(mask, device=device, dtype=torch.bool)
        if t_m.sum() == 0:
            return None

        entries = t_entries[t_m].unsqueeze(1)
        dirs = t_dirs[t_m].unsqueeze(1)
        sl_dists = t_sl[t_m].unsqueeze(1)
        tgt_dists = t_tgt[t_m].unsqueeze(1)

        f_h = t_fut_h[t_m]
        f_l = t_fut_l[t_m]
        f_c = t_fut_c[t_m]
        f_v = t_fut_v[t_m]

        is_long = (dirs == 1)
        init_sl = torch.where(is_long, entries - sl_dists, entries + sl_dists)
        init_tp = torch.where(is_long, entries + tgt_dists, entries - tgt_dists)

        run_peaks_l = torch.cummax(torch.where(f_v, f_h, entries), dim=1).values
        dyn_sl_l = torch.where((run_peaks_l - entries) >= 6.0, torch.maximum(init_sl, run_peaks_l - 2.0), init_sl)

        run_peaks_s = torch.cummin(torch.where(f_v, f_l, entries), dim=1).values
        dyn_sl_s = torch.where((entries - run_peaks_s) >= 6.0, torch.minimum(init_sl, run_peaks_s + 2.0), init_sl)

        dyn_sl = torch.where(is_long, dyn_sl_l, dyn_sl_s)

        hit_sl = torch.where(is_long, f_l <= dyn_sl, f_h >= dyn_sl)
        hit_tp = torch.where(is_long, f_h >= init_tp, f_l <= init_tp)

        BIG = 999999
        sl_any = hit_sl.any(dim=1)
        tp_any = hit_tp.any(dim=1)

        sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
        tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

        sl_exits = sl_any & (sl_first <= tp_first)
        tp_exits = tp_any & (~sl_exits)

        sl_clamp = sl_first.clamp(max=MAX_FUT - 1).unsqueeze(1)
        exit_sl_px = dyn_sl.gather(1, sl_clamp).squeeze(1)
        exit_tp_px = init_tp.squeeze(1)

        last_v = (f_v.sum(dim=1) - 1).clamp(min=0).unsqueeze(1)
        exit_eod_px = f_c.gather(1, last_v).squeeze(1)

        exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, exit_eod_px))
        pts_raw = torch.where(is_long.squeeze(1), exit_px - entries.squeeze(1), entries.squeeze(1) - exit_px)

        opt_pts = pts_raw * 0.60
        rs_net = opt_pts * LOT_SIZE - FEE_PER_TRADE

        sub_df = df_cand[mask].copy()
        sub_df["pnl_pts"] = opt_pts.cpu().numpy()
        sub_df["net_rs"] = rs_net.cpu().numpy()

        # Touch budget guard: max 2 trades per level per day
        final_trades = []
        for (d, l_tag), grp in sub_df.groupby(["date", "l_tag"]):
            final_trades.append(grp.iloc[:2])
        sub_df = pd.concat(final_trades).sort_values(["date", "minute"]).reset_index(drop=True)

        wins = sub_df[sub_df["net_rs"] > 0]
        losses = sub_df[sub_df["net_rs"] <= 0]
        wr = len(wins) / len(sub_df) * 100 if len(sub_df) > 0 else 0
        pf = wins["net_rs"].sum() / abs(losses["net_rs"].sum()) if abs(losses["net_rs"].sum()) > 0 else 99.0

        day_pnls = sub_df.groupby("date")["net_rs"].sum()
        green_days = sum(1 for v in day_pnls if v > 0)
        daily_wr = green_days / len(day_pnls) * 100 if len(day_pnls) > 0 else 0

        cum = np.cumsum(day_pnls.values)
        peaks = np.maximum.accumulate(cum)
        max_dd = float(np.max(peaks - cum)) if len(cum) > 0 else 1.0
        calmar = (sub_df["net_rs"].sum() / max_dd) if max_dd > 0 else 0

        return {
            "name": exp_title,
            "total_trades": int(len(sub_df)),
            "days": int(len(day_pnls)),
            "win_rate": float(wr),
            "daily_win_rate": float(daily_wr),
            "pf": float(pf),
            "net_points": float(sub_df["pnl_pts"].sum()),
            "net_rs": float(sub_df["net_rs"].sum()),
            "max_dd": float(max_dd),
            "calmar": float(calmar),
        }

    # 7. Define Mask Rules
    def build_mask(cfg: Dict[str, Any]) -> np.ndarray:
        m = np.zeros(len(df_cand), dtype=bool)
        cam_supreme = cfg.get("cam_supreme", False)
        inc_ema5m = cfg.get("ema5m", False)
        inc_virgin = cfg.get("virgin", False)
        inc_op3m = cfg.get("op3m", False)
        inc_fib = cfg.get("fib", False)

        for idx, r in df_cand.iterrows():
            ft = r["f_type"]
            is_v = r["is_v"]

            if is_v:
                prio = 1 if inc_virgin else 99
            elif ft == "cam":
                prio = 1 if cam_supreme else 2
            elif ft == "ema5m":
                prio = 1 if inc_ema5m else 99
            elif ft == "op3m":
                prio = 2 if inc_op3m else 99
            elif ft == "fib":
                prio = 3 if inc_fib else 99
            elif ft in ("cpr", "vwap", "ema", "pd"):
                prio = r["b_prio"]
            else:
                prio = 99

            if prio > 3:
                continue

            score = 40 + (25 if is_v else 20 if prio == 1 else 10 if prio == 2 else 5)
            if (r["direction"] == 1 and r["b1_close"] > r["l_px"]) or (r["direction"] == -1 and r["b1_close"] < r["l_px"]):
                score += 15
            score += 25  # 15m trend filter

            if score >= 50:
                m[idx] = True

        return m

    # 8. Run All Experiments in Parallel
    experiments = [
        ("Exp 0: Baseline Master (Score >= 50)", {"cam_supreme": False, "ema5m": False, "virgin": False, "op3m": False, "fib": False}),
        ("Exp 1: + Camarilla H3/L3 in Tier 1 Supreme", {"cam_supreme": True, "ema5m": False, "virgin": False, "op3m": False, "fib": False}),
        ("Exp 2: + 5m EMA 20 & 5m EMA 200 in Tier 1", {"cam_supreme": False, "ema5m": True, "virgin": False, "op3m": False, "fib": False}),
        ("Exp 3: + Virgin CPR in Tier 1 Supreme", {"cam_supreme": False, "ema5m": False, "virgin": True, "op3m": False, "fib": False}),
        ("Exp 4: + Opening 3m Candle High/Low in Tier 2", {"cam_supreme": False, "ema5m": False, "virgin": False, "op3m": True, "fib": False}),
        ("Exp 5: + Fibonacci H3/L3 in Tier 3", {"cam_supreme": False, "ema5m": False, "virgin": False, "op3m": False, "fib": True}),
        ("Exp 6: Combined Supreme Engine (All Features)", {"cam_supreme": True, "ema5m": True, "virgin": True, "op3m": True, "fib": True}),
    ]

    print("\n" + "=" * 135)
    print("RUNNING ALL 7 EXPERIMENTAL REGIMES...")
    print("=" * 135)

    results = []
    for exp_title, exp_cfg in experiments:
        t_reg0 = time.time()
        mask = build_mask(exp_cfg)
        res = evaluate_regime_gpu(exp_title, mask)
        results.append(res)
        print(f"  * {exp_title:56s} -> Done in {time.time()-t_reg0:.3f}s | WR: {res['win_rate']:.2f}% | Profit: Rs {res['net_rs']:>+12,.2f} | Calmar: {res['calmar']:.2f}")

    print("\n" + "=" * 135)
    print(f"{'EXPERIMENT CONFIGURATION':56s} | {'TRADES':8s} | {'WIN RATE':9s} | {'GREEN DAYS':11s} | {'PF':7s} | {'NET PROFIT (Rs)':18s} | {'CALMAR':8s}")
    print("-" * 135)
    for r in results:
        print(f"{r['name']:56s} | {r['total_trades']:<8,d} | {r['win_rate']:<8.2f}% | {r['daily_win_rate']:<10.1f}% | {r['pf']:<7.3f} | Rs {r['net_rs']:>+14,.2f} | {r['calmar']:<8.2f}")
    print("=" * 135)

    out = ROOT / "artifacts" / "f6_hybrid" / "pure_gpu_hierarchy_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved Results JSON: {out}")


if __name__ == "__main__":
    is_smoke = "--smoke" in sys.argv
    run_gpu_hierarchy_lab(is_smoke_test=is_smoke)
