"""Deep Quantitative Research on Dow Jones Index Integration for APEX RUNNER.

Evaluates multiple macro integration mechanisms across 2024–2025:
1. Overnight Dow Return Thresholds (0.1%, 0.2%, 0.3%, 0.5%, 0.75%, 1.0%)
2. Morning Gating (09:30 - 11:30 AM vs All-Day Gating)
3. Bidirectional Asymmetry: Filter CE on Bearish Dow, Filter PE on Bullish Dow
4. High-Volatility Dow Shock Days vs Flat Days
5. Dynamic Profit-Target Expansion on Co-Trend Macro Days
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.optimus_vix_atr_gpu import (
    load_gpu_all_data,
    compute_quad_stochastics_gpu,
    LOT_SIZE,
    FEE,
    BASE_SESSION_START,
    BASE_SESSION_END,
)

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dow Data
def load_dow_metrics() -> pd.DataFrame:
    dow_path = Path("C:/Users/user/Desktop/nifty50 data/DowJones1m.csv")
    df = pd.read_csv(dow_path, usecols=["time", "open", "high", "low", "close"])
    df["dt"] = pd.to_datetime(df["time"], utc=True)
    df["us_date"] = df["dt"].dt.date.astype(str)
    
    # Aggregate US Daily OHLC
    daily = df.groupby("us_date").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).sort_index()
    
    daily["prev_close"] = daily["close"].shift(1)
    daily["dow_ret_pct"] = ((daily["close"] - daily["prev_close"]) / daily["prev_close"]) * 100.0
    daily["dow_range_pct"] = ((daily["high"] - daily["low"]) / daily["prev_close"]) * 100.0
    daily["dow_overnight_gap_pct"] = ((daily["open"] - daily["prev_close"]) / daily["prev_close"]) * 100.0
    daily["dow_body_pct"] = ((daily["close"] - daily["open"]) / daily["prev_close"]) * 100.0
    return daily


# Map Dow US Trading Days to Nifty Trading Days
def build_nifty_dow_table(nifty_days: list[str], dow_df: pd.DataFrame) -> dict[str, dict]:
    dow_dates = dow_df.index.tolist()
    dow_lookup = {}
    
    for nd in nifty_days:
        prior_us = [d for d in dow_dates if d < nd]
        if not prior_us:
            dow_lookup[nd] = {
                "us_date": None, "dow_ret_pct": 0.0, "dow_range_pct": 0.0,
                "dow_body_pct": 0.0, "has_dow": False
            }
            continue
        latest_us = prior_us[-1]
        row = dow_df.loc[latest_us]
        dow_lookup[nd] = {
            "us_date": latest_us,
            "dow_ret_pct": float(row["dow_ret_pct"]) if pd.notna(row["dow_ret_pct"]) else 0.0,
            "dow_range_pct": float(row["dow_range_pct"]) if pd.notna(row["dow_range_pct"]) else 0.0,
            "dow_body_pct": float(row["dow_body_pct"]) if pd.notna(row["dow_body_pct"]) else 0.0,
            "has_dow": True,
        }
    return dow_lookup


d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
N_DAYS = len(days)

s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)
super_setup = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_setup = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
entries_mask = super_setup | flag_setup


@torch.inference_mode()
def simulate_apex_trades_tensor():
    coords = torch.nonzero(entries_mask, as_tuple=False)
    d_idx = coords[:, 0]
    b_idx = coords[:, 1]
    ep = d_c[d_idx, b_idx]

    max_future = 345 - BASE_SESSION_START - 1
    col_offsets = torch.arange(max_future, device=device).unsqueeze(0)
    col_idx = (b_idx + 1).unsqueeze(1) + col_offsets
    valid = (col_idx < BASE_SESSION_END) & (col_idx < 375)
    col_safe = col_idx.clamp(max=374)

    d_exp = d_idx.unsqueeze(1).expand(-1, max_future)
    fut_h = d_h[d_exp, col_safe]
    fut_l = d_l[d_exp, col_safe]

    fut_h_m = torch.where(valid, fut_h, torch.tensor(-1e9, device=device))
    fut_l_m = torch.where(valid, fut_l, torch.tensor(1e9, device=device))
    eod_px = d_c[d_idx, BASE_SESSION_END - 1]

    fut_h_clean = torch.where(valid, fut_h, ep.unsqueeze(1))
    running_peaks = torch.cummax(fut_h_clean, dim=1).values
    ep_exp = ep.unsqueeze(1)
    gains = running_peaks - ep_exp

    initial_sl_pts = 6.0
    lock_trigger_pts = 12.0
    locked_profit_pts = 10.0
    trail_dist_pts = 4.0
    hard_tp_pts = 20.0

    init_sl = ep_exp - (initial_sl_pts * 2.0)
    is_locked = gains >= (lock_trigger_pts * 2.0)
    locked_sl = ep_exp + (locked_profit_pts * 2.0)
    trail_sl = running_peaks - (trail_dist_pts * 2.0)

    dyn_sl = init_sl.clone()
    dyn_sl = torch.where(is_locked, torch.maximum(dyn_sl, locked_sl), dyn_sl)
    dyn_sl = torch.where(is_locked, torch.maximum(dyn_sl, trail_sl), dyn_sl)
    tp_barrier = ep_exp + (hard_tp_pts * 2.0)

    hit_sl = (fut_l_m <= dyn_sl)
    hit_tp = (fut_h_m >= tp_barrier)

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    exit_bar_offset = torch.where(sl_exits, sl_first, torch.where(tp_exits, tp_first, max_future - 1))
    exit_sl_px = dyn_sl.gather(1, sl_first.clamp(max=max_future-1).unsqueeze(1)).squeeze(1)
    exit_tp_px = tp_barrier.squeeze(1)
    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px))

    pts = (exit_px - ep) * 0.50
    rs_net = pts * LOT_SIZE - FEE

    return (
        d_idx.cpu().numpy(),
        b_idx.cpu().numpy(),
        (exit_bar_offset + b_idx + 1).cpu().numpy(),
        pts.cpu().numpy(),
        rs_net.cpu().numpy(),
    )


def deep_dow_research():
    print("=" * 145)
    print("DEEP QUANTITATIVE RESEARCH: DOW JONES MACRO OPTIMIZATION FOR APEX RUNNER")
    print("=" * 145)

    dow_df = load_dow_metrics()
    dow_lookup = build_nifty_dow_table(days, dow_df)

    d_arr, entry_bars, exit_bars, pts_arr, rs_arr = simulate_apex_trades_tensor()

    df = pd.DataFrame({
        "day_idx": d_arr,
        "date": [days[i] for i in d_arr],
        "bar_idx": entry_bars,
        "exit_bar": exit_bars,
        "pts": pts_arr,
        "rs_net": rs_arr,
    })
    df["year"] = df["date"].str[:4]
    df["dow_ret_pct"] = df["date"].map(lambda d: dow_lookup[d]["dow_ret_pct"])
    df["dow_range_pct"] = df["date"].map(lambda d: dow_lookup[d]["dow_range_pct"])
    df["has_dow"] = df["date"].map(lambda d: dow_lookup[d]["has_dow"])

    # Filter to 2024–2025 overlapping window
    df_eval = df[df["has_dow"] & (df["year"].isin(["2024", "2025"]))].copy()
    eval_days = sorted(df_eval["date"].unique())
    n_days_eval = len(eval_days)

    print(f"\nTarget Research Dataset: 2024–2025 Overlapping Period ({n_days_eval} Nifty Days with Clean Dow Data)")
    print(f"Total Base Signals in Window: {len(df_eval):,} signals")

    # =========================================================================
    # PART 1: CORRELATION & REGIME BREAKDOWN (UNFILTERED NATURE OF MACRO BIAS)
    # =========================================================================
    print("\n" + "-" * 45 + " 1. PERFORMANCE BY OVERNIGHT DOW RETURN QUANTILES " + "-" * 45)
    df_eval["dow_tier"] = pd.cut(
        df_eval["dow_ret_pct"],
        bins=[-999, -0.75, -0.30, 0.30, 0.75, 999],
        labels=["Strong Bear (< -0.75%)", "Moderate Bear (-0.75% to -0.30%)", "Neutral (-0.30% to +0.30%)", "Moderate Bull (+0.30% to +0.75%)", "Strong Bull (> +0.75%)"]
    )

    print(f"{'Dow Overnight Regime':35s} | {'Trades':7s} | {'Win Rate':9s} | {'Avg Win':9s} | {'Avg Loss':9s} | {'Avg Pts':8s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'PF':6s}")
    print("-" * 135)
    for tier, grp in df_eval.groupby("dow_tier", observed=True):
        w = grp[grp["rs_net"] > 0]
        l = grp[grp["rs_net"] <= 0]
        wr = len(w) / len(grp) * 100.0
        pf = w["rs_net"].sum() / abs(l["rs_net"].sum()) if len(l) > 0 and abs(l["rs_net"].sum()) > 0 else 99.0
        tot_pts = grp["pts"].sum()
        tot_rs = grp["rs_net"].sum()
        avg_w = w["pts"].mean() if len(w) > 0 else 0
        avg_l = l["pts"].mean() if len(l) > 0 else 0
        print(f"{str(tier):35s} | {len(grp):7d} | {wr:7.1f}% | +{avg_w:5.2f} pt | {avg_l:6.2f} pt | {grp['pts'].mean():+7.2f} | {tot_pts:+10.2f} | Rs {tot_rs:+14.2f} | {pf:6.3f}")

    # =========================================================================
    # PART 2: TESTING DOW FILTERING STRATEGIES
    # =========================================================================
    print("\n" + "=" * 145)
    print("--- 2. COMPREHENSIVE SWEEP OF DOW MACRO FILTER ARCHITECTURES ---")
    print("=" * 145)

    experiments = [
        {"name": "0. Baseline APEX RUNNER (No Dow Filter)", "type": "baseline"},
        # Morning Filters (09:30 to 11:30 AM: Bar 15 to 135)
        {"name": "1. Morning Chop Gate (Skip Morning if Dow Neutral |ret| < 0.20%)", "type": "skip_morning_neutral", "thresh": 0.20},
        {"name": "2. Morning Chop Gate (Skip Morning if Dow Neutral |ret| < 0.30%)", "type": "skip_morning_neutral", "thresh": 0.30},
        {"name": "3. Morning Chop Gate (Skip Morning if Dow Neutral |ret| < 0.40%)", "type": "skip_morning_neutral", "thresh": 0.40},
        {"name": "4. Morning Macro Gating (Only Trade Morning when |Dow| >= 0.50%)", "type": "skip_morning_neutral", "thresh": 0.50},
        # Dow Extreme Shock Filter
        {"name": "5. Dow Bull Wave Accelerator (Only Take Trades on Dow Green Days)", "type": "only_bull", "thresh": 0.0},
        {"name": "6. Dow Trend Days Only (|Dow Return| >= 0.25% All Day)", "type": "all_day_trend", "thresh": 0.25},
        {"name": "7. Dow High Range Volatility Gate (Dow Range >= 0.75%)", "type": "high_range", "thresh": 0.75},
        # Golden Afternoon + Dow Morning Combo
        {"name": "8. Golden Combo: Afternoon Always + Morning Only on Strong Dow (|ret| >= 0.30%)", "type": "golden_combo", "thresh": 0.30},
        {"name": "9. Golden Combo: Afternoon Always + Morning Only on Strong Dow (|ret| >= 0.50%)", "type": "golden_combo", "thresh": 0.50},
    ]

    exp_results = []

    for exp in experiments:
        filtered_trades = []
        etype = exp["type"]
        thresh = exp.get("thresh", 0.0)

        for _, row in df_eval.iterrows():
            b = row["bar_idx"]
            dow_ret = row["dow_ret_pct"]
            dow_rng = row["dow_range_pct"]
            pass_filter = True

            if etype == "baseline":
                pass_filter = True
            elif etype == "skip_morning_neutral":
                # If morning (b < 135 -> before 11:30 AM) and Dow is neutral -> skip
                if b < 135 and abs(dow_ret) < thresh:
                    pass_filter = False
            elif etype == "only_bull":
                if dow_ret <= 0.0:
                    pass_filter = False
            elif etype == "all_day_trend":
                if abs(dow_ret) < thresh:
                    pass_filter = False
            elif etype == "high_range":
                if dow_rng < thresh:
                    pass_filter = False
            elif etype == "golden_combo":
                # Afternoon (b >= 285 -> 14:00 onwards) always active.
                # Morning/Midday (b < 285) only active if strong Dow trend (|ret| >= thresh)
                if b < 285 and abs(dow_ret) < thresh:
                    pass_filter = False

            if pass_filter:
                filtered_trades.append(row)

        df_res = pd.DataFrame(filtered_trades) if filtered_trades else pd.DataFrame(columns=df_eval.columns)
        n_t = len(df_res)
        if n_t == 0:
            continue

        w = df_res[df_res["rs_net"] > 0]
        l = df_res[df_res["rs_net"] <= 0]
        wr = len(w) / n_t * 100.0
        tot_pts = df_res["pts"].sum()
        tot_rs = df_res["rs_net"].sum()
        pf = w["rs_net"].sum() / abs(l["rs_net"].sum()) if len(l) > 0 and abs(l["rs_net"].sum()) > 0 else 99.0
        avg_w = w["pts"].mean() if len(w) > 0 else 0
        avg_l = l["pts"].mean() if len(l) > 0 else 0

        # Max Drawdown across days in 2024-2025
        day_pnl = df_res.groupby("date")["rs_net"].sum().reindex(eval_days, fill_value=0.0)
        eq = np.cumsum(day_pnl.to_numpy())
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
        calmar = tot_rs / max_dd if max_dd > 0 else 0.0

        # Monthly Win Rate
        df_res["month"] = df_res["date"].str[:7]
        eval_months = sorted(list(set(d[:7] for d in eval_days)))
        m_pnl = df_res.groupby("month")["rs_net"].sum().reindex(eval_months, fill_value=0.0)
        pos_m = int((m_pnl > 0).sum())
        month_wr = (pos_m / len(eval_months)) * 100.0

        exp_results.append({
            "name": exp["name"],
            "trades": n_t,
            "win_rate": wr,
            "avg_win": avg_w,
            "avg_loss": avg_l,
            "net_points": tot_pts,
            "net_rs": tot_rs,
            "profit_factor": pf,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "month_wr": month_wr,
            "pos_months": pos_m,
            "tot_months": len(eval_months),
        })

    print(f"{'Strategy Configuration':72s} | {'Trades':7s} | {'Win Rate':9s} | {'Avg Win':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s} | {'Month WR':10s}")
    print("-" * 175)
    for r in exp_results:
        print(f"{r['name']:72s} | {r['trades']:7d} | {r['win_rate']:7.1f}% | +{r['avg_win']:5.2f} pt | {r['net_points']:+10.2f} | Rs {r['net_rs']:+14.2f} | {r['profit_factor']:6.3f} | Rs {r['max_drawdown']:9.2f} | {r['calmar']:7.3f} | {r['month_wr']:6.1f}% ({r['pos_months']}/{r['tot_months']})")

    # Save Results
    out_file = ROOT / "artifacts" / "f6_hybrid" / "dow_macro_research_results.json"
    out_file.write_text(json.dumps(exp_results, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Dow Macro Research Results]: {out_file}")


if __name__ == "__main__":
    deep_dow_research()
