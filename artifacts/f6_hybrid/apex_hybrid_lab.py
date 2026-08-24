"""Quantitative Hybrid Engine: Combining Full-Day Champion + Afternoon Power Champion.

Tests 4 Distinct Hybrid Architectures across 7 Years (1,588 Days, 2020–2026):
  1. Hybrid A: Conviction Morning Gate (|Dow| >= 0.50% / High Range) + Unconditional Afternoon Power
  2. Hybrid B: Dual-Regime Geometry (Quick Morning Target + Uncapped Afternoon Runner)
  3. Hybrid C: Daily Frequency Tiering (Max 2 Morning Trades + Uncapped Afternoon)
  4. Hybrid D: Afternoon Core + Early Morning Breakout Filter (09:30-10:15 Gap Runner + 14:00-15:30 Power)
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
from artifacts.f6_hybrid.deep_dow_macro_research import load_dow_metrics, build_nifty_dow_table

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
N_DAYS, T_BARS = d_c.shape
ALL_MONTHS = sorted(list(set(d[:7] for d in days)))
OOS_DAYS_MASK = np.array([d >= "2023-01-01" for d in days])

dow_df = load_dow_metrics()
dow_lookup = build_nifty_dow_table(days, dow_df)
dow_rets = np.array([dow_lookup[d]["dow_ret_pct"] for d in days], dtype=np.float32)
d_dow_ret = torch.tensor(dow_rets, dtype=torch.float32, device=device)

# Quad Stochastics & Entry Setup Masks
s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)

super_mask = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_mask = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
both_mask = super_mask | flag_mask


@torch.inference_mode()
def simulate_hybrid_trade(
    active_mask: torch.Tensor,
    initial_sl_pts: float,
    lock_trigger_pts: float,
    locked_profit_pts: float,
    trail_dist_pts: float,
    hard_tp_pts: float,
):
    coords = torch.nonzero(active_mask, as_tuple=False)
    M = coords.shape[0]
    if M == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

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

    init_sl = ep_exp - (initial_sl_pts * 2.0)
    is_locked = gains >= (lock_trigger_pts * 2.0)
    locked_sl = ep_exp + (locked_profit_pts * 2.0)
    trail_sl = running_peaks - (trail_dist_pts * 2.0)

    dyn_sl = init_sl.clone()
    dyn_sl = torch.where(is_locked, torch.maximum(dyn_sl, locked_sl), dyn_sl)
    dyn_sl = torch.where(is_locked, torch.maximum(dyn_sl, trail_sl), dyn_sl)
    tp_barrier = ep_exp + (hard_tp_pts * 2.0)

    hit_sl = fut_l_m <= dyn_sl
    hit_tp = fut_h_m >= tp_barrier

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    sl_idx_clamp = sl_first.clamp(max=max_future - 1).unsqueeze(1)
    exit_sl_px = dyn_sl.gather(1, sl_idx_clamp).squeeze(1)
    exit_tp_px = tp_barrier.squeeze(1)

    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px))
    pts = (exit_px - ep) * 0.50
    rs_net = pts * LOT_SIZE - FEE

    return d_idx.cpu().numpy(), b_idx.cpu().numpy(), pts.cpu().numpy(), rs_net.cpu().numpy()


def evaluate_hybrid_portfolio(trades_df: pd.DataFrame, title: str) -> dict:
    n_t = len(trades_df)
    if n_t == 0:
        return {}

    wins = trades_df["rs_net"] > 0
    losses = trades_df["rs_net"] <= 0
    n_w = int(wins.sum())
    n_l = int(losses.sum())
    wr = (n_w / n_t) * 100.0

    tot_pts = float(trades_df["pts"].sum())
    tot_rs = float(trades_df["rs_net"].sum())

    win_sum = float(trades_df.loc[wins, "rs_net"].sum()) if n_w > 0 else 0.0
    loss_sum = abs(float(trades_df.loc[losses, "rs_net"].sum())) if n_l > 0 else 0.0
    pf = win_sum / loss_sum if loss_sum > 0 else 99.0

    avg_w_pts = float(trades_df.loc[wins, "pts"].mean()) if n_w > 0 else 0.0
    avg_l_pts = float(trades_df.loc[losses, "pts"].mean()) if n_l > 0 else 0.0
    asym = abs(avg_w_pts / avg_l_pts) if abs(avg_l_pts) > 0 else 0.0

    # Daily aggregation
    day_pnl = np.zeros(N_DAYS, dtype=np.float32)
    for _, row in trades_df.iterrows():
        day_pnl[int(row["day_idx"])] += row["rs_net"]

    eq = np.cumsum(day_pnl)
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
    calmar = tot_rs / max_dd if max_dd > 0 else 0.0

    df_m = pd.DataFrame({"day": days, "rs": day_pnl})
    df_m["month"] = df_m["day"].str[:7]
    m_pnl = df_m.groupby("month")["rs"].sum().reindex(ALL_MONTHS, fill_value=0.0)
    pos_m = int((m_pnl > 0).sum())
    m_wr = (pos_m / len(ALL_MONTHS)) * 100.0

    # OOS
    oos_pnl = float(day_pnl[OOS_DAYS_MASK].sum())
    oos_pts = float(trades_df[trades_df["date"] >= "2023-01-01"]["pts"].sum())

    return {
        "title": title,
        "trades": n_t,
        "win_rate": round(wr, 2),
        "avg_win_pts": round(avg_w_pts, 2),
        "avg_loss_pts": round(avg_l_pts, 2),
        "asymmetry_ratio": round(asym, 2),
        "net_points": round(tot_pts, 2),
        "net_rs": round(tot_rs, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown": round(max_dd, 2),
        "calmar_ratio": round(calmar, 3),
        "month_win_rate": round(m_wr, 1),
        "pos_months": pos_m,
        "tot_months": len(ALL_MONTHS),
        "oos_net_rs": round(oos_pnl, 2),
        "oos_net_pts": round(oos_pts, 2),
    }


def run_hybrid_comparison():
    print("=" * 145)
    print("HYBRID ARCHITECTURE LAB: COMBINING FULL-DAY CHAMPION + AFTERNOON POWER CHAMPION")
    print("=" * 145)

    # 1. Baseline Full-Day Champion (All-Day, SL=-4, Lock=+7 @ +8, Trail=2.0, Hard TP=25.0)
    mask_full = both_mask.clone()
    d1, b1, p1, r1 = simulate_hybrid_trade(mask_full, 4.0, 8.0, 7.0, 2.0, 25.0)
    df_full = pd.DataFrame({"day_idx": d1, "bar_idx": b1, "pts": p1, "rs_net": r1, "date": [days[i] for i in d1]})
    res_full = evaluate_hybrid_portfolio(df_full, "1. FULL-DAY CHAMPION BASELINE (09:15-15:00)")

    # 2. Baseline Afternoon Power Champion (14:00-15:30, SL=-4, Lock=+7 @ +8, Trail=2.0, Hard TP=999.0)
    mask_pm = both_mask.clone()
    mask_pm[:, :285] = False
    d2, b2, p2, r2 = simulate_hybrid_trade(mask_pm, 4.0, 8.0, 7.0, 2.0, 999.0)
    df_pm = pd.DataFrame({"day_idx": d2, "bar_idx": b2, "pts": p2, "rs_net": r2, "date": [days[i] for i in d2]})
    res_pm = evaluate_hybrid_portfolio(df_pm, "2. AFTERNOON POWER CHAMPION (14:00-15:30)")

    # 3. HYBRID 1: DUAL-REGIME GEOMETRY (Quick Scalp Morning + Uncapped Afternoon Mega-Runner)
    # Morning (bar < 285): SL=-4.0, Lock=+5.0 @ +7.0, Trail=2.0, TP=+15.0 (Locks fast profit, avoids giving back)
    # Afternoon (bar >= 285): SL=-4.0, Lock=+7.0 @ +8.0, Trail=2.0, TP=999.0 (Uncapped mega-runner)
    mask_am = both_mask.clone()
    mask_am[:, 285:] = False
    d_am, b_am, p_am, r_am = simulate_hybrid_trade(mask_am, 4.0, 7.0, 5.0, 2.0, 15.0)
    df_am = pd.DataFrame({"day_idx": d_am, "bar_idx": b_am, "pts": p_am, "rs_net": r_am, "date": [days[i] for i in d_am]})
    df_hyb1 = pd.concat([df_am, df_pm], ignore_index=True)
    res_hyb1 = evaluate_hybrid_portfolio(df_hyb1, "3. HYBRID DUAL-REGIME (Morning Scalp + Afternoon Mega-Runner)")

    # 4. HYBRID 2: MORNING FREQUENCY CAP (Max 2 Morning Trades + Uncapped Afternoon Power)
    # Allows morning trend participation, but strictly caps morning loss to 2 trades max to prevent chop bleed
    df_am_capped_list = []
    for d_i, grp in df_am.groupby("day_idx"):
        df_am_capped_list.append(grp.head(2))
    df_am_capped = pd.concat(df_am_capped_list, ignore_index=True) if df_am_capped_list else pd.DataFrame()
    df_hyb2 = pd.concat([df_am_capped, df_pm], ignore_index=True)
    res_hyb2 = evaluate_hybrid_portfolio(df_hyb2, "4. HYBRID FREQUENCY TIER (Max 2 Morning Trades + Afternoon Power)")

    # 5. HYBRID 3: CONVICTION OPENING SURGE + AFTERNOON POWER (09:30-10:15 Opening + 14:00-15:30 Close)
    # Skips the dead midday chop (10:15 to 14:00) entirely, capturing both the Opening Gap Momentum & Afternoon Expiry Trend
    mask_open = both_mask.clone()
    mask_open[:, :15] = False
    mask_open[:, 60:] = False  # Only 09:30 to 10:15
    d_op, b_op, p_op, r_op = simulate_hybrid_trade(mask_open, 4.0, 8.0, 7.0, 2.0, 20.0)
    df_open = pd.DataFrame({"day_idx": d_op, "bar_idx": b_op, "pts": p_op, "rs_net": r_op, "date": [days[i] for i in d_op]})
    df_hyb3 = pd.concat([df_open, df_pm], ignore_index=True)
    res_hyb3 = evaluate_hybrid_portfolio(df_hyb3, "5. HYBRID TWIN-PEAK (09:30-10:15 Opening Bell + 14:00-15:30 Power)")

    # 6. HYBRID 4: DOW MACRO FILTERED FULL-DAY (Dow Conviction Morning + Always-Active Afternoon)
    # Available on 2024-2025 overlapping data
    df_am_dow = df_full[(df_full["bar_idx"] < 285) & (df_full["date"].map(lambda d: abs(dow_lookup[d]["dow_ret_pct"]) >= 0.50))].copy()
    df_pm_all = df_full[df_full["bar_idx"] >= 285].copy()
    df_hyb4 = pd.concat([df_am_dow, df_pm_all], ignore_index=True)
    res_hyb4 = evaluate_hybrid_portfolio(df_hyb4, "6. HYBRID MACRO-GATED (Dow >=0.50% Morning + Afternoon Always)")

    all_comparisons = [res_full, res_pm, res_hyb1, res_hyb2, res_hyb3, res_hyb4]

    print(f"\n{'Architecture Model':65s} | {'Trades':7s} | {'Win Rate':9s} | {'Avg Win':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s} | {'Month WR':10s}")
    print("-" * 170)
    for r in all_comparisons:
        print(f"{r['title']:65s} | {r['trades']:7d} | {r['win_rate']:7.1f}% | +{r['avg_win_pts']:5.2f} pt | {r['net_points']:+10.2f} | Rs {r['net_rs']:+14.2f} | {r['profit_factor']:6.3f} | Rs {r['max_drawdown']:9.2f} | {r['calmar_ratio']:7.3f} | {r['month_win_rate']:6.1f}% ({r['pos_months']}/{r['tot_months']})")

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "apex_hybrid_champions_results.json"
    out_file.write_text(json.dumps(all_comparisons, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Hybrid Comparison Results]: {out_file}")


if __name__ == "__main__":
    run_hybrid_comparison()
