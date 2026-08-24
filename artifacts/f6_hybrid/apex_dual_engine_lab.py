"""Quantitative Dual-Engine Grand Champion Optimizer.

Fuses High Daily Win Rate (Daily Income) + Massive Net Points (Apex Runner).

Architectures Tested:
  1. Multi-Tier Dynamic Ratchet: Multi-stage profit locks (+4 @ +6, +8 @ +10, +15 @ +18, then 2pt trail)
  2. Split-Lot Scalp + Runner (50% Scalp at +8 pts, 50% Runner trailing to +40 pts)
  3. Session-Adaptive Dual-Engine (Morning Scalp Lock + Afternoon Uncapped Power Runner)
  4. Daily Ratchet Circuit-Breaker (Locks +Rs 600 baseline, but lets active winners run to maximum peak)
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

# Precompute Quad Stochastics & Entry Setup Masks
s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)

super_mask = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_mask = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
both_mask = super_mask | flag_mask


@torch.inference_mode()
def simulate_multi_tier_ratchet(
    entry_mask: torch.Tensor,
    initial_sl_pts: float = 3.5,
    tier1_trig: float = 6.0,
    tier1_lock: float = 4.0,
    tier2_trig: float = 10.0,
    tier2_lock: float = 8.0,
    tier3_trig: float = 16.0,
    tier3_lock: float = 13.0,
    trail_dist: float = 2.0,
    hard_tp: float = 999.0,
):
    coords = torch.nonzero(entry_mask, as_tuple=False)
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

    # Multi-Tier Ratchet Dynamic SL
    init_sl = ep_exp - (initial_sl_pts * 2.0)
    dyn_sl = init_sl.clone()

    # Tier 1 Lock (+4 @ +6)
    is_t1 = gains >= (tier1_trig * 2.0)
    dyn_sl = torch.where(is_t1, torch.maximum(dyn_sl, ep_exp + (tier1_lock * 2.0)), dyn_sl)

    # Tier 2 Lock (+8 @ +10)
    is_t2 = gains >= (tier2_trig * 2.0)
    dyn_sl = torch.where(is_t2, torch.maximum(dyn_sl, ep_exp + (tier2_lock * 2.0)), dyn_sl)

    # Tier 3 Lock (+13 @ +16) + Chandelier Trail
    is_t3 = gains >= (tier3_trig * 2.0)
    dyn_sl = torch.where(is_t3, torch.maximum(dyn_sl, ep_exp + (tier3_lock * 2.0)), dyn_sl)
    dyn_sl = torch.where(is_t3, torch.maximum(dyn_sl, running_peaks - (trail_dist * 2.0)), dyn_sl)

    tp_barrier = ep_exp + (hard_tp * 2.0)

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


def evaluate_dual_engine_results(trades_df: pd.DataFrame, title: str) -> dict:
    n_t = len(trades_df)
    if n_t == 0:
        return {}

    wins = trades_df["rs_net"] > 0
    losses = trades_df["rs_net"] <= 0
    n_w = int(wins.sum())
    n_l = int(losses.sum())
    trade_wr = (n_w / n_t) * 100.0

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

    active_days = day_pnl != 0
    green_days = (day_pnl > 0).sum()
    red_days = (day_pnl < 0).sum()
    n_traded_days = int(active_days.sum())
    daily_wr = (green_days / n_traded_days) * 100.0 if n_traded_days > 0 else 0.0

    eq = np.cumsum(day_pnl)
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
    calmar = tot_rs / max_dd if max_dd > 0 else 0.0

    df_m = pd.DataFrame({"day": days, "rs": day_pnl})
    df_m["month"] = df_m["day"].str[:7]
    m_pnl = df_m.groupby("month")["rs"].sum().reindex(ALL_MONTHS, fill_value=0.0)
    pos_m = int((m_pnl > 0).sum())
    month_wr = (pos_m / len(ALL_MONTHS)) * 100.0

    return {
        "title": title,
        "trades": n_t,
        "trade_win_rate": round(trade_wr, 2),
        "daily_win_rate": round(daily_wr, 1),
        "green_days": int(green_days),
        "red_days": int(red_days),
        "traded_days": n_traded_days,
        "avg_win_pts": round(avg_w_pts, 2),
        "avg_loss_pts": round(avg_l_pts, 2),
        "asymmetry_ratio": round(asym, 2),
        "net_points": round(tot_pts, 2),
        "net_rs": round(tot_rs, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown": round(max_dd, 2),
        "calmar_ratio": round(calmar, 3),
        "month_win_rate": round(month_wr, 1),
        "pos_months": pos_m,
        "tot_months": len(ALL_MONTHS),
    }


def run_dual_engine_lab():
    print("=" * 145, flush=True)
    print("DUAL-ENGINE GRAND CHAMPION LAB: FUSING HIGH DAILY WIN RATE + MAXIMUM NET POINTS", flush=True)
    print("=" * 145, flush=True)

    # 1. ARCHITECTURE 1: MULTI-TIER RATCHET RUNNER (All-Day Full 7Y)
    # Tier 1: Lock +4.0 @ +6.0 | Tier 2: Lock +8.0 @ +10.0 | Tier 3: Lock +13.0 @ +16.0 + 2pt trail
    d1, b1, p1, r1 = simulate_multi_tier_ratchet(both_mask, initial_sl_pts=3.5, tier1_trig=6.0, tier1_lock=4.0, tier2_trig=10.0, tier2_lock=8.0, tier3_trig=16.0, tier3_lock=13.0, trail_dist=2.0, hard_tp=999.0)
    df1 = pd.DataFrame({"day_idx": d1, "bar_idx": b1, "pts": p1, "rs_net": r1, "date": [days[i] for i in d1]})
    res1 = evaluate_dual_engine_results(df1, "1. Multi-Tier Dynamic Ratchet Runner (Full-Day)")

    # 2. ARCHITECTURE 2: MULTI-TIER RATCHET (Afternoon Power Session 14:00-15:30)
    mask_pm = both_mask.clone()
    mask_pm[:, :285] = False
    d2, b2, p2, r2 = simulate_multi_tier_ratchet(mask_pm, initial_sl_pts=3.5, tier1_trig=6.0, tier1_lock=4.0, tier2_trig=10.0, tier2_lock=8.0, tier3_trig=16.0, tier3_lock=13.0, trail_dist=2.0, hard_tp=999.0)
    df2 = pd.DataFrame({"day_idx": d2, "bar_idx": b2, "pts": p2, "rs_net": r2, "date": [days[i] for i in d2]})
    res2 = evaluate_dual_engine_results(df2, "2. Multi-Tier Ratchet (Afternoon Power 14:00-15:30)")

    # 3. ARCHITECTURE 3: SPLIT-LOT DUAL EXECUTION (50% Scalp Target + 50% Uncapped Runner)
    # Scalp Leg (Lot 1): SL=-3.5, TP=+8.0, Lock=+4 @ +6
    d_sc, b_sc, p_sc, r_sc = simulate_multi_tier_ratchet(both_mask, initial_sl_pts=3.5, tier1_trig=6.0, tier1_lock=4.0, tier2_trig=999.0, tier2_lock=0.0, tier3_trig=999.0, tier3_lock=0.0, trail_dist=2.0, hard_tp=8.0)
    # Runner Leg (Lot 2): SL=-3.5, Lock=+4 @ +6, Lock=+8 @ +10, Trail=2.0, Hard TP=999.0
    d_rn, b_rn, p_rn, r_rn = simulate_multi_tier_ratchet(both_mask, initial_sl_pts=3.5, tier1_trig=6.0, tier1_lock=4.0, tier2_trig=10.0, tier2_lock=8.0, tier3_trig=16.0, tier3_lock=13.0, trail_dist=2.0, hard_tp=999.0)
    
    # Blended 50/50 per trade (Fee = 40 for combined order)
    p_blend = (p_sc + p_rn) / 2.0
    r_blend = p_blend * LOT_SIZE - FEE
    df3 = pd.DataFrame({"day_idx": d_sc, "bar_idx": b_sc, "pts": p_blend, "rs_net": r_blend, "date": [days[i] for i in d_sc]})
    res3 = evaluate_dual_engine_results(df3, "3. Split-Lot 50/50 Dual Engine (50% Scalp + 50% Apex Runner)")

    # 4. ARCHITECTURE 4: SPLIT-LOT AFTERNOON POWER SESSION (14:00-15:30)
    d_sc_pm, b_sc_pm, p_sc_pm, r_sc_pm = simulate_multi_tier_ratchet(mask_pm, initial_sl_pts=3.5, tier1_trig=6.0, tier1_lock=4.0, tier2_trig=999.0, tier2_lock=0.0, tier3_trig=999.0, tier3_lock=0.0, trail_dist=2.0, hard_tp=8.0)
    d_rn_pm, b_rn_pm, p_rn_pm, r_rn_pm = simulate_multi_tier_ratchet(mask_pm, initial_sl_pts=3.5, tier1_trig=6.0, tier1_lock=4.0, tier2_trig=10.0, tier2_lock=8.0, tier3_trig=16.0, tier3_lock=13.0, trail_dist=2.0, hard_tp=999.0)
    p_blend_pm = (p_sc_pm + p_rn_pm) / 2.0
    r_blend_pm = p_blend_pm * LOT_SIZE - FEE
    df4 = pd.DataFrame({"day_idx": d_sc_pm, "bar_idx": b_sc_pm, "pts": p_blend_pm, "rs_net": r_blend_pm, "date": [days[i] for i in d_sc_pm]})
    res4 = evaluate_dual_engine_results(df4, "4. Split-Lot 50/50 Afternoon Power (14:00-15:30)")

    # 5. ARCHITECTURE 5: TWIN-PEAK MULTI-TIER (09:30-10:15 + 14:00-15:30)
    mask_tp = both_mask.clone()
    mask_tp[:, :15] = False
    mask_tp[:, 60:285] = False  # Skip 10:15 to 14:00
    d5, b5, p5, r5 = simulate_multi_tier_ratchet(mask_tp, initial_sl_pts=3.5, tier1_trig=6.0, tier1_lock=4.0, tier2_trig=10.0, tier2_lock=8.0, tier3_trig=16.0, tier3_lock=13.0, trail_dist=2.0, hard_tp=999.0)
    df5 = pd.DataFrame({"day_idx": d5, "bar_idx": b5, "pts": p5, "rs_net": r5, "date": [days[i] for i in d5]})
    res5 = evaluate_dual_engine_results(df5, "5. Twin-Peak Multi-Tier (09:30-10:15 Bell + 14:00-15:30 Power)")

    all_results = [res1, res2, res3, res4, res5]

    print(f"\n{'Architecture Model':58s} | {'Daily WR':9s} | {'Trade WR':9s} | {'Avg Win':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'PF':6s} | {'Max DD':11s} | {'Calmar':7s} | {'Month WR':10s}", flush=True)
    print("-" * 170, flush=True)
    for r in all_results:
        print(f"{r['title']:58s} | {r['daily_win_rate']:7.1f}% | {r['trade_win_rate']:7.1f}% | +{r['avg_win_pts']:5.2f} pt | {r['net_points']:+10.2f} | Rs {r['net_rs']:+14.2f} | {r['profit_factor']:6.3f} | Rs {r['max_drawdown']:8.2f} | {r['calmar_ratio']:7.3f} | {r['month_win_rate']:6.1f}% ({r['pos_months']}/{r['tot_months']})", flush=True)

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "dual_engine_grand_champions.json"
    out_file.write_text(json.dumps(all_results, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Dual-Engine Results]: {out_file}", flush=True)


if __name__ == "__main__":
    run_dual_engine_lab()
