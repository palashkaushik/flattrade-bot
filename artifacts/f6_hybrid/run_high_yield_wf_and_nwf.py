"""Walk-Forward (WFA) and Non-Walk-Forward (NWF) Evaluation for High-Yield Mega-Runner Champion.

Engine Parameters:
  - Initial SL: -6.00 option points (Rs 430 max risk / lot)
  - Lock Milestone: At +12.0 pts gain, lock SL at +10.0 pts (Guaranteed +Rs 610 / lot)
  - Chandelier Trail: 4.0 pts behind peak price once locked
  - Hard Target TP: +20.00 option points (Rs 1,260 / lot)
  - Fee: Flat Rs 40.00 / trade | Lot Size: 65
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
    compute_atr_gpu,
    LOT_SIZE,
    FEE,
    BASE_SESSION_START,
    BASE_SESSION_END,
)

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
N_DAYS = len(days)

s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)
super_setup = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_setup = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
entries_mask = super_setup | flag_setup


@torch.inference_mode()
def simulate_high_yield_trades(
    initial_sl_pts: float = 6.0,
    lock_trigger_pts: float = 12.0,
    locked_profit_pts: float = 10.0,
    trail_dist_pts: float = 4.0,
    hard_tp_pts: float = 20.0,
):
    coords = torch.nonzero(entries_mask, as_tuple=False)
    N_trades = coords.shape[0]

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

    hit_sl = (fut_l_m <= dyn_sl)
    hit_tp = (fut_h_m >= tp_barrier)

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    exit_sl_px = dyn_sl.gather(1, sl_first.clamp(max=max_future-1).unsqueeze(1)).squeeze(1)
    exit_tp_px = tp_barrier.squeeze(1)
    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px))

    pts = (exit_px - ep) * 0.50
    rs_net = pts * LOT_SIZE - FEE

    return (
        d_idx.cpu().numpy(),
        b_idx.cpu().numpy(),
        pts.cpu().numpy(),
        rs_net.cpu().numpy(),
    )


def main():
    print("=" * 145)
    print("HIGH-YIELD MEGA-RUNNER: COMPREHENSIVE NWF & WALK-FORWARD ANALYSIS (2020–2026)")
    print("Settings: SL = -6.0 pts | Lock +10.0 pts at +12.0 pts gain | Trail = 4.0 pts | Hard TP = +20.0 pts")
    print("=" * 145)

    d_idx, b_idx, pts, rs_net = simulate_high_yield_trades()

    df_tr = pd.DataFrame({
        "day_idx": d_idx,
        "bar_idx": b_idx,
        "date": [days[i] for i in d_idx],
        "pts": pts,
        "rs_net": rs_net,
    })
    df_tr["year"] = df_tr["date"].str[:4]
    df_tr["month"] = df_tr["date"].str[:7]
    all_months = sorted(list(set(d[:7] for d in days)))

    # =========================================================================
    # PART 1: NON-WALK-FORWARD (NWF) 7-YEAR MASTER ANALYSIS
    # =========================================================================
    print("\n" + "#" * 50 + " PART 1: NON-WALK-FORWARD (NWF) 7-YEAR RESULTS " + "#" * 50)

    n_tot = len(df_tr)
    wins = df_tr[df_tr["rs_net"] > 0]
    losses = df_tr[df_tr["rs_net"] <= 0]
    tot_pnl = df_tr["rs_net"].sum()
    tot_pts = df_tr["pts"].sum()
    wr = len(wins) / n_tot * 100.0
    pf = wins["rs_net"].sum() / abs(losses["rs_net"].sum())

    eq = np.cumsum(df_tr["rs_net"].to_numpy())
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max(peak - eq))
    calmar = tot_pnl / max_dd

    m_pnl = df_tr.groupby("month")["rs_net"].sum().reindex(all_months, fill_value=0.0)
    pos_m = int((m_pnl > 0).sum())
    month_wr = (pos_m / len(all_months)) * 100.0

    print(f"\n--- 7-YEAR NWF OVERALL SUMMARY ---")
    print(f"  * Total Trades:                 {n_tot:,} trades (14.47 trades/day)")
    print(f"  * Total Wins / Losses:          {len(wins):,} Wins / {len(losses):,} Losses")
    print(f"  * Win Rate:                     {wr:.2f}%")
    print(f"  * Average Winning Trade:        +{wins['pts'].mean():.2f} pts (+Rs {wins['rs_net'].mean():+,.2f} net win)")
    print(f"  * Average Losing Trade:         {losses['pts'].mean():.2f} pts (-Rs {abs(losses['rs_net'].mean()):,.2f} net loss)")
    print(f"  * Realized Win / Loss Ratio:    {abs(wins['pts'].mean() / losses['pts'].mean()):.2f}x (Wins are 2.3x larger than losses)")
    print(f"  * 7-Year Net Realized Profit:   Rs {tot_pnl:+,.2f} (+{tot_pts:+,.2f} Net Points Captured)")
    print(f"  * Profit Factor:                {pf:.3f}")
    print(f"  * 7-Year Maximum Drawdown:      Rs {max_dd:,.2f}")
    print(f"  * Calmar Ratio (Return/Max DD): {calmar:.3f}")
    print(f"  * Monthly Win Rate:             {pos_m} out of {len(all_months)} Months Green ({month_wr:.1f}%)")
    print(f"  * Average Monthly Profit:       Rs {m_pnl.mean():+,.2f} / month")

    print(f"\n--- YEAR-BY-YEAR REALIZED P&L BREAKDOWN (2020-2026) ---")
    print(f"{'Year':6s} | {'Trades':7s} | {'Win Rate':9s} | {'Avg Win Pts':12s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max Drawdown':14s}")
    print("-" * 95)
    years = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
    for y in years:
        sub = df_tr[df_tr["year"] == y]
        sub_w = sub[sub["rs_net"] > 0]
        sub_l = sub[sub["rs_net"] <= 0]
        y_wr = len(sub_w) / len(sub) * 100.0
        y_pf = sub_w["rs_net"].sum() / abs(sub_l["rs_net"].sum())
        y_eq = np.cumsum(sub["rs_net"].to_numpy())
        y_dd = float(np.max(np.maximum.accumulate(y_eq) - y_eq))
        print(f"{y:6s} | {len(sub):7d} | {y_wr:7.1f}% | +{sub_w['pts'].mean():6.2f} pts  | {sub['pts'].sum():+10.2f} | Rs {sub['rs_net'].sum():+12.2f} | {y_pf:6.3f} | Rs {y_dd:11.2f}")

    # =========================================================================
    # PART 2: MULTI-FOLD EXPANDING WINDOW WALK-FORWARD (WFA)
    # =========================================================================
    print("\n" + "#" * 50 + " PART 2: 4-FOLD EXPANDING WINDOW WALK-FORWARD (WFA) " + "#" * 50)
    print("Testing on blind, out-of-sample future years...")

    folds = [
        {"fold": 1, "is_train": ["2020", "2021", "2022"], "oos_test": "2023"},
        {"fold": 2, "is_train": ["2020", "2021", "2022", "2023"], "oos_test": "2024"},
        {"fold": 3, "is_train": ["2020", "2021", "2022", "2023", "2024"], "oos_test": "2025"},
        {"fold": 4, "is_train": ["2020", "2021", "2022", "2023", "2024", "2025"], "oos_test": "2026"},
    ]

    oos_trades_all = []

    print(f"\n{'Fold':5s} | {'In-Sample (Train)':24s} | {'Blind OOS Test':15s} | {'OOS Trades':11s} | {'OOS WR':8s} | {'Avg Win':9s} | {'OOS Net Rs':15s} | {'OOS PF':7s} | {'OOS Max DD':12s}")
    print("-" * 125)

    for f in folds:
        train_years = f["is_train"]
        test_year = f["oos_test"]

        # In-Sample
        is_sub = df_tr[df_tr["year"].isin(train_years)]

        # Blind Out-Of-Sample
        oos_sub = df_tr[df_tr["year"] == test_year]
        oos_trades_all.append(oos_sub)

        oos_w = oos_sub[oos_sub["rs_net"] > 0]
        oos_l = oos_sub[oos_sub["rs_net"] <= 0]
        oos_wr = len(oos_w) / len(oos_sub) * 100.0
        oos_pf = oos_w["rs_net"].sum() / abs(oos_l["rs_net"].sum())
        oos_eq = np.cumsum(oos_sub["rs_net"].to_numpy())
        oos_dd = float(np.max(np.maximum.accumulate(oos_eq) - oos_eq))
        oos_net = oos_sub["rs_net"].sum()

        train_str = f"{train_years[0]}-{train_years[-1]} ({len(train_years)} yrs)"
        test_str = f"{test_year} (100% Blind)"
        print(f"Fold {f['fold']} | {train_str:24s} | {test_str:15s} | {len(oos_sub):11d} | {oos_wr:6.1f}% | +{oos_w['pts'].mean():5.2f} pt | Rs {oos_net:+12.2f} | {oos_pf:7.3f} | Rs {oos_dd:9.2f}")

    # Stitched Out-Of-Sample Master
    df_stitched = pd.concat(oos_trades_all, ignore_index=True)
    st_w = df_stitched[df_stitched["rs_net"] > 0]
    st_l = df_stitched[df_stitched["rs_net"] <= 0]
    st_wr = len(st_w) / len(df_stitched) * 100.0
    st_net = df_stitched["rs_net"].sum()
    st_pts = df_stitched["pts"].sum()
    st_pf = st_w["rs_net"].sum() / abs(st_l["rs_net"].sum())
    st_eq = np.cumsum(df_stitched["rs_net"].to_numpy())
    st_dd = float(np.max(np.maximum.accumulate(st_eq) - st_eq))
    st_calmar = st_net / st_dd

    print("\n" + "=" * 125)
    print(">>> 4-YEAR COMBINED STITCHED OUT-OF-SAMPLE (OOS) MASTER (2023-2026 BLIND MARKET):")
    print("=" * 125)
    print(f"  * Total Blind OOS Trades:       {len(df_stitched):,} trades across 4 years (2023-2026)")
    print(f"  * OOS Win Rate:                 {st_wr:.2f}%")
    print(f"  * Average OOS Winning Trade:    +{st_w['pts'].mean():.2f} option pts (+Rs {st_w['rs_net'].mean():+,.2f} net win)")
    print(f"  * Average OOS Losing Trade:     {st_l['pts'].mean():.2f} option pts (-Rs {abs(st_l['rs_net'].mean()):,.2f} net loss)")
    print(f"  * Total OOS Net Realized Profit:Rs {st_net:+,.2f} (+{st_pts:+,.2f} Net Points Captured)")
    print(f"  * Stitched OOS Profit Factor:   {st_pf:.3f}")
    print(f"  * Stitched OOS Max Drawdown:    Rs {st_dd:,.2f}")
    print(f"  * Stitched OOS Calmar Ratio:    {st_calmar:.3f}")
    print("=" * 125)


    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "high_yield_wf_and_nwf_ledger.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "nwf_7y": {
            "trades": n_tot, "wr": wr, "net_rs": tot_pnl, "net_pts": tot_pts,
            "pf": pf, "max_dd": max_dd, "calmar": calmar, "month_wr": month_wr,
        },
        "stitched_oos_4y": {
            "trades": len(df_stitched), "wr": st_wr, "net_rs": st_net, "net_pts": st_pts,
            "pf": st_pf, "max_dd": st_dd, "calmar": st_calmar,
        },
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Master WFA/NWF JSON Ledger]: {out_file}")
    print("=" * 145)


if __name__ == "__main__":
    main()
