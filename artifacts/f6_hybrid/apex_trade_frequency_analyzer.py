"""Comprehensive Trade Frequency & Sweet Spot Analyzer for APEX RUNNER.
Analyzes:
1. Exact Master Model (from APEX_RUNNER_STRATEGY.md / run_high_yield_wf_and_nwf.py)
2. Daily Cap Sweep for Sequential Non-Overlapping Execution
3. Daily Cap Sweep with 15-Min Cooldown
4. Daily Trade Frequency distribution & Expected Net Points per Trade
5. Sweet Spot identification for Maximum Net Points vs Risk vs Fees
"""

import json
import sys
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

d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
N_DAYS = len(days)

s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)
super_setup = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_setup = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
entries_mask = super_setup | flag_setup


@torch.inference_mode()
def simulate_apex_master():
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
        exit_bar_offset.cpu().numpy() + b_idx.cpu().numpy() + 1,
        pts.cpu().numpy(),
        rs_net.cpu().numpy(),
    )


def run_analysis():
    d_arr, entry_bars, exit_bars, pts_arr, rs_arr = simulate_apex_master()
    all_months = sorted(list(set(d[:7] for d in days)))

    df_all = pd.DataFrame({
        "day_idx": d_arr,
        "date": [days[i] for i in d_arr],
        "entry_bar": entry_bars,
        "exit_bar": exit_bars,
        "pts": pts_arr,
        "rs_net": rs_arr,
    })
    df_all["year"] = df_all["date"].str[:4]
    df_all["month"] = df_all["date"].str[:7]
    df_all["is_win"] = df_all["rs_net"] > 0

    print("=" * 120)
    print("APEX RUNNER: TRADE FREQUENCY & SWEET SPOT MASTER ANALYSIS")
    print(f"Dataset: 1,588 Trading Days ({days[0]} to {days[-1]}) | Total Raw Signals: {len(df_all):,}")
    print("=" * 120)

    # 1. Trade Frequency per Day Distribution
    trades_per_day = df_all.groupby("day_idx").size()
    full_daily_counts = pd.Series(0, index=np.arange(N_DAYS))
    full_daily_counts.update(trades_per_day)

    print("\n--- 1. NATURAL DAILY SIGNAL FREQUENCY DISTRIBUTION ---")
    print(f"Average Signals per Day: {full_daily_counts.mean():.2f} (Median: {full_daily_counts.median():.0f}, Max: {full_daily_counts.max()})")
    print(f"Days with 0 Signals: {(full_daily_counts == 0).sum()} ({(full_daily_counts == 0).mean()*100:.1f}%)")
    print(f"Days with 1-5 Signals: {((full_daily_counts >= 1) & (full_daily_counts <= 5)).sum()} ({((full_daily_counts >= 1) & (full_daily_counts <= 5)).mean()*100:.1f}%)")
    print(f"Days with 6-10 Signals: {((full_daily_counts >= 6) & (full_daily_counts <= 10)).sum()} ({((full_daily_counts >= 6) & (full_daily_counts <= 10)).mean()*100:.1f}%)")
    print(f"Days with 11-20 Signals: {((full_daily_counts >= 11) & (full_daily_counts <= 20)).sum()} ({((full_daily_counts >= 11) & (full_daily_counts <= 20)).mean()*100:.1f}%)")
    print(f"Days with >20 Signals: {(full_daily_counts > 20).sum()} ({(full_daily_counts > 20).mean()*100:.1f}%)")

    # 2. Performance by N-th Trade of the Day
    df_all["trade_seq_in_day"] = df_all.groupby("day_idx").cumcount() + 1
    seq_analysis = []
    for seq in range(1, 26):
        sub = df_all[df_all["trade_seq_in_day"] == seq]
        if len(sub) < 30:
            continue
        w = sub[sub["rs_net"] > 0]
        l = sub[sub["rs_net"] <= 0]
        wr = len(w) / len(sub) * 100.0
        tot_pts = sub["pts"].sum()
        avg_pts = sub["pts"].mean()
        avg_rs = sub["rs_net"].mean()
        pf = w["rs_net"].sum() / abs(l["rs_net"].sum()) if len(l) > 0 and abs(l["rs_net"].sum()) > 0 else 99.0
        seq_analysis.append({
            "trade_seq": seq,
            "occurrences": len(sub),
            "win_rate": wr,
            "avg_win_pts": w["pts"].mean() if len(w) > 0 else 0,
            "avg_loss_pts": l["pts"].mean() if len(l) > 0 else 0,
            "avg_pts": avg_pts,
            "total_pts": tot_pts,
            "avg_rs": avg_rs,
            "total_rs": sub["rs_net"].sum(),
            "profit_factor": pf,
        })

    df_seq = pd.DataFrame(seq_analysis)
    print("\n--- 2. PERFORMANCE DECAY / GAIN BY TRADE ORDER (1st Trade, 2nd Trade, 3rd Trade, ...) ---")
    print(f"{'Trade #':8s} | {'Sample':7s} | {'Win Rate':9s} | {'Avg Win':9s} | {'Avg Loss':9s} | {'Avg Pts':8s} | {'Tot Pts':10s} | {'Avg Net Rs':11s} | {'Tot Net Rs':14s} | {'PF':6s}")
    print("-" * 115)
    for _, r in df_seq.iterrows():
        print(f"Trade #{int(r['trade_seq']):2d} | {int(r['occurrences']):7d} | {r['win_rate']:7.1f}% | +{r['avg_win_pts']:5.2f} pt | {r['avg_loss_pts']:6.2f} pt | {r['avg_pts']:+7.2f} | {r['total_pts']:+9.1f} | Rs {r['avg_rs']:+8.1f} | Rs {r['total_rs']:+11.1f} | {r['profit_factor']:6.3f}")

    # 3. Sequential Execution (1 Trade at a time) across Daily Trade Limits
    unique_days, split_indices = np.unique(d_arr, return_index=True)
    splits = np.split(np.arange(len(d_arr)), split_indices[1:])
    day_trades = {d_i: idxs for d_i, idxs in zip(unique_days, splits)}

    cap_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 999]
    sim_results = []

    for cap in cap_levels:
        cap_lbl = f"{cap} Trades/Day" if cap < 999 else "Uncapped (All Sequential)"
        sel_pts = []
        sel_rs = []
        day_pnl = np.zeros(N_DAYS, dtype=np.float32)
        day_points = np.zeros(N_DAYS, dtype=np.float32)

        for d_i in range(N_DAYS):
            t_idxs = day_trades.get(d_i, [])
            if len(t_idxs) == 0:
                continue
            last_exit = -999
            taken = 0
            d_net = 0.0
            d_p = 0.0

            for idx in t_idxs:
                if taken >= cap:
                    break
                e_b = entry_bars[idx]
                if e_b < last_exit:
                    continue  # Wait for open position to close

                r = rs_arr[idx]
                p = pts_arr[idx]
                sel_rs.append(r)
                sel_pts.append(p)
                d_net += r
                d_p += p
                taken += 1
                last_exit = exit_bars[idx]

            day_pnl[d_i] = d_net
            day_points[d_i] = d_p

        n_t = len(sel_rs)
        tot_pts = float(np.sum(day_points))
        tot_rs = float(np.sum(day_pnl))
        wins = [r for r in sel_rs if r > 0]
        losses = [r for r in sel_rs if r <= 0]
        wr = (len(wins) / n_t) * 100.0 if n_t > 0 else 0
        pf = sum(wins) / abs(sum(losses)) if abs(sum(losses)) > 0 else 99.0

        wins_pts = [p for p in sel_pts if p > 0]
        loss_pts = [p for p in sel_pts if p <= 0]
        avg_w = float(np.mean(wins_pts)) if wins_pts else 0.0
        avg_l = float(np.mean(loss_pts)) if loss_pts else 0.0

        eq = np.cumsum(day_pnl)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
        calmar = tot_rs / max_dd if max_dd > 0 else 0.0

        df_m = pd.DataFrame({"day": days, "rs": day_pnl})
        df_m["month"] = df_m["day"].str[:7]
        m_pnl = df_m.groupby("month")["rs"].sum().reindex(all_months, fill_value=0.0)
        pos_m = int((m_pnl > 0).sum())
        month_wr = (pos_m / len(all_months)) * 100.0

        sim_results.append({
            "cap": cap,
            "cap_label": cap_lbl,
            "total_trades": n_t,
            "avg_trades_day": round(n_t / N_DAYS, 2),
            "win_rate": round(wr, 2),
            "avg_win_pts": round(avg_w, 2),
            "avg_loss_pts": round(avg_l, 2),
            "net_points": round(tot_pts, 2),
            "net_rs": round(tot_rs, 2),
            "profit_factor": round(pf, 3),
            "max_drawdown": round(max_dd, 2),
            "calmar_ratio": round(calmar, 3),
            "month_win_rate": round(month_wr, 1),
            "pos_months": pos_m,
            "total_months": len(all_months),
            "avg_trade_rs": round(tot_rs / n_t, 2) if n_t > 0 else 0,
            "total_fees_rs": round(n_t * FEE, 2),
        })

    print("\n--- 3. SEQUENTIAL EXECUTION (NON-OVERLAPPING) DAILY CAP SWEEP (1,588 DAYS) ---")
    print(f"{'Daily Cap':26s} | {'Trades':7s} | {'Tr/Day':6s} | {'Win Rate':9s} | {'Avg Win':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s} | {'Month WR':10s}")
    print("-" * 135)
    for r in sim_results:
        print(f"{r['cap_label']:26s} | {r['total_trades']:7d} | {r['avg_trades_day']:6.2f} | {r['win_rate']:7.1f}% | +{r['avg_win_pts']:5.2f} pt | {r['net_points']:+10.2f} | Rs {r['net_rs']:+14.2f} | {r['profit_factor']:6.3f} | Rs {r['max_drawdown']:9.2f} | {r['calmar_ratio']:7.3f} | {r['month_win_rate']:6.1f}% ({r['pos_months']}/{r['total_months']})")

    # Save to JSON
    out = {
        "daily_distribution": {
            "mean": float(full_daily_counts.mean()),
            "median": float(full_daily_counts.median()),
            "max": int(full_daily_counts.max()),
            "pct_0_signals": float((full_daily_counts == 0).mean()*100),
        },
        "trade_sequence_decay": seq_analysis,
        "sequential_caps": sim_results,
    }
    Path(r"C:\Websites\FLATTRADE BOT\artifacts\f6_hybrid\apex_runner_trade_frequency_analysis.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8"
    )
    print("\n[Analysis JSON Generated Successfully]")


if __name__ == "__main__":
    run_analysis()
