"""Afternoon Focus vs All-Day Trade Frequency Sweet Spot Analysis."""

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
def test_afternoon_sweet_spot():
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

    df = pd.DataFrame({
        "day_idx": d_idx.cpu().numpy(),
        "bar_idx": b_idx.cpu().numpy(),
        "exit_bar": (exit_bar_offset + b_idx + 1).cpu().numpy(),
        "pts": pts.cpu().numpy(),
        "rs_net": rs_net.cpu().numpy(),
        "date": [days[i] for i in d_idx.cpu().numpy()],
    })
    df["month"] = df["date"].str[:7]
    all_months = sorted(list(set(d[:7] for d in days)))

    print("=" * 125)
    print("SESSION WINDOW COMPARISON: ALL-DAY VS AFTERNOON POWER-SESSION (14:00-15:30)")
    print("=" * 125)

    # Filter for 14:00 onwards (bar 285 to 345)
    df_pm = df[df["bar_idx"] >= 285].copy()
    
    # 1. PM Only (Concurrent)
    w_pm = df_pm[df_pm["rs_net"] > 0]
    l_pm = df_pm[df_pm["rs_net"] <= 0]
    wr_pm = len(w_pm) / len(df_pm) * 100.0
    pf_pm = w_pm["rs_net"].sum() / abs(l_pm["rs_net"].sum())
    pts_pm = df_pm["pts"].sum()
    rs_pm = df_pm["rs_net"].sum()

    m_pnl_pm = df_pm.groupby("month")["rs_net"].sum().reindex(all_months, fill_value=0.0)
    pos_m_pm = int((m_pnl_pm > 0).sum())

    print(f"\n1. AFTERNOON ONLY (14:00-15:30, ALL SIGNALS):")
    print(f"  * Total Trades:        {len(df_pm):,} trades ({len(df_pm)/N_DAYS:.2f} trades/day)")
    print(f"  * Win Rate:            {wr_pm:.2f}%")
    print(f"  * Avg Win / Loss:      +{w_pm['pts'].mean():.2f} pts / {l_pm['pts'].mean():.2f} pts (Asymmetry: {abs(w_pm['pts'].mean()/l_pm['pts'].mean()):.2f}x)")
    print(f"  * Net Points Captured: {pts_pm:+,.2f} pts")
    print(f"  * Net Realized Profit: Rs {rs_pm:+,.2f}")
    print(f"  * Profit Factor:       {pf_pm:.3f}")
    print(f"  * Monthly Win Rate:    {pos_m_pm}/{len(all_months)} ({pos_m_pm/len(all_months)*100:.1f}%)")

    # 2. Sequential Daily Cap for PM Session
    print(f"\n2. AFTERNOON ONLY (14:00-15:30) SEQUENTIAL DAILY CAP SWEEP:")
    print(f"{'Cap':15s} | {'Trades':7s} | {'Tr/Day':6s} | {'Win Rate':9s} | {'Avg Win':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Month WR':10s}")
    print("-" * 115)
    
    unique_days_pm = df_pm["day_idx"].unique()
    day_groups_pm = df_pm.groupby("day_idx")

    for cap in [1, 2, 3, 4, 5, 6, 8, 10, 999]:
        cap_lbl = f"{cap} Tr/Day" if cap < 999 else "Uncapped"
        sel_r = []
        sel_p = []
        d_pnl_arr = np.zeros(N_DAYS, dtype=np.float32)

        for d_i in range(N_DAYS):
            if d_i not in day_groups_pm.groups:
                continue
            grp = day_groups_pm.get_group(d_i)
            last_exit = -999
            taken = 0
            d_net = 0.0

            for _, row in grp.iterrows():
                if taken >= cap:
                    break
                if row["bar_idx"] < last_exit:
                    continue
                sel_r.append(row["rs_net"])
                sel_p.append(row["pts"])
                d_net += row["rs_net"]
                taken += 1
                last_exit = row["exit_bar"]

            d_pnl_arr[d_i] = d_net

        n_t = len(sel_r)
        if n_t == 0:
            continue
        w = [r for r in sel_r if r > 0]
        l = [r for r in sel_r if r <= 0]
        wr = len(w) / n_t * 100.0
        pf = sum(w) / abs(sum(l)) if abs(sum(l)) > 0 else 99.0
        tot_pts = sum(sel_p)
        tot_rs = sum(sel_r)

        eq = np.cumsum(d_pnl_arr)
        max_dd = float(np.max(np.maximum.accumulate(eq) - eq))

        df_m = pd.DataFrame({"day": days, "rs": d_pnl_arr})
        df_m["month"] = df_m["day"].str[:7]
        m_pnl = df_m.groupby("month")["rs"].sum().reindex(all_months, fill_value=0.0)
        pos_m = int((m_pnl > 0).sum())

        w_pts = [p for p in sel_p if p > 0]
        print(f"{cap_lbl:15s} | {n_t:7d} | {n_t/N_DAYS:6.2f} | {wr:7.1f}% | +{np.mean(w_pts):5.2f} pt | {tot_pts:+10.2f} | Rs {tot_rs:+12.2f} | {pf:6.3f} | Rs {max_dd:9.2f} | {pos_m/len(all_months)*100:6.1f}% ({pos_m}/{len(all_months)})")


if __name__ == "__main__":
    test_afternoon_sweet_spot()
