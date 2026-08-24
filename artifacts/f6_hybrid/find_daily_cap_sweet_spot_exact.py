"""Exact Daily Trade Cap Sweep for APEX RUNNER on RTX 3060.

Finds the exact sweet spot for maximum net points and net realized profit.
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
def simulate_apex_runner_all():
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


def main():
    print("=" * 145)
    print("APEX RUNNER: DAILY TRADE CAP SWEET SPOT SWEEP (2020–2026, 1,588 DAYS)")
    print("Settings: SL = -6.0 pts | Lock +10.0 pts at +12.0 pts gain | Trail = 4.0 pts | Hard TP = +20.0 pts")
    print("=" * 145)

    d_arr, entry_bars, exit_bars, pts_arr, rs_arr = simulate_apex_runner_all()
    all_months = sorted(list(set(d[:7] for d in days)))

    # Organize signals per day
    unique_days, split_indices = np.unique(d_arr, return_index=True)
    splits = np.split(np.arange(len(d_arr)), split_indices[1:])
    day_trades = {d_i: idxs for d_i, idxs in zip(unique_days, splits)}

    cap_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 999]
    results = []

    for cap in cap_levels:
        cap_label = f"{cap} Trade{'s' if cap>1 else ''}/Day" if cap < 999 else "No Limit (Uncapped)"
        all_selected_rs = []
        all_selected_pts = []
        day_pnl = np.zeros(N_DAYS, dtype=np.float32)
        day_pts = np.zeros(N_DAYS, dtype=np.float32)

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
                    continue  # Position already open

                r = rs_arr[idx]
                p = pts_arr[idx]
                all_selected_rs.append(r)
                all_selected_pts.append(p)
                d_net += r
                d_p += p
                taken += 1
                last_exit = exit_bars[idx]

            day_pnl[d_i] = d_net
            day_pts[d_i] = d_p

        n_t = len(all_selected_rs)
        if n_t == 0:
            continue

        tot_pts = float(np.sum(day_pts))
        tot_rs = float(np.sum(day_pnl))
        wins = [r for r in all_selected_rs if r > 0]
        losses = [r for r in all_selected_rs if r <= 0]
        wr = (len(wins) / n_t) * 100.0
        pf = sum(wins) / abs(sum(losses)) if abs(sum(losses)) > 0 else 99.0

        wins_pts = [p for p in all_selected_pts if p > 0]
        loss_pts = [p for p in all_selected_pts if p <= 0]
        avg_win = float(np.mean(wins_pts)) if wins_pts else 0.0
        avg_loss = float(np.mean(loss_pts)) if loss_pts else 0.0

        eq = np.cumsum(day_pnl)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
        calmar = tot_rs / max_dd if max_dd > 0 else 0.0

        df_m = pd.DataFrame({"day": days, "rs": day_pnl})
        df_m["month"] = df_m["day"].str[:7]
        m_pnl = df_m.groupby("month")["rs"].sum().reindex(all_months, fill_value=0.0)
        pos_m = int((m_pnl > 0).sum())
        month_wr = (pos_m / len(all_months)) * 100.0

        results.append({
            "cap": cap,
            "cap_label": cap_label,
            "total_trades": n_t,
            "avg_trades_day": round(n_t / N_DAYS, 2),
            "win_rate": round(wr, 2),
            "avg_win_pts": round(avg_win, 2),
            "avg_loss_pts": round(avg_loss, 2),
            "net_points": round(tot_pts, 2),
            "net_rs": round(tot_rs, 2),
            "profit_factor": round(pf, 3),
            "max_drawdown": round(max_dd, 2),
            "calmar_ratio": round(calmar, 3),
            "month_win_rate": round(month_wr, 1),
            "pos_months": pos_m,
            "total_months": len(all_months),
            "avg_trade_rs": round(tot_rs / n_t, 2),
            "total_fees_rs": round(n_t * FEE, 2),
        })

    # Print Master Sweet Spot Comparison Table
    print("\n" + "=" * 165)
    print(f"{'Daily Cap':22s} | {'Trades':7s} | {'Tr/Day':6s} | {'Win Rate':9s} | {'Avg Win':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s} | {'Month WR':10s} | {'Fees Paid':12s}")
    print("-" * 165)
    for r in results:
        print(f"{r['cap_label']:22s} | {r['total_trades']:7d} | {r['avg_trades_day']:6.2f} | {r['win_rate']:7.1f}% | +{r['avg_win_pts']:5.2f} pt | {r['net_points']:+10.2f} | Rs {r['net_rs']:+14.2f} | {r['profit_factor']:6.3f} | Rs {r['max_drawdown']:9.2f} | {r['calmar_ratio']:7.3f} | {r['month_win_rate']:6.1f}% ({r['pos_months']}/{r['total_months']}) | Rs {r['total_fees_rs']:9.0f}")

    best_points = max(results, key=lambda x: x["net_points"])
    best_calmar = max(results, key=lambda x: x["calmar_ratio"])
    best_pf = max(results, key=lambda x: x["profit_factor"])
    best_consistency = max(results, key=lambda x: x["month_win_rate"])

    print("\n" + "=" * 165)
    print(">>> SWEET SPOT VERDICTS:")
    print(f"  * MAX NET POINTS CHAMPION:         {best_points['cap_label']} -> {best_points['net_points']:+,.2f} Net Points | Rs {best_points['net_rs']:+,.2f} Net PnL (PF: {best_points['profit_factor']})")
    print(f"  * BEST RISK-ADJUSTED (CALMAR):      {best_calmar['cap_label']} -> Calmar {best_calmar['calmar_ratio']} | Max DD: Rs {best_calmar['max_drawdown']:,.2f}")
    print(f"  * HIGHEST PROFIT FACTOR:           {best_pf['cap_label']} -> Profit Factor: {best_pf['profit_factor']}")
    print(f"  * HIGHEST MONTHLY CONSISTENCY:     {best_consistency['cap_label']} -> {best_consistency['month_win_rate']}% Green Months ({best_consistency['pos_months']}/{best_consistency['total_months']})")
    print("=" * 165)

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "daily_cap_sweet_spot_exact_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "results": results,
        "best_points": best_points,
        "best_calmar": best_calmar,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Sweet Spot Exact JSON Ledger]: {out_file}")


if __name__ == "__main__":
    main()
