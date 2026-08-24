"""Sweep Daily Trade Caps on RTX 3060 GPU to find the Sweet Spot for Maximum Net Points & Realized Profit.

Evaluates Daily Cap levels N in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, No Limit]
across 1,588 Trading Days (2020–2026) using APEX RUNNER parameters:
  - Initial SL: -6.00 option points
  - Lock Milestone: At +12.0 pts gain, lock SL at +10.0 pts
  - Chandelier Trail: 4.0 pts behind peak price once locked
  - Hard Target TP: +20.00 option points
  - 15-Minute Theta Time Stop: Tighten SL to -2.50 pts if stagnant
  - Session Window: 09:30 AM to 03:00 PM
  - Cooldown: 15 minutes between trades
  - Flat Fee: Rs 40.00 / trade | Lot Size: 65
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
d_atr = compute_atr_gpu(d_h, d_l, d_c, period=14)

prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)
super_setup = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_setup = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
entries_mask = super_setup | flag_setup


@torch.inference_mode()
def sweep_daily_cap():
    print("=" * 155)
    print("APEX RUNNER: DAILY TRADE CAP SWEET SPOT OPTIMIZATION (1,588 TRADING DAYS, 2020–2026)")
    print("=" * 155)

    all_months = sorted(list(set(d[:7] for d in days)))

    # Entry mask filtered for 09:30 AM (Bar 15) to 03:00 PM (Bar 345)
    active_mask = entries_mask.clone()
    active_mask[:, :15] = False
    active_mask[:, 345:] = False

    coords = torch.nonzero(active_mask, as_tuple=False)
    d_arr = coords[:, 0].cpu().numpy()
    b_arr = coords[:, 1].cpu().numpy()

    unique_days, split_indices = np.unique(d_arr, return_index=True)
    splits = np.split(np.arange(len(d_arr)), split_indices[1:])
    day_signals = {d_i: b_arr[idxs] for d_i, idxs in zip(unique_days, splits)}

    # APEX RUNNER Parameters
    initial_sl_pts = 6.0
    lock_trigger_pts = 12.0
    locked_profit_pts = 10.0
    trail_dist_pts = 4.0
    hard_tp_pts = 20.0
    time_stop_min = 15
    cooldown_min = 15

    cap_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 999]
    sweep_results = []

    t0 = time.time()

    for cap in cap_levels:
        cap_label = f"{cap} Trade{'s' if cap>1 else ''}/Day" if cap < 999 else "No Limit (Uncapped)"

        all_trades_pts = []
        all_trades_rs = []
        day_pnl_arr = np.zeros(N_DAYS, dtype=np.float32)
        day_pts_arr = np.zeros(N_DAYS, dtype=np.float32)
        total_wins = 0
        total_losses = 0

        for d_i in range(N_DAYS):
            sig_bars = day_signals.get(d_i, [])
            if len(sig_bars) == 0:
                continue

            last_exit_bar = -999
            trades_done = 0
            d_pnl = 0.0
            d_pts = 0.0

            for b in sig_bars:
                if trades_done >= cap:
                    break
                if b < last_exit_bar + cooldown_min:
                    continue  # In cooldown

                ep = float(d_c[d_i, b])
                init_sl = ep - (initial_sl_pts * 2.0)
                tp_barrier = ep + (hard_tp_pts * 2.0)

                curr_sl = init_sl
                peak = ep
                is_locked = False

                exit_px = float(d_c[d_i, 344])
                exit_bar = 344

                for fut_b in range(b + 1, 345):
                    dur = fut_b - b
                    fh = float(d_h[d_i, fut_b])
                    fl = float(d_l[d_i, fut_b])

                    if fh > peak:
                        peak = fh

                    gain_pts = (peak - ep) * 0.50

                    # 15-Min Theta Cut
                    if dur >= time_stop_min and not is_locked:
                        theta_sl = ep - (2.50 * 2.0)
                        if theta_sl > curr_sl:
                            curr_sl = theta_sl

                    # +10 Pt Profit Lock at +12 pt Gain
                    if gain_pts >= lock_trigger_pts:
                        locked_sl = ep + (locked_profit_pts * 2.0)
                        if locked_sl > curr_sl:
                            curr_sl = locked_sl
                            is_locked = True

                    # Chandelier Trail after locking
                    if is_locked:
                        trail_sl = peak - (trail_dist_pts * 2.0)
                        if trail_sl > curr_sl:
                            curr_sl = trail_sl

                    if fl <= curr_sl:
                        exit_px = curr_sl
                        exit_bar = fut_b
                        break
                    elif fh >= tp_barrier:
                        exit_px = tp_barrier
                        exit_bar = fut_b
                        break

                pts_val = (exit_px - ep) * 0.50
                rs_val = pts_val * LOT_SIZE - FEE

                all_trades_pts.append(pts_val)
                all_trades_rs.append(rs_val)
                d_pnl += rs_val
                d_pts += pts_val
                trades_done += 1
                last_exit_bar = exit_bar

                if rs_val > 0:
                    total_wins += 1
                else:
                    total_losses += 1

            day_pnl_arr[d_i] = d_pnl
            day_pts_arr[d_i] = d_pts

        n_t = len(all_trades_rs)
        if n_t == 0:
            continue

        tot_pts = float(np.sum(day_pts_arr))
        tot_rs = float(np.sum(day_pnl_arr))
        wr = (total_wins / n_t) * 100.0
        win_rs_tot = sum(r for r in all_trades_rs if r > 0)
        loss_rs_tot = abs(sum(r for r in all_trades_rs if r <= 0))
        pf = win_rs_tot / loss_rs_tot if loss_rs_tot > 0 else 99.0

        wins_pts = [p for p in all_trades_pts if p > 0]
        loss_pts = [p for p in all_trades_pts if p <= 0]
        avg_win = float(np.mean(wins_pts)) if wins_pts else 0.0
        avg_loss = float(np.mean(loss_pts)) if loss_pts else 0.0

        eq = np.cumsum(day_pnl_arr)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
        calmar = tot_rs / max_dd if max_dd > 0 else 0.0

        df_m = pd.DataFrame({"day": days, "rs": day_pnl_arr})
        df_m["month"] = df_m["day"].str[:7]
        m_pnl = df_m.groupby("month")["rs"].sum().reindex(all_months, fill_value=0.0)
        pos_m = int((m_pnl > 0).sum())
        month_wr = (pos_m / len(all_months)) * 100.0

        total_fees = n_t * FEE

        sweep_results.append({
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
            "total_fees_rs": round(total_fees, 2),
        })

    print(f"Sweep Completed in {time.time()-t0:.2f}s!\n", flush=True)

    # Print Master Sweet Spot Comparison Table
    print("=" * 165)
    print(f"{'Daily Cap':22s} | {'Trades':7s} | {'Tr/Day':6s} | {'Win Rate':9s} | {'Avg Win':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s} | {'Month WR':10s} | {'Fees Paid':12s}")
    print("-" * 165)
    for r in sweep_results:
        print(f"{r['cap_label']:22s} | {r['total_trades']:7d} | {r['avg_trades_day']:6.2f} | {r['win_rate']:7.1f}% | +{r['avg_win_pts']:5.2f} pt | {r['net_points']:+10.2f} | Rs {r['net_rs']:+14.2f} | {r['profit_factor']:6.3f} | Rs {r['max_drawdown']:9.2f} | {r['calmar_ratio']:7.3f} | {r['month_win_rate']:6.1f}% ({r['pos_months']}/{r['total_months']}) | Rs {r['total_fees_rs']:9.0f}")

    # Identify Champions
    best_points = max(sweep_results, key=lambda x: x["net_points"])
    best_calmar = max(sweep_results, key=lambda x: x["calmar_ratio"])
    best_pf = max(sweep_results, key=lambda x: x["profit_factor"])
    best_consistency = max(sweep_results, key=lambda x: x["month_win_rate"])

    print("\n" + "=" * 165)
    print(">>> SWEET SPOT VERDICTS:")
    print(f"  🏆 MAX NET POINTS CHAMPION:         {best_points['cap_label']} -> {best_points['net_points']:+,.2f} Net Points | Rs {best_points['net_rs']:+,.2f} Net PnL (PF: {best_points['profit_factor']})")
    print(f"  🛡️ BEST RISK-ADJUSTED (CALMAR):      {best_calmar['cap_label']} -> Calmar {best_calmar['calmar_ratio']} | Max DD: Rs {best_calmar['max_drawdown']:,.2f}")
    print(f"  💎 HIGHEST PROFIT FACTOR:           {best_pf['cap_label']} -> Profit Factor: {best_pf['profit_factor']}")
    print(f"  📅 HIGHEST MONTHLY CONSISTENCY:     {best_consistency['cap_label']} -> {best_consistency['month_win_rate']}% Green Months ({best_consistency['pos_months']}/{best_consistency['total_months']})")
    print("=" * 165)

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "daily_cap_sweet_spot_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "sweep_results": sweep_results,
        "best_points": best_points,
        "best_calmar": best_calmar,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Daily Cap Sweet Spot JSON Ledger]: {out_file}")


if __name__ == "__main__":
    sweep_daily_cap()
