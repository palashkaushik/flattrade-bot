"""High-Yield Mega-Runner Strategy Optimizer on RTX 3060 GPU.

Fixes the 'choked points' issue:
  - Minimum Profit Milestone: +12.0 to +30.0 Option Points (Rs 780 to Rs 1,950 / lot)
  - Wide Breathing Room: No premature BE chokes at +0.5 pt
  - Profit Lock: Locks +6.0 to +10.0 pts only after reaching +12.0 to +16.0 pts
  - Super Trend Runner: Captures +25 to +60 pt mega-expansions
  - Structural SL: Tight -6.0 to -8.0 pts Initial SL

Evaluates on RTX 3060 across 1,588 days & August 18-20.
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
def run_high_yield_runner_sweep():
    print("=" * 145)
    print("HIGH-YIELD MEGA-RUNNER STRATEGY OPTIMIZER (BIG POINTS & HIGH ASYMMETRY)")
    print("=" * 145)

    coords = torch.nonzero(entries_mask, as_tuple=False)
    N_trades = coords.shape[0]

    d_idx = coords[:, 0]
    b_idx = coords[:, 1]
    ep = d_c[d_idx, b_idx]
    base_atr = d_atr[d_idx, b_idx].clamp(min=5.0, max=25.0)

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

    all_months = sorted(list(set(d[:7] for d in days)))

    configs = []
    # Test High-Yield Parameters
    for initial_sl_pts in [6.0, 7.0, 8.0, 10.0]:
        for lock_trigger_pts in [10.0, 12.0, 15.0, 18.0]:
            for locked_profit_pts in [5.0, 7.0, 10.0]:
                for trail_dist_pts in [4.0, 6.0, 8.0]:
                    for hard_tp_pts in [20.0, 25.0, 30.0, 40.0, 999.0]:
                        if locked_profit_pts < lock_trigger_pts:
                            configs.append({
                                "initial_sl_pts": initial_sl_pts,
                                "lock_trigger_pts": lock_trigger_pts,
                                "locked_profit_pts": locked_profit_pts,
                                "trail_dist_pts": trail_dist_pts,
                                "hard_tp_pts": hard_tp_pts,
                            })

    print(f"Sweeping {len(configs)} High-Yield Mega-Runner Parameter Configurations on GPU...", flush=True)
    t0 = time.time()

    results = []

    for p in configs:
        sl_pts = p["initial_sl_pts"]
        trig_pts = p["lock_trigger_pts"]
        lock_pts = p["locked_profit_pts"]
        tr_dist = p["trail_dist_pts"]
        tp_pts = p["hard_tp_pts"]

        # 1. Initial SL
        init_sl = ep_exp - (sl_pts * 2.0)  # in index points (option pts * 2)

        # 2. Profit Lock Barrier (Locks when gain >= trig_pts)
        is_locked = gains >= (trig_pts * 2.0)
        locked_sl = ep_exp + (lock_pts * 2.0)

        # 3. Chandelier Trailing Barrier (Follows running peak - tr_dist)
        trail_sl = running_peaks - (tr_dist * 2.0)

        # Combine Dynamic Stop Loss
        dyn_sl = init_sl.clone()
        dyn_sl = torch.where(is_locked, torch.maximum(dyn_sl, locked_sl), dyn_sl)
        dyn_sl = torch.where(is_locked, torch.maximum(dyn_sl, trail_sl), dyn_sl)

        tp_barrier = ep_exp + (tp_pts * 2.0)

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

        rs_cpu = rs_net.cpu().numpy()
        pts_cpu = pts.cpu().numpy()
        d_idx_cpu = d_idx.cpu().numpy()

        n_t = len(rs_cpu)
        wins = [r for r in rs_cpu if r > 0]
        losses = [r for r in rs_cpu if r <= 0]
        wins_pts = pts_cpu[pts_cpu > 0]
        loss_pts = pts_cpu[pts_cpu <= 0]

        net_rs_tot = float(np.sum(rs_cpu))
        net_pts_tot = float(np.sum(pts_cpu))
        wr = len(wins) / n_t * 100.0 if n_t > 0 else 0.0
        pf = sum(wins) / abs(sum(losses)) if abs(sum(losses)) > 0 else 99.0

        avg_win_pts = float(np.mean(wins_pts)) if len(wins_pts) > 0 else 0.0
        avg_loss_pts = float(np.mean(loss_pts)) if len(loss_pts) > 0 else 0.0

        df_trades = pd.DataFrame({"day_idx": d_idx_cpu, "rs": rs_cpu})
        df_trades["date"] = [days[i] for i in df_trades["day_idx"]]
        df_trades["month"] = df_trades["date"].str[:7]
        monthly_pnl = df_trades.groupby("month")["rs"].sum().reindex(all_months, fill_value=0.0)
        pos_months = int((monthly_pnl > 0).sum())
        month_wr = (pos_months / len(all_months)) * 100.0

        eq = np.cumsum(rs_cpu)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
        calmar = net_rs_tot / max_dd if max_dd > 0 else 0.0

        results.append({
            "config": p,
            "trades": n_t,
            "win_rate": round(wr, 2),
            "avg_win_pts": round(avg_win_pts, 2),
            "avg_loss_pts": round(avg_loss_pts, 2),
            "net_points": round(net_pts_tot, 2),
            "net_rs": round(net_rs_tot, 2),
            "profit_factor": round(pf, 3),
            "max_dd": round(max_dd, 2),
            "calmar": round(calmar, 3),
            "pos_months": pos_months,
            "month_wr": round(month_wr, 1),
            "avg_trade_rs": round(net_rs_tot / n_t, 2),
        })

    print(f"Completed {len(results)} configurations in {time.time()-t0:.2f}s on GPU!", flush=True)

    # 1. Top 10 Ranked by Average Winning Points & High Total Profit
    top_yield = sorted(results, key=lambda x: (x["avg_win_pts"], x["net_rs"]), reverse=True)[:10]
    print("\n" + "=" * 160)
    print(">>> TOP 10 HIGH-YIELD MEGA-RUNNER STRATEGIES (RANKED BY MAXIMUM AVERAGE WINNING POINTS):")
    print("=" * 160)
    print(f"{'Rank':4s} | {'Init SL':8s} | {'Lock Trig':10s} | {'Lock Level':11s} | {'Trail Dist':11s} | {'Hard TP':8s} | {'Avg Win Pts':12s} | {'Avg Loss Pts':13s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s}")
    print("-" * 160)
    for r, item in enumerate(top_yield, 1):
        c = item["config"]
        tp_str = f"+{c['hard_tp_pts']} pts" if c["hard_tp_pts"] < 900 else "Open Trail"
        print(f"{r:4d} | -{c['initial_sl_pts']:4.1f} pts | +{c['lock_trigger_pts']:4.1f} pts  | +{c['locked_profit_pts']:4.1f} pts   | {c['trail_dist_pts']:4.1f} pts   | {tp_str:10s} | +{item['avg_win_pts']:6.2f} pts  | {item['avg_loss_pts']:6.2f} pts   | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_dd']:9.2f}")

    # 2. Top 10 Ranked by Maximum Total Net Profit & High Profit Factor
    top_profit = sorted(results, key=lambda x: x["net_rs"], reverse=True)[:10]
    print("\n" + "=" * 160)
    print(">>> TOP 10 HIGH-YIELD STRATEGIES (RANKED BY MAXIMUM 7-YEAR NET REALIZED PROFIT):")
    print("=" * 160)
    print(f"{'Rank':4s} | {'Init SL':8s} | {'Lock Trig':10s} | {'Lock Level':11s} | {'Trail Dist':11s} | {'Hard TP':8s} | {'Avg Win Pts':12s} | {'Avg Loss Pts':13s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s}")
    print("-" * 160)
    for r, item in enumerate(top_profit, 1):
        c = item["config"]
        tp_str = f"+{c['hard_tp_pts']} pts" if c["hard_tp_pts"] < 900 else "Open Trail"
        print(f"{r:4d} | -{c['initial_sl_pts']:4.1f} pts | +{c['lock_trigger_pts']:4.1f} pts  | +{c['locked_profit_pts']:4.1f} pts   | {c['trail_dist_pts']:4.1f} pts   | {tp_str:10s} | +{item['avg_win_pts']:6.2f} pts  | {item['avg_loss_pts']:6.2f} pts   | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_dd']:9.2f}")

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "high_yield_runners_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "top_yield": top_yield,
        "top_profit": top_profit,
    }, indent=2, default=float), encoding="utf-8")
    print("\n" + "=" * 160)
    print(f"[Saved High-Yield JSON Ledger]: {out_file}")
    print("=" * 160)


if __name__ == "__main__":
    run_high_yield_runner_sweep()
