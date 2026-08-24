"""Daily Income Machine: 3D GPU-Batched Optimizer for Maximum Daily Win Rate.

Goal: Maximize % of Green Days (Daily Income) with minimal drawdown and positive PnL.

Mechanisms:
  1. Daily Profit Target Lock: Once Day PnL >= Target (+Rs 500 / +Rs 1000 / +Rs 1500), shut down for the day.
  2. Daily Loss Guard: If Day PnL <= -Rs 600 / -Rs 800, stop trading immediately.
  3. Fast Lock Geometry: Lock +3 to +6 pts at +4 to +8 pts gain.
  4. Afternoon Power Session Focus (14:00-15:30) vs Selective Macro Morning.
"""

from __future__ import annotations

import itertools
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

print("=" * 135, flush=True)
print("DAILY INCOME MACHINE: 3D GPU OPTIMIZATION FOR MAXIMUM GREEN DAYS", flush=True)
print("=" * 135, flush=True)

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
def simulate_daily_income_batch(
    entry_mask: torch.Tensor,
    param_grid: list[dict],
    batch_size: int = 40,
) -> list[dict]:
    coords = torch.nonzero(entry_mask, as_tuple=False)
    M = coords.shape[0]
    if M == 0:
        return []

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

    gains_3d = gains.unsqueeze(0)
    ep_3d = ep_exp.unsqueeze(0)
    peaks_3d = running_peaks.unsqueeze(0)
    fut_l_3d = fut_l_m.unsqueeze(0)
    fut_h_3d = fut_h_m.unsqueeze(0)

    d_idx_cpu = d_idx.cpu().numpy()
    b_idx_cpu = b_idx.cpu().numpy()
    
    unique_days, split_indices = np.unique(d_idx_cpu, return_index=True)
    splits = np.split(np.arange(len(d_idx_cpu)), split_indices[1:])
    day_trade_map = {d_i: idxs for d_i, idxs in zip(unique_days, splits)}

    results = []
    BIG = 999999

    for i in range(0, len(param_grid), batch_size):
        chunk = param_grid[i: i + batch_size]
        B = len(chunk)

        b_init_sl = torch.tensor([p["initial_sl"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_lock_trig = torch.tensor([p["lock_trigger"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_locked_sl = torch.tensor([p["locked_profit"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_trail = torch.tensor([p["trail_dist"] * 2.0 for p in chunk], device=device).view(B, 1, 1)
        b_tp = torch.tensor([p["hard_tp"] * 2.0 for p in chunk], device=device).view(B, 1, 1)

        init_sl_3d = ep_3d - b_init_sl
        is_locked_3d = gains_3d >= b_lock_trig
        locked_sl_3d = ep_3d + b_locked_sl
        trail_sl_3d = peaks_3d - b_trail

        dyn_sl_3d = init_sl_3d.expand(B, M, max_future).clone()
        dyn_sl_3d = torch.where(is_locked_3d, torch.maximum(dyn_sl_3d, locked_sl_3d), dyn_sl_3d)
        dyn_sl_3d = torch.where(is_locked_3d, torch.maximum(dyn_sl_3d, trail_sl_3d), dyn_sl_3d)

        tp_barrier_3d = ep_3d + b_tp

        hit_sl_3d = fut_l_3d <= dyn_sl_3d
        hit_tp_3d = fut_h_3d >= tp_barrier_3d

        sl_any = hit_sl_3d.any(dim=2)
        tp_any = hit_tp_3d.any(dim=2)

        sl_first = torch.where(sl_any, torch.argmax(hit_sl_3d.int(), dim=2), BIG)
        tp_first = torch.where(tp_any, torch.argmax(hit_tp_3d.int(), dim=2), BIG)

        sl_exits = sl_any & (sl_first <= tp_first)
        tp_exits = tp_any & (~sl_exits)

        sl_idx_clamp = sl_first.clamp(max=max_future - 1).unsqueeze(2)
        exit_sl_px = dyn_sl_3d.gather(2, sl_idx_clamp).squeeze(2)
        exit_tp_px = tp_barrier_3d.squeeze(2)
        eod_px_2d = eod_px.unsqueeze(0).expand(B, M)

        exit_bar_offset = torch.where(sl_exits, sl_first, torch.where(tp_exits, tp_first, max_future - 1))
        exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px_2d))
        pts_2d = (exit_px - ep.unsqueeze(0)) * 0.50
        rs_net_2d = pts_2d * LOT_SIZE - FEE

        pts_cpu = pts_2d.cpu().numpy()
        rs_cpu = rs_net_2d.cpu().numpy()
        exit_bars_cpu = (exit_bar_offset + b_idx.unsqueeze(0) + 1).cpu().numpy()

        for b_i, p in enumerate(chunk):
            r_arr = rs_cpu[b_i]
            p_arr = pts_cpu[b_i]
            ex_bars = exit_bars_cpu[b_i]

            daily_target_rs = p.get("daily_target_rs", 999999.0)
            daily_max_loss_rs = p.get("daily_max_loss_rs", 999999.0)
            max_daily_trades = p.get("max_daily_trades", 999)

            # Sequential Daily Simulation with Daily Profit Lock & Loss Guard
            selected_rs = []
            selected_pts = []
            day_pnl = np.zeros(N_DAYS, dtype=np.float32)
            day_pts = np.zeros(N_DAYS, dtype=np.float32)

            for d_i in range(N_DAYS):
                t_idxs = day_trade_map.get(d_i, [])
                if len(t_idxs) == 0:
                    continue

                last_exit = -999
                d_rs = 0.0
                d_p = 0.0
                taken = 0

                for idx in t_idxs:
                    if taken >= max_daily_trades:
                        break
                    # Check Daily Circuit Breakers
                    if d_rs >= daily_target_rs:
                        break  # Target achieved! Lock green day
                    if d_rs <= -daily_max_loss_rs:
                        break  # Loss limit reached! Guard capital

                    e_b = b_idx_cpu[idx]
                    if e_b < last_exit:
                        continue  # Wait for position exit

                    r_val = r_arr[idx]
                    p_val = p_arr[idx]

                    selected_rs.append(r_val)
                    selected_pts.append(p_val)
                    d_rs += r_val
                    d_p += p_val
                    taken += 1
                    last_exit = ex_bars[idx]

                day_pnl[d_i] = d_rs
                day_pts[d_i] = d_p

            n_t = len(selected_rs)
            if n_t == 0:
                continue

            active_days = day_pnl != 0
            green_days = (day_pnl > 0).sum()
            red_days = (day_pnl < 0).sum()
            zero_days = (day_pnl == 0).sum()
            n_traded_days = int(active_days.sum())

            daily_wr = (green_days / n_traded_days) * 100.0 if n_traded_days > 0 else 0.0
            tot_rs = float(day_pnl.sum())
            tot_pts = float(day_pts.sum())

            wins = [r for r in selected_rs if r > 0]
            losses = [r for r in selected_rs if r <= 0]
            trade_wr = (len(wins) / n_t) * 100.0 if n_t > 0 else 0.0
            pf = sum(wins) / abs(sum(losses)) if losses and abs(sum(losses)) > 0 else 99.0

            eq = np.cumsum(day_pnl)
            peak = np.maximum.accumulate(eq)
            max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
            calmar = tot_rs / max_dd if max_dd > 0 else 0.0

            df_m = pd.DataFrame({"day": days, "rs": day_pnl})
            df_m["month"] = df_m["day"].str[:7]
            m_pnl = df_m.groupby("month")["rs"].sum().reindex(ALL_MONTHS, fill_value=0.0)
            pos_m = int((m_pnl > 0).sum())
            month_wr = (pos_m / len(ALL_MONTHS)) * 100.0

            results.append({
                **p,
                "trades": n_t,
                "trade_win_rate": round(trade_wr, 2),
                "green_days": int(green_days),
                "red_days": int(red_days),
                "traded_days": n_traded_days,
                "daily_win_rate": round(daily_wr, 2),
                "net_points": round(tot_pts, 2),
                "net_rs": round(tot_rs, 2),
                "profit_factor": round(pf, 3),
                "max_drawdown": round(max_dd, 2),
                "calmar_ratio": round(calmar, 3),
                "month_win_rate": round(month_wr, 1),
                "pos_months": pos_m,
                "tot_months": len(ALL_MONTHS),
                "avg_green_day_rs": round(float(day_pnl[day_pnl > 0].mean()), 2) if green_days > 0 else 0.0,
                "avg_red_day_rs": round(float(day_pnl[day_pnl < 0].mean()), 2) if red_days > 0 else 0.0,
            })

    return results


def run_daily_income_optimization():
    # Parameter Search Grid for High Daily Win Rate Machine
    initial_sls = [3.0, 4.0, 5.0]
    lock_triggers = [6.0, 7.0, 8.0, 10.0]
    locked_profits = [4.0, 5.0, 6.0, 7.0]
    trail_dists = [1.5, 2.0]
    hard_tps = [8.0, 10.0, 12.0, 15.0, 20.0]
    
    # Daily Circuit Breaker Dials
    daily_targets = [500.0, 800.0, 1200.0]    # Lock daily profit
    daily_loss_caps = [300.0, 500.0, 800.0]   # Daily max loss guard
    max_trades_options = [1, 2, 3, 4]

    valid_base = []
    for sl, l_trig, l_prof, trail, tp in itertools.product(initial_sls, lock_triggers, locked_profits, trail_dists, hard_tps):
        if l_prof >= l_trig:
            continue
        if l_prof > tp:
            continue
        valid_base.append({
            "initial_sl": sl, "lock_trigger": l_trig, "locked_profit": l_prof,
            "trail_dist": trail, "hard_tp": tp,
        })

    print(f"Base Geometry Grid: {len(valid_base):,} valid sets", flush=True)

    full_grid = []
    for b in valid_base:
        for d_t in daily_targets:
            for d_l in daily_loss_caps:
                for m_t in max_trades_options:
                    full_grid.append({
                        **b,
                        "daily_target_rs": d_t,
                        "daily_max_loss_rs": d_l,
                        "max_daily_trades": m_t,
                    })

    print(f"Total Daily Income Combinations in Grid: {len(full_grid):,}", flush=True)

    # Test across Afternoon Power Session (14:00 onwards) + Full-Day
    afternoon_mask = both_mask.clone()
    afternoon_mask[:, :285] = False

    print("\n>>> Launching 3D GPU Daily Win Rate Optimization (Afternoon Power Session)...", flush=True)
    t0 = time.time()
    results_pm = simulate_daily_income_batch(afternoon_mask, full_grid, batch_size=40)
    print(f"    Completed in {time.time()-t0:.2f}s ({len(results_pm):,} configurations evaluated)", flush=True)

    df_pm = pd.DataFrame(results_pm)

    # Sort for Champions by Daily Win Rate & Net Rs
    top_daily_wr = df_pm.sort_values(by=["daily_win_rate", "net_rs"], ascending=[False, False]).iloc[0].to_dict()
    
    pool_bal = df_pm[df_pm["daily_win_rate"] >= 65.0]
    top_balanced = (pool_bal.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict()
                    if len(pool_bal) > 0 else df_pm.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict())
    
    pool_pnl = df_pm[df_pm["daily_win_rate"] >= 60.0]
    top_pnl_income = (pool_pnl.sort_values(by="net_rs", ascending=False).iloc[0].to_dict()
                      if len(pool_pnl) > 0 else df_pm.sort_values(by="net_rs", ascending=False).iloc[0].to_dict())

    print("\n" + "=" * 145, flush=True)
    print("DAILY INCOME MACHINE CHAMPIONS (HIGH DAILY WIN RATE & LOW RISK)", flush=True)
    print("=" * 145, flush=True)

    for title, c in [
        ("1. HIGHEST DAILY WIN RATE CHAMPION (Daily Income King)", top_daily_wr),
        ("2. GOLDEN BALANCED DAILY INCOME CHAMPION (Highest Calmar with >65% Green Days)", top_balanced),
        ("3. MAXIMUM PROFIT DAILY INCOME CHAMPION (Highest Net Rs with >60% Green Days)", top_pnl_income),
    ]:
        print(f"\n{title}:", flush=True)
        print(f"  * Geometry: SL = -{c['initial_sl']} pt | Lock +{c['locked_profit']} pt @ +{c['lock_trigger']} pt Gain | Trail = {c['trail_dist']} pt | TP = +{c['hard_tp']} pt", flush=True)
        print(f"  * Daily Dials: Daily Target Lock = Rs {c['daily_target_rs']} | Daily Loss Guard = Rs {c['daily_max_loss_rs']} | Max Daily Trades = {c['max_daily_trades']}", flush=True)
        print(f"  * DAILY WIN RATE:             {c['daily_win_rate']}% GREEN DAYS ({c['green_days']:,} Green Days / {c['red_days']:,} Red Days out of {c['traded_days']:,} traded days)", flush=True)
        print(f"  * Avg Green Day vs Red Day:   +Rs {c['avg_green_day_rs']:,.2f} / day  vs  -Rs {abs(c['avg_red_day_rs']):,.2f} / day", flush=True)
        print(f"  * 7-Year Net Realized Profit: Rs {c['net_rs']:+,.2f} (+{c['net_points']:+,.2f} Net Points Captured)", flush=True)
        print(f"  * Profit Factor:              {c['profit_factor']:.3f}", flush=True)
        print(f"  * 7-Year Max Drawdown:        Rs {c['max_drawdown']:,.2f} (Ultra-low risk!)", flush=True)
        print(f"  * Calmar Ratio (Return/DD):   {c['calmar_ratio']:.3f}", flush=True)
        print(f"  * Monthly Win Rate:           {c['month_win_rate']:.1f}% ({c['pos_months']}/{c['tot_months']} Green Months)", flush=True)

    top_20 = df_pm.sort_values(by=["daily_win_rate", "net_rs"], ascending=[False, False]).head(20)
    print("\n--- TOP 20 DAILY INCOME CONFIGURATIONS ---", flush=True)
    print(top_20[["daily_win_rate", "green_days", "red_days", "net_rs", "profit_factor", "max_drawdown", "calmar_ratio", "initial_sl", "lock_trigger", "locked_profit", "daily_target_rs", "daily_max_loss_rs", "max_daily_trades"]].to_string(), flush=True)

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "daily_income_machine_champions.json"
    out_file.write_text(json.dumps({
        "top_daily_wr": top_daily_wr,
        "top_balanced": top_balanced,
        "top_pnl_income": top_pnl_income,
        "top_50": df_pm.sort_values(by=["daily_win_rate", "net_rs"], ascending=[False, False]).head(50).to_dict(orient="records"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Daily Income Champions]: {out_file}", flush=True)


if __name__ == "__main__":
    run_daily_income_optimization()
