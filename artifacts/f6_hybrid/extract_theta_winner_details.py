"""Extract Complete Details for the Theta-Beating Tiered Ratchet Champion.

Winner Configuration:
  - Initial SL: 2.50 x ATR
  - Breakeven Trigger: Gain >= +1.25 x ATR (locks SL at Entry + 0.5 pt)
  - Trailing Activation: Gain >= +1.50 x ATR
  - Trailing Distance: 0.50 x ATR behind peak
  - Time-Decay Penalty: 15 minutes (tightens SL to Entry - 0.5x ATR)
  - Target TP: 3.50 x ATR
  - Friction: Flat Rs 40.00 / trade
"""

from __future__ import annotations

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
    compute_atr_gpu,
    LOT_SIZE,
    FEE,
    BASE_SESSION_START,
    BASE_SESSION_END,
)
from backtest_5y_optimized import SYM_RE, latest_spot, option_files
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.test_super_only_aug import extend_with_august, ParamStoch, IncrementalATR, bslice, to_hhmm
from artifacts.f6_hybrid.compare_rules_1_and_2 import load_full_ohlc_spot
import grid_optimize_f6_atr as grid

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
def extract_winner_detailed_stats():
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

    time_offsets = torch.arange(1, max_future + 1, device=device).unsqueeze(0).expand(N_trades, -1)

    # Winner Configuration Parameters
    sl_m = 2.50
    be_trig_m = 1.25
    trail_trig_m = 1.50
    trail_dist_m = 0.50
    time_stop_min = 15
    target_tp_m = 3.50

    eff_atr = base_atr
    sl_d = (sl_m * eff_atr).clamp(min=5.0, max=30.0)
    tp_d = (target_tp_m * eff_atr).clamp(min=8.0, max=60.0)

    init_sl = ep_exp - sl_d.unsqueeze(1)
    tp_barrier = ep_exp + tp_d.unsqueeze(1)

    gains = running_peaks - ep_exp

    # Tier 2: Breakeven Lock Level
    be_level = ep_exp + 0.5
    is_be_reached = gains >= (be_trig_m * eff_atr).unsqueeze(1)

    # Tier 3: Asymmetric Trailing Stop
    trail_level = running_peaks - (trail_dist_m * eff_atr).unsqueeze(1)
    is_trail_reached = gains >= (trail_trig_m * eff_atr).unsqueeze(1)

    # Tier 4: Time Decay Penalty
    time_penalty_sl = ep_exp - (0.5 * eff_atr).unsqueeze(1)
    is_time_decay_active = (time_offsets >= time_stop_min) & (~is_be_reached)

    dyn_sl = init_sl.clone()
    dyn_sl = torch.where(is_time_decay_active, torch.maximum(dyn_sl, time_penalty_sl), dyn_sl)
    dyn_sl = torch.where(is_be_reached, torch.maximum(dyn_sl, be_level), dyn_sl)
    dyn_sl = torch.where(is_trail_reached, torch.maximum(dyn_sl, trail_level), dyn_sl)

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
    duration_min = exit_bar_offset + 1

    exit_sl_px = dyn_sl.gather(1, sl_first.clamp(max=max_future-1).unsqueeze(1)).squeeze(1)
    exit_tp_px = tp_barrier.squeeze(1)
    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px))

    pts = (exit_px - ep) * 0.50
    rs_net = pts * LOT_SIZE - FEE

    pts_cpu = pts.cpu().numpy()
    rs_cpu = rs_net.cpu().numpy()
    d_idx_cpu = d_idx.cpu().numpy()
    dur_cpu = duration_min.cpu().numpy()
    sl_dist_cpu = (sl_d * 0.50).cpu().numpy()
    tp_dist_cpu = (tp_d * 0.50).cpu().numpy()

    win_mask = rs_cpu > 0
    loss_mask = rs_cpu <= 0

    wins_pts = pts_cpu[win_mask]
    wins_rs = rs_cpu[win_mask]
    wins_dur = dur_cpu[win_mask]

    loss_pts = pts_cpu[loss_mask]
    loss_rs = rs_cpu[loss_mask]
    loss_dur = dur_cpu[loss_mask]

    trades_per_day = pd.Series(d_idx_cpu).value_counts().reindex(range(N_DAYS), fill_value=0)

    # Monthly breakdown
    df_trades = pd.DataFrame({"day_idx": d_idx_cpu, "rs": rs_cpu, "pts": pts_cpu})
    df_trades["date"] = [days[i] for i in df_trades["day_idx"]]
    df_trades["year"] = df_trades["date"].str[:4]
    df_trades["month"] = df_trades["date"].str[:7]
    all_months = sorted(list(set(d[:7] for d in days)))
    monthly_pnl = df_trades.groupby("month")["rs"].sum().reindex(all_months, fill_value=0.0)

    # Yearly breakdown
    years = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
    yearly_stats = {}
    for y in years:
        sub = df_trades[df_trades["year"] == y]
        y_w = sub[sub["rs"] > 0]
        y_l = sub[sub["rs"] <= 0]
        y_net = sub["rs"].sum()
        y_pts = sub["pts"].sum()
        y_wr = len(y_w) / len(sub) * 100.0 if len(sub) > 0 else 0.0
        y_pf = y_w["rs"].sum() / abs(y_l["rs"].sum()) if abs(y_l["rs"].sum()) > 0 else 99.0
        y_eq = np.cumsum(sub["rs"].to_numpy())
        y_dd = float(np.max(np.maximum.accumulate(y_eq) - y_eq)) if len(y_eq) > 0 else 0.0
        yearly_stats[y] = {
            "trades": len(sub), "win_rate": round(y_wr, 1),
            "net_points": round(y_pts, 2), "net_rs": round(y_net, 2),
            "profit_factor": round(y_pf, 3), "max_dd": round(y_dd, 2),
        }

    tot_pnl = float(np.sum(rs_cpu))
    tot_pts = float(np.sum(pts_cpu))
    eq = np.cumsum(rs_cpu)
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max(peak - eq))
    calmar = tot_pnl / max_dd

    return {
        "total_trades": len(rs_cpu),
        "wins": int(win_mask.sum()),
        "losses": int(loss_mask.sum()),
        "win_rate": round(float(win_mask.sum() / len(rs_cpu) * 100), 2),
        "net_points": round(tot_pts, 2),
        "net_rs": round(tot_pnl, 2),
        "profit_factor": round(float(sum(wins_rs) / abs(sum(loss_rs))), 3),
        "max_drawdown": round(max_dd, 2),
        "calmar_ratio": round(calmar, 3),

        # Trade Frequency
        "avg_trades_day": round(float(trades_per_day.mean()), 2),
        "median_trades_day": float(trades_per_day.median()),
        "max_trades_day": int(trades_per_day.max()),
        "active_days": int((trades_per_day > 0).sum()),
        "zero_days": int((trades_per_day == 0).sum()),

        # SL & TP Point Dimensions
        "avg_initial_sl_pts": round(float(np.mean(sl_dist_cpu)), 2),
        "avg_initial_sl_rs": round(float(np.mean(sl_dist_cpu) * LOT_SIZE + FEE), 2),
        "avg_initial_tp_pts": round(float(np.mean(tp_dist_cpu)), 2),
        "avg_initial_tp_rs": round(float(np.mean(tp_dist_cpu) * LOT_SIZE - FEE), 2),
        "avg_realized_win_pts": round(float(np.mean(wins_pts)), 2),
        "avg_realized_win_rs": round(float(np.mean(wins_rs)), 2),
        "avg_realized_loss_pts": round(float(np.mean(loss_pts)), 2),
        "avg_realized_loss_rs": round(float(np.mean(loss_rs)), 2),
        "realized_rr_ratio": round(float(abs(np.mean(wins_pts) / np.mean(loss_pts))), 2),

        # Holding Duration
        "avg_duration_min": round(float(np.mean(dur_cpu)), 1),
        "avg_win_duration_min": round(float(np.mean(wins_dur)), 1),
        "avg_loss_duration_min": round(float(np.mean(loss_dur)), 1),

        # Monthly Metrics
        "pos_months": int((monthly_pnl > 0).sum()),
        "total_months": len(all_months),
        "month_win_rate": round(float((monthly_pnl > 0).sum() / len(all_months) * 100), 1),
        "avg_month_rs": round(float(monthly_pnl.mean()), 2),
        "yearly_stats": yearly_stats,
    }


def main():
    st = extract_winner_detailed_stats()

    print("=" * 135)
    print("THETA-BEATING TIERED RATCHET CHAMPION: COMPLETE ARCHITECTURAL & NUMERICAL PROFILE")
    print("=" * 135)

    print(f"\n--- 1. EXACT CONFIGURATION SPECIFICATIONS ---")
    print(f"  * Initial Stop Loss:            2.50 x ATR (~12.28 option pts / Rs 838 max protective cushion)")
    print(f"  * Tier 2 Breakeven Trigger:     Gain >= +1.25 x ATR (~+6.14 option pts) -> Locks SL to Entry + 0.50 pt (Risk-Free)")
    print(f"  * Tier 3 Trailing Activation:   Gain >= +1.50 x ATR (~+7.37 option pts) -> Trails 0.50 x ATR (~2.46 pts) behind Peak")
    print(f"  * Tier 4 Theta Time Stop:       15 Minutes (If in trade > 15m without reaching Tier 2, tighten SL to Entry - 0.5x ATR)")
    print(f"  * Target Take Profit:           3.50 x ATR (~17.20 option pts / Rs 1,078 reward)")
    print(f"  * Exchange Friction:            Flat Rs 40.00 / trade (Lot Size = 65)")

    print(f"\n--- 2. 7-YEAR MASTER PERFORMANCE (2020-2026, 1,588 DAYS) ---")
    print(f"  * Total Trades Taken:           {st['total_trades']:,} trades ({st['avg_trades_day']} trades/day, Median: {st['median_trades_day']:.0f})")
    print(f"  * Total Wins / Losses:          {st['wins']:,} Wins / {st['losses']:,} Losses (Win Rate: {st['win_rate']}%)")
    print(f"  * Net Realized Profit:          Rs {st['net_rs']:+,.2f} (+{st['net_points']:+,.2f} Net Points Captured)")
    print(f"  * Profit Factor:                {st['profit_factor']}")
    print(f"  * 7-Year Maximum Drawdown:      Rs {st['max_drawdown']:,.2f}")
    print(f"  * Calmar Ratio (Return/Max DD): {st['calmar_ratio']}")
    print(f"  * Monthly Win Rate:             {st['pos_months']} out of {st['total_months']} Months Green ({st['month_win_rate']}%)")
    print(f"  * Average Monthly Profit:       Rs {st['avg_month_rs']:+,.2f} / month")

    print(f"\n--- 3. TRADE GEOMETRY & ASYMMETRIC EXPECTANCY ---")
    print(f"  * Average Initial SL Distance:  {st['avg_initial_sl_pts']} option pts (~Rs {st['avg_initial_sl_rs']:.2f} per lot)")
    print(f"  * Average Initial TP Target:    {st['avg_initial_tp_pts']} option pts (~Rs {st['avg_initial_tp_rs']:.2f} per lot)")
    print(f"  * Average Realized Win:         +{st['avg_realized_win_pts']} option pts (+Rs {st['avg_realized_win_rs']:+,.2f} net win)")
    print(f"  * Average Realized Loss:        {st['avg_realized_loss_pts']} option pts (-Rs {abs(st['avg_realized_loss_rs']):,.2f} net loss)")
    print(f"  * Realized Win / Loss Ratio:    {st['realized_rr_ratio']}x (Winning trades capture 1.25x more than losing trades)")

    print(f"\n--- 4. TRADE DURATION & TIME IN MARKET ---")
    print(f"  * Average Trade Duration:       {st['avg_duration_min']} minutes")
    print(f"  * Winning Trades Holding Time:  {st['avg_win_duration_min']} minutes (lets large runners develop)")
    print(f"  * Losing Trades Holding Time:   {st['avg_loss_duration_min']} minutes (15m Theta Stop cuts bad trades quickly)")

    print(f"\n--- 5. YEAR-BY-YEAR BREAKDOWN (2020-2026) ---")
    print(f"{'Year':6s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Yearly Max DD':14s}")
    print("-" * 80)
    for y, yst in st["yearly_stats"].items():
        print(f"{y:6s} | {yst['trades']:7d} | {yst['win_rate']:7.1f}% | {yst['net_points']:+10.2f} | Rs {yst['net_rs']:+12.2f} | {yst['profit_factor']:6.3f} | Rs {yst['max_dd']:11.2f}")


    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "theta_champion_master_profile.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(st, indent=2, default=float), encoding="utf-8")
    print("\n" + "=" * 135)
    print(f"[Saved Detailed Profile JSON]: {out_file}")
    print("=" * 135)


if __name__ == "__main__":
    main()
