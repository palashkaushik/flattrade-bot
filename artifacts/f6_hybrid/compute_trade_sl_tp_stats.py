"""Compute Exact Trade Frequency, Average SL, Average TP, and Holding Duration Statistics.

Calculates:
  - Average Trades Per Day (mean, median, min, max, daily distribution)
  - Average Initial SL Distance (pts and Rs)
  - Average Actual Realized Loss (pts and Rs)
  - Average Initial TP Distance (pts and Rs)
  - Average Actual Realized Win (pts and Rs)
  - Average Trade Holding Duration (in minutes)
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
def compute_detailed_trade_metrics(
    sl_mult: float,
    tp_mult: float,
    trail_trigger_atr: float | None,
    trail_dist_atr: float,
    gamma: float = 0.0,
):
    coords = torch.nonzero(entries_mask, as_tuple=False)
    d_idx = coords[:, 0]
    b_idx = coords[:, 1]
    ep = d_c[d_idx, b_idx]
    base_atr = d_atr[d_idx, b_idx].clamp(min=5.0, max=25.0)
    trade_vix = d_vix[d_idx, b_idx].clamp(min=8.0, max=80.0)

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

    vix_scale = torch.pow(trade_vix / 15.0, gamma).clamp(min=0.6, max=2.0)
    eff_atr = base_atr * vix_scale
    sl_dist = (sl_mult * eff_atr).clamp(min=5.0, max=30.0)
    tp_dist = (tp_mult * eff_atr).clamp(min=8.0, max=60.0)

    init_sl = ep_exp - sl_dist.unsqueeze(1)
    tp_barrier = ep_exp + tp_dist.unsqueeze(1)

    if trail_trigger_atr is not None:
        trig_gain = (trail_trigger_atr * eff_atr).unsqueeze(1)
        trail_d = (trail_dist_atr * eff_atr).unsqueeze(1)
        gains = running_peaks - ep_exp
        is_trailing = gains >= trig_gain
        trailing_sl_level = running_peaks - trail_d
        dyn_sl_barrier = torch.where(is_trailing, torch.maximum(init_sl, trailing_sl_level), init_sl)
    else:
        dyn_sl_barrier = init_sl.expand(-1, max_future)

    hit_sl = (fut_l_m <= dyn_sl_barrier)
    hit_tp = (fut_h_m >= tp_barrier)

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    # Duration in bars (minutes)
    exit_bar_offset = torch.where(sl_exits, sl_first, torch.where(tp_exits, tp_first, max_future - 1))
    duration_min = exit_bar_offset + 1

    exit_sl_px = dyn_sl_barrier.gather(1, sl_first.clamp(max=max_future-1).unsqueeze(1)).squeeze(1)
    exit_tp_px = tp_barrier.squeeze(1)
    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px))

    pts = (exit_px - ep) * 0.50
    rs_net = pts * LOT_SIZE - FEE

    # Convert to CPU Numpy arrays
    pts_cpu = pts.cpu().numpy()
    rs_cpu = rs_net.cpu().numpy()
    d_idx_cpu = d_idx.cpu().numpy()
    b_idx_cpu = b_idx.cpu().numpy()
    sl_dist_cpu = (sl_dist * 0.50).cpu().numpy()  # Option delta (0.50)
    tp_dist_cpu = (tp_dist * 0.50).cpu().numpy()  # Option delta (0.50)
    dur_cpu = duration_min.cpu().numpy()
    sl_ex_cpu = sl_exits.cpu().numpy()
    tp_ex_cpu = tp_exits.cpu().numpy()

    # Daily Trade Frequency Distribution
    trades_per_day = pd.Series(d_idx_cpu).value_counts().reindex(range(N_DAYS), fill_value=0)

    # Winners & Losers
    win_mask = rs_cpu > 0
    loss_mask = rs_cpu <= 0

    wins_pts = pts_cpu[win_mask]
    wins_rs = rs_cpu[win_mask]
    wins_dur = dur_cpu[win_mask]

    loss_pts = pts_cpu[loss_mask]
    loss_rs = rs_cpu[loss_mask]
    loss_dur = dur_cpu[loss_mask]

    return {
        "total_trades": len(pts_cpu),
        "total_days": N_DAYS,
        "avg_trades_per_day": round(float(trades_per_day.mean()), 2),
        "median_trades_per_day": float(trades_per_day.median()),
        "max_trades_in_a_day": int(trades_per_day.max()),
        "min_trades_in_a_day": int(trades_per_day.min()),
        "zero_trade_days": int((trades_per_day == 0).sum()),
        "active_days_avg_trades": round(float(trades_per_day[trades_per_day > 0].mean()), 2),

        # Initial SL & TP Target Dimensions (in Option Points & Rs)
        "avg_initial_sl_pts": round(float(np.mean(sl_dist_cpu)), 2),
        "avg_initial_sl_rs": round(float(np.mean(sl_dist_cpu) * LOT_SIZE + FEE), 2),
        "avg_initial_tp_pts": round(float(np.mean(tp_dist_cpu)), 2),
        "avg_initial_tp_rs": round(float(np.mean(tp_dist_cpu) * LOT_SIZE - FEE), 2),

        # Realized Win / Loss Exits (including trailing stops & BE)
        "avg_realized_win_pts": round(float(np.mean(wins_pts)), 2) if len(wins_pts) > 0 else 0.0,
        "avg_realized_win_rs": round(float(np.mean(wins_rs)), 2) if len(wins_rs) > 0 else 0.0,
        "avg_realized_loss_pts": round(float(np.mean(loss_pts)), 2) if len(loss_pts) > 0 else 0.0,
        "avg_realized_loss_rs": round(float(np.mean(loss_rs)), 2) if len(loss_rs) > 0 else 0.0,

        # Holding Duration
        "avg_trade_duration_min": round(float(np.mean(dur_cpu)), 1),
        "avg_win_duration_min": round(float(np.mean(wins_dur)), 1) if len(wins_dur) > 0 else 0.0,
        "avg_loss_duration_min": round(float(np.mean(loss_dur)), 1) if len(loss_dur) > 0 else 0.0,

        # Exit Reason Distribution
        "tp_target_hits": int(tp_ex_cpu.sum()),
        "sl_or_trail_hits": int(sl_ex_cpu.sum()),
        "eod_exits": int((~sl_ex_cpu & ~tp_ex_cpu).sum()),
        "win_rate": round(float(win_mask.sum() / len(pts_cpu) * 100), 2),
        "net_rs": round(float(np.sum(rs_cpu)), 2),
        "net_pts": round(float(np.sum(pts_cpu)), 2),
    }


def main():
    strategies = [
        {
            "name": "Champion Low-Drawdown (Trail=0.75x/0.50x, SL=1.50x, TP=3.00x)",
            "sl_mult": 1.5, "tp_mult": 3.0, "trail_trig": 0.75, "trail_dist": 0.5, "gamma": 0.0,
        },
        {
            "name": "Champion Max-Profit (Trail=1.25x/0.50x, SL=1.50x, TP=3.50x)",
            "sl_mult": 1.5, "tp_mult": 3.5, "trail_trig": 1.25, "trail_dist": 0.5, "gamma": 0.0,
        },
        {
            "name": "Dynamic VIX Champion (Power gamma=1.0, SL=3.00x, TP=6.00x)",
            "sl_mult": 3.0, "tp_mult": 6.0, "trail_trig": None, "trail_dist": 0.5, "gamma": 1.0,
        },
    ]

    all_stats = {}
    print("=" * 115)
    print("EXACT TRADE FREQUENCY, AVERAGE SL, AVERAGE TP, AND HOLDING TIME STATISTICS (2020–2026)")
    print("=" * 115)

    for strat in strategies:
        st = compute_detailed_trade_metrics(strat["sl_mult"], strat["tp_mult"], strat["trail_trig"], strat["trail_dist"], strat["gamma"])
        all_stats[strat["name"]] = st

        print(f"\n{'#'*30} {strat['name'].upper()} {'#'*30}")
        print(f"\n--- TRADE FREQUENCY & VOLUME ---")
        print(f"  * Total Trades (7 Years): {st['total_trades']:,} trades across {st['total_days']:,} trading days")
        print(f"  * Average Trades Per Day: {st['avg_trades_per_day']} trades / day (Median: {st['median_trades_per_day']:.0f} trades / day)")
        print(f"  * Min / Max Daily Range:  {st['min_trades_in_a_day']} to {st['max_trades_in_a_day']} trades in a single day")
        print(f"  * Zero-Trade Days:        {st['zero_trade_days']} days (Active Trading Days: {st['total_days'] - st['zero_trade_days']})")

        print(f"\n--- STOP LOSS & TAKE PROFIT GEOMETRY ---")
        print(f"  * Initial SL Distance:    {st['avg_initial_sl_pts']} pts  (~Rs {st['avg_initial_sl_rs']:.2f} risk per lot)")
        print(f"  * Initial TP Target:      {st['avg_initial_tp_pts']} pts  (~Rs {st['avg_initial_tp_rs']:.2f} reward per lot)")
        print(f"  * Average Realized Win:   +{st['avg_realized_win_pts']} pts (+Rs {st['avg_realized_win_rs']:+,.2f} per winning trade)")
        print(f"  * Average Realized Loss:  {st['avg_realized_loss_pts']} pts (-Rs {abs(st['avg_realized_loss_rs']):,.2f} per losing trade)")
        print(f"  * Realized Win/Loss Ratio: {abs(st['avg_realized_win_pts'] / st['avg_realized_loss_pts']):.2f}x")

        print(f"\n--- TRADE HOLDING DURATION ---")
        print(f"  * Average Trade Duration: {st['avg_trade_duration_min']} minutes")
        print(f"  * Winning Trades Average: {st['avg_win_duration_min']} minutes")
        print(f"  * Losing Trades Average:  {st['avg_loss_duration_min']} minutes")

        print(f"\n--- EXIT REASON BREAKDOWN ---")
        print(f"  * Full TP Target Hits:    {st['tp_target_hits']:,} trades ({st['tp_target_hits']/st['total_trades']*100:.1f}%)")
        print(f"  * Trailing SL / SL Hits:  {st['sl_or_trail_hits']:,} trades ({st['sl_or_trail_hits']/st['total_trades']*100:.1f}%)")
        print(f"  * End of Day (EOD) Exits: {st['eod_exits']:,} trades ({st['eod_exits']/st['total_trades']*100:.1f}%)")

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "trade_sl_tp_detailed_stats.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_stats, indent=2, default=float), encoding="utf-8")
    print("\n" + "=" * 115)
    print(f"[Saved Detailed Stats JSON]: {out_file}")
    print("=" * 115)


if __name__ == "__main__":
    main()
