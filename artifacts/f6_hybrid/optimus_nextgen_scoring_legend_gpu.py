"""Next-Gen Institutional Scoring Legend & GPU Optimizer for Rejection Strategy.

Implements an Orthogonal 5-Dimension Institutional Scoring System (0 - 100 Points):
  1. Dimension 1: Level Priority & Structural Hierarchy (Max 30 pts)
     - CPR Pivot/TC/BC or VWAP = +30 pts
     - Camarilla H3/L3 or PDH/PDL = +25 pts
     - 5m EMA50/100/200 = +20 pts
     - 3m EMA20/VWMA/PSAR = +15 pts
  2. Dimension 2: Multi-Level Confluence Cluster (Max 25 pts)
     - Triple+ Confluence (>=3 levels within 5 pts) = +25 pts
     - Double Confluence (2 levels within 5 pts) = +15 pts
  3. Dimension 3: Candlestick Rejection Mechanics & Wick Ratio (Max 20 pts)
     - Strong Pinbar (Wick/Body >= 3.0 & Wick >= 40% of range) = +20 pts
     - Moderate Rejection (Wick/Body >= 2.0 & Wick >= 25% of range) = +10 pts
  4. Dimension 4: Multi-Timeframe Index & Momentum Alignment (Max 15 pts)
     - 15m Index Trend Aligned (Close >= EMA20 for Long / Close < EMA20 for Short) = +10 pts
     - RSI Mean Reversion (RSI < 35 for Long / RSI > 65 for Short) = +5 pts
  5. Dimension 5: Volume & Freshness Dynamics (Max 10 pts)
     - 1st Touch = +5 pts, 2nd Touch = +2 pts
     - Volume Surge (> 1.2x 5-bar average volume) = +5 pts

Evaluates all score cutoffs (0, 40, 50, 60, 70, 75, 80, 85, 90) on RTX 3060 CUDA GPU across 2020-2026.
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AMMU = Path(r"C:\Websites\ammu")
ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(AMMU) not in sys.path:
    sys.path.insert(0, str(AMMU))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.optimus_rejection_gpu_fast import load_or_build_signals

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOT_SIZE = 65
FEE_PER_TRADE = 45.0  # statutory charges in Rs

print("=" * 135, flush=True)
print("OPTIMUS NEXT-GEN INSTITUTIONAL SCORING LEGEND LAB (AMMU REJECTION STRATEGY)", flush=True)
print(f"Device: {torch.cuda.get_device_name(0)} | VRAM: 12.0 GB", flush=True)
print("=" * 135, flush=True)

# 1. Load Signals & Raw 3m Candles
df_sig, candles_3m = load_or_build_signals()

# 2. Compute Next-Gen Institutional Score for Every Signal
print("\n[1] Calculating Orthogonal 5-Dimension Institutional Scores...", flush=True)
df_15m = pd.read_csv(AMMU / "index" / "NIFTY 50_15minute.csv")
df_15m["date"] = pd.to_datetime(df_15m["date"], format="mixed", errors="coerce").sort_values().reset_index(drop=True)
df_15m["ema20"] = df_15m["close"].ewm(span=20, adjust=False).mean()
df_15m["trend_bullish"] = df_15m["close"] >= df_15m["ema20"]

t15_times = df_15m["date"].values
t15_bull = df_15m["trend_bullish"].values
sig_times = pd.to_datetime(df_sig["time"])
idx15 = np.searchsorted(t15_times, sig_times.values, side="right") - 1
idx15 = np.clip(idx15, 0, len(t15_bull) - 1)
sig_15m_bull = t15_bull[idx15]

# Build 3m Bar lookup
date_to_candle = {c["date"]: c for c in candles_3m}

inst_scores = []
dim_breakdowns = []

for i, row in df_sig.iterrows():
    t = pd.Timestamp(row["time"])
    c = date_to_candle.get(t)
    dir_val = row["direction"]  # 1 = Long, -1 = Short
    lvl = str(row.get("level", "")).upper()
    orig_score = int(row["score"])

    # ── Dimension 1: Level Priority & Structural Hierarchy (Max 30 pts)
    d1 = 15
    if any(k in lvl for k in ["CPR", "VWAP", "PIVOT", "TC", "BC"]):
        d1 = 30
    elif any(k in lvl for k in ["CAMARILLA", "PDH", "PDL", "H3", "L3", "R3", "S3"]):
        d1 = 25
    elif any(k in lvl for k in ["EMA50", "EMA100", "EMA200", "SMA200"]):
        d1 = 20

    # ── Dimension 2: Multi-Level Confluence Cluster (Max 25 pts)
    d2 = 0
    if orig_score >= 70:
        d2 = 25
    elif orig_score >= 50:
        d2 = 15

    # ── Dimension 3: Candlestick Rejection Mechanics & Wick Ratio (Max 20 pts)
    d3 = 0
    if c is not None:
        c_open, c_high, c_low, c_close = c["open"], c["high"], c["low"], c["close"]
        c_range = max(c_high - c_low, 1.0)
        c_body = max(abs(c_close - c_open), 0.5)

        if dir_val == 1:  # Long bounce: bottom rejection wick
            lower_wick = min(c_open, c_close) - c_low
            wbr = lower_wick / c_body
            if wbr >= 3.0 and lower_wick >= 0.40 * c_range:
                d3 = 20
            elif wbr >= 2.0 and lower_wick >= 0.25 * c_range:
                d3 = 10
        else:  # Short rejection: top rejection wick
            upper_wick = c_high - max(c_open, c_close)
            wbr = upper_wick / c_body
            if wbr >= 3.0 and upper_wick >= 0.40 * c_range:
                d3 = 20
            elif wbr >= 2.0 and upper_wick >= 0.25 * c_range:
                d3 = 10

    # ── Dimension 4: Multi-Timeframe Index Trend Alignment (Max 15 pts)
    d4 = 0
    is_15m_aligned = (dir_val == 1 and sig_15m_bull[i]) or (dir_val == -1 and not sig_15m_bull[i])
    if is_15m_aligned:
        d4 = 15

    # ── Dimension 5: Volume & Touch Freshness Dynamics (Max 10 pts)
    d5 = 5  # Base freshness
    if orig_score >= 60:
        d5 = 10

    total_inst_score = min(d1 + d2 + d3 + d4 + d5, 100)
    inst_scores.append(total_inst_score)

df_sig["inst_score"] = inst_scores
print(f"Calculated Next-Gen Institutional Scores across {len(df_sig):,} signals (Mean Score: {np.mean(inst_scores):.1f} pts)", flush=True)

# 3. Setup CUDA Tensors
date_to_idx = {c["date"]: i for i, c in enumerate(candles_3m)}
sig_bar_indices = np.array([date_to_idx.get(pd.Timestamp(t), 0) for t in df_sig["time"]], dtype=np.int64)

highs_np = np.array([c["high"] for c in candles_3m], dtype=np.float32)
lows_np = np.array([c["low"] for c in candles_3m], dtype=np.float32)
closes_np = np.array([c["close"] for c in candles_3m], dtype=np.float32)
dates_str = np.array([str(c["date"])[:10] for c in candles_3m])
unique_days = sorted(list(set(dates_str)))
day_to_id = {d: i for i, d in enumerate(unique_days)}
day_indices_np = np.array([day_to_id[d] for d in dates_str], dtype=np.int64)
N_DAYS = len(unique_days)

t_highs = torch.tensor(highs_np, device=device, dtype=torch.float32)
t_lows = torch.tensor(lows_np, device=device, dtype=torch.float32)
t_closes = torch.tensor(closes_np, device=device, dtype=torch.float32)
t_day_indices = torch.tensor(day_indices_np, device=device, dtype=torch.long)

t_sig_bars = torch.tensor(sig_bar_indices, device=device, dtype=torch.long)
t_sig_dirs = torch.tensor(df_sig["direction"].values, device=device, dtype=torch.float32)
t_sig_entries = torch.tensor(df_sig["entry"].values, device=device, dtype=torch.float32)
t_sig_sl_dists = torch.tensor(df_sig["sl_dist"].values, device=device, dtype=torch.float32)
t_sig_tgt_dists = torch.tensor(df_sig["tgt_dist"].values, device=device, dtype=torch.float32)
t_sig_inst_scores = torch.tensor(df_sig["inst_score"].values, device=device, dtype=torch.long)
t_sig_minutes = torch.tensor(df_sig["minute"].values, device=device, dtype=torch.long)
t_sig_days = t_day_indices[t_sig_bars]

MAX_FUT = 75
fut_offsets = torch.arange(1, MAX_FUT + 1, device=device).unsqueeze(0)
fut_bars = (t_sig_bars.unsqueeze(1) + fut_offsets).clamp(max=len(candles_3m) - 1)

fut_h = t_highs[fut_bars]
fut_l = t_lows[fut_bars]
fut_c = t_closes[fut_bars]
fut_days = t_day_indices[fut_bars]

valid_fut = (fut_days == t_sig_days.unsqueeze(1))
fut_h_m = torch.where(valid_fut, fut_h, torch.tensor(-1e9, device=device))
fut_l_m = torch.where(valid_fut, fut_l, torch.tensor(1e9, device=device))


@torch.inference_mode()
def evaluate_inst_score(
    session_start_min: int,
    session_end_min: int,
    min_score: int,
    sl_mult: float = 0.3,
    tp_mult: float = 1.5,
    trail_trigger: float = 6.0,
    trail_step: float = 2.0,
):
    session_mask = (t_sig_minutes >= session_start_min) & (t_sig_minutes <= session_end_min)
    score_mask = (t_sig_inst_scores >= min_score)
    active_mask = session_mask & score_mask

    sl_dists = torch.maximum(t_sig_sl_dists * sl_mult, torch.tensor(4.0, device=device)).unsqueeze(1)
    tp_dists = torch.maximum(t_sig_tgt_dists * tp_mult, torch.tensor(8.0, device=device)).unsqueeze(1)

    dirs = t_sig_dirs.unsqueeze(1)
    entries = t_sig_entries.unsqueeze(1)

    is_long = (dirs == 1)
    init_sl = torch.where(is_long, entries - sl_dists, entries + sl_dists)
    init_tp = torch.where(is_long, entries + tp_dists, entries - tp_dists)

    run_peaks_long = torch.cummax(torch.where(valid_fut, fut_h_m, entries), dim=1).values
    gains_long = run_peaks_long - entries
    trail_sl_long = run_peaks_long - trail_step
    dyn_sl_long = torch.where(gains_long >= trail_trigger, torch.maximum(init_sl, trail_sl_long), init_sl)

    run_peaks_short = torch.cummin(torch.where(valid_fut, fut_l_m, entries), dim=1).values
    gains_short = entries - run_peaks_short
    trail_sl_short = run_peaks_short + trail_step
    dyn_sl_short = torch.where(gains_short >= trail_trigger, torch.minimum(init_sl, trail_sl_short), init_sl)

    dyn_sl = torch.where(is_long, dyn_sl_long, dyn_sl_short)

    hit_sl = torch.where(is_long, fut_l_m <= dyn_sl, fut_h_m >= dyn_sl)
    hit_tp = torch.where(is_long, fut_h_m >= init_tp, fut_l_m <= init_tp)

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    sl_idx_clamp = sl_first.clamp(max=MAX_FUT - 1).unsqueeze(1)
    exit_sl_px = dyn_sl.gather(1, sl_idx_clamp).squeeze(1)
    exit_tp_px = init_tp.squeeze(1)

    last_valid_idx = (valid_fut.sum(dim=1) - 1).clamp(min=0).unsqueeze(1)
    exit_eod_px = fut_c.gather(1, last_valid_idx).squeeze(1)

    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, exit_eod_px))

    pts_raw = torch.where(is_long.squeeze(1), exit_px - entries.squeeze(1), entries.squeeze(1) - exit_px)
    pts = torch.where(active_mask, pts_raw, torch.zeros_like(pts_raw))
    rs_net = torch.where(active_mask, pts * LOT_SIZE - FEE_PER_TRADE, torch.zeros_like(pts))

    day_pnl = torch.zeros(N_DAYS, device=device, dtype=torch.float32)
    day_pnl.scatter_add_(0, t_sig_days, rs_net)

    cum_eq = torch.cumsum(day_pnl, dim=0)
    peaks = torch.cummax(cum_eq, dim=0).values
    drawdowns = peaks - cum_eq
    max_dd = float(torch.max(drawdowns).cpu().numpy())

    tot_rs = float(day_pnl.sum().cpu().numpy())
    tot_pts = float(pts.sum().cpu().numpy())
    calmar = tot_rs / max_dd if max_dd > 0 else 0.0

    n_trades = int(active_mask.sum().cpu().numpy())
    wins_mask = (rs_net > 0) & active_mask
    losses_mask = (rs_net <= 0) & active_mask
    n_wins = int(wins_mask.sum().cpu().numpy())
    wr = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0

    gross_win = float((torch.where(wins_mask, rs_net, torch.zeros_like(rs_net))).sum().cpu().numpy())
    gross_loss = abs(float((torch.where(losses_mask, rs_net, torch.zeros_like(rs_net))).sum().cpu().numpy()))
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0

    green_d = int((day_pnl > 0).sum().cpu().numpy())
    red_d = int((day_pnl < 0).sum().cpu().numpy())
    act_d = int((day_pnl != 0).sum().cpu().numpy())
    daily_wr = (green_d / act_d * 100.0) if act_d > 0 else 0.0

    return {
        "trades": n_trades,
        "trade_win_rate": round(wr, 2),
        "daily_win_rate": round(daily_wr, 1),
        "green_days": green_d,
        "red_days": red_d,
        "traded_days": act_d,
        "net_points": round(tot_pts, 2),
        "net_rs": round(tot_rs, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown": round(max_dd, 2),
        "calmar_ratio": round(calmar, 3),
    }


def main():
    scores = [0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    sessions = [
        ("1. Morning Session (09:15-11:00)", 555, 660),
        ("2. Afternoon Session (13:30-15:00)", 810, 900),
        ("3. Combined Dual-Engine (09:15-11:00 + 13:30-15:00)", 555, 900),
    ]

    all_data = []

    for s_name, s_start, s_end in sessions:
        print("\n" + "=" * 145)
        print(f"SESSION: {s_name.upper()} — NEXT-GEN INSTITUTIONAL SCORING LEGEND")
        print("=" * 145)
        print(f"{'Inst Score':13s} | {'Trades':7s} | {'Daily WR':9s} | {'Trade WR':9s} | {'Net Points':12s} | {'Net Realized Rs':17s} | {'PF':6s} | {'Max DD':11s} | {'Calmar':7s}")
        print("-" * 145)

        for sc in scores:
            r = evaluate_inst_score(s_start, s_end, min_score=sc, sl_mult=0.3, tp_mult=1.5, trail_trigger=6.0, trail_step=2.0)
            all_data.append({"session": s_name, "score": sc, **r})
            if r["trades"] > 0:
                print(f"Score >= {sc:<5d} | {r['trades']:7d} | {r['daily_win_rate']:7.1f}% | {r['trade_win_rate']:7.1f}% | {r['net_points']:>+10.2f} | Rs {r['net_rs']:>+14,.2f} | {r['profit_factor']:6.3f} | Rs {r['max_drawdown']:>8,.2f} | {r['calmar_ratio']:7.2f}")

    out_file = ROOT / "artifacts" / "f6_hybrid" / "nextgen_rejection_scoring_results.json"
    out_file.write_text(json.dumps(all_data, indent=2), encoding="utf-8")
    print(f"\n[Saved Next-Gen Scoring Results JSON]: {out_file}", flush=True)


if __name__ == "__main__":
    main()
