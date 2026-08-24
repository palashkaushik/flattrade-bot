"""Optimus Pure CUDA Rejection Score & Parameter Optimizer (Ammu Strategy).

Fast multi-process signal generation + 100% Pure CUDA 3D Tensor parameter sweep on RTX 3060.
Evaluates:
  1. Morning Window: 09:15 - 11:00 IST
  2. Afternoon Window: 13:30 - 15:00 IST
  3. Combined Dual-Engine: 09:15 - 11:00 + 13:30 - 15:00 IST
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import itertools
import json
import sys
import time
from bisect import bisect_right
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

AMMU = Path(r"C:\Websites\ammu")
ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(AMMU) not in sys.path:
    sys.path.insert(0, str(AMMU))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kbot.indicators.engine import calculate_rsi
from kbot.strategies.rejection_scalping import (
    Direction,
    RejectionScalping,
)

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOT_SIZE = 65
FEE_PER_TRADE = 45.0  # statutory charges in Rs

print("=" * 135, flush=True)
print("OPTIMUS PURE CUDA REJECTION SCORE & PARAMETER OPTIMIZER", flush=True)
print(f"Device: {torch.cuda.get_device_name(0)} | VRAM: 12.0 GB", flush=True)
print("=" * 135, flush=True)

CACHE_PATH = ROOT / "artifacts" / "f6_hybrid" / "rejection_signals_cache.parquet"


def _process_day_chunk(payload):
    day_candles_3m, candles_5m, s5_times = payload
    strat = RejectionScalping()
    signals = []
    indicator_cache = {}

    rsi_values = calculate_rsi([c["close"] for c in day_candles_3m])
    rsi_cache = {
        id(day_candles_3m[i]): (rsi_values[i], rsi_values[i - 1])
        for i in range(1, len(day_candles_3m))
    }

    s3_history = []
    s5_index = 0
    s5_history = []

    for i, bar in enumerate(day_candles_3m):
        bt = bar["date"]
        s3_history.append(bar)
        while s5_index < len(candles_5m) and s5_times[s5_index] <= bt:
            s5_history.append(candles_5m[s5_index])
            s5_index += 1

        if len(s3_history) < 60 or len(s5_history) < 20:
            continue

        sig = strat.generate_signal(s3_history, s5_history, indicator_cache, rsi_cache)
        if sig:
            signals.append({
                "bar_idx": i,
                "date": str(bt)[:10],
                "time": str(bt),
                "minute": bt.hour * 60 + bt.minute,
                "direction": 1 if sig.direction == Direction.LONG else -1,
                "entry": float(sig.entry_price),
                "sl_dist": float(abs(sig.entry_price - sig.stop_loss)),
                "tgt_dist": float(abs(sig.target - sig.entry_price)),
                "score": int(sig.score),
                "level": sig.level.label,
            })

    return signals


def load_or_build_signals():
    if CACHE_PATH.exists():
        print(f"[1] Loading Cached Causal Signals from: {CACHE_PATH}...", flush=True)
        df_sig = pd.read_parquet(CACHE_PATH)
        df_1m = pd.read_csv(AMMU / "index" / "NIFTY 50_minute.csv")
        df_1m["date"] = pd.to_datetime(df_1m["date"], format="mixed", errors="coerce")
        df_1m = df_1m.sort_values("date").reset_index(drop=True)
        df_1m["bar_3m"] = df_1m["date"].dt.floor("3min")
        agg = df_1m.groupby("bar_3m").agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
        ).reset_index().rename(columns={"bar_3m": "date"})
        candles_3m = agg[agg["date"] >= pd.Timestamp("2020-01-01")].to_dict("records")
        return df_sig, candles_3m

    print("[1] Loading 1m & 5m Raw Data...", flush=True)
    df_1m = pd.read_csv(AMMU / "index" / "NIFTY 50_minute.csv")
    df_1m["date"] = pd.to_datetime(df_1m["date"], format="mixed", errors="coerce")
    df_1m = df_1m.sort_values("date").reset_index(drop=True)

    df_5m = pd.read_csv(AMMU / "index" / "NIFTY 50_5minute.csv")
    df_5m["date"] = pd.to_datetime(df_5m["date"], format="mixed", errors="coerce")
    df_5m = df_5m.sort_values("date").reset_index(drop=True)

    df_1m["bar_3m"] = df_1m["date"].dt.floor("3min")
    agg = df_1m.groupby("bar_3m").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    ).reset_index().rename(columns={"bar_3m": "date"})

    candles_3m = agg[agg["date"] >= pd.Timestamp("2020-01-01")].to_dict("records")
    candles_5m = df_5m[df_5m["date"] >= pd.Timestamp("2019-11-01")].rename(columns={"date": "timestamp"}).to_dict("records")
    s5_times = [c["timestamp"] for c in candles_5m]

    # Split by year/month chunks for parallel processing
    print(f"Generating signals across {len(candles_3m):,} 3m bars using {cpu_count()} CPU cores...", flush=True)
    t0 = time.time()

    # Group into chunks of 15 days with 30-day lookback history
    dates = [c["date"] for c in candles_3m]
    days = sorted(list(set(str(d)[:10] for d in dates)))
    chunks = []
    chunk_size = 15

    for i in range(0, len(days), chunk_size):
        chunk_days = set(days[i: i + chunk_size])
        earliest_day = days[max(0, i - 30)]
        sub_3m = [c for c in candles_3m if str(c["date"])[:10] >= earliest_day and str(c["date"])[:10] in chunk_days or (str(c["date"])[:10] >= earliest_day and str(c["date"])[:10] < days[i])]
        if sub_3m:
            chunks.append((sub_3m, candles_5m, s5_times))

    with Pool(processes=max(2, cpu_count() - 1)) as p:
        results = p.map(_process_day_chunk, chunks)

    all_sigs = []
    seen = set()
    for res in results:
        for s in res:
            k = (s["time"], s["direction"], s["level"])
            if k not in seen:
                seen.add(k)
                all_sigs.append(s)

    all_sigs.sort(key=lambda x: x["time"])
    df_sig = pd.DataFrame(all_sigs)
    df_sig.to_parquet(CACHE_PATH, index=False)
    print(f"Generated and Cached {len(df_sig):,} Rejection Signals in {time.time()-t0:.2f}s", flush=True)
    return df_sig, candles_3m


@torch.inference_mode()
def run_cuda_rejection_sweep(
    df_sig: pd.DataFrame,
    candles_3m: list[dict],
    session_name: str,
    min_time: int,
    max_time: int,
    param_grid: list[dict],
    batch_size: int = 100,
):
    # Filter signals to session time
    sub_sig = df_sig[(df_sig["minute"] >= min_time) & (df_sig["minute"] <= max_time)].reset_index(drop=True)
    if sub_sig.empty:
        return []

    # Build Bar Tensors
    date_to_idx = {c["date"]: i for i, c in enumerate(candles_3m)}
    sig_bar_indices = [date_to_idx.get(pd.Timestamp(t), -1) for t in sub_sig["time"]]
    valid_sig_mask = np.array([idx != -1 for idx in sig_bar_indices], dtype=bool)
    sub_sig = sub_sig[valid_sig_mask].reset_index(drop=True)
    sig_bar_indices = np.array([idx for idx in sig_bar_indices if idx != -1], dtype=np.int64)

    highs_np = np.array([c["high"] for c in candles_3m], dtype=np.float32)
    lows_np = np.array([c["low"] for c in candles_3m], dtype=np.float32)
    closes_np = np.array([c["close"] for c in candles_3m], dtype=np.float32)
    dates_str = np.array([str(c["date"])[:10] for c in candles_3m])
    unique_days = sorted(list(set(dates_str)))
    day_to_id = {d: i for i, d in enumerate(unique_days)}
    day_indices_np = np.array([day_to_id[d] for d in dates_str], dtype=np.int64)
    N_DAYS = len(unique_days)

    # CUDA Tensors
    t_highs = torch.tensor(highs_np, device=device, dtype=torch.float32)
    t_lows = torch.tensor(lows_np, device=device, dtype=torch.float32)
    t_closes = torch.tensor(closes_np, device=device, dtype=torch.float32)
    t_day_indices = torch.tensor(day_indices_np, device=device, dtype=torch.long)

    M = len(sub_sig)
    t_sig_bars = torch.tensor(sig_bar_indices, device=device, dtype=torch.long)
    t_sig_dirs = torch.tensor(sub_sig["direction"].values, device=device, dtype=torch.float32)
    t_sig_entries = torch.tensor(sub_sig["entry"].values, device=device, dtype=torch.float32)
    t_sig_sl_dists = torch.tensor(sub_sig["sl_dist"].values, device=device, dtype=torch.float32)
    t_sig_tgt_dists = torch.tensor(sub_sig["tgt_dist"].values, device=device, dtype=torch.float32)
    t_sig_scores = torch.tensor(sub_sig["score"].values, device=device, dtype=torch.long)
    t_sig_days = t_day_indices[t_sig_bars]

    # Max future bars (up to end of day ~75 3m bars)
    MAX_FUT = 75
    fut_offsets = torch.arange(1, MAX_FUT + 1, device=device).unsqueeze(0)
    fut_bars = (t_sig_bars.unsqueeze(1) + fut_offsets).clamp(max=len(candles_3m) - 1)

    fut_h = t_highs[fut_bars]
    fut_l = t_lows[fut_bars]
    fut_c = t_closes[fut_bars]
    fut_days = t_day_indices[fut_bars]

    # Valid mask (same day)
    valid_fut = (fut_days == t_sig_days.unsqueeze(1))
    fut_h_m = torch.where(valid_fut, fut_h, torch.tensor(-1e9, device=device))
    fut_l_m = torch.where(valid_fut, fut_l, torch.tensor(1e9, device=device))

    results = []

    for chunk_start in range(0, len(param_grid), batch_size):
        chunk = param_grid[chunk_start: chunk_start + batch_size]
        B = len(chunk)

        b_min_score = torch.tensor([p["min_score"] for p in chunk], device=device).view(B, 1)
        b_sl_mult = torch.tensor([p["sl_mult"] for p in chunk], device=device).view(B, 1, 1)
        b_tp_mult = torch.tensor([p["tp_mult"] for p in chunk], device=device).view(B, 1, 1)
        b_trig = torch.tensor([p["trail_trigger"] for p in chunk], device=device).view(B, 1, 1)
        b_step = torch.tensor([p["trail_step"] for p in chunk], device=device).view(B, 1, 1)

        # Filter signals by min score
        score_pass = (t_sig_scores.unsqueeze(0) >= b_min_score)  # (B, M)

        # Compute SL / TP distances
        sl_dists = torch.maximum(t_sig_sl_dists.unsqueeze(0).unsqueeze(2) * b_sl_mult, torch.tensor(4.0, device=device))
        tp_dists = torch.maximum(t_sig_tgt_dists.unsqueeze(0).unsqueeze(2) * b_tp_mult, torch.tensor(8.0, device=device))

        dirs = t_sig_dirs.unsqueeze(0).unsqueeze(2)  # (1, M, 1)
        entries = t_sig_entries.unsqueeze(0).unsqueeze(2)

        # Long vs Short Targets
        is_long = (dirs == 1)
        init_sl = torch.where(is_long, entries - sl_dists, entries + sl_dists)
        init_tp = torch.where(is_long, entries + tp_dists, entries - tp_dists)

        # Running Trailing Stops
        fut_h_3d = fut_h_m.unsqueeze(0)
        fut_l_3d = fut_l_m.unsqueeze(0)
        fut_c_3d = fut_c.unsqueeze(0)

        # Running gains
        if is_long.any():
            run_peaks_long = torch.cummax(torch.where(valid_fut.unsqueeze(0), fut_h_3d, entries), dim=2).values
            gains_long = run_peaks_long - entries
            trail_sl_long = run_peaks_long - b_step
            dyn_sl_long = torch.where(gains_long >= b_trig, torch.maximum(init_sl, trail_sl_long), init_sl)
        else:
            dyn_sl_long = init_sl

        run_peaks_short = torch.cummin(torch.where(valid_fut.unsqueeze(0), fut_l_3d, entries), dim=2).values
        gains_short = entries - run_peaks_short
        trail_sl_short = run_peaks_short + b_step
        dyn_sl_short = torch.where(gains_short >= b_trig, torch.minimum(init_sl, trail_sl_short), init_sl)

        dyn_sl = torch.where(is_long, dyn_sl_long, dyn_sl_short)

        # Hit barriers
        hit_sl = torch.where(is_long, fut_l_3d <= dyn_sl, fut_h_3d >= dyn_sl)
        hit_tp = torch.where(is_long, fut_h_3d >= init_tp, fut_l_3d <= init_tp)

        BIG = 999999
        sl_any = hit_sl.any(dim=2)
        tp_any = hit_tp.any(dim=2)

        sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=2), BIG)
        tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=2), BIG)

        sl_exits = sl_any & (sl_first <= tp_first)
        tp_exits = tp_any & (~sl_exits)

        sl_idx_clamp = sl_first.clamp(max=MAX_FUT - 1).unsqueeze(2)
        exit_sl_px = dyn_sl.gather(2, sl_idx_clamp).squeeze(2)
        exit_tp_px = init_tp.squeeze(2)

        # EOD exit at last valid bar
        last_valid_idx = (valid_fut.sum(dim=1) - 1).clamp(min=0).unsqueeze(0).unsqueeze(2).expand(B, M, 1)
        exit_eod_px = fut_c_3d.expand(B, M, MAX_FUT).gather(2, last_valid_idx).squeeze(2)

        exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, exit_eod_px))

        # PnL points
        pts_raw = torch.where(is_long.squeeze(2), exit_px - entries.squeeze(2), entries.squeeze(2) - exit_px)
        pts = torch.where(score_pass, pts_raw, torch.zeros_like(pts_raw))
        rs_net = torch.where(score_pass, pts * LOT_SIZE - FEE_PER_TRADE, torch.zeros_like(pts))

        # Daily Vector Aggregation on GPU
        d_idx_exp = t_sig_days.unsqueeze(0).expand(B, M)
        day_pnl_cuda = torch.zeros((B, N_DAYS), device=device, dtype=torch.float32)
        day_pnl_cuda.scatter_add_(1, d_idx_exp, rs_net)

        cum_eq_cuda = torch.cumsum(day_pnl_cuda, dim=1)
        peaks_cuda = torch.cummax(cum_eq_cuda, dim=1).values
        drawdowns_cuda = peaks_cuda - cum_eq_cuda
        max_dds_cuda = torch.max(drawdowns_cuda, dim=1).values

        green_days_cuda = (day_pnl_cuda > 0).sum(dim=1)
        red_days_cuda = (day_pnl_cuda < 0).sum(dim=1)
        active_days_cuda = (day_pnl_cuda != 0).sum(dim=1)
        daily_wrs_cuda = (green_days_cuda.float() / active_days_cuda.clamp(min=1).float()) * 100.0

        tot_rs_cuda = day_pnl_cuda.sum(dim=1)
        tot_pts_cuda = pts.sum(dim=1)
        calmar_cuda = tot_rs_cuda / max_dds_cuda.clamp(min=1.0)

        n_trades_cuda = score_pass.sum(dim=1)
        wins_mask = (rs_net > 0) & score_pass
        losses_mask = (rs_net <= 0) & score_pass
        n_wins_cuda = wins_mask.sum(dim=1)
        trade_wrs_cuda = (n_wins_cuda.float() / n_trades_cuda.clamp(min=1).float()) * 100.0

        win_sums_cuda = (torch.where(wins_mask, rs_net, torch.zeros_like(rs_net))).sum(dim=1)
        loss_sums_cuda = (torch.where(losses_mask, rs_net.abs(), torch.zeros_like(rs_net))).sum(dim=1)
        pfs_cuda = win_sums_cuda / loss_sums_cuda.clamp(min=1.0)

        # CPU transfers
        tot_rs_cpu = tot_rs_cuda.cpu().numpy()
        tot_pts_cpu = tot_pts_cuda.cpu().numpy()
        max_dds_cpu = max_dds_cuda.cpu().numpy()
        calmar_cpu = calmar_cuda.cpu().numpy()
        daily_wrs_cpu = daily_wrs_cuda.cpu().numpy()
        green_days_cpu = green_days_cuda.cpu().numpy()
        red_days_cpu = red_days_cuda.cpu().numpy()
        active_days_cpu = active_days_cuda.cpu().numpy()
        trade_wrs_cpu = trade_wrs_cuda.cpu().numpy()
        n_trades_cpu = n_trades_cuda.cpu().numpy()
        pfs_cpu = pfs_cuda.cpu().numpy()

        for b_i, p in enumerate(chunk):
            results.append({
                "session": session_name,
                **p,
                "trades": int(n_trades_cpu[b_i]),
                "trade_win_rate": round(float(trade_wrs_cpu[b_i]), 2),
                "daily_win_rate": round(float(daily_wrs_cpu[b_i]), 1),
                "green_days": int(green_days_cpu[b_i]),
                "red_days": int(red_days_cpu[b_i]),
                "traded_days": int(active_days_cpu[b_i]),
                "net_points": round(float(tot_pts_cpu[b_i]), 2),
                "net_rs": round(float(tot_rs_cpu[b_i]), 2),
                "profit_factor": round(float(pfs_cpu[b_i]), 3),
                "max_drawdown": round(float(max_dds_cpu[b_i]), 2),
                "calmar_ratio": round(float(calmar_cpu[b_i]), 3),
            })

    return results


def main():
    df_sig, candles_3m = load_or_build_signals()

    # Parameter Space to Exhaustively Sweep on GPU
    score_thresholds = [0, 40, 50, 55, 60, 65, 70, 75, 80]
    sl_multipliers = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    tp_multipliers = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    trail_triggers = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    trail_steps = [2.0, 3.0, 4.0, 5.0]

    grid = []
    for sc, sl, tp, tr, st in itertools.product(score_thresholds, sl_multipliers, tp_multipliers, trail_triggers, trail_steps):
        grid.append({
            "min_score": sc,
            "sl_mult": sl,
            "tp_mult": tp,
            "trail_trigger": tr,
            "trail_step": st,
        })

    print(f"\nGenerated {len(grid):,} Parameter Sets per Session Window\n", flush=True)

    sessions = [
        ("1. Morning Session (09:15-11:00)", 555, 660),
        ("2. Afternoon Session (13:30-15:00)", 810, 900),
        ("3. Combined Dual Session (09:15-11:00 + 13:30-15:00)", 555, 900),
    ]

    all_results = []
    t0 = time.time()

    for s_name, min_t, max_t in sessions:
        t_s = time.time()
        print(f">>> Running Pure CUDA Sweep: [{s_name}] ({len(grid):,} configs)...", flush=True)
        res = run_cuda_rejection_sweep(df_sig, candles_3m, s_name, min_t, max_t, grid, batch_size=100)
        all_results.extend(res)
        print(f"    Completed in {time.time()-t_s:.2f}s ({len(res):,} evals | {len(res)/(time.time()-t_s):.1f} configs/sec)", flush=True)

    total_time = time.time() - t0
    print("\n" + "=" * 145, flush=True)
    print(f"CUDA REJECTION SWEEP COMPLETED in {total_time:.2f}s ({len(all_results):,} evaluations | {len(all_results)/total_time:.1f} configs/sec)", flush=True)
    print("=" * 145, flush=True)

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 145, flush=True)
    print("TOP CHAMPION CONFIGURATIONS PER SESSION WINDOW (AMMU REJECTION STRATEGY)", flush=True)
    print("=" * 145, flush=True)

    champions = {}
    for s_name, _, _ in sessions:
        sub = df[df["session"] == s_name]
        valid = sub[sub["trades"] >= 200]
        if valid.empty:
            valid = sub
        champ = valid.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict()
        champions[s_name] = champ

        print(f"\n[CHAMPION FOR]: {s_name}")
        print(f"  * Optimal Min Score Threshold: Score >= {champ['min_score']}")
        print(f"  * SL Multiplier:                {champ['sl_mult']}x ATR")
        print(f"  * Target / TP Multiplier:       {champ['tp_mult']}x ATR")
        print(f"  * Trailing Stop:                Trigger @ +{champ['trail_trigger']} pts | Step = {champ['trail_step']} pts")
        print(f"  * Total Trades:                 {champ['trades']:,}")
        print(f"  * Trade Win Rate:               {champ['trade_win_rate']:.2f}%")
        print(f"  * DAILY WIN RATE:               {champ['daily_win_rate']:.1f}% GREEN DAYS ({champ['green_days']:,} Green / {champ['red_days']:,} Red Days)")
        print(f"  * Net Points Captured:          +{champ['net_points']:+,.2f} pts")
        print(f"  * Net Realized Profit (1 Lot):  Rs {champ['net_rs']:+,.2f}")
        print(f"  * Profit Factor:                {champ['profit_factor']:.3f}")
        print(f"  * Max Drawdown:                 Rs {champ['max_drawdown']:,.2f}")
        print(f"  * Calmar Ratio (Return/DD):     {champ['calmar_ratio']:.3f}")

    out_file = ROOT / "artifacts" / "f6_hybrid" / "ammu_rejection_optimus_champions.json"
    out_file.write_text(json.dumps({
        "champions": champions,
        "top_50": df.sort_values(by="calmar_ratio", ascending=False).head(50).to_dict(orient="records"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Optimus Rejection Champions JSON]: {out_file}", flush=True)


if __name__ == "__main__":
    main()
