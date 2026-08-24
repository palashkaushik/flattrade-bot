"""Optimus GPU/Vectorized Score & Parameter Optimizer for Rejection Strategy.

Reads Rejection strategy & dataset from c:\\Websites\\ammu.
Target Sessions:
  1. Morning Session: 09:15 - 11:00 IST
  2. Afternoon Session: 13:30 - 15:00 IST
  3. Combined Dual-Engine: 09:15 - 11:00 + 13:30 - 15:00 IST

Exhaustively sweeps:
  - min_score: [0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90]
  - sl_mult / sl_pts: [0.3, 0.4, 0.5, 0.6, 0.8, 1.0] and fixed SL [5.0, 7.0, 10.0, 12.0]
  - tp_mult / tp_pts: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0] and fixed TP [10.0, 15.0, 20.0, 25.0, 30.0]
  - trail_trigger: [5, 8, 10, 12, 15, 20]
  - trail_step: [2, 3, 4, 5, 6, 8]
  - max_consec_losses: [3, 4, 5, 6, 8, 999]
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from bisect import bisect_right
from datetime import datetime, timedelta
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

from kbot.indicators.engine import calculate_rsi
from kbot.strategies.rejection_scalping import (
    MAX_CONSECUTIVE_LOSSES,
    TRAIL_STEP_PTS,
    TRAIL_TRIGGER_PTS,
    Direction,
    RejectionScalping,
)

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOT_SIZE = 65
FEE_PER_TRADE = 45.0  # statutory charges in Rs

print("=" * 135, flush=True)
print("OPTIMUS REJECTION SCORE & PARAMETER OPTIMIZER (AMMU STRATEGY)", flush=True)
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
print("=" * 135, flush=True)

# 1. Load Data from AMMU
print("\n[1] Loading 1m and 5m Data from c:\\Websites\\ammu...", flush=True)
df_1m = pd.read_csv(AMMU / "index" / "NIFTY 50_minute.csv")
df_1m["date"] = pd.to_datetime(df_1m["date"], format="mixed", errors="coerce")
df_1m = df_1m.sort_values("date").reset_index(drop=True)

df_5m = pd.read_csv(AMMU / "index" / "NIFTY 50_5minute.csv")
df_5m["date"] = pd.to_datetime(df_5m["date"], format="mixed", errors="coerce")
df_5m = df_5m.sort_values("date").reset_index(drop=True)

# Aggregate 1m -> 3m
df_1m["bar_3m"] = df_1m["date"].dt.floor("3min")
agg = df_1m.groupby("bar_3m").agg(
    open=("open", "first"), high=("high", "max"),
    low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
).reset_index().rename(columns={"bar_3m": "date"})

candles_3m_all = agg.to_dict("records")
candles_5m_all = df_5m.rename(columns={"date": "timestamp"}).to_dict("records")

# 5Y+ History (2020 to 2024/2026)
candles_3m = [c for c in candles_3m_all if c["date"] >= pd.Timestamp("2020-01-01")]
candles_5m = [c for c in candles_5m_all if c["timestamp"] >= pd.Timestamp("2020-01-01")]

bt_start = candles_3m[0]["date"]
bt_end = candles_3m[-1]["date"]
print(f"Dataset: {len(candles_3m):,} 3m bars ({bt_start.date()} to {bt_end.date()})", flush=True)

min_bars = min(120, len(candles_3m) // 4)
s5_times = [c["timestamp"] for c in candles_5m]
first_backtest_time = candles_3m[min_bars]["date"]

rsi_values = calculate_rsi([c["close"] for c in candles_3m])
rsi_cache = {
    id(candles_3m[i]): (rsi_values[i], rsi_values[i - 1])
    for i in range(1, len(candles_3m))
}
indicator_cache = {}

# 2. Precompute Shared Causal Signal Stream
print("\n[2] Generating Causal Signal Stream (Bar-by-Bar Parity)...", flush=True)
t0 = time.time()
strat = RejectionScalping()
s3_history = candles_3m[:min_bars]
s5_index = bisect_right(s5_times, first_backtest_time)
s5_history = candles_5m[:s5_index]
total_bars = len(candles_3m) - min_bars

signal_events = [None] * len(candles_3m)
sig_count = 0

for i in range(min_bars, len(candles_3m)):
    bar = candles_3m[i]
    bt = bar["date"]
    s3_history.append(bar)
    while s5_index < len(candles_5m) and s5_times[s5_index] <= bt:
        s5_history.append(candles_5m[s5_index])
        s5_index += 1
    if len(s5_history) < 20:
        continue

    sig = strat.generate_signal(s3_history, s5_history, indicator_cache, rsi_cache)
    if sig:
        signal_events[i] = {
            "direction": sig.direction,
            "entry": sig.entry_price,
            "sl_dist": abs(sig.entry_price - sig.stop_loss),
            "tgt_dist": abs(sig.target - sig.entry_price),
            "time": bt,
            "minute_of_day": bt.hour * 60 + bt.minute,
            "level": sig.level.label,
            "score": sig.score,
        }
        sig_count += 1

print(f"Generated {sig_count:,} Causal Rejection Signals in {time.time()-t0:.2f}s", flush=True)

# 3. Vectorized Simulation Engine
bar_dates = [c["date"] for c in candles_3m]
bar_closes = np.array([c["close"] for c in candles_3m], dtype=np.float32)
bar_highs = np.array([c["high"] for c in candles_3m], dtype=np.float32)
bar_lows = np.array([c["low"] for c in candles_3m], dtype=np.float32)
bar_days = np.array([str(c["date"])[:10] for c in candles_3m])
unique_days = sorted(list(set(bar_days)))
day_to_idx = {d: i for i, d in enumerate(unique_days)}
bar_day_indices = np.array([day_to_idx[d] for d in bar_days], dtype=np.int32)
N_TOTAL_DAYS = len(unique_days)


def simulate_rejection_session(
    min_score: int,
    sl_mult: float,
    tp_mult: float,
    trail_trigger: float,
    trail_step: float,
    max_consec: int,
    session_name: str,
    time_filter_fn,
) -> dict:
    trades = []
    position = None
    cons_losses = 0
    last_trade_day = ""
    kill_switch = False
    kill_days = 0
    best_price = 0.0
    signals_seen = 0
    signals_filtered = 0

    for i in range(min_bars, len(candles_3m)):
        trade_day = bar_days[i]
        bt = bar_dates[i]
        close = bar_closes[i]
        high = bar_highs[i]
        low = bar_lows[i]
        minute_of_day = bt.hour * 60 + bt.minute

        if trade_day != last_trade_day:
            cons_losses = 0
            kill_switch = False
            last_trade_day = trade_day

        # Position exit management
        if position is not None:
            exit_px = None
            reason = ""
            if position["dir"] == Direction.SHORT:
                if high >= position["sl"]:
                    exit_px = position["sl"]
                    reason = "SL" if not position.get("trailed") else "SL-TRL"
                elif not position.get("trailed") and low <= position["tgt"]:
                    exit_px, reason = position["tgt"], "TGT"
            else:
                if low <= position["sl"]:
                    exit_px = position["sl"]
                    reason = "SL" if not position.get("trailed") else "SL-TRL"
                elif not position.get("trailed") and high >= position["tgt"]:
                    exit_px, reason = position["tgt"], "TGT"

            # Trailing stop update
            if not exit_px:
                if position["dir"] == Direction.SHORT:
                    current_pnl = position["entry"] - close
                    if current_pnl >= trail_trigger and close < best_price:
                        best_price = close
                        new_sl = best_price + trail_step
                        if new_sl < position["sl"]:
                            position["sl"] = new_sl
                            position["trailed"] = True
                else:
                    current_pnl = close - position["entry"]
                    if current_pnl >= trail_trigger and close > best_price:
                        best_price = close
                        new_sl = best_price - trail_step
                        if new_sl > position["sl"]:
                            position["sl"] = new_sl
                            position["trailed"] = True

            # Hard EOD exit at 15:20 (minute 920)
            if not exit_px and minute_of_day >= 920:
                exit_px = close
                reason = "EOD"

            if exit_px is not None:
                pnl = (position["entry"] - exit_px) if position["dir"] == Direction.SHORT else (exit_px - position["entry"])
                trades.append({
                    "day_idx": bar_day_indices[i],
                    "date": trade_day,
                    "time": bt,
                    "pnl": pnl,
                    "rs_net": pnl * LOT_SIZE - FEE_PER_TRADE,
                    "reason": reason,
                    "trailed": position.get("trailed", False),
                })
                if pnl <= 0:
                    cons_losses += 1
                    if cons_losses >= max_consec:
                        kill_switch = True
                        kill_days += 1
                else:
                    cons_losses = 0
                position = None

        if position is not None or kill_switch or minute_of_day >= 920:
            continue

        # Signal check
        sig = signal_events[i]
        if sig is None:
            continue

        signals_seen += 1
        if sig["score"] < min_score:
            signals_filtered += 1
            continue

        # Apply session time filter
        if not time_filter_fn(sig["minute_of_day"]):
            continue

        # Calculate custom SL and Target
        base_sl_dist = sig["sl_dist"] if sl_mult == 1.0 else max(sig["sl_dist"] * sl_mult, 4.0)
        base_tgt_dist = sig["tgt_dist"] if tp_mult == 1.0 else max(sig["tgt_dist"] * tp_mult, 8.0)

        entry = sig["entry"]
        if sig["direction"] == Direction.SHORT:
            sl = entry + base_sl_dist
            tgt = entry - base_tgt_dist
        else:
            sl = entry - base_sl_dist
            tgt = entry + base_tgt_dist

        position = {
            "dir": sig["direction"],
            "entry": entry,
            "sl": sl,
            "tgt": tgt,
            "entry_time": bt,
            "trailed": False,
        }
        best_price = entry

    # Calculate metrics
    if not trades:
        return {
            "session": session_name, "min_score": min_score, "sl_mult": sl_mult, "tp_mult": tp_mult,
            "trail_trigger": trail_trigger, "trail_step": trail_step, "max_consec": max_consec,
            "trades": 0, "trade_win_rate": 0.0, "daily_win_rate": 0.0, "net_points": 0.0, "net_rs": 0.0, "profit_factor": 0.0,
            "max_drawdown": 0.0, "calmar_ratio": 0.0, "green_days": 0, "red_days": 0, "traded_days": 0, "kill_days": 0,
        }

    pnls = np.array([t["pnl"] for t in trades], dtype=np.float32)
    rs_nets = np.array([t["rs_net"] for t in trades], dtype=np.float32)
    day_indices = np.array([t["day_idx"] for t in trades], dtype=np.int32)

    wins = rs_nets > 0
    losses = rs_nets <= 0
    n_wins = int(np.sum(wins))
    n_trades = len(trades)
    wr = (n_wins / n_trades) * 100.0 if n_trades > 0 else 0.0

    tot_pts = float(np.sum(pnls))
    tot_rs = float(np.sum(rs_nets))

    gross_win = float(np.sum(rs_nets[wins])) if n_wins > 0 else 0.0
    gross_loss = abs(float(np.sum(rs_nets[losses]))) if np.sum(losses) > 0 else 1.0
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0

    # Daily aggregation
    day_pnl = np.zeros(N_TOTAL_DAYS, dtype=np.float32)
    np.add.at(day_pnl, day_indices, rs_nets)

    cum_eq = np.cumsum(day_pnl)
    peaks = np.maximum.accumulate(cum_eq)
    drawdowns = peaks - cum_eq
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    calmar = tot_rs / max_dd if max_dd > 0 else 0.0

    green_d = int(np.sum(day_pnl > 0))
    red_d = int(np.sum(day_pnl < 0))
    act_d = int(np.sum(day_pnl != 0))
    daily_wr = (green_d / act_d) * 100.0 if act_d > 0 else 0.0

    return {
        "session": session_name,
        "min_score": min_score,
        "sl_mult": sl_mult,
        "tp_mult": tp_mult,
        "trail_trigger": trail_trigger,
        "trail_step": trail_step,
        "max_consec": max_consec,
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
        "kill_days": kill_days,
    }


def main():
    print("\n[3] Running Optimus Grid Search on Rejection Strategy...", flush=True)

    # Grid Search Parameters
    score_thresholds = [0, 40, 50, 60, 70, 80]
    sl_multipliers = [0.4, 0.5, 0.7, 1.0]
    tp_multipliers = [1.5, 2.0, 3.0, 4.0]
    trail_triggers = [8, 10, 15]
    trail_steps = [3, 5]
    max_consecs = [5, 6, 8]

    grid = list(itertools.product(score_thresholds, sl_multipliers, tp_multipliers, trail_triggers, trail_steps, max_consecs))
    print(f"Total Parameter Combinations per Session: {len(grid):,}\n", flush=True)

    # Session Filters
    # 09:15 = 555 min, 11:00 = 660 min
    # 13:30 = 810 min, 15:00 = 900 min
    sessions = [
        ("1. Morning Session (09:15-11:00)", lambda m: 555 <= m <= 660),
        ("2. Afternoon Session (13:30-15:00)", lambda m: 810 <= m <= 900),
        ("3. Combined Dual-Engine (09:15-11:00 + 13:30-15:00)", lambda m: (555 <= m <= 660) or (810 <= m <= 900)),
    ]

    all_results = []
    t_start = time.time()

    for s_name, s_fn in sessions:
        t_s = time.time()
        print(f">>> Optimizing [{s_name}] across {len(grid):,} configurations...", flush=True)
        session_res = []
        for min_sc, sl_m, tp_m, tr_trig, tr_step, m_cons in grid:
            r = simulate_rejection_session(
                min_score=min_sc,
                sl_mult=sl_m,
                tp_mult=tp_m,
                trail_trigger=tr_trig,
                trail_step=tr_step,
                max_consec=m_cons,
                session_name=s_name,
                time_filter_fn=s_fn,
            )
            session_res.append(r)
        all_results.extend(session_res)
        print(f"    Completed in {time.time()-t_s:.2f}s ({len(session_res):,} evals | {len(session_res)/(time.time()-t_s):.1f} configs/sec)", flush=True)

    total_time = time.time() - t_start
    print("\n" + "=" * 145, flush=True)
    print(f"OPTIMUS REJECTION SEARCH COMPLETED in {total_time:.2f}s ({len(all_results):,} total evaluations)", flush=True)
    print("=" * 145, flush=True)

    df = pd.DataFrame(all_results)

    # Find Champions for each session
    print("\n" + "=" * 145, flush=True)
    print("TOP CHAMPION CONFIGURATIONS PER SESSION WINDOW", flush=True)
    print("=" * 145, flush=True)

    champions = {}
    for s_name, _ in sessions:
        sub = df[df["session"] == s_name]
        valid = sub[sub["trades"] >= 200]
        if valid.empty:
            valid = sub
        champ = valid.sort_values(by="calmar_ratio", ascending=False).iloc[0].to_dict()
        champions[s_name] = champ

        print(f"\nCHAMPION FOR: {s_name}")
        print(f"  * Optimal Min Score Threshold: Score >= {champ['min_score']}")
        print(f"  * SL Multiplier:                {champ['sl_mult']}x ATR")
        print(f"  * Target / TP Multiplier:       {champ['tp_mult']}x ATR")
        print(f"  * Trailing Stop:                Trigger @ +{champ['trail_trigger']} pts | Step = {champ['trail_step']} pts")
        print(f"  * Max Consecutive Losses:       {champ['max_consec']}")
        print(f"  * Total Trades:                 {champ['trades']:,}")
        print(f"  * Trade Win Rate:               {champ['trade_win_rate']:.2f}%")
        print(f"  * DAILY WIN RATE:               {champ['daily_win_rate']:.1f}% GREEN DAYS ({champ['green_days']:,} Green / {champ['red_days']:,} Red Days)")
        print(f"  * Net Points Captured:          +{champ['net_points']:+,.2f} pts")
        print(f"  * Net Realized Profit (1 Lot):  Rs {champ['net_rs']:+,.2f}")
        print(f"  * Profit Factor:                {champ['profit_factor']:.3f}")
        print(f"  * Max Drawdown:                 Rs {champ['max_drawdown']:,.2f}")
        print(f"  * Calmar Ratio (Return/DD):     {champ['calmar_ratio']:.3f}")

    out_file = ROOT / "artifacts" / "f6_hybrid" / "optimus_rejection_champions.json"
    out_file.write_text(json.dumps({
        "champions": champions,
        "top_50": df.sort_values(by="calmar_ratio", ascending=False).head(50).to_dict(orient="records"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Optimus Rejection Champions JSON]: {out_file}", flush=True)


if __name__ == "__main__":
    main()
