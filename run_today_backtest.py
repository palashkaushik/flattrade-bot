"""Run Quad Rotation Strategy on Today's (2026-08-05) Upstox Dataset Warm-warmed with Yesterday's (2026-08-04) Dataset.

Includes Embedded S4 (>25 bars <= 20.0) Trade Reversal Rule:
  - When S4 remains <= 20.0 for >25 consecutive bars, Reverse Mode activates.
  - Instead of buying the signaled option contract (e.g. CE), buy the OPPOSITE side ITM2 option contract (e.g. PE).
  - Resets automatically when S4 moves above 20.0.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np

from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.divergence import DivergenceEngine

DIR_YESTERDAY = Path("C:/Websites/ammu/data/2026-08-04")
DIR_TODAY = Path("C:/Websites/ammu/data/2026-08-05")

SPOT_YESTERDAY = DIR_YESTERDAY / "nifty50_index_1m_2026-08-04.csv"
OPTS_YESTERDAY = DIR_YESTERDAY / "nifty_options_1m_2026-08-04_exp20260811.csv"

SPOT_TODAY = DIR_TODAY / "nifty50_index_1m_2026-08-05.csv"
OPTS_TODAY = DIR_TODAY / "nifty_options_1m_2026-08-05.csv"

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SL_POINTS, TP_POINTS = 10.0, 15.0
SESSION_START, SESSION_END = 560, 900  # 09:20 to 15:00
DAILY_SHUTDOWN_RS = 2000.0
CONSECUTIVE_LOSS_LIMIT = 6


def load_day_dataset(spot_path: Path, opts_path: Path):
    spot_df = pd.read_csv(spot_path, header=None, skiprows=1)
    spot_df.columns = ["timestamp", "open", "high", "low", "close", "vol1", "vol2"][:len(spot_df.columns)]
    spot_df = spot_df.dropna(subset=["timestamp", "close"]).copy()
    spot_df["dt"] = pd.to_datetime(spot_df["timestamp"])
    spot_df["min"] = spot_df["dt"].dt.hour * 60 + spot_df["dt"].dt.minute
    spot_df = spot_df.sort_values("dt").reset_index(drop=True)
    spot_map = dict(zip(spot_df["min"], spot_df["close"]))

    opts_df = pd.read_csv(opts_path, header=None, skiprows=1)
    opts_df.columns = ["side", "strike", "timestamp", "open", "high", "low", "close", "volume", "oi"][:len(opts_df.columns)]
    opts_df = opts_df.dropna(subset=["timestamp", "close"]).copy()
    opts_df["dt"] = pd.to_datetime(opts_df["timestamp"])
    opts_df["min"] = opts_df["dt"].dt.hour * 60 + opts_df["dt"].dt.minute

    opts_groups = {}
    for (side, strike), g in opts_df.groupby(["side", "strike"]):
        g_sorted = g.sort_values("dt").reset_index(drop=True)
        strike_int = int(float(strike))
        opts_groups[(str(side).strip().upper(), strike_int)] = dict(zip(
            g_sorted["min"],
            zip(g_sorted["open"], g_sorted["high"], g_sorted["low"], g_sorted["close"])
        ))

    return spot_map, opts_groups


class SymbolTracker1M:
    def __init__(self):
        self.stoch = QuadStochastics()
        self.divergence = DivergenceEngine()
        self.history_1m: List[Candle] = []
        self.setup_active = False
        self.prev_s1 = None
        self.s4_embedded_count = 0

    def push_1m(self, candle_1m: Candle) -> Tuple[bool, bool]:
        self.history_1m.append(candle_1m)
        if len(self.history_1m) > 50:
            self.history_1m.pop(0)

        stoch_vals = self.stoch.push(candle_1m.high, candle_1m.low, candle_1m.close)
        s1, s2, s3, s4 = (stoch_vals.get(k) for k in ("s1d", "s2d", "s3d", "s4d"))
        self.prev_s1 = s1

        # Embedded S4 counter (S4 <= 20.0)
        if s4 is not None:
            if s4 <= 20.0:
                self.s4_embedded_count += 1
            else:
                self.s4_embedded_count = 0

        is_reverse_mode = self.s4_embedded_count > 25

        self.divergence.update(candle_1m.close, s1)
        has_bull_div = self.divergence.has_bullish_trough_divergence()

        is_flag = False if any(v is None for v in (s1, s4)) else (s4 >= 79.5 and s1 <= 20.5)
        is_super = False if any(v is None for v in (s1, s2, s3, s4)) else all(v <= 20.5 for v in (s1, s2, s3, s4))

        if (is_flag or is_super) and has_bull_div:
            self.setup_active = True

        trig_1m = False
        if self.setup_active:
            if BullishPinBarDetector.check_vicinity_breakout(self.history_1m, max_lookback=10):
                trig_1m = True
                self.setup_active = False

        return trig_1m, is_reverse_mode


def main():
    print("Loading Yesterday's (2026-08-04) dataset for pre-warmup...")
    yest_spot, yest_opts = load_day_dataset(SPOT_YESTERDAY, OPTS_YESTERDAY)
    
    print("Loading Today's (2026-08-05) dataset...")
    today_spot, today_opts = load_day_dataset(SPOT_TODAY, OPTS_TODAY)

    all_symbols = set(yest_opts.keys()) | set(today_opts.keys())
    symbol_trackers = {key: SymbolTracker1M() for key in all_symbols}

    # 1. Warm up with Yesterday's Candles (2026-08-04)
    for key, minute_map in yest_opts.items():
        tracker = symbol_trackers[key]
        for m in sorted(minute_map.keys()):
            o, h, l, c = minute_map[m]
            tracker.push_1m(Candle(open=o, high=h, low=l, close=c, minute=m))

    print("[OK] Stochastics fully pre-warmed! Evaluating Today's (2026-08-05) trading session...")

    # 2. Feed Today's Candles (2026-08-05)
    per_minute_triggers = {}
    for key, minute_map in today_opts.items():
        tracker = symbol_trackers[key]
        side, strike = key

        for m in sorted(minute_map.keys()):
            o, h, l, c = minute_map[m]
            candle_1m = Candle(open=o, high=h, low=l, close=c, minute=m)
            t1, is_rev = tracker.push_1m(candle_1m)

            if t1:
                per_minute_triggers.setdefault(m, []).append((side, strike, c, is_rev))

    # Portfolio execution
    all_today_minutes = sorted(today_spot.keys())
    trades = []
    pos = None
    daily_pnl = 0.0
    consecutive_losses = 0
    shutdown = False

    for minute in all_today_minutes:
        if minute < SESSION_START or minute > SESSION_END:
            continue

        spot_px = today_spot[minute]
        atm = int(round(spot_px / 50.0) * 50)
        active_strikes = {"CE": atm + CE_OFFSET, "PE": atm + PE_OFFSET}

        # 1. Position Exits
        if pos is not None:
            side = pos["side"]
            strike = pos["strike"]
            candle_data = today_opts.get((side, strike), {}).get(minute)

            if candle_data is not None:
                o_px, h_px, l_px, c_px = candle_data
                pos["last_px"] = float(c_px)
                pos["duration_min"] += 1

                if daily_pnl + (c_px - pos["entry"]) * LOT_SIZE <= -DAILY_SHUTDOWN_RS:
                    pts = round(c_px - pos["entry"], 2)
                    trades.append({
                        "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": side, "strike": strike, "entry": pos["entry"],
                        "exit": c_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": "SHUTDOWN", "duration_min": pos["duration_min"], "is_rev": pos["is_rev"]
                    })
                    daily_pnl += pts * LOT_SIZE
                    pos = None
                    shutdown = True
                    continue

                exit_px, reason = None, ""
                if h_px >= pos["tgt"] and l_px <= pos["sl"]:
                    exit_px, reason = pos["sl"], "SL"
                elif h_px >= pos["tgt"]:
                    exit_px, reason = pos["tgt"], "TP"
                elif l_px <= pos["sl"]:
                    exit_px, reason = pos["sl"], "SL"

                if exit_px is None:
                    tracker = symbol_trackers[(side, strike)]
                    tracker.divergence.update(c_px, tracker.prev_s1)
                    if tracker.divergence.has_bearish_peak_divergence():
                        exit_px, reason = c_px, "BEARISH_PEAK_REVERSAL"

                if exit_px is not None:
                    pts = round(exit_px - pos["entry"], 2)
                    trades.append({
                        "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": side, "strike": strike, "entry": pos["entry"],
                        "exit": exit_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": reason, "duration_min": pos["duration_min"], "is_rev": pos["is_rev"]
                    })
                    daily_pnl += pts * LOT_SIZE
                    consecutive_losses = consecutive_losses + 1 if pts <= 0 else 0
                    if consecutive_losses >= CONSECUTIVE_LOSS_LIMIT or daily_pnl <= -DAILY_SHUTDOWN_RS:
                        shutdown = True
                    pos = None

            if minute >= SESSION_END and pos is not None:
                pts = round(pos["last_px"] - pos["entry"], 2)
                trades.append({
                    "entry_min": pos["entry_min"], "exit_min": minute,
                    "side": side, "strike": strike, "entry": pos["entry"],
                    "exit": pos["last_px"], "pts": pts, "rs": round(pts * LOT_SIZE),
                    "reason": "EOD", "duration_min": pos["duration_min"], "is_rev": pos["is_rev"]
                })
                daily_pnl += pts * LOT_SIZE
                pos = None
                break

        if pos is not None or shutdown:
            continue

        # 2. Evaluate Signals with Reverse Mode logic
        trigs = per_minute_triggers.get(minute, [])
        for (signal_side, signal_strike, c_px, is_rev) in trigs:
            if signal_strike == active_strikes[signal_side] and pos is None:
                if is_rev:
                    actual_side = "PE" if signal_side == "CE" else "CE"
                    actual_strike = active_strikes[actual_side]
                else:
                    actual_side = signal_side
                    actual_strike = signal_strike

                actual_candle = today_opts.get((actual_side, actual_strike), {}).get(minute)
                if actual_candle is not None:
                    entry_px = float(actual_candle[3])  # close price
                    pos = {
                        "side": actual_side, "strike": actual_strike, "entry": entry_px,
                        "sl": entry_px - SL_POINTS, "tgt": entry_px + TP_POINTS,
                        "entry_min": minute, "last_px": entry_px, "duration_min": 0, "is_rev": is_rev
                    }
                    break

    # Summary Output
    print("\n" + "=" * 120)
    print("TODAY'S SESSION RESULTS (2026-08-05) - EMBEDDED S4 (>25 BARS <=20) REVERSE TRADE MODE")
    print("=" * 120)
    if not trades:
        print("No trades triggered today under Quad Stochastics + Bullish Trough Divergence + Vicinity Pin Bar Breakout rules.")
    else:
        print(f"{'ENTRY':5s} | {'EXIT':5s} | {'SIDE':4s} | {'STRIKE':6s} | {'MODE':8s} | {'ENTRY_PX':8s} | {'EXIT_PX':8s} | {'PTS':7s} | {'P&L (Rs)':10s} | {'REASON'}")
        print("-" * 120)
        for t in trades:
            e_t = f"{t['entry_min'] // 60:02d}:{t['entry_min'] % 60:02d}"
            x_t = f"{t['exit_min'] // 60:02d}:{t['exit_min'] % 60:02d}"
            pts_s = f"{t['pts']:+.2f}"
            rs_s = f"Rs {t['rs']:+,d}"
            mode_s = "REVERSE" if t["is_rev"] else "NORMAL"
            print(f"{e_t:5s} | {x_t:5s} | {t['side']:4s} | {t['strike']:6d} | {mode_s:8s} | {t['entry']:8.2f} | {t['exit']:8.2f} | {pts_s:>7s} | {rs_s:>10s} | {t['reason']}")
        
        wins = [t for t in trades if t["pts"] > 0]
        pts_tot = sum(t["pts"] for t in trades)
        rs_tot = sum(t["rs"] for t in trades)
        print("=" * 120)
        print(f"Total Trades: {len(trades)} | Wins: {len(wins)} | Win Rate: {len(wins)/len(trades)*100:.1f}% | Net P&L: {pts_tot:+.2f} pts (Rs {rs_tot:+,d})")


if __name__ == "__main__":
    main()
