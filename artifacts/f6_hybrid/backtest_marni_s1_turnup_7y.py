"""
MARNI S1 TURN-UP TRIGGER + TRAILING SL (UNLIMITED PROFIT) — 7-YEAR BACKTEST (2020 - 2026)
=========================================================================================
Strategy Architecture:
  - 4-Timeframe Concurrent MTF Engine (1m, 2m, 3m, 5m)
  - Stochastic Indicators: S1(9,3), S2(14,3), S3(40,4), S4(60,10)
  - Triggers:
      * Super Setup: S1 Turn-Up (S1_t > S1_t-1)
      * Flag Setup: BullishPinBar vicinity breakout
  - Position Management:
      * Trailing SL: +10pt Gain -> +5pt Stop Trail
      * Target: UNLIMITED PROFIT (No static TP cap)
      * Daily Max Loss: Rs 2,000 (30.77 pts)
      * Exit on Bearish Peak Divergence / EOD (15:00)
  - Friction Applied: ₹15/order Brokerage + 0.50 pt/order Slippage
  - Dataset: 2020-01-01 to 2026-05-05 (7 Calendar Years · 1,574 Trading Days)
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from flattrade_bot.indicators.divergence import DivergenceEngine

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_MAX_LOSS_RS = -2000.0
DAILY_MAX_LOSS_PTS = DAILY_MAX_LOSS_RS / LOT_SIZE  # -30.77 pts
CONSECUTIVE_LOSS_LIMIT = 4
ATR_PERIOD = 14
TRAIL_STEP_PTS = 10.0
TRAIL_AMOUNT_PTS = 5.0
SYMBOL_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")

TF_SPECS = {
    "1m": (1, 10, 6.0, 30.0),
    "2m": (2,  5, 10.0, 15.0),
    "3m": (3,  4, 8.0,  25.0),
    "5m": (5,  3, 10.0, 35.0),
}

GLOBAL_SPOT = {}

def init_worker(spot):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot
    source.GLOBAL_CACHE = {}

class IncrementalATR:
    def __init__(self, period=14):
        self.period = period
        self._buf = deque(maxlen=period)
        self.atr = None
        self.prev_close = None
        self._n = 0

    def update(self, h, l, c):
        tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close)) if self.prev_close else h - l
        self._buf.append(tr)
        self._n += 1
        self.prev_close = c
        if self._n < self.period:
            self.atr = None
        elif self._n == self.period:
            self.atr = sum(self._buf) / self.period
        else:
            self.atr = (self.atr * (self.period - 1) + tr) / self.period
        return self.atr

class TFTracker:
    def __init__(self, lb, tf_sl, tf_tp, s1_spec, s2_spec, s3_spec, s4_spec):
        self.lb = lb
        self.tf_sl = tf_sl
        self.tf_tp = tf_tp
        self.s1 = IncrementalStochastic(*s1_spec)
        self.s2 = IncrementalStochastic(*s2_spec)
        self.s3 = IncrementalStochastic(*s3_spec)
        self.s4 = IncrementalStochastic(*s4_spec)
        self.div = DivergenceEngine()
        self.atr = IncrementalATR(ATR_PERIOD)
        self.hist = []
        self.setup = False
        self.stype = ""
        self.prev_s1 = None
        self.s4_emb = 0

    def push(self, c: Candle):
        self.hist.append(c)
        if len(self.hist) > 60:
            self.hist.pop(0)
        s1 = self.s1.push(c.high, c.low, c.close)
        s2 = self.s2.push(c.high, c.low, c.close)
        s3 = self.s3.push(c.high, c.low, c.close)
        s4 = self.s4.push(c.high, c.low, c.close)
        atr_val = self.atr.update(c.high, c.low, c.close)

        if s4 is not None:
            self.s4_emb = self.s4_emb + 1 if s4 <= 20 else 0
        emb = self.s4_emb > 25
        self.div.update(c.close, s1)
        bull_div = self.div.has_bullish_trough_divergence()

        is_flag = s4 is not None and s1 is not None and s4 >= 79.5 and s1 <= 20.5
        is_super = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
        if (is_flag or is_super) and bull_div:
            self.setup = True
            self.stype = "super" if is_super else "flag"

        is_rev = emb and self.stype == "super"
        triggered = False

        if self.setup and len(self.hist) >= 2:
            if self.stype == "super":
                # S1 Turn-Up Trigger for Super Signal
                if s1 is not None and self.prev_s1 is not None and s1 > self.prev_s1:
                    triggered = True
                    self.setup = False
            else:
                # Flag Signal: Pin Bar vicinity breakout
                if BullishPinBarDetector.check_vicinity_breakout(self.hist, self.lb):
                    triggered = True
                    self.setup = False

        self.prev_s1 = s1
        return triggered, is_rev, self.stype, c.close, atr_val

class MTFTracker:
    def __init__(self, s1_spec, s2_spec, s3_spec, s4_spec):
        self.trackers = {
            tf: TFTracker(spec[1], spec[2], spec[3], s1_spec, s2_spec, s3_spec, s4_spec)
            for tf, spec in TF_SPECS.items()
        }
        self.bufs = {tf: [] for tf in TF_SPECS}

    def push_1m(self, c1m: Candle):
        out = []
        for tf, spec in TF_SPECS.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf]) == spec[0]:
                buf = self.bufs[tf]
                ctf = Candle(
                    open=buf[0].open,
                    high=max(x.high for x in buf),
                    low=min(x.low for x in buf),
                    close=buf[-1].close,
                    minute=buf[-1].minute,
                )
                self.bufs[tf] = []
                trig, is_rev, stype, px, atr_val = self.trackers[tf].push(ctf)
                if trig:
                    out.append((tf, is_rev, stype, px, atr_val, spec[2], spec[3]))
        return out

def option_rows(frame, groups, symbol):
    indexes = groups.get(symbol)
    if indexes is None:
        return []
    rows = frame.iloc[indexes].sort_values("time")
    res = []
    for _, row in rows.iterrows():
        t_str = str(row["time"])
        parts = t_str.split(":")
        m = int(parts[0]) * 60 + int(parts[1])
        res.append({
            "time": t_str,
            "minute": m,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        })
    return res

def process_day(task):
    day, opt_file, prev_opt_file, include_fees = task

    spot = GLOBAL_SPOT.get(day)
    if spot is None or len(spot["min"]) < 50:
        return []

    current = source.cached_option(str(opt_file))
    if current is None:
        return []
    frame, groups, prefix = current

    sp0 = float(spot["close"][0])
    atm0 = int(round(sp0 / 50.0) * 50)
    target_strikes = set(range(atm0 - 250, atm0 + 300, 50))

    s1_spec, s2_spec, s3_spec, s4_spec = (9, 3), (14, 3), (40, 4), (60, 10)

    # Warm up trackers on previous day if available
    trk = {}
    if prev_opt_file:
        prev_res = source.cached_option(str(prev_opt_file))
        if prev_res:
            p_frame, p_groups, _ = prev_res
            for sym in p_groups:
                m = SYMBOL_RE.match(sym)
                if not m or int(m.group(2)) not in target_strikes:
                    continue
                p_rows = option_rows(p_frame, p_groups, sym)
                trk[sym] = MTFTracker(s1_spec, s2_spec, s3_spec, s4_spec)
                for r in p_rows:
                    trk[sym].push_1m(Candle(r["open"], r["high"], r["low"], r["close"], minute=r["minute"]))

    # Index current day option bars
    bars = {}
    for sym in groups:
        m = SYMBOL_RE.match(sym)
        if not m or int(m.group(2)) not in target_strikes:
            continue
        if sym not in trk:
            trk[sym] = MTFTracker(s1_spec, s2_spec, s3_spec, s4_spec)
        r_list = option_rows(frame, groups, sym)
        bars[sym] = {r["minute"]: r for r in r_list}

    # Step through minutes to gather trigger signals
    pmtrig = defaultdict(list)
    for m in range(SESSION_START, DAY_LAST + 1):
        for sym, b_map in bars.items():
            if m not in b_map:
                continue
            r = b_map[m]
            c = Candle(r["open"], r["high"], r["low"], r["close"], minute=m)
            trigs = trk[sym].push_1m(c)
            if trigs and m >= SESSION_START:
                match = SYMBOL_RE.match(sym)
                if match:
                    stk = int(match.group(2))
                    side = match.group(3)
                    for (tf, is_rev, stype, px, atr_val, sl_pts, tp_pts) in trigs:
                        pmtrig[m].append((side, stk, sym, px, is_rev, tf, sl_pts, tp_pts, atr_val))

    # Helper function for latest spot lookup
    def get_latest_spot(minute):
        idx = 0
        for i, m_val in enumerate(spot["min"]):
            if int(m_val) <= minute:
                idx = i
            else:
                break
        return float(spot["close"][idx])

    # Position execution & Trailing SL simulation
    pos = None
    trades = []
    dpnl = 0.0
    closs = 0
    shut = False

    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None:
            sym = pos["symbol"]
            b_map = bars.get(sym, {})
            if minute in b_map:
                b = b_map[minute]
                o, h, l, c = b["open"], b["high"], b["low"], b["close"]
                pos["last_px"] = c
                pos["duration_min"] += 1

                # Trailing SL update (+10pt Gain -> +5pt Trail)
                gain = c - pos["entry"]
                steps = int(gain / TRAIL_STEP_PTS)
                if steps > pos["trail_steps"]:
                    pos["sl"] += (steps - pos["trail_steps"]) * TRAIL_AMOUNT_PTS
                    pos["trail_steps"] = steps

                # Daily max loss check (-Rs 2,000)
                unrealized_rs = (c - pos["entry"]) * LOT_SIZE
                if dpnl * LOT_SIZE + unrealized_rs <= DAILY_MAX_LOSS_RS:
                    pts = round(c - pos["entry"], 2)
                    fee = trade_cost(pos["entry"], c, BROKERAGE_PER_ORDER) if include_fees else 0.0
                    rs_net = round(pts * LOT_SIZE - fee, 2)
                    trades.append({
                        "date": day,
                        "entry_min": pos["entry_min"],
                        "exit_min": minute,
                        "side": pos["side"],
                        "symbol": pos["symbol"],
                        "entry": pos["entry"],
                        "exit": c,
                        "points": pts,
                        "fee": fee,
                        "rs_net": rs_net,
                        "reason": "SHUTDOWN_LOSS",
                        "duration_min": pos["duration_min"],
                        "is_rev": pos["is_rev"],
                        "tf": pos["tf"],
                    })
                    dpnl += pts
                    pos = None
                    shut = True
                    continue

                ex, rsn = None, ""
                if l <= pos["sl"]:
                    ex, rsn = pos["sl"], "TRAIL_SL" if pos["trail_steps"] > 0 else "SL"

                if ex is None:
                    t1 = trk.get(pos["symbol"])
                    if t1:
                        t1m = t1.trackers["1m"]
                        t1m.div.update(c, t1m.prev_s1)
                        if t1m.div.has_bearish_peak_divergence():
                            ex, rsn = c, "BEARISH_PEAK_REVERSAL"

                if ex is not None:
                    slip = SLIPPAGE_PTS if include_fees else 0.0
                    entry_f = pos["entry"] + slip
                    exit_f = ex - slip
                    pts = round(exit_f - entry_f, 2)
                    fee = trade_cost(entry_f, exit_f, BROKERAGE_PER_ORDER) if include_fees else 0.0
                    rs_net = round(pts * LOT_SIZE - fee, 2)

                    trades.append({
                        "date": day,
                        "entry_min": pos["entry_min"],
                        "exit_min": minute,
                        "side": pos["side"],
                        "symbol": pos["symbol"],
                        "entry": entry_f,
                        "exit": exit_f,
                        "points": pts,
                        "fee": fee,
                        "rs_net": rs_net,
                        "reason": rsn,
                        "duration_min": pos["duration_min"],
                        "is_rev": pos["is_rev"],
                        "tf": pos["tf"],
                    })
                    dpnl += pts
                    closs = closs + 1 if pts <= 0 else 0
                    if closs >= CONSECUTIVE_LOSS_LIMIT or dpnl <= DAILY_MAX_LOSS_PTS:
                        shut = True
                    pos = None

            if minute >= SESSION_END and pos is not None:
                slip = SLIPPAGE_PTS if include_fees else 0.0
                entry_f = pos["entry"] + slip
                exit_f = pos["last_px"] - slip
                pts = round(exit_f - entry_f, 2)
                fee = trade_cost(entry_f, exit_f, BROKERAGE_PER_ORDER) if include_fees else 0.0
                rs_net = round(pts * LOT_SIZE - fee, 2)

                trades.append({
                    "date": day,
                    "entry_min": pos["entry_min"],
                    "exit_min": minute,
                    "side": pos["side"],
                    "symbol": pos["symbol"],
                    "entry": entry_f,
                    "exit": exit_f,
                    "points": pts,
                    "fee": fee,
                    "rs_net": rs_net,
                    "reason": "EOD",
                    "duration_min": pos["duration_min"],
                    "is_rev": pos["is_rev"],
                    "tf": pos["tf"],
                })
                dpnl += pts
                pos = None
                break

        if pos is not None or shut or minute >= SESSION_END:
            continue

        for (sig_side, sig_stk, sig_sym, c_px, is_rev, tf, sl_pts, tp_pts, atr_val) in pmtrig.get(minute, []):
            sp_cur = get_latest_spot(minute)
            atm_cur = int(round(sp_cur / 50.0) * 50)
            target_stk = atm_cur - 100 if sig_side == "CE" else atm_cur + 100

            if target_stk == sig_stk and pos is None:
                asym = sig_sym
                as2 = sig_side
                if is_rev:
                    as2 = "PE" if sig_side == "CE" else "CE"
                    rev_stk = atm_cur - 100 if as2 == "CE" else atm_cur + 100
                    asym = f"{prefix}{rev_stk}{as2}"

                if asym in bars and minute in bars[asym]:
                    b_bar = bars[asym][minute]
                    ep = b_bar["close"]
                    pos = {
                        "side": as2,
                        "symbol": asym,
                        "entry": ep,
                        "sl": ep - sl_pts,
                        "entry_min": minute,
                        "last_px": ep,
                        "duration_min": 0,
                        "is_rev": is_rev,
                        "tf": tf,
                        "trail_steps": 0,
                    }
                    break

    return trades

def compute_stats(trades: list[dict], days_count: int) -> dict:
    wins = [t for t in trades if t["rs_net"] > 0]
    losses = [t for t in trades if t["rs_net"] <= 0]
    loss_total = abs(sum(t["rs_net"] for t in losses))
    win_total = sum(t["rs_net"] for t in wins)
    net_rs = sum(t["rs_net"] for t in trades)
    net_pts = sum(t["points"] for t in trades)
    wr = round(len(wins) / len(trades) * 100, 2) if trades else 0.0
    pf = round(win_total / loss_total, 4) if loss_total else (float("inf") if win_total else 0.0)
    fees = round(sum(t["fee"] for t in trades), 2)
    avg_trades = round(len(trades) / days_count, 3) if days_count else 0.0

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: (x["date"], x["entry_min"])):
        equity += t["rs_net"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "net_rs": round(net_rs, 2),
        "net_points": round(net_pts, 2),
        "profit_factor": pf,
        "max_drawdown_rs": round(max_dd, 2),
        "fees_rs": fees,
        "avg_trades_per_day": avg_trades,
    }

def main():
    parser = argparse.ArgumentParser(description="Marni S1 Turn-Up Trailing SL 7-Year Backtest")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Run 5-day smoke test only")
    parser.add_argument("--no-fees", action="store_true", help="Disable brokerage & slippage fees")
    args = parser.parse_args()

    include_fees = not args.no_fees
    print(f"\n{'='*125}")
    print(f"MARNI S1 TURN-UP TRIGGER + TRAILING SL (UNLIMITED PROFIT) — 7-YEAR BACKTEST (2020 - 2026)")
    print(f"Date Range: {args.start} to {args.end} | Daily Max Loss: Rs {DAILY_MAX_LOSS_RS:,.0f} | Fees: {'YES' if include_fees else 'NO'}")
    print(f"{'='*125}")

    spot_all = source.load_spot()
    opt_map = source.option_day_files(args.start, args.end)
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days[:5] if args.smoke else all_days

    print(f"Running on {len(days)} trading days {'(SMOKE TEST — 5 DAYS ONLY)' if args.smoke else ''}...")
    previous = {day: max((c for c in all_days if c < day), default="") for day in days}

    tasks = [
        (
            day,
            opt_map[day],
            opt_map.get(previous[day], ""),
            include_fees,
        )
        for day in days
    ]

    t0 = time.time()
    all_trades = []

    if args.smoke or args.workers == 1:
        init_worker(spot_all)
        for task in tasks:
            day_trades = process_day(task)
            all_trades.extend(day_trades)
    else:
        with Pool(processes=args.workers, initializer=init_worker, initargs=(spot_all,)) as pool:
            for day_trades in pool.imap_unordered(process_day, tasks, chunksize=1):
                all_trades.extend(day_trades)

    elapsed = time.time() - t0
    st = compute_stats(all_trades, len(days))

    print(f"\n{'='*125}")
    print(f"MARNI S1 TURN-UP TRIGGER + TRAILING SL (UNLIMITED PROFIT) — 7-YEAR RESULTS")
    print(f"{'='*125}")
    print(f"Total Trading Days:         {len(days):,d} days")
    print(f"Total Qualified Trades:     {len(all_trades):,d} ({st['avg_trades_per_day']:.2f} trades/day)")
    print(f"Winning Trades:             {st['wins']:,d} / {len(all_trades):,d} ({st['win_rate']:.2f}% Win Rate)")
    print(f"Total Realized Points:      {st['net_points']:+,.2f} pts")
    print(f"Profit Factor:              {st['profit_factor']:.2f}")
    print(f"Brokerage & Slippage:       Rs {st['fees_rs']:,.2f}")
    print(f"Max Drawdown:               Rs {st['max_drawdown_rs']:,.2f}")
    print(f"Net Realized P&L:           Rs {st['net_rs']:+,.2f}")
    print(f"Backtest Elapsed Time:      {elapsed:.2f}s")
    print(f"{'='*125}\n")

    # Year-by-Year Breakdown
    by_year = defaultdict(list)
    for t in all_trades:
        by_year[t["date"][:4]].append(t)

    print(f"{'Year':6s} | {'Trades':8s} | {'Win Rate':9s} | {'Points':12s} | {'Profit Factor':14s} | {'Max DD (Rs)':14s} | {'Net Realized P&L (Rs)':22s}")
    print("-" * 95)
    for y in sorted(by_year.keys()):
        y_trades = by_year[y]
        yst = compute_stats(y_trades, len(set(t["date"] for t in y_trades)))
        print(f"{y:6s} | {yst['trades']:8d} | {yst['win_rate']:8.1f}% | {yst['net_points']:+11.2f}p | {yst['profit_factor']:13.2f} | Rs {yst['max_drawdown_rs']:11,.2f} | Rs {yst['net_rs']:+19,.2f}")
    print("-" * 95)

    # Breakdown by Timeframe
    by_tf = defaultdict(list)
    for t in all_trades:
        by_tf[t["tf"]].append(t)

    print(f"\n{'='*95}")
    print("PERFORMANCE BREAKDOWN BY ENTRY TIMEFRAME (1m, 2m, 3m, 5m)")
    print(f"{'='*95}")
    for tf_key in ["1m", "2m", "3m", "5m"]:
        tf_trades = by_tf[tf_key]
        if tf_trades:
            tf_st = compute_stats(tf_trades, len(days))
            print(f"TF {tf_key:2s} | Trades: {tf_st['trades']:5d} | WR: {tf_st['win_rate']:5.1f}% | Points: {tf_st['net_points']:+9.2f}p | PF: {tf_st['profit_factor']:5.2f} | Net Rs: Rs {tf_st['net_rs']:+12,.2f}")
    print(f"{'='*95}\n")

    # Save to JSON
    out_path = Path("artifacts/f6_hybrid/marni_s1_turnup_7y_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"stats": st, "trades": all_trades}, f, indent=2)
    print(f"Saved complete 7-year trade ledger to: {out_path}")

if __name__ == "__main__":
    main()
