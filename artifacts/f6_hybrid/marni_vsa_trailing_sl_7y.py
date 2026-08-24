"""
MARNI VSA ENGINE — DYNAMIC TRAILING STOP LOSS & UNLIMITED PROFIT 7-YEAR BACKTEST (2020 - 2026)
=============================================================================================
Strategy: Marni VSA with Dynamic Trailing Stop Loss & Adaptive Profit Locking
Tests:
  1. Fixed Baseline: TP = 0.786 Retest, SL = 1.155 Ext
  2. Trailing SL (+10pt Gain -> +5pt SL Trail, Unlimited TP)
  3. Tight Trailing SL (+5pt Gain -> +5pt SL Trail, Unlimited TP)
  4. Adaptive Span Trailing (0.25x Opt Span Step -> 0.25x Opt Span Trail)
  5. S1 Turn-Up Trigger + Trailing SL
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.causal_live_parity_research import IncrementalATR

LOT_SIZE = 65
SESSION_START = 555
SESSION_END = 900
DAY_LAST = 930
CONSECUTIVE_LOSS_LIMIT = 4
SYMBOL_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")
GLOBAL_SPOT = {}

def init_worker(spot):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot
    source.GLOBAL_CACHE = {}

class IncrementalStochastic:
    def __init__(self, k_period: int = 9, d_period: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.highs = deque(maxlen=k_period)
        self.lows = deque(maxlen=k_period)
        self.closes = deque(maxlen=k_period)
        self.k_history = deque(maxlen=d_period)

    def update(self, high: float, low: float, close: float) -> Tuple[Optional[float], Optional[float]]:
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

        if len(self.highs) < self.k_period:
            return None, None

        highest_high = max(self.highs)
        lowest_low = min(self.lows)

        if highest_high == lowest_low:
            fast_k = 50.0
        else:
            fast_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100.0

        self.k_history.append(fast_k)
        if len(self.k_history) < self.d_period:
            return fast_k, None

        slow_d = sum(self.k_history) / float(self.d_period)
        return fast_k, slow_d

class UTBotState:
    def __init__(self, key: float = 1.0, period: int = 10):
        self.key = key
        self.atr = IncrementalATR(period)
        self.trailing_stop = 0.0
        self.previous_source = None
        self.position = 0

    def update(self, candle: Candle, source_close: float | None = None) -> str:
        source_price = candle.close if source_close is None else source_close
        atr = self.atr.update(candle.high, candle.low, candle.close)
        previous_source = self.previous_source
        self.previous_source = source_price

        if previous_source is None or atr is None or atr == 0.0:
            return "none"

        n_loss = self.key * atr
        if source_price > self.trailing_stop:
            proposed = source_price - n_loss
            self.trailing_stop = max(self.trailing_stop, proposed) if previous_source > self.trailing_stop else proposed
        else:
            proposed = source_price + n_loss
            self.trailing_stop = min(self.trailing_stop, proposed) if previous_source < self.trailing_stop else proposed

        if previous_source <= self.trailing_stop and source_price > self.trailing_stop:
            self.position = 1
        elif previous_source >= self.trailing_stop and source_price < self.trailing_stop:
            self.position = -1

        return "green" if self.position == 1 else ("red" if self.position == -1 else "none")

class StrictHTFBiasState:
    def __init__(self, period: int = 15, linreg_len: int = 11, ut_key: float = 1.0, ut_period: int = 10):
        self.period = period
        self.linreg_len = linreg_len
        self.buf: List[Candle] = []
        self.ha_prev: Optional[Candle] = None
        self.ha_closes: deque[float] = deque(maxlen=linreg_len + 5)
        self.ut = UTBotState(key=ut_key, period=ut_period)
        self.ut_color = "none"
        self.current_linreg_plot: Optional[float] = None
        self.current_ha_close: Optional[float] = None

    def _calc_linreg(self) -> Optional[float]:
        if len(self.ha_closes) < self.linreg_len:
            return None
        closes = list(self.ha_closes)[-self.linreg_len:]
        x = list(range(self.linreg_len))
        x_mean = (self.linreg_len - 1) / 2.0
        y_mean = sum(closes) / float(self.linreg_len)
        denom = sum((xi - x_mean) ** 2 for xi in x)
        if denom == 0.0:
            return y_mean
        numer = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, closes))
        slope = numer / denom
        intercept = y_mean - slope * x_mean
        return intercept + slope * (self.linreg_len - 1)

    def update_1m(self, c: Candle) -> None:
        self.buf.append(c)
        if len(self.buf) == self.period:
            agg_open = self.buf[0].open
            agg_high = max(b.high for b in self.buf)
            agg_low = min(b.low for b in self.buf)
            agg_close = self.buf[-1].close
            self.buf.clear()

            if self.ha_prev is None:
                ha_open = (agg_open + agg_close) / 2.0
                ha_close = (agg_open + agg_high + agg_low + agg_close) / 4.0
                ha_high = agg_high
                ha_low = agg_low
            else:
                ha_close = (agg_open + agg_high + agg_low + agg_close) / 4.0
                ha_open = (self.ha_prev.open + self.ha_prev.close) / 2.0
                ha_high = max(agg_high, ha_open, ha_close)
                ha_low = min(agg_low, ha_open, ha_close)

            ha_c = Candle(ha_open, ha_high, ha_low, ha_close, minute=c.minute)
            self.ha_prev = ha_c
            self.ha_closes.append(ha_close)
            self.current_ha_close = ha_close
            self.current_linreg_plot = self._calc_linreg()
            self.ut_color = self.ut.update(ha_c)

    def snapshot(self) -> dict:
        return {
            "linreg_plot": self.current_linreg_plot,
            "ut_color": self.ut_color,
            "ha_close": self.current_ha_close,
        }

def spot_row(spot_dict, idx):
    return {
        "minute": int(spot_dict["min"][idx]),
        "open": float(spot_dict["open"][idx]),
        "high": float(spot_dict["high"][idx]),
        "low": float(spot_dict["low"][idx]),
        "close": float(spot_dict["close"][idx]),
    }

def active_strikes(spot_dict, minute, side):
    idx = 0
    for i, m in enumerate(spot_dict["min"]):
        if int(m) == minute:
            idx = i
            break
    price = float(spot_dict["close"][idx])
    atm = int(round(price / 50.0) * 50)
    return atm - 100 if side == "CE" else atm + 100

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
    (
        day,
        opt_file,
        prev_day,
        min_span,
        mode, # "fixed", "trail_10_5", "trail_5_5", "trail_span_25", "trail_s1_turnup"
        include_fees,
    ) = task

    spot = GLOBAL_SPOT.get(day)
    if spot is None or len(spot["min"]) < 50:
        return []

    current = source.cached_option(str(opt_file))
    if current is None:
        return []
    frame, groups, prefix = current

    active_keys = set()
    for index, minute in enumerate(spot["min"]):
        if SESSION_START <= minute <= DAY_LAST:
            atm = int(round(float(spot["close"][index]) / 50.0) * 50)
            active_keys.add(("CE", atm - 100))
            active_keys.add(("PE", atm + 100))

    bars = {}
    symbol_by_key = {}

    for sym in groups:
        m = SYMBOL_RE.match(sym)
        if not m:
            continue
        strike = int(m.group(2))
        side = m.group(3)
        key = (side, strike)
        if key not in active_keys:
            continue

        symbol_by_key[key] = sym
        r_list = option_rows(frame, groups, sym)
        bars[key] = {r["minute"]: r for r in r_list}

    htf = StrictHTFBiasState(period=15, linreg_len=11)
    ut_1m = UTBotState(key=1.0, period=10)
    s1_stoch = IncrementalStochastic(k_period=9, d_period=3)

    if prev_day and prev_day in GLOBAL_SPOT:
        prev_spot = GLOBAL_SPOT[prev_day]
        for i in range(len(prev_spot["min"])):
            prow = spot_row(prev_spot, i)
            pc = Candle(prow["open"], prow["high"], prow["low"], prow["close"], minute=prow["minute"])
            htf.update_1m(pc)
            ut_1m.update(pc)
            s1_stoch.update(prow["high"], prow["low"], prow["close"])

    history = []
    ce_setups = []
    pe_setups = []
    events = []

    open_high = float(spot["high"][0])
    open_low = float(spot["low"][0])
    prev_s1 = None

    for idx in range(len(spot["min"])):
        row = spot_row(spot, idx)
        m = row["minute"]
        c = Candle(row["open"], row["high"], row["low"], row["close"], minute=m)
        
        htf.update_1m(c)
        htf_snap = htf.snapshot()
        col = ut_1m.update(c)
        s1_k, _ = s1_stoch.update(row["high"], row["low"], row["close"])
        history.append((c, col, s1_k))

        # Check 09:15 open anchor drop setup at 09:32
        if m == 572:
            trough_low = min(history[i][0].low for i in range(len(history)))
            span = open_high - trough_low
            if span >= min_span:
                pe_setups.append({
                    "origin_high": open_high,
                    "peak_low": trough_low,
                    "span": span,
                    "triggered": False,
                })

        # 3-Phase Bullish (CE): 1 Red -> >= 5 Green -> 1 Red
        if col == "red" and len(history) >= 7:
            g_count = 0
            k = len(history) - 2
            while k >= 0 and history[k][1] == "green":
                g_count += 1
                k -= 1
            if g_count >= 5 and k >= 0 and history[k][1] == "red":
                pat = [history[i][0] for i in range(k, len(history))]
                pk = max(p.high for p in pat)
                orig = min(p.low for p in pat)
                sp = pk - orig
                if sp >= min_span:
                    ce_setups.append({
                        "peak_high": pk,
                        "origin_low": orig,
                        "span": sp,
                        "triggered": False,
                    })

        # 3-Phase Bearish (PE): 1 Green -> >= 5 Red -> 1 Green
        if col == "green" and len(history) >= 7:
            r_count = 0
            k = len(history) - 2
            while k >= 0 and history[k][1] == "red":
                r_count += 1
                k -= 1
            if r_count >= 5 and k >= 0 and history[k][1] == "green":
                pat = [history[i][0] for i in range(k, len(history))]
                orig = max(p.high for p in pat)
                pk = min(p.low for p in pat)
                sp = orig - pk
                if sp >= min_span:
                    pe_setups.append({
                        "origin_high": orig,
                        "peak_low": pk,
                        "span": sp,
                        "triggered": False,
                    })

        # Determine active strikes
        ce_strike = active_strikes(spot, m, "CE")
        pe_strike = active_strikes(spot, m, "PE")
        ce_key = ("CE", ce_strike)
        pe_key = ("PE", pe_strike)

        # S1 Turn-Up Trigger variant
        s1_turn_up = (s1_k is not None and prev_s1 is not None and s1_k > prev_s1 and s1_k <= 30.0)
        s1_turn_down = (s1_k is not None and prev_s1 is not None and s1_k < prev_s1 and s1_k >= 70.0)

        # Check CE Touches in Pocket [0.618 - 0.786] on Spot Index
        valid_ce = []
        for s in ce_setups:
            if s.get("triggered", False):
                continue
            pk, orig, sp = s["peak_high"], s["origin_low"], s["span"]
            if c.low < orig - 0.25 * sp:
                continue
            f618 = pk - 0.618 * sp
            f786 = pk - 0.786 * sp
            in_zone = (c.low <= f618 + 0.5) and (c.high >= f786 - 0.5)

            linreg_p = htf_snap.get("linreg_plot")
            ut_col = htf_snap.get("ut_color")

            if in_zone:
                trigger_ok = s1_turn_up if mode == "trail_s1_turnup" else True
                if trigger_ok and linreg_p is not None and c.close > linreg_p and ut_col == "green":
                    events.append({
                        "minute": m,
                        "side": "CE",
                        "strike": ce_strike,
                        "symbol": symbol_by_key.get(ce_key, f"NIFTY{ce_strike}CE"),
                        "span": sp,
                    })
                    s["triggered"] = True
                    continue
            valid_ce.append(s)
        ce_setups = valid_ce

        # Check PE Touches in Pocket [0.618 - 0.786] on Spot Index
        valid_pe = []
        for s in pe_setups:
            if s.get("triggered", False):
                continue
            orig, pk, sp = s["origin_high"], s["peak_low"], s["span"]
            if c.high > orig + 0.25 * sp:
                continue
            f618 = pk + 0.618 * sp
            f786 = pk + 0.786 * sp
            in_zone = (c.high >= f618 - 0.5) and (c.low <= f786 + 0.5)

            linreg_p = htf_snap.get("linreg_plot")
            ut_col = htf_snap.get("ut_color")

            if in_zone:
                trigger_ok = s1_turn_down if mode == "trail_s1_turnup" else True
                if trigger_ok and linreg_p is not None and c.close < linreg_p and (ut_col == "red" or m <= 600):
                    events.append({
                        "minute": m,
                        "side": "PE",
                        "strike": pe_strike,
                        "symbol": symbol_by_key.get(pe_key, f"NIFTY{pe_strike}PE"),
                        "span": sp,
                    })
                    s["triggered"] = True
                    continue
            valid_pe.append(s)
        pe_setups = valid_pe
        prev_s1 = s1_k

    # Simulate trades with Trailing SL
    trades = []
    consecutive_losses = 0

    for ev in events:
        m = ev["minute"]
        if m < SESSION_START or m >= SESSION_END:
            continue
        if consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            continue

        side = ev["side"]
        strike = ev["strike"]
        key = (side, strike)

        if key not in bars or m not in bars[key]:
            continue

        opt_entry = bars[key][m]["close"]
        opt_span = ev["span"] * 0.5
        
        # SL and TP definitions based on mode
        sl_dist = opt_span * 0.369 # standard initial stop distance
        current_sl = opt_entry - sl_dist
        
        if mode == "fixed":
            tp_price = opt_entry + (opt_span * 0.786)
            trail_step = None
            trail_amt = None
        elif mode in ("trail_10_5", "trail_s1_turnup"):
            tp_price = None # Unlimited upside
            trail_step = 10.0
            trail_amt = 5.0
        elif mode == "trail_5_5":
            tp_price = None # Unlimited upside
            trail_step = 5.0
            trail_amt = 5.0
        elif mode == "trail_span_25":
            tp_price = None # Unlimited upside
            trail_step = max(5.0, opt_span * 0.25)
            trail_amt = max(3.0, opt_span * 0.20)
        elif mode == "fixed_plus_trail":
            tp_price = opt_entry + (opt_span * 0.786)
            trail_step = 10.0
            trail_amt = 5.0
        else:
            tp_price = opt_entry + (opt_span * 0.786)
            trail_step = None
            trail_amt = None

        trail_steps_taken = 0
        exit_fill, exit_m, rsn = None, None, ""

        for bar_m in range(m + 1, DAY_LAST + 1):
            if bar_m not in bars[key]:
                continue
            b = bars[key][bar_m]
            h, l, cl = b["high"], b["low"], b["close"]

            # Dynamic Trailing SL update
            if trail_step is not None:
                gain = cl - opt_entry
                steps = int(gain / trail_step)
                if steps > trail_steps_taken:
                    current_sl += (steps - trail_steps_taken) * trail_amt
                    trail_steps_taken = steps

            # Check TP / SL hits
            if tp_price is not None and l <= current_sl and h >= tp_price:
                exit_fill, exit_m, rsn = current_sl, bar_m, "SL"
                break
            elif tp_price is not None and h >= tp_price:
                exit_fill, exit_m, rsn = tp_price, bar_m, "TP"
                break
            elif l <= current_sl:
                exit_fill, exit_m, rsn = current_sl, bar_m, "TRAIL_SL" if trail_steps_taken > 0 else "SL"
                break
            elif bar_m >= SESSION_END:
                exit_fill, exit_m, rsn = cl, bar_m, "EOD"
                break

        if exit_fill is not None:
            slip = SLIPPAGE_PTS if include_fees else 0.0
            entry_f = opt_entry + slip
            exit_f = exit_fill - slip
            pts = round(exit_f - entry_f, 2)
            fee = trade_cost(entry_f, exit_f, BROKERAGE_PER_ORDER) if include_fees else 0.0
            rs_net = round(pts * LOT_SIZE - fee, 2)

            if rs_net > 0:
                consecutive_losses = 0
            else:
                consecutive_losses += 1

            trades.append({
                "date": day,
                "entry_min": m,
                "exit_min": exit_m,
                "side": side,
                "strike": strike,
                "symbol": ev["symbol"],
                "span": ev["span"],
                "entry": entry_f,
                "exit": exit_f,
                "reason": rsn,
                "points": pts,
                "fee": fee,
                "rs_net": rs_net,
            })

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
    parser = argparse.ArgumentParser(description="Marni VSA Trailing SL Multi-Year Backtest")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Run 5-day smoke test only")
    parser.add_argument("--no-fees", action="store_true", help="Disable brokerage & slippage fees")
    parser.add_argument("--min-span", type=float, default=20.0, help="Minimum impulse span in points (default: 20.0)")
    args = parser.parse_args()

    include_fees = not args.no_fees
    print(f"\n{'='*120}")
    print(f"MARNI VSA ENGINE — TRAILING SL & UNLIMITED PROFIT MULTI-YEAR STUDY (2020 - 2026)")
    print(f"Date Range: {args.start} to {args.end} | Min Span: {args.min_span} pts | Fees: {'YES' if include_fees else 'NO'}")
    print(f"{'='*120}")

    spot_all = source.load_spot()
    opt_map = source.option_day_files(args.start, args.end)
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days[:5] if args.smoke else all_days

    print(f"Running on {len(days)} trading days {'(SMOKE TEST — 5 DAYS ONLY)' if args.smoke else ''}...")
    previous = {day: max((c for c in all_days if c < day), default="") for day in days}

    modes = [
        ("fixed", "Fixed Baseline (0.786 TP / 1.155 SL)"),
        ("trail_10_5", "Trailing SL (+10pt Gain -> +5pt Trail, Unlimited TP)"),
        ("trail_5_5", "Tight Trailing SL (+5pt Gain -> +5pt Trail, Unlimited TP)"),
        ("trail_span_25", "Adaptive Span Trailing (0.25x Span Step/Trail)"),
        ("trail_s1_turnup", "S1 Turn-Up Trigger + Trailing SL (+10/+5)"),
    ]

    all_results = {}

    for mode_key, mode_label in modes:
        tasks = [
            (
                day,
                opt_map[day],
                previous[day],
                args.min_span,
                mode_key,
                include_fees,
            )
            for day in days
        ]

        t0 = time.time()
        mode_trades = []

        if args.smoke or args.workers == 1:
            init_worker(spot_all)
            for task in tasks:
                day_trades = process_day(task)
                mode_trades.extend(day_trades)
        else:
            with Pool(processes=args.workers, initializer=init_worker, initargs=(spot_all,)) as pool:
                for day_trades in pool.imap_unordered(process_day, tasks, chunksize=1):
                    mode_trades.extend(day_trades)

        elapsed = time.time() - t0
        st = compute_stats(mode_trades, len(days))
        all_results[mode_key] = {"label": mode_label, "stats": st, "trades": mode_trades}
        print(f"Completed {mode_label:55s} in {elapsed:.2f}s | Trades: {st['trades']:5d} | WR: {st['win_rate']:5.1f}% | Net Rs: Rs {st['net_rs']:+12,.2f} | PF: {st['profit_factor']:.2f}")

    print(f"\n{'='*130}")
    print(f"COMPARATIVE SUMMARY LEADERBOARD ({args.start[:4]} - {args.end[:4]})")
    print(f"{'='*130}")
    print(f"{'Strategy Configuration':55s} | {'Trades':6s} | {'Win Rate':9s} | {'Net Points':12s} | {'Profit Factor':14s} | {'Max DD (Rs)':14s} | {'Net Realized P&L (Rs)':22s}")
    print("-" * 135)
    for mk, res in sorted(all_results.items(), key=lambda x: x[1]["stats"]["net_rs"], reverse=True):
        st = res["stats"]
        print(f"{res['label']:55s} | {st['trades']:6d} | {st['win_rate']:8.1f}% | {st['net_points']:+11.2f}p | {st['profit_factor']:13.2f} | Rs {st['max_drawdown_rs']:11,.2f} | Rs {st['net_rs']:+19,.2f}")
    print("-" * 135)

    # Save to json
    out_path = Path("artifacts/f6_hybrid/marni_vsa_trailing_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: {"label": v["label"], "stats": v["stats"]} for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved comparison JSON to: {out_path}")

if __name__ == "__main__":
    main()
