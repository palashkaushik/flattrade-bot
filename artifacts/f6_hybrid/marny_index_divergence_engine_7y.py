"""Marny Core Index Engine with Mirrored Divergence Filter (7 Years: 2020-2026 | 1574 Days).

Mirrored Index Architecture:
1. CALL (CE) Execution:
   - Bullish Impulse on Index (1 Red -> >= 5 Green -> 1 Red).
   - Index 0.786 Retracement Touch (entry = peak_high - 0.786 * span).
   - Filter 1: 5m Index HTF Gate (5m HA Close > LinReg Plot & 5m UT == "green").
   - Filter 2: Bullish Trough Divergence on Index (Lower Low Index Price + Higher Low S1).
   - Action: BUY ATM-100 CE Option contract.

2. PUT (PE) Execution (Mirrored):
   - Bearish Impulse on Index (1 Green -> >= 5 Red -> 1 Green).
   - Index 0.786 Retracement Touch (entry = trough_low + 0.786 * span).
   - Filter 1: 5m Index HTF Gate (5m HA Close < LinReg Plot & 5m UT == "red").
   - Filter 2: Bearish Peak Divergence on Index (Higher High Index Price + Lower High S1).
   - Action: BUY ATM+100 PE Option contract.

3. Option Risk & Accounting:
   - Evaluated across all 1,574 trading days (2020-01-01 to 2026-05-05).
   - SL & TP: Marny Core Option Fibonacci Levels (TP=0.0 Peak Retest, SL=1.155 Ext) and Fixed/ATR options.
   - Full institutional costs + Zero Fee toggles, Daily Loss = -30 pts & No Cap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost
from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.indicators.divergence import DivergenceEngine
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from artifacts.f6_hybrid.marni_fib_5y_fast import (
    UTBotState, HeikinAshiState, StrictHTFBiasState, IncrementalATR
)

LOT_SIZE = 65
SESSION_START = 560
SESSION_END = 900
DAY_LAST = 930
GLOBAL_SPOT = {}


def init_worker(spot):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot
    source.GLOBAL_CACHE = {}


class IndexFibDivergenceTracker:
    """Tracks both Bullish (CE) and Bearish (PE) Marny 3-phase swings on Index with Mirrored Divergence."""
    def __init__(self, min_span: float = 15.0, min_candles: int = 5):
        self.min_span = min_span
        self.min_candles = min_candles
        self.ut = UTBotState(key=1.0, period=10)
        self.history = []  # (Candle, ut_color)
        self.ce_setups = []  # Bullish impulse setups
        self.pe_setups = []  # Bearish impulse setups

    def push(self, candle: Candle, htf_bias: dict) -> list[dict]:
        col = self.ut.update(candle)
        self.history.append((candle, col))

        # Check for completed Bullish Impulse (CE): 1 RED -> >= 5 GREEN -> 1 RED
        if col == "red" and len(self.history) >= self.min_candles + 2:
            green_count = 0
            k = len(self.history) - 2
            while k >= 0 and self.history[k][1] == "green":
                green_count += 1
                k -= 1
            if green_count >= self.min_candles and k >= 0 and self.history[k][1] == "red":
                pattern = [self.history[i][0] for i in range(k, len(self.history))]
                peak_high = max(c.high for c in pattern)
                origin_low = min(c.low for c in pattern)
                span = peak_high - origin_low
                if span >= self.min_span:
                    self.ce_setups.append({
                        "peak_high": peak_high,
                        "origin_low": origin_low,
                        "span": span,
                    })

        # Check for completed Bearish Impulse (PE): 1 GREEN -> >= 5 RED -> 1 GREEN (Mirrored)
        if col == "green" and len(self.history) >= self.min_candles + 2:
            red_count = 0
            k = len(self.history) - 2
            while k >= 0 and self.history[k][1] == "red":
                red_count += 1
                k -= 1
            if red_count >= self.min_candles and k >= 0 and self.history[k][1] == "green":
                pattern = [self.history[i][0] for i in range(k, len(self.history))]
                origin_high = max(c.high for c in pattern)
                peak_low = min(c.low for c in pattern)
                span = origin_high - peak_low
                if span >= self.min_span:
                    self.pe_setups.append({
                        "origin_high": origin_high,
                        "peak_low": peak_low,
                        "span": span,
                    })

        events = []

        # Evaluate CE Touches (Bullish Retracement)
        valid_ce = []
        for s in self.ce_setups:
            peak, origin, span = s["peak_high"], s["origin_low"], s["span"]
            if candle.low < origin - 0.25 * span:
                continue
            entry_level = peak - 0.786 * span
            if candle.high >= entry_level - 1.5 and candle.low <= entry_level + 1.5:
                if htf_bias.get("bullish", False):
                    events.append({
                        "side": "CE",
                        "minute": candle.minute,
                        "index_entry": candle.close,
                        "index_peak": peak,
                        "index_origin": origin,
                        "index_span": span,
                    })
                    continue
            valid_ce.append(s)
        self.ce_setups = valid_ce

        # Evaluate PE Touches (Bearish Retracement - Mirrored)
        valid_pe = []
        for s in self.pe_setups:
            origin, peak, span = s["origin_high"], s["peak_low"], s["span"]
            if candle.high > origin + 0.25 * span:
                continue
            entry_level = peak + 0.786 * span
            if candle.high >= entry_level - 1.5 and candle.low <= entry_level + 1.5:
                if htf_bias.get("bearish", False):
                    events.append({
                        "side": "PE",
                        "minute": candle.minute,
                        "index_entry": candle.close,
                        "index_peak": peak,
                        "index_origin": origin,
                        "index_span": span,
                    })
                    continue
            valid_pe.append(s)
        self.pe_setups = valid_pe

        return events


def process_day_index_divergence(args):
    day, opt_path, prev_opt_path, p = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not opt_path:
        return []

    min_span = p.get("min_span", 15.0)
    include_fees = p.get("include_fees", True)
    use_marny_levels = p.get("use_marny_levels", True)
    fixed_sl = p.get("fixed_sl", 10.0)
    fixed_tp = p.get("fixed_tp", 15.0)
    daily_loss_pts = p.get("daily_loss_pts", -30.0)

    rec = source.cached_option(str(opt_path))
    if rec is None:
        return []
    df, groups, prefix = rec
    if prefix is None:
        return []

    spot_mins = spot["min"]
    spot_opens = spot["open"]
    spot_highs = spot["high"]
    spot_lows = spot["low"]
    spot_closes = spot["close"]

    # 1. State machines on Index
    index_5m = StrictHTFBiasState(period=5)
    index_fib = IndexFibDivergenceTracker(min_span=min_span, min_candles=5)
    index_div = DivergenceEngine(max_history=40, min_lookback=3, max_lookback=30)
    index_s1 = IncrementalStochastic(9, 3)

    # 2. Warmup on Spot
    prev_spot = GLOBAL_SPOT.get(p.get("prev_day", ""))
    if prev_spot is not None:
        p_mins = prev_spot["min"]
        p_opens = prev_spot["open"]
        p_highs = prev_spot["high"]
        p_lows = prev_spot["low"]
        p_closes = prev_spot["close"]
        for j in range(len(p_mins)):
            pc = Candle(float(p_opens[j]), float(p_highs[j]), float(p_lows[j]), float(p_closes[j]), minute=int(p_mins[j]))
            index_5m.update_1m(pc)
            index_fib.push(pc, index_5m.snapshot())
            s1_v = index_s1.push(pc.high, pc.low, pc.close)
            index_div.update(pc.close, s1_v, pc.low, pc.high)

    # 3. Current day evaluation on Spot
    index_events = []
    for j in range(len(spot_mins)):
        m = int(spot_mins[j])
        c = Candle(float(spot_opens[j]), float(spot_highs[j]), float(spot_lows[j]), float(spot_closes[j]), minute=m)
        index_5m.update_1m(c)
        htf = index_5m.snapshot()

        s1_v = index_s1.push(c.high, c.low, c.close)
        index_div.update(c.close, s1_v, c.low, c.high)

        has_bull_div = index_div.has_bullish_trough_divergence()
        has_bear_div = index_div.has_bearish_peak_divergence()

        raw_events = index_fib.push(c, htf)
        for ev in raw_events:
            if ev["side"] == "CE" and has_bull_div:
                index_events.append({**ev, "strike_offset": -100})
            elif ev["side"] == "PE" and has_bear_div:
                index_events.append({**ev, "strike_offset": 100})

    if not index_events:
        return []

    # 4. Map index events to Option Contracts
    all_events = []
    option_bars_by_key = {}

    for ev in index_events:
        m = ev["minute"]
        idx = np.searchsorted(spot_mins, m, side="right") - 1
        if idx < 0:
            continue
        spot_px = float(spot_closes[idx])
        atm = int(round(spot_px / 50) * 50)
        strike = atm + ev["strike_offset"]
        side = ev["side"]
        sym = f"{prefix}{strike}{side}"
        key = (side, strike)

        if key not in option_bars_by_key:
            sl = source.make_slice(df, groups, sym)
            if sl is None or len(sl["times"]) < 15:
                continue
            option_bars_by_key[key] = {}
            for j in range(len(sl["times"])):
                m_j = int(sl["times"][j])
                option_bars_by_key[key][m_j] = {
                    "minute": m_j,
                    "open": float(sl["open"][j]),
                    "high": float(sl["high"][j]),
                    "low": float(sl["low"][j]),
                    "close": float(sl["close"][j]),
                }

        bar = option_bars_by_key[key].get(m)
        if bar is not None:
            all_events.append({
                **ev,
                "symbol": sym,
                "key": key,
                "option_entry": bar["close"],
                "span": ev["index_span"] * 0.5,  # Option delta adjusted span (~0.5 delta)
                "peak_high": bar["close"] + (ev["index_span"] * 0.5 * 0.786),
                "origin_low": bar["close"] - (ev["index_span"] * 0.5 * 0.369),
            })

    # 5. Trade Simulation
    events_by_min = defaultdict(list)
    for ev in all_events:
        events_by_min[ev["minute"]].append(ev)

    trades = []
    pos = None
    dpnl = 0.0
    closs = 0
    shut = False

    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None and minute > pos["entry_min"]:
            bar = option_bars_by_key[pos["key"]].get(minute)
            if bar is not None:
                high, low, close = bar["high"], bar["low"], bar["close"]
                sl_px = pos["sl"]
                tp_px = pos["tp"]

                ex, rsn = None, ""
                if dpnl + (low - pos["entry"]) <= daily_loss_pts:
                    ex, rsn = pos["entry"] + (daily_loss_pts - dpnl), "DAILY_LOSS"
                    shut = True
                elif low <= sl_px and high >= tp_px:
                    ex, rsn = sl_px, "SL"
                elif high >= tp_px:
                    ex, rsn = tp_px, "TP"
                elif low <= sl_px:
                    ex, rsn = sl_px, "SL"
                elif minute >= SESSION_END:
                    ex, rsn = close, "EOD"

                if ex is not None:
                    slip = SLIPPAGE_PTS if include_fees else 0.0
                    brokerage = BROKERAGE_PER_ORDER if include_fees else 0.0
                    entry_fill = pos["entry"] + slip
                    exit_fill = ex - slip
                    pts = round(exit_fill - entry_fill, 2)
                    fee = trade_cost(entry_fill, exit_fill, brokerage) if include_fees else 0.0
                    net_rs = round(pts * LOT_SIZE - fee, 2)
                    trades.append({
                        "date": day,
                        "entry_min": pos["entry_min"],
                        "exit_min": minute,
                        "side": pos["side"],
                        "symbol": pos["symbol"],
                        "entry": entry_fill,
                        "exit": exit_fill,
                        "reason": rsn,
                        "points": pts,
                        "rs_net": net_rs,
                        "fee": fee,
                    })
                    dpnl += pts
                    closs = closs + 1 if net_rs <= 0 else 0
                    if closs >= 6 or dpnl <= daily_loss_pts:
                        shut = True
                    pos = None

        if pos is not None or shut or minute >= SESSION_END:
            continue

        for ev in events_by_min.get(minute, []):
            if pos is not None:
                break
            
            if use_marny_levels:
                tp_val = ev["peak_high"]
                sl_val = ev["origin_low"]
            else:
                sl_val = ev["option_entry"] - fixed_sl
                tp_val = ev["option_entry"] + fixed_tp

            pos = {
                **ev,
                "entry_min": minute,
                "entry": ev["option_entry"],
                "sl": sl_val,
                "tp": tp_val,
                "highest": ev["option_entry"],
            }

    return trades


def run_index_divergence_backtest(params, days_subset=None, workers=8):
    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days if days_subset is None else days_subset
    previous = {day: max((c for c in all_days if c < day), default="") for day in days}

    tasks = [
        (day, opt_map[day], opt_map.get(previous[day], ""), {**params, "prev_day": previous[day]})
        for day in days
    ]

    with Pool(processes=min(cpu_count(), workers), initializer=init_worker, initargs=(spot_all,)) as pool:
        all_day_trades = pool.map(process_day_index_divergence, tasks)

    all_trades = [t for day_trs in all_day_trades for t in day_trs]
    wins = [t for t in all_trades if t["rs_net"] > 0]
    losses = [t for t in all_trades if t["rs_net"] <= 0]
    loss_tot = abs(sum(t["rs_net"] for t in losses))
    win_tot = sum(t["rs_net"] for t in wins)
    net_rs = sum(t["rs_net"] for t in all_trades)
    net_pts = sum(t["points"] for t in all_trades)
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0.0
    pf = win_tot / loss_tot if loss_tot else 0.0
    fees = sum(t["fee"] for t in all_trades)

    equity, peak, max_dd = 0.0, 0.0, 0.0
    for t in sorted(all_trades, key=lambda x: (x["date"], x["entry_min"])):
        equity += t["rs_net"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr, 2),
        "net_rs": round(net_rs, 2),
        "net_points": round(net_pts, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown_rs": round(max_dd, 2),
        "fees_rs": round(fees, 2),
        "all_trades": all_trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Marny Core Index & Mirrored Divergence Filter Backtest")
    parser.add_argument("--smoke", action="store_true", help="Run 5-day smoke test")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPU workers")
    args = parser.parse_args()

    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days[:5] if args.smoke else all_days

    print("=" * 135)
    print(f"MARNY CORE INDEX + MIRRORED DIVERGENCE ENGINE (7 YEARS: 2020-2026 | {len(days)} DAYS)")
    print("CE: Bullish Impulse + Bullish Div | PE: Bearish Impulse + Bearish Div")
    print("=" * 135)

    spans = [0.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
    
    print(f"{'Configuration':72s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'Profit Factor':13s} | {'Max Drawdown':14s} | {'Total Fees':12s}")
    print("-" * 135)

    for span in spans:
        for inc_fees in [False, True]:
            fee_label = "With Fees" if inc_fees else "Zero Fees"
            label = f"Index Marny + Mirrored Div | Span>={span:4.1f} | Marny TP/SL | {fee_label}"
            params = {
                "min_span": span,
                "include_fees": inc_fees,
                "use_marny_levels": True,
                "daily_loss_pts": -1e9,
            }
            res = run_index_divergence_backtest(params, days, args.workers)
            print(f"{label:72s} | {res['trades']:7d} | {res['win_rate']:7.1f}% | {res['net_points']:+10.2f} | Rs {res['net_rs']:+13.2f} | {res['profit_factor']:13.2f} | Rs {res['max_drawdown_rs']:11.2f} | Rs {res['fees_rs']:10.2f}")

    print("=" * 135)


if __name__ == "__main__":
    main()
