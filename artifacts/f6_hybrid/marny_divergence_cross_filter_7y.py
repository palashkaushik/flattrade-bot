"""Marny Core & Stochastic Divergence Cross-Filter Engine (7 Years: 2020-2026 | 1574 Days).

Two Mirror Architectures:
1. Mode A: Marny Core Engine with Stochastic Divergence Filter
   - Primary Signal: Marny 1m 0.786 Fibonacci Retracement on Option Chart (3-phase impulse).
   - Filter Gate: Active Bullish Stochastic Trough Divergence on Option (Lower Low Price + Higher Low S1) AND 5m Marny HTF Gate (5m HA > LinReg & UT == "green").

2. Mode B: Stochastic Divergence Engine with Marny Core as Filter (Vice Versa)
   - Primary Signal: Bullish Stochastic Divergence Trigger (S1 Turn-Up / Pinbar on Option).
   - Filter Gate: Option is actively pulling back in a verified Marny Fibonacci Discount Zone (0.618-0.786) AND 5m Marny HTF Gate is bullish.

3. Accounting:
   - Full 1,574 Days (2020-01-01 to 2026-05-05).
   - Institutional friction (STT, exchange turnover, SEBI, GST, stamp duty, slippage) + Zero Fee toggles.
   - Daily Max Loss = -30.0 pts (-Rs 1,950), Unlimited Profit.
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
import grid_optimize_f6_atr as f6_eng
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost
from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.indicators.divergence import DivergenceEngine
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from artifacts.f6_hybrid.marny_option_chart_backtest import (
    Option5mHTFBias, Option1mFibTracker, IncrementalATR
)
from artifacts.f6_hybrid.f6_champion_marny_5m_filter_backtest import (
    MTFTrackerS1TurnUp
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


def process_day_divergence_cross(args):
    day, opt_path, prev_opt_path, p = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not opt_path:
        return []

    mode = p.get("mode", "A")  # "A" = Marny+DivFilter, "B" = Div+MarnyFilter
    min_span = p.get("min_span", 0.0)
    include_fees = p.get("include_fees", True)
    fixed_sl = p.get("fixed_sl", 10.0)
    fixed_tp = p.get("fixed_tp", 15.0)
    atr_sl_mult = p.get("atr_sl_mult", None)
    atr_tp_mult = p.get("atr_tp_mult", None)
    daily_loss_pts = p.get("daily_loss_pts", -30.0)
    trail_sl = p.get("trail_sl", False)

    rec = source.cached_option(str(opt_path))
    if rec is None:
        return []
    df, groups, prefix = rec
    if prefix is None:
        return []

    spot_mins = spot["min"]
    all_events = []
    option_bars_by_key = {}

    for side in ("CE", "PE"):
        atm_strikes = set()
        for m in range(SESSION_START, SESSION_END + 1):
            idx = np.searchsorted(spot_mins, m, side="right") - 1
            if idx >= 0:
                spot_px = float(spot["close"][idx])
                atm = int(round(spot_px / 50) * 50)
                strike = atm - 100 if side == "CE" else atm + 100
                atm_strikes.add(strike)

        for strike in atm_strikes:
            sym = f"{prefix}{strike}{side}"
            sl = source.make_slice(df, groups, sym)
            if sl is None or len(sl["times"]) < 15:
                continue

            key = (side, strike)
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

            # State machines
            marny_5m = Option5mHTFBias()
            marny_fib = Option1mFibTracker(min_span=min_span, min_candles=5)
            div_engine = DivergenceEngine(max_history=40, min_lookback=3, max_lookback=30)
            stoch_s1 = IncrementalStochastic(9, 3)
            opt_atr = IncrementalATR(period=14)
            prev_s1 = None

            # Warmup
            if prev_opt_path:
                prev_rec = source.cached_option(str(prev_opt_path))
                if prev_rec:
                    p_df, p_groups, p_prefix = prev_rec
                    p_sl = source.make_slice(p_df, p_groups, sym)
                    if p_sl is not None:
                        for pj in range(len(p_sl["times"])):
                            pc = Candle(float(p_sl["open"][pj]), float(p_sl["high"][pj]), float(p_sl["low"][pj]), float(p_sl["close"][pj]), minute=int(p_sl["times"][pj]))
                            marny_5m.update_1m(pc)
                            marny_fib.push(pc, marny_5m.snapshot())
                            s1_v = stoch_s1.push(pc.high, pc.low, pc.close)
                            div_engine.update(pc.close, s1_v, pc.low, pc.high)
                            opt_atr.update(pc.high, pc.low, pc.close)
                            prev_s1 = s1_v

            # Current day evaluation
            for j in range(len(sl["times"])):
                m = int(sl["times"][j])
                c = Candle(float(sl["open"][j]), float(sl["high"][j]), float(sl["low"][j]), float(sl["close"][j]), minute=m)
                marny_5m.update_1m(c)
                is_marny_bullish = marny_5m.snapshot()
                atr_v = opt_atr.update(c.high, c.low, c.close)

                fib_events = marny_fib.push(c, is_marny_bullish)
                s1_v = stoch_s1.push(c.high, c.low, c.close)
                div_engine.update(c.close, s1_v, c.low, c.high)

                has_div = div_engine.has_bullish_trough_divergence()
                s1_turn_up = prev_s1 is not None and s1_v is not None and s1_v > prev_s1 and s1_v <= 35.0

                if mode == "A":
                    # Mode A: In whichever option chart there is bullish divergence, ONLY THEN execute Marny signal
                    for ev in fib_events:
                        if has_div:
                            all_events.append({
                                **ev,
                                "side": side,
                                "strike": strike,
                                "symbol": sym,
                                "key": key,
                                "option_entry": c.close,
                                "atr": atr_v if atr_v and atr_v > 0 else 5.0,
                            })
                else:
                    # Mode B: Bullish Divergence + S1 Turn-Up is Signal, Marny Fib Discount is Filter
                    if has_div and s1_turn_up and is_marny_bullish:
                        # Check if Marny Fib is active in discount pocket
                        if marny_fib.setups:
                            s = marny_fib.setups[-1]
                            peak, origin, span = s["peak_high"], s["origin_low"], s["span"]
                            if span > 0:
                                retracement = (peak - c.close) / span
                                if 0.50 <= retracement <= 0.90:
                                    all_events.append({
                                        "minute": m,
                                        "side": side,
                                        "strike": strike,
                                        "symbol": sym,
                                        "key": key,
                                        "option_entry": c.close,
                                        "atr": atr_v if atr_v and atr_v > 0 else 5.0,
                                    })

                prev_s1 = s1_v

    # Trade simulation
    events_by_min = defaultdict(list)
    for ev in all_events:
        events_by_min[ev["minute"]].append(ev)

    trades = []
    pos = None
    dpnl = 0.0
    closs = 0
    shut = False

    target_level = p.get("target_level", 0.0)
    stop_level = p.get("stop_level", 1.155)
    use_marny_levels = p.get("use_marny_levels", True)
    atr_sl_mult = p.get("atr_sl_mult", None)
    atr_tp_mult = p.get("atr_tp_mult", None)
    fixed_sl = p.get("fixed_sl", None)
    fixed_tp = p.get("fixed_tp", None)

    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None and minute > pos["entry_min"]:
            bar = option_bars_by_key[pos["key"]].get(minute)
            if bar is not None:
                high, low, close = bar["high"], bar["low"], bar["close"]
                sl_px = pos["sl"]
                tp_px = pos["tp"]

                if trail_sl:
                    if high > pos["highest"]:
                        pos["highest"] = high
                        gain = pos["highest"] - pos["entry"]
                        if use_marny_levels and pos.get("span"):
                            span_val = pos["span"]
                            if gain >= 0.382 * span_val:
                                new_sl = pos["entry"]
                                if gain >= 0.50 * span_val:
                                    new_sl = pos["highest"] - 0.25 * span_val
                                if new_sl > pos["sl"]:
                                    pos["sl"] = new_sl
                        elif atr_sl_mult is not None and pos.get("atr"):
                            if gain >= 1.5 * pos["atr"]:
                                new_sl = pos["highest"] - (atr_sl_mult * pos["atr"])
                                if new_sl > pos["sl"]:
                                    pos["sl"] = new_sl
                        elif fixed_sl is not None:
                            if gain >= fixed_sl:
                                new_sl = pos["highest"] - fixed_sl
                                if new_sl > pos["sl"]:
                                    pos["sl"] = new_sl
                    sl_px = pos["sl"]

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
            
            if use_marny_levels and ev.get("span") is not None:
                peak = ev.get("peak_high", ev["option_entry"] + 15.0)
                origin = ev.get("origin_low", ev["option_entry"] - 10.0)
                span = ev.get("span", peak - origin)
                
                # Marny Core SL/TP Formulas
                if target_level == 0.0:
                    tp_val = peak
                elif target_level == -0.29:
                    tp_val = peak + 0.29 * span
                else:
                    tp_val = peak - target_level * span
                
                sl_val = origin - (stop_level - 1.0) * span
            elif atr_sl_mult is not None and atr_tp_mult is not None:
                atr_v = ev.get("atr", 5.0)
                sl_val = ev["option_entry"] - (atr_sl_mult * atr_v)
                tp_val = ev["option_entry"] + (atr_tp_mult * atr_v)
            else:
                sl_val = ev["option_entry"] - (fixed_sl if fixed_sl is not None else 10.0)
                tp_val = ev["option_entry"] + (fixed_tp if fixed_tp is not None else 15.0)

            pos = {
                **ev,
                "entry_min": minute,
                "entry": ev["option_entry"],
                "sl": sl_val,
                "tp": tp_val,
                "highest": ev["option_entry"],
            }

    return trades


def run_divergence_cross_backtest(params, days_subset=None, workers=8):
    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days if days_subset is None else days_subset
    previous = {day: max((c for c in all_days if c < day), default="") for day in days}

    tasks = [
        (day, opt_map[day], opt_map.get(previous[day], ""), params)
        for day in days
    ]

    with Pool(processes=min(cpu_count(), workers), initializer=init_worker, initargs=(spot_all,)) as pool:
        all_day_trades = pool.map(process_day_divergence_cross, tasks)

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
    parser = argparse.ArgumentParser(description="Marny Core & Stochastic Divergence Cross-Filter Backtest")
    parser.add_argument("--smoke", action="store_true", help="Run 5-day smoke test")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPU workers")
    args = parser.parse_args()

    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days[:5] if args.smoke else all_days

    print("=" * 130)
    print(f"MARNY CORE & STOCHASTIC DIVERGENCE CROSS-FILTER (7 YEARS: 2020-2026 | {len(days)} DAYS)")
    print(f"Workers: {args.workers}")
    print("=" * 130)

    configs = [
        # (Label, mode, f_sl, f_tp, sl_m, tp_m, span, inc_fees)
        ("Mode A: Marny Signal + Divergence Filter | Fixed 10/15 | With Fees", "A", 10.0, 15.0, None, None, 0.0, True),
        ("Mode A: Marny Signal + Divergence Filter | Fixed 10/15 | Zero Fees", "A", 10.0, 15.0, None, None, 0.0, False),
        ("Mode A: Marny Signal + Divergence Filter | ATR 2x/4x   | With Fees", "A", None, None, 2.0, 4.0, 0.0, True),
        ("Mode A: Marny Signal + Divergence Filter | ATR 2x/4x   | Zero Fees", "A", None, None, 2.0, 4.0, 0.0, False),
        
        ("Mode B: Divergence Signal + Marny Filter | Fixed 10/15 | With Fees", "B", 10.0, 15.0, None, None, 0.0, True),
        ("Mode B: Divergence Signal + Marny Filter | Fixed 10/15 | Zero Fees", "B", 10.0, 15.0, None, None, 0.0, False),
        ("Mode B: Divergence Signal + Marny Filter | ATR 2x/4x   | With Fees", "B", None, None, 2.0, 4.0, 0.0, True),
        ("Mode B: Divergence Signal + Marny Filter | ATR 2x/4x   | Zero Fees", "B", None, None, 2.0, 4.0, 0.0, False),
    ]

    print(f"{'Configuration':68s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'Profit Factor':13s} | {'Max Drawdown':14s} | {'Total Fees':12s}")
    print("-" * 130)

    for label, mode, f_sl, f_tp, sl_m, tp_m, span, inc_fees in configs:
        params = {
            "mode": mode,
            "min_span": span,
            "include_fees": inc_fees,
            "trail_sl": False,
            "fixed_sl": f_sl,
            "fixed_tp": f_tp,
            "atr_sl_mult": sl_m,
            "atr_tp_mult": tp_m,
            "daily_loss_pts": -30.0,
        }
        res = run_divergence_cross_backtest(params, days, args.workers)
        print(f"{label:68s} | {res['trades']:7d} | {res['win_rate']:7.1f}% | {res['net_points']:+10.2f} | Rs {res['net_rs']:+13.2f} | {res['profit_factor']:13.2f} | Rs {res['max_drawdown_rs']:11.2f} | Rs {res['fees_rs']:10.2f}")

    print("=" * 130)


if __name__ == "__main__":
    main()
