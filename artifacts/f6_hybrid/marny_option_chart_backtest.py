"""Marny Option-Chart Backtest Engine (2020-2026).

Exact Rules:
1. Chart Source: Evaluated directly on individual Option Charts (CE and PE).
2. HTF Filter (5-Minute):
   - 5m Heikin-Ashi on the option chart.
   - 11-period Linear Regression Candles Signal Curve (11-period SMA of 5m HA Closes).
   - 5m UT Bot Alerts (Key=1.0, Period=10) on the option chart.
   - Bullish Bias = (5m HA Close > 11-period LinReg Plot) AND (5m UT Bot Color == "green").
3. 1-Minute Marny Fibonacci Setup on Option Chart:
   - 3-Phase Bullish Impulse Rally: 1 RED UT -> >= 5 Consecutive GREEN UT -> 1 RED UT.
   - Origin Low = Lowest low in the impulse pattern.
   - Peak High = Highest high in the impulse pattern.
   - Span = Peak High - Origin Low (filtered by min_span).
   - 0.786 Retracement Entry Level = Peak High - 0.786 * Span.
   - Trigger: When 1m option candle touches 0.786 entry level AND 5m HTF bias is Bullish.
4. Exits on Option Chart:
   - TP = 0.290 or TP = 0.000 (Peak High retest).
   - SL = 1.079, 1.155, or 1.250 (below Origin Low).
   - EOD = 15:00 Market Close.
   - Transaction fees & slippage toggleable.
"""

from __future__ import annotations

import argparse
import json
import re
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
from artifacts.f6_hybrid.causal_live_parity_research import IncrementalATR

LOT_SIZE = 65
SESSION_START = 560
SESSION_END = 900
DAY_LAST = 930
CONSECUTIVE_LOSS_LIMIT = 4
GLOBAL_SPOT = {}


def init_worker(spot):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot
    source.GLOBAL_CACHE = {}


class UTBotState:
    """Causal O(1) UT Bot calculation."""
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
        previous_stop = self.trailing_stop
        self.previous_source = source_price
        if atr is None or previous_source is None:
            return "blue"

        loss = self.key * atr
        if source_price > previous_stop and previous_source > previous_stop:
            self.trailing_stop = max(previous_stop, source_price - loss)
        elif source_price < previous_stop and previous_source < previous_stop:
            self.trailing_stop = min(previous_stop, source_price + loss)
        elif source_price > previous_stop:
            self.trailing_stop = source_price - loss
        else:
            self.trailing_stop = source_price + loss

        if previous_source < previous_stop and source_price > previous_stop:
            self.position = 1
        elif previous_source > previous_stop and source_price < previous_stop:
            self.position = -1
        return "green" if self.position == 1 else "red" if self.position == -1 else "blue"


class HeikinAshiState:
    """Causal O(1) Heikin-Ashi transformer."""
    def __init__(self):
        self.open = None
        self.close = None

    def update(self, candle: Candle) -> Candle:
        ha_close = (candle.open + candle.high + candle.low + candle.close) / 4.0
        ha_open = (
            (candle.open + candle.close) / 2.0
            if self.open is None
            else (self.open + self.close) / 2.0
        )
        ha_high = max(candle.high, ha_open, ha_close)
        ha_low = min(candle.low, ha_open, ha_close)
        self.open = ha_open
        self.close = ha_close
        return Candle(ha_open, ha_high, ha_low, ha_close, minute=candle.minute)


class Option5mHTFBias:
    """Causal 5-Minute HTF Bias computed directly on Option bars."""
    def __init__(self):
        self.ha = HeikinAshiState()
        self.ut = UTBotState(key=1.0, period=10)
        self.ha_closes = deque(maxlen=11)
        self.buffer = []
        self.bullish = False
        self.ut_color = "blue"

    def update_1m(self, candle: Candle):
        self.buffer.append(candle)
        if candle.minute % 5 != 0 or not self.buffer:
            return
        buf = self.buffer
        self.buffer = []
        raw_5m = Candle(
            open=buf[0].open,
            high=max(c.high for c in buf),
            low=min(c.low for c in buf),
            close=buf[-1].close,
            minute=candle.minute,
        )
        ha_5m = self.ha.update(raw_5m)
        self.ha_closes.append(ha_5m.close)
        self.ut_color = self.ut.update(raw_5m)
        if len(self.ha_closes) >= 11:
            linreg_plot = sum(self.ha_closes) / len(self.ha_closes)
            self.bullish = ha_5m.close > linreg_plot and self.ut_color == "green"
        else:
            self.bullish = False

    def snapshot(self) -> bool:
        return self.bullish


class Option1mFibTracker:
    """Exact Marni 3-Phase Bullish Impulse & 0.786 Retracement Tracker on Option Chart."""
    def __init__(self, min_span: float = 10.0, min_candles: int = 5):
        self.min_span = min_span
        self.min_candles = min_candles
        self.ut = UTBotState(key=1.0, period=10)
        self.history = []  # (Candle, ut_color)
        self.setups = []

    def push(self, candle: Candle, htf_bullish: bool) -> list[dict]:
        col = self.ut.update(candle)
        self.history.append((candle, col))

        # Check for completed Bullish Impulse: 1 RED -> >= 5 GREEN -> 1 RED
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
                    self.setups.append({
                        "peak_high": peak_high,
                        "origin_low": origin_low,
                        "span": span,
                    })

        # Evaluate 0.786 Retracement Touches
        events = []
        valid_setups = []
        for s in self.setups:
            peak = s["peak_high"]
            origin = s["origin_low"]
            span = s["span"]

            # Invalidation: breached beyond origin stop before touch
            if candle.low < origin - 0.25 * span:
                continue

            entry_level = peak - 0.786 * span

            # Touch condition on 1m option candle
            if candle.high >= entry_level - 1.0 and candle.low <= entry_level + 1.0:
                if htf_bullish:
                    events.append({
                        "minute": candle.minute,
                        "entry_level": entry_level,
                        "entry_price": candle.close,
                        "peak_high": peak,
                        "origin_low": origin,
                        "span": span,
                    })
                    continue
            valid_setups.append(s)

        self.setups = valid_setups
        return events


def process_day_option_chart(args):
    day, opt_path, prev_opt_path, p = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not opt_path:
        return {}

    min_span = p.get("min_span", 10.0)
    target_levels = p.get("target_levels", (0.29, 0.0))
    stop_levels = p.get("stop_levels", (1.079, 1.155, 1.25))
    include_fees = p.get("include_fees", True)
    daily_loss_pts = p.get("daily_loss_pts", -30.0)
    daily_profit_pts = p.get("daily_profit_pts", 30.0)

    rec = source.cached_option(str(opt_path))
    if rec is None:
        return {}
    df, groups, prefix = rec
    if prefix is None:
        return {}

    spot_mins = spot["min"]
    all_events = []
    option_bars_by_key = {}
    symbol_name_by_key = {}

    for side in ("CE", "PE"):
        # Find active strikes for the day
        atm_strikes = set()
        for m in range(SESSION_START, SESSION_END + 1):
            idx = np.searchsorted(spot_mins, m, side="right") - 1
            if idx >= 0:
                spot_px = float(spot["close"][idx])
                atm = int(round(spot_px / 50) * 50)
                strike = atm - 50 if side == "CE" else atm + 50
                atm_strikes.add(strike)

        for strike in atm_strikes:
            sym = f"{prefix}{strike}{side}"
            sl = source.make_slice(df, groups, sym)
            if sl is None or len(sl["times"]) < 15:
                continue

            key = (side, strike)
            symbol_name_by_key[key] = sym
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

            # Initialize 5m HTF bias and 1m Fib tracker on this Option Chart
            htf_5m = Option5mHTFBias()
            fib_1m = Option1mFibTracker(min_span=min_span, min_candles=5)

            # Warmup with previous day option bars if available
            if prev_opt_path:
                prev_rec = source.cached_option(str(prev_opt_path))
                if prev_rec:
                    p_df, p_groups, p_prefix = prev_rec
                    p_sl = source.make_slice(p_df, p_groups, sym)
                    if p_sl is not None:
                        for pj in range(len(p_sl["times"])):
                            pc = Candle(float(p_sl["open"][pj]), float(p_sl["high"][pj]), float(p_sl["low"][pj]), float(p_sl["close"][pj]), minute=int(p_sl["times"][pj]))
                            htf_5m.update_1m(pc)
                            fib_1m.push(pc, htf_5m.snapshot())

            # Process current day option bars
            for j in range(len(sl["times"])):
                m = int(sl["times"][j])
                c = Candle(float(sl["open"][j]), float(sl["high"][j]), float(sl["low"][j]), float(sl["close"][j]), minute=m)
                htf_5m.update_1m(c)
                is_bullish = htf_5m.snapshot()
                evs = fib_1m.push(c, is_bullish)
                for ev in evs:
                    all_events.append({
                        **ev,
                        "side": side,
                        "strike": strike,
                        "symbol": sym,
                        "key": key,
                        "option_entry": c.close,
                    })

    # Simulate trades across parameter configurations
    output = {}
    for tp in target_levels:
        for sl_lvl in stop_levels:
            k = f"opt_chart|tp{tp}|sl{sl_lvl}"
            trades = simulate_option_chart_trades(
                all_events,
                option_bars_by_key,
                target_level=tp,
                stop_level=sl_lvl,
                include_fees=include_fees,
                daily_loss_pts=daily_loss_pts,
                daily_profit_pts=daily_profit_pts,
            )
            for t in trades:
                t["date"] = day
            output[k] = trades

    return output


def simulate_option_chart_trades(events, option_bars, target_level, stop_level, include_fees=True, daily_loss_pts=-30.0, daily_profit_pts=30.0):
    events_by_min = defaultdict(list)
    for ev in events:
        events_by_min[ev["minute"]].append(ev)

    trades = []
    pos = None
    dpnl = 0.0
    closs = 0
    shut = False

    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None and minute > pos["entry_min"]:
            bar = option_bars[pos["key"]].get(minute)
            if bar is not None:
                peak = pos["peak_high"]
                origin = pos["origin_low"]
                span = pos["span"]

                # Target on option chart: TP = 0.29 => peak - 0.29 * span; TP = 0.0 => peak
                tgt_price = peak if target_level == 0.0 else peak - target_level * span
                # Stop loss on option chart: SL = 1.079, 1.155, 1.25 => origin - (sl_lvl - 1.0) * span
                sl_price = origin - (stop_level - 1.0) * span

                high, low, close = bar["high"], bar["low"], bar["close"]
                ex, rsn = None, ""

                # Check intraday daily loss / profit cap hit on current bar
                if dpnl + (low - pos["entry"]) <= daily_loss_pts:
                    ex, rsn = pos["entry"] + (daily_loss_pts - dpnl), "SHUTDOWN_LOSS"
                    shut = True
                elif dpnl + (high - pos["entry"]) >= daily_profit_pts:
                    ex, rsn = pos["entry"] + (daily_profit_pts - dpnl), "SHUTDOWN_PROFIT"
                    shut = True
                elif low <= sl_price and high >= tgt_price:
                    ex, rsn = sl_price, "SL"
                elif high >= tgt_price:
                    ex, rsn = tgt_price, "TP"
                elif low <= sl_price:
                    ex, rsn = sl_price, "SL"
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
                    if closs >= CONSECUTIVE_LOSS_LIMIT or dpnl <= daily_loss_pts or dpnl >= daily_profit_pts:
                        shut = True
                    pos = None

        if pos is not None or shut or minute >= SESSION_END:
            continue

        for ev in events_by_min.get(minute, []):
            if pos is not None:
                break
            pos = {
                **ev,
                "entry_min": minute,
                "entry": ev["option_entry"],
            }

    return trades


def run_option_chart_backtest(params, days_subset=None, workers=8):
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
        all_day_outputs = pool.map(process_day_option_chart, tasks)

    trades_by_config = defaultdict(list)
    for day_res in all_day_outputs:
        for k, trs in day_res.items():
            trades_by_config[k].extend(trs)

    summary = {}
    for k, all_trades in trades_by_config.items():
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

        summary[k] = {
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

    return summary


def main():
    parser = argparse.ArgumentParser(description="Marny Option-Chart Backtest Engine")
    parser.add_argument("--smoke", action="store_true", help="Run 5-day smoke test")
    parser.add_argument("--no-fees", action="store_true", help="Disable transaction fees")
    parser.add_argument("--min-span", type=float, default=10.0, help="Minimum impulse span on option chart")
    parser.add_argument("--daily-loss", type=float, default=-30.0, help="Daily max loss cap in points (default: -30.0)")
    parser.add_argument("--daily-profit", type=float, default=30.0, help="Daily max profit cap in points (default: 30.0)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPU workers")
    args = parser.parse_args()

    include_fees = not args.no_fees
    print("=" * 95)
    print("MARNY ENGINE — OPTION-CHART 5M HTF + 1M FIB 0.786 BACKTEST")
    print(f"Min Span: {args.min_span} pts | Daily Caps: [{args.daily_loss:+.1f} / {args.daily_profit:+.1f} pts] | Include Fees: {include_fees} | Workers: {args.workers}")
    print("=" * 95)

    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days[:5] if args.smoke else all_days

    params = {
        "min_span": args.min_span,
        "target_levels": (0.29, 0.0),
        "stop_levels": (1.079, 1.155, 1.25),
        "include_fees": include_fees,
        "daily_loss_pts": args.daily_loss,
        "daily_profit_pts": args.daily_profit,
    }

    t0 = time.time()
    res = run_option_chart_backtest(params, days, args.workers)
    el = time.time() - t0

    print(f"\nExecution finished in {el:.2f}s ({len(days)/el:.1f} days/sec)")
    print("\n" + "=" * 115)
    print("OPTION-CHART BACKTEST RESULTS SUMMARY")
    print("=" * 115)
    print(f"{'Configuration':25s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'Profit Factor':13s} | {'Max Drawdown':14s} | {'Total Fees':12s}")
    print("-" * 115)
    for k, s in res.items():
        print(f"{k:25s} | {s['trades']:7d} | {s['win_rate']:7.1f}% | {s['net_points']:+10.2f} | Rs {s['net_rs']:+13.2f} | {s['profit_factor']:13.2f} | Rs {s['max_drawdown_rs']:11.2f} | Rs {s['fees_rs']:10.2f}")
    print("=" * 115)

    # Save detailed JSON report
    out_file = ROOT / "artifacts" / "f6_hybrid" / "marny_option_chart_results.json"
    with open(out_file, "w") as f:
        # Exclude large trade list for concise saving
        clean_summary = {k: {k2: v2 for k2, v2 in v.items() if k2 != "all_trades"} for k, v in res.items()}
        json.dump(clean_summary, f, indent=2)
    print(f"\nSaved detailed summary to: {out_file}")


if __name__ == "__main__":
    main()
