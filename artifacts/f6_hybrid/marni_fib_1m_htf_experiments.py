"""1-Minute Marni Fib Backtest evaluating 5m, 10m, and 15m HTF Trend Filters (2020-2026)."""

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
ENTRY_LEVEL = 0.786
TARGET_LEVELS = (0.618, 0.50, 0.382, 0.29, 0.236, 0.0)
STOP_LEVELS = (1.059, 1.079, 1.115, 1.155, 1.25)
HTF_CONFIGS = {"5m": 5, "10m": 10, "15m": 15}
SYMBOL_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")
GLOBAL_SPOT = {}


def init_worker(spot):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot
    source.GLOBAL_CACHE = {}


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


def linreg_value(values: deque[float]) -> float | None:
    if len(values) < 11:
        return None
    n = len(values)
    x_sum = n * (n - 1) / 2.0
    x2_sum = (n - 1) * n * (2 * n - 1) / 6.0
    y_sum = sum(values)
    xy_sum = sum(index * value for index, value in enumerate(values))
    denominator = n * x2_sum - x_sum * x_sum
    slope = (n * xy_sum - x_sum * y_sum) / denominator
    intercept = (y_sum - slope * x_sum) / n
    return intercept + slope * (n - 1)


class StrictHTFBiasState:
    def __init__(self, period: int):
        self.period = period
        self.buffer = []
        self.ha = HeikinAshiState()
        self.ut = UTBotState()
        self.closes = deque(maxlen=11)
        self.ha_candle = None
        self.ut_color = "blue"
        self.linreg_plot = None

    def update_1m(self, candle: Candle):
        self.buffer.append(candle)
        if len(self.buffer) != self.period:
            return
        buf = self.buffer
        self.buffer = []
        aggregate = Candle(
            open=buf[0].open,
            high=max(i.high for i in buf),
            low=min(i.low for i in buf),
            close=buf[-1].close,
            minute=buf[-1].minute,
        )
        ha = self.ha.update(aggregate)
        self.ha_candle = ha
        self.ut_color = self.ut.update(ha)
        self.closes.append(ha.close)
        self.linreg_plot = linreg_value(self.closes)

    def snapshot(self) -> dict:
        if self.ha_candle is None or self.linreg_plot is None:
            return {"bullish": False, "bearish": False}
        close = self.ha_candle.close
        open_p = self.ha_candle.open
        plot = self.linreg_plot
        return {
            "bullish": close > plot and self.ut_color == "green",
            "bearish": close < plot and self.ut_color == "red",
            "ut_color": self.ut_color,
            "linreg_plot": plot,
        }


class MultiHTFFeed:
    def __init__(self):
        self.states = {name: StrictHTFBiasState(p) for name, p in HTF_CONFIGS.items()}

    def warmup(self, rows):
        for r in rows:
            self.push(r)

    def push(self, row):
        c = Candle(float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), int(row["minute"]))
        for state in self.states.values():
            state.update_1m(c)

    def snapshot(self, name: str) -> dict:
        return self.states[name].snapshot()


class FibPattern:
    def __init__(self, direction="bullish", first="red", middle="green", final="red", orientation="high_to_low"):
        self.direction = direction
        self.first_color = first
        self.middle_color = middle
        self.final_color = final
        self.orientation = orientation
        self.previous_color = None
        self.previous_candle = None
        self.phase = "idle"
        self.range_high = None
        self.range_low = None
        self.middle_count = 0

    def reset_session(self):
        self.previous_color = None
        self.previous_candle = None
        self.phase = "idle"
        self.range_high = None
        self.range_low = None
        self.middle_count = 0

    def update(self, candle: Candle, color: str):
        completed = None
        if self.phase == "in_middle":
            if color == self.middle_color:
                self.middle_count += 1
                self.range_high = max(self.range_high, candle.high)
                self.range_low = min(self.range_low, candle.low)
            elif color == self.final_color:
                if self.middle_count >= 5:
                    self.range_high = max(self.range_high, candle.high)
                    self.range_low = min(self.range_low, candle.low)
                    completed = (self.direction, self.range_high, self.range_low, self.orientation)
                self.phase = "idle"
                self.range_high = None
                self.range_low = None
                self.middle_count = 0
            else:
                self.phase = "idle"
                self.range_high = None
                self.range_low = None
                self.middle_count = 0
        elif color == self.middle_color and self.previous_color == self.first_color and self.previous_candle is not None:
            self.phase = "in_middle"
            self.middle_count = 1
            self.range_high = max(self.previous_candle.high, candle.high)
            self.range_low = min(self.previous_candle.low, candle.low)

        self.previous_color = color
        self.previous_candle = candle
        return completed


class Fib1mEngine:
    def __init__(self):
        self.ut = UTBotState()
        specs = [
            ("bullish", "red", "green", "red", "high_to_low"),
            ("bearish", "green", "red", "green", "low_to_high"),
        ]
        self.patterns = [FibPattern(*s) for s in specs]
        self.setups = []

    def reset_session(self):
        for p in self.patterns:
            p.reset_session()
        self.setups = []

    def push(self, candle: Candle):
        color = self.ut.update(candle)
        valid = []
        for direction, high, low, orientation in self.setups:
            if orientation == "high_to_low" and candle.low < low:
                continue
            if orientation == "low_to_high" and candle.high > high:
                continue
            valid.append((direction, high, low, orientation))

        new_setups = []
        for p in self.patterns:
            comp = p.update(candle, color)
            if comp is not None:
                new_setups.append(comp)

        events = []
        rem = []
        for direction, high, low, orientation in valid:
            entry_level = (
                high - ENTRY_LEVEL * (high - low)
                if orientation == "high_to_low"
                else low + ENTRY_LEVEL * (high - low)
            )
            if candle.high >= entry_level - 1.0 and candle.low <= entry_level + 1.0:
                events.append({
                    "minute": candle.minute,
                    "entry_level": entry_level,
                    "entry_price": candle.close,
                    "fib_high": high,
                    "fib_low": low,
                    "direction": direction,
                    "orientation": orientation,
                })
            else:
                rem.append((direction, high, low, orientation))

        self.setups = rem + new_setups
        return events


def fib_price(high: float, low: float, level: float, orientation: str) -> float:
    span = high - low
    if orientation == "high_to_low":
        return high - level * span
    return low + level * span


def spot_row(spot, index):
    return {
        "open": float(spot["open"][index]),
        "high": float(spot["high"][index]),
        "low": float(spot["low"][index]),
        "close": float(spot["close"][index]),
        "minute": int(spot["min"][index]),
    }


def active_strikes(spot, minute: int, side: str) -> int:
    idx = list(spot["min"]).index(minute) if minute in spot["min"] else 0
    ref_price = spot["open"][idx]
    atm = int(round(ref_price / 50.0) * 50)
    return atm - 50 if side == "CE" else atm + 50


def option_rows(frame, groups, symbol):
    indexes = groups.get(symbol)
    if indexes is None:
        return []
    rows = frame.iloc[indexes].sort_values("time")
    return [
        {
            "time": row["time"],
            "minute": int(str(row["time"]).split(":")[0]) * 60 + int(str(row["time"]).split(":")[1]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for _, row in rows.iterrows()
    ]


def load_day_symbols(day, current_path, previous_path, spot):
    current = source.cached_option(str(current_path))
    previous = source.cached_option(str(previous_path)) if previous_path else None
    if current is None:
        return [], {}
    frame, groups, prefix = current
    previous_rows = {}
    if previous is not None:
        previous_rows = {
            symbol: option_rows(previous[0], previous[1], symbol)
            for symbol in previous[1]
        }
    current_rows = {
        symbol: option_rows(frame, groups, symbol)
        for symbol in groups
    }
    active_keys = set()
    for index, minute in enumerate(spot["min"]):
        if SESSION_START <= minute <= DAY_LAST:
            atm = int(round(float(spot["close"][index]) / 50.0) * 50)
            active_keys.add(("CE", atm - 50))
            active_keys.add(("PE", atm + 50))

    symbols = []
    rows_by_symbol = {}
    for symbol in current_rows:
        match = SYMBOL_RE.match(symbol)
        if not match:
            continue
        key = (match.group(3), int(match.group(2)))
        if key not in active_keys:
            continue
        symbols.append(key)
        rows_by_symbol[key] = {
            "symbol": symbol,
            "previous": previous_rows.get(symbol, []),
            "current": current_rows[symbol],
        }
    return sorted(symbols), rows_by_symbol


def simulate(events, bars, index_bars, spot, target_level, stop_level, include_fees=True):
    events_by_minute = defaultdict(list)
    for event in events:
        events_by_minute[event["minute"]].append(event)

    position = None
    stopped = False
    consecutive_losses = 0
    trades = []
    timeline = [int(m) for m in spot["min"] if SESSION_START <= m <= DAY_LAST]

    for minute in timeline:
        if position is not None and minute > position["entry_min"]:
            row = bars[position["key"]].get(minute)
            if row:
                stop = fib_price(position["fib_high"], position["fib_low"], stop_level, position["orientation"])
                target = fib_price(position["fib_high"], position["fib_low"], target_level, position["orientation"])
                price_row = index_bars[minute]

                if position["side"] == "CE":
                    hit_stop = price_row["low"] <= stop
                    hit_target = price_row["high"] >= target
                else:
                    hit_stop = price_row["high"] >= stop
                    hit_target = price_row["low"] <= target

                reason = "SL" if hit_stop else "TP" if hit_target else None
                if minute >= SESSION_END and reason is None:
                    reason = "EOD"

                if reason:
                    slip = SLIPPAGE_PTS if include_fees else 0.0
                    brokerage = BROKERAGE_PER_ORDER if include_fees else 0.0
                    entry_fill = position["option_entry"] + slip
                    exit_fill = row["close"] - slip
                    points = round(exit_fill - entry_fill, 2)
                    fee = trade_cost(entry_fill, exit_fill, brokerage) if include_fees else 0.0
                    net_rs = round(points * LOT_SIZE - fee, 2)

                    trades.append({
                        "entry_min": position["entry_min"],
                        "exit_min": minute,
                        "side": position["side"],
                        "symbol": position["symbol"],
                        "entry": entry_fill,
                        "exit": exit_fill,
                        "reason": reason,
                        "points": points,
                        "rs_net": net_rs,
                        "fee": fee,
                    })
                    consecutive_losses = consecutive_losses + 1 if net_rs <= 0 else 0
                    stopped = consecutive_losses >= CONSECUTIVE_LOSS_LIMIT
                    position = None

        if position is not None or stopped or minute >= SESSION_END:
            continue
        for event in events_by_minute.get(minute, []):
            if position is not None:
                break
            position = {
                **event,
                "entry_min": minute,
                "key": (event["side"], event["strike"]),
            }
    return trades


def process_day(args):
    day, current_path, previous_path, target_levels, stop_levels, include_fees = args
    spot = GLOBAL_SPOT[day]
    previous_days = sorted(key for key in GLOBAL_SPOT if key < day)
    symbols, symbol_rows = load_day_symbols(day, current_path, previous_path, spot)
    bars = {}
    symbol_by_key = {}
    for side, strike in symbols:
        record = symbol_rows[(side, strike)]
        symbol_by_key[(side, strike)] = record["symbol"]
        bars[(side, strike)] = {row["minute"]: row for row in record["current"]}

    feed_1m = Fib1mEngine()
    htf_feed = MultiHTFFeed()

    if previous_days:
        previous_spot = GLOBAL_SPOT[previous_days[-1]]
        prows = [spot_row(previous_spot, i) for i in range(len(previous_spot["min"]))]
        htf_feed.warmup(prows)
        for prow in prows:
            c = Candle(prow["open"], prow["high"], prow["low"], prow["close"], prow["minute"])
            feed_1m.push(c)
    feed_1m.reset_session()

    events_by_htf = {name: [] for name in HTF_CONFIGS}

    for idx in range(len(spot["min"])):
        row = spot_row(spot, idx)
        htf_feed.push(row)
        c = Candle(row["open"], row["high"], row["low"], row["close"], row["minute"])
        m = row["minute"]
        raw_events = feed_1m.push(c)
        if not raw_events:
            continue

        for ev in raw_events:
            side = "CE" if ev["direction"] == "bullish" else "PE"
            strike = active_strikes(spot, m, side)
            key = (side, strike)
            if key not in bars or m not in bars[key]:
                continue
            option_close = bars[key][m]["close"]

            for htf_name in HTF_CONFIGS:
                bias = htf_feed.snapshot(htf_name)
                allowed = bias["bullish"] if side == "CE" else bias["bearish"]
                if allowed:
                    events_by_htf[htf_name].append({
                        **ev,
                        "side": side,
                        "strike": strike,
                        "symbol": symbol_by_key[key],
                        "minute": m,
                        "option_entry": option_close,
                    })

    index_bars = {int(spot["min"][i]): spot_row(spot, i) for i in range(len(spot["min"]))}
    output = {}
    for htf_name in HTF_CONFIGS:
        for tp in target_levels:
            for sl in stop_levels:
                key = f"1m_htf{htf_name}|tp{tp}|sl{sl}"
                trades = simulate(events_by_htf[htf_name], bars, index_bars, spot, tp, sl, include_fees=include_fees)
                for t in trades:
                    t["date"] = day
                output[key] = trades
    return output


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
    parser = argparse.ArgumentParser(description="1-Minute Marni Fib HTF Comparison (5m vs 10m vs 15m)")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-fees", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/marni_fib_1m_htf_experiments.json")
    args = parser.parse_args()

    include_fees = not args.no_fees
    print(f"=== 1-MINUTE MARNI FIB: 5m vs 10m vs 15m HTF COMPARISON ===")
    print(f"Date Range: {args.start} to {args.end}")
    print(f"HTF Eval Mode: NORMAL (At 0.786 Touch) | Include Fees: {include_fees} | Workers: {args.workers}")

    spot_all = source.load_spot()
    opt_map = source.option_day_files(args.start, args.end)
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days[:5] if args.smoke else all_days
    print(f"Running on {len(days)} trading days {'(SMOKE TEST - 5 DAYS ONLY)' if args.smoke else ''}...")

    previous = {day: max((c for c in all_days if c < day), default="") for day in days}
    target_levels = TARGET_LEVELS
    stop_levels = STOP_LEVELS

    tasks = [
        (
            day,
            opt_map[day],
            opt_map.get(previous[day], ""),
            target_levels,
            stop_levels,
            include_fees,
        )
        for day in days
    ]

    t0 = time.time()
    aggregated = defaultdict(list)

    if args.smoke or args.workers == 1:
        init_worker(spot_all)
        for task in tasks:
            res = process_day(task)
            for k, v in res.items():
                aggregated[k].extend(v)
    else:
        with Pool(processes=args.workers, initializer=init_worker, initargs=(spot_all,)) as pool:
            for res in pool.imap_unordered(process_day, tasks, chunksize=1):
                for k, v in res.items():
                    aggregated[k].extend(v)

    elapsed = time.time() - t0
    print(f"\nExecution finished in {elapsed:.2f} seconds ({len(days)/elapsed:.1f} days/sec).")

    summary = {}
    print(f"\n{'='*105}")
    print(f"COMPARATIVE SUMMARY: 1-MINUTE BASE WITH 5m vs 10m vs 15m HTF FILTERS (NORMAL TOUCH EVAL)")
    print(f"{'='*105}")
    print(f"{'Configuration':25s} | {'Trades':6s} | {'WinRate':7s} | {'Net Rs':12s} | {'Net Pts':8s} | {'PF':6s} | {'MaxDD Rs':10s} | {'Fees Rs':8s}")
    print(f"{'-'*105}")

    for k in sorted(aggregated.keys()):
        trades = aggregated[k]
        st = compute_stats(trades, len(days))
        summary[k] = {"stats": st, "trades": trades}
        print(f"{k:25s} | {st['trades']:6d} | {st['win_rate']:6.1f}% | {st['net_rs']:+12,.2f} | {st['net_points']:+8.2f} | {st['profit_factor']:6.2f} | {st['max_drawdown_rs']:10,.2f} | {st['fees_rs']:8,.2f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: {"stats": v["stats"], "trades": v["trades"]} for k, v in summary.items()}, f, indent=2)
    print(f"\nDetailed JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
