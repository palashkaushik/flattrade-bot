"""Marny Engine — High Performance ATR-Dynamic SL/TP Optimizer (2020-2026).

Combines:
1. Exact Marny 0.786 Fibonacci Retracement + 15m HTF Bias Trigger Engine.
2. Dynamic Volatility-Adaptive ATR Target (TP) and Stop-Loss (SL) on Options.
3. Fast parallel multi-worker Optuna / Grid search across 2020-2026 dataset.
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
import pandas as pd

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


class StrictHTFBiasState:
    def __init__(self, period: int = 15):
        self.period = period
        self.ha = HeikinAshiState()
        self.ut = UTBotState(key=1.0, period=10)
        self.closes = deque(maxlen=11)
        self.buffer = []
        self.bullish = False
        self.bearish = False
        self.ut_color = "blue"

    def update_1m(self, candle: Candle):
        self.buffer.append(candle)
        if candle.minute % self.period != 0 or not self.buffer:
            return
        buf = self.buffer
        self.buffer = []
        raw_bar = Candle(
            open=buf[0].open,
            high=max(c.high for c in buf),
            low=min(c.low for c in buf),
            close=buf[-1].close,
            minute=candle.minute,
        )
        ha_bar = self.ha.update(raw_bar)
        self.closes.append(ha_bar.close)
        self.ut_color = self.ut.update(raw_bar)
        if len(self.closes) >= 11:
            linreg_plot = sum(self.closes) / len(self.closes)
            self.bullish = ha_bar.close > linreg_plot and self.ut_color == "green"
            self.bearish = ha_bar.close < linreg_plot and self.ut_color == "red"
        else:
            self.bullish = False
            self.bearish = False

    def snapshot(self) -> dict:
        return {"bullish": self.bullish, "bearish": self.bearish, "ut_color": self.ut_color}


class MarnyFibTriggerFeed:
    def __init__(self, min_span: float = 25.0, min_candles: int = 5):
        self.min_span = min_span
        self.min_candles = min_candles
        self.ut = UTBotState()
        self.history = []
        self.setups = []
        self.curr_day = 0

    def push(self, candle: Candle, current_bias: dict | None = None) -> list[dict]:
        if candle.minute == 555:
            self.curr_day += 1
        col = self.ut.update(candle)
        self.history.append((candle, col, self.curr_day))

        # Check Bearish
        if col == "green" and len(self.history) >= self.min_candles + 2:
            red_count, k = 0, len(self.history) - 2
            while k >= 0 and self.history[k][1] == "red":
                red_count += 1
                k -= 1
            if red_count >= self.min_candles:
                pattern = [self.history[i][0] for i in range(max(0, k), len(self.history))]
                origin_high, trough_low = max(c.high for c in pattern), min(c.low for c in pattern)
                span = origin_high - trough_low
                if span >= self.min_span:
                    self.setups.append(("bearish", origin_high, trough_low, "low_to_high", current_bias or {}))

        # Check Bullish
        if col == "red" and len(self.history) >= self.min_candles + 2:
            green_count, k = 0, len(self.history) - 2
            while k >= 0 and self.history[k][1] == "green":
                green_count += 1
                k -= 1
            if green_count >= self.min_candles:
                pattern = [self.history[i][0] for i in range(max(0, k), len(self.history))]
                peak_high, origin_low = max(c.high for c in pattern), min(c.low for c in pattern)
                span = peak_high - origin_low
                if span >= self.min_span:
                    self.setups.append(("bullish", peak_high, origin_low, "high_to_low", current_bias or {}))

        events, valid = [], []
        for direction, high, low, orientation, bias_c in self.setups:
            span = high - low
            if (orientation == "high_to_low" and candle.low < low - 0.25 * span) or \
               (orientation == "low_to_high" and candle.high > high + 0.25 * span):
                continue
            entry_level = high - 0.786 * span if orientation == "high_to_low" else low + 0.786 * span
            if candle.high >= entry_level - 1.0 and candle.low <= entry_level + 1.0:
                side = "CE" if direction == "bullish" else "PE"
                if (current_bias or {}).get("bullish" if side == "CE" else "bearish", False):
                    events.append({
                        "minute": candle.minute,
                        "entry_level": entry_level,
                        "entry_price": candle.close,
                        "fib_high": high,
                        "fib_low": low,
                        "direction": direction,
                        "side": side,
                    })
                    continue
            valid.append((direction, high, low, orientation, bias_c))
        self.setups = valid
        return events


def option_atr_fast(bars_list: list[dict], upto_minute: int, period: int = 14) -> float | None:
    valid = [b for b in bars_list if b["minute"] <= upto_minute]
    if len(valid) < period:
        return None
    h = np.array([b["high"] for b in valid])
    l = np.array([b["low"] for b in valid])
    c = np.array([b["close"] for b in valid])
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
    return atr


def process_day_marny_atr(args):
    day, opt_path, prev_opt_path, p = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not opt_path:
        return []

    min_span = p.get("min_span", 25.0)
    atr_period = p.get("atr_period", 14)
    atr_sl_mult = p.get("atr_sl_mult", 1.5)
    atr_tp_mult = p.get("atr_tp_mult", 3.5)
    trail_sl = p.get("trail_sl", False)

    rec = source.cached_option(str(opt_path))
    if rec is None:
        return []
    df, groups, prefix = rec
    if prefix is None:
        return []

    # Prepare 1m candles for spot
    spot_mins = spot["min"]
    spot_candles = [
        Candle(float(spot["open"][i]), float(spot["high"][i]), float(spot["low"][i]), float(spot["close"][i]), minute=int(spot_mins[i]))
        for i in range(len(spot_mins))
    ]

    htf = StrictHTFBiasState(15)
    marny = MarnyFibTriggerFeed(min_span=min_span, min_candles=5)

    # Warmup HTF on previous day if available
    prev_days = sorted(k for k in GLOBAL_SPOT if k < day)
    if prev_days:
        prev_spot = GLOBAL_SPOT[prev_days[-1]]
        for i in range(len(prev_spot["min"])):
            c = Candle(float(prev_spot["open"][i]), float(prev_spot["high"][i]), float(prev_spot["low"][i]), float(prev_spot["close"][i]), minute=int(prev_spot["min"][i]))
            htf.update_1m(c)
            marny.push(c, htf.snapshot())

    # Option slices cache
    slices = {}
    def get_slice_bars(side, minute):
        spot_px = float(spot["close"][np.searchsorted(spot_mins, minute, side="right") - 1])
        atm = int(round(spot_px / 50) * 50)
        strike = atm - 50 if side == "CE" else atm + 50
        sym = f"{prefix}{strike}{side}"
        if sym not in slices:
            sl = source.make_slice(df, groups, sym)
            if sl is None:
                return None, None
            bars = []
            for j in range(len(sl["times"])):
                bars.append({"minute": int(sl["times"][j]), "open": float(sl["open"][j]), "high": float(sl["high"][j]), "low": float(sl["low"][j]), "close": float(sl["close"][j])})
            slices[sym] = (sym, bars, sl)
        return slices[sym][0], slices[sym][1]

    trades = []
    pos = None
    dpnl = 0.0
    closs = 0
    shut = False

    for c in spot_candles:
        m = c.minute
        if m < SESSION_START or m > DAY_LAST:
            continue

        htf.update_1m(c)
        bias = htf.snapshot()

        # Check existing open position
        if pos is not None:
            sym, opt_bars = get_slice_bars(pos["side"], m)
            cur_bar = next((b for b in opt_bars if b["minute"] == m), None) if opt_bars else None
            if cur_bar is not None:
                high, low, close = cur_bar["high"], cur_bar["low"], cur_bar["close"]
                pos["highest"] = max(pos["highest"], high)
                
                # Trailing SL update
                if trail_sl and pos["highest"] > pos["entry"] + pos["atr_val"] * 1.5:
                    pos["sl"] = max(pos["sl"], pos["highest"] - pos["atr_val"] * atr_sl_mult)

                ex, rsn = None, ""
                if low <= pos["sl"] and high >= pos["tgt"]:
                    ex, rsn = pos["sl"], "SL"
                elif high >= pos["tgt"]:
                    ex, rsn = pos["tgt"], "TP"
                elif low <= pos["sl"]:
                    ex, rsn = pos["sl"], "SL"
                elif m >= SESSION_END:
                    ex, rsn = close, "EOD"

                if ex is not None:
                    slip = SLIPPAGE_PTS
                    entry_fill = pos["entry"] + slip
                    exit_fill = ex - slip
                    pts = round(exit_fill - entry_fill, 2)
                    fee = trade_cost(entry_fill, exit_fill, BROKERAGE_PER_ORDER)
                    net_rs = round(pts * LOT_SIZE - fee, 2)
                    trades.append({
                        "date": day,
                        "entry_min": pos["entry_min"],
                        "exit_min": m,
                        "side": pos["side"],
                        "symbol": pos["symbol"],
                        "entry": entry_fill,
                        "exit": exit_fill,
                        "points": pts,
                        "rs_net": net_rs,
                        "fee": fee,
                        "reason": rsn,
                    })
                    dpnl += pts
                    closs = closs + 1 if pts <= 0 else 0
                    if closs >= CONSECUTIVE_LOSS_LIMIT:
                        shut = True
                    pos = None

        if pos is not None or shut or m >= SESSION_END:
            continue

        raw_events = marny.push(c, bias)
        for ev in raw_events:
            if pos is not None:
                break
            sym, opt_bars = get_slice_bars(ev["side"], m)
            if not opt_bars:
                continue
            cur_bar = next((b for b in opt_bars if b["minute"] == m), None)
            if not cur_bar or cur_bar["close"] <= 0:
                continue
            
            atr_v = option_atr_fast(opt_bars, m, atr_period)
            if not atr_v or atr_v < 1.0:
                atr_v = 8.0 # default fallback ATR

            entry_px = cur_bar["close"]
            sl_dist = atr_v * atr_sl_mult
            tp_dist = atr_v * atr_tp_mult

            pos = {
                "side": ev["side"],
                "symbol": sym,
                "entry": entry_px,
                "sl": entry_px - sl_dist,
                "tgt": entry_px + tp_dist,
                "atr_val": atr_v,
                "highest": entry_px,
                "entry_min": m,
            }

    return trades


def run_marny_atr_backtest(params, days_subset=None, workers=8):
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
        all_day_outputs = pool.map(process_day_marny_atr, tasks)

    all_trades = [t for day_trades in all_day_outputs for t in day_trades]
    wins = [t for t in all_trades if t["rs_net"] > 0]
    losses = [t for t in all_trades if t["rs_net"] <= 0]
    loss_tot = abs(sum(t["rs_net"] for t in losses))
    win_tot = sum(t["rs_net"] for t in wins)
    net_rs = sum(t["rs_net"] for t in all_trades)
    net_pts = sum(t["points"] for t in all_trades)
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0.0
    pf = win_tot / loss_tot if loss_tot else 0.0
    fees = sum(t["fee"] for t in all_trades)

    # Max Drawdown
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
    parser = argparse.ArgumentParser(description="Marny Engine ATR Dynamic SL/TP Optimizer")
    parser.add_argument("--smoke", action="store_true", help="Run 5-day smoke test")
    parser.add_argument("--grid", action="store_true", help="Run grid search optimization")
    parser.add_argument("--workers", type=int, default=8, help="Parallel CPU workers")
    args = parser.parse_args()

    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days[:5] if args.smoke else all_days

    if args.smoke:
        print("=== SMOKE TEST: MARNY ENGINE WITH ATR DYNAMIC SL/TP (5 DAYS) ===")
        p = {"min_span": 25.0, "atr_period": 14, "atr_sl_mult": 1.5, "atr_tp_mult": 3.5, "trail_sl": True}
        res = run_marny_atr_backtest(p, days, args.workers)
        print(f"Trades: {res['trades']} | Win Rate: {res['win_rate']}% | Net Pts: {res['net_points']:+,.2f} | Net Rs: Rs {res['net_rs']:+,.2f} | PF: {res['profit_factor']} | MaxDD: Rs {res['max_drawdown_rs']:,.2f}")
        return

    if args.grid:
        print(f"=== MARNY ENGINE ATR GRID SEARCH ACROSS {len(days)} DAYS ===")
        grid = [
            {"min_span": s, "atr_period": 14, "atr_sl_mult": sl, "atr_tp_mult": tp, "trail_sl": tr}
            for s in [20.0, 30.0, 40.0]
            for sl in [1.2, 1.5, 2.0]
            for tp in [2.5, 3.5, 5.0]
            for tr in [True, False]
        ]
        print(f"Total Combinations: {len(grid)}")
        best_res, best_p = None, None
        for idx, p in enumerate(grid, 1):
            t0 = time.time()
            res = run_marny_atr_backtest(p, days, args.workers)
            el = time.time() - t0
            print(f"[{idx:2d}/{len(grid)}] Span={p['min_span']:4.1f} | SL={p['atr_sl_mult']:3.1f} | TP={p['atr_tp_mult']:3.1f} | Trail={str(p['trail_sl']):5s} -> Trades={res['trades']:4d} | WR={res['win_rate']:5.1f}% | Net Rs=Rs {res['net_rs']:+11.2f} | PF={res['profit_factor']:4.2f} | DD=Rs {res['max_drawdown_rs']:9.2f} ({el:.1f}s)")
            if best_res is None or res["net_rs"] > best_res["net_rs"]:
                best_res, best_p = res, p

        print("\n" + "=" * 90)
        print("OPTIMIZATION WINNER:")
        print("=" * 90)
        print(f"Parameters: {best_p}")
        print(f"Trades: {best_res['trades']} | WR: {best_res['win_rate']}% | Net Rs: Rs {best_res['net_rs']:+,.2f} | PF: {best_res['profit_factor']} | MaxDD: Rs {best_res['max_drawdown_rs']:,.2f}")


if __name__ == "__main__":
    main()
