"""Index-signal to option-execution backtest matrix.

Signals are generated from NIFTY index candles. The active ITM2 option is used
only for entry/exit fills. Fixed exits are index points (SL20/TP30); ATR exits
use index ATR(14) with SL x2/TP x4, matching the existing futures champion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as futures
from backtest_5y_optimized import option_files
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost
from flattrade_bot.indicators.divergence import DivergenceEngine
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.causal_live_parity_research import LegacyDivergence, IncrementalATR


LOT_SIZE = 65
SESSION_START = 560
SESSION_END = 900
DAY_LAST = 930
CE_OFFSET, PE_OFFSET = -100, 100
INDEX_ATR_PERIOD = 14
INDEX_ATR_SL_MULT = 2.0
INDEX_ATR_TP_MULT = 4.0
FIXED_SL_POINTS = 20.0
FIXED_TP_POINTS = 30.0
CONSECUTIVE_LOSS_LIMIT = 4
SYMBOL_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")

GLOBAL_SPOT = {}


class SymmetricLegacyDivergence(LegacyDivergence):
    def has_bearish(self):
        first, second = self._pair(peak=True)
        return first is not None and second[0] > first[0] and second[1] < first[1]


class IndexSignalFeed:
    """Index-only MTF feed implementing the existing futures Flag/Super rules."""

    def __init__(self, embed: int, divergence_mode: str):
        self.trackers = {tf: futures.FuturesQuadTriggers(embed) for tf in futures.TF_SPECS}
        self.stochs = {tf: futures.QuadStoch() for tf in futures.TF_SPECS}
        self.divs = {
            tf: (
                SymmetricLegacyDivergence()
                if divergence_mode == "previous_divergence"
                else DivergenceEngine()
            )
            for tf in futures.TF_SPECS
        }
        self.atrs = {tf: IncrementalATR(INDEX_ATR_PERIOD) for tf in futures.TF_SPECS}
        self.buffers = {tf: [] for tf in futures.TF_SPECS}
        self.values = {}
        self.divergence_mode = divergence_mode

    def reset_session(self):
        self.trackers = {
            tf: futures.FuturesQuadTriggers(self.trackers[tf].embed_n)
            for tf in futures.TF_SPECS
        }
        self.buffers = {tf: [] for tf in futures.TF_SPECS}
        self.values = {}

    def warmup(self, rows: dict):
        for index in range(len(rows["min"])):
            self.push_1m(
                rows["high"][index],
                rows["low"][index],
                rows["close"][index],
                rows["min"][index],
            )
        self.reset_session()

    def _has_divergence(self, tf: str, signal: str) -> bool:
        if self.divergence_mode == "no_divergence":
            return True
        div = self.divs[tf]
        bullish = signal in ("bull_flag", "supersignal_bull")
        if self.divergence_mode == "previous_divergence":
            return div.has_bullish() if bullish else div.has_bearish()
        return (
            div.has_bullish_trough_divergence()
            if bullish
            else div.has_bearish_peak_divergence()
        )

    def push_1m(self, high: float, low: float, close: float, minute: int):
        completed = []
        for tf, spec in futures.TF_SPECS.items():
            self.buffers[tf].append((high, low, close, minute))
            if len(self.buffers[tf]) != spec[0]:
                continue
            buf = self.buffers[tf]
            self.buffers[tf] = []
            aggregate = Candle(
                open=buf[0][0],
                high=max(item[0] for item in buf),
                low=min(item[1] for item in buf),
                close=buf[-1][2],
                minute=buf[-1][3],
            )
            values = self.stochs[tf].push(aggregate.high, aggregate.low, aggregate.close)
            self.values[tf] = values
            self.trackers[tf].update_embed(values.get("s4d"))
            self.atrs[tf].update(aggregate.high, aggregate.low, aggregate.close)
            self.divs[tf].update(
                aggregate.close,
                values.get("s1d"),
                low_price=aggregate.low,
                high_price=aggregate.high,
            )
            completed.append(tf)

        signals = []
        for tf in completed:
            values = self.values[tf]
            signal = self.trackers[tf].evaluate(values, signal_mode="all")
            if signal is None or not self._has_divergence(tf, signal):
                continue
            signals.append((tf, signal, self.atrs[tf].value))
        return signals


def init_worker(spot):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot
    futures.GLOBAL_CACHE = {}


def bar_at(option_slice, minute):
    if option_slice is None:
        return None
    index = np.searchsorted(option_slice["times"], minute)
    if index < len(option_slice["times"]) and option_slice["times"][index] == minute:
        return (
            option_slice["open"][index],
            option_slice["high"][index],
            option_slice["low"][index],
            option_slice["close"][index],
        )
    return None


def process_day(args):
    day, option_path, params, divergence_mode, exit_mode = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not option_path:
        return []
    option_data = futures.cached_option(option_path)
    if option_data is None:
        return []
    frame, groups, prefix = option_data
    if prefix is None:
        return []

    feed = IndexSignalFeed(params["embed"], divergence_mode)
    previous_days = sorted(key for key in GLOBAL_SPOT if key < day)
    if previous_days:
        feed.warmup(GLOBAL_SPOT[previous_days[-1]])

    slices = {}
    position = None
    consecutive_losses = 0
    stopped = False
    trades = []

    def active_option(side, minute):
        spot_px = futures.latest_value(spot, minute)
        if spot_px is None:
            return None
        atm = int(round(spot_px / 50.0) * 50)
        strike = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        symbol = f"{prefix}{strike}{side}"
        option_slice = slices.get(symbol)
        if option_slice is None:
            option_slice = futures.make_slice(frame, groups, symbol)
            if option_slice is not None:
                slices[symbol] = option_slice
        return (symbol, strike, option_slice) if option_slice is not None else None

    for index in range(len(spot["min"])):
        minute = int(spot["min"][index])
        if minute < SESSION_START or minute > DAY_LAST:
            continue
        index_open = float(spot["open"][index])
        index_high = float(spot["high"][index])
        index_low = float(spot["low"][index])
        index_close = float(spot["close"][index])
        signals = feed.push_1m(index_high, index_low, index_close, minute)

        if position is not None and minute > position["entry_min"]:
            option_bar = bar_at(position["slice"], minute)
            if option_bar is not None:
                stop_hit = (
                    index_low <= position["index_stop"]
                    if position["direction"] == "bullish"
                    else index_high >= position["index_stop"]
                )
                target_hit = (
                    index_high >= position["index_target"]
                    if position["direction"] == "bullish"
                    else index_low <= position["index_target"]
                )
                reason = None
                if stop_hit:
                    reason = "SL"
                elif target_hit:
                    reason = "TP"
                if minute >= SESSION_END and reason is None:
                    reason = "EOD"
                if reason:
                    entry_fill = position["entry_option"] + SLIPPAGE_PTS
                    exit_fill = float(option_bar[3]) - SLIPPAGE_PTS
                    points = round(exit_fill - entry_fill, 2)
                    fee = trade_cost(entry_fill, exit_fill, BROKERAGE_PER_ORDER)
                    net_rs = round(points * LOT_SIZE - fee, 2)
                    trades.append({
                        "date": day,
                        "entry_min": position["entry_min"],
                        "exit_min": minute,
                        "side": position["side"],
                        "signal": position["signal"],
                        "symbol": position["symbol"],
                        "entry": entry_fill,
                        "exit": exit_fill,
                        "pts": points,
                        "rs_net": net_rs,
                        "fee": fee,
                        "reason": reason,
                        "tf": position["tf"],
                        "sl_points": position["sl_points"],
                        "tp_points": position["tp_points"],
                    })
                    consecutive_losses = consecutive_losses + 1 if net_rs <= 0 else 0
                    stopped = consecutive_losses >= CONSECUTIVE_LOSS_LIMIT
                    position = None
        if position is not None or stopped or minute >= SESSION_END:
            continue

        for tf, signal, index_atr in signals:
            side = "CE" if signal in ("bull_flag", "supersignal_bull") else "PE"
            active = active_option(side, minute)
            if active is None:
                continue
            symbol, strike, option_slice = active
            option_bar = bar_at(option_slice, minute)
            if option_bar is None:
                continue
            entry_option = float(option_bar[3])
            if exit_mode == "fixed_index":
                sl_points, tp_points = FIXED_SL_POINTS, FIXED_TP_POINTS
            else:
                atr_value = index_atr if index_atr and index_atr > 0.5 else None
                sl_points = atr_value * INDEX_ATR_SL_MULT if atr_value else FIXED_SL_POINTS
                tp_points = atr_value * INDEX_ATR_TP_MULT if atr_value else FIXED_TP_POINTS
            direction = "bullish" if side == "CE" else "bearish"
            position = {
                "entry_min": minute,
                "side": side,
                "symbol": symbol,
                "strike": strike,
                "signal": signal,
                "tf": tf,
                "slice": option_slice,
                "entry_option": entry_option,
                "index_entry": index_close,
                "direction": direction,
                "index_stop": index_close - sl_points if direction == "bullish" else index_close + sl_points,
                "index_target": index_close + tp_points if direction == "bullish" else index_close - tp_points,
                "sl_points": sl_points,
                "tp_points": tp_points,
            }
            break

    return trades


def stats(trades, day_count):
    wins = [trade for trade in trades if trade["rs_net"] > 0]
    losses = [trade for trade in trades if trade["rs_net"] <= 0]
    gross_loss = abs(sum(trade["rs_net"] for trade in losses))
    return {
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "net_rs": round(sum(trade["rs_net"] for trade in trades)),
        "net_points": round(sum(trade["rs_net"] for trade in trades) / LOT_SIZE, 2),
        "profit_factor": round(sum(trade["rs_net"] for trade in wins) / gross_loss, 4) if gross_loss else float("inf"),
        "avg_trades_per_day": round(len(trades) / day_count, 3) if day_count else 0.0,
        "fees_rs": round(sum(trade["fee"] for trade in trades), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/index_signal_option_matrix.json")
    args = parser.parse_args()

    spot_all = futures.load_spot()
    option_day_map = futures.option_day_files(args.start, args.end)
    days = sorted(set(option_day_map) & set(spot_all))
    if args.smoke:
        days = days[:5]
    params = {"embed": 14, "atr_period": INDEX_ATR_PERIOD, "atr_sl_mult": INDEX_ATR_SL_MULT, "atr_tp_mult": INDEX_ATR_TP_MULT}
    divergence_modes = ("no_divergence", "new_divergence", "previous_divergence")
    exit_modes = ("fixed_index", "atr_index")
    tasks = [
        (day, str(option_day_map[day]), params, divergence, exit_mode)
        for divergence in divergence_modes
        for exit_mode in exit_modes
        for day in days
    ]

    results = []
    with Pool(max(1, min(8, args.workers)), initializer=init_worker, initargs=(spot_all,)) as pool:
        grouped = {}
        for task, trades in zip(tasks, pool.imap(process_day, tasks)):
            grouped.setdefault((task[3], task[4]), []).extend(trades)
        for divergence in divergence_modes:
            for exit_mode in exit_modes:
                results.append({
                    **stats(grouped[(divergence, exit_mode)], len(days)),
                    "divergence_mode": divergence,
                    "exit_mode": exit_mode,
                })

    result = {
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "smoke": args.smoke,
        "signal_source": "NIFTY index OHLC",
        "signal_rules": "existing futures Flag/Super rules",
        "divergence_modes": divergence_modes,
        "exit_modes": {"fixed_index": "index SL20 / TP30", "atr_index": "index ATR14 x2 / x4"},
        "consecutive_loss_limit": CONSECUTIVE_LOSS_LIMIT,
        "daily_caps": False,
        "results": results,
    }
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
