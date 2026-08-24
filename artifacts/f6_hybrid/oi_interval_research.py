"""Research-only F6 backtest with intervalized aggregate OI trend gating.

This file intentionally does not modify the live strategy. Interval ``0`` is
the no-OI baseline; positive intervals sample the put-minus-call OI gap at
that many-minute boundaries and allow CE only on bullish gap momentum and PE
only on bearish gap momentum.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import SYM_RE, latest_spot, load_spot, option_files, to_minutes
from backtest_monthly_ramp import resolve_exit_points


SESSION_START = 560
SESSION_END = 900
DAY_LAST = 930
LOT_SIZE = 65
OI_RADIUS = 4
OI_THRESHOLD = 3_000_000.0
EXECUTION_TIMEFRAMES = {"1m", "2m"}
GLOBAL_SPOT = {}


def init_worker(spot):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot


def load_groups(path: str) -> dict:
    frame = pd.read_csv(
        path,
        usecols=["time", "symbol", "open", "high", "low", "close", "oi", "volume"],
        engine="c",
    )
    if frame.empty:
        return {}
    frame["min"] = np.array([to_minutes(value) for value in frame["time"]])
    frame = frame.drop_duplicates(subset=["symbol", "min"], keep="last")
    frame = frame.sort_values(["symbol", "min"], kind="stable")
    output = {}
    for symbol, group in frame.groupby("symbol"):
        output[symbol] = {
            "min": group["min"].to_numpy(),
            "open": group["open"].to_numpy(),
            "high": group["high"].to_numpy(),
            "low": group["low"].to_numpy(),
            "close": group["close"].to_numpy(),
            "oi": group["oi"].to_numpy(),
            "volume": group["volume"].to_numpy(),
        }
    return output


def filter_symbols(groups: dict, target_strikes: set[int]) -> dict:
    return {
        symbol: group
        for symbol, group in groups.items()
        if (match := SYM_RE.match(symbol)) and int(match.group(2)) in target_strikes
    }


def value_at(group: dict, minute: int, field: str):
    index = np.searchsorted(group["min"], minute)
    if index < len(group["min"]) and group["min"][index] == minute:
        return float(group[field][index])
    return None


def oi_sentiment(groups: dict, spot: dict, interval: int) -> dict[int, str]:
    if interval <= 0:
        return {}

    snapshots = {}
    previous_gap = None
    current_sentiment = "NEUTRAL"
    for minute in range(SESSION_START - 30, DAY_LAST + 1):
        if minute % interval != 0:
            continue
        spot_price = latest_spot(spot, minute)
        if spot_price is None:
            continue
        base = int(round(spot_price / 50.0) * 50)
        strikes = {base + offset * 50 for offset in range(-OI_RADIUS, OI_RADIUS + 1)}
        call_oi = 0.0
        put_oi = 0.0
        for symbol, group in groups.items():
            match = SYM_RE.match(symbol)
            if not match or int(match.group(2)) not in strikes:
                continue
            oi = value_at(group, minute, "oi")
            if oi is None:
                continue
            if match.group(3) == "CE":
                call_oi += oi
            else:
                put_oi += oi
        gap = put_oi - call_oi
        if previous_gap is not None:
            if gap >= OI_THRESHOLD and gap > previous_gap:
                current_sentiment = "BULLISH"
            elif gap <= -OI_THRESHOLD and gap < previous_gap:
                current_sentiment = "BEARISH"
            else:
                current_sentiment = "NEUTRAL"
        previous_gap = gap
        snapshots[minute] = current_sentiment

    carried = "NEUTRAL"
    output = {}
    for minute in range(SESSION_START, DAY_LAST + 1):
        if minute in snapshots:
            carried = snapshots[minute]
        output[minute] = carried
    return output


def build_signals(groups: dict, previous_groups: dict, params: dict, spot: dict) -> tuple[dict, dict, str]:
    first_symbol = next(iter(groups), "")
    match = SYM_RE.match(first_symbol)
    if not match:
        return {}, {}, ""
    prefix = match.group(1)
    start_spot = latest_spot(spot, 555) or latest_spot(spot, SESSION_START)
    if start_spot is None:
        return {}, {}, prefix
    base = int(round(start_spot / 50.0) * 50)
    target_strikes = set(range(base - 250, base + 300, 50))
    current = filter_symbols(groups, target_strikes)
    previous = filter_symbols(previous_groups, target_strikes)

    trackers = {}
    for symbol, group in previous.items():
        tracker = grid.MTFTracker(params)
        for index in range(len(group["min"])):
            tracker.push_1m(grid.Candle(
                open=group["open"][index], high=group["high"][index],
                low=group["low"][index], close=group["close"][index],
                minute=group["min"][index],
            ))
        trackers[symbol] = tracker

    triggers = {}
    slices = {}
    for symbol, group in current.items():
        tracker = trackers.setdefault(symbol, grid.MTFTracker(params))
        slices[symbol] = group
        for index in range(len(group["min"])):
            minute = group["min"][index]
            for tf, is_reverse, signal_type, price, atr in tracker.push_1m(grid.Candle(
                open=group["open"][index], high=group["high"][index],
                low=group["low"][index], close=group["close"][index], minute=minute,
            )):
                if tf not in EXECUTION_TIMEFRAMES:
                    continue
                triggers.setdefault(minute, []).append(
                    (symbol, int(SYM_RE.match(symbol).group(2)), SYM_RE.match(symbol).group(3),
                     price, is_reverse, tf, atr)
                )
    return triggers, {"slices": slices, "trackers": trackers}, prefix


def simulate(
    day: str,
    spot: dict,
    triggers: dict,
    state: dict,
    prefix: str,
    params: dict,
    interval: int,
    sentiment_override: dict[int, str] | None = None,
) -> list[dict]:
    slices = state["slices"]
    trackers = copy.deepcopy(state["trackers"])
    sentiment = sentiment_override if sentiment_override is not None else oi_sentiment(state["oi_groups"], spot, interval)
    daily_loss_pts = grid.DAILY_LOSS_RS / LOT_SIZE
    position = None
    daily_points = 0.0
    consecutive_losses = 0
    stopped = False
    trades = []

    def bar_at(group, minute):
        if group is None:
            return None
        index = np.searchsorted(group["min"], minute)
        if index < len(group["min"]) and group["min"][index] == minute:
            return (
                group["open"][index], group["high"][index],
                group["low"][index], group["close"][index],
            )
        return None

    def active_info(side, minute):
        spot_price = latest_spot(spot, minute)
        if spot_price is None:
            return None
        atm = int(round(spot_price / 50.0) * 50)
        strike = atm + (grid.CE_OFFSET if side == "CE" else grid.PE_OFFSET)
        symbol = f"{prefix}{strike}{side}"
        group = slices.get(symbol)
        return (symbol, group, strike) if group is not None else None

    for minute in range(SESSION_START, DAY_LAST + 1):
        if position is not None:
            held = bar_at(position["slice"], minute)
            if held is not None:
                _, high, low, close = held
                position["last_px"] = float(close)
                position["duration_min"] += 1
                if daily_points + (close - position["entry"]) <= daily_loss_pts:
                    exit_price, reason = close, "SHUTDOWN_LOSS"
                else:
                    exit_price, reason = None, ""
                    if high >= position["target"] and low <= position["sl"]:
                        exit_price, reason = position["sl"], "SL"
                    elif high >= position["target"]:
                        exit_price, reason = position["target"], "TP"
                    elif low <= position["sl"]:
                        exit_price, reason = position["sl"], "SL"
                    if exit_price is None:
                        tracker = trackers.get(position["symbol"])
                        if tracker:
                            one_minute = tracker.trackers["1m"]
                            one_minute.div.update(close, one_minute.prev_s1, low_price=low, high_price=high)
                            if one_minute.div.has_bearish_peak_divergence():
                                exit_price, reason = close, "BEARISH_PEAK_REVERSAL"
                if exit_price is not None:
                    points = round(exit_price - position["entry"], 2)
                    trades.append({
                        "date": day, "entry_min": position["entry_min"], "exit_min": minute,
                        "side": position["side"], "symbol": position["symbol"],
                        "entry": position["entry"], "exit": exit_price, "pts": points,
                        "rs": round(points * LOT_SIZE), "reason": reason,
                        "duration_min": position["duration_min"], "tf": position["tf"],
                        "oi_interval": interval,
                    })
                    daily_points += points
                    consecutive_losses = consecutive_losses + 1 if points <= 0 else 0
                    stopped = consecutive_losses >= params["consec_loss"] or daily_points <= daily_loss_pts
                    position = None
        if position is not None and minute >= SESSION_END:
            points = round(position["last_px"] - position["entry"], 2)
            trades.append({
                "date": day, "entry_min": position["entry_min"], "exit_min": minute,
                "side": position["side"], "symbol": position["symbol"],
                "entry": position["entry"], "exit": position["last_px"], "pts": points,
                "rs": round(points * LOT_SIZE), "reason": "EOD",
                "duration_min": position["duration_min"], "tf": position["tf"],
                "oi_interval": interval,
            })
            position = None
        if position is not None or stopped or minute >= SESSION_END:
            continue

        for symbol, strike, side, trigger_price, is_reverse, tf, atr in triggers.get(minute, []):
            source = active_info(side, minute)
            if source is None or source[2] != strike or position is not None:
                continue
            actual_side = ("PE" if side == "CE" else "CE") if is_reverse else side
            actual = active_info(actual_side, minute)
            if actual is None:
                continue
            if interval > 0:
                expected = "BULLISH" if actual_side == "CE" else "BEARISH"
                if sentiment.get(minute, "NEUTRAL") != expected:
                    continue
            entry_bar = bar_at(actual[1], minute)
            if entry_bar is None:
                continue
            entry = float(entry_bar[3])
            sl, tp = resolve_exit_points(atr, params["atr_sl_mult"], params["atr_tp_mult"], 10.0, 15.0, params)
            position = {
                "side": actual_side, "symbol": actual[0], "slice": actual[1],
                "entry": entry, "sl": entry - sl, "target": entry + tp,
                "sl_pts": sl, "tp_pts": tp, "entry_min": minute,
                "last_px": entry, "duration_min": 0, "tf": tf,
            }
            break
    return trades


def process_day(args):
    day, path, previous_path, params, intervals = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None:
        return {interval: [] for interval in intervals}
    groups = load_groups(path)
    previous = load_groups(previous_path) if previous_path else {}
    if not groups:
        return {interval: [] for interval in intervals}
    triggers, state, prefix = build_signals(groups, previous, params, spot)
    if not state.get("slices"):
        return {interval: [] for interval in intervals}
    state["oi_groups"] = groups
    return {
        interval: simulate(day, spot, triggers, state, prefix, params, interval)
        for interval in intervals
    }


def summarize(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "wr": 0.0, "pts": 0.0, "rs": 0, "pf": 0.0}
    wins = [trade for trade in trades if trade["pts"] > 0]
    losses = [trade for trade in trades if trade["pts"] <= 0]
    gross_wins = sum(trade["pts"] for trade in wins)
    gross_losses = abs(sum(trade["pts"] for trade in losses))
    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) * 100,
        "pts": round(sum(trade["pts"] for trade in trades), 2),
        "rs": round(sum(trade["rs"] for trade in trades)),
        "pf": round(gross_wins / gross_losses, 6) if gross_losses else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--intervals", default="0,1,2,3,5,15")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-divergence", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/oi_interval_research.json")
    args = parser.parse_args()
    params = json.loads(Path(args.params_file).read_text(encoding="utf-8"))["params"]
    params["use_divergence"] = not args.no_divergence
    intervals = [int(value) for value in args.intervals.split(",")]
    spot_all = load_spot()
    files = option_files(args.start, args.end)
    days = sorted(set(files) & set(spot_all))
    if args.smoke:
        days = days[:5]
    tasks = [
        (day, str(files[day]), str(files[days[index - 1]]) if index else "", params, intervals)
        for index, day in enumerate(days)
    ]
    aggregate = {interval: [] for interval in intervals}
    with Pool(max(1, min(8, args.workers)), initializer=init_worker, initargs=(spot_all,)) as pool:
        for day_results in pool.imap(process_day, tasks):
            for interval, trades in day_results.items():
                aggregate[interval].extend(trades)
    result = {
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "workers": max(1, min(8, args.workers)),
        "timeframes": sorted(EXECUTION_TIMEFRAMES),
        "oi_threshold": OI_THRESHOLD,
        "use_divergence": not args.no_divergence,
        "params": params,
        "results": {str(interval): summarize(trades) for interval, trades in aggregate.items()},
    }
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
