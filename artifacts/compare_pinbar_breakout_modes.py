"""Compare legacy and first-break pinbar behavior on cached Flattrade days."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import date, datetime, time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.flattrade_day_cache import decode_active_strikes, load_day_cache
from artifacts.replay_flattrade_signals import parse_time
from flattrade_bot.indicators.patterns import BullishPinBarDetector
from flattrade_bot.main import row_to_candle
from flattrade_bot.strategies.quad_pinbar_divergence import QuadPinbarDivergenceStrategy


ModeCheck = Callable[[type, list], bool]
TIMEFRAME_ORDER = {"1m": 0, "2m": 1, "3m": 2, "5m": 3}


def legacy_breakout(cls, candle_history: list, max_lookback: int = 10) -> bool:
    """The profitable legacy behavior: reuse any already-broken pinbar high."""
    if len(candle_history) < 2:
        return False
    current = candle_history[-1]
    lookback = min(len(candle_history) - 1, max_lookback)
    for i in range(1, lookback + 1):
        past = candle_history[-1 - i]
        if cls.is_bullish_pin_bar(past) and current.high > past.high:
            return True
    return False


def first_break(cls, candle_history: list, max_lookback: int = 10) -> bool:
    """Only accept the first candle that trades above a pinbar high."""
    if len(candle_history) < 2:
        return False
    current = candle_history[-1]
    lookback = min(len(candle_history) - 1, max_lookback)
    for i in range(1, lookback + 1):
        past = candle_history[-1 - i]
        intervening = candle_history[-i:-1]
        if (
            cls.is_bullish_pin_bar(past)
            and current.high > past.high
            and all(candle.high <= past.high for candle in intervening)
        ):
            return True
    return False


@contextmanager
def breakout_mode(check: ModeCheck):
    descriptor = BullishPinBarDetector.__dict__["check_vicinity_breakout"]
    BullishPinBarDetector.check_vicinity_breakout = classmethod(check)
    try:
        yield
    finally:
        BullishPinBarDetector.check_vicinity_breakout = descriptor


def collect_signals(cache: dict, target: date, check: ModeCheck) -> tuple[list[dict], dict]:
    active = decode_active_strikes(cache["active_strikes"])
    bars: dict[tuple[str, int], dict[str, dict]] = {}
    signals = []
    with breakout_mode(check):
        for key, info in sorted(cache["contracts"].items()):
            side, strike_text = key.split(":", 1)
            strike = int(strike_text)
            rows = sorted(
                {row["time"]: row for row in info["rows"]}.values(),
                key=lambda row: row["time"],
            )
            bars[(side, strike)] = {
                row["time"]: row
                for row in rows
                if parse_time(row["time"]).date() == target
            }
            strategy = QuadPinbarDivergenceStrategy()
            for row in rows:
                timestamp = parse_time(row["time"])
                triggers = strategy.push_spot_candle(row_to_candle(row), side)
                if timestamp.date() != target:
                    continue
                minute = timestamp.hour * 60 + timestamp.minute
                if strike not in active.get((side, minute), set()):
                    continue
                for tf, is_reverse, signal, entry, sl_points, tp_points in triggers:
                    target_side = ("PE" if side == "CE" else "CE") if is_reverse else side
                    target_strikes = active.get((target_side, minute), set())
                    target_strike = min(target_strikes) if target_strikes else strike
                    signals.append({
                        "dt": timestamp,
                        "time": row["time"],
                        "key": (target_side, target_strike),
                        "tf": tf,
                        "signal": signal,
                        "entry": float(entry),
                        "sl": float(sl_points),
                        "tp": float(tp_points),
                        "source": f"{side}{strike}",
                    })
    signals.sort(key=lambda item: (item["dt"], TIMEFRAME_ORDER[item["tf"]], item["source"]))
    return signals, bars


def execute_signals(
    signals: list[dict],
    bars: dict,
    spot_rows: list[dict],
    fixed_tp: float | None = None,
    fixed_sl: float | None = None,
) -> dict:
    timeline = sorted(
        parse_time(row["time"])
        for row in spot_rows
        if parse_time(row["time"]).date() == parse_time(spot_rows[0]["time"]).date()
    )
    by_time: dict[datetime, list[dict]] = {}
    for signal in signals:
        by_time.setdefault(signal["dt"], []).append(signal)

    position = None
    trades = []
    blocked = []
    for timestamp in timeline:
        if position and timestamp > position["dt"]:
            row = bars.get(position["key"], {}).get(timestamp.strftime("%d-%m-%Y %H:%M:%S"))
            if row:
                stop = position["entry"] - position["sl"]
                target = position["entry"] + position["tp"]
                reason = None
                exit_price = None
                if float(row["low"]) <= stop:
                    reason, exit_price = "SL", stop
                elif float(row["high"]) >= target:
                    reason, exit_price = "TP", target
                if reason:
                    trades.append(close_trade(position, timestamp, exit_price, reason))
                    position = None

        for signal in by_time.get(timestamp, []):
            if position is None and timestamp.time() <= time(15, 0):
                position = signal.copy()
                if fixed_tp is not None:
                    position["tp"] = fixed_tp
                if fixed_sl is not None:
                    position["sl"] = fixed_sl
                break
            blocked.append({
                "time": signal["time"],
                "side": f"{signal['key'][0]}{signal['key'][1]}",
                "timeframe": signal["tf"],
                "signal": signal["signal"],
            })

        if timestamp.time() == time(15, 0) and position:
            row = bars.get(position["key"], {}).get(timestamp.strftime("%d-%m-%Y %H:%M:%S"))
            exit_price = float(row["close"]) if row else position["entry"]
            trades.append(close_trade(position, timestamp, exit_price, "EOD"))
            position = None

    return {
        "signal_count": len(signals),
        "trades": trades,
        "blocked_count": len(blocked),
        "blocked": blocked,
        "net_points": round(sum(trade["points"] for trade in trades), 2),
        "net_rupees": round(sum(trade["pnl"] for trade in trades), 2),
    }


def close_trade(position: dict, timestamp: datetime, exit_price: float, reason: str) -> dict:
    points = round(exit_price - position["entry"], 2)
    return {
        "entry_time": position["time"],
        "side": f"{position['key'][0]}{position['key'][1]}",
        "timeframe": position["tf"],
        "signal": position["signal"],
        "entry": round(position["entry"], 2),
        "exit_time": timestamp.strftime("%d-%m-%Y %H:%M:%S"),
        "exit": round(exit_price, 2),
        "reason": reason,
        "points": points,
        "pnl": round(points * 65, 2),
    }


def compare_day(
    cache_dir: Path,
    target: date,
    fixed_tp: float | None = None,
    fixed_sl: float | None = None,
) -> dict:
    cache = load_day_cache(cache_dir, target)
    if cache is None:
        raise SystemExit(f"Missing cache for {target.isoformat()}")
    modes = {
        "legacy_high_break": legacy_breakout,
        "first_break_high": first_break,
    }
    return {
        name: execute_signals(
            *collect_signals(cache, target, check),
            cache["spot_rows"],
            fixed_tp,
            fixed_sl,
        )
        for name, check in modes.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        nargs="+",
        default=["2026-08-12", "2026-08-13"],
        help="Cached trading dates to compare",
    )
    parser.add_argument("--cache-dir", default="artifacts/flattrade_day_cache")
    parser.add_argument("--output", default="artifacts/pinbar_breakout_comparison.json")
    parser.add_argument(
        "--fixed-tp",
        type=float,
        default=None,
        help="Override ATR target with this many points while keeping ATR SL",
    )
    parser.add_argument(
        "--fixed-sl",
        type=float,
        default=None,
        help="Override ATR stop with this many points",
    )
    args = parser.parse_args()
    result = {
        "config": {
            "stop": "fixed" if args.fixed_sl is not None else "ATR-controlled",
            "fixed_tp_points": args.fixed_tp,
            "fixed_sl_points": args.fixed_sl,
            "lot_size": 65,
        },
        "days": {
            day: compare_day(
                Path(args.cache_dir),
                date.fromisoformat(day),
                args.fixed_tp,
                args.fixed_sl,
            )
            for day in args.dates
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Comparison written to {output}")


if __name__ == "__main__":
    main()
