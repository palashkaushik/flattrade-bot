"""Read-only Flattrade candle replay for a live signal timestamp."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattrade_bot.broker.auto_login import automated_flattrade_login
from flattrade_bot.broker.history import FlattradeHistoryFetcher
from flattrade_bot.config import settings
from flattrade_bot.main import row_to_candle
from flattrade_bot.strategies.quad_pinbar_divergence import (
    QuadPinbarDivergenceStrategy,
)


def parse_row_time(value: str) -> datetime:
    return datetime.strptime(value, "%d-%m-%Y %H:%M:%S")


def replay_side(rows, target: datetime, side: str) -> dict:
    strategy = QuadPinbarDivergenceStrategy()
    triggers = []
    watch = []
    processed = 0
    watch_start = target - timedelta(minutes=20)
    for row in rows:
        row_time = parse_row_time(row["time"])
        if row_time > target:
            continue
        processed += 1
        row_triggers = list(
            strategy.push_spot_candle(row_to_candle(row), side)
        )
        tracker = strategy.ce_tracker.trackers["1m"] if side == "CE" else strategy.pe_tracker.trackers["1m"]
        if watch_start <= row_time <= target:
            t1, t2 = tracker.div._find_troughs()
            watch.append({
                "time": row_time.isoformat(" "),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "s1": tracker.stoch.latest_s1,
                "s2": tracker.stoch.latest_s2,
                "s3": tracker.stoch.latest_s3,
                "s4": tracker.stoch.latest_s4,
                "bullish_divergence": tracker.has_bull_divergence,
                "super_ready": tracker.super_ready,
                "super_setup_active": tracker.setup_active and tracker.stype == "super",
                "setup_type": tracker.stype,
                "trough_1": t1,
                "trough_2": t2,
                "triggers": row_triggers,
            })
        for tf, is_reverse, signal_type, entry, sl_points, tp_points in row_triggers:
            if target.replace(hour=0, minute=0, second=0) <= row_time <= target:
                triggers.append({
                    "time": row_time.isoformat(" "),
                    "timeframe": tf,
                    "signal": signal_type,
                    "reverse": is_reverse,
                    "entry": entry,
                    "sl_points": sl_points,
                    "tp_points": tp_points,
                })

    summary = strategy.get_stoch_summary(side)
    return {
        "side": side,
        "processed_rows": processed,
        "api_first": rows[0]["time"] if rows else None,
        "api_last": rows[-1]["time"] if rows else None,
        "triggers_to_target": triggers,
        "watch_window_14_18_to_target": watch,
        "summary": summary,
    }


async def run(args) -> None:
    target_date = date.fromisoformat(args.date)
    target = datetime.combine(target_date, time.fromisoformat(args.time))
    token = os.getenv("FLATTRADE_TOKEN", "")
    if not token:
        token = automated_flattrade_login(
            user_id=settings.FLATTRADE_USER_ID,
            password=settings.FLATTRADE_PASSWORD,
            totp_key=settings.FLATTRADE_TOTP_KEY,
            api_key=settings.FLATTRADE_API_KEY,
            api_secret=settings.FLATTRADE_API_SECRET,
            headless=True,
        )
    if not token:
        raise SystemExit("Read-only login failed")

    fetcher = FlattradeHistoryFetcher(token)
    spot_rows, ce_rows, pe_rows = await asyncio.gather(
        fetcher.fetch_historical_candles("26000", "NSE", "1", args.days_back),
        fetcher.fetch_historical_candles(args.ce_token, "NFO", "1", args.days_back),
        fetcher.fetch_historical_candles(args.pe_token, "NFO", "1", args.days_back),
    )

    def spot_at_target():
        rows = [
            row for row in spot_rows
            if parse_row_time(row["time"]) <= target
        ]
        if not rows:
            return None
        row = rows[-1]
        return {"time": row["time"], "close": row["close"]}

    result = {
        "target": target.isoformat(" "),
        "tokens": {"spot": "26000", "ce": args.ce_token, "pe": args.pe_token},
        "spot_at_target": spot_at_target(),
        "ce": replay_side(ce_rows, target, "CE"),
        "pe": replay_side(pe_rows, target, "PE"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"Diagnosis written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--time", default="14:38:00")
    parser.add_argument("--ce-token", default="41005")
    parser.add_argument("--pe-token", default="41014")
    parser.add_argument("--days-back", type=int, default=2)
    parser.add_argument(
        "--output",
        default="artifacts/live_signal_diagnosis.json",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
