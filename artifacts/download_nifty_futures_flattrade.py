"""Download missing NIFTY futures 1-minute candles from Flattrade."""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.replay_flattrade_signals import fetch_range
from flattrade_bot.broker.auto_login import automated_flattrade_login
from flattrade_bot.broker.history import FlattradeHistoryFetcher
from flattrade_bot.config import settings


FILE_RE = re.compile(r"nifty_fut_(\d{2})_(\d{2})_(\d{4})\.csv$", re.IGNORECASE)
FIELDS = ["date", "time", "symbol", "open", "high", "low", "close", "oi", "volume"]
MIN_VALID_NIFTY_PRICE = 10_000.0


def parse_api_time(value: str) -> datetime:
    for fmt in ("%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported Flattrade timestamp: {value!r}")


def existing_dates(destination: Path) -> set[date]:
    dates = set()
    for path in destination.rglob("nifty_fut_*.csv"):
        match = FILE_RE.search(path.name)
        if match:
            day, month, year = (int(value) for value in match.groups())
            dates.add(date(year, month, day))
    return dates


def month_chunks(start: date, end: date, chunk_days: int):
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def normalize_rows(rows: list[dict], symbol: str) -> dict[date, list[dict]]:
    by_day: dict[date, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        parsed = parse_api_time(str(row["time"]))
        timestamp = parsed.strftime("%Y-%m-%d %H:%M:%S")
        by_day[parsed.date()][timestamp] = {
            "date": parsed.date().isoformat(),
            "time": parsed.strftime("%H:%M:%S"),
            "symbol": symbol,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "oi": float(row.get("oi", 0.0)),
            "volume": float(row.get("volume", 0.0)),
        }
    return {
        day: [items[key] for key in sorted(items)]
        for day, items in by_day.items()
    }


def write_day(destination: Path, day: date, rows: list[dict]) -> Path:
    target_dir = destination / str(day.year) / str(day.month)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"nifty_fut_{day:%d_%m_%Y}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


async def download(args: argparse.Namespace) -> None:
    destination = Path(args.destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    present = existing_dates(destination)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    token = automated_flattrade_login(
        user_id=settings.FLATTRADE_USER_ID,
        password=settings.FLATTRADE_PASSWORD,
        totp_key=settings.FLATTRADE_TOTP_KEY,
        api_key=settings.FLATTRADE_API_KEY,
        api_secret=settings.FLATTRADE_API_SECRET,
        headless=True,
    )
    if not token:
        raise SystemExit("Flattrade read-only login failed")

    fetcher = FlattradeHistoryFetcher(token)
    futures = await fetcher.search_futures_token("NIFTY")
    if not futures:
        raise SystemExit("Could not resolve the NIFTY futures contract on Flattrade")
    if "FPI" in futures.get("tsym", "").upper():
        raise SystemExit(
            f"Resolved non-price FPI instrument {futures['tsym']}; refusing to write data"
        )

    print(f"Futures contract: {futures['tsym']}")
    print(f"Destination: {destination}")
    print(f"Existing daily files skipped: {len(present)}")

    fetched_rows = 0
    created_files = 0
    skipped_days = 0
    for chunk_start, chunk_end in month_chunks(start, end, args.chunk_days):
        print(f"Fetching {chunk_start} to {chunk_end} ...", flush=True)
        rows = await fetch_range(
            fetcher,
            futures["token"],
            "NFO",
            chunk_start,
            chunk_end,
            intrv="1",
        )
        grouped = normalize_rows(rows, futures["tsym"])
        fetched_rows += len(rows)
        for day, day_rows in grouped.items():
            if not start <= day <= end or day in present:
                skipped_days += 1
                continue
            if max(row["close"] for row in day_rows) < MIN_VALID_NIFTY_PRICE:
                skipped_days += 1
                print(f"  skipped invalid futures values for {day.isoformat()}")
                continue
            path = write_day(destination, day, day_rows)
            present.add(day)
            created_files += 1
            print(f"  wrote {path.name}: {len(day_rows)} candles")
        await asyncio.sleep(0.15)

    print(
        f"Complete: {created_files} files written, {fetched_rows} API rows read, "
        f"{skipped_days} existing/out-of-range day groups skipped"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument(
        "--destination",
        default=r"C:\Users\user\Desktop\nifty50 data\nifty_futures",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=14,
        help="Maximum API date span per request",
    )
    args = parser.parse_args()
    if args.chunk_days < 1:
        raise SystemExit("--chunk-days must be positive")
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(download(args))


if __name__ == "__main__":
    main()
