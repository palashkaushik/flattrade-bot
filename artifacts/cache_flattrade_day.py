"""Download one Flattrade day once for repeated offline replays."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.flattrade_day_cache import save_day_cache
from artifacts.replay_flattrade_signals import download_day_data
from flattrade_bot.broker.auto_login import automated_flattrade_login
from flattrade_bot.broker.history import FlattradeHistoryFetcher
from flattrade_bot.config import settings


async def run(args: argparse.Namespace) -> Path:
    target = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
    token = automated_flattrade_login(
        user_id=settings.FLATTRADE_USER_ID,
        password=settings.FLATTRADE_PASSWORD,
        totp_key=settings.FLATTRADE_TOTP_KEY,
        api_key=settings.FLATTRADE_API_KEY,
        api_secret=settings.FLATTRADE_API_SECRET,
        headless=not args.visible,
    )
    if not token:
        raise SystemExit("Read-only Flattrade login failed")
    spot_rows, active, contracts, futures_rows = await download_day_data(
        FlattradeHistoryFetcher(token), target, with_futures=True, itm=args.itm
    )
    destination = save_day_cache(Path(args.cache_dir), target, spot_rows, active, contracts, futures_rows)
    print(f"Cached {target.isoformat()} to {destination}")
    print(f"Spot candles: {len(spot_rows)}")
    print(f"Option contracts: {len(contracts)}")
    print(f"Option candles: {sum(len(info.get('rows', [])) for info in contracts.values())}")
    print(f"Futures 5m candles: {len(futures_rows) if futures_rows else 0}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Trading date in YYYY-MM-DD; defaults to yesterday")
    parser.add_argument(
        "--cache-dir",
        default="artifacts/flattrade_day_cache",
        help="Directory for compressed day snapshots",
    )
    parser.add_argument("--visible", action="store_true", help="Show the broker login browser")
    parser.add_argument("--itm", type=int, default=2, help="Option ITM depth (1=ATM-100, 2=ATM-200)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
