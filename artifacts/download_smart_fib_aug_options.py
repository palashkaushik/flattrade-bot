"""Download exact Smart Fib option contracts for Aug 12-14, 2026.

This is a read-only historical download. It keeps the existing replay cache
untouched and stores a separate cache containing ATM, first-ITM, and
second-ITM candidates for both CE and PE.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.flattrade_day_cache import dedupe_rows, save_day_cache
from artifacts.replay_flattrade_signals import fetch_range
from flattrade_bot.broker.auto_login import automated_flattrade_login
from flattrade_bot.broker.history import FlattradeHistoryFetcher
from flattrade_bot.config import settings


DEFAULT_DATES = ("2026-08-12", "2026-08-13", "2026-08-14")
HISTORICAL_EXPIRY = {
    "2026-08-11": "18AUG26",
    "2026-08-12": "18AUG26",
    "2026-08-13": "18AUG26",
    "2026-08-14": "18AUG26",
}
SESSION_START = 555
SESSION_END = 900


def parse_broker_time(value: str):
    from datetime import datetime

    return datetime.strptime(value, "%d-%m-%Y %H:%M:%S")


def smart_fib_active_strikes(spot_rows: list[dict], target: date) -> dict[tuple[str, int], set[int]]:
    """Track every candidate that can be selected at each trigger minute."""
    active = {
        (side, minute): set()
        for side in ("CE", "PE")
        for minute in range(SESSION_START, SESSION_END + 1)
    }
    for row in spot_rows:
        parsed = parse_broker_time(row["time"])
        if parsed.date() != target:
            continue
        minute = parsed.hour * 60 + parsed.minute
        if not SESSION_START <= minute <= SESSION_END:
            continue
        atm = int(round(float(row["close"]) / 50.0) * 50)
        active[("CE", minute)].update((atm, atm - 50, atm - 100))
        active[("PE", minute)].update((atm, atm + 50, atm + 100))
    return active


def expected_expiry(target: date) -> str:
    value = HISTORICAL_EXPIRY.get(target.isoformat())
    if value is None:
        raise ValueError(f"No historical expiry mapping for {target.isoformat()}")
    return value


async def resolve_contract(
    fetcher: FlattradeHistoryFetcher,
    target: date,
    side: str,
    strike: int,
) -> dict[str, str] | None:
    expiry = expected_expiry(target)
    exact = await fetcher.search_option_token(f"NIFTY {expiry} {strike} {side}")
    if exact:
        identity = f"{exact.get('tsym', '')} {exact.get('dname', '')}".upper()
        if expiry in identity:
            return exact

    # A generic query is only accepted when the broker still returns the
    # historical expiry. This prevents silently downloading a later series.
    generic = await fetcher.search_option_token(f"NIFTY {strike} {side}")
    if generic:
        identity = f"{generic.get('tsym', '')} {generic.get('dname', '')}".upper()
        if expiry in identity:
            return generic
    return None


async def download_day(
    fetcher: FlattradeHistoryFetcher,
    target: date,
    cache_dir: Path,
) -> dict:
    spot_rows = dedupe_rows(await fetch_range(fetcher, "26000", "NSE", target, target))
    if not spot_rows:
        raise RuntimeError(f"No NIFTY spot rows returned for {target.isoformat()}")

    active = smart_fib_active_strikes(spot_rows, target)
    unique = sorted({
        (side, strike)
        for (side, _minute), strikes in active.items()
        for strike in strikes
    })

    resolved: dict[tuple[str, int], dict] = {}
    for side, strike in unique:
        info = await resolve_contract(fetcher, target, side, strike)
        if info is None:
            raise RuntimeError(
                f"Historical contract not found: {target.isoformat()} {side} {strike} "
                f"{expected_expiry(target)}"
            )
        resolved[(side, strike)] = info

    rows_by_contract = await asyncio.gather(*(
        fetch_range(
            fetcher,
            info["token"],
            "NFO",
            target - timedelta(days=1),
            target,
        )
        for info in resolved.values()
    ))
    for info, rows in zip(resolved.values(), rows_by_contract):
        info["rows"] = dedupe_rows(rows)

    destination = save_day_cache(cache_dir, target, spot_rows, active, resolved, None)
    return {
        "date": target.isoformat(),
        "expiry": expected_expiry(target),
        "cache": str(destination),
        "spot_rows": len(spot_rows),
        "contracts": len(resolved),
        "option_rows": sum(len(info.get("rows", [])) for info in resolved.values()),
        "missing_rows": [
            f"{side}:{strike}"
            for (side, strike), info in resolved.items()
            if not info.get("rows")
        ],
        "contract_symbols": sorted(
            f"{side}:{strike}={info.get('tsym', '')}"
            for (side, strike), info in resolved.items()
        ),
    }


async def run(args: argparse.Namespace) -> list[dict]:
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

    fetcher = FlattradeHistoryFetcher(token)
    cache_dir = Path(args.cache_dir)
    reports = []
    for value in args.dates:
        target = date.fromisoformat(value)
        report = await download_day(fetcher, target, cache_dir)
        reports.append(report)
        print(
            f"{value}: {report['contracts']} contracts, "
            f"{report['option_rows']} option candles -> {report['cache']}"
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument(
        "--cache-dir",
        default="artifacts/flattrade_day_cache_smart_fib",
    )
    parser.add_argument("--visible", action="store_true")
    args = parser.parse_args()
    reports = asyncio.run(run(args))
    print(__import__("json").dumps(reports, indent=2))


if __name__ == "__main__":
    main()
