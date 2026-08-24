"""Read-only Flattrade replay for historical F6 signal verification."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grid_optimize_f6_atr as grid
from artifacts.flattrade_day_cache import (
    decode_active_strikes,
    dedupe_rows,
    load_day_cache,
    save_day_cache,
)
from flattrade_bot.broker.auto_login import automated_flattrade_login
from flattrade_bot.broker.history import FlattradeHistoryFetcher
from flattrade_bot.config import settings
from flattrade_bot.broker.network import force_ipv4


SESSION_START = 9 * 60 + 20
SESSION_END = 15 * 60
TIMEFRAMES = {"1m", "2m", "3m", "5m"}

HISTORICAL_CONTRACTS = {
    "2026-08-11": {
        ("CE", 24500): "41011",
        ("CE", 24450): "41009",
        ("CE", 24400): "41007",
        ("CE", 24350): "41005",
        ("PE", 24700): "41024",
        ("PE", 24650): "41019",
        ("PE", 24600): "41016",
        ("PE", 24550): "41014",
    }
}


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%d-%m-%Y %H:%M:%S")


def minute_of_day(value: str) -> int:
    parsed = parse_time(value)
    return parsed.hour * 60 + parsed.minute


def row_to_grid_candle(row: dict) -> grid.Candle:
    return grid.Candle(
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        minute=minute_of_day(row["time"]),
    )


async def fetch_range(
    fetcher: FlattradeHistoryFetcher,
    token: str,
    exchange: str,
    start_date: date,
    end_date: date,
    intrv: str = "1",
) -> list[dict]:
    """Fetches an exact date range instead of the rolling API cap."""
    import httpx

    timezone = ZoneInfo(settings.TRADING_TIMEZONE)
    start = datetime.combine(start_date, time.min, timezone)
    end = datetime.combine(end_date + timedelta(days=1), time.min, timezone)
    payload = {
        "uid": settings.FLATTRADE_USER_ID,
        "exch": exchange,
        "token": token,
        "st": str(int(start.timestamp())),
        "et": str(int(end.timestamp())),
        "intrv": intrv,
    }
    body = f"jData={json.dumps(payload)}&jKey={fetcher.auth_token}"
    with force_ipv4():
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(f"{fetcher.base_url}TPSeries", data=body)
            data = response.json()
    if not isinstance(data, list):
        return []
    rows = [
        {
            "time": row.get("time"),
            "open": float(row.get("into", 0.0)),
            "high": float(row.get("inth", 0.0)),
            "low": float(row.get("intl", 0.0)),
            "close": float(row.get("intc", 0.0)),
            "volume": float(row.get("v", 0.0)),
        }
        for row in data
    ]
    rows.reverse()
    return rows


def active_strikes(spot_rows: list[dict], target: date, itm: int = 1) -> dict[tuple[str, int], set[int]]:
    result: dict[tuple[str, int], set[int]] = {("CE", m): set() for m in range(SESSION_START, SESSION_END + 1)}
    result.update({("PE", m): set() for m in range(SESSION_START, SESSION_END + 1)})
    offset = 100 * itm
    for row in spot_rows:
        parsed = parse_time(row["time"])
        if parsed.date() != target:
            continue
        minute = parsed.hour * 60 + parsed.minute
        if not SESSION_START <= minute <= SESSION_END:
            continue
        atm = int(round(float(row["close"]) / 50.0) * 50)
        result[("CE", minute)].add(atm - offset)
        result[("PE", minute)].add(atm + offset)
    return result


async def resolve_contracts(
    fetcher: FlattradeHistoryFetcher,
    target: date,
    strikes: dict[tuple[str, int], set[int]],
) -> dict[tuple[str, int], dict]:
    """Resolve every option contract referenced by the target day's spot path."""
    expiry = target.strftime("%d%b%y").upper()
    contracts = {}
    for side in ("CE", "PE"):
        unique_strikes = sorted({
            strike
            for (mapped_side, _), values in strikes.items()
            if mapped_side == side
            for strike in values
        })
        for strike in unique_strikes:
            # SearchScrip exposes the currently listed weekly contract by
            # strike/side. On 2026-08-13 that is the 18AUG26 expiry; querying
            # the session date (13AUG26) returns no result because it was not
            # the expiry symbol used by the broker that day.
            info = await fetcher.search_option_token(f"NIFTY {strike} {side}")
            if not info:
                info = await fetcher.search_option_token(f"NIFTY {expiry} {strike} {side}")
            fallback_token = HISTORICAL_CONTRACTS.get(target.isoformat(), {}).get((side, strike))
            if info is None and fallback_token:
                info = {
                    "token": fallback_token,
                    "tsym": f"NIFTY{expiry}{'C' if side == 'CE' else 'P'}{strike}",
                    "dname": f"NIFTY {expiry} {strike} {side}",
                }
            if info:
                contracts[(side, strike)] = info
    return contracts


async def download_day_data(
    fetcher: FlattradeHistoryFetcher,
    target: date,
    with_futures: bool = False,
    itm: int = 1,
) -> tuple[list[dict], dict[tuple[str, int], set[int]], dict[tuple[str, int], dict], list[dict] | None]:
    """Fetch spot plus warm-up/session candles for all target-day contracts."""
    spot_rows = dedupe_rows(await fetch_range(fetcher, "26000", "NSE", target, target))
    strikes = active_strikes(spot_rows, target, itm=itm)
    contracts = await resolve_contracts(fetcher, target, strikes)
    rows_by_contract = await asyncio.gather(*(
        fetch_range(fetcher, info["token"], "NFO", target - timedelta(days=1), target)
        for info in contracts.values()
    ))
    for info, rows in zip(contracts.values(), rows_by_contract):
        info["rows"] = dedupe_rows(rows)
    futures_rows = None
    if with_futures:
        fut_info = await fetcher.search_futures_token("NIFTY")
        if fut_info:
            futures_rows = dedupe_rows(
                await fetch_range(fetcher, fut_info["token"], "NFO", target - timedelta(days=1), target, intrv="5")
            )
    return spot_rows, strikes, contracts, futures_rows


def replay_token(
    rows: list[dict],
    target: date,
    side: str,
    strike: int,
    active: dict[tuple[str, int], set[int]],
    params: dict,
    price_lookup: dict[tuple[str, int, str], float],
) -> list[dict]:
    signals = []
    for use_divergence in (True, False):
        replay_params = dict(params)
        replay_params["use_divergence"] = use_divergence
        tracker = grid.MTFTracker(replay_params)
        seen_times = set()
        for row in rows:
            parsed = parse_time(row["time"])
            if parsed.date() > target:
                continue
            timestamp = row["time"]
            if timestamp in seen_times:
                continue
            seen_times.add(timestamp)
            for tf, is_reverse, signal_type, entry, _atr in tracker.push_1m(row_to_grid_candle(row)):
                if parsed.date() != target or tf not in TIMEFRAMES:
                    continue
                minute = parsed.hour * 60 + parsed.minute
                if not SESSION_START <= minute <= SESSION_END:
                    continue
                if strike not in active.get((side, minute), set()):
                    continue
                target_side = ("PE" if side == "CE" else "CE") if is_reverse else side
                target_strikes = active.get((target_side, minute), set())
                target_strike = min(target_strikes) if target_strikes else strike
                target_entry = price_lookup.get((target_side, target_strike, timestamp), float(entry))
                signals.append({
                    "time": timestamp,
                    "side": target_side,
                    "source_side": side,
                    "source_strike": strike,
                    "strike": target_strike,
                    "timeframe": tf,
                    "signal": signal_type,
                    "reverse": is_reverse,
                    "entry_reference": round(float(target_entry), 2),
                    "trigger_reference": round(float(entry), 2),
                    "sl_points": 10.0,
                    "tp_points": 15.0,
                    "sl_reference": round(float(target_entry) - 10.0, 2),
                    "tp_reference": round(float(target_entry) + 15.0, 2),
                    "divergence": use_divergence,
                })
    return signals


async def run(args) -> dict:
    target = date.fromisoformat(args.date)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    cached = None if args.refresh_cache else (
        load_day_cache(cache_dir, target) if cache_dir else None
    )
    if cached:
        spot_rows = cached["spot_rows"]
        strikes = decode_active_strikes(cached["active_strikes"])
        contracts = {}
        for key in sorted(cached["contracts"]):
            side, strike = key.split(":", 1)
            contracts[(side, int(strike))] = cached["contracts"][key]
        futures_rows = cached.get("futures_rows")
        cache_status = "hit"
    else:
        if args.offline:
            raise SystemExit(f"No cache exists for {target.isoformat()}")
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
            raise SystemExit("Read-only Flattrade login failed")
        spot_rows, strikes, contracts, futures_rows = await download_day_data(
            FlattradeHistoryFetcher(token), target, with_futures=args.with_futures, itm=args.itm
        )
        if cache_dir:
            save_day_cache(cache_dir, target, spot_rows, strikes, contracts, futures_rows)
        cache_status = "written" if cache_dir else "disabled"

    price_lookup = {
        (side, strike, row["time"]): float(row["close"])
        for (side, strike), info in contracts.items()
        for row in info.get("rows", [])
    }
    params = {
        "s1_k": args.s1_k,
        "s1_d": 3,
        "s4_k": args.s4_k,
        "atr_period": 14,
        "atr_sl_mult": 2.0,
        "atr_tp_mult": 6.0,
        "f6_s4_thresh": 79.5,
        "f6_s1_thresh": 20.5,
        "consec_loss": 4,
    }
    signals = []
    for (side, strike), info in contracts.items():
        signals.extend(replay_token(
            info.get("rows", []), target, side, strike, strikes, params, price_lookup
        ))
    signals.sort(key=lambda item: (item["time"], item["divergence"], item["side"], item["timeframe"]))
    return {
        "date": args.date,
        "source": "Flattrade TPSeries read-only replay",
        "timeframes": sorted(TIMEFRAMES),
        "params": params,
        "fixed_exits": {"sl_points": 10.0, "tp_points": 15.0},
        "spot_window": {
            "rows": len(spot_rows),
            "first": spot_rows[0]["time"] if spot_rows else None,
            "last": spot_rows[-1]["time"] if spot_rows else None,
            "session_sample": [
                {"time": row["time"], "close": row["close"]}
                for row in spot_rows
                if parse_time(row["time"]).date() == target
                and SESSION_START <= minute_of_day(row["time"]) <= SESSION_END
            ][:10],
        },
        "contracts": {
            f"{side}:{strike}": {
                "token": info["token"],
                "tsym": info.get("tsym", ""),
                "dname": info.get("dname", ""),
            }
            for (side, strike), info in contracts.items()
        },
        "cache": {"status": cache_status, "path": str(cache_path(args, target))},
        "signals": signals,
    }


def cache_path(args, target: date) -> Path:
    """Keep output metadata independent of the cache implementation details."""
    return (Path(args.cache_dir) / f"{target.isoformat()}.json.gz") if args.cache_dir else Path()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-08-11")
    parser.add_argument("--days-back", type=int, default=2)
    parser.add_argument("--s1-k", type=int, default=9)
    parser.add_argument("--s4-k", type=int, default=60)
    parser.add_argument("--output", default="artifacts/flattrade_signals_2026-08-11.json")
    parser.add_argument(
        "--cache-dir",
        default="artifacts/flattrade_day_cache",
        help="Compressed local day cache; used before broker access",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Fail instead of logging in when cache is absent")
    parser.add_argument("--with-futures", action="store_true", help="Also fetch Nifty futures 5m for bias")
    parser.add_argument("--itm", type=int, default=1, help="Option ITM depth (1=ATM-100, 2=ATM-200)")
    args = parser.parse_args()
    result = asyncio.run(run(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"Signals written to {output}")


if __name__ == "__main__":
    main()
