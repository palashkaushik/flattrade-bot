"""Local compressed cache for one Flattrade replay day."""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path
from typing import Any


CACHE_VERSION = 1


def cache_path(cache_dir: Path, target: date) -> Path:
    """Return the stable cache filename for a trading date."""
    return cache_dir / f"{target.isoformat()}.json.gz"


def dedupe_rows(rows: list[dict]) -> list[dict]:
    """Keep one candle per broker timestamp and restore chronological order."""
    by_time = {str(row.get("time")): row for row in rows if row.get("time")}
    return [by_time[key] for key in sorted(by_time)]


def encode_active_strikes(active: dict[tuple[str, int], set[int]]) -> dict[str, list[int]]:
    return {
        f"{side}:{minute}": sorted(strikes)
        for (side, minute), strikes in sorted(active.items())
    }


def decode_active_strikes(payload: dict[str, list[int]]) -> dict[tuple[str, int], set[int]]:
    active: dict[tuple[str, int], set[int]] = {}
    for key, strikes in payload.items():
        side, minute = key.split(":", 1)
        active[(side, int(minute))] = {int(strike) for strike in strikes}
    return active


def save_day_cache(
    cache_dir: Path,
    target: date,
    spot_rows: list[dict],
    active: dict[tuple[str, int], set[int]],
    contracts: dict[tuple[str, int], dict[str, Any]],
    futures_rows: list[dict] | None = None,
) -> Path:
    """Write a complete replay snapshot without credentials or session tokens."""
    payload = {
        "schema_version": CACHE_VERSION,
        "date": target.isoformat(),
        "interval": "1",
        "warmup_start": target.fromordinal(target.toordinal() - 1).isoformat(),
        "spot_rows": dedupe_rows(spot_rows),
        "active_strikes": encode_active_strikes(active),
        "contracts": {
            f"{side}:{strike}": {
                "token": info["token"],
                "tsym": info.get("tsym", ""),
                "dname": info.get("dname", ""),
                "rows": dedupe_rows(info.get("rows", [])),
            }
            for (side, strike), info in sorted(contracts.items())
        },
        "futures_rows": dedupe_rows(futures_rows) if futures_rows else [],
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_path(cache_dir, target)
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return destination


def load_day_cache(cache_dir: Path, target: date) -> dict[str, Any] | None:
    """Load and validate a local replay snapshot, returning None when absent."""
    source = cache_path(cache_dir, target)
    if not source.exists():
        return None
    try:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != CACHE_VERSION or payload.get("date") != target.isoformat():
        return None
    return payload
