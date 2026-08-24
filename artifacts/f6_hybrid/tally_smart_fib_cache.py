"""Tally Smart Fib trades against a downloaded day-cache snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.flattrade_day_cache import load_day_cache
from artifacts.f6_hybrid import marni_fib_core_combo_cache as smart_core


DEFAULT_DATES = ("2026-08-12", "2026-08-13", "2026-08-14")
TIMEFRAMES = ("1m", "2m", "3m", "5m", "combined")
TARGET_LEVEL = 0.29
STOP_LEVELS = (1.155, 1.25)


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%d-%m-%Y %H:%M:%S")


def cache_coverage(cache: dict) -> dict:
    contracts = cache.get("contracts", {})
    return {
        "spot_rows": len(cache.get("spot_rows", [])),
        "contracts": len(contracts),
        "option_rows": sum(len(info.get("rows", [])) for info in contracts.values()),
        "symbols": sorted(
            f"{key}={info.get('tsym', '')}" for key, info in contracts.items()
        ),
    }


def trade_match(cache: dict, day: str, trade: dict) -> dict:
    symbol = trade.get("symbol", "")
    rows = []
    for info in cache.get("contracts", {}).values():
        if info.get("tsym") == symbol:
            rows = info.get("rows", [])
            break
    current_minutes = {
        parse_time(row["time"]).hour * 60 + parse_time(row["time"]).minute
        for row in rows
        if row.get("time") and parse_time(row["time"]).strftime("%Y-%m-%d") == day
    }
    entry_ok = int(trade["entry_min"]) in current_minutes
    exit_ok = int(trade["exit_min"]) in current_minutes
    return {
        "symbol": symbol,
        "entry_row": entry_ok,
        "exit_row": exit_ok,
        "matched": entry_ok and exit_ok,
    }


def run(cache_dir: Path, dates: list[str]) -> dict:
    coverage = {}
    results = {}
    all_trades = {}
    for day in dates:
        cache = load_day_cache(cache_dir, datetime.strptime(day, "%Y-%m-%d").date())
        if cache is None:
            raise FileNotFoundError(f"Missing cache for {day}: {cache_dir}")
        coverage[day] = cache_coverage(cache)
        day_result = smart_core.process_day(
            day,
            TIMEFRAMES,
            (TARGET_LEVEL,),
            STOP_LEVELS,
            cache_dir=cache_dir,
            debug=False,
        )
        for key, trades in day_result.items():
            matched = [trade_match(cache, day, trade) for trade in trades]
            results.setdefault(key, {})[day] = {
                "trades": len(trades),
                "matched_trades": sum(item["matched"] for item in matched),
                "unmatched_trades": sum(not item["matched"] for item in matched),
                "net_points": round(sum(float(t.get("points", 0.0)) for t in trades), 2),
                "net_rs": round(sum(float(t.get("rs_net", 0.0)) for t in trades), 2),
                "trades_detail": [
                    {
                        "entry": f"{trade['entry_min'] // 60:02d}:{trade['entry_min'] % 60:02d}",
                        "exit": f"{trade['exit_min'] // 60:02d}:{trade['exit_min'] % 60:02d}",
                        "side": trade["side"],
                        "symbol": trade["symbol"],
                        "reason": trade["reason"],
                        "points": trade["points"],
                        "rs_net": trade["rs_net"],
                        "match": item,
                    }
                    for trade, item in zip(trades, matched)
                ],
            }
            all_trades.setdefault(key, []).extend(trades)

    aggregate = {}
    for key, trades in all_trades.items():
        aggregate[key] = {
            "trades": len(trades),
            "matched_trades": sum(
                trade_match(
                    load_day_cache(cache_dir, datetime.strptime(t["date"], "%Y-%m-%d").date()),
                    t["date"],
                    t,
                )["matched"]
                for t in trades
            ),
            "net_points": round(sum(float(t.get("points", 0.0)) for t in trades), 2),
            "net_rs": round(sum(float(t.get("rs_net", 0.0)) for t in trades), 2),
        }

    return {
        "cache_dir": str(cache_dir),
        "dates": dates,
        "coverage": coverage,
        "by_configuration": results,
        "aggregate": aggregate,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Smart Fib Historical Trade Tally",
        "",
        f"- Cache: `{report['cache_dir']}`",
        f"- Dates: {', '.join(report['dates'])}",
        "- Entry/exit matching: every reported trade must have both timestamps in its downloaded option contract rows.",
        "",
        "## Download Coverage",
        "",
        "| Date | Spot rows | Contracts | Option rows |",
        "|---|---:|---:|---:|",
    ]
    for day in report["dates"]:
        item = report["coverage"][day]
        lines.append(f"| {day} | {item['spot_rows']} | {item['contracts']} | {item['option_rows']} |")
    lines.extend([
        "",
        "## Aggregate Tally",
        "",
        "| Configuration | Trades | Matched | Net points | Net Rs |",
        "|---|---:|---:|---:|---:|",
    ])
    for key, item in sorted(report["aggregate"].items()):
        lines.append(
            f"| `{key}` | {item['trades']} | {item['matched_trades']} | "
            f"{item['net_points']:+.2f} | {item['net_rs']:+.2f} |"
        )
    lines.extend(["", "## Daily Details", ""])
    for key, by_day in sorted(report["by_configuration"].items()):
        lines.append(f"### `{key}`")
        lines.append("")
        for day in report["dates"]:
            item = by_day.get(day, {"trades": 0, "matched_trades": 0, "net_points": 0, "net_rs": 0, "trades_detail": []})
            lines.append(
                f"- {day}: {item['trades']} trades, {item['matched_trades']} matched, "
                f"net {item['net_points']:+.2f} points / Rs {item['net_rs']:+.2f}"
            )
            for trade in item["trades_detail"]:
                lines.append(
                    f"  - {trade['entry']} -> {trade['exit']} {trade['side']} "
                    f"`{trade['symbol']}` {trade['reason']} "
                    f"{trade['points']:+.2f} points / Rs {trade['rs_net']:+.2f}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="artifacts/flattrade_day_cache_smart_fib")
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument("--output", default="artifacts/f6_hybrid/smart_fib_aug_12_14_tally.md")
    args = parser.parse_args()
    report = run(Path(args.cache_dir), args.dates)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"Markdown tally written to {output}")


if __name__ == "__main__":
    main()
