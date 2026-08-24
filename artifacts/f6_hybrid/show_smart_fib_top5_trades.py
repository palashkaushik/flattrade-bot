"""Print Aug 12-14 trades for the five distinct Smart Fib top contenders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.marni_fib_backtest import simulate
from artifacts.f6_hybrid.marni_fib_core_combo_cache import extract_day_events


CACHE_DIR = ROOT / "artifacts" / "flattrade_day_cache"
DAYS = ("2026-08-12", "2026-08-13", "2026-08-14")
SIGNAL = {
    "s1_k_period": 12,
    "s1_d_period": 4,
    "min_span": 15.0,
    "setup_max_age": 45,
    "touch_buffer": 0.5,
    "zone_start": 0.5,
    "zone_end": 0.786,
}
STOPS = (1.13, 1.155, 1.25, 1.382, 1.618)


def fmt_minute(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def main() -> int:
    prepared_by_day = {}
    for day in DAYS:
        prepared_by_day[day] = extract_day_events(
            day,
            cache_dir=CACHE_DIR,
            **SIGNAL,
            debug=False,
        )

    report = {"signal": SIGNAL, "contenders": {}}
    for stop in STOPS:
        contender = f"stop_{stop:g}"
        report["contenders"][contender] = {}
        for day in DAYS:
            prepared = prepared_by_day[day]
            events = [{**event, "timeframe": "combined"} for event in prepared["signals"]]
            trades = simulate(
                events,
                prepared["bars"],
                prepared["index_bars"],
                prepared["spot"],
                "combined",
                0.5,
                stop,
                concurrent=False,
                option_point_threshold=5.0,
                fallback_target_level=0.0,
            )
            event_by_key = {
                (int(event["minute"]), event["side"], event["symbol"]): event
                for event in events
            }
            rows = []
            for trade in trades:
                event = event_by_key.get(
                    (trade["entry_min"], trade["side"], trade["symbol"]),
                    {},
                )
                rows.append({
                    "entry_time": fmt_minute(int(trade["entry_min"])),
                    "exit_time": fmt_minute(int(trade["exit_min"])),
                    "side": trade["side"],
                    "strike": int(event.get("strike", 0)),
                    "trigger": event.get("trigger"),
                    "entry": trade["entry"],
                    "exit": trade["exit"],
                    "reason": trade["reason"],
                    "points": trade["points"],
                    "net_rs": trade["rs_net"],
                    "fee_rs": trade["fee"],
                })
            report["contenders"][contender][day] = {
                "trades": len(rows),
                "net_points": round(sum(row["points"] for row in rows), 2),
                "net_rs": round(sum(row["net_rs"] for row in rows), 2),
                "trades_detail": rows,
            }

    output = ROOT / "artifacts" / "f6_hybrid" / "smart_fib_top5_trades_2026-08-12_14.json"
    output.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for contender, days in report["contenders"].items():
        print(f"\n=== {contender} ===")
        for day, result in days.items():
            print(f"{day}: {result['trades']} trades | {result['net_points']:+.2f} pts | Rs {result['net_rs']:+,.2f}")
            for index, trade in enumerate(result["trades_detail"], start=1):
                print(
                    f"  {index:02d}. {trade['entry_time']}-{trade['exit_time']} "
                    f"{trade['side']} {trade['strike']} {trade['trigger']} "
                    f"{trade['entry']:.2f}->{trade['exit']:.2f} {trade['reason']} "
                    f"pts={trade['points']:+.2f} net=Rs {trade['net_rs']:+,.2f}"
                )
    print(f"\nJSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
