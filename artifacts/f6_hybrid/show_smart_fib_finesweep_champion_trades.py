"""Print chart-verifiable trades for the fine-sweep Smart Fib champion exit.

Signal: S1(12,4)/span15/age45/buf0.5/zone0.5-0.786
Exit:   target 0.618, fallback 0, threshold 5, stop 1.05
Cost:   fixed all-in Rs40 per completed trade (brokerage + taxes approx).
Days:   2026-08-12, 2026-08-13, 2026-08-14 (live tick cache).
"""

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
PARAMS = {
    "s1_k_period": 12,
    "s1_d_period": 4,
    "min_span": 15.0,
    "setup_max_age": 45,
    "touch_buffer": 0.5,
    "zone_start": 0.5,
    "zone_end": 0.786,
    "target_level": 0.618,
    "fallback_target_level": 0.0,
    "option_point_threshold": 5.0,
    "stop_level": 1.05,
    "fixed_cost_per_trade": 40.0,
}


def fmt_minute(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def main() -> int:
    report = {"parameters": PARAMS, "days": {}}
    for day in DAYS:
        prepared = extract_day_events(
            day,
            cache_dir=CACHE_DIR,
            min_span=PARAMS["min_span"],
            touch_buffer=PARAMS["touch_buffer"],
            setup_max_age=PARAMS["setup_max_age"],
            zone_start=PARAMS["zone_start"],
            zone_end=PARAMS["zone_end"],
            s1_k_period=PARAMS["s1_k_period"],
            s1_d_period=PARAMS["s1_d_period"],
            debug=False,
        )
        if not prepared:
            raise RuntimeError(f"no cache payload for {day}")
        events = [{**event, "timeframe": "combined"} for event in prepared["signals"]]
        trades = simulate(
            events,
            prepared["bars"],
            prepared["index_bars"],
            prepared["spot"],
            "combined",
            PARAMS["target_level"],
            PARAMS["stop_level"],
            concurrent=False,
            option_point_threshold=PARAMS["option_point_threshold"],
            fallback_target_level=PARAMS["fallback_target_level"],
            fixed_cost_per_trade=PARAMS["fixed_cost_per_trade"],
            brokerage_per_order=0.0,
        )
        event_by_key = {
            (int(event["minute"]), event["side"], event["symbol"]): event
            for event in events
        }
        rows = []
        for trade in trades:
            event = event_by_key.get((trade["entry_min"], trade["side"], trade["symbol"]), {})
            rows.append({
                "entry_time": fmt_minute(int(trade["entry_min"])),
                "exit_time": fmt_minute(int(trade["exit_min"])),
                "side": trade["side"],
                "strike": int(event.get("strike", 0)),
                "symbol": trade["symbol"],
                "trigger": event.get("trigger"),
                "fib_source": event.get("fib_source"),
                "signal_minute": fmt_minute(int(event["signal_minute"])) if event.get("signal_minute") else None,
                "fib_high": event.get("fib_high"),
                "fib_low": event.get("fib_low"),
                "orientation": event.get("orientation"),
                "s1_value": event.get("s1_value"),
                "s1_turn": event.get("s1_turn"),
                "entry": trade["entry"],
                "exit": trade["exit"],
                "reason": trade["reason"],
                "points": trade["points"],
                "net_rs": trade["rs_net"],
                "fee_rs": trade["fee"],
            })
        report["days"][day] = {
            "trades": len(rows),
            "wins": sum(row["points"] > 0 for row in rows),
            "net_points": round(sum(row["points"] for row in rows), 2),
            "net_rs": round(sum(row["net_rs"] for row in rows), 2),
            "trades_detail": rows,
        }

    output = ROOT / "artifacts" / "f6_hybrid" / "smart_fib_finesweep_champion_trades_2026-08-12_14.json"
    output.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    old_path = ROOT / "artifacts" / "f6_hybrid" / "smart_fib_max_net_trades_2026-08-12_14.json"
    print("=== NEW CHAMPION: target 0.618 / stop 1.05 / Rs40 fixed cost ===")
    for day, result in report["days"].items():
        print(f"\n{day}: {result['trades']} trades | {result['wins']} wins | "
              f"{result['net_points']:+.2f} points | Rs {result['net_rs']:+,.2f}")
        for trade in result["trades_detail"]:
            print(
                f"  {trade['entry_time']} -> {trade['exit_time']} "
                f"{trade['side']} {trade['symbol']} "
                f"trigger={trade['trigger']} source={trade['fib_source']} "
                f"entry={trade['entry']:.2f} exit={trade['exit']:.2f} "
                f"{trade['reason']} points={trade['points']:+.2f} "
                f"net=Rs {trade['net_rs']:+,.2f}"
            )

    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        print("\n=== COMPARISON vs OLD CHAMPION (target 0.5 / stop 1.13) ===")
        print(f"{'DAY':<12} {'OLD tr':>7} {'OLD pts':>10} {'OLD Rs':>12} | {'NEW tr':>7} {'NEW pts':>10} {'NEW Rs':>12}")
        for day in DAYS:
            o = old["days"][day]
            n = report["days"][day]
            print(f"{day:<12} {o['trades']:>7} {o['net_points']:>+10.2f} {o['net_rs']:>+12.2f} | "
                  f"{n['trades']:>7} {n['net_points']:>+10.2f} {n['net_rs']:>+12.2f}")

    print(f"\nJSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())