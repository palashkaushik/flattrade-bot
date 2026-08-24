"""Probe exit-reason mechanics for wide-target sweep configs (5-day CPU oracle)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.marni_fib_backtest import simulate
from artifacts.f6_hybrid.marni_fib_core_combo_cache import extract_day_events


CACHE_DIR = ROOT / "artifacts" / "flattrade_day_cache"
DAYS = ("2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14")
BASE = {
    "s1_k_period": 12,
    "s1_d_period": 4,
    "min_span": 15.0,
    "setup_max_age": 45,
    "touch_buffer": 0.5,
    "zone_start": 0.5,
    "zone_end": 0.786,
    "fallback_target_level": 0.0,
    "option_point_threshold": 5.0,
}
CONFIGS = {
    "0.618/1.05": dict(BASE, target_level=0.618, stop_level=1.05),
    "0.786/1.13": dict(BASE, target_level=0.786, stop_level=1.13),
    "1.272/1.13": dict(BASE, target_level=1.272, stop_level=1.13),
    "1.618/1.618": dict(BASE, target_level=1.618, stop_level=1.618),
}


def fmt_minute(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def main() -> int:
    prepared = {}
    for day in DAYS:
        payload = extract_day_events(
            day,
            cache_dir=CACHE_DIR,
            min_span=BASE["min_span"],
            touch_buffer=BASE["touch_buffer"],
            setup_max_age=BASE["setup_max_age"],
            zone_start=BASE["zone_start"],
            zone_end=BASE["zone_end"],
            s1_k_period=BASE["s1_k_period"],
            s1_d_period=BASE["s1_d_period"],
            debug=False,
        )
        if payload and "signals" in payload:
            prepared[day] = payload

    for name, params in CONFIGS.items():
        reasons = Counter()
        total_pts = 0.0
        total_rs = 0.0
        total = 0
        wins = 0
        print(f"\n=== CONFIG {name} (5 days) ===")
        for day in prepared:
            p = prepared[day]
            events = [{**event, "timeframe": "combined"} for event in p["signals"]]
            trades = simulate(
                events,
                p["bars"],
                p["index_bars"],
                p["spot"],
                "combined",
                params["target_level"],
                params["stop_level"],
                concurrent=False,
                option_point_threshold=params["option_point_threshold"],
                fallback_target_level=params["fallback_target_level"],
                fixed_cost_per_trade=40.0,
                brokerage_per_order=0.0,
            )
            for trade in trades:
                reasons[trade["reason"]] += 1
                total_pts += trade["points"]
                total_rs += trade["rs_net"]
                total += 1
                wins += trade["points"] > 0
        print(f"  trades={total} wins={wins} net_pts={total_pts:+.2f} net_rs={total_rs:+,.2f}")
        for reason, count in reasons.most_common():
            print(f"    {reason}: {count}")

    print("\n=== SAMPLE TRADES (config 1.272/1.13) — with computed level prices ===")
    from artifacts.f6_hybrid.marni_fib_backtest import fib_price
    params = CONFIGS["1.272/1.13"]
    for day in list(prepared)[:2]:
        p = prepared[day]
        events = [{**event, "timeframe": "combined"} for event in p["signals"]]
        trades = simulate(
            events,
            p["bars"],
            p["index_bars"],
            p["spot"],
            "combined",
            params["target_level"],
            params["stop_level"],
            concurrent=False,
            option_point_threshold=params["option_point_threshold"],
            fallback_target_level=params["fallback_target_level"],
            fixed_cost_per_trade=40.0,
            brokerage_per_order=0.0,
        )
        for trade in trades:
            event = next(
                (e for e in events if e["minute"] == trade["entry_min"] and e["side"] == trade["side"]),
                {},
            )
            fh, fl = event.get("fib_high"), event.get("fib_low")
            orient = event.get("orientation")
            target = fib_price(fh, fl, params["target_level"], orient) if fh is not None else None
            stop = fib_price(fh, fl, params["stop_level"], orient) if fh is not None else None
            print(
                f"  {day} {fmt_minute(int(trade['entry_min']))}->{fmt_minute(int(trade['exit_min']))} "
                f"{trade['side']} src={event.get('fib_source')} fib=({fh},{fl}) o={orient} "
                f"target_lvl={target} stop_lvl={stop} entry={trade['entry']} exit={trade['exit']} "
                f"{trade['reason']} pts={trade['points']:+.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())