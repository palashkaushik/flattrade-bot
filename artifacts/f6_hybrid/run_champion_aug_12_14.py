"""Run the Smart Fib champion (0.786/1.13) on specific day-cache days and print every trade."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid import marni_fib_core_combo_cache as smart_core
from artifacts.f6_hybrid import smart_fib_optimus_gpu as optimus
from artifacts.f6_hybrid.marni_fib_backtest import LOT_SIZE, simulate

CHAMPION = dict(
    min_span=15.0,
    touch_buffer=0.5,
    setup_max_age=45,
    zone_start=0.5,
    zone_end=0.786,
    s1_k_period=12,
    s1_d_period=4,
)
TARGET = 0.786
STOP = 1.13
FIXED_COST = 40.0
DAYS = ("2026-08-12", "2026-08-13", "2026-08-14")


def clock(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def main() -> None:
    all_trades = []
    for day in DAYS:
        payload = smart_core.extract_day_events(day, debug=False, **CHAMPION)
        if not payload:
            print(f"{day}: no cache payload")
            continue
        selected, _ = optimus._select_day_events(payload)
        events = [{**signal, "timeframe": "combined"} for signal in selected]
        first_event_by_minute = {}
        for signal in selected:
            first_event_by_minute.setdefault(signal["minute"], signal)
        trades = simulate(
            events,
            payload["bars"],
            payload["index_bars"],
            payload["spot"],
            "combined",
            target_level=TARGET,
            stop_level=STOP,
            fixed_cost_per_trade=FIXED_COST,
        )
        all_trades.extend(trades)
        print(f"\n=== {day} — Smart Fib champion S1(12,4) span15 age45 buf0.5 zone0.5-0.786 | TP {TARGET} SL {STOP} ===")
        print(f"{'In':6s} {'Out':6s} {'Src':6s} {'Side':3s} {'Strike':7s} {'Entry':7s} {'Exit':7s} {'Rsn':4s} {'Pts':7s} {'Fee':6s} {'Net Rs':9s}")
        print("-" * 80)
        for t in trades:
            source = first_event_by_minute.get(t["entry_min"], {}).get("trigger", "?")
            print(
                f"{clock(t['entry_min']):6s} {clock(t['exit_min']):6s} {source:6s} {t['side']:3s} "
                f"{t['symbol']:7s} {t['entry']:7.2f} {t['exit']:7.2f} {t['reason']:4s} "
                f"{t['points']:+7.2f} {t['fee']:6.1f} {t['rs_net']:+9.2f}"
            )
        if trades:
            wins = [t for t in trades if t["rs_net"] > 0]
            losses = [t for t in trades if t["rs_net"] <= 0]
            loss_total = abs(sum(t["rs_net"] for t in losses))
            pf = round(sum(t["rs_net"] for t in wins) / loss_total, 4) if loss_total else float("inf")
            net = round(sum(t["rs_net"] for t in trades), 2)
            print(
                f"--- {len(trades)} trades | WR {len(wins)/len(trades)*100:.1f}% | "
                f"net {net:+,.2f} Rs ({net/LOT_SIZE:+,.2f} pts) | PF {pf}"
            )
        else:
            print("--- 0 trades")

    print(f"\n=== AGGREGATE {DAYS[0]}..{DAYS[2]} ===")
    if all_trades:
        wins = [t for t in all_trades if t["rs_net"] > 0]
        losses = [t for t in all_trades if t["rs_net"] <= 0]
        loss_total = abs(sum(t["rs_net"] for t in losses))
        pf = round(sum(t["rs_net"] for t in wins) / loss_total, 4) if loss_total else float("inf")
        net = round(sum(t["rs_net"] for t in all_trades), 2)
        print(
            f"{len(all_trades)} trades | WR {len(wins)/len(all_trades)*100:.1f}% | "
            f"net {net:+,.2f} Rs ({net/LOT_SIZE:+,.2f} pts) | PF {pf}"
        )
    else:
        print("0 trades")


if __name__ == "__main__":
    main()