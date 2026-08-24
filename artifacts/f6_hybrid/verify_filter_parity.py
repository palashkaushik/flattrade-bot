"""Synthetic live-vs-reference parity check for the 5m index filter.

Problem: the grid CSV ends 2026-05-15, so the Aug 19 red-day bug cannot be
replayed from real data. This script generates a synthetic market:

  - 12 warm days: GREEN (steady uptrend) -> filter ends green, CE bias
  - target day: RED (steady downtrend from 09:15) -> PE only

It then asserts, minute by minute (555..900):
  1. LIVE filter (IndexFilter) allows PE and NEVER CE at any minute,
     including BEFORE the first 5m bar completes (09:15..09:19) — this is
     the exact scenario that produced the 09:20/09:21 CE bug on Aug 19.
  2. LIVE vs REFERENCE (PocketHTFFilter + filter_allows) agree on every minute.
  3. Day scoping: after a "restart" mid-day (seed only prior days, then feed
     live ticks from 10:30), prior-day snapshots are ignored -> PE, never CE.

Run: python verify_filter_parity.py
Exit 0 on full agreement, 1 otherwise.
"""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattrade_bot.strategies.pocket_money import IndexFilter, PocketMoneyEngine
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.pocket_money_backtest import (
    PocketHTFFilter, build_index_filter, filter_allows,
)

SESSION_START, SESSION_END = 555, 900  # 09:15 .. 15:00
MINUTES = list(range(SESSION_START, SESSION_END + 1))


def synth_day(day: str, direction: float, base: float = 18000.0) -> list:
    """One trading day of 376 1m rows; direction>0 -> uptrend (green)."""
    rows = []
    for i in range(376):
        price = base + direction * i * 0.9
        rows.append({
            "time": f"{day} {555 + i // 1:02d}:{0:02d}:00".replace(":00:00", ""),
        })
    # build proper minute timestamps
    out = []
    for i in range(376):
        m = SESSION_START + i
        hh, mm = divmod(m, 60)
        out.append({
            "time": f"{day} {hh:02d}:{mm:02d}:00",
            "open": base + direction * i * 0.9,
            "high": base + direction * (i * 0.9 + 0.6),
            "low": base + direction * (i * 0.9 - 0.6),
            "close": base + direction * (i * 0.9 + 0.3),
        })
    return out


def warm_days(n: int, direction: float = 1.0) -> list:
    days = []
    for i in range(n):
        d = datetime(2026, 8, 1 + i).strftime("%d-%m-%Y")
        days.append(synth_day(d, direction, base=18000.0 + i * 200))
    return days


def live_allowed_at(engine, minute: int) -> str:
    return engine.filter.allowed_side(minute)


def reference_snapshots(spot_rows, day: str, warm: list) -> dict:
    from artifacts.f6_hybrid.pocket_money_backtest import build_index_filter as bif
    # build a spot dict in the reference format
    spot = {"min": [], "open": [], "high": [], "low": [], "close": []}
    for r in spot_rows:
        hh, mm = divmod(SESSION_START + len(spot["min"]), 60)
        spot["min"].append(len(spot["min"]) + SESSION_START)
        spot["open"].append(r["open"])
        spot["high"].append(r["high"])
        spot["low"].append(r["low"])
        spot["close"].append(r["close"])
    grid_spots = {}
    for i, d in enumerate(warm):
        g = {"min": [], "open": [], "high": [], "low": [], "close": []}
        for r in d:
            g["min"].append(len(g["min"]) + SESSION_START)
            g["open"].append(r["open"])
            g["high"].append(r["high"])
            g["low"].append(r["low"])
            g["close"].append(r["close"])
        grid_spots[f"2026-08-{i+1:02d}"] = g
    import artifacts.f6_hybrid.pocket_money_backtest as pmb
    import grid_optimize_f6_atr as grid
    grid.GLOBAL_SPOT = grid_spots
    day_iso = f"2026-08-{len(warm):02d}"
    return bif(spot, day=day_iso, warm_days=len(warm))


def main() -> int:
    fails = 0

    # ── Scenario 1: warm GREEN days + RED target day ─────────────────────
    print("Scenario 1: 12 green warm days + red target day")
    warm = warm_days(12, direction=1.0)
    target = synth_day("19-08-2026", direction=-1.0)
    target_iso = "2026-08-19"

    ref_snaps = reference_snapshots(target, target_iso, warm)
    print(f"  reference snapshots: {len(ref_snaps)}")

    engine = PocketMoneyEngine()
    engine.set_today("19-08-2026")
    # seed warm days only (as the live API does: completed days only)
    for d in warm:
        engine.seed_spot_1m(d, today=None)
    # live feed minute by minute, replicating push_spot_tick semantics
    for i, row in enumerate(target):
        m = SESSION_START + i
        bar = {"open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"]}
        engine.filter.update_1m(bar, m, day="19-08-2026")
        engine.filter.update_forming(row["close"], m)
        live = live_allowed_at(engine, m)
        ref = filter_allows(ref_snaps, m)
        if live != ref:
            fails += 1
            print(f"  MISMATCH minute {m}: live={live} ref={ref}")
        if live == "CE":
            fails += 1
            print(f"  FAIL minute {m}: CE allowed on red day! live={live}")
            break
        if m in (560, 561, 570, 585, 630, 700, 780, 850):
            print(f"  minute {m}: live={live} ref={ref} forming={engine.filter.forming is not None}")
    # first completion minute = 555 + 4 = 559 (bar 555..559 commits at 560)
    first_ok = all(live_allowed_at(engine, m) == "PE" for m in range(555, 560))
    print(f"  pre-first-completion (555..559) all PE: {first_ok}")
    if not first_ok:
        fails += 1
    if fails == 0:
        print("  Scenario 1 PASS")

    # ── Scenario 2: mid-day restart — seed prior days, ticks from 10:30 ──
    print("\nScenario 2: mid-day restart (seed prior days, live from 10:30)")
    print("  NOTE: live has no 09:15..10:29 rows (API returns completed days only),")
    print("  so its UT trail legitimately diverges from the full-day reference.")
    print("  Assertion = SAFETY: never CE on a red day; PE/None both block the trade.")
    engine2 = PocketMoneyEngine()
    engine2.set_today("19-08-2026")
    for d in warm:
        engine2.seed_spot_1m(d, today=None)
    # restart at 10:30 (minute 630): no today rows seeded
    for m in range(630, 900):
        i = m - SESSION_START
        price = target[i]["close"]
        engine2.filter.update_1m(
            {"open": target[i]["open"], "high": target[i]["high"],
             "low": target[i]["low"], "close": target[i]["close"]},
            m, day="19-08-2026",
        )
        engine2.filter.update_forming(price, m)
        live = live_allowed_at(engine2, m)
        if live == "CE":
            fails += 1
            print(f"  FAIL minute {m}: CE allowed after restart on red day!")
    if fails == 0:
        print("  Scenario 2 PASS (no CE at any minute 630..899)")

    # ── Scenario 3: forming bar flips with live price (red day) ──────────
    print("\nScenario 3: forming bar tracks live price, not last close")
    engine3 = PocketMoneyEngine()
    engine3.set_today("19-08-2026")
    for d in warm:
        engine3.seed_spot_1m(d, today=None)
    engine3.filter.update_1m(
        {"open": target[0]["open"], "high": target[0]["high"],
         "low": target[0]["low"], "close": target[0]["close"]},
        555, day="19-08-2026",
    )
    # forming bar minute 556: live price spikes ABOVE white line while UT red
    spike = target[0]["close"] + 400
    engine3.filter.update_forming(spike, 556)
    live = live_allowed_at(engine3, 556)
    print(f"  spike price, UT red -> live={live} (expected None or PE, never CE)")
    if live == "CE":
        fails += 1
        print("  FAIL: CE allowed when UT red")
    # then price drops back below -> PE again
    engine3.filter.update_forming(target[0]["close"], 557)
    live = live_allowed_at(engine3, 557)
    print(f"  price back below, UT red -> live={live} (expected PE)")
    if live != "PE":
        fails += 1
        print("  FAIL: expected PE")

    print("\n" + ("ALL PASS" if fails == 0 else f"{fails} FAILURES"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())