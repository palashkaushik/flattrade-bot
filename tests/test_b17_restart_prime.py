"""Regression test: restart mid-session must not re-route historical B17 events.

The strategy's ``last_evaluated_minute`` starts at -1, so the first
evaluation after a restart replays the whole morning's buffers and returns
every past signal as "new". main.py primes the strategy by discarding the
first evaluation; this test pins that contract: after priming at minute M,
no event with minute <= M may ever be routed again.
"""

import json
import os
from datetime import date

import pytest

from artifacts.flattrade_day_cache import load_day_cache
from artifacts.f6_hybrid.marni_fib_core_combo_cache import GLOBAL_CACHE_DIR
from flattrade_bot.strategies.smart_fib_combined import LiveSmartFibCombinedStrategy

CACHED_DAY = date(2026, 8, 13)
RESTART_MINUTE = 600  # 10:00


@pytest.fixture(scope="module")
def day_payload():
    payload = load_day_cache(GLOBAL_CACHE_DIR, CACHED_DAY)
    if payload is None:
        pytest.skip(f"No cached payload for {CACHED_DAY}")
    return payload


def _build_strategy(day_payload) -> LiveSmartFibCombinedStrategy:
    strategy = LiveSmartFibCombinedStrategy()
    strategy.set_today(CACHED_DAY)
    strategy.add_spot_rows(day_payload["spot_rows"])
    for key, info in day_payload["contracts"].items():
        side, strike = key.split(":", 1)
        strategy.add_contract_rows(
            side,
            int(strike),
            info.get("tsym") or f"{side}:{strike}",
            info.get("token") or "",
            info["rows"],
        )
    return strategy


def test_restart_prime_suppresses_historical_replay(day_payload):
    strategy = _build_strategy(day_payload)
    assert len(strategy.spot_rows.get(CACHED_DAY.isoformat(), [])) > 0

    replay_events = strategy.evaluate(RESTART_MINUTE)
    assert strategy.last_evaluated_minute == RESTART_MINUTE

    # Evaluating the same minute again must never return anything.
    assert strategy.evaluate(RESTART_MINUTE) == []

    # Minutes after the prime may only carry events newer than the prime.
    later = strategy.evaluate(RESTART_MINUTE + 5)
    for event in later:
        assert int(event["minute"]) > RESTART_MINUTE, (
            f"event at minute {event['minute']} leaked past the restart prime at {RESTART_MINUTE}"
        )


def test_restart_prime_never_routes_pre_restart_minutes(day_payload):
    strategy = _build_strategy(day_payload)
    strategy.evaluate(RESTART_MINUTE)  # prime at 10:00

    minute = RESTART_MINUTE + 1
    while minute < 900:
        events = strategy.evaluate(minute)
        for event in events:
            assert int(event["minute"]) > RESTART_MINUTE
        minute += 5


def test_single_instance_guard_blocks_live_pid(tmp_path, monkeypatch):
    import subprocess
    import sys

    from flattrade_bot import config
    from flattrade_bot.main import TradingEngine

    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        runtime = tmp_path / "bot.runtime.json"
        runtime.write_text(
            json.dumps({"pid": sleeper.pid, "live_orders": True}), encoding="utf-8"
        )
        monkeypatch.setattr(config.settings, "BOT_RUNTIME_FILE", runtime)
        assert TradingEngine._single_instance_guard() is False
    finally:
        sleeper.terminate()


def test_single_instance_guard_allows_dead_pid(tmp_path, monkeypatch):
    from flattrade_bot import config
    from flattrade_bot.main import TradingEngine

    runtime = tmp_path / "bot.runtime.json"
    runtime.write_text(json.dumps({"pid": 999999, "live_orders": True}), encoding="utf-8")
    monkeypatch.setattr(config.settings, "BOT_RUNTIME_FILE", runtime)
    assert TradingEngine._single_instance_guard() is True


def test_single_instance_guard_allows_missing_file(tmp_path, monkeypatch):
    from flattrade_bot import config
    from flattrade_bot.main import TradingEngine

    runtime = tmp_path / "bot.runtime.json"
    monkeypatch.setattr(config.settings, "BOT_RUNTIME_FILE", runtime)
    assert TradingEngine._single_instance_guard() is True
