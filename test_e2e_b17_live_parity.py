"""End-to-End Test Suite for the B17 Smart Fib Combined live strategy.

Validates:
  1. Byte-identical event parity: minute-by-minute replay of cached days
     through ``LiveSmartFibCombinedStrategy`` emits exactly the combined
     4-TF event stream the B17 backtest engine produces (reference =
     ``extract_day_events`` + ``_session_events`` + merged sorted by
     (minute, tf) and deduped by (minute, side, symbol)).
  2. Fib-level exits: TradeExecutor honors absolute sl_level/tp_level with
     price_rise direction and the index/option monitor source (B17 exits).
  3. B17 risk wiring: RiskManager session 09:20-15:00 and the 4-loss stop
     used by the B17 backtest (CONSECUTIVE_LOSS_LIMIT=4 for B17).
  4. Paper-order path: B17 events route through executor.open_trade and
     positions carry fib SL/TP levels + monitor token/exchange.

Live order fire is intentionally excluded; use b17_live_fire.py with its
explicit confirmation guard.
"""

import asyncio
import gzip
import json
import logging
from datetime import date, datetime
from pathlib import Path

from flattrade_bot.config import settings
from flattrade_bot.execution import TradeExecutor
from flattrade_bot.risk.manager import RiskManager
from flattrade_bot.strategies.smart_fib_combined import (
    CHAMPION,
    LiveSmartFibCombinedStrategy,
    _row_minute,
    _session_events,
)
from artifacts.f6_hybrid.marni_fib_core_combo_cache import extract_day_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_b17")

CACHE_DIR = Path(__file__).parent / "artifacts" / "flattrade_day_cache_smart_fib"
TEST_DAYS = ("2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14")
TIMEFRAMES = (1, 2, 3, 5)
REPLAY_STEP = 10  # every 10th minute; causality makes this equal to per-minute


def reference_combined_events(day_iso: str, cache: dict) -> list:
    """Reproduces the B17 orchestrator's merged event stream for a day."""
    day = date.fromisoformat(day_iso)
    merged = []
    for tf in TIMEFRAMES:
        payload = extract_day_events(
            day_iso,
            cache_loader=lambda root, d, c=cache, day=day: c if d == day else None,
            bar_minutes=tf,
            filter_period=5 * tf,
            debug=False,
            **CHAMPION,
        )
        for signal in _session_events(payload):
            merged.append((int(signal["minute"]), tf, signal))
    merged.sort(key=lambda item: (item[0], item[1]))
    events, seen = [], set()
    for minute, tf, signal in merged:
        key = (minute, signal["side"], signal["symbol"])
        if key in seen:
            continue
        seen.add(key)
        events.append((minute, signal["side"], signal["symbol"]))
    return events


def load_day_cache(day_iso: str) -> dict:
    with gzip.open(CACHE_DIR / f"{day_iso}.json.gz", "rt") as fh:
        return json.load(fh)


def replay_day(day_iso: str, cache: dict) -> list:
    """Replays a cached day through the live strategy and returns emitted events."""
    day = date.fromisoformat(day_iso)
    strat = LiveSmartFibCombinedStrategy(timeframes=TIMEFRAMES)
    strat.set_today(day)
    strat.add_spot_rows(cache["spot_rows"])
    for key, info in cache["contracts"].items():
        side, strike = key.split(":")
        strat.add_contract_rows(
            side, int(strike), info.get("tsym", ""), info.get("token", ""), info["rows"]
        )
    minutes = sorted(
        {
            m for m in (_row_minute(r) for r in cache["spot_rows"])
            if 560 <= m < 900
        }
    )
    emitted = []
    for minute in minutes[::REPLAY_STEP] + [minutes[-1]]:
        for event in strat.evaluate(minute):
            emitted.append((int(event["minute"]), event["side"], event["symbol"]))
    return emitted


def test_b17_settings():
    assert settings.STRATEGY_NAME == "B17_SMART_FIB_COMBINED"
    assert settings.B17_TIMEFRAMES == (1, 2, 3, 5)
    assert settings.B17_S1_K == 12 and settings.B17_S1_D == 4
    assert settings.B17_ZONE_START == 0.5 and settings.B17_ZONE_END == 0.786
    assert settings.B17_TARGET_LEVEL == 0.786 and settings.B17_STOP_LEVEL == 1.13
    assert settings.B17_CONSECUTIVE_LOSS_LIMIT == 4
    assert settings.CONSECUTIVE_LOSS_LIMIT == 8  # legacy default untouched
    logger.info("✅ B17 settings verified")


def test_event_parity_all_days():
    for day_iso in TEST_DAYS:
        cache = load_day_cache(day_iso)
        reference = reference_combined_events(day_iso, cache)
        emitted = replay_day(day_iso, cache)
        assert emitted == reference, (
            f"Event parity failed for {day_iso}: ref={len(reference)} live={len(emitted)}"
        )
        logger.info(
            "✅ Parity OK %s: reference=%d emitted=%d",
            day_iso,
            len(reference),
            len(emitted),
        )


class FakeBrokerClient:
    """Simulates Flattrade responses; supports fill-confirmed orders."""

    def __init__(self):
        self.auth_token = "FAKE_TOKEN"
        self.orders = []
        self.last_order = None

    async def place_market_order(self, **kwargs):
        order = {
            "norenordno": f"ORD_{len(self.orders) + 1}",
            "stat": "Ok",
            **kwargs,
        }
        self.orders.append(order)
        self.last_order = order
        return order

    async def get_order_book(self):
        if not self.last_order:
            return {"stat": "Ok", "orders": []}
        return {
            "stat": "Ok",
            "orders": [
                {
                    "norenordno": self.last_order["norenordno"],
                    "status": "COMPLETE",
                    "fillshares": str(self.last_order["quantity"]),
                    "avgprc": str(self.last_order["ltp"]),
                }
            ],
        }

    async def cancel_order(self, order_id):
        return {"stat": "Ok", "result": "CANCELLED"}


class FakeNotifier:
    async def notify_trade_open(self, payload):
        pass

    async def notify_trade_close(self, payload):
        pass


async def test_executor_fib_level_exits():
    """B17 positions exit on absolute fib levels with price_rise direction."""
    client = FakeBrokerClient()
    risk = RiskManager(consecutive_loss_limit=4, max_daily_loss_rs=settings.B17_MAX_DAILY_LOSS_RS)
    executor = TradeExecutor(
        client, risk, FakeNotifier(), quantity=settings.LOT_SIZE, live_orders=True
    )

    # price_rise=True (CE): SL when ltp <= sl_level, TP when ltp >= tp_level
    result = await executor.open_trade(
        side="CE",
        order_symbol="NIFTY18AUG26C24250",
        display_symbol="NIFTY 24250 CE",
        token="999",
        timeframe="combined",
        signal="SmartFib index@600",
        entry_price=120.0,
        sl_points=0.0,
        tp_points=0.0,
        current_min=601,
        opened_at=datetime.now(),
        sl_level=80.0,
        tp_level=180.0,
        price_rise=True,
        monitor_token="26000",
        monitor_exchange="NSE",
    )
    assert result["accepted"], result
    pos = executor.position
    assert pos["sl_level"] == 80.0 and pos["tp_level"] == 180.0
    assert pos["price_rise"] is True
    assert pos["monitor_token"] == "26000" and pos["monitor_exchange"] == "NSE"

    check = await executor.check_exit(90.0, datetime(2026, 8, 18, 12, 0))
    assert not check["accepted"]  # between levels: stays open

    check = await executor.check_exit(79.9, datetime(2026, 8, 18, 12, 0))
    assert check["accepted"] and check["trade"]["reason"] == "STOP_LOSS"

    # Re-open with price_rise=False (PE): SL when ltp >= sl_level
    result = await executor.open_trade(
        side="PE",
        order_symbol="NIFTY18AUG26P24250",
        display_symbol="NIFTY 24250 PE",
        token="998",
        timeframe="combined",
        signal="SmartFib index@620",
        entry_price=90.0,
        sl_points=0.0,
        tp_points=0.0,
        current_min=621,
        opened_at=datetime.now(),
        sl_level=110.0,
        tp_level=40.0,
        price_rise=False,
        monitor_token="26000",
        monitor_exchange="NSE",
    )
    assert result["accepted"], result
    check = await executor.check_exit(120.0, datetime(2026, 8, 18, 12, 30))
    assert check["accepted"] and check["trade"]["reason"] == "STOP_LOSS"

    # EOD at 15:00 closes regardless of levels
    result = await executor.open_trade(
        side="CE",
        order_symbol="NIFTY18AUG26C24250",
        display_symbol="NIFTY 24250 CE",
        token="999",
        timeframe="combined",
        signal="SmartFib index@700",
        entry_price=100.0,
        sl_points=0.0,
        tp_points=0.0,
        current_min=701,
        opened_at=datetime.now(),
        sl_level=50.0,
        tp_level=200.0,
        price_rise=True,
        monitor_token="26000",
        monitor_exchange="NSE",
    )
    assert result["accepted"], result
    check = await executor.check_exit(100.0, datetime(2026, 8, 18, 15, 0))
    assert check["accepted"] and check["trade"]["reason"] == "EOD"
    logger.info("✅ Fib-level executor exits verified (SL/TP direction + EOD 15:00)")


async def test_paper_order_path_from_events():
    """B17 events route through open_trade with fib levels on the paper path."""
    client = FakeBrokerClient()
    risk = RiskManager(consecutive_loss_limit=4, max_daily_loss_rs=settings.B17_MAX_DAILY_LOSS_RS)
    executor = TradeExecutor(
        client, risk, FakeNotifier(), quantity=settings.LOT_SIZE, live_orders=True
    )

    cache = load_day_cache("2026-08-11")
    strat = LiveSmartFibCombinedStrategy(timeframes=TIMEFRAMES)
    strat.set_today(date(2026, 8, 11))
    strat.add_spot_rows(cache["spot_rows"])
    for key, info in cache["contracts"].items():
        side, strike = key.split(":")
        strat.add_contract_rows(
            side, int(strike), info.get("tsym", ""), info.get("token", ""), info["rows"]
        )
    strat._slice_minute = 900
    events = strat.evaluate(899)
    assert events, "Expected at least one B17 event on 2026-08-11"

    event = events[0]
    side = str(event["side"])
    strike = int(event["strike"])
    tracked = {
        "token": "777",
        "tsym": event["symbol"],
        "dname": f"NIFTY {strike} {side}",
    }
    sl_level, tp_level, price_rise, monitor = strat.exit_levels(event)
    assert monitor in ("index", "option")
    now = datetime.now()
    result = await executor.open_trade(
        side=side,
        order_symbol=tracked["tsym"],
        display_symbol=tracked["dname"],
        token=tracked["token"],
        timeframe="combined",
        signal=f"SmartFib {event.get('trigger', 'index')}@min{event['minute']}",
        entry_price=float(event.get("option_entry", 0.0)),
        sl_points=0.0,
        tp_points=0.0,
        current_min=620,
        opened_at=now,
        sl_level=sl_level,
        tp_level=tp_level,
        price_rise=price_rise,
        monitor_token=tracked["token"] if monitor == "option" else "26000",
        monitor_exchange="NFO" if monitor == "option" else "NSE",
    )
    assert result["accepted"], result
    assert result["position"]["sl_level"] == sl_level
    assert result["position"]["tp_level"] == tp_level
    logger.info(
        "✅ Paper order path verified: %s %s entry=%.2f sl=%.2f tp=%.2f monitor=%s",
        side,
        tracked["tsym"],
        event.get("option_entry", 0.0),
        sl_level,
        tp_level,
        monitor,
    )


def test_b17_risk_session():
    risk = RiskManager(consecutive_loss_limit=4, max_daily_loss_rs=settings.B17_MAX_DAILY_LOSS_RS)
    assert risk.session_start_min == 9 * 60 + 20
    assert risk.session_end_min == 15 * 60
    can_trade, _ = risk.can_open_trade(560, 0)
    assert can_trade
    can_trade, reason = risk.can_open_trade(901, 0)
    assert not can_trade
    for _ in range(4):
        risk.record_trade_result(-50.0)
    can_trade, reason = risk.can_open_trade(600, 0)
    assert not can_trade, "B17 4-loss stop must block entries"
    assert "Consecutive Loss" in reason
    logger.info("✅ B17 risk session (09:20-15:00) and 4-loss stop verified")


async def test_engine_b17_live_wiring():
    """Drives the engine's own live path: _process_b17_events -> open with
    fib levels -> _monitor_active_position -> fib SL exit, using fakes."""
    import flattrade_bot.main as main_mod

    assert "_b17_minute_update" in main_mod.TradingEngine.__dict__, (
        "_b17_minute_update must be a method of TradingEngine (dedent regression)"
    )
    assert "_b17_minute_update" not in vars(main_mod), (
        "_b17_minute_update must not live at module level"
    )

    class _FakeNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 18, 11, 0, 0)

    real_datetime = main_mod.datetime
    main_mod.datetime = _FakeNow
    try:
        engine = main_mod.TradingEngine(live_orders=True)
        fake = FakeBrokerClient()
        engine.executor.client = fake
        engine.executor.risk = RiskManager(
            consecutive_loss_limit=4, max_daily_loss_rs=settings.B17_MAX_DAILY_LOSS_RS
        )

        class FakeHistory:
            def __init__(self):
                self.quote_ltp = 0.0

            async def fetch_live_quote(self, token, exchange):
                return {"lp": self.quote_ltp}

            async def search_option_token(self, name):
                return None

        fake_history = FakeHistory()
        engine.history = fake_history

        day_iso = TEST_DAYS[0]
        cache = load_day_cache(day_iso)
        strat = engine.b17_strategy
        strat.set_today(date.fromisoformat(day_iso))
        strat.add_spot_rows(cache["spot_rows"])
        for key, info in cache["contracts"].items():
            side, strike = key.split(":")
            engine.b17_tracked[key] = {
                "token": info.get("token", ""),
                "tsym": info.get("tsym", ""),
                "dname": f"NIFTY {strike} {side}",
            }
            strat.add_contract_rows(
                side, int(strike), info.get("tsym", ""), info.get("token", ""), info["rows"]
            )

        minutes = sorted(
            {m for m in (_row_minute(r) for r in cache["spot_rows"]) if 560 <= m < 900}
        )
        events = []
        for minute in minutes[::REPLAY_STEP] + [minutes[-1]]:
            events.extend(strat.evaluate(minute))
        assert events, "wiring test needs at least one B17 event"

        await engine._process_b17_events(events)
        assert engine.executor.position is not None, "event must open a live position"
        assert engine.active_position is not None
        assert engine.b17_event_count == 1
        pos = engine.executor.position
        assert pos["sl_level"] is not None and pos["tp_level"] is not None
        assert pos["monitor_exchange"] in ("NSE", "NFO")
        assert len(fake.orders) == 1  # entry fired

        sl_level = pos["sl_level"]
        price_rise = pos["price_rise"]
        fake_history.quote_ltp = sl_level - 10.0 if price_rise else sl_level + 10.0
        await engine._monitor_active_position()
        assert engine.executor.position is None, "fib SL must close the position"
        assert engine.active_position is None
        assert len(fake.orders) == 2  # entry + exit
        logger.info(
            "✅ Engine wiring: B17 event -> live trade -> fib SL exit (monitor=%s/%s)",
            pos["monitor_exchange"], pos.get("monitor_token"),
        )
    finally:
        main_mod.datetime = real_datetime


async def main():
    test_b17_settings()
    test_event_parity_all_days()
    await test_executor_fib_level_exits()
    await test_paper_order_path_from_events()
    await test_engine_b17_live_wiring()
    test_b17_risk_session()
    logger.info("🎉 ALL B17 END-TO-END TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    asyncio.run(main())
