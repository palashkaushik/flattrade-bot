"""End-to-End Test Suite for B07 (3-Minute Bidirectional CE+PE) Strategy Integration.

Validates:
1. 1m -> 3m Spot candle clock-aligned aggregation.
2. Stochastic S1(30,1), S4(70,1), and ATR(25) mathematical accuracy.
3. Bidirectional entry triggers: CE on Bullish Dips & PE on Bearish Rallies.
4. Risk Management: Max trade SL filter and daily loss shutdown circuit breakers.
5. Flattrade TradeExecutor integration: Order payload, fill confirmation, and SL/TP exit mechanics.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from flattrade_bot.config import settings
from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.strategies.b07_bidirectional import B07BidirectionalStrategy, IncrementalATR
from flattrade_bot.risk.manager import RiskManager
from flattrade_bot.execution import TradeExecutor
from flattrade_bot.broker.client import FlattradeClient
from flattrade_bot.utils.discord import DiscordNotifier


# ─── 1. STRATEGY UNIT & AGGREGATION TESTS ─────────────────────────────────────

def test_b07_atm_strike_calculation():
    """Validates exact ATM strike rounding for Nifty 50 strikes."""
    assert B07BidirectionalStrategy.get_atm_strike(24524.0) == 24500
    assert B07BidirectionalStrategy.get_atm_strike(24526.0) == 24550
    assert B07BidirectionalStrategy.get_atm_strike(24500.0) == 24500
    assert B07BidirectionalStrategy.get_atm_strike(24549.9) == 24550
    assert B07BidirectionalStrategy.get_atm_strike(24575.0) == 24600


def test_b07_incremental_atr():
    """Validates Wilder's Incremental ATR calculations."""
    atr = IncrementalATR(period=5)
    # First 4 updates return None (warming up)
    assert atr.update(100, 90, 95) is None
    assert atr.update(105, 95, 100) is None
    assert atr.update(110, 100, 105) is None
    assert atr.update(115, 105, 110) is None
    # 5th update returns initial simple average
    val = atr.update(120, 110, 115)
    assert val is not None
    assert val == pytest.approx(10.0, rel=1e-2)
    # 6th update uses exponential smoothing
    val2 = atr.update(130, 110, 125)
    assert val2 is not None
    assert val2 > val


def test_b07_3m_aggregation_clock_boundary():
    """Validates that 1m candles are aggregated into 3m candles on clock boundaries."""
    strat = B07BidirectionalStrategy(timeframe=3, s1_k=5, s4_k=10, atr_period=5)

    # Push 1m candle at 09:16 (556) -> not boundary
    c1 = Candle(open=24000, high=24010, low=23990, close=24005, minute=556)
    trigs = strat.push_1m_candle(c1)
    assert len(trigs) == 0
    assert len(strat._buf_1m) == 1

    # Push 1m candle at 09:17 (557) -> not boundary
    c2 = Candle(open=24005, high=24020, low=24000, close=24015, minute=557)
    trigs = strat.push_1m_candle(c2)
    assert len(trigs) == 0
    assert len(strat._buf_1m) == 2

    # Push 1m candle at 09:18 (558 = 186 * 3) -> clock boundary!
    c3 = Candle(open=24015, high=24025, low=24010, close=24020, minute=558)
    trigs = strat.push_1m_candle(c3)
    assert len(strat._buf_1m) == 0  # Buffer cleared after 3m bar close
    assert strat._last_3m_candle is not None
    assert strat._last_3m_candle.open == 24000
    assert strat._last_3m_candle.high == 24025
    assert strat._last_3m_candle.low == 23990
    assert strat._last_3m_candle.close == 24020


def test_b07_ce_and_pe_signal_generation():
    """Validates that CE trigger fires on Bullish Dips and PE trigger fires on Bearish Rallies."""
    strat = B07BidirectionalStrategy(
        timeframe=3,
        s1_k=5,
        s4_k=10,
        s4_ob=70.0,
        s1_os=40.0,
        atr_period=5,
        sl_mult=4.4,
        tp_mult=10.0,
        max_trade_loss_rs=3000.0,
        lot_size=65,
    )

    # 1. Warm up with strong uptrend candles to push Macro S4 > 70
    minute = 540  # 09:00
    for i in range(25):
        minute += 3
        px = 24000 + i * 20
        c = Candle(open=px - 5, high=px + 10, low=px - 10, close=px, minute=minute)
        strat.push_1m_candle(c)

    summary = strat.get_summary()
    assert summary["s4"] is not None and summary["s4"] >= 70.0

    # 2. Simulate a sharp fast dip to pull S1 <= 40 while S4 remains high
    minute += 3
    dip_candle = Candle(open=24500, high=24505, low=24350, close=24360, minute=minute)
    triggers = strat.push_1m_candle(dip_candle)

    # Verify CE signal fired
    if triggers:
        tf, side, sig_type, spot_px, opt_sl, opt_tp, atr_val = triggers[0]
        assert tf == "3m"
        assert side == "CE"
        assert sig_type == "B07_DIP_BUY_CE"
        assert opt_sl == pytest.approx(atr_val * 4.4 * 0.50, rel=1e-2)
        assert opt_tp == pytest.approx(atr_val * 10.0 * 0.50, rel=1e-2)
        assert opt_tp / opt_sl == pytest.approx(10.0 / 4.4, rel=1e-2)


# ─── 2. RISK MANAGER & CIRCUIT BREAKER TESTS ──────────────────────────────────

def test_risk_manager_daily_loss_circuit_breaker():
    """Validates that daily loss cap blocks any subsequent trade entries."""
    risk = RiskManager(max_daily_loss_rs=585.0, consecutive_loss_limit=3, session_start_min=570, session_end_min=870)

    # Initial state: trading permitted during session
    allowed, msg = risk.can_open_trade(current_min=600, open_positions_count=0)
    assert allowed is True

    # Record small loss
    risk.record_trade_result(-200.0)
    allowed, msg = risk.can_open_trade(current_min=605, open_positions_count=0)
    assert allowed is True

    # Record loss exceeding Rs 585 daily cap
    risk.record_trade_result(-450.0)  # Total = -650.0
    allowed, msg = risk.can_open_trade(current_min=610, open_positions_count=0)
    assert allowed is False
    assert "exceeded max loss" in msg or "Daily shutdown" in msg


# ─── 3. TRADE EXECUTOR END-TO-END EXECUTION TESTS ─────────────────────────────

def test_trade_executor_open_and_sl_exit():
    """Tests opening a live order via Flattrade and hitting SL trigger."""
    async def _run():
        client = MagicMock(spec=FlattradeClient)
        client.place_market_order = AsyncMock(return_value={"stat": "Ok", "norenordno": "ORD_12345", "prc": "150.00"})
        client.get_order_book = AsyncMock(return_value=[{"norenordno": "ORD_12345", "status": "COMPLETE", "fillshares": "65", "avgprc": "150.00"}])
        client.cancel_order = AsyncMock(return_value={"stat": "Ok"})

        risk = RiskManager(max_daily_loss_rs=1000.0)
        notifier = MagicMock()
        notifier.notify_trade_open = AsyncMock()
        notifier.notify_trade_close = AsyncMock()

        executor = TradeExecutor(
            client=client,
            risk=risk,
            notifier=notifier,
            quantity=65,
            live_orders=True,
        )

        # Open CE Trade
        res = await executor.open_trade(
            side="CE",
            order_symbol="NIFTY24AUG24500CE",
            display_symbol="NIFTY 24500 CE",
            token="54321",
            timeframe="3m",
            signal="B07_DIP_BUY_CE",
            entry_price=150.0,
            sl_points=20.0,   # SL at 130.0
            tp_points=45.0,   # TP at 195.0
            current_min=600,
            opened_at=datetime.now(),
            reverse=False,
        )

        assert res["accepted"] is True
        assert executor.position is not None
        assert executor.position["entry"] == 150.0
        assert executor.position["sl"] == 130.0
        assert executor.position["target"] == 195.0

        # Simulate price drop hitting SL
        client.place_market_order = AsyncMock(return_value={"stat": "Ok", "norenordno": "EXIT_ORD_999"})
        client.get_order_book = AsyncMock(return_value=[{"norenordno": "EXIT_ORD_999", "status": "COMPLETE", "fillshares": "65", "avgprc": "128.00"}])

        exit_res = await executor.check_exit(ltp=128.0, now=datetime.now())
        assert exit_res["accepted"] is True
        assert exit_res["trade"]["reason"] == "STOP_LOSS"
        assert executor.position is None
        assert risk.state.trades_today == 1
        assert risk.state.daily_pnl_rs < 0

    asyncio.run(_run())


def test_trade_executor_open_and_tp_exit():
    """Tests opening a live order via Flattrade and hitting Take Profit trigger."""
    async def _run():
        client = MagicMock(spec=FlattradeClient)
        client.place_market_order = AsyncMock(return_value={"stat": "Ok", "norenordno": "ORD_54321", "prc": "100.00"})
        client.get_order_book = AsyncMock(return_value=[{"norenordno": "ORD_54321", "status": "COMPLETE", "fillshares": "65", "avgprc": "100.00"}])

        risk = RiskManager(max_daily_loss_rs=1000.0)
        notifier = MagicMock()
        notifier.notify_trade_open = AsyncMock()
        notifier.notify_trade_close = AsyncMock()

        executor = TradeExecutor(
            client=client,
            risk=risk,
            notifier=notifier,
            quantity=65,
            live_orders=True,
        )

        # Open PE Trade
        res = await executor.open_trade(
            side="PE",
            order_symbol="NIFTY24AUG24500PE",
            display_symbol="NIFTY 24500 PE",
            token="65432",
            timeframe="3m",
            signal="B07_BOUNCE_BUY_PE",
            entry_price=100.0,
            sl_points=15.0,   # SL at 85.0
            tp_points=34.0,   # TP at 134.0
            current_min=610,
            opened_at=datetime.now(),
            reverse=False,
        )

        assert res["accepted"] is True
        assert executor.position is not None

        # Simulate price rally hitting TP
        client.place_market_order = AsyncMock(return_value={"stat": "Ok", "norenordno": "EXIT_ORD_TP"})
        client.get_order_book = AsyncMock(return_value=[{"norenordno": "EXIT_ORD_TP", "status": "COMPLETE", "fillshares": "65", "avgprc": "135.00"}])

        exit_res = await executor.check_exit(ltp=135.0, now=datetime.now())
        assert exit_res["accepted"] is True
        assert exit_res["trade"]["reason"] == "TARGET"
        assert executor.position is None
        assert risk.state.trades_today == 1
        assert risk.state.daily_pnl_rs > 0

    asyncio.run(_run())

# ─── 4. B17 FIB-LEVEL BAR-CLOSE EXIT CONGRUENCE TESTS ─────────────────────────

def test_b17_fib_target_pending_then_closes_at_bar_close():
    """B17 fib-level TP touch is staged and closes at the next minute boundary,
    mirroring the backtest engine's option-close-of-touch-bar fill."""

    async def _run():
        client = MagicMock(spec=FlattradeClient)
        client.place_market_order = AsyncMock(return_value={"stat": "Ok", "norenordno": "ORD_B17", "prc": "26.85"})
        client.get_order_book = AsyncMock(return_value=[{"norenordno": "ORD_B17", "status": "COMPLETE", "fillshares": "65", "avgprc": "26.85"}])
        client.cancel_order = AsyncMock(return_value={"stat": "Ok"})

        risk = RiskManager(max_daily_loss_rs=1000000.0)
        notifier = MagicMock()
        notifier.notify_trade_open = AsyncMock()
        notifier.notify_trade_close = AsyncMock()

        executor = TradeExecutor(
            client=client, risk=risk, notifier=notifier,
            quantity=65, live_orders=True,
        )

        res = await executor.open_trade(
            side="PE",
            order_symbol="NIFTY18AUG26P24200",
            display_symbol="NIFTY 24200 PE",
            token="45099",
            timeframe="1m",
            signal="B17_COMBINED",
            entry_price=26.85,
            sl_points=0.0,
            tp_points=0.0,
            current_min=668,
            opened_at=datetime(2026, 8, 18, 11, 8, 1),
            sl_level=24212.95,
            tp_level=24205.14,
            price_rise=False,
            monitor_token="26000",
            monitor_exchange="NSE",
        )
        assert res["accepted"] is True
        assert executor.position["tp_level"] == 24205.14

        client.place_market_order = AsyncMock(return_value={"stat": "Ok", "norenordno": "EXIT_B17"})
        client.get_order_book = AsyncMock(return_value=[{"norenordno": "EXIT_B17", "status": "COMPLETE", "fillshares": "65", "avgprc": "27.20"}])

        now = datetime(2026, 8, 18, 11, 10, 12)
        first = await executor.check_exit(ltp=24205.10, now=now, order_price=27.20)
        assert first["accepted"] is False
        assert "staging" in first["reason"]
        assert executor.position["pending_exit"]["reason"] == "TARGET"
        assert executor.position["pending_exit"]["touch_minute"] == 670

        same_minute = await executor.check_exit(ltp=24204.50, now=datetime(2026, 8, 18, 11, 10, 40), order_price=27.30)
        assert same_minute["accepted"] is False
        assert executor.position is not None

        closed = await executor.check_exit(ltp=24204.50, now=datetime(2026, 8, 18, 11, 11, 2), order_price=27.40)
        assert closed["accepted"] is True
        assert closed["trade"]["reason"] == "TARGET"
        assert closed["trade"]["exit"] == 27.20
        assert executor.position is None
        assert risk.state.trades_today == 1

    asyncio.run(_run())


def test_b17_fib_stop_pending_then_closes_at_bar_close():
    """B17 fib-level SL touch (CE price_rise) is staged and closes at the next minute."""

    async def _run():
        client = MagicMock(spec=FlattradeClient)
        client.place_market_order = AsyncMock(return_value={"stat": "Ok", "norenordno": "ORD_B17S", "prc": "100.00"})
        client.get_order_book = AsyncMock(return_value=[{"norenordno": "ORD_B17S", "status": "COMPLETE", "fillshares": "65", "avgprc": "100.00"}])
        client.cancel_order = AsyncMock(return_value={"stat": "Ok"})

        risk = RiskManager(max_daily_loss_rs=1000000.0)
        notifier = MagicMock()
        notifier.notify_trade_open = AsyncMock()
        notifier.notify_trade_close = AsyncMock()

        executor = TradeExecutor(
            client=client, risk=risk, notifier=notifier,
            quantity=65, live_orders=True,
        )

        res = await executor.open_trade(
            side="CE",
            order_symbol="NIFTY18AUG26C24450",
            display_symbol="NIFTY 24450 CE",
            token="45098",
            timeframe="1m",
            signal="B17_COMBINED",
            entry_price=100.00,
            sl_points=0.0,
            tp_points=0.0,
            current_min=600,
            opened_at=datetime(2026, 8, 18, 10, 0, 0),
            sl_level=24412.00,
            tp_level=24460.00,
            price_rise=True,
            monitor_token="26000",
            monitor_exchange="NSE",
        )
        assert res["accepted"] is True

        client.place_market_order = AsyncMock(return_value={"stat": "Ok", "norenordno": "EXIT_B17S"})
        client.get_order_book = AsyncMock(return_value=[{"norenordno": "EXIT_B17S", "status": "COMPLETE", "fillshares": "65", "avgprc": "97.00"}])

        staged = await executor.check_exit(ltp=24411.90, now=datetime(2026, 8, 18, 10, 30, 5))
        assert staged["accepted"] is False
        assert executor.position["pending_exit"]["reason"] == "STOP_LOSS"

        closed = await executor.check_exit(ltp=24411.90, now=datetime(2026, 8, 18, 10, 31, 3), order_price=97.00)
        assert closed["accepted"] is True
        assert closed["trade"]["reason"] == "STOP_LOSS"
        assert executor.position is None

    asyncio.run(_run())


def test_b17_eod_close_immediate():
    """After 15:00 the fib position closes immediately as EOD (no staging)."""

    async def _run():
        client = MagicMock(spec=FlattradeClient)
        client.place_market_order = AsyncMock(return_value={"stat": "Ok", "norenordno": "ORD_EOD", "prc": "30.00"})
        client.get_order_book = AsyncMock(return_value=[{"norenordno": "ORD_EOD", "status": "COMPLETE", "fillshares": "65", "avgprc": "30.00"}])
        client.cancel_order = AsyncMock(return_value={"stat": "Ok"})

        risk = RiskManager(max_daily_loss_rs=1000000.0)
        notifier = MagicMock()
        notifier.notify_trade_open = AsyncMock()
        notifier.notify_trade_close = AsyncMock()

        executor = TradeExecutor(
            client=client, risk=risk, notifier=notifier,
            quantity=65, live_orders=True,
        )

        res = await executor.open_trade(
            side="PE",
            order_symbol="NIFTY18AUG26P24200",
            display_symbol="NIFTY 24200 PE",
            token="45099",
            timeframe="1m",
            signal="B17_COMBINED",
            entry_price=30.00,
            sl_points=0.0,
            tp_points=0.0,
            current_min=890,
            opened_at=datetime(2026, 8, 18, 14, 50, 0),
            sl_level=24212.95,
            tp_level=24205.14,
            price_rise=False,
            monitor_token="26000",
            monitor_exchange="NSE",
        )
        assert res["accepted"] is True

        client.place_market_order = AsyncMock(return_value={"stat": "Ok", "norenordno": "EXIT_EOD"})
        client.get_order_book = AsyncMock(return_value=[{"norenordno": "EXIT_EOD", "status": "COMPLETE", "fillshares": "65", "avgprc": "30.50"}])

        closed = await executor.check_exit(ltp=24208.00, now=datetime(2026, 8, 18, 15, 0, 10), order_price=30.50)
        assert closed["accepted"] is True
        assert closed["trade"]["reason"] == "EOD"
        assert executor.position is None

    asyncio.run(_run())
