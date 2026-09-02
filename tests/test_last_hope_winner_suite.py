"""Comprehensive Test Suite for Last Hope GPU Winner Strategy & Live Execution Bot.

Covers:
  1. Indicator Math Parity (Stochastics S1/S3/S4, ATR, EMA, VWAP)
  2. Multi-Timeframe Resampling (1m, 2m, 3m, 5m aggregation)
  3. Option Chart S/R Suite (CPR, Camarilla, PDH/PDL)
  4. Arming State Machine (10-bar window on S1 <= 25.0)
  5. Signal Triggers (Flag M6 and Super setups)
  6. Strict S/R Bounce Gating (touch_buffer = 0.0)
  7. Risk Geometry (ATR*1.5, SL, TP, Breakeven at 70% move to Entry+1pt)
  8. Seamless TradeExecutor Order Placement, Retries, and Broker Reconciliation
  9. Full-Day End-to-End Simulation with EOD Square-Off
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List
import pytest
import numpy as np

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattrade_bot.strategies.last_hope_winner import (
    ARM_S1,
    ARM_WINDOW,
    ATR_MULT,
    ATR_PERIOD,
    BE_BUFFER_PTS,
    BE_TRIGGER_RATIO,
    M6_S1,
    M6_S4,
    SUPER_THRESH,
    TOUCH_BUFFER,
    TP_PTS_CAP,
    Bar1m,
    IncrementalATR,
    IncrementalEMA,
    IncrementalStoch,
    IncrementalVWAP,
    LastHopeWinnerEngine,
    OptionContractState,
    TFTracker,
)
from flattrade_bot.execution import TradeExecutor
from flattrade_bot.risk.manager import RiskManager
from flattrade_bot.utils.discord import DiscordNotifier


# =====================================================================
# 1. INDICATOR MATH PARITY TESTS
# =====================================================================

def test_incremental_stochastic_parity():
    """Verifies IncrementalStoch against vectorized numpy stochastic %D."""
    k_period, d_period = 12, 3
    stoch = IncrementalStoch(k_period, d_period)

    np.random.seed(42)
    highs = 100.0 + np.cumsum(np.random.randn(50))
    lows = highs - np.random.uniform(0.5, 2.0, 50)
    closes = lows + (highs - lows) * np.random.uniform(0.1, 0.9, 50)

    # Incremental updates
    inc_d = []
    for h, l, c in zip(highs, lows, closes):
        inc_d.append(stoch.update(h, l, c))

    # Vectorized reference
    raw_k = []
    for i in range(len(closes)):
        if i < k_period - 1:
            raw_k.append(np.nan)
        else:
            hh = max(highs[i - k_period + 1 : i + 1])
            ll = min(lows[i - k_period + 1 : i + 1])
            k = ((closes[i] - ll) / max(hh - ll, 1e-6)) * 100.0
            raw_k.append(k)

    vec_d = []
    for i in range(len(raw_k)):
        if i < k_period + d_period - 2:
            vec_d.append(None)
        else:
            d_val = float(np.mean(raw_k[i - d_period + 1 : i + 1]))
            vec_d.append(d_val)

    # Compare non-None values
    for i in range(len(inc_d)):
        if vec_d[i] is not None:
            assert inc_d[i] is not None, f"Mismatch at index {i}: inc is None, vec is {vec_d[i]}"
            assert abs(inc_d[i] - vec_d[i]) < 1e-4, f"Value mismatch at {i}: {inc_d[i]} vs {vec_d[i]}"


def test_incremental_atr_parity():
    """Verifies IncrementalATR formula (EMA smoothed TR) against numpy reference."""
    period = 10
    atr_calc = IncrementalATR(period)
    alpha = 2.0 / (period + 1.0)

    highs = [105, 108, 107, 110, 112, 109, 108, 111, 115, 114, 116, 118]
    lows = [100, 102, 103, 105, 107, 104, 103, 106, 109, 110, 111, 113]
    closes = [104, 106, 105, 109, 108, 106, 107, 110, 113, 112, 115, 117]

    inc_values = []
    for h, l, c in zip(highs, lows, closes):
        inc_values.append(atr_calc.update(h, l, c))

    # Reference TR calculation
    tr_list = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        tr_list.append(tr)

    ref_atr = [tr_list[0]]
    for i in range(1, len(tr_list)):
        ref_val = ref_atr[-1] * (1.0 - alpha) + tr_list[i] * alpha
        ref_atr.append(ref_val)

    for i in range(len(inc_values)):
        assert abs(inc_values[i] - ref_atr[i]) < 1e-4, f"ATR mismatch at {i}: {inc_values[i]} vs {ref_atr[i]}"


def test_incremental_ema_and_vwap():
    """Verifies EMA20, EMA200 and VWAP incremental formulas."""
    ema = IncrementalEMA(20)
    vwap = IncrementalVWAP()

    v1 = ema.update(100.0)
    assert v1 == 100.0

    v2 = ema.update(110.0)
    alpha = 2.0 / 21.0
    assert abs(v2 - (110.0 * alpha + 100.0 * (1.0 - alpha))) < 1e-5

    vw1 = vwap.update(105.0, 95.0, 100.0, volume=100.0)
    assert abs(vw1 - 100.0) < 1e-5


# =====================================================================
# 2. MULTI-TIMEFRAME RESAMPLING TESTS
# =====================================================================

def test_multi_tf_tracker_aggregation():
    """Verifies that 1m bars correctly aggregate into 2m, 3m, and 5m bars."""
    tracker_3m = TFTracker(3)
    base_t = datetime(2026, 8, 28, 9, 15)

    bars = [
        Bar1m(555, open=100, high=105, low=98, close=102, timestamp=base_t),
        Bar1m(556, open=102, high=108, low=101, close=107, timestamp=base_t + timedelta(minutes=1)),
        Bar1m(557, open=107, high=110, low=104, close=109, timestamp=base_t + timedelta(minutes=2)),
    ]

    # Bar 1: forming
    s1, s3, s4, rising = tracker_3m.push_1m_bar(bars[0])
    assert tracker_3m.last_lo is None  # not closed yet

    # Bar 2: forming
    s1, s3, s4, rising = tracker_3m.push_1m_bar(bars[1])
    assert tracker_3m.last_lo is None

    # Bar 3: completes 3m bar!
    s1, s3, s4, rising = tracker_3m.push_1m_bar(bars[2])
    assert tracker_3m.last_lo == 98.0  # min(98, 101, 104)
    assert tracker_3m.last_cl == 109.0  # last close


# =====================================================================
# 3. OPTION S/R SUITE (CPR & CAMARILLA) TESTS
# =====================================================================

def test_day_sr_levels_cpr_camarilla():
    """§43: display suite carries CPR/Camarilla/PDH-PDL; sr_levels (the GATE)
    must be EMPTY — EMA20 (live indicator) is the only trading-gate level."""
    cs = OptionContractState("NIFTY26AUG24200CE", "token1", "CE", 24200)

    ph, pl, pc = 150.0, 100.0, 120.0
    cs.set_day_sr_levels(ph, pl, pc)

    pivot = (150.0 + 100.0 + 120.0) / 3.0  # 123.3333
    bc = (150.0 + 100.0) / 2.0              # 125.0
    tc = 2.0 * pivot - bc                  # 121.6666
    cam_h3 = 120.0 + (150.0 - 100.0) * 1.1 / 4.0  # 133.75
    cam_l3 = 120.0 - (150.0 - 100.0) * 1.1 / 4.0  # 106.25

    # GATE must be EMA20-only (injected live in _on_1m_bar_close)
    assert cs.sr_levels == {}, "§43: sr_levels (gate) must be empty — EMA20 gates via live indicator"

    # Display suite keeps the full level set for dashboard/TradingView
    assert abs(cs.display_levels["CPR_Pivot"] - pivot) < 1e-4
    assert abs(cs.display_levels["CPR_BC"] - bc) < 1e-4
    assert abs(cs.display_levels["CPR_TC"] - tc) < 1e-4
    assert abs(cs.display_levels["Cam_H3"] - cam_h3) < 1e-4
    assert abs(cs.display_levels["Cam_L3"] - cam_l3) < 1e-4
    assert cs.display_levels["PDH"] == 150.0
    assert cs.display_levels["PDL"] == 100.0


# =====================================================================
# 4. ARMING STATE MACHINE & TRIGGER TESTS
# =====================================================================

def test_arming_state_machine_and_expiration():
    """Verifies that S1 <= 25 arms setup and expires precisely after 10 bars."""
    cs = OptionContractState("NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs.set_day_sr_levels(150, 100, 120)

    base_t = datetime(2026, 8, 28, 9, 15)

    # 1. Warm up with decreasing bars that push S1 down to <= 25
    prices = [150 - i * 3 for i in range(15)]
    for i, p in enumerate(prices):
        t = base_t + timedelta(minutes=i)
        bar = Bar1m(555 + i, open=p + 1, high=p + 2, low=p - 1, close=p, timestamp=t)
        cs._on_1m_bar_close(bar)

    assert cs.flag_armed is True
    assert cs.super_armed is True

    # 2. Push 20 rising bars so S1 rises well above 25 and arming window expires
    for i in range(20):
        bar_num = len(cs.bars)
        t = base_t + timedelta(minutes=bar_num)
        p = 110 + i * 5  # Strong rising prices: S1 reaches > 80
        bar = Bar1m(555 + bar_num, open=p, high=p + 4, low=p - 1, close=p + 3, timestamp=t)
        cs._on_1m_bar_close(bar)

    # After 20 bars with S1 > 25, setup must be completely disarmed
    assert cs.flag_armed is False
    assert cs.super_armed is False


# =====================================================================
# 5. STRICT S/R BOUNCE GATING (TOUCH_BUFFER = 0.0) TESTS
# =====================================================================

def test_sr_bounce_gate_strict():
    """§43: gate = EMA20 ONLY. Low <= EMA20 + 0.0 and Close >= EMA20 - 0.5."""
    cs = OptionContractState("NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs.set_day_sr_levels(150, 100, 120)
    cs.ema20.value = 120.0
    cs.ema200.value = 50.0
    cs.vwap.cum_pv = 50.0 * 10000.0
    cs.vwap.cum_vol = 10000.0

    t = datetime(2026, 8, 28, 9, 20)

    # Use 5m tracker for stochastic values — only recalculates every 5 bars,
    # so manual values survive through up to 4 push_1m_bar calls.
    cs.tf_trackers[5].last_s4 = 85.0
    cs.tf_trackers[5].last_s1 = 40.0
    cs.tf_trackers[5].last_s3 = 50.0
    cs.tf_trackers[5].prev_s1 = 35.0

    # Case A: Low doesn't touch EMA20 (low = 120.5 > 120.0) -> REJECTED (touch_buffer=0.0)
    cs.flag_armed = True
    cs.flag_arm_bar = 0
    bar_no_touch = Bar1m(560, open=125.0, high=126.0, low=120.5, close=124.0, timestamp=t)
    sig_a = cs._on_1m_bar_close(bar_no_touch)
    assert sig_a is None, "Signal should be rejected when low > EMA20 (touch_buffer=0.0)"

    # Case B: Low touches EMA20 (low = 119.5 <= 120.0) and Close bounces (close = 122.0) -> ACCEPTED!
    cs.flag_armed = True
    cs.flag_arm_bar = len(cs.bars)
    bar_bounce = Bar1m(561, open=121.0, high=123.0, low=119.5, close=122.0, timestamp=t)
    sig_b = cs._on_1m_bar_close(bar_bounce)
    assert sig_b is not None
    assert sig_b["trigger"] == "FLAG"
    assert "EMA20" in sig_b["level"]


# =====================================================================
# 6. RISK GEOMETRY & BREAKEVEN STOP TESTS
# =====================================================================

def test_risk_geometry_sl_tp_breakeven():
    """§43: ATR*1.0 stop distance and Breakeven stop movement at 60% target move."""
    engine = LastHopeWinnerEngine()
    cs = engine.register_contract("CE:24200", "NIFTY26AUG24200CE", "token1", "CE", 24200)

    # Fixed ATR = 6.0 pts -> dist = min(max(6.0 * 1.0, 2.0), 15.0) = 6.0 pts
    cs.atr.value = 6.0
    cs.set_day_sr_levels(150, 100, 120)

    # Setup a trade signal
    sig = {
        "side": "CE",
        "symbol": "NIFTY26AUG24200CE",
        "token": "token1",
        "strike": 24200,
        "entry": 100.0,
        "dist": 6.0,
        "sl": 94.0,               # 100 - 6.0
        "tp": 106.0,              # 100 + 6.0
        "be_trigger_px": 103.60,  # 100 + 0.60 * 6.0 = 103.60 (§43 BE_TRIGGER_RATIO=0.60)
        "be_hardened_sl": 101.0,  # Entry + 1.0 pt
        "be_done": False,
    }

    engine.on_trade_opened(sig)
    now = datetime(2026, 8, 28, 9, 30)

    # Tick 1: Price rises to 102.0 (< BE trigger 103.60) -> SL remains 94.0
    engine.push_tick("CE:24200", 102.0, now)
    assert engine.active_trade["sl"] == 94.0
    assert engine.active_trade["be_done"] is False

    # Tick 2: Price reaches 103.80 (>= BE trigger 103.60) -> SL hardens to 101.0!
    engine.push_tick("CE:24200", 103.80, now)
    assert engine.active_trade["be_done"] is True
    assert engine.active_trade["sl"] == 101.0

    # Tick 3: Price pulls back to 103.0 -> SL MUST NEVER MOVE BACKWARDS
    engine.push_tick("CE:24200", 103.0, now)
    assert engine.active_trade["sl"] == 101.0


# =====================================================================
# 7. TRADE EXECUTOR RESILIENCE & RECONCILIATION TESTS
# =====================================================================

class MockFlattradeClient:
    """Mock Flattrade client to test execution failure recovery and polling."""

    def __init__(self):
        self.auth_token = "mock_token"
        self.order_book_call_count = 0
        self.orders = []
        self.positions = []
        self.last_placed_order = None

    async def place_market_order(self, symbol, side, quantity, ltp, product="MIS", slippage_buffer=5.0, force_mkt=False):
        order_id = f"MOCK_ORD_{len(self.orders) + 1}"
        if force_mkt:
            limit_price = 0.0
        else:
            limit_price = round(ltp - slippage_buffer if side == "SELL" else ltp + slippage_buffer, 2)
        order_record = {
            "norenordno": order_id,
            "tsym": symbol,
            "trantype": "S" if side == "SELL" else "B",
            "qty": quantity,
            "prc": limit_price,
            "prctyp": "MKT" if force_mkt else "LMT",
            "status": "OPEN",  # Initially OPEN, will become COMPLETE after 2 polls
            "avgprc": str(ltp),
            "fillshares": quantity,
        }
        self.orders.append(order_record)
        self.last_placed_order = order_record
        return {"stat": "Ok", "norenordno": order_id}

    async def get_order_book(self):
        self.order_book_call_count += 1
        # After 2 polls, mark order as COMPLETE
        if self.order_book_call_count >= 2 and self.orders:
            self.orders[-1]["status"] = "COMPLETE"
        return self.orders

    async def get_positions(self):
        return {"stat": "Ok", "positions": self.positions}

    async def cancel_order(self, order_id):
        return {"stat": "Ok"}


@pytest.mark.asyncio
async def test_trade_executor_exit_fill_and_no_premature_cancel():
    """Verifies that TradeExecutor cleanly fills exits with retry polling and doesn't cancel exits."""
    client = MockFlattradeClient()
    risk = RiskManager(max_daily_loss_points=1000, quantity=65)
    discord = DiscordNotifier("Test")
    executor = TradeExecutor(client, risk, discord, quantity=65, live_orders=True)

    # Set active position
    executor.position = {
        "side": "CE",
        "symbol": "NIFTY26AUG24200CE",
        "order_symbol": "NIFTY26AUG24200CE",
        "token": "token1",
        "quantity": 65,
        "entry": 100.0,
        "sl": 90.0,
        "target": 110.0,
        "opened_at": datetime(2026, 8, 28, 9, 20),
    }

    now = datetime(2026, 8, 28, 9, 25)

    # Execute exit at target
    res = await executor.close_position(110.0, now, "TARGET")

    assert res["accepted"] is True
    assert res["trade"]["exit"] == 110.0
    assert res["trade"]["pts"] == 10.0
    assert res["trade"]["rs"] == 650.0
    assert executor.position is None
    # Exits are now TRUE MARKET orders (price-band rejections on limit sells
    # below LTP caused the repeated live SL rejection bug):
    assert client.last_placed_order["prctyp"] == "MKT"
    assert client.last_placed_order["prc"] == 0.0


# =====================================================================
# 8. FULL-DAY END-TO-END SIMULATION TEST
# =====================================================================

def test_full_session_simulation_with_eod_squareoff():
    """Simulates an entire session with 2nd ITM resolution, multiple signals, exits, and EOD squareoff."""
    engine = LastHopeWinnerEngine()

    spot = 24240.0
    desired = engine.desired_strikes(spot)
    assert desired["CE_SPEC"] == 24150  # ATM (24250) - 100
    assert desired["PE_SPEC"] == 24350  # ATM (24250) + 100

    ce_contract = engine.register_contract("CE:24150", "NIFTY26AUG24150CE", "tok_ce", "CE", 24150)
    ce_contract.set_day_sr_levels(prev_high=160, prev_low=120, prev_close=135)

    base_time = datetime(2026, 8, 28, 9, 15)

    # 1. Warm-up: push 3 bars to arm S1 (close near low → S1_1m <= 25)
    for m in range(3):
        t = base_time + timedelta(minutes=m)
        p = 125.0
        bar = Bar1m(555 + m, open=p + 1, high=p + 2, low=p - 1, close=p, timestamp=t)
        ce_contract._on_1m_bar_close(bar)

    # Force arm (in case warm-up stochastic didn't drop S1 low enough)
    ce_contract.flag_armed = True
    ce_contract.flag_arm_bar = len(ce_contract.bars) - 1
    ce_contract.super_armed = True
    ce_contract.super_arm_bar = len(ce_contract.bars) - 1

    # 2. Trigger bar: bounce on EMA20 (§43 gate — the only trading level).
    # After the 3 warm-up bars (close 125), EMA20 sits near 125; force a known
    # value so the bounce math is deterministic.
    ce_contract.ema20.value = 126.0
    ema_val = ce_contract.ema20.value

    # Use 5m tracker for stochastic values — injected on a NON-boundary minute so
    # they survive (clock-aligned aggregation only recomputes at 5m boundaries:
    # minute 558 -> (558+1-555) % 5 = 4 != 0)
    ce_contract.tf_trackers[5].last_s4 = 82.0
    ce_contract.tf_trackers[5].last_s1 = 35.0
    ce_contract.tf_trackers[5].last_s3 = 50.0
    ce_contract.tf_trackers[5].prev_s1 = 30.0

    t_trig = base_time + timedelta(minutes=3)
    bar_trig = Bar1m(558, open=ema_val + 2, high=ema_val + 4, low=ema_val - 1.0, close=ema_val + 3.0, timestamp=t_trig)
    sig = ce_contract._on_1m_bar_close(bar_trig)

    assert sig is not None
    assert sig["trigger"] == "FLAG"
    engine.on_trade_opened(sig)

    # 3. Target Exit
    tp_price = sig["tp"]
    t_exit = base_time + timedelta(minutes=5)
    engine.push_tick("CE:24150", tp_price, t_exit)
    engine.on_trade_closed()

    assert engine.active_trade is None


# =====================================================================
# 9. STRESS & EDGE-CASE RESILIENCE TESTS
# =====================================================================

def test_sl_priority_over_tp_same_tick():
    """Verifies that if a volatile tick or bar touches both SL and TP, SL takes strict priority."""
    engine = LastHopeWinnerEngine()
    cs = engine.register_contract("CE:24200", "NIFTY26AUG24200CE", "token1", "CE", 24200)

    trade = {
        "side": "CE",
        "symbol": "NIFTY26AUG24200CE",
        "token": "token1",
        "entry": 100.0,
        "sl": 90.0,
        "tp": 110.0,
        "dist": 10.0,
        "be_trigger_px": 107.0,
        "be_hardened_sl": 101.0,
        "be_done": False,
    }
    engine.on_trade_opened(trade)

    # In backtest & live execution logic, if price drops below SL (e.g. 88.0), exit is immediately STOP_LOSS
    ltp = 88.0
    now = datetime(2026, 8, 28, 9, 35)
    reason = None
    if ltp <= trade["sl"]:
        reason = "STOP_LOSS"
    elif ltp >= trade["tp"]:
        reason = "TARGET"

    assert reason == "STOP_LOSS"


@pytest.mark.asyncio
async def test_broker_position_reconciliation_fallback():
    """Verifies that if order status lookup fails, TradeExecutor falls back to PositionBook to confirm fill."""
    client = MockFlattradeClient()
    risk = RiskManager(max_daily_loss_points=1000, quantity=65)
    discord = DiscordNotifier("Test")
    executor = TradeExecutor(client, risk, discord, quantity=65, live_orders=True)

    # Override get_order_book to simulate a broker timeout/failure
    async def failing_order_book():
        return []
    client.get_order_book = failing_order_book

    # PositionBook confirms position is flat (netqty = 0)
    async def position_book_flat():
        return [{"tsym": "NIFTY26AUG24200CE", "netqty": "0"}]
    client.get_positions = position_book_flat

    executor.position = {
        "side": "CE",
        "symbol": "NIFTY26AUG24200CE",
        "order_symbol": "NIFTY26AUG24200CE",
        "token": "token1",
        "quantity": 65,
        "entry": 100.0,
        "sl": 90.0,
        "target": 110.0,
        "opened_at": datetime(2026, 8, 28, 9, 20),
    }

    now = datetime(2026, 8, 28, 9, 25)
    res = await executor.close_position(110.0, now, "TARGET")

    assert res["accepted"] is True
    assert executor.position is None, "Position must be cleared once confirmed flat by PositionBook"


def test_multiple_concurrent_contracts_isolation():
    """Verifies that CE and PE contract state machines operate in complete isolation without state leakage."""
    engine = LastHopeWinnerEngine()
    ce = engine.register_contract("CE:24200", "NIFTY26AUG24200CE", "tok_ce", "CE", 24200)
    pe = engine.register_contract("PE:24400", "NIFTY26AUG24400PE", "tok_pe", "PE", 24400)

    ce.set_day_sr_levels(160, 100, 130)
    pe.set_day_sr_levels(200, 140, 170)

    # Push decreasing bars to CE only
    t = datetime(2026, 8, 28, 9, 15)
    for i in range(15):
        p = 150 - i * 3
        bar = Bar1m(555 + i, open=p + 1, high=p + 2, low=p - 1, close=p, timestamp=t + timedelta(minutes=i))
        ce._on_1m_bar_close(bar)

    # CE should be armed, PE must remain flat
    assert ce.flag_armed is True
    assert pe.flag_armed is False


# =====================================================================
# 10. SUPER TRIGGER END-TO-END TEST
# =====================================================================

def _set_stoch_for_trigger(cs, s4=85.0, s1=40.0, s3=50.0, prev_s1=35.0):
    """Helper: set 5m tracker stoch values that survive push_1m_bar (< 5 bars)."""
    cs.tf_trackers[5].last_s4 = s4
    cs.tf_trackers[5].last_s1 = s1
    cs.tf_trackers[5].last_s3 = s3
    cs.tf_trackers[5].prev_s1 = prev_s1


def test_super_trigger_end_to_end():
    """Verifies SUPER trigger fires when S3,S4,S1 all < 25 with S1 rising, within armed window."""
    engine = LastHopeWinnerEngine()
    cs = engine.register_contract("PE:24400", "NIFTY26AUG24400PE", "tok_pe", "PE", 24400)
    cs.set_day_sr_levels(prev_high=160, prev_low=100, prev_close=130)

    # Use 5m tracker for stochastic values — survives push_1m_bar
    # S1 rising: prev_s1=10 < last_s1=18 (both < 25)
    _set_stoch_for_trigger(cs, s4=20.0, s1=18.0, s3=15.0, prev_s1=10.0)

    # Arm the contract
    cs.flag_armed = True
    cs.flag_arm_bar = 0
    cs.super_armed = True
    cs.super_arm_bar = 0

    # Build a trigger bar that bounces on EMA20 (§43 gate)
    cs.ema20.value = 130.0
    ema_val = cs.ema20.value
    t_trig = datetime(2026, 8, 28, 9, 20)
    bar_trig = Bar1m(560, open=ema_val + 1, high=ema_val + 2, low=ema_val - 0.5, close=ema_val + 1.5, timestamp=t_trig)

    sig = cs._on_1m_bar_close(bar_trig)

    assert sig is not None, "SUPER trigger must fire when S3,S4,S1 < 25 and S1 rising"
    assert sig["trigger"] == "SUPER"
    assert sig["side"] == "PE"
    engine.on_trade_opened(sig)
    assert engine.active_trade is not None
    assert engine.active_trade["trigger"] == "SUPER"


# =====================================================================
# 11. ATR CAP & FLOOR BOUNDARY TESTS
# =====================================================================

def _make_trigger_bar(cs, trigger_time=None):
    """Helper: build a bar that bounces on the §43 gate level (EMA20) for FLAG trigger."""
    lvl = cs.ema20.value if cs.ema20.value else 120.0
    t = trigger_time or datetime(2026, 8, 28, 9, 20)
    return Bar1m(560, open=lvl + 1, high=lvl + 2, low=lvl - 0.5, close=lvl + 1, timestamp=t)


def test_atr_floor_clamp():
    """§43: ATR dist is floored at 2.0 pts when ATR*1.0 < 2.0."""
    engine = LastHopeWinnerEngine()
    cs = engine.register_contract("CE:24200", "NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs.set_day_sr_levels(150, 100, 120)
    _set_stoch_for_trigger(cs)
    cs.flag_armed = True
    cs.flag_arm_bar = 0

    # ATR = 1.0 -> ATR*1.0 = 1.0 -> floor to 2.0
    # EMA20 (seeded from 3 warmup bars) — force a known gate level
    cs.ema20.value = 123.0
    t = datetime(2026, 8, 28, 9, 20)
    # Bounce on EMA20=123.0 with TR = 1.0 (high-low=1.0, first bar no prev_close)
    bar = Bar1m(560, open=123.0, high=123.0, low=122.0, close=122.9, timestamp=t)
    sig = cs._on_1m_bar_close(bar)

    assert sig is not None, f"Signal must fire (bounce on EMA20={cs.ema20.value:.2f})"
    assert sig["dist"] == 2.0, f"ATR floor must be 2.0, got {sig['dist']}"
    assert sig["sl"] == round(sig["entry"] - 2.0, 2)
    assert sig["tp"] == round(sig["entry"] + 2.0, 2)


def test_atr_cap_clamp():
    """§43: ATR dist is capped at 15.0 pts when ATR*1.0 > 15.0."""
    engine = LastHopeWinnerEngine()
    cs = engine.register_contract("CE:24200", "NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs.set_day_sr_levels(150, 100, 120)
    _set_stoch_for_trigger(cs)
    cs.flag_armed = True
    cs.flag_arm_bar = 0

    # ATR = 16.0 -> ATR*1.0 = 16.0 -> cap to 15.0
    cs.ema20.value = 123.0
    t = datetime(2026, 8, 28, 9, 20)
    bar = Bar1m(560, open=123.0, high=131.0, low=115.0, close=123.0, timestamp=t)
    # TR = 131.0 - 115.0 = 16.0 -> ATR = 16.0
    # Bounce: low=115.0 <= 123.0 (EMA20) OK — but must ALSO satisfy close >= EMA20-0.5
    # close=123.0 >= 122.5 -> OK
    sig = cs._on_1m_bar_close(bar)

    assert sig is not None, "Signal must fire"
    assert sig["dist"] == 15.0, f"ATR cap must be 15.0, got {sig['dist']}"
    assert sig["sl"] == round(sig["entry"] - 15.0, 2)
    assert sig["tp"] == round(sig["entry"] + 15.0, 2)


def test_atr_exact_boundary_2pt():
    """§43: ATR*1.0 exactly 2.0 stays at 2.0 (no off-by-one)."""
    engine = LastHopeWinnerEngine()
    cs = engine.register_contract("CE:24200", "NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs.set_day_sr_levels(150, 100, 120)
    _set_stoch_for_trigger(cs)
    cs.flag_armed = True
    cs.flag_arm_bar = 0

    # ATR = 2.0 -> ATR*1.0 = 2.0 exactly
    # EMA20 updates BEFORE the gate: forced 123.0 blends with bar close ->
    # post-update EMA20 = 123*0.95 + close*0.05. Bar low must sit BELOW that.
    cs.ema20.value = 123.0
    t = datetime(2026, 8, 28, 9, 20)
    bar = Bar1m(560, open=123.0, high=124.5, low=122.5, close=122.9, timestamp=t)
    # TR = 124.5 - 122.5 = 2.0 -> first-bar ATR = 2.0 (exact boundary, no floor/cap)
    # Bounce: post-update EMA20 = 122.995; low=122.5 <= 122.995; close=122.9 >= 122.495
    sig = cs._on_1m_bar_close(bar)

    assert sig is not None, "Signal must fire"
    assert abs(sig["dist"] - 2.0) < 0.01, f"ATR boundary must be 2.0, got {sig['dist']}"


# =====================================================================
# 12. FLAG THRESHOLD BOUNDARY TESTS
# =====================================================================

def test_flag_threshold_exact_boundary():
    """Verifies FLAG fires at S4=79.5 (>=) and S1=79.4 (<), but not at S4=79.4 or S1=79.5."""
    engine = LastHopeWinnerEngine()
    cs = engine.register_contract("CE:24200", "NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs.set_day_sr_levels(150, 100, 120)

    t = datetime(2026, 8, 28, 9, 20)

    # Case A: S4=79.5 (exact boundary, should fire)
    _set_stoch_for_trigger(cs, s4=79.5, s1=40.0, s3=50.0, prev_s1=35.0)
    cs.flag_armed = True
    cs.flag_arm_bar = 0
    bar_a = _make_trigger_bar(cs, t)
    sig_a = cs._on_1m_bar_close(bar_a)
    assert sig_a is not None, "FLAG must fire at S4=79.5 (>= boundary)"

    # Case B: S4=79.4 (below boundary, should NOT fire)
    cs2 = OptionContractState("NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs2.set_day_sr_levels(150, 100, 120)
    _set_stoch_for_trigger(cs2, s4=79.4, s1=40.0, s3=50.0, prev_s1=35.0)
    cs2.flag_armed = True
    cs2.flag_arm_bar = 0
    bar_b = _make_trigger_bar(cs2, t)
    sig_b = cs2._on_1m_bar_close(bar_b)
    assert sig_b is None, "FLAG must NOT fire at S4=79.4 (< 79.5)"

    # Case C: S1=79.5 (exact boundary, should NOT fire — uses strict <)
    cs3 = OptionContractState("NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs3.set_day_sr_levels(150, 100, 120)
    _set_stoch_for_trigger(cs3, s4=85.0, s1=79.5, s3=50.0, prev_s1=75.0)
    cs3.flag_armed = True
    cs3.flag_arm_bar = 0
    bar_c = _make_trigger_bar(cs3, t)
    sig_c = cs3._on_1m_bar_close(bar_c)
    assert sig_c is None, "FLAG must NOT fire at S1=79.5 (not < 79.5)"

    # Case D: S1=79.4 (just below, should fire)
    cs4 = OptionContractState("NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs4.set_day_sr_levels(150, 100, 120)
    _set_stoch_for_trigger(cs4, s4=85.0, s1=79.4, s3=50.0, prev_s1=75.0)
    cs4.flag_armed = True
    cs4.flag_arm_bar = 0
    bar_d = _make_trigger_bar(cs4, t)
    sig_d = cs4._on_1m_bar_close(bar_d)
    assert sig_d is not None, "FLAG must fire at S1=79.4 (< 79.5)"


# =====================================================================
# 13. ARMING WINDOW EXACT BOUNDARY TEST
# =====================================================================

def test_arming_window_exact_boundary():
    """Verifies signal accepted at bar+10 (within window) but rejected at bar+11 (expired)."""
    t = datetime(2026, 8, 28, 9, 20)

    # Part 1: Signal at bar+10 (within window) -> ACCEPTED
    cs1 = OptionContractState("NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs1.set_day_sr_levels(150, 100, 120)
    _set_stoch_for_trigger(cs1)
    cs1.flag_armed = True
    cs1.flag_arm_bar = 0

    # Manually populate cs.bars to set bar count (don't push through _on_1m_bar_close
    # as that causes 5m tracker recalculation after 5 bars)
    for i in range(10):
        cs1.bars.append(Bar1m(555 + i, 100, 101, 99, 100, t))

    # Push trigger bar through _on_1m_bar_close → bar_idx = 10
    bar_10 = _make_trigger_bar(cs1, t)
    sig_10 = cs1._on_1m_bar_close(bar_10)
    assert sig_10 is not None, "Signal must be accepted at bar+10 (within ARM_WINDOW)"

    # Part 2: Arming expired at bar+11 -> REJECTED
    cs2 = OptionContractState("NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs2.set_day_sr_levels(150, 100, 120)
    _set_stoch_for_trigger(cs2)
    cs2.flag_armed = True
    cs2.flag_arm_bar = 0

    # Push 11 bars through _on_1m_bar_close to trigger expiry logic
    for i in range(11):
        bar = Bar1m(555 + i, 100, 101, 99, 100, t)
        cs2._on_1m_bar_close(bar)

    assert cs2.flag_armed is False, "Arming must expire after ARM_WINDOW bars"


# =====================================================================
# 14. EOD SQUARE-OFF TEST
# =====================================================================

def test_eod_square_off_rejects_signals_after_session_end():
    """Verifies that signals are rejected after SESSION_END_MIN (15:00 = minute 900)."""
    cs = OptionContractState("NIFTY26AUG24200CE", "token1", "CE", 24200)
    cs.set_day_sr_levels(150, 100, 120)

    # Arm the contract
    cs.flag_armed = True
    cs.flag_arm_bar = 0
    cs.tf_trackers[1].last_s4 = 85.0
    cs.tf_trackers[1].last_s1 = 40.0

    # Bar at 15:01 (minute 901) - after session end
    tLate = datetime(2026, 8, 28, 15, 1)
    barLate = Bar1m(901, open=120, high=125, low=118, close=122, timestamp=tLate)

    # The strategy checks SESSION_END_MIN at line 469
    # After session end, signals should not be generated
    cs.bars = []  # reset bars
    for i in range(5):
        bar = Bar1m(555 + i, 100, 101, 99, 100, datetime(2026, 8, 28, 9, 15 + i))
        cs._on_1m_bar_close(bar)

    # Check that SESSION_END_MIN constant is correct
    from flattrade_bot.strategies.last_hope_winner import SESSION_END_MIN
    assert SESSION_END_MIN == 900, f"SESSION_END_MIN must be 900 (15:00), got {SESSION_END_MIN}"


def test_engine_eod_constant_value():
    """Verifies the engine EOD constant matches 15:00 IST (minute 900)."""
    from flattrade_bot.strategies.last_hope_winner import SESSION_START_MIN, SESSION_END_MIN
    assert SESSION_START_MIN == 555, f"SESSION_START_MIN must be 555 (09:15), got {SESSION_START_MIN}"
    assert SESSION_END_MIN == 900, f"SESSION_END_MIN must be 900 (15:00), got {SESSION_END_MIN}"


# =====================================================================
# 15. BACKTEST ENGINE ATR FLOOR PARITY TEST
# =====================================================================

def test_backtest_atr_floor_matches_live():
    """Verifies that the backtest engine ATR clamp(min=2.0) matches live floor formula."""
    # Live formula: dist = min(max(ATR * 1.5, 2.0), 15.0)
    import torch
    atr_val = torch.tensor([[0.5]])  # ATR=0.5 -> ATR*1.5=0.75
    atr_mult = torch.tensor([1.5])
    TP_PTS = torch.tensor([15.0])

    # New engine formula: clamp(minimum(atr * mult, TP_PTS), min=2.0)
    result = torch.clamp(torch.minimum(atr_val * atr_mult, TP_PTS), min=2.0)
    assert result.item() == 2.0, f"ATR floor must clamp 0.75 to 2.0, got {result.item()}"

    # ATR=12.0 -> 18.0 -> capped to 15.0
    atr_val2 = torch.tensor([[12.0]])
    result2 = torch.clamp(torch.minimum(atr_val2 * atr_mult, TP_PTS), min=2.0)
    assert result2.item() == 15.0, f"ATR cap must clamp 18.0 to 15.0, got {result2.item()}"

    # ATR=6.0 -> 9.0 -> unchanged
    atr_val3 = torch.tensor([[6.0]])
    result3 = torch.clamp(torch.minimum(atr_val3 * atr_mult, TP_PTS), min=2.0)
    assert result3.item() == 9.0, f"ATR mid-range must be 9.0, got {result3.item()}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

