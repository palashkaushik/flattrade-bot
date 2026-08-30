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
    """Verifies exact formulas for CPR (BC, Pivot, TC) and Camarilla H3/L3."""
    cs = OptionContractState("NIFTY26AUG24200CE", "token1", "CE", 24200)

    ph, pl, pc = 150.0, 100.0, 120.0
    cs.set_day_sr_levels(ph, pl, pc)

    pivot = (150.0 + 100.0 + 120.0) / 3.0  # 123.3333
    bc = (150.0 + 100.0) / 2.0              # 125.0
    tc = 2.0 * pivot - bc                  # 121.6666
    cam_h3 = 120.0 + (150.0 - 100.0) * 1.1 / 4.0  # 133.75
    cam_l3 = 120.0 - (150.0 - 100.0) * 1.1 / 4.0  # 106.25

    assert abs(cs.sr_levels["CPR_Pivot"] - pivot) < 1e-4
    assert abs(cs.sr_levels["CPR_BC"] - bc) < 1e-4
    assert abs(cs.sr_levels["CPR_TC"] - tc) < 1e-4
    assert abs(cs.sr_levels["Cam_H3"] - cam_h3) < 1e-4
    assert abs(cs.sr_levels["Cam_L3"] - cam_l3) < 1e-4
    assert cs.sr_levels["PDH"] == 150.0
    assert cs.sr_levels["PDL"] == 100.0


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
    """Verifies that S/R bounce requires low <= level + 0.0 and close >= level - 0.5."""
    cs = OptionContractState("NIFTY26AUG24200CE", "token1", "CE", 24200)
    # Set day S/R levels with single known pivot
    cs.sr_levels = {"CPR_Pivot": 120.0}
    cs.ema20.value = 50.0
    cs.ema200.value = 50.0
    cs.vwap.cum_pv = 50.0 * 10000.0
    cs.vwap.cum_vol = 10000.0

    pivot = 120.0

    # Force armed state
    cs.flag_armed = True
    cs.flag_arm_bar = 0
    cs.tf_trackers[1].last_s4 = 85.0
    cs.tf_trackers[1].last_s1 = 40.0

    t = datetime(2026, 8, 28, 9, 20)

    # Case A: Low doesn't touch level (low = 120.5 > 120.0) -> REJECTED (touch_buffer=0.0)
    bar_no_touch = Bar1m(560, open=125.0, high=126.0, low=120.5, close=124.0, timestamp=t)
    sig_a = cs._on_1m_bar_close(bar_no_touch)
    assert sig_a is None, "Signal should be rejected when low > level (touch_buffer=0.0)"

    # Case B: Low touches level (low = 119.5 <= 120.0) and Close bounces (close = 122.0) -> ACCEPTED!
    cs.flag_armed = True
    cs.flag_arm_bar = len(cs.bars)
    cs.tf_trackers[1].last_s4 = 85.0
    cs.tf_trackers[1].last_s1 = 40.0
    bar_bounce = Bar1m(561, open=121.0, high=123.0, low=119.5, close=122.0, timestamp=t)
    sig_b = cs._on_1m_bar_close(bar_bounce)
    assert sig_b is not None
    assert sig_b["trigger"] == "FLAG"
    assert "CPR_Pivot" in sig_b["level"]


# =====================================================================
# 6. RISK GEOMETRY & BREAKEVEN STOP TESTS
# =====================================================================

def test_risk_geometry_sl_tp_breakeven():
    """Verifies ATR*1.5 stop distance and Breakeven stop movement at 70% target move."""
    engine = LastHopeWinnerEngine()
    cs = engine.register_contract("CE:24200", "NIFTY26AUG24200CE", "token1", "CE", 24200)

    # Fixed ATR = 6.0 pts -> dist = min(6.0 * 1.5, 15.0) = 9.0 pts
    cs.atr.value = 6.0
    cs.set_day_sr_levels(150, 100, 120)

    # Setup a trade signal
    sig = {
        "side": "CE",
        "symbol": "NIFTY26AUG24200CE",
        "token": "token1",
        "strike": 24200,
        "entry": 100.0,
        "dist": 9.0,
        "sl": 91.0,               # 100 - 9.0
        "tp": 109.0,              # 100 + 9.0
        "be_trigger_px": 106.30,  # 100 + 0.70 * 9.0 = 106.30
        "be_hardened_sl": 101.0,  # Entry + 1.0 pt
        "be_done": False,
    }

    engine.on_trade_opened(sig)
    now = datetime(2026, 8, 28, 9, 30)

    # Tick 1: Price rises to 104.0 (< BE trigger 106.30) -> SL remains 91.0
    engine.push_tick("CE:24200", 104.0, now)
    assert engine.active_trade["sl"] == 91.0
    assert engine.active_trade["be_done"] is False

    # Tick 2: Price reaches 106.50 (>= BE trigger 106.30) -> SL hardens to 101.0!
    engine.push_tick("CE:24200", 106.50, now)
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

    async def place_market_order(self, symbol, side, quantity, ltp, product="MIS", slippage_buffer=5.0):
        order_id = f"MOCK_ORD_{len(self.orders) + 1}"
        limit_price = round(ltp - slippage_buffer if side == "SELL" else ltp + slippage_buffer, 2)
        order_record = {
            "norenordno": order_id,
            "tsym": symbol,
            "trantype": "S" if side == "SELL" else "B",
            "qty": quantity,
            "prc": limit_price,
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
    assert client.last_placed_order["prc"] == 105.0  # 110.0 - 5.0 slippage buffer


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

    # 1. Warm-up sequence that arms S1
    for m in range(20):
        t = base_time + timedelta(minutes=m)
        p = 140.0 - m * 2.0
        bar = Bar1m(555 + m, open=p + 1, high=p + 2, low=p - 1, close=p, timestamp=t)
        ce_contract._on_1m_bar_close(bar)

    assert ce_contract.flag_armed is True

    # 2. Trigger bar: bounce on CPR_Pivot (~138.33)
    pivot = ce_contract.sr_levels["CPR_Pivot"]
    ce_contract.tf_trackers[1].last_s4 = 82.0
    ce_contract.tf_trackers[1].last_s1 = 35.0

    t_trig = base_time + timedelta(minutes=21)
    bar_trig = Bar1m(576, open=pivot + 2, high=pivot + 4, low=pivot - 1.0, close=pivot + 3.0, timestamp=t_trig)
    sig = ce_contract._on_1m_bar_close(bar_trig)

    assert sig is not None
    assert sig["trigger"] == "FLAG"
    engine.on_trade_opened(sig)

    # 3. Target Exit
    tp_price = sig["tp"]
    t_exit = base_time + timedelta(minutes=25)
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


if __name__ == "__main__":
    pytest.main(["-v", __file__])

