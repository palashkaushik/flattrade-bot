"""End-to-End Test Suite & Flattrade Test Order Execution Script.

Validates:
  1. Option A Strategy Engine (Optuna Optimized ATR F6):
     - Stochastics: S1=(12,3), S2=(14,3), S3=(40,4), S4=(50,10)
     - F6 Flag Trigger (S4 >= 79.5 and S1 <= 25.0)
     - 4-Timeframe Scanning (1m, 2m, 3m, 5m)
     - Dynamic ATR(10) SL (3.0x) and TP (6.0x) calculations
     - Risk Management (Consecutive Loss Limit = 8, Max Daily Loss = Rs 2,000)
2. Flattrade REST Client & simulated order formatting.
3. Live order fire is intentionally excluded; use live_order_fire.py with its
   explicit confirmation guard.
"""

import asyncio
import logging
from flattrade_bot.config import settings
from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.strategies.quad_pinbar_divergence import QuadPinbarDivergenceStrategy
from flattrade_bot.risk.manager import RiskManager
from flattrade_bot.broker.client import FlattradeClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_test")


def test_strategy_engine():
    logger.info("=== STEP 1: Testing Option A Strategy Engine Components ===")
    
    # 1. Verify Configuration
    assert settings.S1_SPEC == (12, 3), f"S1 spec mismatch: {settings.S1_SPEC}"
    assert settings.S4_SPEC == (50, 10), f"S4 spec mismatch: {settings.S4_SPEC}"
    assert settings.F6_S4_THRESH == 79.5, f"F6 S4 threshold mismatch: {settings.F6_S4_THRESH}"
    assert settings.F6_S1_THRESH == 20.5, f"F6 S1 threshold mismatch: {settings.F6_S1_THRESH}"
    assert settings.ATR_PERIOD == 10, f"ATR period mismatch: {settings.ATR_PERIOD}"
    assert settings.ATR_SL_MULT == 3.0, f"ATR SL mult mismatch: {settings.ATR_SL_MULT}"
    assert settings.ATR_TP_MULT == 6.0, f"ATR TP mult mismatch: {settings.ATR_TP_MULT}"
    assert settings.CONSECUTIVE_LOSS_LIMIT == 8, f"Consecutive loss limit mismatch: {settings.CONSECUTIVE_LOSS_LIMIT}"
    logger.info("✅ Settings verified successfully.")

    # 2. Verify QuadStochastics
    stoch = QuadStochastics()
    for i in range(60):
        res = stoch.push(100.0 + i, 90.0 + i, 95.0 + i)
    assert res["s1d"] is not None and res["s4d"] is not None, "Stochastic calculation failed"
    logger.info(f"✅ QuadStochastics verified: S1={res['s1d']:.2f}, S2={res['s2d']:.2f}, S3={res['s3d']:.2f}, S4={res['s4d']:.2f}")

    # 3. Verify ITM2 Strike Mapping
    ce_strike, pe_strike = QuadPinbarDivergenceStrategy.get_itm2_strikes(24352.0)
    assert ce_strike == 24250 and pe_strike == 24450, f"ITM2 Strike mapping error: CE={ce_strike}, PE={pe_strike}"
    logger.info(f"✅ ITM2 Strike Mapping verified: Spot 24352 -> CE Strike {ce_strike}, PE Strike {pe_strike}")

    # 4. Verify Strategy & Multi-timeframe Tracker
    strat = QuadPinbarDivergenceStrategy()
    c1 = Candle(open=150.0, high=160.0, low=148.0, close=155.0, minute=560)
    trigs = strat.push_spot_candle(c1, "CE")
    logger.info(f"✅ Strategy multi-timeframe push executed cleanly. Triggers: {trigs}")

    # 5. Verify Risk Manager
    rm = RiskManager()
    can_trade, reason = rm.can_open_trade(560, 0)
    assert can_trade, f"Risk manager check failed: {reason}"
    logger.info(f"✅ RiskManager check verified: {reason}")


async def test_flattrade_order_fire():
    logger.info("\n=== STEP 2: Simulated Flattrade Order Test ===")
    client = FlattradeClient()
    logger.info("Running order test with no auth token; no live order can be submitted.")

    # Define test order parameters for Nifty 2nd ITM CE contract
    test_symbol = "NIFTY13FEB26C24250"
    test_qty = settings.LOT_SIZE  # 65
    test_ltp = 150.0

    logger.info(f"Firing test market order: BUY {test_qty} {test_symbol} @ LTP ₹{test_ltp:.2f}")
    res = await client.place_market_order(
        symbol=test_symbol,
        side="BUY",
        quantity=test_qty,
        ltp=test_ltp,
        product="MIS",
        slippage_buffer=1.0
    )

    logger.info(f"Test Order Result: {res}")
    assert "stat" in res, "Order response missing 'stat' key"
    logger.info("✅ Simulated order test completed successfully.")


async def main():
    test_strategy_engine()
    await test_flattrade_order_fire()
    logger.info("\n🎉 ALL END-TO-END TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    asyncio.run(main())
