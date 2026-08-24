"""Verification and Audit Suite for Backtest & Indicator Logic."""

import unittest
from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.divergence import DivergenceEngine


class TestStrategyComponents(unittest.TestCase):

    def test_bullish_pin_bar_detection(self):
        """Test Bullish Pin Bar pattern recognition."""
        # Hammer candle: High 100, Low 90, Open 98, Close 99 -> Range 10, Body 1, Lower 8, Upper 1
        pin = Candle(open=98.0, high=100.0, low=90.0, close=99.0)
        self.assertTrue(BullishPinBarDetector.is_bullish_pin_bar(pin))

        # Normal bullish candle: High 100, Low 90, Open 91, Close 99 -> Lower shadow small
        normal = Candle(open=91.0, high=100.0, low=90.0, close=99.0)
        self.assertFalse(BullishPinBarDetector.is_bullish_pin_bar(normal))

    def test_pin_bar_breakout_confirmation(self):
        """Test Pin Bar breakout confirmation."""
        pin = Candle(open=98.0, high=100.0, low=90.0, close=99.0)
        
        # Next candle closes above Pin Bar High (100.0) -> Confirmed
        next_valid = Candle(open=99.5, high=102.0, low=99.0, close=101.5)
        self.assertTrue(BullishPinBarDetector.is_breakout_confirmed(pin, next_valid))

        # Next candle closes below Pin Bar High -> Not confirmed
        next_invalid = Candle(open=99.5, high=100.5, low=98.0, close=99.8)
        self.assertFalse(BullishPinBarDetector.is_breakout_confirmed(pin, next_invalid))

    def test_bullish_trough_divergence(self):
        """Test Bullish Trough Divergence engine."""
        div = DivergenceEngine()
        
        # Trough 1 at t=5: Price 100, S1 15
        for i in range(10):
            p = 100.0 if i == 5 else 105.0 + i
            s = 15.0 if i == 5 else 30.0 + i
            div.update(p, s)

        # Trough 2 at t=15: Price 95 (Lower than 100), S1 22 (Higher than 15)
        for i in range(10):
            p = 95.0 if i == 5 else 102.0 + i
            s = 22.0 if i == 5 else 35.0 + i
            div.update(p, s)

        self.assertTrue(div.has_bullish_trough_divergence())


if __name__ == "__main__":
    unittest.main()
