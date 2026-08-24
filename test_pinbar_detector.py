"""Regression tests for the lenient bullish pin-bar definition."""

from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector


def test_visual_0920_ce_candle_is_accepted_as_a_pin_bar():
    candle = Candle(open=236.35, high=240.0, low=232.4, close=239.25)

    assert BullishPinBarDetector.is_bullish_pin_bar(candle)


def test_breakout_after_visual_0920_ce_candle_is_confirmed():
    pin_bar = Candle(open=236.35, high=240.0, low=232.4, close=239.25)
    breakout = Candle(open=237.25, high=244.85, low=237.25, close=242.65)

    assert BullishPinBarDetector.check_vicinity_breakout([pin_bar, breakout])


def test_intrabar_high_break_after_visual_0935_ce_pinbar_is_confirmed():
    pin_bar = Candle(open=184.25, high=185.50, low=182.50, close=184.80)
    breakout = Candle(open=184.55, high=186.90, low=183.75, close=183.75)

    assert BullishPinBarDetector.check_vicinity_breakout([pin_bar, breakout])


def test_delayed_candle_is_not_a_new_break_after_high_was_already_broken():
    pin_bar = Candle(open=189.80, high=190.30, low=185.80, close=187.90)
    first_break = Candle(open=187.25, high=195.45, low=185.35, close=195.00)
    delayed_candle = Candle(open=194.00, high=194.60, low=186.75, close=186.75)

    assert not BullishPinBarDetector.check_vicinity_breakout(
        [pin_bar, first_break, delayed_candle]
    )


def test_body_that_is_still_too_large_is_rejected():
    candle = Candle(open=100.0, high=105.0, low=95.0, close=105.0)

    assert not BullishPinBarDetector.is_bullish_pin_bar(candle)
