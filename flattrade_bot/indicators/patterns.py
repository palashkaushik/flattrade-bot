"""Candlestick Pattern Engine — Bullish Pin Bar & Vicinity Breakout Confirmation."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    minute: int = 0

    @property
    def range(self) -> float:
        return max(0.001, self.high - self.low)

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)


class BullishPinBarDetector:
    """Detects Bullish Pin Bar candles and validates vicinity breakout confirmation."""

    # Lenient hammer-style geometry: keep a meaningful lower rejection while
    # allowing moderately larger bodies like the 09:20 CE candle.
    LOWER_SHADOW_MIN_RATIO = 0.45
    BODY_MAX_RATIO = 0.45
    UPPER_SHADOW_MAX_RATIO = 0.25

    @classmethod
    def is_bullish_pin_bar(cls, candle: Candle) -> bool:
        """Evaluates if a candle is a Bullish Pin Bar (lower price rejection).

        Rules:
          - Lower shadow (tail) >= 45% of total candle range.
          - Real body <= 45% of total candle range (located in top section).
          - Upper shadow <= 25% of total candle range.
        """
        r = candle.range
        if r <= 0.001:
            return False

        lower_ratio = candle.lower_shadow / r
        body_ratio = candle.body / r
        upper_ratio = candle.upper_shadow / r

        return (
            lower_ratio >= cls.LOWER_SHADOW_MIN_RATIO
            and body_ratio <= cls.BODY_MAX_RATIO
            and upper_ratio <= cls.UPPER_SHADOW_MAX_RATIO
        )

    @classmethod
    def check_vicinity_breakout(cls, candle_history: List[Candle], max_lookback: int = 10) -> bool:
        """Checks if current candle is the first break above a nearby pin-bar high.

        Parameters:
          candle_history: List of candles up to current candle.
          max_lookback: 10 candles for 1m, 5 candles for 2m.

        Rule: Any past candle C_k in last max_lookback is a Bullish Pin Bar,
        no intervening candle has broken C_k.high, and current_candle.high > C_k.high.
        """
        if len(candle_history) < 2:
            return False

        current_candle = candle_history[-1]
        lookback = min(len(candle_history) - 1, max_lookback)

        for i in range(1, lookback + 1):
            past_candle = candle_history[-1 - i]
            if cls.is_bullish_pin_bar(past_candle):
                intervening = candle_history[-i:-1]
                if (
                    current_candle.high > past_candle.high
                    and all(candle.high <= past_candle.high for candle in intervening)
                ):
                    return True

        return False
