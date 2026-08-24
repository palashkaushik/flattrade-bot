"""Incremental Elder Impulse System for option-chart closes."""

from typing import Optional

from flattrade_bot.indicators.ema import IncrementalEMA


class IncrementalElderImpulse:
    """EMA(13) slope plus MACD(12,26,9) histogram slope."""

    def __init__(self):
        self.ema13 = IncrementalEMA(13)
        self.ema12 = IncrementalEMA(12)
        self.ema26 = IncrementalEMA(26)
        self.macd_ema9 = IncrementalEMA(9)
        self.prev_ema13: Optional[float] = None
        self.prev_hist: Optional[float] = None
        self.color = "blue"

    def update(self, close: float) -> str:
        ema13 = self.ema13.update(close)
        ema12 = self.ema12.update(close)
        ema26 = self.ema26.update(close)
        color = "blue"
        if ema12 is not None and ema26 is not None:
            histogram = self.macd_ema9.update(ema12 - ema26)
            if (
                ema13 is not None
                and histogram is not None
                and self.prev_ema13 is not None
                and self.prev_hist is not None
            ):
                if ema13 > self.prev_ema13 and histogram > self.prev_hist:
                    color = "green"
                elif ema13 < self.prev_ema13 and histogram < self.prev_hist:
                    color = "red"
            self.prev_hist = histogram
        self.prev_ema13 = ema13
        self.color = color
        return color


def elder_allows(color: str, side: str, mode: str = "permissive") -> bool:
    """Return whether an Elder color permits a CE/PE entry."""
    if mode == "permissive":
        return color in (("green", "blue") if side == "CE" else ("red", "blue"))
    if mode == "strict":
        return color == ("green" if side == "CE" else "red")
    raise ValueError(f"Unknown Elder mode: {mode}")
