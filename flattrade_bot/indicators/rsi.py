"""Incremental RSI(14) for live bar-by-bar computation."""

from typing import Optional


class IncrementalRSI:
    """Wilder's RSI computed incrementally on each bar close."""

    def __init__(self, period: int = 14):
        self.period = period
        self.closes: list[float] = []
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.value: Optional[float] = None

    def update(self, close: float) -> Optional[float]:
        self.closes.append(close)
        if len(self.closes) < 2:
            return None

        if self.avg_gain is None:
            # Need period+1 closes for initial calculation
            if len(self.closes) < self.period + 1:
                return None
            # Initial SMA of gains/losses
            gains = []
            losses = []
            for i in range(1, self.period + 1):
                delta = self.closes[i] - self.closes[i - 1]
                if delta > 0:
                    gains.append(delta)
                    losses.append(0.0)
                else:
                    gains.append(0.0)
                    losses.append(abs(delta))
            self.avg_gain = sum(gains) / self.period
            self.avg_loss = sum(losses) / self.period
        else:
            # Wilder's smoothing
            delta = close - self.closes[-2]
            gain = max(delta, 0.0)
            loss = abs(min(delta, 0.0))
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        if self.avg_loss == 0:
            self.value = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            self.value = round(100.0 - (100.0 / (1.0 + rs)), 2)
        return self.value
