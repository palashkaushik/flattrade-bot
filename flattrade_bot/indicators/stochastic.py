"""4-Stochastic Calculator Engine for Options Chart (S1, S2, S3, S4).

Parameters:
  S1: Fast %D (9, 3)
  S2: Medium %D (14, 3)
  S3: Slow %D (40, 4)
  S4: Trend %D (60, 10)
"""

from collections import deque
from typing import Dict, Optional, Tuple


class IncrementalStochastic:
    """Computes single stochastic %D with rolling high/low windows."""

    def __init__(self, k_period: int, d_period: int):
        self.k_period = k_period
        self.d_period = d_period
        self.highs = deque(maxlen=k_period)
        self.lows = deque(maxlen=k_period)
        self.raw_k_history = deque(maxlen=d_period)

    def push(self, high: float, low: float, close: float) -> Optional[float]:
        self.highs.append(high)
        self.lows.append(low)
        if len(self.highs) < self.k_period:
            return None

        hh = max(self.highs)
        ll = min(self.lows)
        raw_k = 50.0 if hh == ll else ((close - ll) / (hh - ll)) * 100.0
        self.raw_k_history.append(raw_k)

        if len(self.raw_k_history) < self.d_period:
            return None

        return sum(self.raw_k_history) / len(self.raw_k_history)


class QuadStochastics:
    """Calculates all 4 Stochastics (S1, S2, S3, S4) for option chart candles."""

    def __init__(
        self,
        s1_spec: Tuple[int, int] = (12, 3),
        s2_spec: Tuple[int, int] = (14, 3),
        s3_spec: Tuple[int, int] = (40, 4),
        s4_spec: Tuple[int, int] = (50, 10),
    ):
        self.s1 = IncrementalStochastic(*s1_spec)
        self.s2 = IncrementalStochastic(*s2_spec)
        self.s3 = IncrementalStochastic(*s3_spec)
        self.s4 = IncrementalStochastic(*s4_spec)

    def push(self, high: float, low: float, close: float) -> Dict[str, Optional[float]]:
        return {
            "s1d": self.s1.push(high, low, close),
            "s2d": self.s2.push(high, low, close),
            "s3d": self.s3.push(high, low, close),
            "s4d": self.s4.push(high, low, close),
        }
