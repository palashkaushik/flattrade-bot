"""Incremental exponential moving average."""


class IncrementalEMA:
    """Causal EMA seeded with the simple average of its first period closes."""

    def __init__(self, period: int = 20):
        if period < 1:
            raise ValueError("EMA period must be positive")
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self._seed = []
        self.value = None

    def update(self, close: float):
        if self.value is None:
            self._seed.append(float(close))
            if len(self._seed) < self.period:
                return None
            self.value = sum(self._seed) / self.period
            self._seed.clear()
            return self.value
        self.value = self.alpha * float(close) + (1.0 - self.alpha) * self.value
        return self.value
