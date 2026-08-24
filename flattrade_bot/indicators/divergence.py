"""Dual-Mode Divergence Engine on Option Charts.

1. Bullish Trough Divergence (SUPER entry gate, user-specified rule):
   - A trough forms the moment the (1m/10s) S1 turns up: the declining leg has
     bottomed. The trough is the LOWEST LOW of that completed decline, with
     the S1 and S2 values at that bar.
   - The trough is only FULLY FORMED once S1 crosses back ABOVE the 20 level
     (the turn-up candle's recovery is confirmed). That crossing is when the
     divergence is looked for.
   - Bullish divergence: current confirmed trough price < previous confirmed
     trough price AND (current trough S1 > previous trough S1 OR current
     trough S2 > previous trough S2) — price makes a lower low while momentum
     (S1 or S2) makes a higher low.
   - The SUPER entry fires on the bar where S1 crosses above 20 AND the
     divergence is confirmed (close of that bar).

2. Legacy turn-up trough divergence (research scripts):
   - Same trough formation, compared on S1 only, without the 20-crossing
     confirmation (used by F6-style research backtests).

3. Bearish Peak Divergence (Reversal Trade Exit):
   - Price Peak 2 > Price Peak 1, Stochastic S1 Peak 2 < S1 Peak 1.
   - Pivot-confirmed (legacy) — used by reversal-exit research scripts.
"""

from collections import deque
from typing import List, Optional, Tuple


class DivergenceEngine:
    """Detects causal Trough and Peak divergences on Option Price and Stochastic S1/S2 charts."""

    def __init__(
        self,
        max_history: int = 40,
        min_lookback: int = 3,
        max_lookback: int = 30,
        pivot_left: int = 1,
        pivot_right: int = 1,
        confirm_cross_above: float = 20.0,
    ):
        self.max_history = max_history
        self.min_lookback = min_lookback
        self.max_lookback = max_lookback
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.confirm_cross_above = confirm_cross_above
        self.price_history = deque(maxlen=max_history)
        self.low_history = deque(maxlen=max_history)
        self.high_history = deque(maxlen=max_history)
        self.s1_history = deque(maxlen=max_history)

        # Turn-up trough tracking.
        self._n = 0                # total bars fed (monotonic index)
        self._last_s1 = None       # previous bar's S1
        self._declining = False    # S1 is currently in a declining leg
        self._leg = []             # (low, s1, s2) of the active declining leg
        self._troughs: List[Tuple[int, float, float]] = []  # (index, low, s1) — turn-up troughs (legacy)
        self._pending: Optional[Tuple[int, float, float, float]] = None  # (index, low, s1, s2) — unconfirmed trough
        self._confirmed: List[Tuple[int, float, float, float]] = []      # troughs confirmed by S1 crossing above 20
        self._crossed20 = False    # S1 crossed above 20 on the last update

    def update(
        self,
        close_price: float,
        s1_val: Optional[float],
        s2_val: Optional[float] = None,
        low_price: Optional[float] = None,
        high_price: Optional[float] = None,
    ):
        if s1_val is None:
            return
        if s2_val is None:
            s2_val = s1_val
        self.price_history.append(close_price)
        self.low_history.append(close_price if low_price is None else low_price)
        self.high_history.append(close_price if high_price is None else high_price)
        self.s1_history.append(s1_val)
        low = self.low_history[-1]
        self._crossed20 = False

        last = self._last_s1
        if last is None:
            # first bar — nothing to compare
            self._declining = False
            self._leg = []
        elif s1_val > last and self._declining:
            # S1 turn-up: the declining leg has bottomed. The trough is the
            # lowest low of this leg, with the S1 and S2 values at that bar.
            leg = self._leg + [(low, s1_val, s2_val)]
            min_low = min(p for p, s1_, s2_ in leg)
            trough_s1 = next(s1_ for p, s1_, s2_ in leg if p == min_low)
            trough_s2 = next(s2_ for p, s1_, s2_ in leg if p == min_low)
            self._troughs.append((self._n, min_low, trough_s1))
            self._pending = (self._n, min_low, trough_s1, trough_s2)
            if len(self._troughs) > self.max_history:
                self._troughs.pop(0)
            self._leg = []
            self._declining = False
        elif s1_val < last:
            # extending the declining leg
            self._leg.append((low, s1_val, s2_val))
            self._declining = True
        elif s1_val == last:
            # flat: keep the leg open if we were declining
            if self._declining:
                self._leg.append((low, s1_val, s2_val))
            else:
                self._leg = []
        else:
            # rising without a prior decline — no trough
            self._leg = []
            self._declining = False

        # S1 crossing back ABOVE the 20 level: the current trough becomes
        # fully formed (the recovery is confirmed). This is when the
        # divergence is looked for.
        if last is not None and last <= self.confirm_cross_above and s1_val > self.confirm_cross_above:
            if self._pending is not None:
                self._confirmed.append(self._pending)
                if len(self._confirmed) > self.max_history:
                    self._confirmed.pop(0)
                self._pending = None
            self._crossed20 = True

        self._last_s1 = s1_val
        self._n += 1

    def divergence_confirmed_at_last_update(self) -> Optional[Tuple[int, int]]:
        """Bullish divergence confirmed on the bar where S1 crossed above 20.

        Current confirmed trough price < previous confirmed trough price AND
        (current trough S1 > previous trough S1 OR current trough S2 > previous
        trough S2). Only valid right after an update() whose bar crossed above
        the confirm level; None otherwise (including when only one trough has
        been confirmed so far)."""
        if not self._crossed20:
            return None
        if len(self._confirmed) < 2:
            return None
        (i2, p2, s1_2, s2_2) = self._confirmed[-1]
        (i1, p1, s1_1, s2_1) = self._confirmed[-2]
        if p2 < p1 and (s1_2 > s1_1 or s2_2 > s2_1):
            return i1, i2
        return None

    def bullish_divergence_id(self) -> Optional[Tuple[int, int]]:
        """Return the current trough pair that forms bullish divergence.

        Legacy semantics (S1 only, turn-up troughs, no 20-crossing
        confirmation): current trough vs previous trough, price lower AND S1
        higher. Used by F6-style research scripts."""
        if len(self._troughs) < 2:
            return None
        (i2, p2, s2) = self._troughs[-1]
        (i1, p1, s1) = self._troughs[-2]
        if p2 < p1 and s2 > s1:
            return i1, i2
        return None

    def has_bullish_trough_divergence(self) -> bool:
        """Bullish Trough Divergence: Lower Price Trough & Higher S1 Trough (legacy)."""
        return self.bullish_divergence_id() is not None

    def _find_pivots(self, prices: List[float]) -> List[Tuple[int, float, float]]:
        """Find confirmed local pivots using a causal left/right window."""
        s1_vals = list(self.s1_history)
        n = len(prices)
        start = self.pivot_left
        end = n - self.pivot_right
        if end <= start:
            return []

        pivots = []
        for index in range(start, end):
            center = prices[index]
            left = prices[index - self.pivot_left:index]
            right = prices[index + 1:index + self.pivot_right + 1]
            neighbors = left + right
            if neighbors and center <= min(neighbors) and center < max(neighbors):
                pivots.append((index, center, s1_vals[index]))
        return pivots

    def _find_troughs(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Return the latest confirmed trough pair within the causal lookback.

        Legacy pivot-based detection — kept for diagnostic scripts."""
        pivots = self._find_pivots(list(self.low_history))
        if len(pivots) < 2:
            return None, None
        _, t2_price, t2_s1 = pivots[-1]
        t2_index = pivots[-1][0]
        prior = [pivot for pivot in pivots[:-1]
                 if self.min_lookback <= t2_index - pivot[0] <= self.max_lookback]
        if not prior:
            return None, None
        _, t1_price, t1_s1 = prior[-1]
        return (t1_price, t1_s1), (t2_price, t2_s1)

    def _find_peaks(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Return the latest confirmed peak pair within the causal lookback."""
        pivots = self._find_pivots(list(self.high_history))
        if len(pivots) < 2:
            return None, None
        _, p2_price, p2_s1 = pivots[-1]
        p2_index = pivots[-1][0]
        prior = [pivot for pivot in pivots[:-1]
                 if self.min_lookback <= p2_index - pivot[0] <= self.max_lookback]
        if not prior:
            return None, None
        _, p1_price, p1_s1 = prior[-1]
        return (p1_price, p1_s1), (p2_price, p2_s1)

    def has_bearish_peak_divergence(self) -> bool:
        """Bearish Peak Divergence (Reversal Exit): Higher Price Peak & Lower S1 Peak."""
        p1, p2 = self._find_peaks()
        if p1 is None or p2 is None:
            return False
        p1_price, p1_s1 = p1
        p2_price, p2_s1 = p2
        return p2_price > p1_price and p2_s1 < p1_s1
