import unittest

import grid_optimize_f6_atr as grid
from flattrade_bot.indicators.divergence import DivergenceEngine


class PivotDivergenceTests(unittest.TestCase):
    def test_reference_tracker_can_disable_divergence_filter(self):
        params = dict(grid.CHAMPION)
        params["use_divergence"] = False

        tracker = grid.TFTracker(10, params)

        self.assertFalse(tracker.use_divergence)

    def test_bullish_divergence_uses_price_lows_and_can_skip_weaker_pivot(self):
        engine = DivergenceEngine(pivot_left=1, pivot_right=1)
        rows = [
            (120.0, 120.0, 50.0),
            (118.0, 116.2, 7.2),   # first chart trough
            (120.0, 120.0, 20.0),
            (115.0, 115.0, 40.0),
            (112.0, 111.2, 17.0),  # intervening lower price pivot
            (115.0, 115.0, 20.0),
            (110.0, 110.0, 40.0),
            (109.0, 105.0, 13.7),  # lower price, higher S1 vs first trough
            (112.0, 107.6, 19.5),  # confirms the second pivot
        ]

        for close, low, s1 in rows:
            engine.update(close, s1, low_price=low, high_price=close)

        self.assertTrue(engine.has_bullish_trough_divergence())


if __name__ == "__main__":
    unittest.main()
