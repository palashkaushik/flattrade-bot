import tempfile
import unittest
from pathlib import Path

import pandas as pd

import grid_optimize_f6_atr as grid


class DayDataNormalizationTests(unittest.TestCase):
    def test_cached_day_deduplicates_symbol_minute_rows(self):
        frame = pd.DataFrame([
            {"time": "09:15:00", "symbol": "NIFTYCE", "open": 1, "high": 2, "low": 1, "close": 2},
            {"time": "09:15:00", "symbol": "NIFTYCE", "open": 1, "high": 2, "low": 1, "close": 2},
            {"time": "09:16:00", "symbol": "NIFTYCE", "open": 2, "high": 3, "low": 2, "close": 3},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "day.csv"
            frame.to_csv(path, index=False)
            grid.init_worker_local({})
            data = grid.cached_day(str(path))

        self.assertEqual(data["NIFTYCE"]["min"].tolist(), [555, 556])
        self.assertEqual(data["NIFTYCE"]["close"].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
