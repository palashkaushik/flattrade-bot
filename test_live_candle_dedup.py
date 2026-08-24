import unittest

from flattrade_bot.main import latest_new_candle, unseen_candles


class LiveCandleDedupTests(unittest.TestCase):
    def test_returns_latest_row_only_when_timestamp_is_new(self):
        rows = [{"time": "11-08-2026 14:37:00", "close": 100.0}]

        row, timestamp = latest_new_candle(rows, "11-08-2026 14:36:00")

        self.assertIs(row, rows[-1])
        self.assertEqual(timestamp, "11-08-2026 14:37:00")

    def test_skips_duplicate_latest_row(self):
        rows = [{"time": "11-08-2026 14:37:00", "close": 100.0}]

        row, timestamp = latest_new_candle(rows, "11-08-2026 14:37:00")

        self.assertIsNone(row)
        self.assertEqual(timestamp, "11-08-2026 14:37:00")

    def test_skips_rows_without_timestamp(self):
        row, timestamp = latest_new_candle([{"close": 100.0}], None)

        self.assertIsNone(row)
        self.assertIsNone(timestamp)

    def test_returns_all_unseen_rows_in_chronological_order(self):
        rows = [
            {"time": "11-08-2026 14:37:00", "close": 100.0},
            {"time": "11-08-2026 14:38:00", "close": 101.0},
            {"time": "11-08-2026 14:39:00", "close": 102.0},
        ]

        result = unseen_candles(rows, "11-08-2026 14:37:00")

        self.assertEqual([row["close"] for row in result], [101.0, 102.0])


if __name__ == "__main__":
    unittest.main()
