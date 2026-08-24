import unittest


from backtest_blind_2024_2026 import select_days


class BlindBacktestTests(unittest.TestCase):
    def test_select_days_requires_both_option_and_spot_dates(self):
        files = {
            "2024-12-31": "option-file",
            "2025-01-02": "option-file",
            "2025-01-03": "option-file",
        }
        spot = {
            "2025-01-02": object(),
            "2025-01-04": object(),
        }

        self.assertEqual(
            select_days(files, spot, "2025-01-01", "2025-01-03"),
            ["2025-01-02"],
        )
