import unittest

from artifacts.f6_hybrid.backtest_champion_walkforward_2020_2026 import (
    FOLDS,
    select_fold_days,
)


class ChampionWalkForwardTests(unittest.TestCase):
    def test_folds_cover_true_oos_years_through_2026_without_future_leakage(self):
        days = [
            "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-29",
            "2024-12-31", "2025-12-31", "2026-01-02",
        ]
        selected = select_fold_days(days, FOLDS[2])

        self.assertEqual(selected["oos"], ["2025-12-31"])
        self.assertEqual(selected["warmup"], ["2024-12-31"])
        self.assertLess(selected["warmup"][-1], selected["oos"][0])

    def test_fold_definitions_are_ordered_and_true_oos_only(self):
        self.assertEqual(
            [(fold["is_end"], fold["oos_year"]) for fold in FOLDS],
            [("2022", "2023"), ("2023", "2024"), ("2024", "2025"), ("2025", "2026")],
        )


if __name__ == "__main__":
    unittest.main()
