import unittest
from pathlib import Path

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files
from f6_hybrid.incremental import (
    build_day_signal_state,
    run_incremental_candidates,
    simulate_day_signal_state,
)
from f6_hybrid.raw_features import FactorizedCandidatePool, run_factorized_candidates


DATA_ROOT = Path("C:/Websites/ammu/nifty_options")


@unittest.skipUnless(DATA_ROOT.exists(), "historical option data is unavailable")
class IncrementalReferenceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = dict(grid.CHAMPION)
        cls.spot_all = load_spot()
        cls.files = option_files("2020-01-01", "2020-01-07")
        cls.days = sorted(set(cls.files) & set(cls.spot_all))[:5]
        grid.init_worker_local(cls.spot_all)

    @staticmethod
    def canonical(trades):
        fields = (
            "date", "entry_min", "exit_min", "side", "symbol", "entry", "exit",
            "pts", "rs", "sl_pts", "tp_pts", "reason", "duration_min", "tf",
        )
        return [tuple(trade.get(field) for field in fields) for trade in trades]

    def test_signal_cache_split_matches_reference_trade_for_trade(self):
        for index, day in enumerate(self.days):
            fpath = str(self.files[day])
            fprev = str(self.files[self.days[index - 1]]) if index else ""
            reference = grid.process_day((day, fpath, fprev, self.params))
            state = build_day_signal_state(
                day, fpath, fprev, self.params, self.spot_all[day]
            )
            self.assertIsNotNone(state, day)
            optimized = simulate_day_signal_state(state, self.params)
            self.assertEqual(self.canonical(optimized), self.canonical(reference), day)

    def test_execution_variants_share_signal_builds(self):
        alternate = dict(self.params)
        alternate.update(atr_sl_mult=3.0, atr_tp_mult=6.0, consec_loss=8)
        trades, signal_builds = run_incremental_candidates(
            [self.params, alternate],
            self.days,
            self.files,
            self.spot_all,
            workers=1,
        )

        self.assertEqual(signal_builds, len(self.days))
        self.assertEqual(self.canonical(trades[0]), self.canonical(
            [
                trade
                for day_index, day in enumerate(self.days)
                for trade in grid.process_day(
                    (
                        day,
                        str(self.files[day]),
                        str(self.files[self.days[day_index - 1]]) if day_index else "",
                        self.params,
                    )
                )
            ]
        ))

    def test_factorized_feature_path_matches_reference(self):
        alternate = dict(self.params)
        alternate.update(atr_sl_mult=3.0, atr_tp_mult=6.0, consec_loss=8)
        trades, base_builds, signal_builds = run_factorized_candidates(
            [self.params, alternate],
            self.days,
            self.files,
            self.spot_all,
            workers=1,
        )

        expected = []
        for params in (self.params, alternate):
            candidate_trades = []
            for index, day in enumerate(self.days):
                candidate_trades.extend(
                    grid.process_day(
                        (
                            day,
                            str(self.files[day]),
                            str(self.files[self.days[index - 1]]) if index else "",
                            params,
                        )
                    )
                )
            expected.append(candidate_trades)

        self.assertEqual(base_builds, len(self.days))
        self.assertEqual(signal_builds, len(self.days))
        for actual, reference in zip(trades, expected):
            self.assertEqual(self.canonical(actual), self.canonical(reference))

    def test_factorized_candidate_pool_reuses_workers_across_batches(self):
        alternate = dict(self.params)
        alternate.update(atr_sl_mult=1.5, atr_tp_mult=6.0, consec_loss=6)
        with FactorizedCandidatePool(self.spot_all, workers=1) as pool:
            first, first_base, first_signal = pool.run(
                [self.params], self.days, self.files
            )
            second, second_base, second_signal = pool.run(
                [alternate], self.days, self.files
            )

        expected, _, _ = run_factorized_candidates(
            [alternate], self.days, self.files, self.spot_all, workers=1
        )
        self.assertEqual(self.canonical(second[0]), self.canonical(expected[0]))
        self.assertGreater(first_base, 0)
        self.assertGreater(first_signal, 0)
        self.assertGreater(second_base, 0)
        self.assertGreater(second_signal, 0)


if __name__ == "__main__":
    unittest.main()
