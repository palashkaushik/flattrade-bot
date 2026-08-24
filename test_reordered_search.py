import unittest

from artifacts.f6_hybrid.reordered_search import (
    build_stage_resources,
    canonical_trade,
    params_from_trial,
    score_candidate,
    stage_day_block,
)


class ReorderedSearchUnitTests(unittest.TestCase):
    def test_stage_resources_are_unique_increasing_and_end_at_full_window(self):
        days = [f"2020-01-{index:02d}" for index in range(1, 81)]
        self.assertEqual(build_stage_resources(days), [5, 20, 60, 80])
        self.assertEqual(build_stage_resources(days[:3]), [3])

    def test_canonical_trade_keeps_reference_fields_and_order(self):
        trade = {
            "date": "2020-01-02", "entry_min": 698, "exit_min": 699,
            "side": "CE", "symbol": "NIFTY24JAN12150CE", "entry": 94.15,
            "exit": 0.15, "pts": -94.0, "rs": -6110.0, "sl_pts": 6.0,
            "tp_pts": 30.0, "reason": "BEARISH_PEAK_REVERSAL",
            "duration_min": 1, "tf": "1m", "extra": "ignored",
        }
        self.assertEqual(
            canonical_trade(trade),
            ("2020-01-02", 698, 699, "CE", "NIFTY24JAN12150CE",
             94.15, 0.15, -94.0, -6110.0, 6.0, 30.0,
             "BEARISH_PEAK_REVERSAL", 1, "1m"),
        )

    def test_params_from_trial_sets_fixed_s1_d(self):
        class Trial:
            def suggest_categorical(self, name, values):
                return values[0]

        params = params_from_trial(Trial())
        self.assertEqual(params["s1_d"], 3)
        self.assertEqual(set(params) - {"s1_d"}, {
            "s1_k", "s4_k", "atr_period", "atr_sl_mult", "atr_tp_mult",
            "f6_s4_thresh", "f6_s1_thresh", "consec_loss",
        })

    def test_cost_aware_score_uses_net_trade_values(self):
        trades = [
            {"pts": 10.0, "entry": 100.0, "exit": 110.0},
            {"pts": -5.0, "entry": 100.0, "exit": 95.0},
        ]

        raw_score, _ = score_candidate(trades, cost_aware=False)
        net_score, stats = score_candidate(trades, cost_aware=True)

        self.assertLess(net_score, raw_score)
        self.assertEqual(stats["rs"], 45)

    def test_stage_blocks_only_add_new_days_and_include_previous_day_warmup(self):
        days = [f"2020-01-{index:02d}" for index in range(1, 31)]

        first_eval, first_block = stage_day_block(days, 0, 5)
        next_eval, next_block = stage_day_block(days, 5, 20)

        self.assertEqual(first_block, days[:5])
        self.assertEqual(first_eval, days[:5])
        self.assertEqual(next_block, days[5:20])
        self.assertEqual(next_eval, [days[4], *days[5:20]])


if __name__ == "__main__":
    unittest.main()
