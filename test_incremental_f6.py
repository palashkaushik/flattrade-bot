import unittest

from backtest_monthly_ramp import resolve_exit_points, resolve_timeframes
from f6_hybrid.incremental import (
    DaySignalState,
    ExecutionKey,
    MinuteCursor,
    SignalKey,
    SignalStateCache,
    simulate_day_signal_state,
)


class SignalStateCacheTests(unittest.TestCase):
    def test_reuses_one_signal_state_for_multiple_execution_variants(self):
        cache = SignalStateCache()
        key = SignalKey(
            day="2020-01-02",
            previous_day="2020-01-01",
            s1_k=7,
            s1_d=3,
            s4_k=50,
            atr_period=10,
            f6_s4_thresh=75.0,
            f6_s1_thresh=20.5,
        )
        builds = []

        def build():
            builds.append(key)
            return {"events": [560]}

        first = cache.get_or_build(key, build)
        second = cache.get_or_build(key, build)

        self.assertIs(first, second)
        self.assertEqual(builds, [key])
        self.assertEqual(cache.build_count, 1)

    def test_signal_key_does_not_hide_signal_changing_parameters(self):
        base = dict(
            day="2020-01-02",
            previous_day="2020-01-01",
            s1_k=7,
            s1_d=3,
            s4_k=50,
            atr_period=10,
            f6_s4_thresh=75.0,
            f6_s1_thresh=20.5,
        )
        base_key = SignalKey(**base)

        for field, value in {
            "s1_k": 9,
            "s1_d": 4,
            "s4_k": 60,
            "atr_period": 14,
            "f6_s4_thresh": 79.5,
            "f6_s1_thresh": 25.0,
        }.items():
            changed = dict(base)
            changed[field] = value
            self.assertNotEqual(base_key, SignalKey(**changed), field)

    def test_execution_key_contains_only_stateful_execution_parameters(self):
        first = ExecutionKey(atr_sl_mult=2.0, atr_tp_mult=4.0, consec_loss=6)
        second = ExecutionKey(atr_sl_mult=3.0, atr_tp_mult=4.0, consec_loss=6)

        self.assertNotEqual(first, second)
        self.assertEqual(first.as_tuple(), (2.0, 4.0, 6))

    def test_fixed_exit_points_override_atr_exit_parameters(self):
        self.assertEqual(
            resolve_exit_points(
                atr_val=8.0,
                sl_mult=3.0,
                tp_mult=6.0,
                fallback_sl=10.0,
                fallback_tp=15.0,
                params={"fixed_sl_points": 10.0, "fixed_tp_points": 15.0},
            ),
            (10.0, 15.0),
        )

    def test_timeframe_filter_keeps_only_requested_execution_timeframes(self):
        self.assertEqual(
            resolve_timeframes({"timeframes": ["1m", "2m"]}),
            {"1m", "2m"},
        )


class MinuteCursorTests(unittest.TestCase):
    def test_advances_monotonically_without_binary_search(self):
        cursor = MinuteCursor(
            minutes=(560, 562, 565),
            values=((1, 2, 0, 1), (2, 3, 1, 2), (3, 4, 2, 3)),
        )

        self.assertEqual(cursor.at(560), (1, 2, 0, 1))
        self.assertIsNone(cursor.at(561))
        self.assertEqual(cursor.at(565), (3, 4, 2, 3))
        self.assertEqual(cursor.index, 2)

    def test_rejects_backward_queries(self):
        cursor = MinuteCursor(minutes=(560, 562), values=("a", "b"))
        cursor.at(562)

        with self.assertRaises(ValueError):
            cursor.at(560)

    def test_returns_latest_value_at_or_before_minute(self):
        cursor = MinuteCursor(minutes=(559, 562), values=(100.0, 102.0))

        self.assertEqual(cursor.latest_at_or_before(560), 100.0)
        self.assertEqual(cursor.latest_at_or_before(562), 102.0)
        self.assertIsNone(MinuteCursor((562,), (102.0,)).latest_at_or_before(560))


class SignalExecutionSplitTests(unittest.TestCase):
    def test_empty_signal_state_has_reference_empty_trade_result(self):
        state = DaySignalState(
            day="2020-01-02",
            spot={},
            prefix="",
            trackers={},
            slices={},
            pmtrig={},
        )
        params = {
            "atr_sl_mult": 2.0,
            "atr_tp_mult": 4.0,
            "consec_loss": 6,
        }

        self.assertEqual(simulate_day_signal_state(state, params), [])


if __name__ == "__main__":
    unittest.main()
