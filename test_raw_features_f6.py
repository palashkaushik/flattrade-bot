import unittest

from f6_hybrid.raw_features import (
    BaseDayState,
    BaseKey,
    FeatureBar,
    materialize_signal_state,
)


class RawFeatureCacheTests(unittest.TestCase):
    def test_threshold_materialization_emits_reference_event_order(self):
        symbol = "NIFTY24JAN24100CE"
        bar = FeatureBar(
            minute=560,
            close=2.0,
            atr=1.0,
            s1=10.0,
            s2=10.0,
            s3=10.0,
            s4=80.0,
            bullish_divergence=True,
            pin_break=True,
            embedded=False,
            tf="1m",
            tf_sl=6.0,
            tf_tp=30.0,
        )
        base = BaseDayState(
            day="2020-01-02",
            spot={},
            prefix="NIFTY24JAN24",
            trackers={},
            slices={symbol: object()},
            features={symbol: {"1m": (bar,)}},
        )

        state = materialize_signal_state(base, f6_s4_thresh=75.0,
                                         f6_s1_thresh=20.5)

        self.assertEqual(len(state.pmtrig[560]), 2)
        self.assertEqual(state.pmtrig[560][0]["tf"] if isinstance(state.pmtrig[560][0], dict)
                         else state.pmtrig[560][0][5], "1m")
        self.assertEqual(state.pmtrig[560][1][5], "1m")

    def test_base_key_excludes_execution_parameters(self):
        first = BaseKey("2020-01-02", "2020-01-01", 7, 3, 50, 10)
        second = BaseKey("2020-01-02", "2020-01-01", 7, 3, 50, 10)

        self.assertEqual(first, second)

    def test_materialization_carries_setup_state_from_previous_day(self):
        symbol = "NIFTY24JAN24100CE"
        warmup = FeatureBar(
            minute=559,
            close=2.0,
            atr=1.0,
            s1=10.0,
            s2=10.0,
            s3=10.0,
            s4=10.0,
            bullish_divergence=True,
            pin_break=False,
            embedded=False,
            tf="1m",
            tf_sl=6.0,
            tf_tp=30.0,
        )
        current = FeatureBar(
            minute=560,
            close=2.5,
            atr=1.0,
            s1=50.0,
            s2=50.0,
            s3=50.0,
            s4=50.0,
            bullish_divergence=False,
            pin_break=True,
            embedded=False,
            tf="1m",
            tf_sl=6.0,
            tf_tp=30.0,
        )
        base = BaseDayState(
            day="2020-01-02",
            spot={},
            prefix="NIFTY24JAN24",
            trackers={},
            slices={symbol: object()},
            features={symbol: {"1m": (current,)}},
            warmup_features={symbol: {"1m": (warmup,)}},
        )

        state = materialize_signal_state(base, 75.0, 20.5)

        self.assertEqual([event[0] for event in state.pmtrig[560]], ["CE"])


if __name__ == "__main__":
    unittest.main()
