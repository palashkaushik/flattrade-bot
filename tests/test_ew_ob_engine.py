"""Unit tests for the Elliott OB engine core."""

from artifacts.ew_ob.ew_ob_engine import (
    Anchor,
    Bar,
    CandidateOB,
    EWOBEngine,
    GREEN,
    Impulse,
    OrderBlock,
    OrderBlockTracker,
    RED,
    Wave,
    WaveDetector,
)
from artifacts.ew_ob.ew_ob_runner import resample_ohlc_tf
import numpy as np


def bar(gi, open_, high, low, close, day="2026-08-20", minute=None):
    if minute is None:
        minute = 555 + gi
    return Bar(gi, day, minute, open_, high, low, close)


def anchor(gi, price, kind):
    return Anchor(gi=gi, minute=555 + gi, price=price, kind=kind)


def impulse(direction, start_gi=0, w2_gi=2, w2_price=103,
            w4_gi=4, w4_price=105, end_gi=5, timeframe=1):
    values = [
        anchor(start_gi, 100 if direction == "bull" else 120,
               RED if direction == "bull" else GREEN),
        anchor(start_gi + 1, 110 if direction == "bull" else 90,
               GREEN if direction == "bull" else RED),
        anchor(w2_gi, w2_price, RED if direction == "bull" else GREEN),
        anchor(start_gi + 3, 108 if direction == "bull" else 88,
               GREEN if direction == "bull" else RED),
        anchor(w4_gi, w4_price, RED if direction == "bull" else GREEN),
        anchor(end_gi, 115 if direction == "bull" else 80,
               GREEN if direction == "bull" else RED),
    ]
    waves = [Wave(a.gi, a.gi, a.price, a.price, (a.kind,)) for a in values[1:]]
    return Impulse(direction, start_gi, end_gi, *waves, timeframe)


def test_bull_anchor_window_arms_on_b():
    detector = WaveDetector()
    detector.history = [bar(i, 70 + i, 80 + i, 60 + i, 75 + i) for i in range(8)]
    detector.anchors = [
        anchor(0, 100, RED),
        anchor(1, 110, GREEN),
        anchor(2, 103, RED),
        anchor(3, 108, GREEN),
        anchor(4, 105, RED),
        anchor(5, 115, GREEN),
        anchor(6, 107, RED),
    ]
    detector._advance()

    assert detector.impulse is not None
    assert detector.impulse.direction == "bull"
    assert detector.impulse.start_gi == 0
    assert detector.impulse.end_gi == 5
    assert detector.impulse.w5_second_last_open == 74
    assert detector.armed is False

    detector.anchors.append(anchor(7, 112, GREEN))
    detector._advance()
    assert detector.armed is True
    assert detector.armed_gi == 7
    assert detector.consume_arm() is not None
    assert detector.consume_arm() is None


def test_bear_condition_allows_half_point_origin_tolerance():
    detector = WaveDetector()
    detector.anchors = [
        anchor(0, 100.0, GREEN),
        anchor(1, 90.0, RED),
        anchor(2, 100.4, GREEN),
        anchor(3, 85.0, RED),
        anchor(4, 99.0, GREEN),
        anchor(5, 80.0, RED),
        anchor(6, 92.0, GREEN),
    ]
    detector._advance()

    assert detector.impulse is not None
    assert detector.impulse.direction == "bear"


def test_nested_bull_window_is_rejected_when_c_breaks_origin():
    detector = WaveDetector()
    detector.anchors = [
        anchor(0, 100, RED), anchor(1, 110, GREEN), anchor(2, 103, RED),
        anchor(3, 108, GREEN), anchor(4, 105, RED), anchor(5, 115, GREEN),
        anchor(6, 95, RED),
    ]
    detector._advance()
    assert detector.impulse is None
    assert detector._pending is None


def test_same_direction_impulse_requires_a_deeper_origin():
    detector = WaveDetector()
    detector._last_impulse = impulse("bull", end_gi=5)
    detector.anchors = [
        anchor(10, 121, RED), anchor(11, 130, GREEN), anchor(12, 123, RED),
        anchor(13, 128, GREEN), anchor(14, 125, RED), anchor(15, 135, GREEN),
        anchor(16, 126, RED),
    ]
    detector._advance()
    assert detector.impulse is None


def test_failed_window_slides_to_the_next_anchor():
    detector = WaveDetector()
    detector.anchors = [
        anchor(0, 100, RED), anchor(1, 110, GREEN), anchor(2, 103, RED),
        anchor(3, 108, GREEN), anchor(4, 105, RED), anchor(5, 109, GREEN),
        anchor(6, 95, RED), anchor(7, 90, GREEN),
    ]
    detector._advance()

    assert detector.impulse is not None
    assert detector.impulse.direction == "bear"
    assert detector.impulse.start_gi == 1


def test_resampled_wave_candle_uses_start_timestamp_and_end_index():
    spot = {
        "min": np.array([555, 556, 557, 558, 559, 560]),
        "open": np.array([100, 101, 102, 103, 104, 105], dtype=float),
        "high": np.array([101, 102, 103, 104, 105, 106], dtype=float),
        "low": np.array([99, 100, 101, 102, 103, 104], dtype=float),
        "close": np.array([101, 102, 101, 104, 105, 104], dtype=float),
    }
    opens, highs, lows, closes, ends, starts = resample_ohlc_tf(spot, 3)

    assert starts.tolist() == [555, 558]
    assert ends.tolist() == [2, 5]
    assert opens.tolist() == [100.0, 103.0]
    assert closes.tolist() == [101.0, 104.0]


def test_engine_has_shared_wave_detectors_without_four_minute_tf():
    engine = EWOBEngine()

    assert set(engine.wave_detectors) == {1, 2, 3, 5}
    assert engine.wave is engine.wave_detectors[1]


def test_order_block_selection_targets_w2_for_bull():
    tracker = OrderBlockTracker()
    tracker.registry = [
        OrderBlock(1, 24195.95, 24199.80, 14),
        OrderBlock(2, 24195.95, 24202.85, 15),
        OrderBlock(1, 24208.90, 24213.35, 30),
    ]
    imp = impulse("bull", start_gi=10, w2_gi=12, w2_price=24196,
                  w4_gi=14, w4_price=24200, end_gi=15)
    selected = tracker.select_for_impulse(imp)

    assert len(selected) == 1
    assert selected[0].ob.formed_gi == 14
    assert selected[0].ob.lo == 24195.95


def test_order_block_selection_targets_one_minute_w4_for_bear():
    tracker = OrderBlockTracker()
    tracker.registry = [
        OrderBlock(2, 24241.90, 24247.85, 20),
        OrderBlock(1, 24243.50, 24249.85, 21),
        OrderBlock(3, 24242.05, 24249.85, 22),
    ]
    imp = impulse("bear", start_gi=10, w2_gi=12, w2_price=24252,
                  w4_gi=20, w4_price=24250, end_gi=25)
    selected = tracker.select_for_impulse(imp)

    assert len(selected) == 1
    assert selected[0].ob.tf == 1
    assert selected[0].ob.formed_gi == 21


def test_three_minute_setup_can_use_shared_w5_order_block():
    tracker = OrderBlockTracker()
    tracker.registry = [
        OrderBlock(1, 100, 110, 18),
        OrderBlock(5, 80, 90, 21),
    ]
    imp = impulse("bear", start_gi=10, w2_gi=12, w2_price=103,
                  w4_gi=18, w4_price=105, end_gi=20, timeframe=3)
    selected = tracker.select_for_impulse(imp)

    assert len(selected) == 1
    assert selected[0].ob.tf == 5
    assert selected[0].ob.formed_gi == 21


def test_pattern_a_ob():
    tracker = OrderBlockTracker()
    tracker.feed_tf_bars(tf=1, high=[100, 99, 101, 102], low=[98, 96, 97, 100], gi=[0, 1, 2, 3])

    assert len(tracker.registry) == 1
    ob = tracker.registry[0]
    assert (ob.lo, ob.hi, ob.formed_gi) == (96, 99, 2)


def test_pattern_b_ob():
    tracker = OrderBlockTracker()
    tracker.feed_tf_bars(tf=5, high=[98, 102, 103, 102], low=[95, 99, 97, 100], gi=[0, 1, 2, 3])

    assert len(tracker.registry) == 1
    ob = tracker.registry[0]
    assert (ob.lo, ob.hi, ob.formed_gi) == (99, 102, 2)


def test_snapshot_filters_by_impulse_window():
    tracker = OrderBlockTracker()
    tracker.registry = [
        OrderBlock(1, 96, 99, 0),
        OrderBlock(1, 95, 101, 2),
        OrderBlock(1, 90, 100, 5),
    ]
    imp = type("I", (), {"direction": "bull", "start_gi": 1, "end_gi": 3})()
    candidates = tracker.snapshot(imp)

    assert len(candidates) == 1
    assert candidates[0].ob.formed_gi == 2
    assert candidates[0].side == "bull"


def _fake_resolver(close_map=None):
    close_map = close_map or {}
    return lambda day, side, minute, spot_px, strike=None: close_map.get(minute, 100.0)


def _armed_engine(direction="bull", **kwargs):
    engine = EWOBEngine(tol=0.5, **kwargs)
    engine.resolve_option = _fake_resolver()
    engine._trs = [1.0] * 10
    engine._prev_close = 100.0
    engine.armed = True
    engine._arm_gi = 0
    engine._arm_minute = 555
    side = direction
    ob = OrderBlock(1, 139, 151, 1)
    candidate = CandidateOB(
        ob=ob,
        side=side,
        untouched_top=ob.hi,
        untouched_bot=ob.lo,
        eligible_from_gi=1,
    )
    engine.candidates = [candidate]
    engine._fallback_candidate = candidate
    return engine


def test_engine_accepts_half_point_near_edge_entry():
    engine = _armed_engine()
    engine.feed(bar(1, 180, 182, 151.4, 151.5, minute=556))

    assert engine.pos is not None
    assert engine.pos.entry_min == 556


def test_engine_enters_and_tps():
    engine = _armed_engine()
    engine.feed(bar(1, 180, 182, 180, 181, minute=556))
    engine.feed(bar(2, 181, 185, 179, 184, minute=557))
    engine.feed(bar(3, 184, 152, 148, 149, minute=558))
    assert engine.pos is not None
    assert engine.pos.side == "CE"
    assert engine.pos.entry_min == 558
    assert engine.pos.sl == 139

    engine.feed(bar(4, 200, 210, 199, 209, minute=559))
    assert engine.pos is None
    assert engine.trades[0]["exit_reason"] == "TP"


def test_engine_supports_atr_sl_and_tp_multipliers():
    engine = _armed_engine(risk_mode="atr", sl_mult=3.0, tp_atr_mult=5.0)
    engine._prev_close = 149.0
    engine.feed(bar(1, 180, 150, 148, 149, minute=556))

    assert engine.pos is not None
    assert round(engine.pos.sl, 2) == 145.70
    assert round(engine.pos.tp, 2) == 154.50


def test_option_exit_reuses_entry_strike():
    calls = []

    def resolver(day, side, minute, spot_px, strike=None):
        calls.append((minute, strike))
        return 100.0 if minute == 556 else 130.0

    engine = _armed_engine()
    engine.resolve_option = resolver
    engine.feed(bar(1, 180, 182, 148, 149, minute=556))
    entry_strike = engine.pos.strike
    engine.feed(bar(2, 200, 210, 199, 209, minute=557))

    assert engine.pos is None
    assert calls == [(556, entry_strike), (557, entry_strike)]


def test_engine_can_use_same_timeframe_ob_stop():
    engine = _armed_engine(risk_mode="ob_same_tf")
    engine._same_tf_stop_ob = OrderBlock(5, 137, 153, 1)
    engine.feed(bar(1, 180, 182, 148, 149, minute=556))

    assert engine.pos is not None
    assert engine.pos.sl == 137


def test_three_minute_bear_setup_enters_on_breakout_of_w5_block():
    engine = _armed_engine("bear")
    engine.candidates[0].entry_mode = "breakout"
    engine.candidates[0].ob = OrderBlock(3, 100, 110, 1)
    engine.candidates[0].untouched_bot = 100

    engine.feed(bar(1, 105, 109, 100.4, 103, minute=556))

    assert engine.pos is not None
    assert engine.pos.side == "PE"
    assert engine.pos.sl == 110


def test_engine_sl_first_on_same_bar():
    engine = _armed_engine()
    engine.feed(bar(1, 184, 152, 148, 149, minute=556))
    engine.feed(bar(2, 200, 300, 100, 200, minute=557))

    assert engine.pos is None
    assert engine.trades[0]["exit_reason"] == "SL"


def test_bear_setup_has_only_dynamic_block_and_one_fallback():
    engine = _armed_engine("bear")
    fallback = engine.candidates[0]
    fallback.ob = OrderBlock(1, 105, 115, 1)
    fallback.untouched_bot = 105
    engine._active_impulse = impulse("bear")

    formation = bar(12, 100, 110, 100, 105, minute=567)
    engine.feed(formation)
    assert len(engine.candidates) == 2
    dynamic = engine.candidates[0]
    assert dynamic.ob.formed_gi == 12
    assert dynamic.eligible_from_gi == 13

    engine.feed(bar(13, 110, 111, 104, 110, minute=568))
    assert dynamic.dead is True
    assert engine.candidates[1] is fallback


def test_no_option_bar_does_not_open_position():
    engine = _armed_engine()
    engine.resolve_option = lambda *args: None
    engine.feed(bar(1, 184, 152, 148, 149, minute=556))

    assert engine.pos is None
    assert engine.trades == []


def test_eod_flattens_position():
    engine = _armed_engine()
    engine.resolve_option = _fake_resolver({930: 55.0})
    engine.feed(bar(1, 184, 152, 148, 149, minute=556))
    assert engine.pos is not None
    engine.close_day()

    assert engine.pos is None
    assert engine.trades[0]["exit_reason"] == "EOD"
