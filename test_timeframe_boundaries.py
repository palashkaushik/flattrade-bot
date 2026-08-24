from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.strategies.quad_pinbar_divergence import MTFTracker


def test_live_timeframe_buffers_do_not_mix_sessions():
    tracker = MTFTracker()

    tracker.push_1m(Candle(100.0, 101.0, 99.0, 100.5, minute=929))
    tracker.push_1m(Candle(100.5, 102.0, 100.0, 101.5, minute=555))

    assert all(
        candle.minute != 929
        for buffer in tracker.bufs.values()
        for candle in buffer
    )


def test_session_rollover_clears_pending_setup_state():
    tracker = MTFTracker()
    tracker._last_minute = 929

    for tf in tracker.trackers:
        tf_tracker = tracker.trackers[tf]
        tf_tracker.hist.append(Candle(100.0, 101.0, 99.0, 100.5, minute=929))
        tf_tracker.setup_active = True
        tf_tracker.stype = "super"
        tf_tracker.flag_ready = True
        tf_tracker.super_ready = True
        tf_tracker.has_bull_divergence = True
        tf_tracker._armed_bullish_divergence = (1, 2)
        tf_tracker.s4_embedded_count = 26
        tracker.f6scans[tf]._fired = True

    tracker._reset_timeframe_buffers_if_session_rolled(555)

    assert all(not tf_tracker.hist for tf_tracker in tracker.trackers.values())
    assert all(not tf_tracker.setup_active for tf_tracker in tracker.trackers.values())
    assert all(tf_tracker.stype == "" for tf_tracker in tracker.trackers.values())
    assert all(not tf_tracker.flag_ready for tf_tracker in tracker.trackers.values())
    assert all(not tf_tracker.super_ready for tf_tracker in tracker.trackers.values())
    assert all(not tf_tracker.has_bull_divergence for tf_tracker in tracker.trackers.values())
    assert all(tf_tracker._armed_bullish_divergence is None for tf_tracker in tracker.trackers.values())
    assert all(tf_tracker.s4_embedded_count == 0 for tf_tracker in tracker.trackers.values())
    assert all(not scanner._fired for scanner in tracker.f6scans.values())
