from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.strategies.quad_pinbar_divergence import MTFTracker
import grid_optimize_f6_atr as grid
from artifacts.f6_hybrid import causal_live_parity_research as causal


def _trigger(_candle):
    return True, False, "super", 100.0, 2.0


def _flag_trigger(_candle):
    return True, False, "flag", 100.0, 2.0


def test_embedded_one_minute_reverses_super_signal_on_two_minute(monkeypatch):
    tracker = MTFTracker()
    tracker.trackers["1m"].s4_embedded_count = 25
    monkeypatch.setattr(tracker.trackers["2m"], "push", _trigger)

    signals = tracker.push_1m(Candle(100.0, 101.0, 99.0, 100.0, minute=560))

    assert any(signal[0] == "2m" and signal[1] is True for signal in signals)


def test_global_embedded_regime_does_not_reverse_flag_signal(monkeypatch):
    tracker = MTFTracker()
    tracker.trackers["1m"].s4_embedded_count = 25
    monkeypatch.setattr(tracker.trackers["3m"], "push", _flag_trigger)

    signals = tracker.push_1m(Candle(100.0, 101.0, 99.0, 100.0, minute=561))

    assert any(signal[0] == "3m" and signal[1] is False for signal in signals)


def test_global_embedded_regime_clears_when_source_s4_recovers(monkeypatch):
    tracker = MTFTracker()
    tracker.trackers["1m"].s4_embedded_count = 0
    monkeypatch.setattr(tracker.trackers["2m"], "push", _trigger)

    signals = tracker.push_1m(Candle(100.0, 101.0, 99.0, 100.0, minute=560))

    assert any(signal[0] == "2m" and signal[1] is False for signal in signals)


def test_grid_tracker_applies_global_reverse_regime(monkeypatch):
    params = {
        "s1_k": 12,
        "s1_d": 3,
        "s4_k": 50,
        "atr_period": 10,
        "atr_sl_mult": 3.0,
        "atr_tp_mult": 6.0,
        "f6_s4_thresh": 79.5,
        "f6_s1_thresh": 20.5,
        "consec_loss": 8,
    }
    tracker = grid.MTFTracker(params)
    tracker.trackers["1m"].s4_emb = 25
    monkeypatch.setattr(tracker.trackers["2m"], "push", _trigger)

    tracker.push_1m(grid.Candle(100.0, 101.0, 99.0, 100.0, minute=559))
    signals = tracker.push_1m(grid.Candle(100.0, 101.0, 99.0, 100.0, minute=560))

    assert any(signal[0] == "2m" and signal[1] is True for signal in signals)


def test_causal_tracker_applies_global_reverse_regime(monkeypatch):
    params = {
        "s1_k": 12,
        "s1_d": 3,
        "s4_k": 50,
        "atr_period": 10,
        "atr_sl_mult": 3.0,
        "atr_tp_mult": 6.0,
        "f6_s4_thresh": 79.5,
        "f6_s1_thresh": 20.5,
        "consec_loss": 8,
    }
    tracker = causal.CausalMTF(
        params,
        "previous_divergence",
        "legacy_high_break",
        "pinbar",
        True,
    )
    tracker.trackers["1m"].s4_embedded = 25
    monkeypatch.setattr(tracker.trackers["2m"], "push", _trigger)

    signals = tracker.push(causal.Candle(100.0, 101.0, 99.0, 100.0, minute=560))

    assert any(signal[0] == "2m" and signal[1] is True for signal in signals)
