from datetime import date
from pathlib import Path

from artifacts.flattrade_day_cache import load_day_cache
from artifacts.f6_hybrid import marni_fib_flattrade_cache as cache_runner
from artifacts.f6_hybrid.marni_fib_flattrade_cache import parse_row
from artifacts.f6_hybrid.marni_fib_backtest import (
    BiasState,
    FibPattern,
    SymbolFibFeed,
    bias_confirmed_for_event,
    combined_bias_allows,
    confirm_event,
    row_to_candle,
    simulate,
)
from flattrade_bot.indicators.patterns import Candle


def candle():
    return Candle(100.0, 105.0, 95.0, 102.0)


def test_red_green_red_requires_five_green_ut_candles():
    pattern = FibPattern("bullish", "red", "green", "red", "high_to_low")
    pattern.update(candle(), "red")
    for _ in range(4):
        pattern.update(candle(), "green")

    assert pattern.update(candle(), "red") is None


def test_green_red_green_requires_five_red_ut_candles():
    pattern = FibPattern("bearish", "green", "red", "green", "low_to_high")
    pattern.update(candle(), "green")
    for _ in range(5):
        pattern.update(candle(), "red")

    completed = pattern.update(candle(), "green")

    assert completed[0] == "bearish"


def test_five_minute_bias_is_confirmed_only_after_fifth_minute():
    state = BiasState(5)

    for minute in range(560, 564):
        state.update_1m(row_to_candle({
            "minute": minute,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
        }))

    assert state.snapshot()["confirmed_minute"] is None

    state.update_1m(row_to_candle({
        "minute": 564,
        "open": 100.5,
        "high": 102.0,
        "low": 100.0,
        "close": 101.5,
    }))

    assert state.snapshot()["confirmed_minute"] == 564


def test_event_waits_for_higher_timeframe_confirmation():
    event = {"minute": 763}

    assert not bias_confirmed_for_event(event, {"confirmed_minute": 759})
    assert bias_confirmed_for_event(event, {"confirmed_minute": 764})


def test_bullish_bias_requires_green_heikin_ashi_body():
    state = BiasState(5)
    state.ha_candle = row_to_candle({
        "minute": 564,
        "open": 102.0,
        "high": 103.0,
        "low": 98.0,
        "close": 100.0,
    })
    state.linreg_signal = 99.0
    state.ut_color = "green"

    assert state.snapshot()["bullish"] is False


def test_confirmed_event_uses_confirmation_minute_for_entry():
    event = {"minute": 763, "entry": 222.5}

    assert confirm_event(event, {"confirmed_minute": 763}, 763)["minute"] == 763
    assert confirm_event(event, {"confirmed_minute": 759}, 763) is None

    confirmed = confirm_event(event, {"confirmed_minute": 764}, 764)

    assert confirmed["signal_minute"] == 763
    assert confirmed["minute"] == 764


def test_cached_ce_signal_waits_for_five_minute_bias_confirmation():
    cache = load_day_cache(Path("artifacts/flattrade_day_cache"), date(2026, 8, 12))
    rows = [parse_row(row) for row in cache["contracts"]["CE:24200"]["rows"]]
    previous = [row for row in rows if not row["time"].startswith("12-08-2026")]
    current = [row for row in rows if row["time"].startswith("12-08-2026")]
    feed = SymbolFibFeed("option")
    feed.warmup(previous)

    events = [
        event
        for row in current
        for event in feed.push(row)
        if event.get("signal_minute") == 763
    ]

    assert len(events) == 1
    assert events[0]["minute"] == 764
    assert events[0]["bias"]["confirmed_minute"] == 764


def test_warmup_does_not_confirm_new_session_with_previous_day_minute():
    cache = load_day_cache(Path("artifacts/flattrade_day_cache"), date(2026, 8, 12))
    rows = [parse_row(row) for row in cache["contracts"]["CE:24200"]["rows"]]
    previous = [row for row in rows if not row["time"].startswith("12-08-2026")]
    feed = SymbolFibFeed("option")

    feed.warmup(previous)

    assert feed.bias.snapshot("1m")["confirmed_minute"] is None


def test_bias_ut_uses_regular_close_when_heikin_ashi_source_is_disabled():
    class RecordingUT:
        def __init__(self):
            self.source_close = None

        def update(self, candle, source_close=None):
            self.source_close = candle.close if source_close is None else source_close
            return "green"

    state = BiasState(5)
    recorder = RecordingUT()
    state.ut = recorder

    for minute in range(560, 565):
        state.update_1m(row_to_candle({
            "minute": minute,
            "open": 100.0 + minute - 560,
            "high": 105.0 + minute - 560,
            "low": 99.0 + minute - 560,
            "close": 104.0 + minute - 560,
        }))

    assert recorder.source_close == 108.0
    assert recorder.source_close != state.ha_candle.close


def test_index_signal_also_requires_selected_option_side_bias():
    index_bearish = {"bullish": False, "bearish": True}
    option_green = {"bullish": True, "bearish": False}
    option_red = {"bullish": False, "bearish": True}

    assert combined_bias_allows(index_bearish, option_green, "PE") is False
    assert combined_bias_allows(index_bearish, option_red, "PE") is True


def test_index_mode_rejects_pe_trade_when_selected_option_bias_is_bullish():
    cache_runner.GLOBAL_CACHE_DIR = Path("artifacts/flattrade_day_cache")
    results = cache_runner.run_day(
        "2026-08-12",
        ("1m",),
        (0.0,),
        (1.079,),
        "index",
    )

    trades = results["index|1m|tp0.0|sl1.079"]

    assert not any(
        trade["entry_min"] == 853
        and trade["symbol"] == "NIFTY18AUG26P24400"
        for trade in trades
    )


def test_index_feed_keeps_the_0956_bearish_setup_until_the_1034_touch():
    cache = load_day_cache(Path("artifacts/flattrade_day_cache"), date(2026, 8, 12))
    spot = cache_runner.normalize_spot(cache["spot_rows"])
    feed = SymbolFibFeed("index")
    events = []

    for index in range(len(spot["min"])):
        events.extend(feed.push(cache_runner.spot_row(spot, index)))

    assert any(
        event["minute"] == 633
        and event["direction"] == "bearish"
        and event["timeframe"] == "1m"
        for event in events
    )


def test_warmup_keeps_1033_touch_aligned_and_initializes_bias():
    cache = load_day_cache(Path("artifacts/flattrade_day_cache"), date(2026, 8, 12))
    previous = load_day_cache(Path("artifacts/flattrade_day_cache"), date(2026, 8, 11))
    spot = cache_runner.normalize_spot(cache["spot_rows"])
    feed = SymbolFibFeed("index")
    feed.warmup(cache_runner.normalize_spot_rows(previous["spot_rows"]))
    events = []

    for index in range(len(spot["min"])):
        events.extend(feed.push(cache_runner.spot_row(spot, index)))

    assert any(
        event["minute"] == 633
        and event["direction"] == "bearish"
        and event["timeframe"] == "1m"
        for event in events
    )
    assert feed.bias.snapshot("1m")["linreg_signal"] is not None


def test_short_pe_trade_profits_when_price_falls():
    event = {
        "side": "PE",
        "strike": 24400,
        "symbol": "NIFTY18AUG26P24400",
        "minute": 600,
        "option_entry": 151.85,
        "fib_source": "option",
        "orientation": "low_to_high",
        "direction": "bearish",
        "fib_high": 155.0,
        "fib_low": 145.0,
        "timeframe": "1m",
    }
    bars = {
        ("PE", 24400): {
            601: {"minute": 601, "open": 150.0, "high": 152.0, "low": 149.0, "close": 150.0},
            602: {"minute": 602, "open": 150.0, "high": 151.0, "low": 144.0, "close": 146.1},
        }
    }
    index_bars = {
        600: {"minute": 600, "open": 24300.0, "high": 24310.0, "low": 24290.0, "close": 24305.0},
        601: {"minute": 601, "open": 24305.0, "high": 24312.0, "low": 24295.0, "close": 24300.0},
        602: {"minute": 602, "open": 24300.0, "high": 24305.0, "low": 24280.0, "close": 24285.0},
    }
    spot = {
        "min": [600, 601, 602],
        "open": [24300.0, 24305.0, 24300.0],
        "high": [24310.0, 24312.0, 24305.0],
        "low": [24290.0, 24295.0, 24280.0],
        "close": [24305.0, 24300.0, 24285.0],
    }
    trades = simulate(events=[event], bars=bars, index_bars=index_bars, spot=spot,
                      timeframe_mode="1m", target_level=0.0, stop_level=1.079)
    assert len(trades) == 1
    assert trades[0]["reason"] == "TP"
    assert trades[0]["points"] > 0
    assert trades[0]["rs_net"] > 0


def test_short_pe_trade_loses_when_price_rises_to_stop():
    event = {
        "side": "PE",
        "strike": 24400,
        "symbol": "NIFTY18AUG26P24400",
        "minute": 600,
        "option_entry": 151.85,
        "fib_source": "option",
        "orientation": "low_to_high",
        "direction": "bearish",
        "fib_high": 155.0,
        "fib_low": 145.0,
        "timeframe": "1m",
    }
    bars = {
        ("PE", 24400): {
            601: {"minute": 601, "open": 152.0, "high": 156.5, "low": 151.0, "close": 154.0},
        }
    }
    index_bars = {
        600: {"minute": 600, "open": 24300.0, "high": 24310.0, "low": 24290.0, "close": 24305.0},
        601: {"minute": 601, "open": 24305.0, "high": 24325.0, "low": 24300.0, "close": 24315.0},
    }
    spot = {
        "min": [600, 601],
        "open": [24300.0, 24305.0],
        "high": [24310.0, 24320.0],
        "low": [24290.0, 24300.0],
        "close": [24305.0, 24315.0],
    }
    trades = simulate(events=[event], bars=bars, index_bars=index_bars, spot=spot,
                      timeframe_mode="1m", target_level=0.0, stop_level=1.079)
    assert len(trades) == 1
    assert trades[0]["reason"] == "SL"
    assert trades[0]["points"] < 0
    assert trades[0]["rs_net"] < 0
