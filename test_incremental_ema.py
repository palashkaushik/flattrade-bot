from flattrade_bot.indicators.ema import IncrementalEMA


def test_incremental_ema_seeds_then_updates():
    ema = IncrementalEMA(3)

    assert ema.update(1.0) is None
    assert ema.update(2.0) is None
    assert ema.update(3.0) == 2.0
    assert ema.update(4.0) == 3.0
