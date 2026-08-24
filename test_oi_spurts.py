import pytest

from flattrade_bot.indicators.oi import (
    LONG_BUILDUP,
    LONG_UNWINDING,
    SHORT_BUILDUP,
    SHORT_COVERING,
    NEUTRAL,
    allows_option_side,
    classify_oi_spurt,
)


@pytest.mark.parametrize(
    ("current_oi", "previous_price", "current_price", "expected"),
    [
        (110.0, 100.0, 105.0, LONG_BUILDUP),
        (110.0, 100.0, 95.0, SHORT_BUILDUP),
        (90.0, 100.0, 105.0, SHORT_COVERING),
        (90.0, 100.0, 95.0, LONG_UNWINDING),
    ],
)
def test_classifies_the_four_oi_spurt_regimes(current_oi, previous_price, current_price, expected):
    result = classify_oi_spurt(
        previous_oi=100.0,
        current_oi=current_oi,
        previous_price=100.0,
        current_price=current_price,
        min_oi_change_pct=0.05,
        min_price_change_pct=0.01,
    )

    assert result.regime == expected


def test_small_oi_or_price_changes_are_neutral():
    result = classify_oi_spurt(
        previous_oi=100.0,
        current_oi=102.0,
        previous_price=100.0,
        current_price=100.5,
        min_oi_change_pct=0.05,
        min_price_change_pct=0.01,
    )

    assert result.regime == NEUTRAL
    assert result.bias == "neutral"


def test_abnormal_oi_and_price_expansion_sets_caution():
    result = classify_oi_spurt(
        previous_oi=100.0,
        current_oi=130.0,
        previous_price=100.0,
        current_price=110.0,
        min_oi_change_pct=0.05,
        min_price_change_pct=0.01,
        caution_oi_change_pct=0.20,
        caution_price_change_pct=0.05,
    )

    assert result.regime == LONG_BUILDUP
    assert result.bias == "bullish"
    assert result.caution is True


def test_prices_must_be_positive_for_percentage_classification():
    with pytest.raises(ValueError):
        classify_oi_spurt(0.0, 100.0, 100.0, 101.0)


def test_regime_filter_maps_bullish_and_bearish_option_sides():
    bullish = classify_oi_spurt(100.0, 110.0, 100.0, 105.0)
    bearish = classify_oi_spurt(100.0, 110.0, 100.0, 95.0)

    assert allows_option_side(bullish, "CE") is True
    assert allows_option_side(bullish, "PE") is False
    assert allows_option_side(bearish, "PE") is True
    assert allows_option_side(bearish, "CE") is False
