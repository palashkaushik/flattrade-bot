from artifacts.f6_hybrid.causal_live_parity_research import (
    resolve_daily_loss_limit,
    resolve_sl_points,
)


def test_fixed_sl_override_keeps_sl_independent_of_atr():
    assert resolve_sl_points(9.0, 3.0, 10.0, fixed_sl_points=10.0) == 10.0


def test_no_daily_caps_disables_the_default_daily_loss_limit():
    assert resolve_daily_loss_limit(None, no_daily_caps=True) == float("-inf")
