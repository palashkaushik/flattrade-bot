from artifacts.f6_hybrid.causal_live_parity_research import ordered_stochastics


def test_ordered_stochastics_requires_strict_descending_values():
    assert ordered_stochastics((80.0, 60.0, 40.0, 20.0)) is True
    assert ordered_stochastics((80.0, 80.0, 40.0, 20.0)) is False
    assert ordered_stochastics((60.0, 80.0, 40.0, 20.0)) is False


def test_ordered_stochastics_rejects_unwarmed_values():
    assert ordered_stochastics((80.0, None, 40.0, 20.0)) is False
