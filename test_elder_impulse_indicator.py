from flattrade_bot.indicators.elder import elder_allows


def test_permissive_elder_allows_neutral_and_directional_colors():
    assert elder_allows("green", "CE", "permissive")
    assert elder_allows("blue", "CE", "permissive")
    assert elder_allows("red", "PE", "permissive")
    assert elder_allows("blue", "PE", "permissive")


def test_permissive_elder_rejects_opposite_impulse():
    assert not elder_allows("red", "CE", "permissive")
    assert not elder_allows("green", "PE", "permissive")
