"""Candle-level open-interest and price regime classification."""

from __future__ import annotations

from dataclasses import dataclass


LONG_BUILDUP = "LONG_BUILDUP"
SHORT_BUILDUP = "SHORT_BUILDUP"
SHORT_COVERING = "SHORT_COVERING"
LONG_UNWINDING = "LONG_UNWINDING"
NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class OISpurt:
    """Classification for one candle-to-candle OI/price transition."""

    regime: str
    bias: str
    oi_change: float
    price_change: float
    oi_change_pct: float
    price_change_pct: float
    caution: bool


def classify_oi_spurt(
    previous_oi: float,
    current_oi: float,
    previous_price: float,
    current_price: float,
    *,
    min_oi_change_pct: float = 0.01,
    min_price_change_pct: float = 0.001,
    caution_oi_change_pct: float = 0.20,
    caution_price_change_pct: float = 0.05,
) -> OISpurt:
    """Classifies OI/price direction while filtering insignificant movement.

    The percentages are decimal fractions: ``0.01`` means 1%. OI rising with
    price rising is long buildup; OI rising with price falling is short
    buildup; falling OI with rising price is short covering; falling OI with
    falling price is long unwinding.
    """
    if previous_oi <= 0 or previous_price <= 0 or current_oi < 0 or current_price <= 0:
        raise ValueError("OI must be non-negative with positive previous/current prices")

    oi_change = current_oi - previous_oi
    price_change = current_price - previous_price
    oi_change_pct = oi_change / previous_oi
    price_change_pct = price_change / previous_price
    oi_active = abs(oi_change_pct) >= min_oi_change_pct
    price_active = abs(price_change_pct) >= min_price_change_pct
    caution = (
        abs(oi_change_pct) >= caution_oi_change_pct
        and abs(price_change_pct) >= caution_price_change_pct
    )

    if not oi_active or not price_active:
        regime, bias = NEUTRAL, "neutral"
    elif oi_change > 0 and price_change > 0:
        regime, bias = LONG_BUILDUP, "bullish"
    elif oi_change > 0 and price_change < 0:
        regime, bias = SHORT_BUILDUP, "bearish"
    elif oi_change < 0 and price_change > 0:
        regime, bias = SHORT_COVERING, "bullish_reversal"
    else:
        regime, bias = LONG_UNWINDING, "bearish_reversal"

    return OISpurt(
        regime=regime,
        bias=bias,
        oi_change=round(oi_change, 6),
        price_change=round(price_change, 6),
        oi_change_pct=round(oi_change_pct, 6),
        price_change_pct=round(price_change_pct, 6),
        caution=caution,
    )


def allows_option_side(
    spurt: OISpurt,
    side: str,
    *,
    allow_reversals: bool = True,
    block_caution: bool = True,
) -> bool:
    """Returns whether an OI regime supports a CE or PE entry."""
    if block_caution and spurt.caution:
        return False
    normalized_side = side.upper()
    if normalized_side == "CE":
        return spurt.regime == LONG_BUILDUP or (
            allow_reversals and spurt.regime == SHORT_COVERING
        )
    if normalized_side == "PE":
        return spurt.regime == SHORT_BUILDUP or (
            allow_reversals and spurt.regime == LONG_UNWINDING
        )
    raise ValueError(f"Unsupported option side: {side!r}")
