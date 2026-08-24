# OI Spurt Filter

`flattrade_bot.indicators.oi` classifies the transition from one candle to the
next using price and open interest:

| OI | Price | Regime | Bias |
|---|---|---|---|
| Rising | Rising | `LONG_BUILDUP` | Bullish |
| Rising | Falling | `SHORT_BUILDUP` | Bearish |
| Falling | Rising | `SHORT_COVERING` | Bullish reversal |
| Falling | Falling | `LONG_UNWINDING` | Bearish reversal |

Small changes return `NEUTRAL`. Percentage thresholds are configurable because
OI scales differ between contracts. Large simultaneous OI and price expansion
sets `caution=True`; the option-side filter blocks those regimes by default.

```python
from flattrade_bot.indicators.oi import allows_option_side, classify_oi_spurt

state = classify_oi_spurt(
    previous_oi=1_000_000,
    current_oi=1_080_000,
    previous_price=100.0,
    current_price=103.0,
    min_oi_change_pct=0.03,
    min_price_change_pct=0.01,
)

if allows_option_side(state, "CE"):
    print(state.regime)
```

The local option archive contains OI, and `test_trending_oi_filter.py` already
contains an aggregate ATM +/- 4 strike filter. The live TPSeries adapter
currently drops OI from its normalized candle rows, so this classifier is not
yet enabled for live orders. Live integration requires preserving OI and
fetching a consistent option-chain window at each candle.
