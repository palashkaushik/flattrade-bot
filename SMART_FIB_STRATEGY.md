# Smart Fib Strategy

## Purpose

Smart Fib is a two-path NIFTY intraday options strategy driven by 1-minute UT
color ranges, a 5-minute index bias, Heikin-Ashi candles, the Humble LinReg
plot, and S1 stochastic confirmation.

The implementation replays cached Flattrade index and option candles. It does
not use TradingView data directly.

## Common 5m Index Bias

The common bias is calculated only from the NIFTY index 5-minute chart:

1. Aggregate index 1-minute candles into 5-minute candles.
2. Convert every raw 5-minute candle to Heikin-Ashi OHLC.
3. Apply the UT Bot recurrence to the Heikin-Ashi candle using key `1` and ATR
   period `10`.
4. Calculate the Humble LinReg plot from Heikin-Ashi closes using regression
   length `11` and SMA signal length `11`.
5. Use the latest confirmed 5-minute candle for the bias decision.

Bias rules:

- Bullish: HA close is above the LinReg plot and UT color is green. CE entries
  are allowed.
- Bearish: HA close is below the LinReg plot and UT color is red. PE entries
  are allowed.
- No option or index trade is allowed without the matching index bias.

## S1 Stochastic

S1 uses the repository convention `(12, 3)` and only the `%D` output.

- S1 turn up: current `%D` rises after a falling previous slope.
- S1 turn down: current `%D` falls after a rising previous slope.
- The S1 turn and price-zone condition must occur on the same 1-minute candle.
- The active Fib zone is `0.618` through `1.0`.

## Trade Type 1: Index Fib

The Fib range is created from the index 1-minute UT pattern:

- `red -> green -> red`: bullish index range, CE direction, S1 turn up.
- `green -> red -> green`: bearish index range, PE direction, S1 turn down.

Entry sequence:

1. Complete the UT pattern using at least five middle-color candles.
2. Use the full pattern extremes as Fib high and Fib low.
3. Wait for the index close to enter the `0.618-1.0` Fib zone.
4. Require the matching index S1 turn and 5-minute index bias.
5. Buy the matching option side.

The index setup may produce multiple sequential signals. The engine does not
allow concurrent positions, but later signals may enter after an earlier
position exits.

## Trade Type 2: Option Fib

Each CE and PE chart is evaluated independently.

1. Only `red -> green -> red` is valid on an option chart.
2. `green -> red -> green` option setups are ignored.
3. Plot the Fib immediately after the RGR setup completes using full pattern
   extremes.
4. Wait for the option price to enter its own `0.618-1.0` Fib zone.
5. Require an option S1 turn up.
6. Require the matching common index 5-minute bias.
7. Buy the option chart that generated the valid setup.

## Strike Selection

The engine tracks three strike candidates at each signal minute:

- CE: ATM, ATM-50, ATM-100
- PE: ATM, ATM+50, ATM+100

The first available candidate is used by the single-position event replay.

## Exits

Each trade is tested with separate stop configurations:

- Fib stop `1.155`
- Fib stop `1.25`

The primary target is Fib `0.29`.

At the primary target:

- If gross option premium movement is at least `10` points, exit at Fib `0.29`.
- If gross option premium movement is below `10` points, continue toward Fib
  `0.0`.

The premium-point check uses actual option prices. The configured delta value
is `0.5` metadata only and does not replace actual premium movement.

Positions exit at the stop, selected target, or end of session when no target
or stop has been reached.

## Data Quality

The index loader removes an obvious stale terminal quote when several preceding
rows are identical flat quotes and the final flat quote jumps abnormally. This
prevents a bad last row from corrupting an overnight Fib extreme.

## Engine

Primary implementation:

`artifacts/f6_hybrid/marni_fib_core_combo_cache.py`

Exit simulation:

`artifacts/f6_hybrid/marni_fib_backtest.py`
