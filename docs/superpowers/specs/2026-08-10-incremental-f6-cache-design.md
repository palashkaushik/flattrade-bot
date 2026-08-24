# Incremental F6 Signal Cache Design

## Goal

Reduce repeated F6 candidate work without changing the reference engine's trade
semantics. The first implementation must remain parity-safe; pointer and JIT
optimizations are allowed only after the split is proven equivalent.

## Boundaries

`grid_optimize_f6_atr.process_day()` currently performs two different jobs:

1. It parses/caches option data, warms trackers, builds MTF/F6 signal events, and
   retains the tracker state used by bearish-reversal exits.
2. It executes the stateful position loop with ATR SL/TP, daily shutdown,
   consecutive-loss shutdown, reversal entries, and EOD exits.

The new path will expose those jobs as an internal `SignalState` plus an exact
execution function. The reference `process_day()` remains unchanged until parity
tests pass. A signal cache key includes the day, previous-day file, all signal
parameters, fixed F6 configuration, and an implementation version. Execution
parameters are `atr_sl_mult`, `atr_tp_mult`, and `consec_loss`.

## Reuse Model

The current grid factors into:

- 36 raw indicator-period combinations: `s1_k × s4_k × atr_period`.
- 9 threshold combinations: `f6_s4_thresh × f6_s1_thresh`.
- 48 execution combinations: `atr_sl_mult × atr_tp_mult × consec_loss`.

Raw OHLC arrays and canonical minute/slot mappings are reused first. Exact signal
events and the tracker state are then built once per signal key and reused for
the 48 execution variants. Execution state is never reused between candidates:
SL/TP and shutdown rules are path-dependent.

## Pointer Path

The optimized executor will use sorted integer arrays and monotonic cursors:

- `minute -> spot index` for ATM lookup.
- `(strike, side) -> slot id` for option lookup.
- `slot -> current bar index` for advancing OHLC slices.
- `minute -> signal range` for consuming already-sorted events.

This removes repeated regex/string mapping and avoids `searchsorted()` in the
monotonic per-minute loop. Missing bars retain the reference behavior and are
covered by tests.

## Correctness Gates

- Unit tests prove cache keys distinguish every signal-changing parameter.
- A cache reuse test proves two execution variants build signals once.
- Synthetic fixtures cover empty data, missing bars, same-bar SL/TP precedence,
  reversal side selection, daily shutdown, consecutive-loss shutdown, and EOD.
- Five-day champion parity compares trade count, ordered trade fields, and
  summary metrics against the unchanged reference engine.
- No full search runs until the five-day parity and timing benchmark pass.

## Scope Exclusions

The first slice does not change indicator formulas, divergence logic, option
selection, fees, precision, or GPU execution. Numba batching and disk-backed
feature caches are follow-up optimizations after CPU parity.
