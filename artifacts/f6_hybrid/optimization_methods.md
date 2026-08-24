# F6 Backtest Optimization Matrix

This records the methods from the supplied Luna/Sonnet research notes and the
measured result in this repository. Accuracy-preserving methods are enabled only
after parity; approximate methods are never used for the final audit.

## Implemented and Measured

| Method | Status | Evidence |
|:---|:---|:---|
| Signal/execution separation | Enabled | 48 execution variants reuse one signal state per day |
| Incremental signal-state cache | Enabled | 48-variant five-day run: 3.258x faster than reference |
| Monotonic minute pointers | Enabled | Option bars and spot values avoid repeated `searchsorted()` |
| Day-batched candidate evaluation | Enabled | One worker builds each signal key once, then executes variants |
| Fixed 8-worker pool | Enabled | Avoids oversubscription and is recorded in benchmark output |
| Five-day trade-for-trade parity | Passing | All 48 benchmark candidate summaries match |
| Parquet/Arrow columnar cache | Opt-in | 3.405x total speedup; only 1.06x over CSV incremental, so not default |

## Next Accuracy-Preserving Optimizations

| Method | Plan |
|:---|:---|
| Raw feature cache | Share the 36 `(s1_k, s4_k, atr_period)` feature bases across the 9 threshold pairs |
| Numba execution batch | Compile the remaining position loop over packed arrays; require exact five-day parity |
| Memmap/shared read-only data | Avoid duplicate raw-array copies across Windows workers after the packed layout is stable |
| Event scheduling | Jump over idle no-position minutes only when no signal or state transition can be missed |
| Walk-forward batching | Reuse cached signals separately inside each IS/OOS fold; never share future data across folds |
| Hyperband/TPE screening | Use partial day blocks for pruning, then full fidelity for finalists |
| PBO/DSR/CPCV filters | Reject weak candidates before expensive full-history and walk-forward runs |

## Walk-Forward Measurement

The incremental refit run used 40 trials per true-OOS window, batch size 8, and
8 workers. It completed both 2023 and 2024 folds. Arbitrary TPE batches mostly
had unique signal keys: a batch built about `8 * days` signal states, so the
cache's 3.2x gain applies strongly to grouped execution variants but not yet to
random signal-parameter exploration. A raw feature cache keyed only by
`(s1_k, s4_k, atr_period)` is the next high-impact optimization for TPE.

The refit OOS stitch was negative after costs (`-Rs 134,485`, PF 0.90), while
the fixed champion was negative in 2023 and positive in 2024. This is evidence
that in-sample optimization is not sufficient and validates keeping walk-forward
as a mandatory gate.

The factorized runner subsequently passed five-day trade parity and improved the
48-variant benchmark from 37.604 seconds to 8.290 seconds (4.536x), with five
base builds and five materialized signal builds. The full two-window refit took
about 5,977 seconds versus about 7,023 seconds for the prior incremental path,
with identical OOS results.

## Numba Packed Execution — Benchmarked and Rejected

A full numba-packed position loop (`f6_hybrid/packed.py`) was implemented with
flat arrays and an exact mirror of the reference exit machine (SL/TP precedence,
daily shutdown, bearish-peak reversal via a ported `DivergenceEngine`, EOD).
Measured on the same 48 candidates x 5 days, 8 workers:

| Variant | Seconds | Speedup vs factorized | Parity (48 candidates) |
|:---|:---:|:---:|:---:|
| Factorized (Python) | 9.181 | 1.0x | reference |
| Packed serial (njit) | 8.442 | 1.088x | exact, trade-for-trade |
| Packed parallel (prange) | 159.6 | 0.058x | exact |

The serial kernel is essentially at parity speed: after the factorized cache
eliminates indicator math, the remaining per-candidate work is a small
branch-heavy loop where Python overhead is a minor share. `prange` over 48 tiny
per-candidate states is badly oversubscribed inside the 8-worker pool and is
17x slower. Verdict: do not enable numba in the engine; keep the factorized
Python path as the default. The packed module and its five-day smoke
(`smoke_numba_packed.py`, parity gate) remain as a benchmark artifact and as a
reference implementation if the workload ever shifts to batch-heavy indicator
math.

## Reordered Batched Search — Measured

The standalone `reordered_search.py` harness was smoke-tested against the
reference engine and then run with 40 TPE candidates in batches of 8 over the
2020-2022 window. It evaluates cumulative prefixes of 5, 20, 60, and 748 days,
using Hyperband to prune before full fidelity while retaining the exact
factorized evaluator.

| Measure | Result |
|:---|:---|
| Command | `python artifacts\\f6_hybrid\\reordered_search.py --trials 40 --batch-size 8 --workers 8` |
| Search window | 748 available days, 2020-2022 |
| Smoke | 25 champion trades, exact trade-for-trade parity, 2/2 candidates complete |
| Full trials | 40 requested; 13 full-fidelity, 27 pruned |
| Wall time | 1,568.4 seconds (26.1 minutes) |
| Stage resources | 5 / 20 / 60 / 748 days |
| Feature builds | 8,760 base / 11,299 signal |
| Output | `reordered_search_20260811T051735Z.json` and matching CSV |

The pruning schedule is retained as an accuracy-preserving **screening
harness**: the smoke gate passed and no approximate execution was introduced.
The top bounded candidate scored `693,922` in-sample, below existing 200-trial
results, so this run is not evidence of a better optimum. Full-fidelity
candidates still require the existing walk-forward and cost gates before any
configuration can be promoted.

## Performance Phase 2026-08-12 — Implemented and Measured

The next performance phase added four correctness-preserving changes:

- Full-archive `(symbol, minute)` normalization at the shared cache boundary;
  the 2026 archive contained 193 duplicate groups.
- `FactorizedCandidatePool`, which keeps one eight-worker Windows pool alive
  across candidate batches.
- Fidelity-stage continuation, which evaluates only newly added day blocks plus
  one prior-day warmup instead of recomputing every prefix.
- Cost-aware Optuna ranking using the configured 1.0-point slippage and fee
  model, with F6 fixed at `79.5/20.5`.

The corrected cost-aware 40-trial run completed in `4,607.8s` (76.8 minutes),
with 15 full-fidelity candidates and 25 pruned candidates. Its best candidate
was:

```text
s1_k=12, s4_k=50, atr_period=14,
atr_sl_mult=2.0, atr_tp_mult=6.0,
f6_s4_thresh=79.5, f6_s1_thresh=20.5, consec_loss=4
```

The independent cost-adjusted 2020-2026 runner exactly matched the factorized
result:

| Metric | Result |
|:---|:---:|
| Trades | 6,226 |
| Win rate | 26.9% |
| Net points | -11,173.06 |
| Net P&L | -Rs 819,964 |
| Profit factor | 0.80 |
| Fees | Rs 93,714.98 |

The cost-aware walk-forward Optuna run used 8 trials per fold and completed in
`3,292.3s`:

| OOS year | Best parameters | Net P&L | PF |
|:---:|:---|---:|:---:|
| 2023 | s1=12, s4=50, ATR14, SL1.5/TP6, CL8 | -Rs 124,593 | 0.81 |
| 2024 | s1=9, s4=50, ATR10, SL2/TP6, CL8 | -Rs 74,425 | 0.90 |
| 2025 | s1=12, s4=75, ATR10, SL1.5/TP6, CL4 | -Rs 158,855 | 0.75 |
| 2026 | s1=12, s4=75, ATR10, SL1.5/TP6, CL4 | -Rs 96,784 | 0.56 |
| **Stitched** | 3,616 trades | **-Rs 454,657** | -- |

Conclusion: the optimization is now cost-aware and parity-aligned, but the
corrected pivot-divergence strategy family remains unprofitable over this
archive. The remaining high-impact performance work is a persistent 36-base
feature cache and grouped exhaustive execution; another raw Optuna trial count
increase is not justified yet.

## Divergence Pivot Correction — Verified on Live Flattrade Data

The original divergence engine selected a single rolling minimum close as the
prior trough. This could reject a valid chart-style divergence when an
intermediate price low did not have a corresponding oscillator low. The engine
now uses causal confirmed price pivots from candle lows, compares eligible
prior pivots within the configured lookback, and arms each divergence pair only
once. Critical live/reference callers pass candle low/high values explicitly.

Verification on Flattrade token `41005` (NIFTY 11AUG26 24350 CE), target
`2026-08-11 14:38`:

| Check | Result |
|:---|:---|
| Price/indicator pair | 14:24 trough to 14:37 lower low with higher S1 |
| CE 1m trigger | `super` at 14:38, entry 111.90 |
| ATR exits | SL 12.51 / TP 25.02 |
| Reference five-day smoke | 20 trades, 45.0% WR, smoke gate passed |
| Reference/factorized parity | 7 focused parity tests passed |

This is a semantic strategy correction, not a performance-only optimization.
All historical performance figures produced before this correction must be
re-benchmarked before being compared with future results.

## Deliberately Not Used Yet

- Approximate fill/event skipping is not final-result safe until same-bar SL/TP
  precedence and daily shutdown parity are proven.
- GPU execution is deferred until the packed CPU/Numba path is benchmarked; the
  RTX 3060 has poor FP64 throughput and the current workload is branch-heavy.
- DuckDB is not installed and Parquet predicate pushdown is not a clear win for
  one small daily file; the cache remains optional.
- Higher-timeframe screening is not used for final F6 signals because it can
  change minute-level entry and exit semantics. Stratified whole-day screening
  is acceptable for early candidate pruning.

## Required Benchmark Contract

Every optimization must report reference time, optimized time, speedup, cache
builds/hits, memory use when available, and exact trade/summary parity. A faster
result without parity is rejected.
