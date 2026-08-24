# F6 Backtest Performance Research

Date: 2026-08-12

## Executive Summary

The current 2020-2026 cost-aware Optuna run is too slow because the full
fidelity stage dominates the work, not because Optuna sampling itself is slow.
The recorded fixed-threshold raw search spent `3,618.7s` of `4,059.1s` (89.1%)
on only 10 full-fidelity candidates across 1,574 days. The next optimization
should remove repeated full-history feature construction and repeated Windows
pool startup before attempting another large trial count.

With F6 fixed at `79.5/20.5`, the declared grid is only:

```text
4 s1 values x 3 s4 values x 3 ATR periods
  x 4 SL values x 4 TP values x 3 consecutive-loss values
= 1,728 execution combinations
```

The strongest next architecture is a grouped exhaustive/factorized engine:
build the 36 signal bases once per day, then evaluate the 48 execution variants
for each base. Persist normalized features in a disk-backed cache and keep one
8-worker pool alive for the entire run.

## Current Evidence

| Observation | Evidence | Implication |
|:---|:---|:---|
| Factorized execution is already valuable | 48 candidates x 5 days: `37.604s -> 8.290s`, `4.536x`, exact summaries | Keep signal/execution separation |
| Incremental cache helps | Reference-to-incremental `3.059x` on the same benchmark | Keep pointer/cache path |
| Numba serial did not materially help the small test | `1.088x` serial, exact parity | Do not enable the current serial packed path blindly |
| Numba nested parallelism failed | `0.058x` with `prange` inside the 8-process pool | Avoid process + native-thread oversubscription |
| Full current search is dominated by finalists | 10 full candidates: `3,618.7s`; early stages: `440.2s` | Optimize full candidate execution first |
| Current search has low information efficiency | 40 trials, 30 pruned, 10 completed; fixed thresholds | Use a smaller/cleaner resource ladder or exhaustive grouped execution |
| 2026 archive had duplicate rows | 193 symbol/day groups contained duplicate minutes; some were 2x copies | Normalize once before any worker sees data |
| Windows uses spawn | Python documentation: spawn starts a fresh interpreter and is slower than fork | Avoid repeatedly creating pools and re-pickling data |

## Findings

### 1. Remove repeated staged-prefix work

`reordered_search.py` evaluates cumulative prefixes `5, 20, 60, 1574` by
restarting `run_factorized_candidates` for each stage. A survivor is therefore
recomputed from day one multiple times. This is the largest direct algorithmic
waste.

Recommended implementation:

- Add a study-wide incremental evaluator that keeps per-candidate state across
  day blocks.
- Process only the new block after a candidate survives a stage.
- Carry previous-day warmup exactly once at each fold/window boundary.
- Keep the reference prefix evaluator as a parity oracle.
- Compare cumulative incremental results against a fresh full-prefix run on a
  five-day and 20-day fixture before enabling it.

Expected benefit: high. It removes repeated work proportional to the number of
promotion stages, while changing no strategy semantics.

### 2. Keep one worker pool alive for the study

`run_factorized_candidates` creates a new Windows multiprocessing pool for each
candidate batch and each stage. Python's documentation notes that Windows uses
`spawn`, which starts a fresh interpreter and has higher startup/serialization
cost than fork-based execution.

Recommended implementation:

- Create one pool at the search-run boundary with exactly 8 workers.
- Send batch tasks through that pool instead of creating a pool inside every
  evaluator call.
- Move candidate context into task arguments or a controlled worker update;
  never rely on stale global candidate state.
- Report pool creations, worker initialization time, and task serialization
  time in the benchmark.

Expected benefit: medium to high for staged searches and walk-forward folds.

### 3. Normalize and persist day data once

The 2026 archive exposed duplicated `(symbol, minute)` rows. The cache now
deduplicates and sorts them, but it does so when each worker first reads a day.

Recommended implementation:

- Add a one-time 2020-2026 normalization command.
- Deduplicate `(symbol, minute)` deterministically and record removed-row
  counts.
- Write sorted, typed arrays to Parquet/Arrow or compact `.npz` shards.
- Store an input-data manifest with source hash, row count, duplicate count, and
  date coverage.
- Make the backtest refuse stale or incomplete normalized caches.

Apache Arrow documents Parquet as a typed columnar format suitable for selective
column reads. The repository's earlier Parquet test was only a marginal win over
CSV on a tiny five-day workload, so this must be rebenchmarked on the full
1,574-day archive rather than assumed to help.

Expected benefit: medium for I/O and startup; high for reproducibility/data
integrity.

### 4. Precompute the 36 signal bases

With F6 thresholds fixed at `79.5/20.5`, only `(s1_k, s4_k, atr_period)` affect
raw indicator features. There are 36 bases. SL, TP, and consecutive-loss values
affect execution only.

Recommended implementation:

- Build each base once per day using typed contiguous arrays.
- Persist or memoize the base features using a key containing data hash, day,
  previous day, s1 period, s4 period, ATR period, and pivot-divergence version.
- Materialize the fixed F6 threshold state once per base.
- Execute the 48 `(SL, TP, consecutive-loss)` combinations against the same
  signal state.
- Use a grouped exhaustive sweep instead of asking TPE to rediscover a small
  structured grid.

Expected benefit: high. This is the direct generalization of the measured
`4.536x` factorized benchmark to the full fixed-threshold grid.

### 5. Change the Optuna resource ladder

Optuna's documentation recommends Hyperband with TPE for pruning, but notes that
each Hyperband bracket needs enough trials for TPE to adapt. Hyperband chooses
the number of brackets from `min_resource`, `max_resource`, and
`reduction_factor`.

The current `min=5`, `max=1574`, `factor=3` schedule creates roughly six
brackets. Forty trials cannot provide ten startup trials per bracket, so the
TPE model receives weak evidence while still paying for repeated stages.

Recommended alternatives to benchmark:

- `resources=[20, 60, 200, 1574]`, reducing noisy five-day promotion decisions.
- `reduction_factor=4`, reducing bracket count.
- Use at least 60 trials if Hyperband/TPE adaptation is the objective.
- Prefer grouped exhaustive execution when all six remaining axes are fixed and
  finite; use Optuna for exploratory extensions, not for rediscovering 1,728
  known combinations.
- Persist the study with Optuna JournalStorage or SQLite/RDB so an interruption
  resumes instead of losing all in-memory trials.

Expected benefit: medium for pruning; high for resilience and avoiding wasted
runs; not a substitute for faster full evaluation.

### 6. Revisit Numba only as a single batch kernel

Numba's official guidance says nopython loops can be fast and `prange` requires
independent iterations with no unsafe shared writes. The current benchmark used
`prange` inside an 8-process pool over small candidate batches, causing severe
oversubscription.

If profiling shows execution is still dominant after feature caching:

- Pack all candidates in a sufficiently large batch into contiguous arrays.
- Run one native Numba worker with up to 8 Numba threads, not 8 Python workers
  each launching Numba threads.
- Use `NUMBA_NUM_THREADS`/`set_num_threads` explicitly and record it.
- Keep signal generation and branch-heavy state transitions in a parity-tested
  kernel.
- Benchmark serial njit, one-process parallel njit, and the Python factorized
  path on the full workload.

Expected benefit: uncertain but worth a second benchmark only after the data
and feature layers are fixed. The existing `0.058x` nested-prange result must
not be repeated.

### 7. Event-schedule idle execution

The execution loop scans every minute from session start to EOD for every
candidate. When there is no open position, it can jump to the next signal minute
while preserving all state transitions that can occur without a position.

Recommended implementation:

- Convert `pmtrig` keys into sorted event-minute arrays.
- When flat, jump to the next signal minute or session end.
- When holding a position, retain per-minute bars for SL/TP, divergence exit,
  daily shutdown, and EOD semantics.
- Prove same-minute ordering and no missed signal/state transitions with a
  trade-for-trade fixture.

Expected benefit: medium to high for sparse signals; lower if most minutes have
open positions or triggers.

### 8. Compact the Python state boundary

The current feature path creates `Candle` objects, nested dictionaries, tuples,
and deep-copied tracker state per day/candidate. This is convenient for parity
but expensive across 1,574 days.

Recommended implementation after parity fixtures exist:

- Use integer symbol slots and structure-of-arrays OHLC/minute buffers.
- Store trigger streams in compact NumPy arrays rather than nested tuples.
- Replace generic `deepcopy` with a small immutable numeric tracker snapshot.
- Keep a Python adapter only at the reference boundary.

Expected benefit: medium to high; implementation risk is high, so it follows
the cache/pool work.

### 9. Control nested native parallelism

The scikit-learn parallelism guidance highlights oversubscription: outer
processes multiplied by inner BLAS/OpenMP/native threads can create far more
workers than the machine can run. The same problem caused the current Numba
parallel result to collapse.

Recommended runtime policy:

- Hard ceiling: 8 total worker processes/threads for this project.
- If Python multiprocessing uses 8 workers, set Numba/BLAS/OpenMP threads to 1.
- If a native kernel uses 8 threads, use one Python worker.
- Record both process count and native thread count in every benchmark.

## Priority Plan

| Priority | Phase | Change | Gate |
|:---:|:---|:---|:---|
| P0 | Data | Normalize/deduplicate full archive once | manifest + duplicate test |
| P1 | Evaluation | Persistent pool and incremental stage continuation | exact 5/20/full parity |
| P2 | Features | Disk-backed 36-base cache and grouped 48-execution sweep | base-build count + runtime |
| P3 | Runtime | Event scheduling and compact trigger arrays | trade-for-trade parity |
| P4 | Search | Cost-aware Optuna with persisted Journal/SQLite study | resumability + net objective |
| P5 | Native | Single-batch Numba or compiled execution kernel | full-workload parity + benchmark |

Do not proceed to P3-P5 until P0-P2 are measured. The current evidence says
P1/P2 are more promising than another raw Optuna trial increase.

## Evidence Quality

- **High:** repository benchmarks, current 2020-2026 search timings, duplicate
  row scan, exact parity tests, and official Python/Optuna/Numba documentation.
- **Medium:** expected gains for persistent pools, event scheduling, and compact
  arrays; these require repository-specific benchmarks.
- **Low until measured:** GPU acceleration, Cython/Rust rewrites, and larger
  worker counts. They may help, but current evidence does not justify them.

## Sources

- Python multiprocessing and Windows spawn behavior:
  https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
- Python shared memory:
  https://docs.python.org/3/library/multiprocessing.shared_memory.html
- Numba performance tips:
  https://numba.readthedocs.io/en/stable/user/performance-tips.html
- Numba parallel loops and scheduling:
  https://numba.readthedocs.io/en/stable/user/parallel.html
- Optuna efficient samplers/pruners:
  https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
- Optuna multiprocessing/storage:
  https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
- Optuna Hyperband details:
  https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html
- Apache Arrow Parquet:
  https://arrow.apache.org/docs/python/parquet.html
- Parallelism and oversubscription guidance:
  https://scikit-learn.org/stable/computing/parallelism.html
- GitHub code-search examples for shared-memory backtests and Numba kernels:
  https://github.com/search?q=multiprocessing.shared_memory+backtest&type=code
  https://github.com/search?q=prange+backtest&type=code
