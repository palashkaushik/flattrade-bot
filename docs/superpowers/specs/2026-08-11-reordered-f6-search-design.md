# Reordered F6 Search Design

## Goal

Reduce wasted full-fidelity evaluations during the F6 eight-axis search without
changing signal or execution semantics. The existing reference engine remains
the source of truth; this work changes only candidate scheduling and fidelity.

## Constraints

- Keep the existing eight-axis `SEARCH_SPACE` unchanged.
- Keep `s1_d=3`, fees, costs, and walk-forward gates outside this raw search.
- Use the factorized runner for candidate evaluation so shared feature state is
  reused within a batch.
- Preserve exact trade output and summary parity with the existing engine.
- Use the fixed eight-worker pool and never launch a long search before the
  five-day smoke test passes.
- Do not enable Numba or approximate event/fill logic.

## Design

Add a standalone artifact harness at
`artifacts/f6_hybrid/reordered_search.py`. It imports the existing search space,
statistics, composite score, day-file loading, and
`run_factorized_candidates`; it does not modify `grid_optimize_f6_atr.py`.

The harness uses Optuna's ask/tell API in batches:

1. Ask a batch of candidate trials from a multivariate TPE sampler.
2. Evaluate all active candidates on the same staged day block with
   `run_factorized_candidates`, allowing duplicate signal bases to be reused.
3. Report each candidate's composite score to a Hyperband pruner at the stage's
   day count. Pruned candidates are told to Optuna as `PRUNED` and are not
   evaluated at later stages.
4. Continue survivors through the next day block. Tell completed candidates
   their full-fidelity score and write one durable CSV row per candidate.

The default raw-search stages are 5, 20, 60, and all days in the 2020-2022
search window. The five-day stage is intentionally small and noisy; it is used
as a screening resource, not as final evidence. Hyperband's resource is the
number of evaluated days and its maximum is the full search window.

## Smoke Contract

`--smoke` runs only a champion and one alternate candidate on the first five
available days. It must verify:

- the champion's factorized trade list matches `grid.run_days` exactly;
- champion totals stay within the established 15-40 trades / five-day gate;
- every requested candidate returns a result and no stage silently drops a
  trial; and
- the staged runner can complete a batch with the fixed worker count.

The smoke command exits non-zero on any mismatch. Only after it passes may the
full reordered search be run.

## Outputs

- `artifacts/f6_hybrid/reordered_search.py`: standalone search harness.
- `artifacts/f6_hybrid/reordered_search_<run_id>.csv`: durable trial/stage
  ledger.
- `artifacts/f6_hybrid/reordered_search_<run_id>.json`: run configuration,
  stage counts, pruning counts, timings, best full-fidelity candidates, and
  parity result.
- `artifacts/f6_hybrid/optimization_methods.md`: measured method and decision.

The CSV records trial number, stage/resource, state, score, raw statistics,
parameters, elapsed time, and factorized base/signal build counts. JSON records
the exact day lists and command configuration so a result can be reproduced.

## Error Handling

- A failed candidate evaluation fails the run rather than being represented as
  a poor score.
- A candidate is told exactly once to Optuna, either `PRUNED` or with its final
  score.
- Existing result files are never overwritten implicitly; the harness requires
  an explicit output path or appends with a run identifier.
- A parity mismatch stops the smoke test and prevents full-search execution.

## Validation

1. Compile the harness and its imported modules.
2. Run `python artifacts\\f6_hybrid\\reordered_search.py --smoke`.
3. Inspect the smoke JSON and CSV for exact champion parity and complete stage
   accounting.
4. Run the validated search with a bounded trial count first, then compare its
   best candidates against the existing Phase 1 results and retain the
   walk-forward validation as the acceptance gate.
