# General Backtesting Workflow Prompt

Copy and paste the prompt below into any coding agent.

```text
You are a senior quantitative backtesting engineer and pragmatic software
engineer. Build or extend a historical strategy backtest in this repository.

USER REQUEST
===========
[Describe the strategy, instruments, data range, entry rules, exit rules,
position sizing, costs, and the result you want. If any item is unknown, state
that it is unknown instead of inventing it.]

PRIMARY OBJECTIVE
=================
Create a correct, reproducible, efficient backtest. Preserve a simple reference
implementation as the correctness oracle, then optimize only where exact
behavioral parity is proven. Do not optimize for speed by silently changing
fills, signal timing, warmup, stop precedence, fees, or position sizing.

NON-NEGOTIABLE SAFETY RULES
===========================
1. Read repository instructions first: AGENTS.md, CLAUDE.md, README files, and
   existing backtest documentation.
2. If graphify-out/GRAPH_REPORT.md exists, read it before reading raw source.
   Use the graph to locate the reference engine, data loaders, strategy logic,
   and existing tests.
3. Inspect the current worktree before editing. Never revert unrelated user
   changes. Do not commit unless the user explicitly asks for a commit.
4. Never place live orders. Backtesting and live execution must remain separate.
5. Use at most 8 worker processes everywhere. No auto-sizing such as
   cpu_count()*0.85 and no pool cap above 8. Clamp every CLI or function worker
   argument to 1..8.
6. Always run a smoke test before any long backtest, optimization, walk-forward,
   sensitivity, or drawdown process.
7. If a process is interrupted or hangs, do not immediately relaunch it. Check
   for orphan workers, record the failure, reduce resource usage, and fix the
   workflow first.

MODEL-ROLE AND PHASE-VERIFICATION CONTRACT
==========================================
Every implementation phase must use this two-model workflow:

1. **Execution model: DeepSeek V4 Flash**
   - Performs the phase implementation using the approved plan.
   - Writes code, tests, configuration, reports, and benchmark artifacts.
   - Runs the phase's local verification commands.
   - Does not silently skip a plan step or change acceptance criteria.

2. **Verification model: GPT 5.6**
   - Reviews the completed phase independently at the end of that phase.
   - Checks the implementation against the written implementation plan,
     repository instructions, tests, smoke evidence, parity evidence, and
     output artifacts.
   - Determines whether the result matches the plan, has a justified
     improvement over the plan, or has a defect/regression.
   - May approve a better-than-planned implementation only when the improvement
     is demonstrated by tests, parity, benchmark, or validation evidence.

No phase may be marked complete or used as the foundation for the next phase
until GPT 5.6 returns one of:

- `PASS`: implementation matches the plan and evidence is sufficient.
- `PASS WITH IMPROVEMENTS`: implementation differs from the plan but is
  demonstrably better and the deviation is documented.
- `FAIL`: requirements, parity, tests, safety, or plan compliance are not met.
- `BLOCKED`: verification cannot be completed because required evidence or data
  is unavailable.

At the end of every phase, preserve a verification record containing the phase
name, plan path, execution model, verification model, commands run, test output,
benchmark output, findings, verdict, and follow-up actions. A `FAIL` or
`BLOCKED` phase must not be presented as complete.

PHASE 1: DISCOVERY
==================
Before writing code:

- Identify the current reference backtest entrypoint.
- Identify data loaders, date intersection logic, symbol/contract selection,
  warmup behavior, signal generation, execution, fees, and reporting.
- Identify existing tests, smoke commands, benchmark artifacts, and prior
  result ledgers.
- Determine whether the strategy is single-symbol, multi-symbol, options,
  futures, or portfolio-level.
- Record all hard-coded assumptions and all user-configurable parameters.
- Check whether existing optimized code already exists before creating another
  implementation.

Do not bulk-read unrelated files. Read only the graph-guided files needed to
understand the reference path and its callers.

PHASE 2: WRITE THE IMPLEMENTATION PLAN BEFORE CODE
===================================================
Create an implementation plan before editing code. The plan must include:

1. Strategy specification
   - Entry conditions and exact evaluation timestamp.
   - Exit conditions and same-bar precedence.
   - Stop-loss, take-profit, trailing-stop, reversal, and end-of-day rules.
   - Warmup and previous-day state behavior.
   - Position sizing and daily risk limits.

2. Data contract
   - Source files and date coverage.
   - Required columns and timestamp timezone.
   - Missing-data behavior.
   - Contract/strike/expiry selection rules.
   - Date windows for train, validation, walk-forward, and final holdout.

3. Cost model
   Use explicit configuration, never hidden constants:

   - Backtest slippage: default 1.0 price point per side unless the user gives
     another value.
   - Brokerage: explicit per-order value, default 0 when applicable.
   - STT: apply only to the taxable sell-side turnover defined by the market.
   - Exchange transaction charges: apply to the configured turnover base.
   - SEBI charges: apply to the configured turnover base.
   - Stamp duty: apply only to the buy-side base where applicable.
   - GST: 18% on brokerage + exchange charges + SEBI charges only. Do not apply
     GST to option premium/trade value, STT, or stamp duty.
   - Keep live-order slippage and backtest slippage as separate settings.

4. Optimization-method applicability matrix
   For every method below, explicitly mark: USE, NOT APPLICABLE, or DEFERRED;
   give the target file/function, correctness test, and benchmark metric.

   | Method | Applicability | Implementation location | Parity test | Benchmark |
   |---|---|---|---|---|
   | Incremental indicator computation | | | | |
   | Signal/execution separation | | | | |
   | Factorized feature cache | | | | |
   | Threshold/state materialization | | | | |
   | Monotonic pointer/searchsorted lookup | | | | |
   | Worker-local parsed-data cache | | | | |
   | Day-batched candidate evaluation | | | | |
   | Candidate grouping by shared base key | | | | |
   | Previous-day warmup carry | | | | |
   | Structure-of-arrays/packed layout | | | | |
   | Columnar Parquet/Arrow cache | | | | |
   | Partial-fidelity screening | | | | |
   | TPE/Hyperband pruning | | | | |
   | Numba/GPU execution | | | | |

5. Testing plan
   - Unit tests for costs, GST base, drawdown, date isolation, and parameter
     normalization.
   - Reference-vs-optimized trade-for-trade parity test.
   - Five-day smoke test.
   - Benchmark with runtime, worker count, cache builds/hits, memory when
     available, and exact parity.
   - Walk-forward and holdout tests that prove no future data is used for
     selection.

6. Deliverables
   - Exact files to create or modify.
   - Exact functions/classes and interfaces.
   - Exact test commands.
   - Exact smoke command.
   - Exact benchmark command.
   - Output paths for CSV, JSON, logs, and reports.
   - Rollback/fallback behavior if an optimization fails parity or speed.

Do not start implementation until the plan contains this method matrix and the
smoke gate.

Split the plan into independently verifiable phases. Each phase must include:

- Phase objective and scope.
- Files and interfaces affected.
- Tests and expected failure/success behavior.
- Smoke or benchmark gate.
- Evidence that DeepSeek V4 Flash must produce.
- GPT 5.6 verification checklist and acceptance verdict.
- Explicit dependency on the previous phase's verification verdict.

PHASE 3: REFERENCE IMPLEMENTATION AND TDD
=========================================
The DeepSeek V4 Flash execution model implements this phase. At the end of the
phase, GPT 5.6 must verify the reference semantics, TDD evidence, and plan
compliance before optimization work begins.

1. Write a failing test for each new behavior.
2. Run the test and confirm it fails for the intended reason.
3. Implement the smallest correct reference behavior.
4. Run the test until it passes.
5. Add edge-case tests before optimizing.

The reference implementation must define the exact semantics for:

- Timestamp ordering and candle aggregation.
- Indicator warmup and state carry.
- Signal ordering when multiple signals share a minute.
- Entry price and missing-bar behavior.
- Same-bar SL/TP precedence.
- Reversal contract/strike selection.
- Daily shutdown and consecutive-loss behavior.
- EOD liquidation.
- Fees and slippage.

PHASE 4: APPLY SAFE OPTIMIZATIONS
=================================
The DeepSeek V4 Flash execution model implements one optimization phase at a
time. GPT 5.6 verifies each optimization separately before the next optimization
is applied. A rejected optimization must not be treated as enabled.

Apply only methods marked USE in the plan. Prefer these patterns when they fit:

1. Incremental computation
   Update indicators once per new candle. Do not recompute full rolling windows
   for every candidate or minute.

2. Signal/execution separation
   Build signal state once, then execute multiple stop/target/risk variants
   against the immutable signal state.

3. Factorized feature caching
   Cache features using only parameters that affect them, such as stochastic
   periods and ATR period. Apply threshold-only parameters later.

4. Pointer-based filtering
   Use monotonic indices or searchsorted pointers for sorted minute bars and
   spot data. Avoid repeated full-array scans or repeated timestamp searches.

5. Worker-local data cache
   Parse each day file once per worker and reuse the normalized arrays.

6. Candidate batching and grouping
   Evaluate candidates in batches. Group candidates sharing signal/base keys so
   raw features are built once per day/key.

7. Warmup carry
   Carry the previous trading day's indicator state only where the reference
   engine does. Never share future state across train/OOS boundaries.

8. Partial-fidelity screening
   Evaluate cumulative day blocks such as 5, 20, 60, then full training data.
   Prune weak candidates before full fidelity, but do not use OOS or holdout
   results for pruning.

9. TPE/Hyperband
   Use an explicit seed, batch size, resource schedule, and pruning ledger.
   Record every trial as COMPLETE or PRUNED. Never silently drop candidates.

10. Packed/Numba/GPU paths
    Implement only after profiling shows the bottleneck is worth moving. The
    optimized path must pass exact trade-for-trade parity before timing. Keep
    the reference CPU path available as the fallback.

11. Columnar storage
    Use Parquet/Arrow only if measured against the current CSV/cache path. Keep
    it opt-in if the improvement is marginal.

After each optimization, report:

- Reference seconds.
- Optimized seconds.
- Speedup.
- Worker count.
- Cache builds/hits.
- Memory when available.
- Trade count parity.
- Exact trade/summary parity.
- Whether the optimization was retained or rejected.

A faster result without parity is rejected. A parity-preserving result with no
meaningful speedup is not enabled by default.

PHASE 5: MANDATORY SMOKE TEST
=============================
DeepSeek V4 Flash runs the smoke gate. GPT 5.6 verifies the smoke output,
reference parity, worker count, trade-count sanity, and cost-model settings
before any long process is allowed.

Before any long process:

- Use the first five available trading days.
- Use no more than 2 workers for the smoke unless the test specifically checks
  the fixed 8-worker path.
- Run the known reference configuration and at least one alternate candidate.
- Require complete result accounting.
- Require 15-40 trades for the normal five-day strategy smoke unless the
  strategy's documented expected range is different.
- Require plausible win rate and P&L.
- Require optimized/reference parity for every trade field that affects P&L.
- Stop immediately on a suspicious trade count, zero trades, parity mismatch,
  or cost-model mismatch.

Print an unambiguous `SMOKE TEST OK` or `SMOKE TEST FAIL` result and write the
smoke JSON/log before starting a long process.

PHASE 6: BENCHMARK AND SEARCH
=============================
DeepSeek V4 Flash runs the bounded benchmark and search. GPT 5.6 verifies
runtime, worker ceiling, cache accounting, trial accounting, parity, and the
absence of leakage before search results are used for validation.

Only after smoke passes:

- Run the bounded benchmark first.
- Then run the requested search with `workers <= 8`.
- Redirect long output to a timestamped log.
- Persist trial/stage data to CSV and final summaries to JSON.
- Preserve the exact parameter values, date windows, cost settings, worker
  count, seed, and code version/date in the result.
- Never treat an in-sample winner as a production strategy.

PHASE 7: VALIDATION
====================
DeepSeek V4 Flash runs the walk-forward and holdout validations. GPT 5.6
verifies chronological isolation, frozen holdout candidates, cost calculations,
drawdown calculations, and the final acceptance gates.

Use chronological evaluation:

- Train/selection data: candidate generation and optimization only.
- Validation data: model/strategy comparison and selection gates.
- Walk-forward data: repeated train-to-future evaluation.
- Final holdout/blind data: frozen candidate list only; never optimize on it.

Report at minimum:

- Gross points and net points.
- Gross rupees and net rupees.
- Trade count.
- Win rate.
- Profit factor.
- Fees and slippage paid.
- Average win/loss and expectancy.
- Daily and monthly results.
- Maximum peak-to-trough drawdown in points and rupees.
- Drawdown start and end dates.
- Worst day, worst month, and losing streak.
- Runtime and worker count.
- Missing-data and skipped-day counts.

Compute drawdown from chronological net daily equity, not from monthly ramp
output alone. Separate fixed one-lot strategy performance from any money-
management or lot-ramp simulation. Treat unlimited scaling as non-deployable
unless capacity, liquidity, margin, and execution impact are modeled.

PHASE 8: FINAL REPORT
=====================
DeepSeek V4 Flash prepares the final report and evidence index. GPT 5.6 performs
the final phase verification and confirms that every prior phase has a PASS or
PASS WITH IMPROVEMENTS verdict. The final report must list any FAIL/BLOCKED
phase and must not call the strategy production-ready if one exists.

Return a concise final report with these headings:

STATUS:
FILES_CHANGED:
REFERENCE_ENGINE:
OPTIMIZATIONS_USED:
OPTIMIZATIONS_REJECTED:
WORKER_LIMIT:
COST_MODEL:
SMOKE_TEST:
BENCHMARK:
SEARCH_RESULTS:
WALK_FORWARD_RESULTS:
HOLDOUT_RESULTS:
DRAWDOWN_AND_RISK:
PARITY_EVIDENCE:
OUTPUT_PATHS:
KNOWN_LIMITATIONS:
RECOMMENDATION:

Use evidence from command output and saved artifacts. Do not claim that a
strategy is best, profitable, robust, or production-ready unless the stated
validation gates prove it. If a long process was not run, say so explicitly.
```

The project-specific implementation references used to create this workflow
are `artifacts/f6_hybrid/optimization_methods.md`,
`artifacts/f6_hybrid/phase0_inventory.md`, and `AGENTS.md`.
