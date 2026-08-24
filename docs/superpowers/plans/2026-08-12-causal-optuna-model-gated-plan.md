# Causal F6 Optuna Plan

> Superseded for execution by `docs/superpowers/plans/SHORTCUT_BACKTEST.md`.
> GPT-5.6 Luna is now the sole executor and verifier; the DeepSeek role below
> is retained only as historical planning context.

## Goal

Build one causally correct research engine that can identify a robust F6
configuration under realistic costs and drawdown constraints.

The final research result must report:

- Net points after costs
- Net rupees after costs
- Maximum drawdown
- Average trades per day
- Average SL points
- Average TP points
- Win rate and profit factor
- Year-by-year results
- Train, validation, and blind performance

This plan does **not** add paper trading or change live trading behavior.

## Model Roles

### DeepSeek V4 Flash

DeepSeek is the implementation model. It may:

- Edit code only inside the current phase scope.
- Run smoke tests and approved backtests.
- Produce phase artifacts and logs.
- Stop when the phase acceptance checklist is complete.

DeepSeek must not:

- Change live configuration.
- Skip smoke tests.
- Promote a candidate based only on gross P&L.
- Continue to the next phase without verification.

### GPT-5.6 Luna

GPT-5.6 Luna is the verification model. It must:

- Review the diff and phase artifacts.
- Check for look-ahead bias and live/backtest drift.
- Recalculate important metrics independently where possible.
- Return exactly `PASS` or `FAIL` with findings.

GPT-5.6 Luna must not silently edit code during verification.

## Model Switching Rule

At the start of every phase:

1. Switch to **DeepSeek V4 Flash**.
2. Give it the phase instruction below.
3. Wait for the phase report.
4. Switch to **GPT-5.6 Luna**.
5. Paste the verification prompt below.
6. Continue only if GPT-5.6 Luna returns `PASS`.

If verification returns `FAIL`:

1. Keep the phase open.
2. Switch back to DeepSeek V4 Flash.
3. Give DeepSeek only the listed failures to fix.
4. Run the phase checks again.
5. Switch back to GPT-5.6 Luna.
6. Repeat until `PASS`.

### Exact Handoff Prompt

Use this after every DeepSeek phase:

```text
Switch to GPT-5.6 Luna. Verify only Phase [N] of the Causal F6 Optuna Plan.
Read the phase artifacts, tests, logs, and diff. Do not edit files or launch
long backtests. Check every acceptance criterion, especially look-ahead bias,
data leakage, cost handling, and live/backtest parity. Return:

VERDICT: PASS or FAIL
FINDINGS: numbered, with file and line references
REQUIRED_FIXES: exact fixes if FAIL
REMAINING_RISKS: concise list

Do not approve based on gross P&L alone.
```

## Canonical Strategy Profiles

The engine must support explicit divergence profiles. Never mix their results:

- `no_divergence`: F6 immediate entry plus no divergence requirement for the normal path.
- `current_pivot`: confirmed OHLC pivot divergence.
- `previous_rolling`: legacy rolling-close extrema divergence for historical reproduction.

The old champion baseline is frozen as:

```text
S1=(12,3)
S4=(50,10)
ATR period=10
ATR SL multiplier=3.0
ATR TP multiplier=6.0
F6 S4 threshold=79.5
F6 S1 threshold=25.0
Consecutive-loss limit=8
Pin bar=45% lower shadow / 45% body / 25% upper shadow
Timeframes=1m, 2m, 3m, 5m
```

The baseline is for reproduction only. It is not automatically the final
live configuration.

## Phase 0: Freeze Evidence

### DeepSeek instruction

```text
Phase 0 only. Do not edit strategy or engine code. Record the exact baseline
parameters, divergence profile, pin-bar thresholds, timeframe rules, cost
model, data manifest, date ranges, worker count, Python version, package
versions, and current git revision. Create a reproducibility JSON manifest.
Run only file/schema checks, not a long backtest.
```

### Deliverables

- `artifacts/f6_hybrid/canonical_manifest.json`
- Data file list and SHA-256 manifest
- Parameter profile JSON
- Environment/version report

### Acceptance criteria

- Every parameter is explicit.
- Data range and missing days are recorded.
- Divergence mode is explicit.
- Cost assumptions are explicit.
- No live files are changed.

**Switch to GPT-5.6 Luna for verification. Advance only on `PASS`.**

## Phase 1: Causal Engine

### DeepSeek instruction

```text
Phase 1 only. Build the research-only causal engine. Process one day in
chronological order. For every candle: update indicators, form clock-aligned
1m/2m/3m/5m bars, generate signals, check the current position, apply exits,
then advance. Never precompute end-of-day indicator state for an earlier exit.
Implement no_divergence, current_pivot, and previous_rolling profiles behind
one explicit switch. Do not change live bot code.
```

### Required behavior

- Clock-aligned timeframe boundaries, not row-count boundaries.
- Confirmed pivot divergence uses only data available at that candle.
- Legacy divergence is isolated and labeled.
- F6 immediate signal remains separate from normal divergence/pin-bar setup.
- Same-bar SL/TP precedence is explicit and tested.
- Reverse trades are handled explicitly.
- EOD and daily loss behavior matches the live specification.

### Tests

- Synthetic candle test for each divergence profile.
- Test that future candles cannot change a previous signal or exit.
- Test 1m/2m/3m/5m clock boundaries with missing minutes.
- Test same-bar stop/target precedence.
- Test reverse-side symbol selection.

**Switch to GPT-5.6 Luna for verification. Advance only on `PASS`.**

## Phase 2: Data and Speed Layer

### DeepSeek instruction

```text
Phase 2 only. Optimize data access without changing engine output. Convert
daily CSV data into sorted, deduplicated Parquet or Arrow files with explicit
numeric dtypes. Build worker-local caches and reuse one process pool across
trials. Preserve a parity test proving raw CSV and cached data produce the
same trades and metrics.
```

### Required optimizations

- Polars lazy CSV scan or PyArrow/Parquet conversion.
- `usecols` and explicit dtypes.
- Sorted arrays for `numpy.searchsorted`.
- One persistent 8-worker pool.
- No nested multiprocessing and Numba `prange` together.
- Cache key includes data hash and engine version.

### Acceptance criteria

- Cached and uncached outputs match exactly.
- Duplicate rows are reported and handled deterministically.
- Speed benchmark records wall time and memory.
- No approximation is introduced.

**Switch to GPT-5.6 Luna for verification. Advance only on `PASS`.**

## Phase 3: Integrated Costs and Metrics

### DeepSeek instruction

```text
Phase 3 only. Apply realistic costs during the simulation. Model entry and
exit fills conservatively from the candle/quote data, apply slippage on both
legs, apply statutory fees, and calculate risk stops from the same net state.
Add max drawdown, average trades/day, average SL, average TP, yearly equity,
and trade-reason fields.
```

### Required metrics

- Gross points and net points
- Gross P&L and net P&L
- Fees and slippage separately
- Maximum drawdown and drawdown duration
- Average trades per trading day
- Average SL points
- Average TP points
- Win rate, PF, expectancy, median trade
- Yearly and fold-level results

**Switch to GPT-5.6 Luna for verification. Advance only on `PASS`.**

## Phase 4: Optuna Screening

### DeepSeek instruction

```text
Phase 4 only. Build the Optuna search around the causal engine. Use a
multivariate grouped TPE sampler with seed 42. Use a scalar screening score
for pruning: net points after costs minus a drawdown penalty. Apply minimum
trade-count and maximum-drawdown constraints. Use contiguous resource stages
of 20, 60, 200, and full training days. Save every trial and never use the
blind period during optimization.
```

### Recommended Optuna settings

```python
TPESampler(
    multivariate=True,
    group=True,
    constant_liar=True,
    seed=42,
)
```

- Hyperband or Successive Halving for single-objective screening.
- Do not use Optuna `n_jobs` with the internal worker pool.
- Enqueue the known baseline as a reference trial.
- Keep the top Pareto candidates, not only the top P&L candidate.

### Search space

- S1 k: `7, 9, 12, 14`
- S4 k: `50, 60, 75`
- ATR period: `10, 14, 20`
- SL multiplier: `1.0, 1.5, 2.0, 2.5, 3.0`
- TP multiplier: `2.0, 3.0, 4.0, 5.0, 6.0`
- F6 S4 threshold: `75, 79.5, 85`
- F6 S1 threshold: `15, 20.5, 25`
- Consecutive-loss limit: `4, 6, 8`
- Divergence profile: separate studies, never mixed silently

**Switch to GPT-5.6 Luna for verification. Advance only on `PASS`.**

## Phase 5: Chronological Validation

### DeepSeek instruction

```text
Phase 5 only. Freeze the top candidates from Phase 4. Evaluate them without
parameter changes on train 2020-2022, validation 2023-2024, and blind 2025-2026.
Report every candidate by fold, year, regime, and cost scenario. Do not select
the best blind result after looking at it.
```

### Required gates

- Training result is positive after costs.
- Validation result is positive after costs.
- Blind result is positive after costs.
- No single year supplies most of the profit.
- Drawdown remains within the declared limit.
- Results survive 0.5, 1.0, and 2.0 point slippage scenarios.

**Switch to GPT-5.6 Luna for verification. Advance only on `PASS`.**

## Phase 6: Statistical Overfit Audit

### DeepSeek instruction

```text
Phase 6 only. Audit the frozen finalists for multiple-testing and backtest
overfitting. Calculate Probability of Backtest Overfitting, Deflated Sharpe
Ratio, block bootstrap confidence intervals, and parameter sensitivity. Use
the free research tools only as independent audits; do not let an audit tool
replace the causal engine.
```

### Free audit tools

- `https://github.com/esvhd/pypbo`
- `https://github.com/Aliipou/backtest-audit`
- `https://github.com/eslazarev/purged-cross-validation`

**Switch to GPT-5.6 Luna for verification. Advance only on `PASS`.**

## Phase 7: Final Report

### DeepSeek instruction

```text
Phase 7 only. Produce the final report. Clearly separate gross and net
results, training/validation/blind results, drawdown, average trades/day,
average SL/TP, cost assumptions, data hashes, engine revision, and all known
limitations. Do not recommend live deployment automatically.
```

### Final decision categories

- `PROMISING`: positive blind net result and passes overfit audit.
- `PAPER_ONLY`: positive but fragile, low sample, or high drawdown.
- `REJECT`: negative blind result, cost failure, or leakage concern.

**Switch to GPT-5.6 Luna for final verification. Only a final `PASS` closes the plan.**

## Explicit User Workflow

At the beginning of Phase 0, tell the user:

```text
Switch to DeepSeek V4 Flash now. Execute Phase 0 only.
When Phase 0 reports complete, switch to GPT-5.6 Luna and paste the handoff prompt.
Do not start Phase 1 until Luna returns PASS.
```

At every later phase, tell the user:

```text
Switch to DeepSeek V4 Flash for Phase [N]. Stop after its acceptance checklist.
Then switch to GPT-5.6 Luna for verification. Continue only after PASS.
```

No paper-trading phase is part of this plan. Live changes require a separate
explicit decision after the final verification.
