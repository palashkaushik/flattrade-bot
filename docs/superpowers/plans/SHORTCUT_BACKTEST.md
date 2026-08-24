# Shortcut Backtest

## Purpose

Shortcut Backtest is the project name for the fast-but-accurate research method:

```text
cached data -> cheap multi-block screening -> causal finalist rerun
            -> chronological validation -> blind test -> statistical audit
```

It is a research workflow only. It does not change live trading or add paper
trading.

## PDF-Derived Safeguards

The supplied optimization references support these methods when they preserve
the same causal engine:

- Constrained, meaningful parameter ranges before trials are submitted.
- Cached immutable features and incremental indicator updates.
- Separate signal generation from execution and accounting logic.
- Cheap chronological screening followed by a full-engine finalist rerun.
- Early pruning only after a minimum chronological block has been evaluated.
- Risk-adjusted scoring, walk-forward validation, and sensitivity checks.

The shortcut layer must not replace execution simulation with a simplified
fill model, lower-frequency approximation, random train/test splits, or any
future-looking feature. Every finalist is rerun through the full causal engine.

## Model Gate

GPT-5.6 Luna executes and verifies every phase. The process still stops at
each phase checkpoint; Luna must review the acceptance criteria before moving
forward.

At every phase checkpoint, use:

```text
GPT-5.6 Luna: verify Shortcut Backtest Phase [N].
Do not move to the next phase until the acceptance criteria,
look-ahead protection, data integrity, cost model, and reproducibility.
Return exactly:

VERDICT: PASS or FAIL
FINDINGS:
REQUIRED_FIXES:
REMAINING_RISKS:
```

## Phase 0: Freeze

GPT-5.6 Luna must record:

- Git revision
- Python and dependency versions
- Data file manifest and hashes
- Duplicate and missing-row counts
- Canonical strategy parameters
- Divergence profile
- Pin-bar thresholds
- Timeframe rules
- Cost and slippage assumptions

Do not edit strategy code in Phase 0.

**Checkpoint:** GPT-5.6 Luna verifies. Continue only after `PASS`.

## Phase 1: Causal Engine

Build one candle-by-candle engine:

- Clock-aligned 1m, 2m, 3m, and 5m bars
- No future-state indicator access
- Explicit divergence profile
- F6 immediate branch separated from normal divergence/pin-bar branch
- Same-bar SL/TP precedence
- Contract rotation and reverse trades
- Conservative quote-based fills
- Fees and slippage applied during simulation

**Checkpoint:** GPT-5.6 Luna verifies. Continue only after `PASS`.

## Phase 2: Fast Data Layer

- Convert daily CSVs to sorted Parquet or Arrow data.
- Use Polars lazy scans or typed column reads.
- Cache immutable day arrays inside workers.
- Use NumPy sorted arrays and `searchsorted`.
- Keep one persistent worker pool.
- Do not combine Optuna parallel jobs with the internal worker pool.

Prove cached and uncached results are identical.

**Checkpoint:** GPT-5.6 Luna verifies. Continue only after `PASS`.

## Phase 3: Optuna Screening

Before any long Optuna run, execute and pass:

```text
python artifacts/f6_hybrid/causal_optuna_shortcut.py \
  --mode new_divergence --trials 2 --smoke
```

If the smoke command fails, hangs, produces zero trades, or violates the
expected metric range, do not start the long run.

Use multivariate grouped TPE:

```python
TPESampler(
    multivariate=True,
    group=True,
    constant_liar=True,
    seed=42,
)
```

Use a single screening score:

```text
net points after costs - drawdown penalty
```

Reject invalid parameter combinations before simulation, and do not compare
trials with different data windows, fees, slippage, or warm-up rules.

Screen on balanced chronological blocks:

- 20 days
- 60 days
- 200 days
- Full 2020–2022 training period

Use Hyperband or Successive Halving only for this single-objective screening
stage. Keep the top Pareto candidates for final evaluation.

Screening is allowed to rank and prune candidates only. It is not an
authoritative performance result.

**Checkpoint:** GPT-5.6 Luna verifies. Continue only after `PASS`.

## Phase 4: Validation

Freeze the selected candidates and run without changes:

- Train: 2020–2022
- Validate: 2023–2024
- Blind: 2025–2026

Report:

- Net points and rupees
- Maximum drawdown
- Average trades/day
- Average SL and TP
- Win rate and PF
- Fees and slippage
- Yearly equity

**Handoff:** Switch to GPT-5.6 Luna. Continue only after `PASS`.

## Phase 5: Statistical Audit

Run only on finalists:

- Probability of Backtest Overfitting
- Deflated Sharpe Ratio
- Block bootstrap confidence intervals
- Parameter sensitivity
- Slippage sensitivity at 0.5, 1.0, and 2.0 points/side

**Checkpoint:** GPT-5.6 Luna performs final approval.

## Current Start Command

```text
GPT-5.6 Luna: execute Shortcut Backtest Phase 0 only. Freeze the manifest and
canonical parameters, make no strategy changes, run only bounded checks, and
stop with a Phase 0 report. Verify the report before starting Phase 1.
```
