# F6 Hybrid Backtest Implementation Plan

> **For agentic workers:** Execute one phase at a time. The token-cheap worker implements a phase, then 5.6 Luna independently verifies it. Do not start the next phase until Luna returns `PASS`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fast and accurate hybrid F6 backtest that exhaustively evaluates the declared F6 parameter grid, preserves the existing CPU engine as the correctness oracle, uses CUDA batching where it is semantically safe, and produces cost-adjusted walk-forward and holdout results.

**Architecture:** The existing `grid_optimize_f6_atr.py` remains the CPU reference implementation. New F6-specific modules normalize data and candidates, batch exact candidate execution on CUDA through CuPy kernels, and keep walk-forward selection, fees, drawdown, reporting, and final audit under CPU control. GPU results are never trusted until they pass CPU parity fixtures, smoke parity, and known-window parity.

**Tech Stack:** Python 3.13, NumPy, pandas, Numba CPU JIT where useful, CuPy `RawKernel` for CUDA, multiprocessing for the CPU oracle, unittest, CUDA events or `cupyx.profiler` for timing.

## Global Constraints

- Work only in `C:\Websites\FLATTRADE BOT`; the temporary sweep copy is read-only research material, not a source of truth.
- Read `AGENTS.md`, `graphify-out/GRAPH_REPORT.md`, and the relevant graph nodes before reading source files.
- Leave all existing unrelated worktree changes untouched.
- Do not commit unless the user explicitly requests a commit.
- Use `apply_patch` for manual source edits.
- Write tests before production code for every new behavior.
- Run the mandatory five-day smoke test before every modified backtest and before any full run.
- Never claim GPU correctness without CPU parity.
- Never silently sample, prune, or cap an exhaustive grid. Any user-approved cap must be reported as non-exhaustive.
- The current eight `SEARCH_SPACE` axes produce 15,552 combinations. Phase 0 must verify whether additional fixed F6 parameters should become search axes before this count is treated as final.
- Keep price and P&L calculations in float64 until a measured parity test proves another precision safe.
- Use `backend=auto` for fallback behavior and `backend=cuda` for an explicit CUDA requirement. CUDA unavailability under `backend=cuda` is a hard failure.
- Do not modify live trading parameters, `flattrade_bot/main.py`, or broker execution code.
- Do not update `BACKTEST_LEDGER.md` until the user reviews the completed result.

## What The Six-To-Eight-Hour Estimate Means

The six-to-eight-hour estimate applies to the first verified hybrid slice:

- CPU baseline and parameter inventory.
- Canonical data layout.
- GPU indicator and candidate-batch prototype.
- First exact GPU execution path or a clearly documented parity blocker.
- Synthetic and five-day CPU/GPU parity checks.

It does not promise that the full 15,552-candidate multi-year walk-forward run will finish in six to eight hours. Full runtime depends on the actual CUDA installation, batch size, data coverage, and whether the exact F6 state machine fits the GPU kernel without semantic divergence. The system must benchmark before launching the full grid.

## Phase Gates

Every phase has two workers:

- **Token-cheap worker:** edits only the phase files, runs targeted commands, and returns a concise handoff.
- **5.6 Luna verifier:** independently reads the plan and diff, runs the required checks, and returns `PASS`, `FAIL`, or `BLOCKED` with file/line findings.

The workflow is:

1. Give the phase implementation prompt to the token-cheap worker.
2. Give the phase verification prompt and the cheap worker's report to Luna.
3. If Luna returns `FAIL`, give only the findings to the cheap worker for a repair pass.
4. Re-run Luna on the repair.
5. Advance only after `PASS`.

The cheap worker must not declare a phase complete based only on intent. Luna must inspect evidence.

## Shared Prompt Rules

Prepend this context to every worker prompt:

```text
Repository: C:\Websites\FLATTRADE BOT
Task: F6-specific hybrid CPU-oracle + CUDA-batch backtest.
Plan: docs/superpowers/plans/2026-08-10-f6-hybrid-token-phased.md
Hardware observed: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 610.74.

Read AGENTS.md and graphify-out/GRAPH_REPORT.md before raw source. Use graph.json/codebase-memory to locate symbols before reading implementation files. Preserve unrelated dirty changes. Do not use the temporary sweep copy as production source. Do not commit. Use apply_patch for manual edits. Follow TDD: write a failing test, run it, implement the smallest fix, then run the targeted and regression tests. Never run a full backtest before the five-day smoke test passes.
```

## Phase 0: Baseline And F6 Parameter Inventory

**Time budget:** 30-45 minutes.

**Purpose:** Establish the CPU reference, verify the environment, and list every F6 parameter rather than assuming `SEARCH_SPACE` is exhaustive.

**Phase checklist:**

- [ ] Run the five-day CPU smoke and environment probes.
- [ ] Inventory searched axes, fixed constants, and known reference values.
- [ ] Write the two evidence files without changing production Python.
- [ ] Obtain a Luna `PASS`.

**Files:**
- Read: `grid_optimize_f6_atr.py`
- Read: `backtest_5y_optimized.py`
- Read: `backtest_walkforward_fees.py`
- Read: `backtest_monthly_ramp.py`
- Read: `validate_top_candidates.py`
- Read: `BACKTEST_LEDGER.md`
- Read-only comparison: temporary `f6_gpu_batch.py`, `f6_gpu_execution.py`, `f6_numba_primitives.py`
- Create generated evidence: `artifacts/f6_hybrid/phase0_inventory.md`
- Create generated evidence: `artifacts/f6_hybrid/phase0_baseline.json`

**Required baseline commands:**

```text
python grid_optimize_f6_atr.py --smoke
python -m py_compile grid_optimize_f6_atr.py backtest_5y_optimized.py backtest_walkforward_fees.py backtest_monthly_ramp.py validate_top_candidates.py
python -c "import importlib.util; print('cupy', bool(importlib.util.find_spec('cupy'))); print('numba', bool(importlib.util.find_spec('numba')))"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

The inventory must record current `SEARCH_SPACE`, `CHAMPION`, `CHAMPION_EXPECTED`, `TF_SPECS`, offsets, session boundaries, ATR fallback rules, daily loss, consecutive-loss behavior, divergence, reversal, same-bar exit precedence, cost constants, known reference values, and every parameter that is currently fixed but could reasonably be searched.

### Token-Cheap Implementation Prompt

```text
Apply Shared Prompt Rules. Execute Phase 0 only.

Do not edit production Python files. Read the listed F6 files through graph-guided navigation, run the four baseline commands, and create the two evidence files under artifacts/f6_hybrid/. The inventory must distinguish searched axes from fixed engine constants and must quote exact source locations. Record actual smoke output, CUDA/CuPy availability, and any pre-existing test failures. Compare the temporary GPU prototypes only as research notes; do not copy them.

Return exactly:
STATUS: PASS or BLOCKED
FILES: changed/generated paths
BASELINE: smoke trades, WR, net Rs, PF, CUDA/CuPy status
INVENTORY: searched axes, fixed parameters, proposed additional axes
BLOCKERS: concrete blockers or NONE
NEXT: the smallest Phase 1 prerequisite
```

### 5.6 Luna Verification Prompt

```text
You are 5.6 Luna. Verify Phase 0 against the plan; do not implement fixes.

Read AGENTS.md, graphify-out/GRAPH_REPORT.md, the plan, the cheap worker report, and the two phase0 evidence files. Inspect the exact source lines for every inventory claim. Re-run the smoke, compile, dependency, and nvidia-smi commands. Confirm the smoke is not zero trades and is within the repository sanity range. Confirm no production Python or unrelated files changed.

Return:
VERDICT: PASS, FAIL, or BLOCKED
FINDINGS: ordered by severity, with path and line references; write NONE if clean
EVIDENCE: commands actually run and relevant outputs
PLAN_CONGRUENCY: exact statement of whether the inventory covers all F6 behavior-changing constants
```

**Gate:** Phase 1 cannot start until the baseline and parameter inventory are independently verified.

## Phase 1: Contracts, Candidate Grid, And CPU Oracle

**Time budget:** 60-90 minutes.

**Purpose:** Make the CPU behavior reproducible and define immutable candidate/result contracts before CUDA work.

**Phase checklist:**

- [ ] Write failing contract, grid, and oracle tests.
- [ ] Implement deterministic candidates and the thin CPU oracle.
- [ ] Verify the current 15,552-candidate count or update it from Phase 0 evidence.
- [ ] Obtain a Luna `PASS`.

**Files:**
- Create: `f6_hybrid/__init__.py`
- Create: `f6_hybrid/contracts.py`
- Create: `f6_hybrid/params.py`
- Create: `f6_hybrid/cpu_oracle.py`
- Create: `test_f6_hybrid_contracts.py`
- Create: `test_f6_hybrid_params.py`
- Create: `test_f6_hybrid_cpu_oracle.py`

**Required interfaces:**

```python
F6Candidate.from_mapping(values) -> F6Candidate
F6Candidate.key() -> tuple
F6Candidate.sha256() -> str
run_cpu_oracle(candidate, days, files, spot_all, workers=1) -> list[dict]
summarize_cpu_trades(trades) -> dict
```

`F6Candidate` must include every search axis approved by Phase 0. With the current engine this is the eight axes `s1_k`, `s4_k`, `atr_period`, `atr_sl_mult`, `atr_tp_mult`, `f6_s4_thresh`, `f6_s1_thresh`, and `consec_loss`, plus fixed `s1_d=3`. Fixed engine configuration must be represented separately and visibly, not silently excluded from the hash.

The CPU oracle must call the existing `grid_optimize_f6_atr.run_days()` and preserve its returned trade dictionaries. It must not fork or rewrite signal logic.

**Required tests:**

- Candidate serialization and hash are deterministic.
- Current eight-axis grid enumerates exactly 15,552 candidates.
- Candidate order is stable across processes.
- Invalid or missing axes fail clearly.
- A small CPU fixture produces the expected trade summary.
- The known champion smoke/reference path can be invoked without changing the engine.

### Token-Cheap Implementation Prompt

```text
Apply Shared Prompt Rules. Execute Phase 1 only, after reading Phase 0 evidence.

Write the failing unittest cases first and run them to prove they fail. Then create the f6_hybrid package contracts, deterministic candidate enumeration, and a thin CPU-oracle wrapper around grid_optimize_f6_atr.run_days. Do not rewrite or copy the F6 signal engine. Include all Phase 0-approved axes and keep fixed values visible in the candidate/config hash. Run the new tests, existing targeted tests, and py_compile.

Required commands:
python -m unittest -v test_f6_hybrid_contracts test_f6_hybrid_params test_f6_hybrid_cpu_oracle
python -m compileall -q f6_hybrid

Return STATUS, FILES, TESTS, GRID_COUNT, ORACLE_RESULT, BLOCKERS, and NEXT in that order.
```

### 5.6 Luna Verification Prompt

```text
You are 5.6 Luna. Verify Phase 1; do not implement fixes.

Read the Phase 0 inventory, plan, changed files, and cheap report. Check that the candidate hash includes every behavior-changing approved parameter and that enumeration is deterministic. Run the three targeted unittest modules and py_compile. Trace the oracle call to confirm it uses the unchanged reference engine. Reject the phase if the 15,552 count is asserted without matching the actual approved axes, if fixed parameters are hidden, or if the wrapper changes trade semantics.

Return VERDICT, FINDINGS with path/line references, COMMANDS, GRID_AND_HASH_CHECK, ORACLE_CHECK, and PLAN_CONGRUENCY.
```

**Gate:** Candidate identity and CPU oracle must pass before data packing or GPU implementation.

## Phase 2: Canonical F6 Data Layout

**Time budget:** 60-90 minutes.

**Purpose:** Convert irregular CSV input into deterministic dense arrays suitable for both CPU and CUDA without moving data per candidate.

**Phase checklist:**

- [ ] Write failing packing and warmup tests with synthetic files.
- [ ] Implement deterministic day and batch packing with documented shapes.
- [ ] Verify slot mapping, missing values, and day boundaries.
- [ ] Obtain a Luna `PASS`.

**Files:**
- Create: `f6_hybrid/data.py`
- Create: `test_f6_hybrid_data.py`
- Modify only if required: `f6_hybrid/contracts.py`

**Required interfaces:**

```python
CanonicalDay.from_files(day, option_file, previous_option_file, spot) -> CanonicalDay
CanonicalDay.to_numpy() -> dict[str, object]
pack_days(days, files, spot_all) -> CanonicalDataset
```

The layout must document shapes and dtypes. It must preserve minute order, day boundaries, option symbol/slot mapping, CE/PE identity, previous-day warmup data, session-end markers, and missing values. The GPU-facing arrays should be bar-major with fixed slot indexing for a batch. The CPU oracle remains responsible for exact dynamic mapping until parity proves the packed representation equivalent.

**Required tests:**

- Synthetic CSV fixture packs stable arrays and slot maps.
- Missing symbol/bar behavior is deterministic and reported.
- Previous-day warmup is not confused with trading-day output.
- Repacking the same files yields identical arrays and hashes.
- No pandas objects enter the GPU interface.

### Token-Cheap Implementation Prompt

```text
Apply Shared Prompt Rules. Execute Phase 2 only.

Write failing tests for deterministic minute/slot packing before implementation. Build CanonicalDay and CanonicalDataset around the exact F6 data requirements from Phase 0. Use small synthetic CSV fixtures; do not run a multi-year backtest. Record array shapes, dtypes, missing values, day IDs, session markers, and previous-day warmup behavior. Keep the output compatible with both NumPy and CuPy without copying per candidate. Run the targeted tests, the Phase 1 tests, and py_compile.

Return STATUS, FILES, ARRAY_CONTRACT, TESTS, BLOCKERS, and NEXT.
```

### 5.6 Luna Verification Prompt

```text
You are 5.6 Luna. Verify Phase 2; do not implement fixes.

Check the data contract against grid_optimize_f6_atr.cached_day/process_day and the Phase 0 inventory. Run test_f6_hybrid_data plus all Phase 1 tests. Inspect that day boundaries, previous-day warmup, option slots, CE/PE mapping, NaN handling, and session markers are explicit. Confirm no candidate loop reparses CSV files and no pandas object crosses the GPU boundary. Return VERDICT, FINDINGS, COMMANDS, DATA_CONTRACT_CHECK, and PLAN_CONGRUENCY.
```

**Gate:** The same canonical input must be usable by the CPU oracle and GPU path before GPU kernels are added.

## Phase 3: GPU Indicator And Batch Primitives

**Time budget:** 60-90 minutes.

**Purpose:** Port only batchable primitives first and establish CUDA availability, timing, memory, and CPU parity.

**Phase checklist:**

- [ ] Write CPU and optional CUDA primitive tests.
- [ ] Implement CuPy RawKernel primitives and CPU oracles.
- [ ] Record device, memory, warmup, transfer, and kernel evidence.
- [ ] Obtain a Luna `PASS`.

**Files:**
- Create: `f6_hybrid/gpu_primitives.py`
- Create: `f6_hybrid/cpu_primitives.py`
- Create: `test_f6_hybrid_gpu_primitives.py`
- Create: `artifacts/f6_hybrid/phase3_gpu_probe.json`

**Required interfaces:**

```python
gpu_available() -> bool
device_info() -> dict
stochastic_batch(high, low, close, k_values, d_values) -> array
atr_batch(high, low, close, periods) -> array
```

Use optional CuPy imports so CPU-only environments remain importable. Use CuPy `RawKernel` for custom CUDA code, float64 arrays, device-side persistence, and explicit synchronization only at result boundaries. Use Numba CPU JIT or NumPy for the CPU primitive oracle. Use CUDA events or `cupyx.profiler` for timings, include warmup, and record CuPy memory-pool usage.

**Required tests:**

- CPU primitive tests always run.
- CUDA tests skip cleanly only when no CUDA device is available.
- GPU stochastic and ATR outputs match CPU within declared tolerance and equal-NaN rules.
- Candidate batches produce stable parameter-major output.
- The probe records device, driver, CuPy version, dtype, batch size, warmup, kernel time, transfer time, and memory peak.

### Token-Cheap Implementation Prompt

```text
Apply Shared Prompt Rules. Execute Phase 3 only.

Write failing CPU/GPU primitive tests first. Implement optional CuPy RawKernel primitives for the approved F6 stochastic and ATR calculations, with CPU equivalents as the oracle. Do not implement full trade execution yet. Do not use Numba-CUDA as the new GPU stack. Keep arrays on-device until the primitive result is required, and include GPU-event timing rather than CPU wall time alone. If CUDA is unavailable, CPU tests must pass and the GPU probe must record a clear unavailable status.

Run:
python -m unittest -v test_f6_hybrid_gpu_primitives test_f6_hybrid_data test_f6_hybrid_contracts test_f6_hybrid_params test_f6_hybrid_cpu_oracle
python -m compileall -q f6_hybrid

Return STATUS, FILES, CUDA_STATUS, PARITY, BENCHMARK, TESTS, BLOCKERS, and NEXT.
```

### 5.6 Luna Verification Prompt

```text
You are 5.6 Luna. Verify Phase 3; do not implement fixes.

Inspect the GPU code for optional imports, dtype consistency, bounds checks, asynchronous timing, device synchronization, and host/device transfer placement. Run the full targeted suite. If CUDA is available, verify the actual parity assertions and probe output; if unavailable, verify that skip/fallback behavior is explicit and not reported as GPU success. Confirm the implementation uses CuPy RawKernel and does not introduce Numba-CUDA GPU dependencies. Return VERDICT, FINDINGS, COMMANDS, CUDA_EVIDENCE, PARITY_EVIDENCE, and PLAN_CONGRUENCY.
```

**Gate:** Primitive parity must pass before a GPU trade kernel can influence candidate ranking.

## Phase 4: Exact GPU F6 Candidate Execution

**Time budget:** 90-150 minutes for the first implementation; stop and report a blocker if exact parity cannot be achieved.

**Purpose:** Evaluate many candidate configurations concurrently while preserving the stateful F6 semantics.

**Phase checklist:**

- [ ] Write synthetic execution and parity tests before the kernel.
- [ ] Implement the exact stateful F6 batch execution path.
- [ ] Run synthetic and five-day parity without launching the full grid.
- [ ] Obtain a Luna `PASS` or record a concrete parity blocker.

**Files:**
- Create: `f6_hybrid/gpu_engine.py`
- Create: `test_f6_hybrid_gpu_engine.py`
- Modify only if required: `f6_hybrid/data.py`, `f6_hybrid/contracts.py`

**Required interface:**

```python
run_gpu_batch(dataset, candidates, batch_size, return_traces=False) -> list[CandidateSummary]
```

The kernel may assign one thread/configuration and loop sequentially over bars. It must implement the approved F6 behavior, including 1m/2m/3m/5m aggregation, S1/S2/S3/S4 state, bullish trough divergence, bearish peak reversal exit, `flag_nodiv`, super setup, pinbar vicinity breakout, embedded S4 reversal, ATR warmup/fallback, CE/PE offset selection, daily shutdown, consecutive-loss shutdown, same-bar stop-first precedence, and EOD exit. Any Phase 0-approved additions must be included.

The GPU returns compact summaries for screening and an optional trace for parity debugging. It must not apply a different strategy merely because the kernel is faster. The temporary execution prototype is not sufficient without these semantics; in particular, its external signal input, missing fees, missing consecutive-loss behavior, and zero-slippage wrapper are not acceptable as final behavior.

**Required tests:**

- Synthetic entry/stop/target/EOD fixture.
- Same-bar stop and target fixture.
- Daily reset and consecutive-loss fixture.
- Reversal and bearish-peak fixture.
- F6 smoke fixture against the CPU oracle.
- Known champion fixture against the CPU oracle.
- Exact trade count and reasons for debug traces; numeric values use declared float64 tolerance.

### Token-Cheap Implementation Prompt

```text
Apply Shared Prompt Rules. Execute Phase 4 only.

Write the synthetic execution and CPU-oracle parity tests first. Implement the smallest exact CuPy RawKernel batch executor that consumes CanonicalDataset and F6Candidate batches. Preserve all F6 semantics listed in the plan; do not substitute the simplified temporary kernel. Keep the CPU oracle unchanged. Return compact summaries for screening and optional full traces for debugging. If a required semantic cannot be represented exactly, stop with BLOCKED and document the missing state instead of implementing an approximation.

Run the targeted GPU engine tests, all prior tests, and py_compile. Run the five-day smoke parity only after the synthetic tests pass. Do not run the full grid.

Return STATUS, FILES, SEMANTICS_IMPLEMENTED, TESTS, CPU_GPU_DIFF, CUDA_STATUS, BLOCKERS, and NEXT.
```

### 5.6 Luna Verification Prompt

```text
You are 5.6 Luna. Verify Phase 4; do not implement fixes.

Compare the kernel state transitions against grid_optimize_f6_atr.TFTracker, MTFTracker, process_day, and the Phase 0 inventory. Run every synthetic test and the five-day parity test. Inspect same-bar precedence, daily resets, warmup, reversal, ATR fallback, consecutive losses, and option mapping. Reject any implementation that omits a listed behavior, returns only aggregate numbers without a trace path for debugging, or silently substitutes the temporary simplified kernel. Return VERDICT, ordered FINDINGS with paths/lines, COMMANDS, SEMANTICS_CHECKLIST, PARITY_RESULT, and PLAN_CONGRUENCY.
```

**Gate:** No GPU ranking is permitted until Luna confirms exact F6 semantic coverage and parity on smoke fixtures.

## Phase 5: CPU/GPU Parity And Performance Benchmark

**Time budget:** 45-90 minutes.

**Purpose:** Prove that the hybrid path is faster without changing results.

**Phase checklist:**

- [ ] Add independent parity comparisons and synchronized timing.
- [ ] Compare synthetic, smoke, and known-reference workloads.
- [ ] Record total wall time, transfers, memory, batch size, and speedup.
- [ ] Obtain a Luna `PASS`.

**Files:**
- Create: `f6_hybrid/parity.py`
- Create: `f6_hybrid/benchmark.py`
- Create: `test_f6_hybrid_parity.py`
- Create generated evidence: `artifacts/f6_hybrid/phase5_parity.json`
- Create generated evidence: `artifacts/f6_hybrid/phase5_benchmark.json`

**Required checks:**

- Compare synthetic traces.
- Compare the first five available trading days.
- Compare the known reference window for the established champion.
- Compare candidate summaries, trade counts, wins, points, rupees, and drawdown.
- Record first-run CUDA context and kernel compilation separately from warmed runs.
- Report CPU time, GPU kernel time, host/device transfer time, total wall time, batch size, device memory, and speedup.
- Benchmark with CUDA events or `cupyx.profiler`; do not use an unsynchronized CPU timer as GPU kernel time.

GPU is faster only if total wall time, including required transfers and setup, improves over the CPU baseline for the actual candidate batch size.

### Token-Cheap Implementation Prompt

```text
Apply Shared Prompt Rules. Execute Phase 5 only after Phase 4 PASS.

Add failing parity and benchmark assertions first. Implement independent comparison helpers and a benchmark that separates warmup, compile, transfer, kernel, and total wall time. Run synthetic, five-day, and known-reference comparisons. Do not weaken tolerances to force a pass. If the GPU is slower at the measured workload, record that honestly and keep the CPU fallback; do not optimize blindly.

Run:
python -m unittest -v test_f6_hybrid_parity test_f6_hybrid_gpu_engine test_f6_hybrid_gpu_primitives test_f6_hybrid_data test_f6_hybrid_contracts test_f6_hybrid_params test_f6_hybrid_cpu_oracle
python -m compileall -q f6_hybrid

Return STATUS, FILES, PARITY, TIMINGS, SPEEDUP, MEMORY, CUDA_STATUS, BLOCKERS, and NEXT.
```

### 5.6 Luna Verification Prompt

```text
You are 5.6 Luna. Verify Phase 5; do not implement fixes.

Run the complete Phase 5 suite and inspect the raw evidence JSON. Confirm that parity includes actual smoke/reference data, not only synthetic fixtures. Confirm GPU timings use CUDA events or cupyx.profiler and include transfer/setup costs in total wall time. Check that failed parity or a slower GPU run is reported as such rather than hidden. Return VERDICT, FINDINGS, COMMANDS, PARITY_MATRIX, PERFORMANCE_CHECK, and PLAN_CONGRUENCY.
```

**Gate:** If parity fails, the GPU path is screening-only and cannot select final candidates. If GPU is slower, retain it as optional but use the measured CPU path.

## Phase 6: Cost Model And Walk-Forward Coordinator

**Time budget:** 90-120 minutes.

**Purpose:** Apply realistic costs and perform chronological selection without holdout leakage.

**Phase checklist:**

- [ ] Write failing fee, drawdown, gate, and isolation tests.
- [ ] Implement immutable cost adjustment and walk-forward orchestration.
- [ ] Verify selection, OOS, and holdout candidate-list isolation.
- [ ] Obtain a Luna `PASS`.

**Files:**
- Create: `f6_hybrid/costs.py`
- Create: `f6_hybrid/walkforward.py`
- Create: `f6_hybrid/report.py`
- Create: `test_f6_hybrid_costs.py`
- Create: `test_f6_hybrid_walkforward.py`

**Required interfaces:**

```python
apply_costs(trades, slippage_pts, brokerage_per_order=0.0) -> list[dict]
net_stats(trades) -> dict
drawdown_stats(trades) -> dict
run_walkforward(candidates, folds, backend, config) -> dict
run_holdout(selected_candidates, holdout, backend, config) -> dict
```

Use the statutory fee formula from `backtest_walkforward_fees.py`: STT 0.0625% sell leg, exchange 0.035% both legs, SEBI 0.0001% both legs, stamp 0.003% buy leg, GST 18% on exchange plus SEBI, and zero brokerage unless configured. Keep raw trades immutable and evaluate slippage scenarios independently. The configured primary and sensitivity cases must be visible in the report.

The coordinator must use the current validation windows from the inventory, with explicit train/selection, OOS, and final holdout periods. It must generate candidates and rank only on pre-OOS data, pass unchanged candidates to OOS, and pass only frozen selected candidates to the final holdout. Tests must prove that holdout data cannot alter the candidate list.

Selection gates must include positive net P&L, net PF greater than 1.0, minimum trade counts, and the configured peak-to-trough daily drawdown cap. Robustness must include fold pass rate, median/worst OOS result, slippage sensitivity, and immediate parameter-neighbor stability.

### Token-Cheap Implementation Prompt

```text
Apply Shared Prompt Rules. Execute Phase 6 only after Phase 5 PASS.

Write failing unit tests for fee arithmetic, copied trade dictionaries, daily drawdown dates, selection gates, fold isolation, and holdout candidate-list isolation. Implement costs and the walk-forward coordinator around the already-verified CPU/GPU interfaces. Do not run the full multi-year grid yet. Use small deterministic fixtures for the tests. Preserve the exact existing fee formula and make every date window and cost scenario explicit in config/output.

Run the Phase 6 tests, all prior tests, py_compile, and git diff --check. Return STATUS, FILES, TESTS, COST_CHECK, ISOLATION_CHECK, BLOCKERS, and NEXT.
```

### 5.6 Luna Verification Prompt

```text
You are 5.6 Luna. Verify Phase 6; do not implement fixes.

Inspect costs.py, walkforward.py, report.py, tests, and the Phase 0 date/config inventory. Run all tests and py_compile. Recalculate fee arithmetic independently from backtest_walkforward_fees.py. Verify raw trades are not mutated, drawdown is daily peak-to-trough with dates, selection never reads holdout results, and the holdout receives exactly the frozen candidate list. Reject any use of holdout metrics for ranking or parameter expansion. Return VERDICT, FINDINGS, COMMANDS, COST_RECALCULATION, HOLDOUT_ISOLATION, and PLAN_CONGRUENCY.
```

**Gate:** No full run until costs and holdout isolation pass independent verification.

## Phase 7: Runner, Smoke, Reference, And Full Results

**Time budget:** 60-90 minutes of implementation plus the actual compute runtime.

**Purpose:** Provide a reproducible CLI and generate the final F6 walk-forward report.

**Phase checklist:**

- [ ] Write CLI gate and output tests.
- [ ] Implement smoke, reference, benchmark, walk-forward, and holdout commands.
- [ ] Run the full job only after all gates pass.
- [ ] Obtain a Luna `PASS` on the final report.

**Files:**
- Create: `run_f6_hybrid.py`
- Create: `test_run_f6_hybrid.py`
- Create generated output: `artifacts/f6_hybrid/F6_HYBRID_REPORT.md`
- Create generated output: `artifacts/f6_hybrid/f6_hybrid_results.csv`
- Create generated output: `artifacts/f6_hybrid/f6_hybrid_benchmark.json`

**Required CLI:**

```text
python run_f6_hybrid.py --backend auto --smoke
python run_f6_hybrid.py --backend auto --reference
python run_f6_hybrid.py --backend auto --walk-forward
python run_f6_hybrid.py --backend auto --holdout
```

The CLI must refuse `--walk-forward` until the smoke and reference checks pass in the same run state. It must print the approved candidate count, backend, GPU device, batch size, date windows, cost scenarios, and whether any CPU fallback occurred. It must write a report that distinguishes CPU reference, GPU screening, selected, OOS, and holdout rows.

The full run sequence is:

1. Run five-day smoke and verify trade count, WR range, and P&L sanity.
2. Run the known reference candidate and compare expected metrics from the ledger/inventory.
3. Benchmark warmed CPU and CUDA paths.
4. Run the exhaustive approved grid on the configured training/selection windows.
5. Run only selected candidates on OOS windows.
6. Run only the frozen selected list on final holdout.
7. Generate the report and stop. Do not update the ledger or live configuration.

### Token-Cheap Implementation Prompt

```text
Apply Shared Prompt Rules. Execute Phase 7 only after Phase 6 PASS.

Write failing CLI tests for backend selection, smoke gating, reference gating, output paths, and holdout-only frozen candidates. Implement run_f6_hybrid.py with explicit backend, date, batch-size, worker, cost, and output arguments. Run smoke first and record its evidence. Run the known reference before any full walk-forward run. Do not launch the full exhaustive grid if either gate fails. If both pass, run the configured full job with a long timeout and preserve logs/results under artifacts/f6_hybrid/.

Run the CLI tests, `python -m unittest -v discover -s . -p "test_f6_hybrid*.py"`, `python -m compileall -q f6_hybrid`, and git diff --check. Return STATUS, FILES, SMOKE, REFERENCE, BENCHMARK, FULL_RUN_STATUS, RESULT_PATHS, BLOCKERS, and NEXT.
```

### 5.6 Luna Verification Prompt

```text
You are 5.6 Luna. Verify Phase 7; do not implement fixes.

Read the plan, all phase reports, CLI source, tests, logs, CSV, and final report. Re-run the CLI tests and inspect the smoke/reference evidence. Confirm the full job could not bypass smoke/reference gates, the candidate count matches the approved parameter inventory, the holdout rows contain only frozen candidates, and the report records backend, GPU device, precision, timing, costs, drawdown, and fallback status. Check that no ledger/live files were modified. Return VERDICT, FINDINGS with path/line references, COMMANDS, GATE_CHECK, HOLDOUT_CHECK, RESULT_CHECK, and PLAN_CONGRUENCY.
```

**Gate:** The implementation is complete only when Luna verifies the report and all required gates. A compute timeout is not a correctness pass; report it separately from implementation status.

## Final Result Requirements

The final report must answer all of these without relying on undocumented assumptions:

- What exact F6 parameters and fixed settings were tested?
- How many combinations were evaluated and how many were skipped, if any?
- Which backend ran each stage, on what GPU, with what precision and batch size?
- Did GPU and CPU agree on parity fixtures, smoke, and reference data?
- What were gross and net points, rupees, WR, PF, fees, trade count, and daily drawdown?
- Which candidates passed each walk-forward fold?
- What were the median and worst OOS results?
- How sensitive were results to slippage and one-step parameter neighbors?
- What happened on the untouched holdout?
- Was any candidate actually confirmed, or did all candidates fail?

No result may be presented as a live recommendation solely because it wins the in-sample or selection period.

## Handoff Format Between Workers

The cheap worker must return a short report with the exact keys requested by its phase prompt. Luna receives:

- This plan file.
- The phase-specific implementation prompt.
- The cheap worker report.
- `git status --short` and `git diff --stat`.
- Targeted test output and generated evidence paths.

Luna returns `PASS`, `FAIL`, or `BLOCKED`. A `FAIL` report must list concrete findings before any summary. The cheap worker may repair only those findings before the next verification cycle.

## Plan Self-Review

- Scope is F6-specific; no general strategy black box is included.
- CUDA is optional at import time but mandatory when explicitly requested.
- CPU remains the authoritative fallback and final audit path.
- Existing F6 semantics are enumerated instead of replaced by the temporary simplified kernel.
- Exhaustive count, cost model, smoke gate, reference gate, walk-forward isolation, and holdout isolation are explicit.
- Six-to-eight-hour work is bounded to the first verified hybrid slice; full compute runtime is reported separately.
- No live trading code or unrelated worktree changes are in scope.
