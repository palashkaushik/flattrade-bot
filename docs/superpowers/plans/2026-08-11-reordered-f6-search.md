# Reordered F6 Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use inline execution with the
> smoke-test checkpoint below. Steps use checkbox syntax for tracking.

**Goal:** Add a standalone, staged Optuna search harness that screens F6
candidates in batches with exact factorized execution and Hyperband pruning.

**Architecture:** Keep `grid_optimize_f6_atr.py` as the reference engine. Add
`artifacts/f6_hybrid/reordered_search.py` as an orchestration-only layer that
samples parameters, calls `run_factorized_candidates`, reports staged scores,
and writes durable results. Add pure unit tests for stage construction and
trade canonicalization, then use a five-day integration smoke to prove the
factorized path matches the reference before running a bounded search.

**Tech Stack:** Python 3.11 hermes environment, Optuna ask/tell API, NumPy/
existing multiprocessing factorized runner, `unittest`, JSON and CSV.

## Global Constraints

- Keep the existing eight-axis `SEARCH_SPACE` unchanged.
- Keep `s1_d=3`, fees, costs, and walk-forward gates outside this raw search.
- Use the factorized runner for candidate evaluation so shared feature state is reused within a batch.
- Preserve exact trade output and summary parity with the existing engine.
- Use the fixed eight-worker pool and never launch a long search before the five-day smoke test passes.
- Do not enable Numba or approximate event/fill logic.
- Do not modify the reference engine in `grid_optimize_f6_atr.py`.
- Do not commit changes unless the user explicitly requests a commit.

---

### Task 1: Add Failing Scheduler Unit Tests

**Files:**
- Create: `test_reordered_search.py`
- Test: `artifacts/f6_hybrid/reordered_search.py`

**Interfaces:**
- The tests import `build_stage_resources`, `canonical_trade`, and
  `params_from_trial` from `artifacts.f6_hybrid.reordered_search`.
- The implementation must expose these pure helpers without requiring market
  data or creating a process pool at import time.

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from artifacts.f6_hybrid.reordered_search import (
    build_stage_resources,
    canonical_trade,
    params_from_trial,
)


class ReorderedSearchUnitTests(unittest.TestCase):
    def test_stage_resources_are_unique_increasing_and_end_at_full_window(self):
        days = [f"2020-01-{index:02d}" for index in range(1, 81)]
        self.assertEqual(build_stage_resources(days), [5, 20, 60, 80])
        self.assertEqual(build_stage_resources(days[:3]), [3])

    def test_canonical_trade_keeps_reference_fields_and_order(self):
        trade = {
            "date": "2020-01-02", "entry_min": 698, "exit_min": 699,
            "side": "CE", "symbol": "NIFTY24JAN12150CE", "entry": 94.15,
            "exit": 0.15, "pts": -94.0, "rs": -6110.0, "sl_pts": 6.0,
            "tp_pts": 30.0, "reason": "BEARISH_PEAK_REVERSAL",
            "duration_min": 1, "tf": "1m", "extra": "ignored",
        }
        self.assertEqual(
            canonical_trade(trade),
            ("2020-01-02", 698, 699, "CE", "NIFTY24JAN12150CE",
             94.15, 0.15, -94.0, -6110.0, 6.0, 30.0,
             "BEARISH_PEAK_REVERSAL", 1, "1m"),
        )

    def test_params_from_trial_sets_fixed_s1_d(self):
        class Trial:
            def suggest_categorical(self, name, values):
                return values[0]

        params = params_from_trial(Trial())
        self.assertEqual(params["s1_d"], 3)
        self.assertEqual(set(params) - {"s1_d"}, {
            "s1_k", "s4_k", "atr_period", "atr_sl_mult", "atr_tp_mult",
            "f6_s4_thresh", "f6_s1_thresh", "consec_loss",
        })


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify they fail for the expected reason**

Run:

```powershell
python -m unittest test_reordered_search.py -v
```

Expected: import failure because
`artifacts/f6_hybrid/reordered_search.py` does not exist yet. Do not create the
implementation before observing this failure.

---

### Task 2: Implement Pure Helpers and Staged Batch Search

**Files:**
- Create: `artifacts/f6_hybrid/reordered_search.py`

**Interfaces:**
- `build_stage_resources(days: Sequence[str]) -> list[int]`: returns the unique
  increasing cumulative resource counts from 5, 20, 60, and `len(days)`,
  clamped to the available days.
- `canonical_trade(trade: dict) -> tuple`: returns the same 14 fields used by
  `test_incremental_f6_integration.py`.
- `params_from_trial(trial) -> dict`: samples every key in
  `grid.SEARCH_SPACE` and adds `s1_d=3`.
- `run_search(days, files, spot_all, n_trials, batch_size, workers, output_prefix)`:
  returns a JSON-serializable run summary and writes the CSV/JSON outputs.
- `main()` supports `--smoke`, `--trials`, `--batch-size`, `--workers`,
  `--start`, `--end`, and `--output-prefix`.

- [ ] **Step 1: Implement the pure helpers first**

```python
def build_stage_resources(days):
    available = len(days)
    resources = [min(limit, available) for limit in (5, 20, 60, available)]
    return list(dict.fromkeys(resource for resource in resources if resource > 0))


TRADE_FIELDS = (
    "date", "entry_min", "exit_min", "side", "symbol", "entry", "exit",
    "pts", "rs", "sl_pts", "tp_pts", "reason", "duration_min", "tf",
)


def canonical_trade(trade):
    return tuple(trade.get(field) for field in TRADE_FIELDS)


def params_from_trial(trial):
    params = {
        name: trial.suggest_categorical(name, values)
        for name, values in grid.SEARCH_SPACE.items()
    }
    params["s1_d"] = 3
    return params
```

- [ ] **Step 2: Run the unit tests and verify they pass**

Run:

```powershell
python -m unittest test_reordered_search.py -v
```

Expected: all three tests pass.

- [ ] **Step 3: Implement the staged ask/tell loop**

Use `optuna.samplers.TPESampler(multivariate=True, seed=42)` and
`optuna.pruners.HyperbandPruner(min_resource=resources[0],
max_resource=resources[-1], reduction_factor=3)`. For each batch, call
`study.ask()` and `params_from_trial()` once per candidate. For every resource,
evaluate the cumulative prefix `days[:resource]` in one call:

```python
trade_lists, base_builds, signal_builds = run_factorized_candidates(
    [params for _, params in active],
    list(days[:resource]), files, spot_all, workers=workers,
)
```

For each returned candidate, compute `stats_for(trades)` and
`composite_score(stats)`, call `trial.report(score, step=resource)`, and call
`trial.should_prune()` only before the final resource. Tell each trial exactly
once: `study.tell(trial, state=optuna.trial.TrialState.PRUNED)` for pruned
candidates, or `study.tell(trial, score)` after full fidelity. Carry only
survivors to the next resource.

- [ ] **Step 4: Add durable CSV/JSON output and explicit failure behavior**

The CSV header must be:

```text
run_id,trial,resource,state,score,trades,wr,net_rs,pf,elapsed_s,base_builds,signal_builds,s1_k,s4_k,atr_period,atr_sl_mult,atr_tp_mult,f6_s4_thresh,f6_s1_thresh,consec_loss
```

Write one row per trial/resource. JSON must include `run_id`, command
configuration, exact `days`, `resources`, `trial_count`, `completed_count`,
`pruned_count`, `stage_timings`, `best_full_fidelity`, and output paths. Let
candidate exceptions propagate after closing the pool; never turn failures into
numeric scores.

- [ ] **Step 5: Run unit tests, then compile the harness**

Run:

```powershell
python -m unittest test_reordered_search.py -v
python -m py_compile artifacts\f6_hybrid\reordered_search.py test_reordered_search.py
```

Expected: all tests pass and both files compile without warnings.

---

### Task 3: Run the Mandatory Five-Day Smoke Gate

**Files:**
- Modify: `artifacts/f6_hybrid/reordered_search.py` only if the smoke exposes a defect
- Create: `artifacts/f6_hybrid/reordered_search_smoke.json`
- Create: `artifacts/f6_hybrid/reordered_search_smoke.csv`

**Interfaces:**
- `--smoke` uses `grid.CHAMPION` plus one alternate execution candidate.
- It evaluates the first five available days with `workers=2` to keep the gate
  short, while the full harness defaults to the fixed eight workers.

- [ ] **Step 1: Implement the smoke comparison**

Compare the champion's factorized result against `grid.run_days` using
`canonical_trade` for every trade, and enforce:

```python
if canonical_factorized != canonical_reference:
    raise SystemExit("SMOKE FAIL: champion trade parity mismatch")
if not 15 <= len(factorized_trades) <= 40:
    raise SystemExit("SMOKE FAIL: suspicious champion trade count")
if len(results) != 2:
    raise SystemExit("SMOKE FAIL: candidate result dropped")
```

Record `parity_ok`, trade counts, resource list, and stage accounting in the
smoke JSON.

- [ ] **Step 2: Run the smoke test before any full search**

Run:

```powershell
python artifacts\f6_hybrid\reordered_search.py --smoke
```

Expected: `SMOKE TEST OK`, `parity_ok: true`, and a champion total between 15
and 40 trades. If it fails, fix the harness and rerun the smoke; do not start a
search.

---

### Task 4: Run a Bounded Reordered Search and Record the Measurement

**Files:**
- Create: `artifacts/f6_hybrid/reordered_search_*.csv`
- Create: `artifacts/f6_hybrid/reordered_search_*.json`
- Modify: `artifacts/f6_hybrid/optimization_methods.md`

**Interfaces:**
- Full run uses 2020-01-01 through 2022-12-31, the existing eight-axis space,
  `--trials 40`, `--batch-size 8`, and `--workers 8`.
- A full run is allowed only after Task 3 passes.

- [ ] **Step 1: Launch the bounded search**

Run:

```powershell
python artifacts\f6_hybrid\reordered_search.py --trials 40 --batch-size 8 --workers 8
```

Expected: the process completes with no candidate exceptions, each trial is
either `PRUNED` or `COMPLETE`, and JSON `completed_count + pruned_count` equals
40.

- [ ] **Step 2: Validate output accounting**

Run:

```powershell
python -c "import glob,json; p=sorted(glob.glob('artifacts/f6_hybrid/reordered_search_*.json'))[-1]; d=json.load(open(p)); assert d['trial_count']==40; assert d['completed_count']+d['pruned_count']==40; print(p, d['completed_count'], d['pruned_count'])"
```

Expected: one JSON path followed by counts summing to 40. Inspect the top
full-fidelity candidates against existing `optuna_results.csv`; do not treat
raw in-sample rank as evidence until walk-forward validation passes.

- [ ] **Step 3: Document the measured result**

Append a section to `artifacts/f6_hybrid/optimization_methods.md` containing
the command, stages, trial count, completed/pruned counts, wall time, feature
build totals, and whether the staged result preserved smoke parity. State
whether the method is retained for candidate screening or rejected if pruning
is too aggressive or adds no measured benefit.

---

### Task 5: Final Verification and Graph Refresh

**Files:**
- Modify: `graphify-out/GRAPH_REPORT.md`, `graphify-out/graph.json`, and related
  generated graph files through `run_graphify.py`.

- [ ] **Step 1: Run focused and regression verification**

```powershell
python -m unittest test_reordered_search.py test_raw_features_f6.py -v
python -m py_compile artifacts\f6_hybrid\reordered_search.py
```

- [ ] **Step 2: Refresh the project graph after Python changes**

```powershell
python run_graphify.py
```

- [ ] **Step 3: Inspect the final diff and status without reverting unrelated work**

```powershell
git diff -- artifacts/f6_hybrid/reordered_search.py test_reordered_search.py artifacts/f6_hybrid/optimization_methods.md docs/superpowers/specs/2026-08-11-reordered-f6-search-design.md docs/superpowers/plans/2026-08-11-reordered-f6-search.md
git status --short
```

Report the smoke parity, bounded-search accounting, measured pruning/timing,
and any residual walk-forward validation gap. Do not claim production readiness
from raw search results alone.
