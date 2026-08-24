# Realistic Strategy Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate established ATR/F6 strategy candidates on 2025 selection data, confirm unchanged winners on 2026 holdout data, and rank net return, win rate, and drawdown after realistic costs.

**Architecture:** Keep the existing `grid_optimize_f6_atr.py` engine as the signal and execution source of truth. Add a small pure-metrics/catalog module for candidate normalization, cost adjustment, drawdown, and chronological selection, plus a CLI runner that owns multiprocessing and report generation. The holdout phase receives only the selected 2025 candidates and never invokes candidate generation.

**Tech Stack:** Python 3.13, pandas, NumPy, multiprocessing, unittest, existing `grid_optimize_f6_atr.py` and `backtest_walkforward_fees.py`.

## Global Constraints

- Use the validated option files under `C:/Websites/ammu/nifty_options` and spot data under `C:/Websites/ammu/index/NIFTY 50_minute.csv`.
- Selection period is `2025-01-01` through `2025-12-31`.
- Confirmation holdout is `2026-01-01` through `2026-05-05`.
- Primary slippage is 1.0 point per side; sensitivity cases are 0.5 and 2.0 points per side.
- Apply STT 0.0625%, exchange 0.035%, SEBI 0.0001%, stamp 0.003%, GST 18%, and Rs 0 brokerage unless overridden.
- The maximum allowed peak-to-trough daily net drawdown is Rs 100,000.
- Require at least 300 selection-period trades and 50 holdout trades.
- Do not optimize or select candidates using holdout results.
- Do not modify live trading parameters automatically.
- Run the mandatory five-day smoke test before the full sweep.
- Do not commit changes unless the user explicitly requests a commit.

---

### Task 1: Add Pure Sweep Metrics And Candidate Catalog

**Files:**
- Create: `strategy_sweep.py`
- Test: `test_strategy_sweep.py`

**Interfaces:**
- `select_days(files: dict[str, str], spot: dict[str, object], start: str, end: str) -> list[str]`
- `normalize_candidate(values: dict[str, object]) -> dict[str, object]`
- `load_base_candidates(results_csv: str) -> list[dict[str, object]]`
- `expand_neighbors(candidates: list[dict[str, object]], limit: int = 500) -> list[dict[str, object]]`
- `trade_cost(entry_px: float, exit_px: float, slippage_pts: float, brokerage_per_order: float = 0.0) -> float`
- `apply_costs(trades: list[dict], slippage_pts: float, brokerage_per_order: float = 0.0) -> list[dict]`
- `net_stats(trades: list[dict]) -> dict[str, float]`
- `drawdown_stats(trades: list[dict]) -> dict[str, object]`
- `passes_selection(metrics: dict[str, object]) -> bool`

- [ ] **Step 1: Write failing unit tests for date intersection, cost arithmetic, and drawdown.**

```python
import unittest

from strategy_sweep import apply_costs, drawdown_stats, select_days


class StrategySweepTests(unittest.TestCase):
    def test_select_days_requires_option_and_spot_data(self):
        files = {"2025-01-01": "a", "2025-01-02": "b"}
        spot = {"2025-01-02": object(), "2025-01-03": object()}
        self.assertEqual(select_days(files, spot, "2025-01-01", "2025-01-03"), ["2025-01-02"])

    def test_apply_costs_subtracts_two_sides_of_slippage_and_fees(self):
        trades = [{"entry": 100.0, "exit": 110.0, "pts": 10.0}]
        adjusted = apply_costs(trades, slippage_pts=1.0)
        self.assertEqual(adjusted[0]["pts_net"], 8.0)
        self.assertGreater(adjusted[0]["fee"], 0.0)

    def test_drawdown_uses_daily_net_equity(self):
        trades = [
            {"date": "2025-01-01", "rs_net": 100.0, "pts_net": 1.0},
            {"date": "2025-01-02", "rs_net": -40.0, "pts_net": -0.4},
            {"date": "2025-01-03", "rs_net": -80.0, "pts_net": -0.8},
        ]
        result = drawdown_stats(trades)
        self.assertEqual(result["max_drawdown_rs"], 120.0)
        self.assertEqual(result["start"], "2025-01-01")
        self.assertEqual(result["end"], "2025-01-03")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new test before implementation.**

Run: `python -m unittest -v test_strategy_sweep`

Expected: FAIL with `ModuleNotFoundError: No module named 'strategy_sweep'`.

- [ ] **Step 3: Implement the pure metric functions.**

Use the existing fee constants from `backtest_walkforward_fees.py` and preserve
the existing fee formula. `apply_costs` must return copied trade dictionaries,
not mutate the engine output, so the same gross trades can be evaluated at
0.5, 1.0, and 2.0 points per side.

Use this drawdown algorithm:

```python
def drawdown_stats(trades):
    daily = {}
    for trade in trades:
        daily.setdefault(trade["date"], 0.0)
        daily[trade["date"]] += float(trade["rs_net"])

    equity = 0.0
    peak = 0.0
    peak_day = None
    result = {"max_drawdown_rs": 0.0, "start": None, "end": None}
    for day in sorted(daily):
        if peak_day is None:
            peak_day = day
        equity += daily[day]
        if equity > peak:
            peak = equity
            peak_day = day
        drawdown = peak - equity
        if drawdown > result["max_drawdown_rs"]:
            result = {"max_drawdown_rs": round(drawdown, 2),
                      "start": peak_day, "end": day}
    result["max_drawdown_pts"] = round(result["max_drawdown_rs"] / 65.0, 2)
    return result
```

Candidate loading must read only `pruned == 0` rows from `optuna_results.csv`,
deduplicate on the eight parameter axes, add the current Section 16 candidate
and the legacy grid champion, and always set `s1_d` to 3. Neighbor expansion
must vary one axis at a time using `grid_optimize_f6_atr.SEARCH_SPACE`, dedupe,
and stop at 500 candidates.

- [ ] **Step 4: Run the unit tests and compile the module.**

Run: `python -m unittest -v test_strategy_sweep`

Expected: all three tests PASS.

Run: `python -m py_compile strategy_sweep.py test_strategy_sweep.py`

Expected: exit code 0 with no output.

### Task 2: Add Chronological Sweep Runner

**Files:**
- Create: `run_strategy_sweep.py`
- Modify: `strategy_sweep.py` only if Task 1 interfaces need runner support

**Interfaces:**
- CLI command: `python run_strategy_sweep.py --selection-start 2025-01-01 --selection-end 2025-12-31 --holdout-start 2026-01-01 --holdout-end 2026-05-05`
- `run_candidates(candidates, days, files, spot, workers) -> list[dict]`
- `evaluate_candidate(trades, slippages, brokerage) -> list[dict]`
- `rank_selection(results) -> dict[str, list[dict]]`
- `confirm_holdout(selected, results) -> list[dict]`

- [ ] **Step 1: Write failing tests for selection gates and holdout isolation.**

```python
import unittest

from strategy_sweep import passes_selection


class StrategyGateTests(unittest.TestCase):
    def test_selection_requires_positive_net_pf_drawdown_and_trade_count(self):
        base = {
            "net_rs": 1000.0,
            "pf": 1.1,
            "max_drawdown_rs": 99999.0,
            "trades": 300,
        }
        self.assertTrue(passes_selection(base))
        self.assertFalse(passes_selection({**base, "max_drawdown_rs": 100001.0}))
        self.assertFalse(passes_selection({**base, "trades": 299}))


if __name__ == "__main__":
    unittest.main()
```

Use the repository's standard `unittest.TestCase` style because pytest is not
installed in this workspace.

- [ ] **Step 2: Run the test and verify it fails for the missing gate helper.**

Run: `python -m unittest -v test_strategy_sweep`

Expected: FAIL because `passes_selection` does not yet implement the gate.

- [ ] **Step 3: Implement the runner.**

The runner must:

1. Load spot data and option files.
2. Select 2025 and 2026 dates independently with `select_days`.
3. Load the base candidate catalog and evaluate every base candidate only on
   2025 using one persistent multiprocessing pool initialized with
   `grid_optimize_f6_atr.init_worker_local`.
4. Evaluate the optional neighbor expansion only on 2025.
5. Rank candidates at the primary 1.0-point slippage case.
6. Select the top passing candidates by maximum net P&L, maximum net win rate,
   and best `net_rs / max_drawdown_rs` ratio.
7. Run exactly those selected candidates on 2026 without generating neighbors
   or reading 2026 metrics before selection.
8. Evaluate all selected candidates at 0.5, 1.0, and 2.0 points per side.
9. Write `strategy_sweep_results.csv` with stage, period, slippage, all
   parameters, trades, gross/net points, net P&L, net WR, PF, fees, drawdown,
   and drawdown dates.
10. Print separate selection ranking, holdout confirmation, and sensitivity
    tables. If no candidate passes holdout, print `NO STRATEGY CONFIRMED`.

Use `grid.run_days(pool, params, days, files, spot)` for engine execution so
the established signal and exit implementation remains unchanged.

- [ ] **Step 4: Run unit tests and compile the runner.**

Run: `python -m unittest -v test_strategy_sweep`

Expected: all tests PASS.

Run: `python -m py_compile strategy_sweep.py run_strategy_sweep.py`

Expected: exit code 0 with no output.

### Task 3: Smoke Test And Reference Verification

**Files:**
- No source changes unless a test exposes an implementation defect.
- Generated: `strategy_sweep_results.csv`

- [ ] **Step 1: Verify the historical Section 16 reproduction before the sweep.**

Run the existing exact-grid verification for the Section 16 candidate on
`2020-01-01` through `2024-10-31` and confirm:

```text
days 1203
trades 6398
wr approximately 50.875
rs 1659198
pf approximately 1.832
```

- [ ] **Step 2: Run the five-day smoke test.**

Run: `python run_strategy_sweep.py --smoke --selection-start 2025-01-06 --selection-end 2025-01-10`

Expected: five dates, 15-40 trades, no empty candidate list, and no holdout
evaluation.

- [ ] **Step 3: Run the complete bounded sweep.**

Run: `python run_strategy_sweep.py --selection-start 2025-01-01 --selection-end 2025-12-31 --holdout-start 2026-01-01 --holdout-end 2026-05-05 --primary-slippage 1.0 --sensitivity-slippage 0.5 --stress-slippage 2.0 --drawdown-cap 100000`

Expected: a selection table, a holdout table containing only 2025-selected
candidates, and three slippage scenarios for each reported winner.

- [ ] **Step 4: Verify the result file and ranking gates.**

Run: `python -c "import pandas as pd; df=pd.read_csv('strategy_sweep_results.csv'); assert {'selection','holdout'} <= set(df['stage']); assert df['max_drawdown_rs'].notna().all(); print(len(df), 'verified rows')"`

Expected: exit code 0; all rows contain drawdown metrics and both stages exist.

### Task 4: Report The Recommendation

**Files:**
- Modify: `BACKTEST_LEDGER.md` only after the sweep has completed and the
  user has reviewed the result.

- [ ] **Step 1: Extract the three requested winners.**

Report the maximum-net, maximum-net-win-rate, and best-balanced candidates from
the confirmed holdout set. Include exact parameters, selection metrics,
holdout metrics, primary/stress slippage results, fees, max drawdown, and
drawdown dates.

- [ ] **Step 2: State the confirmation outcome.**

If no candidate passes the 2026 gates, explicitly state that no strategy was
confirmed and do not recommend live deployment. If one or more pass, recommend
only the balanced confirmed candidate for paper soak, not automatic live use.

- [ ] **Step 3: Run the final verification commands.**

Run: `python -m unittest -v test_strategy_sweep`

Run: `python -m py_compile strategy_sweep.py run_strategy_sweep.py test_strategy_sweep.py`

Run: `git diff --check`

Review `git status --short` and leave unrelated worktree changes untouched.
