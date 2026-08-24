# General Strategy Research And Walk-Forward Black Box

## Objective

Build a reusable black box that accepts a natural-language trading strategy,
researches and clarifies its NIFTY option-buying interpretation, translates it
into a versioned execution specification, exhaustively evaluates its declared
finite parameter space, and identifies robust candidates through chronological
walk-forward validation.

The system must distinguish the research/compiler layer from the deterministic
backtest layer. An agent may interpret, research, and ask questions, but it may
not silently invent rules, execute arbitrary generated code, or select a final
strategy from holdout data.

## Design Principles

- Ambiguity becomes a questionnaire item, not an undocumented assumption.
- Every external research claim is cited with its URL, retrieval date, and evidence quality.
- Every executed strategy has a canonical JSON representation and SHA-256 hash.
- Strategy-specific behavior lives behind an approved adapter contract.
- The kernel exhaustively evaluates finite declared domains; it never silently samples.
- Training and selection data are isolated from chronological OOS confirmation data.
- Costs, fills, drawdown, trade sufficiency, and missing-data behavior are explicit.
- No result automatically changes live trading configuration.

## End-To-End Flow

1. The user submits a natural-language strategy and optional files or links.
2. The compiler extracts a provisional model: signals, filters, timeframes,
   entries, option selection, exits, risk, execution, and parameter axes.
3. The research layer may use any public source and the repository to resolve
   NIFTY option mechanics. It records citations and confidence for each claim.
4. The compiler generates one complete questionnaire grouped by signal logic,
   timeframe, contract selection, entry, exit, risk, execution, costs, data,
   and parameter domains. Each item includes why it affects the result,
   researched choices, and a recommended default.
5. The user answers the questionnaire. Required unanswered or contradictory
   items produce `needs_clarification`; the system does not backtest yet.
6. The compiler emits a `StrategySpec`, displays its interpretation and exact
   parameter grid, and waits for user approval.
7. The approved spec is handed to an allowlisted strategy adapter.
8. The walk-forward kernel evaluates the complete finite grid on each training
   or selection window, ranks candidates, and runs only unchanged selected
   candidates on the following OOS window.
9. A final confirmation holdout is run only after all choices are frozen.
10. The report includes the spec hash, questionnaire answers, sources,
    assumptions, candidate counts, fold results, sensitivity, rejection
    reasons, and confirmed recommendations.

## StrategySpec Contract

The compiler output is a provider-neutral JSON document. Its required sections
are:

- `schema_version`: versioned schema identifier.
- `strategy_id` and `source_text`: stable name and original user description.
- `questionnaire`: questions, answers, unresolved items, and answer sources.
- `research_sources`: URL, title, retrieval date, claim, and evidence quality.
- `assumptions`: explicit defaults accepted by the user.
- `market`: underlying, instrument type, timezone, lot size, strike step, and
  allowed option sides.
- `data`: spot and option fields, minimum resolution, date coverage, and
  missing-data policy.
- `signal`: indicators, conditions, filters, timeframe relationships, and
  trigger semantics.
- `option_selection`: expiry rule, strike rule, CE/PE mapping, selection time,
  rollover behavior, and contract availability requirements.
- `entry`, `exit`, and `risk`: fill timing, same-bar precedence, stop/target/
  trailing rules, daily limits, consecutive-loss limits, and position limits.
- `execution`: session boundaries, slippage scenarios, brokerage, statutory
  charges, and order assumptions.
- `parameters`: typed finite domains. A range must be normalized to explicit
  values with a documented step before execution.
- `validation`: training windows, OOS folds, final holdout, minimum trades,
  drawdown cap, and robustness requirements.
- `adapter`: allowlisted adapter identifier and capability version.

The canonical serialization sorts keys and uses stable numeric formatting.
The hash covers the full spec, including questionnaire answers, assumptions,
parameter values, and validation settings.

## Questionnaire Rules

The questionnaire is generated after preliminary extraction and research. It
must cover every result-changing decision, including:

- Indicator definitions, lookbacks, source series, thresholds, and confirmation.
- Timeframe construction, candle alignment, warmup, and signal timing.
- NIFTY option expiry, strike distance, moneyness, CE/PE mapping, and selection
  time relative to the signal.
- Entry price, fill model, spread/slippage, same-bar behavior, and re-entry.
- Stop-loss, target, trailing, reversal, end-of-day, and exit precedence.
- Daily loss/profit limits, consecutive-loss shutdown, position count, and lot
  sizing for the one-lot evaluation.
- Parameter domains, grid steps, constraints, and whether each axis is fixed or
  searched.
- Required data, unavailable periods, contract continuity, and fee assumptions.

Each question has a stable identifier. The answer record stores the selected
option or free text, user confirmation status, and the source of any proposed
default. The agent asks the complete questionnaire in one batch, then may
regenerate a smaller follow-up only when an answer creates a contradiction.

## Research And NIFTY Adaptation

Public web sources are allowed. The compiler must prefer primary exchange,
regulatory, broker, and data-provider documentation when available, but it may
use secondary research when clearly labeled. It must separate sourced facts
from strategy assumptions and show low-confidence claims before asking for
confirmation.

NIFTY option-buying validation must explicitly check contract availability,
expiry and strike selection, option side, lot size, market hours, timestamp
alignment, premium data fields, fees, slippage, and whether the requested rule
can be represented without look-ahead. A strategy that requires unavailable
data or unsupported mechanics is rejected with a concrete explanation and a
possible questionnaire alternative.

## Adapter Contract

The backtest core depends on an approved adapter interface rather than a
particular strategy module. The initial interface is conceptually:

```python
class StrategyAdapter(Protocol):
    adapter_id: str

    def validate(self, spec: StrategySpec) -> list[str]: ...
    def parameter_space(self, spec: StrategySpec) -> ParameterSpace: ...
    def run(self, params: dict[str, object], dates: list[str],
            data: MarketData) -> list[Trade]: ...
```

Adapters expose capabilities and normalize their output to one immutable trade
schema. The first adapter may wrap the existing ATR/F6 implementation because
it is a proven reference engine, but the kernel must not import its constants
or parameter names directly. Later adapters can cover trailing, S1 turn-up,
fixed exits, or other approved strategy families.

Agent-generated code is not executed directly. New adapter code requires the
same validation, smoke test, parity test, and allowlisting as any other engine
implementation.

## Compute Backends And GPU Acceleration

The black box has a separate compute-backend contract so strategy semantics do
not depend on whether a run uses CPU or GPU:

```python
class ComputeBackend(Protocol):
    backend_id: str

    def available(self) -> bool: ...
    def execute(self, adapter: StrategyAdapter, candidates: list[dict[str, object]],
                dates: list[str], data: MarketData) -> list[CandidateResult]: ...
```

The CPU reference backend is the correctness oracle. It may use multiprocessing
and Numba for preprocessing, but its outputs are normalized through the same
trade and metric contracts as every other backend.

The first accelerated backend targets CUDA-capable machines and batches many
parameter configurations over normalized OHLCV/option arrays. The temporary
sweep work already provides reusable precedents: a CuPy device probe, a
float64 CuPy batch executor, and Numba array-aggregation primitives with parity
tests. Those experiments must be ported into the actual repository only after
they pass the generic adapter contract; they are not treated as production
code merely because they exist in the temporary copy.

GPU execution requirements are:

- Preprocess and transfer each fold's market data once where possible.
- Batch candidate configurations within an explicit device-memory budget.
- Use float64 for price and P&L calculations unless a parity test proves a
  lower precision is safe for a specific kernel.
- Return deterministic candidate results in the same order as the CPU backend.
- Compare GPU output with the CPU oracle on synthetic fixtures, the five-day
  smoke window, and the known reference window before accepting a result.
- Record device name, backend version, precision, batch size, elapsed time,
  memory behavior, and parity status in the run report.
- `backend=cuda` fails clearly when CUDA is unavailable; `backend=auto` may
  fall back to CPU only when the report records that fallback.

Not every strategy is GPU-compatible on its first implementation. Adapters
declare capabilities. Stateful or irregular rules can run on the CPU reference
backend while vectorizable candidate evaluation uses CUDA. This preserves one
black-box interface without pretending that every Python strategy can be
translated into a safe GPU kernel automatically.

## Exhaustive Search And Walk-Forward Kernel

The kernel expands the adapter's finite parameter domains into a deterministic
ordered Cartesian product. It records total combinations, completed
combinations, and any explicit user-approved cap. If the grid is too large for
the configured compute budget, execution pauses for a decision rather than
claiming an exhaustive result.

For each chronological fold:

1. Build the data intersection using only dates with both spot and option data.
2. Evaluate all permitted candidates on the training or selection window.
3. Apply primary costs and selection gates.
4. Rank by a Pareto view: maximum net result, maximum net win rate, and best
   net-return-to-drawdown ratio with PF and stability tie-breakers.
5. Run the selected candidates unchanged on the next OOS window.
6. Record fold-level and aggregate results without changing parameters from OOS
   performance.

The final confirmation period is never used to generate candidates, expand
neighbors, choose parameters, or tune thresholds. It receives only the frozen
candidate list and reports confirmation or failure.

## Metrics And Robustness

Every candidate and cost scenario reports trades, gross/net points, gross/net
rupees, win rate, profit factor, fees, peak-to-trough daily drawdown in rupees
and points, drawdown start/end dates, and missing-data counts.

Robustness includes:

- Number and percentage of OOS folds passed.
- Median, worst, and dispersion of OOS net result and PF.
- Minimum OOS trade count and maximum OOS drawdown.
- Slippage and brokerage sensitivity.
- Immediate one-step parameter-neighborhood results.
- Stability of the selected candidate across folds instead of only its best fold.

The report must distinguish exploratory, selected, OOS, and final-confirmation
rows. If no candidate passes the confirmation gates, it must state that no
strategy was confirmed and must not recommend live deployment.

## Safety And Auditability

- Natural-language strategy text, files, and web pages are untrusted inputs.
- The compiler emits data, not executable code.
- Adapters are allowlisted and run in a controlled process with explicit data
  and resource boundaries.
- Every run stores the spec hash, adapter version, source list, data coverage,
  parameter grid, command/configuration, and software version.
- Research claims that cannot be verified remain assumptions or block execution.
- The system never changes live parameters automatically.

## Implementation Sequence

1. Add the provider-neutral schema, questionnaire result model, canonical hash,
   validation errors, and normalized trade/data contracts.
2. Add the adapter protocol and deterministic grid expansion with unit tests.
3. Add the CPU reference backend, GPU backend contract, and deterministic
   candidate batching with CPU/GPU parity fixtures.
4. Add the generic cost, metrics, drawdown, and walk-forward kernel with strict
   train/OOS/holdout isolation.
5. Wrap the existing ATR/F6 engine as the first adapter and run CPU/GPU parity
   plus mandatory five-day smoke tests.
6. Add the research/compiler integration as a replaceable agent-facing
   boundary; keep the core usable with a hand-authored spec.
7. Add further strategy adapters only after each passes the same contract tests.
8. Produce the reproducible report and review it before any ledger or live-use
   recommendation.

## Acceptance Criteria

- A free-form strategy produces a complete questionnaire before execution.
- Ambiguous or unsupported strategies cannot enter the backtest kernel.
- The same approved spec produces the same hash, grid order, and metrics.
- Finite domains are fully enumerated or an explicit cap is reported.
- CPU and CUDA backends agree within declared tolerances on fixtures, smoke,
  and the known reference window, or CUDA is rejected for that adapter.
- GPU runs report device, precision, batch size, timing, and fallback status.
- The ATR/F6 adapter reproduces its known reference smoke and parity results.
- No candidate selection reads final holdout results.
- Walk-forward reports include costs, drawdown dates, robustness, and rejection
  reasons.
- No live configuration changes occur automatically.

## Non-Goals

- No unrestricted natural-language-to-Python execution.
- No hidden optimizer, random sampling, or unreported pruning.
- No direct final-holdout optimization.
- No claim that a backtest result is a live trading recommendation without a
  separate paper-soak decision.
