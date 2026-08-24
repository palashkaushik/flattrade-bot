# Phase 0 — F6 Parameter Inventory (2026-08-10)

Source of truth: `grid_optimize_f6_atr.py` (CPU reference engine, 600 lines).
Comparisons: `backtest_monthly_ramp.py`, `backtest_walkforward_fees.py`, `backtest_walkforward_refit.py`, `BACKTEST_LEDGER.md`, temp GPU prototypes (read-only).

## 1. Engine constants (grid_optimize_f6_atr.py)

| Constant | Value | Line |
|:---|:---|:---|
| LOT_SIZE | 65 | 56 |
| CE_OFFSET / PE_OFFSET | -100 / +100 | 57 |
| SESSION_START / END / DAY_LAST | 560 / 900 / 930 | 58 |
| DAILY_LOSS_RS | -2000.0 | 59 |
| DAILY_PROFIT_PTS | inf | 60 |
| MAX_CACHE_ENTRIES | 1600 | 87 |
| WORKERS | max(2, 85% cpu) | 89 |

TF_SPECS (lines 62-67): `agg, lookback(lb), tf_sl, tf_tp`
- 1m: (1, 10, 6.0, 30.0)
- 2m: (2, 5, 10.0, 15.0)
- 3m: (3, 4, 8.0, 25.0)
- 5m: (5, 3, 10.0, 35.0)

## 2. Search axes (SEARCH_SPACE, lines 76-85) — 15,552 combos (4·3·3·4·4·3·3·3)

| Axis | Values | Type |
|:---|:---|:---|
| s1_k | [7, 9, 12, 14] | fast stoch %K |
| s4_k | [50, 60, 75] | slow stoch %K |
| atr_period | [10, 14, 20] | ATR lookback |
| atr_sl_mult | [1.5, 2.0, 2.5, 3.0] | ATR SL mult |
| atr_tp_mult | [3.0, 4.0, 5.0, 6.0] | ATR TP mult |
| f6_s4_thresh | [75.0, 79.5, 85.0] | F6 flag S4 >= |
| f6_s1_thresh | [15.0, 20.5, 25.0] | F6 flag S1 <= |
| consec_loss | [4, 6, 8] | consecutive-loss shutdown |

HARD-FIXED (not axes, must stay fixed): s1_d=3 (line 474), s2=(14,3), s3=(40,4), s4_d=10 (lines 154-156), lookbacks per TF, tf_sl/tf_tp per TF, offsets ±100, session 560-930, daily loss -2000 Rs, lot 65, ATM = round(spot/50)*50, strike window atm-250..atm+300 step 50, super threshold 20.5 (hardcoded line 191), emb: s4<=20 count > 25 (lines 186-187).

## 3. Candidate parameter sets in repo

| Set | File:line | Values |
|:---|:---|:---|
| CHAMPION | grid_optimize_f6_atr.py:69-73 | s1_k=9, s1_d=3, s4_k=60, atr_period=14, sl=2.0, tp=4.0, f6s4=79.5, f6s1=20.5, cl=6 |
| CHAMPION_EXPECTED | grid_optimize_f6_atr.py:74 | trades=7,843, wr=48.0, rs=1,030,642, pf=1.45 (full 5Y) |
| CHAMPION_PARAMS | backtest_monthly_ramp.py:47-50 | s1_k=12, s1_d=3, s4_k=50, atr_period=10, sl=3.0, tp=6.0, f6s4=79.5, f6s1=25.0, cl=8 (Section-16 winner) |
| refit_results/summary.json | per-window refit | s1_k in {7, 9, 12}, cl in {6, 8} across windows |

## 4. Signal → trade semantics to port EXACTLY on GPU

- **Warmup:** previous trading day file is fully pushed through all MTFTrackers before current-day bars (`fprev` handling, lines 276-286). First day of a run has no warmup.
- **Trigger sources:** (a) TFTracker: setup requires (is_flag OR is_super) AND bullish trough divergence; fire via `BullishPinBarDetector.check_vicinity_breakout(hist, lb)`, setup consumed on fire (lines 188-200); (b) FlagNoDivScanner: no divergence, fires once per flag zone, resets when flag clears (lines 204-223). Both run per-TF; 1m bar aggregation: agg bars from 1m buffer, Candle = open of first, high=max, low=min, close=last, minute=last (lines 233-247).
- **Entry:** fill at trigger-bar CLOSE of the option's 1m slice at signal minute (bslice bar[3], line 405). Strike must be current ATM at signal minute (`ainfo` uses latest_spot at that minute, atm=round(spot/50)*50, strike = atm ± 100; entry rejected if slice missing, lines 391-402).
- **Reversal (is_rev):** entry on the OPPOSITE side strike, same ATM±100 rule (lines 393-399).
- **SL/TP:** if atr_val > 0.5 → ATR-derived (atr×mult), else TF fallback (tf_sl/tf_tp from TF_SPECS) (lines 406-411). Entry SL ep-sl_use, TP ep+tp_use.
- **Exit precedence (same-bar):** h>=tgt AND l<=sl → SL wins (line 352-356); else TP if h>=tgt; else SL if l<=sl; else 1m bearish peak divergence → exit at CLOSE (lines 358-364).
- **Daily shutdown:** open P&L check `dpnl*LOT + (c-entry)*LOT <= -2000` closes at close with reason SHUTDOWN_LOSS (lines 339-349); after any closed loss, closs increments, closs>=consec_loss OR dpnl<=-2000 → shut=True, no more entries today (lines 373-375).
- **EOD:** minute>=900 closes at last_px (tracked close) reason EOD, then day loop breaks (lines 377-386).
- **Same-minute ordering:** pmtrig per minute iterated in symbol/insertion order; first eligible fills, `break` after entry; pos is None guard (lines 390-415).
- **Fees/slippage NOT in this engine** — raw points; costs applied by walkforward scripts post-hoc (see section 6).

## 5. Baseline commands (run 2026-08-10)

- `python -m py_compile grid_optimize_f6_atr.py backtest_5y_optimized.py backtest_walkforward_fees.py backtest_monthly_ramp.py validate_top_candidates.py` → EXIT=0 (all compile)
- `python grid_optimize_f6_atr.py --smoke` → **Trades: 25 | WR: 44.0% | Net Rs: +3,578 | PF: 1.70** → SMOKE TEST OK (5 days 2020-01-01..2020-01-07, champion params)
- CuPy: **NOT INSTALLED** · Numba: **NOT INSTALLED** → GPU phases (3+) must `pip install cupy-cuda11x numba` (verify driver 610.74 → CUDA 12.x era driver; cupy-cuda12x candidate; confirm via `nvidia-smi` topo before install)
- nvidia-smi: `NVIDIA GeForce RTX 3060, 610.74, 12288 MiB`

## 6. Cost model source of truth (backtest_walkforward_fees.py:60-97)

SLIPPAGE_PTS=2.0/side; BROKERAGE=0; STT 0.0625% sell leg; EXCHANGE 0.035% both; SEBI 0.0001% both; STAMP 0.003% buy leg; GST 18% on exchange+SEBI. `trade_cost(entry_px, exit_px, brokerage)` → Rs deducted. Monthly ramp: lots = max(1, floor(equity_month_start/40,000)); daily stop on raw points (-2000 Rs grid; backtest_monthly_ramp uses 30 pts & cl=8). WF windows: (2020-22→2023), (2021-23→2024), (2020-21→2022 pseudo-OOS).

## 7. Walk-forward precedent

`backtest_walkforward_refit.py` already exists: per-window re-optimization harness (writes refit_results/summary.json, refits 8 axes). Phase 6 must reuse/extend its window scheme; verify it matches the plan's sequential walk-forward semantics.

## 8. Overfitting evidence (for Phase 6 baseline)

- 15,552 configs, selected AND measured on 2020-2024 → winner's curse; Section-16 best (+1,659,198 Rs, 6,398 trades, 50.9%, PF 1.83) is the max of the sample, not evidence of edge.
- `phase2.log` sensitivity: s1_k ±5-9%, consec_loss -0.2% (7.9%→6, 8→6 deltas), f6_s4_thresh=79.5 non-round optimum.
- Honest tests: Phase 6 sequential walk-forward + untouched 2026 holdout (2026-01-01..2026-05-05 window as per earlier plan; confirm in Phase 6).

## 9. Proposed additional axes (recorded, NOT approved for search)

Candidate axes if exhaustiveness ever widens: s1_d (currently 3), super threshold 20.5, emb threshold (20 / count>25), s2/s3 periods, strike window, ATM rounding (50/100), session bounds. Do NOT add to the hybrid grid without explicit user decision — current scope is the 8 axes = 15,552.