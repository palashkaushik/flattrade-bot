# Deep Research — Faster & Accurate F6 Parameter Search (2026-08-10)

Three parallel web-research threads (CPU acceleration / GPU acceleration / search+validation).
Sources: marketmaker.cc speed ladder + framework-tax paper, numba.readthedocs.io, pythonspeed.com,
JMLR Bergstra & Bengio 2012, Bailey & López de Prado (DSR, PBO/SSCV), Optuna docs (Hyperband,
RDB warm-start), CuPy docs, NVIDIA MC sample + numba-cuda deprecation, Springer 2025, gpu-backtester,
VectorAlpha, QuantStart.

## 1. CPU acceleration — the biggest win is the cheapest

- Event-driven engines pay a measurable "framework tax": 13–23× slower than vectorized/compiled on
  identical trades; **Numba njit of the SAME loop = 13.7× and ties vectorized numpy** (parity-checked,
  40,779 bit-identical trades). HIGH
- Speed ladder (identical PnL, one laptop): pandas loop → numpy 22.7× → njit+event-loop 35.3× →
  **njit prange across combos 217.6× → process pool 297.9×** (340 combos/s). HIGH
- **99.3% of compiled runtime is feature math (stochastic/ATR), 0.7% is the stateful trade loop** →
  indicator batch computation dominates; combo-level parallelism is the cheapest big win. HIGH
- Numba njit compiles stateful per-bar loops well (branches, continue, scalar state OK). Pitfalls:
  no pandas/dicts/sets (numpy arrays/typed containers), object-mode = ~zero speedup, `fastmath=True`,
  GIL released (threads viable). HIGH
- Windows spawn: args are pickled → pass paths/scalars; parse CSV once per worker (initializer);
  `imap_unordered(chunksize≈4)`; `multiprocessing.shared_memory`/np.memmap for read-only data. HIGH
- Vectorized indicators (2D symbols×minutes, no `rolling().apply()`) = 100–130× vs Python loops
  (RSI 120×, ATR 100×). `df.iterrows` = anti-pattern (15–100× slower than itertuples, 1,000–5,000×
  slower than vectorized). HIGH

## 2. GPU — viable but NOT the missing piece at our scale

- **RTX 3060 FP64 ≈ 0.2 TFLOPS (1/64 of FP32, ~60× slower)** → float32 throughout + Kahan
  summation for P&L (NIFTY ~25,000 → fp32 eps ≈ 0.003 pts, fine intraday). MEDIUM/HIGH
- One-thread-per-sim is standard (QuantStart MC 537×) but NVIDIA warns long serial loops
  underutilize unless threads ≫ cores — 15,552 configs × 3,584 cores passes, yet
  **thread-per-(config × day) with per-day GPU reductions is better occupancy**. HIGH
- **Numba's CUDA target is deprecated** → maintained `numba-cuda` package; CuPy RawKernel =
  lowest-risk, near-CUDA-C speed (2025 Springer study: parity across Numba/CuPy/CUDA-C). HIGH
- Real published GPU backtests: gpu-backtester 465× (OCO sims, but via vectorized batched-tensor
  reformulation, NOT thread-per-config) and ~80× walk-forward (648 params); VectorAlpha ~12×
  median for 1M-candle × 250-param indicator batches; GPU/CPU bit-for-bit parity impossible
  (tolerance ~1e-9). MEDIUM
- **Counterpoint (controlled, equivalence-gated): 298× on a laptop CPU for the same class of
  sweep — "a GPU is not the missing piece at finite width".** HIGH

## 3. Search — probabilistic beats exhaustive on 8 discrete dims

- Bergstra & Bengio (JMLR 2012): in high-dim spaces few dims matter; grid ~90% waste; **random
  beats grid; no deterministic guarantee exists — coverage is statistical**. HIGH
- Optuna: TPE natively handles discrete/categoricals; **HyperbandPruner/SuccessiveHalving beats
  MedianPruner**; BOHB result: Hyperband finds acceptable configs fast, BO finds the best →
  **Hyperband early + TPE refinement late**. HIGH
- Warm-start fully supported: sqlite storage (`load_if_exists=True`), fixed seed, `enqueue_trial`
  (re-test champion), `add_trial` (ingest our 200 logged trials) — nothing wasted. HIGH
- Overfit proofing for the winner (daily returns per config = we have them):
  - **PBO via SSCV** (Bailey et al.): needs ≥50 configs; PBO < 0.3 = strong. HIGH
  - **Deflated Sharpe** with N = effective trials (and worst-case 15,552). HIGH
  - White's Reality Check / Hansen SPA vs no-skill baseline. HIGH
  - Plateau test: ±1-step Hamming neighbors of winner similar P&L (local CV < 0.35 robust);
    parameter persistence across top-5%. MEDIUM
  - Walk-forward (re-tune per window, test next) + OOS degradation ≤ ~30-50% of IS. MEDIUM

## 4. Recommended pipeline (faster AND accurate)

| Stage | Method | Budget |
|:---|:---|:---|
| 0 | Ingest existing 200 trials into sqlite study (fixed seed 42, enqueue champion) | 0 runs |
| 1 | Subsample screening: ~1 regime-stratified year, TPE + HyperbandPruner (η=3, min=5%), proxy = Sharpe/PF + P&L | ~500 trials ≈ 100 yr-equiv |
| 2 | Refine top ~50 with multivariate TPE at full-year fidelity | ~200 trials |
| 3 | Full-5Y validation of top 10-20 finalists (champion + per-cluster best) | ~15 runs |
| 4 | Walk-forward 5 annual folds (opt years 1..k, test k+1) + 2026 holdout | 5×(50-100) quick |

Total ≈ 700-900 trial-equivalents vs 15,552 exhaustive (~15-20× cheaper).

Overfit-proofing checklist for the winner: PBO < 0.3 · DSR with N≥trials · SPA · CPCV · plateau
test · parameter persistence · ±2× cost robustness · seed robustness.

## 5. Recommendations for THIS project (effort → payoff)

1. **Numba njit of `process_day`** (trackers + trade loop one compiled function) — ~30×;
   parity-testable against current engine. Days of work, no new deps beyond numba.
2. **precompute feature stage vectorized (2D symbols×minutes) then njit trade loop** + prange
   across configs — combined 100-300× territory (ladder-proven pattern).
3. **Batch across configs that share params** — per-(symbol,day) indicator arrays cached across
   combos (search-side price-lookup precompute reported 1,556×).
4. **Multiprocessing hygiene on Windows**: worker-local CSV parse cache already exists; switch to
   `imap_unordered(chunksize≈4)`; avoid DataFrame in hot paths.
5. **GPU only if CPU-compiled still too slow** — then CuPy RawKernel: float32 + Kahan, batch
   feature computation on GPU, RawKernel only for the residual event loop, thread-per-(config,day),
   pinned memory + streams, parity tolerance 1e-9. RTX 3060 FP64 would be a mistake.
6. **Search restructure (accuracy)**: warm-start 200 trials, HyperbandPruner, two-stage screening.
7. **Honest winner validation**: PBO + DSR + SPA + plateau + walk-forward OOS before believing any
   ₹ + figure.

## Verdict for the plan

The current plan's Phase 3→4 (CuPy GPU primitives + exact GPU execution) is the highest-risk,
lowest-certainty part. Evidence says: build Numba-CPU parity path FIRST (Phase 3 becomes "CPU
compiled engine", GPU becomes an optional Phase 3.5 gated on a benchmark), and restructure the
search (Phase 6) around warm-started TPE+Hyperband + plateau/PBO validation instead of exhaustive
enumeration. Estimated total effort drops ~40-60% with the same or better accuracy.