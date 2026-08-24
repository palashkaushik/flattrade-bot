# Optimus Backtest — Agent Runbook  (engine file: `optimized_gpu_backtest.py`)

> **Engine file:** `artifacts/f6_hybrid/optimized_gpu_backtest.py` — branded **"Optimus Backtest"** after the bug-hardening pass.
> **Regression guard:** `artifacts/f6_hybrid/test_optimus_regression.py` (speed + accuracy; exits non-zero on regression).
> **Depends on:** `opt_futures_quad` (repo-root data module: `load_spot()`, `option_day_files()`)
> **Hardware target:** NVIDIA GPU (RTX 3060 12 GB used here), CUDA, PyTorch 2.5.1+cu121
> **Status:** fused `(B,N,T)` 3D-batch engine, **59.3× faster** than the pre-fix version after the scalar-readback fix (see §4). Hardened: batching daily-cap fix, mixed-precision fp16 safety, ensemble SoC refactor. Verified regression-free (see `optimus_baseline.json`).

This runbook lets any agent run a causal, GPU-accelerated backtest of **any** options strategy on this engine, reproduce results, and extend it **without reintroducing the performance bug**.

---

## 1. What the engine does

A full Optuna batch of `B` trials is evaluated in **one fused `(B, N, T)` GPU pass** (`N` = ~1574 days, `T` = 375 1-minute bars) instead of `B` separate dispatches.

- **Entry signal:** stochastic `S4` (%K, fast) vs `s4_ob`, and `S1` (%D, slow) vs `s1_os`, inside a valid session window, with a tradable SL distance.
- **Direction:** CE (call side) and, for bidirectional strategies, PE (put side). Ensemble/merge via `merge_results`.
- **Trade management:** SL/TP in points off the entry bar, ₹30 fee + 1pt slippage, ITM strike offset by `moneyness`.
- **Causal pillars (DO NOT break):** zero lookahead (`F.pad(x,(K-1,0))` left-only), TF clock alignment, exchange drag, **position lock = 1 trade/day/direction** (unless re-entry mode is enabled), daily loss/profit circuit breaker.

### Strategy IDs (timeframes)
| ID | TF | Notes |
|----|----|-------|
| B01 | 1m | CE-only (baseline, unidirectional) |
| B02 | 1m | CE+PE bidirectional |
| B03 | 2m | CE+PE bidirectional |
| B04 | 3m | CE+PE bidirectional |
| B05 | 5m | CE+PE bidirectional |
| B06 | 1m | CE+PE, tight-drawdown variant |
| B07 | free `timeframe∈{1,2,3,5}` | best-TF, win-rate-first scoring |

### Param dict schema (what `suggest_params` / `suggest_candidate` return)
```python
{
  "timeframe": int,            # 1,2,3,5
  "s1_k": int,                 # %D slow stochastic period (5..30)
  "s4_k": int,                 # %K fast stochastic period (20..120)
  "s1_os": float,              # %D oversold threshold (lower = fewer entries)
  "s4_ob": float,              # %K overbought threshold (higher = fewer entries)
  "atr_p": int,                # ATR period (8..35)
  "sl_m": float,               # SL distance (× ATR × 0.5)
  "tp_m": float,               # TP distance (× ATR × 0.5); must be >= 1.5× sl_m
  "daily_loss_pts": int,       # daily loss cap (points)
  "daily_profit_pts": int,     # daily profit cap (points)
  "moneyness": 0.5|0.6|0.7,    # strike selection (0.5=ATM)
  "max_trade_loss_rs": int,    # max SL distance allowed (filters untradable params)
  "sess_start_off": int,       # session start offset (bars after BASE_SESSION_START=5)
  "sess_end_off": int,         # session end offset (sess_end = 345 - sess_end_off)
  "sess_end": int,             # = BASE_SESSION_END - sess_end_off
}
```

---

## 2. Environment & invocation

The module **auto-loads all data into VRAM on import** (~3 s, ~4.2 GB). Run from anywhere — it inserts the repo root on `sys.path` and imports `opt_futures_quad`.

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "backend:cudaMallocAsync"   # required: holds VRAM flat, no growth
$py = "C:\Users\user\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
cd "C:\Websites\FLATTRADE BOT\artifacts\f6_hybrid"
```

### Run modes
| Goal | Command |
|------|---------|
| Full 7-strategy study (NW + WF-OOS) | `$py optimized_gpu_backtest.py` |
| One strategy only | `STRAT=B07 $py optimized_gpu_backtest.py` |
| Candidate cream mode (top-5 of each phase, re-optimized) | `MODE=CANDIDATES CTRIALS=800 $py optimized_gpu_backtest.py` |
| Parity check (fused 3D vs sequential 2D) | `PARITY=1 $py optimized_gpu_backtest.py` |

### Key env vars
- `BATCH` — fused batch size (default **100**). Larger batches amortize better; keep `PROCS=1`.
- `PROCS` — parallel workers (default **1**). >1 uses `spawn` (each worker reloads data → ~4.2 GB VRAM **each**; can OOM a 12 GB card). Prefer `PROCS=1`.
- `TRIALS` — override `TRIALS_PER_STRATEGY` (default 3000) for the 7-strategy run.
- `STRAT` — filter to one strategy ID.
- `CTRIALS` / `CAND_LIMIT` — candidate-mode trial count / candidate cap.
- `MODE=CANDIDATES`, `PARITY=1` — see table.

### IS / OOS masks (data splits)
The module exposes three masks you pass as `day_mask` to the eval functions:
- `None` → full-period (NW / non-walk-forward)
- `d_is_mask` → In-Sample, days `< "2024-01-01"` (~994 days)
- `d_oos_mask` → Out-of-Sample, days `>= "2024-01-01"` (~580 days)

### Evaluate a single param set (most common "any strategy" call)
```python
import optuna, optimized_gpu_backtest as m
params = dict(timeframe=3, s1_k=7, s4_k=50, s1_os=25.0, s4_ob=70.0, atr_p=10,
              sl_m=1.5, tp_m=5.0, daily_loss_pts=8, daily_profit_pts=50, moneyness=0.5,
              max_trade_loss_rs=1500, sess_start_off=0, sess_end_off=30, sess_end=315)
res = m.evaluate_batch("B07", [params], None)[0]          # full period
res_is = m.evaluate_batch("B07", [params], m.d_is_mask)[0]
res_oos = m.evaluate_batch("B07", [params], m.d_oos_mask)[0]
# res keys: trades, win_rate, net_rs, pf, max_dd, ce_trades, pe_trades, ce_pnl, pe_pnl
```

### Run your own Optuna study
```python
study = m.search("B07", None, n_trials=500, seed=42)      # full period
best = study.best_trial.params
```

---

## 3. Step-by-step: backtest ANY strategy

1. **Conform the signal to the entry schema.** The engine enters when `S4 >= s4_ob` (fast stochastic overbought) AND `S1 <= s1_os` (slow stochastic oversold). Map your strategy's entry rule onto these two stochastic comparisons (and, for PE, their mirror: `S4 <= 100-s4_ob` AND `S1 >= 100-s1_os`). If your idea isn't a stochastic-divergence entry, extend `evaluate_batch` (see §5) rather than hacking the mask.
2. **Pick a strategy ID / timeframe** (`timeframe` in the param dict; B01 is CE-only).
3. **Set risk params**: `sl_m`, `tp_m` (TP ≥ 1.5× SL or the trial is pruned), `daily_loss_pts`, `daily_profit_pts`, `max_trade_loss_rs`, `moneyness`, session window.
4. **Smoke-test first** (mandatory): run `evaluate_batch` on 5 days by passing a tiny `day_mask` (e.g. `m.d_is_mask[:5]`) — verify trades ∈ [1,30]/day, WR 30–50%, no crash. Restore full mask before the real run.
5. **Run** via `search()` (custom) or the CLI modes in §2. Results dicts carry `net_rs`, `pf`, `win_rate`, `max_dd`, `trades`, `ce_*`/`pe_*`.
6. **Always verify with `PARITY=1`** after any change to the simulate/`_finalize` path — it asserts fused 3D == sequential 2D within 1e-2 on B01/B02/B07 across NW/WF/OOS.

---

## 4. The 59× optimization (READ BEFORE TOUCHING HOT PATH)

Profiling (`torch.profiler`, CPU+CUDA) showed launch overhead was **negligible (1.0×)** and CUDA Graphs (P0) are **not viable** (variable-size `torch.where`/`torch.nonzero` gathers can't be captured — `IndexError "Dimension out of range (expected [-2,1], but got 2)"`).

The real cost was **~58% of GPU time spent in GPU→CPU scalar readback**: `_finalize`'s per-trade circuit-breaker loop compared `new_cum < -loss_cap` where `loss_cap`/`profit_cap` were passed in as **0-dim GPU tensors**, forcing a sync on every one of ~68,700 trades.

**Fix:** `_to_scalar(x)` coerces caps to a plain Python float **once per `_finalize` call** (not per trade). `evaluate_batch(B=100)` went **8.24 s → 0.14 s (59.3×)** with bit-for-bit identical P&L.

---

## 5. DO's and DON'Ts (critical)

### DON'T (these kill performance or correctness)
- ❌ **Never call `.item()`, `bool(tensor)`, `tensor.is_nonzero()`, or `float(tensor)` inside any per-trade / per-element / per-day loop.** Force a sync once before the loop, never inside it. This is exactly the bug that cost 58% of runtime.
- ❌ Don't assume CUDA Graphs help here — they don't (variable-size gathers, negligible launch overhead).
- ❌ Don't use `torch.compile` / Triton — no wheel for torch 2.5.1+cu121 on Windows.
- ❌ Don't recompute `get_stoch`/`get_atr` per trial — they're cached (`STOCH_CACHE`/`ATR_CACHE`). Call them; never re-derive.
- ❌ Don't introduce lookahead: indicator padding must be **left-only** `F.pad(x,(K-1,0))`. Right padding = future leak.
- ❌ Don't set `PROCS>1` on a 12 GB card without checking VRAM (each spawn worker reloads ~4.2 GB).
- ❌ Don't drop the `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` env var — VRAM creeps without it.
- ❌ Don't change `BATCH_SIZE` semantics (constant `(B,N,T)` shapes) without re-running parity.

### DO
- ✅ Keep `_finalize` receiving **plain Python floats** for `loss_cap`/`profit_cap` (use `_to_scalar`).
- ✅ Pass `day_mask=None` (full/NW), `d_is_mask` (IS), or `d_oos_mask` (OOS) — never hand-roll the split.
- ✅ Keep the fused single-pass sim; extend entry-mask generation, not the core arithmetic.
- ✅ Run `PARITY=1` after any edit to `simulate_direction_locked_batch` / `_finalize` / `merge_results`.
- ✅ Use `BATCH=100` (default). If you raise it, confirm VRAM headroom.
- ✅ Respect causal pillars (§1): position lock, daily circuit breaker, clock alignment.

---

## 6. Reference papers (sent for review) — what to reuse, what to ignore

Location of the source PDFs (re-read if a new optimization is contemplated):

| PDF | Path | Applicable to our backtest? | Takeaway for an agent |
|-----|------|------------------------------|-----------------------|
| **GPU-Accelerated Computing with Python 3 and CUDA** (Cautaerts & Ghorbanfekr, Packt 2026) | `C:\Users\user\Downloads\Documents\GPU-Accelerated Computing with Python 3 and CUDA_ From low-level kernels to real-world applicatio...{Niels Cautaerts_ Hossein Ghorbanfekr}(2026, Packt Publishing){115889524} libgen.li.pdf` | Partially — principles only (it teaches Numba/CuPy/JAX, **not PyTorch**) | Profile before optimizing; fused single-pass already done; **CUDA Graphs don't fit our variable-size gathers**; preallocate/reuse buffers (indicator cache already does); `cudaMallocAsync` already on; fp16 is risky (gate behind parity). The launch-overhead advice does **not** apply (measured 1.0×). |
| **Parallelizing High-Frequency Trading using GPGPU** (Anil et al.) | `C:\Users\user\Downloads\Telegram Desktop\aditya-anil-parallelizing-high-frequency-trading-using.pdf` | Conceptual only — no tunable technique | Confirms our stochastic formula (`get_stoch` `(C−L14)/(H14−L14)×100` == their `%K`). Confirms scalar readback is *the* HFT bottleneck (matches our fix). Nothing to implement. |
| **GPU Accelerated Option Pricing Algorithms** (Michaeli, 2025, BSc thesis) | `C:\Users\user\Downloads\Documents\Michaeli_Daniel_2025.pdf` | No — option *pricing* (CRR/Monte-Carlo), not backtesting | Generic "move work to GPU / minimize host sync" already applied. Skip for backtest work. |
| **GPU-Accelerated Optimization of Discrete Ricci Flow for High-Resolution Triangular Meshes** (Wei et al., 2025) | `C:\Users\user\Documents\wei-et-al-2025-gpu-accelerated-optimization-of-discrete-ricci-flow-for-high-resolution-triangular-meshes.pdf` | No — 3D mesh geometry, irrelevant domain | Discard for this project. |

**Rule for agents:** before adding any optimization inspired by these PDFs, re-profile with `torch.profiler` (CPU+CUDA) and confirm the bottleneck is real. The only change that measurably helped was removing the per-trade GPU→CPU scalar sync (§4).

### Additional PDFs received — NOW EXTRACTED (machine-readable)
These 4 PDFs were sent for review. They **cannot be parsed as PDF by this model**, but text was
extracted with `PyPDF2` and saved (so any agent can read them) under
`artifacts/f6_hybrid/pdf_text/`:

| PDF | Extracted text | Verdict |
|-----|---------------|---------|
| **GPU Programming Fundamentals with CUDA** | `pdf_text/pdf_GPU_Programming_Fundamentals_with_CUDA_(--)_...txt` | Standard kernel techniques (shared mem, streams, pinned mem, fp16, warp coalescing). Already inside PyTorch or N/A to resident-VRAM fused engine. Only lever: fp16 (risky, needs parity). |
| **Handbook of AI & Big Data Applications in Investments** (Larry Cao) | `pdf_text/pdf_Handbook_AI_BigData_Investments.txt` | Conceptual (backtest overfitting, ML features). No GPU technique. |
| **Python for Algorithmic Trading Cookbook** (Jason Strimpel) | `pdf_text/pdf_Python_AlgoTrading_Cookbook.txt` | VectorBT / Numba / Polars-GPU / Zipline — CPU/DataFrame vectorization; our `(B,N,T)` PyTorch engine already surpasses it. |
| **MSDN Magazine** | `pdf_text/pdf_MSDN_Magazine_(...)...txt` | General MS dev; no relevant optimization. |

**Conclusion:** no new applicable optimization. The PDFs confirm the patterns the engine already
embeds (minimize host sync, fuse passes, vectorize single-pass). The remaining real lever is
**batching multiple param-sets per GPU pass** (structural — not from these PDFs).

To re-extract after edits: `python -m pip install PyPDF2` then `PyPDF2.PdfReader(path)` → `extract_text()`.

---

## 7. Cross-strategy extension (see `cross_strategy_ensemble_gpu.py`)

The ensemble reuses this engine's data/indicators and adds: (a) **meta-confirmation** — multiple component strategies vote on entry, trade fires when ≥`confirm_k` agree; (b) **wider entry bands** (`band_relax` widens `s4_ob`/`s1_os`); (c) **intraday re-entries** (`reentry=True` lifts the 1-trade/day lock after a position exits). All three feed the same optimized `_finalize`/`simulate` path, so the §5 rules still apply.

**Status (verified):** `PARITY=1 python cross_strategy_ensemble_gpu.py` → `ENSEMBLE PARITY: PASS`
(4 identical components + `confirm_k=1` + `band_relax=0` + `reentry=False` reproduces the
single-strategy baseline bit-for-bit on NW/WF/OOS). A 150-trial study ran in ~6 s and produced
NW +₹576k @ 73% WR, OOS +₹260k @ 78% WR (saved to `ensemble_candidates.json`).

**Bugs caught and fixed during bring-up** (keep these in mind when extending):
1. `_sim_exits` must mask invalid future bars to `±1e9` (else clamped EOD bars false-trigger SL/TP).
2. `_sim_exits` indexes price tensors by the **day** index `en`, not the batch index `eb`.
3. PE take-profit sign must be `- ATRe*sl_m` (below entry), not `+`.
4. Ensemble trade TP must use `tp_m`, not `sl_m`.
5. `main()` writes JSON via `open(out,"w")`, not `json.dump(obj, Path)`.

### Batching (multi-param-set) — ALREADY present, now VERIFIED correct
The fused engine already batches **`B=100` trials per fused `(B,N,T)` pass** — both `search()` (base, `optimized_gpu_backtest.py`) and `search_ensemble()` (ensemble) call `evaluate_batch`/`evaluate_ensemble_batch` with `B=BATCH_SIZE` param-dicts in one GPU op. This is the real multi-param-set optimization (not a new addition), and `BATCH` is the env knob (default 100).

**Verification (2026-08-16):** batched `evaluate_batch(B=3)` vs 3× `evaluate_batch(B=1)` is **bit-for-bit identical** on trades/net/WR/PF/DD for all three TFs (1m/3m/5m). This caught a **latent batching bug**:

> **BATCHING BUG (fixed):** in `simulate_direction_locked_batch` the daily circuit-breaker cap was built as `dl = max_daily_loss[b_idx]` (a per-row array) and then read back as `dl[bi]` — indexing the *per-row* array at position `bi`, which returns **another trade's** daily-loss cap, not param-`bi`'s cap. At `B>1` this corrupted the daily loss/profit circuit breaker and silently dropped trades (e.g. 277→244 for the 1m param). The parity test had only ever exercised `B=1` (single pair), where per-row index `0` coincides with param index `0`, so the bug was masked. **Fix:** index the per-param `(B,)` cap tensor directly — `dl_param[bi]`, `dp_param[bi]`.

**Rule:** after any edit to the batch sim, verify with `B>1` (not just parity's `B=1`). Run a quick `evaluate_batch([p,p,p])` vs 3× `evaluate_batch([p])` and assert trades+net match.

### fp16 / mixed-precision path — IMPLEMENTED as SAFE mixed precision (`HALF=1`)
A naive full `float16` recast was tried first and **rejected**: it diverged ~50% in trades (678→337) because pips-based exits subtract two ~24,000 values (difference ~20 pts) and `float16` (~2-pt resolution at 24,000) suffers catastrophic cancellation in `exit_px − entry_eff`. Web research (Micikevicius 2017 mixed-precision; PyTorch AMP "keep the residual/critical stream in fp32"; FPChecker/Wikipedia Sterbenz — *the only fix for cancellation is to avoid subtracting approximations of nearby quantities*) gave the cure: **mixed precision**.

**Implementation (current):** `enable_half()` casts **only `STOCH_CACHE` to `float16`**. Everything else — `d_high/d_low/d_close/prev_c/d_tr`, `TF_DATA`, `ATR_CACHE`, and all control tensors (`T()`) — stays **float32**, so the exit P&L (`exit_px − entry_eff`, SL/TP) is computed in fp32 and is **bit-exact**. The stochastic cache feeds threshold comparisons only; `float16→float32` upcast is lossless, so entry masks differ from fp32 only at the ~0.4% of bars where stoch sits within one fp16 ULP of threshold.

**Verification (2026-08-16) — SAFE:**
| metric | fp32 | HALF (mixed) | delta |
|--------|------|--------------|------|
| NW trades (3m param) | 678 | 677 | −1 (−0.15%) |
| NW net (₹) | 429,193 | 427,663 | −0.36% |
| PF | 4.13 | 4.13 | identical |
| peak VRAM (B=100) | 3458 MB | 3115 MB | **−342 MB (~10%)** |
| batching B=3 vs 3×B=1 | — | identical | OK |

Ensemble parity also `PASS` under `HALF=1` (both the single-strategy and ensemble paths share the identical fp16 stoch cache, so their 0.4% entry noise cancels exactly).

---

## 8. Optimus Backtest — regression guard

`test_optimus_regression.py` is the speed + accuracy regression gate. Run it after **any** change to `optimized_gpu_backtest.py` or `cross_strategy_ensemble_gpu.py`:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "backend:cudaMallocAsync"
& .\venv\Scripts\python.exe artifacts/f6_hybrid/test_optimus_regression.py
```

It asserts (exit 0 = clean):
- **Accuracy anchors** — base B07 3m NW: T=678, net=₹429,193.5, PF=4.13 (hard-coded golden).
- **Batching** — `B=3 == 3×B=1` (trades + net bit-exact).
- **Ensemble parity** — meta path == base engine for NW/WF/OOS (exact).
- **fp16** — HALF trades within 2, net within 1.5% of fp32 (measured in an isolated subprocess, since `enable_half()` mutates `STOCH_CACHE` in place).
- **Speed budget** — `evaluate_batch(B=100) < 2.0s`, ensemble parity `< 120s`.

**CRITICAL invariant (2026-08-16 regression):** `_finalize()` must **count the trade that breaches the daily loss/profit cap before halting the day**. The day is `break`ed *after* `kept.append(r)`, never before — otherwise the loser that triggered the halt is silently dropped, understating losses (this was the source of the invalid ₹1.9M / 71% WR Optimus number). Any change to `_finalize` must keep the breach trade in `kept`.

Measured baseline (RTX 3060 12 GB, 2026-08-16): B=1 ≈ 0.007s, B=100 ≈ 0.27s, parity ≈ 1.6s; stored in `optimus_baseline.json`. The HALF path saves ~342 MB peak VRAM (~10%).

**Conclusion:** `HALF=1` is now a **safe VRAM-saver** (~10% peak, more headroom for `BATCH`/`PROCS>1`) with negligible signal error (<1 trade, <0.4% net). Use it when VRAM-constrained; keep fp32 for bit-exact reproduction. (If even more VRAM is needed, casting `ATR_CACHE` to fp16 is the next lever but shifts SL/TP stops by ~3 pts — verify before trusting.)
