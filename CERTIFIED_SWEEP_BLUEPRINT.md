# CERTIFIED SWEEP PIPELINE — Blueprint & Agent Handbook

> **Status:** PRODUCTION. This is the ONLY sanctioned way to validate any strategy
> in this repository. Every number produced before 2026-09-05 by `gpu_sim_last_hope.py`,
> `dyn_strike_engine.py`, or any §41–§44 sweep is **research-grade/convicted** —
> do not cite it as truth. The dyn engine was convicted of drift (see §1, Fix D).
>
> **Proof of correctness:** the pipeline reproduces the live bot's Sep 4 2026
> session trade-for-trade (12:55 FLAG @139.35→TP, 13:08 FLAG @134.35→SL) and the
> certified April 2026 harness 48/48 (3 variants × 16 days, exact to the paisa).
>
> **Speed:** 4.46M full-fidelity replays (2,940 configs × 1,518 days) in **3.9s**
> on the RTX 3060; extended grid 105M replays in minutes. Phase 1 recording:
> 1,526 days in 145s with 8 CPU workers.

---

## 1. Architecture — Why Three Tiers

A strategy backtest is a **path-dependent state machine** (position, stops,
ratchet latches). It cannot be vectorized away without corrupting trade
sequence (see GPU guide §26 `be_done` — the costliest bug in repo history).
But it parallelizes perfectly across **(config, day)** cells, because state
resets daily and configs are independent.

```
                    TIER 1 — RECORD (CPU, 8 workers)            once per day
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ lh_sweep_build2.py: run the REAL live engine (LastHopeWinnerEngine) over  │
  │ every spec-eligible contract of every day. Record per (contract, minute):  │
  │   OHLC · bar index · RAW indicator values (stochastics, ATRs, EMAs, VWAP,│
  │   TF bucket lo/cl) · rise-bits · prior-day HLC · spec-strike map           │
  │ Output: one .npz per day → lh_events_v2/<stochset>/ev_YYYYMMDD.npz         │
  └──────────────────────────────────────────────────────────────────────────┘
                    TIER 2 — CPU TWIN (njit)                        parity oracle
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ lh_sweep_replay_cpu.py: @njit replay of ONE (config, day) from the npz.   │
  │ Authoritative semantics document. Ground truth for the CUDA kernel.      │
  └──────────────────────────────────────────────────────────────────────────┘
                    TIER 3 — CUDA SWEEP (100% GPU)                    the sweep
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ lh_sweep_cuda2.py: one thread = one (config, day). fp64. Day-major        │
  │ mapping. Banker's rounding. 1D flat outputs. Chunked launches.           │
  │ Then: top-K re-verified on the CPU twin; CSV ledger.                    │
  └──────────────────────────────────────────────────────────────────────────┘
```

**The non-negotiable rule:** Tier 3 must reproduce Tier 2 EXACTLY (48/48 style
gate) before any sweep number is reported. Tier 2 must reproduce the
live-validated harness before that.

---

## 2. File Index

| File | Tier | Purpose |
|:---|:---:|:---|
| `lh_sweep_build2.py` | 1 | Event-stream recorder (parametrizable stoch set via `--stochset`) |
| `lh_sweep_replay_cpu.py` | 2 | njit CPU twin — the executable spec |
| `lh_sweep_cuda.py` | 3 | v1 CUDA sweep (fixed thresholds — parity-certified 48/48) |
| `lh_sweep_cuda2.py` | 3 | v2 CUDA sweep — sweeps ALL thresholds/indicator periods from raw recordings |
| `lh_parity_gate.py` | gate | 48-check gate vs the certified April harness |
| `run_full_window.py` | ref | Live-validated replay harness (Aug 26–Sep 4) — matched the live bot 1:1 |
| `april_certified_run.py` | ref | Certified April harness (3 variants × 16 days) |
| `lh_cuda_sweep_results.csv` | out | v1 sweep ledger (2,940 configs) |
| `lh_cuda_sweep2_results.csv` | out | v2 extended sweep ledger |

Reference truth chain (each level certifies the next):

```
LIVE BOT Sep 4 session  ==  run_full_window.py  ==  CPU twin  ==  CUDA kernel
     (ground truth)          (validated)         (executable spec)  (the sweep)
```

---

## 3. The Fix Ledger (every bug this pipeline killed)

These are the reasons naive engines lie. Any new port will hit them again.

| # | Fix | Symptom | Root cause |
|:---:|:---|:---|:---|
| **A** | `be_done` reset at BOTH entry sites | WR jumped 78.5%→84.4% after unrelated fix | Vectorized state flag never reset on position-open — every trade after the first BE rode to full SL/TP (GPU guide §26) |
| **B** | Bar-space arm windows | Entry 1–2 min drift | Engine counts BARS (`len(cs.bars)`), replay counted MINUTES. With gaps/missing bars these diverge — record `baridx` and compare `baridx[t] − baridx[anchor] ≤ arm` |
| **C** | `clear_all` on exit | Extra entries right after a position closed | Certified harness disarms EVERY contract when a position exits; kernels that only reset the position contract over-trade |
| **D** | **dyn engine conviction** | dyn: Sep 4 +₹1,532 / 100% WR vs live −₹23.50 | Vectorized TF-bucket + precomputed gate matrix drifted from engine semantics. **Self-parity is not correctness** — ground truth must be an independent implementation |
| **E** | Engine consumption pass | Replay over-triggered vs certified 31 vs 21 trades | The harness calls `_on_1m_bar_close` on EVERY contract EVERY minute — this CONSUMES arming whenever trigger+bounce fires, even while a position is open or on non-spec contracts. Replay must model the consumption, not just the entry |
| **F** | Disarm asymmetry per family | FixedB/Full10 trades never fired | S44 (engine path) disarms on signal-generation; FixedB/Full10 (armed-bounce paths) disarm on entry or consumption-hit only |
| **G** | Strict vs loose SUPER rising | Full10 WR wrong by points | Engine: `is_rising` needs `prev_s1 is not None`; Full10 harness: `s1 > (prev_s1 or 0)`. Record BOTH as separate bits |
| **H** | CUDA flat-index on 2D array | `CUDA_ERROR_ILLEGAL_ADDRESS` on full grid | `out[cfg*ND+day]` on a shape-(NC,ND) array is ROW indexing → OOB writes 4.46M past a 2,940-row array. Outputs must be 1D `NC*ND` |
| **I** | Local array bounds | `compute-sanitizer` 1-byte OOB reads | `cuda.local.array(128, int64)` per thread; assert `max(C) < 128` at pack time |
| **J** | Contiguous device transfer | `ValueError: non-contiguous buffer` | `cuda.to_device` requires `np.ascontiguousarray` on every slice |
| **K** | v2 parquet expiry rule | 16 trading days missing (Apr 8–13 etc.) | Old builder hard-coded Thu→Tue; NSE ran holiday-shifted MONDAY expiries (02MAR26, 30MAR26, 13APR26). New rule: nearest expiry ≥ day within 9 days, per-day from source CSVs |
| **L** | Uniform `day` dtype | `ArrowNotImplementedError: greater_equal(string, timestamp)` | Rebuilt parquet had mixed string/timestamp partitions → all predicate filters fail. Normalize to ISO string at write time |
| **M** | `iterrows()` index trap | Zero trades on a port | `for m, r in df.iterrows()` yields DataFrame INDEX, not the minute column → spot lookups never matched. Use `itertuples` + `r.minute` |
| **N** | April "high LTP" scare | 2nd-ITM premiums 200–450 vs Aug 80–200 | NOT a data bug — verified bar-for-bar vs independent source CSVs (0/375 mismatch). April 2026 was a genuine high-IV gap regime (−620 index gap days, put-call parity holds, TV math checks) |

---

## 4. Optimizations (what makes it fast)

1. **Record once, sweep everything else.** The engine pass (expensive: Python
   objects, per-bar indicator updates) runs ONCE per day per indicator set.
   All threshold/exit/gate params replay from raw recorded values — no
   re-recording per config.
2. **8-worker CPU recorder** (Phase 1): 1,526 days in 145s. Each worker reads
   only its day-range from the parquet (predicate pushdown, `filters=[("day",
   "in", ...)]`) — never the full 461MB file.
3. **One thread per (config, day)** (Tier 3): each CUDA thread walks its day's
   345 minutes sequentially (causality preserved), threads parallelize across
   105M (config, day) cells. This is the only design that is both
   event-accurate and GPU-fast.
4. **Day-major thread mapping** (`day = tid // NC`): neighboring threads share
   the same day's arrays → coalesced reads.
5. **fp64 everywhere in the audit path** (RTX 3060 fp64 is 1/64 rate but the
   grid is so parallel it doesn't matter — 4.46M threads in 3.9s). No TF32,
   no fastmath, no atomics for P&L.
6. **Banker's rounding ported** (`round(x,2)` = round-half-to-even) — a
   `dist` of 7.125 rounds differently under round-half-away and breaks paisa
   parity with Python.
7. **Chunked launches** for WDDM/TDR safety and VRAM: configs chunked at ~8K,
   day tensors resident (~1.2GB), outputs per chunk (~500MB).
8. **`np.savez_compressed` per day**: ~220MB per indicator set; stoch values
   compress extremely well.
9. **Prior-day seeding with tail-300 same-symbol bars** — matches the live
   bot's broker TPSeries warmup (TradingView parity). COLD on weekly-token
   rollover days (by design — mirrors live).
10. **Fee model in every trade:** `(exit−entry)×65 − 45`. Fees decided the
    entire last sweep — never report gross.

---

## 5. DO's — Certified-Sweep Rules (MANDATORY)

1. **DO run the parity gate first, every time.** New kernel, new recording
   build, new data — 48/48 (or equivalent vs your reference window) BEFORE any
   sweep number leaves your mouth.
2. **DO use an independent ground truth.** The chain must terminate at
   live-bot logs or the live-validated harness. Two engines sharing a core
   verify nothing (Fix D).
3. **DO compare trade lists, not aggregates.** Diff per-day nets AND per-trade
   (entry minute, exit minute, prices, kind). One trade of drift is a bug.
4. **DO record RAW indicator values, not booleans**, wherever a threshold might
   ever be swept (stochastics, ATR, EMA). Booleans bake the threshold in and
   force re-recording per config (that's why v1's 2,940-config sweep needed
   v2's raw recording for the extended grid).
5. **DO model engine side effects** (consumption/disarm, `clear_all`) — the
   strategy is not just entry conditions; it's the whole state machine.
6. **DO assert array bounds at pack time** (`max(C) < MAXC`, spec indices in
   range) — a 1-byte OOB on GPU is a silent wrong number or a dead context.
7. **DO re-verify the top-K on the CPU twin** after every GPU sweep. GPU
   screens, CPU confirms.
8. **DO write results to a CSV ledger** with config, net, trades, WR, PF,
   maxDD, Calmar — and record the data fingerprint + git hash.
9. **DO smoke-test on 5 days** before any multi-year recording (AGENTS.md
   smoke-test mandate still applies).
10. **DO use `compute-sanitizer --tool memcheck` and `NUMBA_ENABLE_CUDASIM=1`**
    when a kernel misbehaves — the simulator gives exact Python-level
    IndexError lines; the sanitizer gives the faulting thread/address.

## 6. DON'Ts

1. **DON'T trust a vectorized engine against itself.** Self-parity (metrics
   path vs trade-list path of the same core) is the `be_done` trap.
2. **DON'T use center padding, future fill, or same-bar signal+fill.**
   Left-pad `(K−1, 0)` only (GPU guide §3).
3. **DON'T pin instruments at session start.** Dynamic strike at trigger
   minute from live spot (`round(spot/50)*50`; CE=ATM−100, PE=ATM+100) —
   this was worth +29% in §44 and is a semantic, not an optimization.
4. **DON'T overlap positions** — position lock is part of the strategy.
5. **DON'T omit fees/slippage.** The 2026-09-05 sweep proved the entire
   family is gross-profitable, fee-destroyed.
6. **DON'T write 2D outputs with flat indices** (Fix H) — the single most
   likely crash in any new kernel.
7. **DON'T put Python objects, dicts, or `.item()` calls in hot loops.**
   `.item()` on GPU scalars = 68,700 syncs (the 59.3× lesson).
8. **DON'T run multiple CUDA processes** on the 3060 (context exhaustion).
   One process, chunked launches.
9. **DON'T re-read the parquet per day.** Predicate-filter per worker range,
   read once, cache npz.
10. **DON'T cite pre-2026-09-05 backtest numbers** (§41–§44 sweeps, dyn
    corpus, FixedB walk-forward) — convicted or unverified. The v1 certified
    ledger (`lh_cuda_sweep_results.csv`) supersedes them.

---

## 7. Running a Sweep for a NEW General Strategy

Example: an RSI-based strategy with different parameters. The pipeline is
strategy-agnostic — only the RECORDED EVENTS and the KERNEL TRIGGER BLOCK change.

### Step 1 — Define the semantics (write it down first)
Write the exact entry/exit state machine as prose, then encode it in a CPU
twin. The CPU twin IS the executable spec (GPU guide §26 rule 4).

### Step 2 — Phase 1: record (adapt `lh_sweep_build2.py`)
- Replace the indicator instantiation block with YOUR indicators (RSI period
  variants, Bollinger, whatever) — one column per period variant you might
  sweep. Record RAW values.
- Keep the fixed scaffolding: session clip [555, 900), tail-300 same-symbol
  prior-day seeding, spec-strike map from spot, `has`/`baridx`/OHLC, npz per
  day under `lh_events_v2/<variantset>/`.
- Smoke: record 5 days, verify trade counts 1–10/day and WR 30–50%.

### Step 3 — Phase 2: CPU twin (adapt `lh_sweep_replay_cpu.py`)
- Port your state machine into `@njit replay_day(...)` with params as scalar
  args. Every per-trade flag RESETS at position-open (Fix A).
- Bar-space windows via recorded `baridx` (Fix B). Model consumption/disarm
  side effects (Fix E/F).

### Step 4 — Certify the twin against an independent truth
- Build ONE live-validated day (replay harness on fetched broker bars, or live
  bot logs) and require trade-for-trade equality. No independent truth = no
  sweep.

### Step 5 — CUDA kernel (adapt `lh_sweep_cuda2.py`)
- Same skeleton: 1D flat outputs, `MAXC` local arrays with pack-time assert,
  day-major mapping, fp64, banker's rounding if the strategy rounds.
- Insert your trigger block where v2 computes m6/super/rise from raw stoch.
- Gate: 48/48-style parity vs the twin across your reference window, PLUS a
  10-day × N-config random diff vs the twin.

### Step 6 — Sweep, verify top-K, ledger
- Chunked full grid → CSV → re-run top-10 configs on the CPU twin → equal?
  → report. Record: data fingerprint, git hash, grid definition, wall time.

### Grid design guidance (from the 2026-09-05 experience)
- Sweep at minimum: trigger thresholds, arming window + threshold, stop/target
  multiplier, indicator periods, BE on/off × ratio × buffer, touch buffer,
  gate family, trigger requirement.
- Constrain ranges to market-meaningful values (GPU guide §11).
- 20K–100K configs × 1,500 days is comfortable on this hardware (minutes).

---

## 8. Data Contract (v2 canonical)

- **Parquet:** `nifty50_options_master_v2.parquet` (see AGENTS.md — the ONLY
  option source). `day` ISO string, `minute` 555–899, symbols
  `NIFTY<DDMMMYY><STRIKE><CE|PE>`, plus `exp_date`.
- **Index:** `C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv`
  (Flattrade TPSeries starts ~Jun 2022 only — desktop CSV is the long source).
- **Non-synced cache (ALWAYS read from here):**
  `C:\Users\user\AppData\Local\Temp\opencode\data\nifty50_options_master_v2.parquet`
  (OneDrive stalls the Desktop path — GPU guide §16 BUG #5).
- Sessions: [555, 900). Post-close bars clipped; pre-open 09:14 bar excluded.
- TPSeries stamp convention: bar labels = minute START (TV-compatible), no shift.

---

## 9. The 2026-09-05 Sweep Verdict (the reason this pipeline exists)

2,940 configs × 1,518 days, fee-complete, parity-certified:

- **Every configuration loses money over 2020–2026.** Range −₹1.83M … −₹129K.
- §44 champion (deployed live!): **−₹821,080**, WR 69.2%, PF 0.777.
- Best config (arm26/×1.0/be0.3+1.5/tb0.5/full10/trigON): −₹128,575 —
  +₹668K gross − ₹796K fees on 17,696 trades.
- Structure: high WR (72–75%) with avg win ≪ avg loss (BE churn +₹20 vs SL
  −₹500) and ~35 trades/day × ₹45 fees is structurally fatal.
- Median effects: BE ratchet saves the most (−543K vs −955K OFF); full10 gate
  beats ema20-only (opposite of the dyn sweep's ranking); trig ON ≈ OFF.

**Decision protocol going live:** no config goes to the VPS unless its
certified-sweep net is positive AND it survives a walk-forward split on this
harness.
