# Optimus GPU Backtest Reference Findings

This document records the useful findings from all 15 extracted references in
`C:\Users\user\Downloads\pdf_text_md`, interpreted for the Smart Fib/Optimus
backtest on this machine.

## Scope And Hardware

- Hardware: NVIDIA RTX 3060, 12 GB VRAM.
- Runtime: CUDA 12.1, PyTorch `2.5.1+cu121`, Windows/WDDM.
- Available: CUDA PyTorch, Polars, NumPy, Numba `0.66.0`, and Numba-CUDA
  `0.30.4`.
- Installed toolkit: CUDA `12.9.41`, with NVVM and NVRTC available.
- Verified: `numba.cuda.is_available() = True` on the RTX 3060.
- Audit requirement: preserve causal/live parity, event order, stop/target
  precedence, fees, slippage, dynamic strikes, and daily risk limits.

## Main Conclusion

The best architecture is a hybrid:

1. Keep the CPU implementation as the reference oracle.
2. Keep invariant OHLC and event tensors resident on the GPU.
3. Evaluate independent events and parameter configurations in GPU batches.
4. Move only high-volume, independent numerical scans to the GPU.
5. Keep path-dependent ordering and final audit aggregation deterministic.

The references do **not** support moving every stateful operation to a GPU
just because a GPU is available. Small, irregular, causal state machines can
lose time to launch, transfer, and synchronization overhead. Any GPU signal
extractor must first pass event-level parity, not only aggregate P&L parity.

## Implemented From These Findings

The current working tree now contains a parity-checked GPU preprocessing module
at `artifacts/f6_hybrid/smart_fib_gpu_statescan.py`:

- Device-vectorized Wilder ATR, UT color, and S1 scans with no per-series
  `.item()` or `.tolist()` calls.
- A fused ATR + UT + S1 scan for callers that need all three outputs.
- Closed-bar 5m Heikin-Ashi + UT + LinReg snapshots.
- Fixed Smart Fib bullish/bearish UT swing completion records.
- Causal pending-setup snapshots with age and swing-base invalidation.
- Float64 audit defaults and NaN/missing-bar state preservation.
- An installed and verified Numba-CUDA toolchain is now available for the real
  single-kernel implementation: CUDA Toolkit `12.9.41`, NVVM true, Numba-CUDA
  `0.30.4`, and a real `@cuda.jit` smoke kernel passed.
- The compiled implementation is in
  `artifacts/f6_hybrid/smart_fib_numba_cuda.py`; it is explicit opt-in and uses
  the PyTorch scan as fallback.

The focused GPU regression file is
`tests/test_smart_fib_gpu_statescan.py`. It covers padded tails, middle gaps,
closed 5m filter calculations, swing completions, pending setup state, and
parameter-control dtype propagation. The current run passed `4` tests. On a
representative warm RTX 3060 run with `B=4096, T=375`, the fused scan measured
about `0.84 s` versus `0.92 s` for the two separate vectorized scans, or about
`1.09x` for that combined workload. This is a preprocessing-kernel timing,
not a full-window Optimus wall-clock claim.

The new preprocessing functions are intentionally not silently wired into the
historical event oracle yet. The next integration gate is a complete event
stream comparison against `extract_day_events`, including previous-day
warmup, date-aware 5m bucket formation, option reachability, and same-minute
ordering. Until that gate passes, the existing CPU oracle remains authoritative.

### Verified compiled-kernel result

On the RTX 3060 with `B=4096, T=375`, the one-thread-per-series Numba kernel
measured approximately `0.008-0.009 s` versus `0.806-0.993 s` for the PyTorch
fused scan, or roughly **100x faster** (`98.7x` in the first stable run and
`107.6x` in a later warm run), with CPU parity passing on padded and middle-gap
data.
At `B=100,000`, block sizes `64`, `128`, and `256` were approximately tied at
`0.137 s`; the implementation keeps `128` as the default and exposes the block
size for measured tuning. This is still a preprocessing benchmark, not a
full-window Optimus claim.

## Why The Fused Scan Was Only 1.09x Faster

That number is not comparable to the runbook's Optimus headline numbers.

The measured `0.84 s` versus `0.92 s` compared two already-vectorized PyTorch
preprocessing paths on one `(B=4096, T=375)` batch. The new function fused two
Python time loops, but it still launches many CUDA kernels per time step for
`gather`, `scatter_`, `where`, `max`, `min`, and reductions. It is not one
compiled CUDA kernel.

The runbook's measured `59.3x` result came from a different, much larger bug:
the Optimus `_finalize` loop read GPU scalar tensors once per trade. Removing
approximately `68,700` GPU-to-CPU scalar synchronizations changed `8.24 s` to
`0.14 s` with bit-for-bit identical P&L. The runbook's approximately `41x` and
larger suite figures come from resident `(B,N,T)` evaluation, 3D parameter
batching, matrix-first first-hit exits, and avoiding repeated data preparation.

The correct comparison is therefore:

- Preprocessing microbenchmark: measures indicator/state-scan implementation.
- Optimus benchmark: measures CPU preparation, H2D transfer, GPU batch
  evaluation, D2H summary, and finalization separately.
- Full suite speedup: compare the same number of trials, same batch size, same
  date mask, and same parity configuration.

Do not use the `1.09x` microbenchmark to judge the runbook's `59x`/`404x`
architecture.

## Numba CUDA Diagnosis And Recovery

The original diagnosis before installing the toolkit was:

```text
nvidia-smi: RTX 3060, driver 610.74, CUDA UMD 13.3, WDDM
cuda.detect(): 1 supported device, compute capability 8.6
cuda.current_context(): succeeds
Numba driver availability: True
Numba NVVM availability: False
Numba cuda.is_available(): False
```

Numba's `cuda.is_available()` requires both a working CUDA driver and the NVVM
compiler. PyTorch's `+cu121` wheel supplies the CUDA runtime needed by PyTorch;
it does not supply the full CUDA toolkit/NVVM compiler needed to JIT-compile a
Numba kernel. There is also no system CUDA Toolkit, `nvcc`, `cuda-nvrtc`, or
`numba-cuda` package in the Hermes environment.

The direct compile smoke test confirms the failure is exactly:

```text
NvvmSupportError: libNVVM cannot be found
Could not find module 'nvvm.dll' (or one of its dependencies)
```

This rules out replacing the NVIDIA driver as the first response. The driver
and CUDA context are already working; the missing compiler/toolkit is the
blocker.

### Installation result

The manual NVIDIA CUDA Toolkit `12.9` installer was completed, followed by the
small wrapper installation in the Hermes venv:

```text
CUDA Toolkit: 12.9.41
NVVM: True
Numba CUDA: True
cuda-bindings: 12.9.7 (matched to the CUDA 12.9 toolkit)
GPU: NVIDIA GeForce RTX 3060, compute capability 8.6
@cuda.jit smoke kernel: PASS
PyTorch 2.5.1+cu121 CUDA: True
```

The old `libNVVM cannot be found` failure is resolved. Open a new PowerShell
after the installer so the CUDA Toolkit PATH is visible; scripts may also set
`CUDA_HOME` and prepend `CUDA_HOME\bin` and `CUDA_HOME\nvvm\bin` explicitly.
On Windows, preload the toolkit `nvJitLink_120_0.dll` before PyTorch loads its
older same-named DLL; `smart_fib_cuda_bootstrap.py` now does this.

### Preferred installation path

Use the project venv only. The current NVIDIA Numba-CUDA documentation provides
the CUDA 12 package extra:

```powershell
$py = "C:\Users\user\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
& $py -m pip install "numba-cuda[cu12]"
```

If the package download times out, retry with a longer network timeout and
retries rather than installing globally:

```powershell
& $py -m pip install --timeout 300 --retries 5 "numba-cuda[cu12]"
```

The current package resolver selects a CUDA 12 toolkit/compiler dependency
set. The installed NVIDIA driver is newer than CUDA 12 and is expected to run
it, but PyTorch must be rechecked after installation. Do not assume that
installing the package is harmless just because the driver is healthy.

### Alternative installation path

Install a matching CUDA 12 Toolkit using the NVIDIA installer, then point the
venv at the toolkit root before importing Numba:

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
$env:PATH = "$env:CUDA_HOME\bin;$env:CUDA_HOME\nvvm\bin;$env:PATH"
```

The exact `v12.x` directory must exist and contain `bin\nvcc.exe` and the NVVM
compiler libraries. On Windows, do not rely on Linux-only CUDA minor-version
compatibility instructions; Numba documents MVC as unsupported on Windows.

### Verification gate after installation

Run all checks in a fresh process:

```powershell
& $py -c "from numba import cuda; from numba.cuda.cudadrv import nvvm; print('driver=', cuda.current_context().device.name); print('nvvm=', nvvm.is_available()); print('numba_cuda=', cuda.is_available()); cuda.detect()"
& $py -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then compile a one-element `@cuda.jit` smoke kernel and run the existing
float64 state-scan parity suite. Numba is not considered enabled until:

- NVVM is true.
- `cuda.is_available()` is true.
- A kernel compiles and executes.
- PyTorch CUDA still works in a separate process.
- The 5-day smoke and event-level parity gates pass.

## Faster Accurate Design With Numba Enabled

The PyTorch vectorized scan is a correctness baseline. The real speed path is
one compiled kernel, not more Python-level function fusion:

1. Convert OHLC inputs once to contiguous time-major SoA arrays `[T, series]`.
2. Launch one CUDA thread per independent day/contract series.
3. Loop over time inside that thread to preserve causal order.
4. Keep ATR/UT scalar state and fixed S1/ATR ring buffers in registers or small
   local arrays.
5. Write only the required outputs, preferably `[T, series]` color/S1/event
   arrays.
6. Use 128 and 256 threads per block as the first launch candidates.
7. Keep `fastmath=False` and float64 for the audit kernel.
8. Inspect register spills and local-memory traffic before increasing block
   size.

This removes the hundreds of per-time-step PyTorch launches and the repeated
temporary tensor traffic. It is the optimization that can materially improve
the `1.09x` preprocessing result. It must remain behind an explicit feature
flag until it matches the CPU oracle; installation alone is not permission to
replace the reference path.

For the full Optimus engine, prioritize the proven runbook path first:

- Keep the `_finalize` scalar-readback fix.
- Keep the full dataset and precomputed indicators resident in VRAM.
- Keep fixed `(B,N,T)` trial batches and `PROCS=1` on a 12 GB card.
- Cache indicator tensors by `(timeframe, period)` instead of recomputing them
  per trial.
- Keep matrix-first first-hit exits and chronological `argmax` semantics.
- Move only aggregate summaries back to the host during search.
- Use `torch.cummax` only for a trailing barrier whose associativity has been
  proven against the CPU implementation.
- Optimize CPU preparation separately with shared raw-day caching and projected
  columnar reads; the full-window measurement is about `680.8 s` CPU prep versus
  `3.3 s` CUDA grid evaluation.

## GPU-Only Optimus Architecture

"GPU-only" should mean all repeated strategy computation runs on the GPU. Raw
CSV parsing and disk reads are still a one-time host staging operation; forcing
CSV parsing through a GPU dataframe on this Windows/RTX 3060 setup would add a
new dependency and is not the main bottleneck once the raw tensors are cached.

The efficient architecture is:

1. Parse each source file once into a normalized host cache.
2. Pack all index OHLC as `[day, time]` and all reachable option OHLC as
   `[day, contract, time]` float64 arrays.
3. Pin and transfer the complete raw tensor set once.
4. Flatten independent index/option/variant streams into a series axis and run
   one Numba CUDA kernel per state-machine family.
5. Emit GPU event masks and compact event metadata, including dynamic strike
   lookup, without Python dictionaries or per-variant CPU extraction.
6. Evaluate all exit configurations in fixed `(B, day, time)` batches using the
   resident matrix-first engine.
7. Keep train/validation masks on GPU for walk-forward folds; never reload or
   reparse the raw archive for each fold.
8. Copy only aggregate summaries during search. Copy trade traces only for the
   winner and parity audits.

The target tensor shapes are:

```text
index_ohlc       [D, T, 4]
option_ohlc      [D, C, T, 4]
variant_signals  [V, D, T]
exit_parameters  [B]
```

For a custom Numba kernel, use time-major views `[T, series]` internally so a
warp reads adjacent series values at the same time. Use one thread per series,
not one thread per time bar: time is causally sequential inside each thread,
while days/contracts/variants are independent across threads.

### GPU state-machine rules

- Fuse ATR, UT color, S1, swing state, and setup state only when the complete
  event stream is compared with the CPU oracle.
- Use fixed local ring buffers for known periods; do not allocate Python lists,
  deques, dictionaries, or objects inside kernels.
- Use integer arithmetic for slot/strike routing where possible.
- Write fixed-width masks first; compact events with a deterministic stable
  prefix-sum pass only if the sparse matrix is too large.
- Keep the CPU implementation available as a shadow oracle for every fold.
- Chunk long kernels on Windows/WDDM to avoid watchdog/TDR resets.

### Walk-forward without CPU reprocessing

Prepare the full raw tensor cache once. For each expanding fold:

- Apply an in-device training mask to the same resident arrays.
- Evaluate the custom parameter batch on train days.
- Select the best train score using net points minus the configured drawdown
  penalty.
- Evaluate that one selected configuration once on the unseen validation mask.
- Stitch each validation day exactly once in chronological order.

No fold should call `collect_variant_cpu_dataset` for every variant. That is the
slow path the aborted Numba run exposed.

## Optimus Do's

- Do profile the end-to-end stages separately: CPU prep, H2D, kernels, D2H,
  finalization, and serialization.
- Do benchmark the runbook engine at `B=1`, `B=50`, and `B=100` with identical
  data and masks before claiming a speedup.
- Do keep all invariant data in VRAM and avoid CSV/tensor creation inside trial
  loops.
- Do use 3D `(B,N,T)` parameter batching for independent trials.
- Do keep `PROCS=1` on the RTX 3060; multiple spawned processes create multiple
  CUDA contexts and can exhaust VRAM.
- Do use `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` before importing
  PyTorch in the grid scripts.
- Do use contiguous SoA layouts and benchmark time-major views for custom
  kernels.
- Do convert GPU scalar thresholds/caps to host scalars once before a sequential
  host loop, never inside the loop.
- Do keep float64 data and controls in the audited path when comparison
  boundaries matter.
- Do use left-only causal padding and closed-bucket clock alignment.
- Do retain dynamic ATM strike selection, fees, slippage, position locks, and
  circuit breakers.
- Do run the 5-day smoke, three-date parity, event-level parity, and full-window
  audit before publishing a result.
- Do record GPU driver, toolkit, PyTorch, Numba-CUDA, dtype, batch size,
  allocator, data fingerprint, and git/source hash.

## Optimus Don'ts

- Do not compare a preprocessing microbenchmark with the runbook's full-engine
  `59x` or suite-level `404x` claims.
- Do not call `.item()`, `bool(tensor)`, `float(tensor)`, or `.tolist()` inside
  a per-trade, per-element, or per-day hot loop.
- Do not assume a Python function called "fused" is a fused CUDA kernel.
- Do not add CUDA Graphs without profiling and fixed-shape validation; the
  existing matrix engine has variable-size gathers and a documented graph
  capture failure.
- Do not enable TF32, FP16, BF16, AMP, fast math, approximate intrinsics, or
  unordered floating-point reductions in the audited path.
- Do not use floating-point atomics for P&L, fees, ATR, or indicator sums.
- Do not install CUDA packages globally or mix a system Python with the Hermes
  venv.
- Do not install a random CUDA toolkit version to fix Numba; match the intended
  CUDA major version and verify PyTorch afterward.
- Do not mix `cuda-bindings 13.3.x` or an unpinned pip `nvjitlink` DLL with the
  CUDA 12.9 toolkit; use the matching `cuda-bindings 12.9.7` and toolkit DLL.
- Do not treat `cuda.detect()` alone as proof that Numba kernels can compile;
  check NVVM and run a real kernel.
- Do not wire the new GPU event extractor into the historical oracle until its
  complete event stream matches `extract_day_events`.
- Do not use center/right padding, forward-fill future option prices, shuffled
  time-series validation, static ATM strikes, or overlapping positions.
- Do not run a full multi-year backtest before the smoke test.
- Do not call the CPU signal extractor once per variant when a shared raw tensor
  cache and GPU-batched state scan can reuse the same OHLC data.
- Do not confuse GPU-resident tensors with GPU-only execution if Python is still
  parsing files, constructing dictionaries, or looping over variants/trades in
  the hot path.

## Current Bottleneck

The strongest measured bottleneck is CPU preparation, not the matrix exit
kernel:

- Full-window float64 grid CPU preparation: about `680.8 s`.
- Same grid CUDA evaluation: about `3.3 s`.
- Peak allocated VRAM: about `6.45 GB`; reserved: about `7.07 GB`.

Therefore, optimizing only CUDA arithmetic will not materially improve total
wall time until repeated CSV parsing, Python object construction, and repeated
signal preparation are reduced.

The experimental `smart_fib_gpu_statescan.py` also has a separate local
bottleneck: `.item()`, `.tolist()`, and per-series Python loops force repeated
GPU/CPU synchronization. The safe immediate fix is a vectorized PyTorch
time-scan. The faster path is now available: one fused Numba CUDA kernel per
series, with the CPU/PyTorch implementations retained as parity oracles.

## Priority 1: Reduce CPU Preparation

### Shared normalized raw-day cache

Normalize each index and option day once, then reuse it across all Smart Fib
signal variants. Do not reread and reparsed the same CSVs for each S1, zone,
span, age, or touch-buffer combination.

Store a cache fingerprint containing:

- Source root and file paths.
- File size and modification time, or a content hash.
- Schema and row counts.
- Cache version.
- Date range and sorted day list.

The current cache identity checks should be extended with source-content
identity so stale results cannot silently survive data changes.

### Projected columnar storage

For repeated full-window runs, convert the raw archive to projected Parquet or
Arrow partitions by date and contract family. Read only timestamp, symbol,
open, high, low, close, and fields required by the selected strategy.

Use Polars lazy scans or DuckDB predicate/projection pushdown. Do not build a
giant all-strike wide table; the current compact reachable-contract axis is a
better fit.

### Fold-specific compaction

For walk-forward runs, compact the train/validation day set before constructing
future-bar and event matrices. Do not build outcomes for all resident events
when a fold mask will discard most of them later.

### Preserve warmup semantics

Prior-day warmup must be retained. Never use broad forward-fill, interpolation,
or backfill that can introduce future information into an indicator or option
fill.

## Priority 2: Faster Causal State Scans

### Safe PyTorch baseline

Use a Python loop over time and vectorized tensors over independent series. The
loop over time is required by the ATR, UT trailing stop, stochastic, and other
causal recurrences. Remove all of the following from the hot path:

- `.item()`
- `.tolist()`
- `bool(cuda_tensor)`
- Per-step `torch.nonzero`
- Per-series Python loops
- Per-step allocations
- Repeated `.to(device)` calls

Use time-major, structure-of-arrays tensors for custom scans:

```text
[time, series]
```

At a fixed time, neighboring threads then read neighboring series values. The
current `[series, time]` layout is convenient for Python but causes strided
loads when a kernel maps one thread to one series.

### Fused Numba CUDA path

When the driver path is available, use one `@cuda.jit` kernel with one thread
owning one complete series:

```text
for series in grid(1):
    initialize scalar state and small ring buffers
    for time in range(T):
        update ATR
        update UT color
        update S1 stochastic
        write only required outputs
```

This preserves causal time order while parallelizing across independent days,
contracts, or option series. Use fixed-size local ring buffers for the known
periods. Keep scalar state in registers where possible and inspect register
spills after compilation.

Use small inline device helpers for true range, Wilder ATR, UT stop/color, and
rolling stochastic. Do not use Python classes, dictionaries, deques, or dynamic
lists inside a CUDA kernel.

### State-scan memory rules

- Use no shared memory for per-series ATR or stochastic state; neighboring
  threads do not reuse it.
- Use shared memory only for genuine block-level reuse, reductions, scans, or
  compaction.
- Avoid floating-point atomics.
- If events must be compacted, use deterministic flags, counts, an exclusive
  prefix sum, and stable chronological scatter.
- Specialize kernels for the known S1 variants `(9,3)`, `(12,3)`, `(14,3)`, and
  `(12,4)` if dynamic period bounds cause register pressure.

### Exact state semantics

The GPU implementation must preserve:

- ATR NaN/warmup behavior.
- First ATR value as the simple mean of the first period true ranges.
- Wilder recurrence after warmup.
- Invalid bars not advancing state.
- UT `previous_source` updating during ATR warmup.
- UT warmup color using the previous trailing stop without changing the stop.
- Flat stochastic windows returning `50.0`.
- S1 output only after both K and D warmups.
- Chronological rolling-sum order where exact floating-point parity matters.

## Priority 3: Matrix-First Exit Engine

The current matrix-first design is the right shape for GPU execution:

- Pack sparse events once.
- Keep OHLC tensors resident.
- Gather future bars in batches.
- Use `argmax` for the first chronological stop/target hit.
- Use `cummax` only for a monotonic trailing barrier where the CPU semantics
  are proven equivalent.
- Keep the position lock, daily loss cap, consecutive-loss breaker, and
  chronological re-entry logic in a deterministic ordered pass.

Do not replace a sequential position-lock loop with unordered reductions. A
GPU result with the same aggregate P&L but a different event order is not a
valid parity result.

### Reduce device-to-host traffic

During grid or Optuna search, return aggregate metrics only. Copy daily arrays,
trade traces, and detailed exits only for:

- Parity checks.
- Top candidates.
- Final audit reports.

Avoid synchronizing inside the time-slot loop. CUDA events are appropriate for
benchmark timing, but timing synchronization should be optional in production
runs.

### Tune batch size empirically

Benchmark at least `32`, `64`, `100`, `128`, and `256`. Larger batches create
larger temporary `(B, events, future)` masks and can reduce throughput or
exhaust VRAM. Choose using measured end-to-end wall time, peak VRAM, and parity,
not theoretical occupancy alone.

### Layout experiments

The current contract arrays use `(day, contract, time)` and sometimes expose a
permuted `(day, time, contract)` view. Benchmark a contiguous copy for the
future-bar gather before duplicating all float64 OHLC arrays; the RTX 3060 has
limited headroom after the full float64 grid allocation.

## Precision And Determinism

### Audit mode

Use float64 consistently for:

- OHLC.
- ATR and indicator state.
- Fibonacci levels.
- Entry and exit prices.
- Fees, points, rupee P&L, and drawdown.
- Parameter/control tensors that participate in comparisons.

The current grid's float64 market arrays should not be paired with float32 stop,
target, or threshold controls when strict parity is required.

Disable or avoid in the audit path:

- FP16.
- BF16.
- AMP.
- TF32/high-precision FP32 matmul modes.
- `fastmath`.
- Approximate math intrinsics.
- Unordered floating-point reductions.
- Floating-point atomics.

Floating-point addition is not associative. Kernel fusion, FMA formation,
parallel reductions, and changed accumulation order can alter threshold
decisions even with float64.

### Throughput mode

A separate exploratory float32 path may be useful, but it must be clearly
marked non-audited and validated against the full smoke window. It must not
overwrite or replace the float64 audit artifact.

## Launch And Occupancy Guidance

For custom kernels:

- Start with 128 and 256 threads per block.
- Test 64 and 512 only after profiling.
- Use block sizes that are multiples of 32.
- Ensure enough blocks to keep all SMs busy.
- Inspect achieved occupancy, register count, local-memory traffic, global
  memory efficiency, warp stalls, and branch efficiency.
- Do not maximize occupancy blindly; register spills can make a nominally
  high-occupancy kernel slower.
- Avoid one giant long-running kernel on Windows/WDDM because of TDR risk.

## Streams, Pinned Memory, And CUDA Graphs

The current one-time pinned host transfer and resident-GPU design is correct.
Streams help only when independent CPU staging, H2D, kernel, and D2H work can
overlap. Use explicit events and dependencies; do not add streams by default.

CUDA Graphs are conditional. They require fixed shapes, stable pointers, no
allocations, and no host synchronization inside the captured region. Current
variable event counts and batch shapes make graph capture a later experiment,
not the first optimization.

Zero-copy and texture memory are low-priority or unsuitable for repeatedly
accessed discrete-GPU OHLC data.

## Validation Protocol

Every optimization must pass, in order:

1. Five-day smoke test.
2. Indicator-level parity.
3. Event-level parity: event minute, pattern, side, strike, Fib bounds, and
   entry price.
4. Trade-level parity: entry minute, exit minute, reason, prices, fees, and
   position sequence.
5. Three-date GPU/CPU parity gate.
6. Full-window audit for the winning configuration.

Add regression cases for:

- Missing bars and invalid padding.
- Prior-day warmup.
- Closed 5m bucket timing.
- Dynamic strike selection.
- Same-bar stop/target ties.
- EOD exits.
- Missing option bars.
- Fallback targets.
- Consecutive-loss and daily-loss breakers.
- Stable contract-slot ordering.

Record these with every run:

- Git/source hash.
- Data-root and source-file fingerprint.
- Date range and sorted days.
- Python, NumPy, Torch, Numba, CUDA, driver, and GPU versions.
- Dtype and TF32 policy.
- Allocator and batch size.
- Worker count.
- Fees, slippage, lot size, and rounding policy.
- Parity tolerances and failed experiment notes.

## Recommended Implementation Order

1. Keep a strict float64 audit configuration and fix float32 control tensors.
2. Remove `.item()`-driven synchronization from the state-scan baseline.
3. Fuse ATR, UT, and S1 in one vectorized time-scan and validate it.
4. Fix Numba CUDA driver availability, then benchmark a fused one-thread-per-
   series kernel against the PyTorch baseline.
5. Add shared normalized raw-day caching and projected columnar reads to reduce
   the dominant CPU preparation time.
6. Compact fold-specific event data before future-bar matrix construction.
7. Split aggregate-only and detailed device-to-host result paths.
8. Benchmark time-major/contiguous layouts and batch sizes.
9. Profile with Nsight Systems, then Nsight Compute on representative kernels.
10. Consider custom fusion, streams, or CUDA Graphs only after all parity gates
    pass.

## Techniques To Keep Out Of The Audit Path

- FP16/BF16/AMP.
- TF32.
- Fast math and approximate intrinsics.
- Floating-point atomics.
- Unordered P&L or indicator reductions.
- Randomized or shuffled time-series validation.
- Broad forward-fill or interpolation of missing option OHLC.
- Full VectorBT/Zipline replacement of the stateful event machine.
- DDP, AllReduce, distributed training, TensorRT, pruning, and quantization.
- Zero-copy for the resident RTX 3060 OHLC dataset.
- Multiple CUDA processes competing for one GPU.

## Source Coverage

All files listed in `C:\Users\user\Downloads\pdf_text_md\manifest.json` were
reviewed. The two `CUDA by Example` files are duplicate extractions of the
same title and were compared for consistency.

| Reference | Main useful contribution |
|---|---|
| GPU-Accelerated Computing with Python 3 and CUDA | Numba kernel structure, device helpers, time/series mapping, streams, events, PTX and register inspection. |
| Parallel Programming with Python, 2nd ed. | CUDA grid mapping, coalescing, streams, shared memory, launch benchmarking, Numba examples. |
| Michaeli Daniel 2025 | Work/span analysis, dependency-chain limits, batching, partitioning, register pressure, divergence. |
| CUDA Programming | CUDA execution model, memory hierarchy, synchronization, profiling, precision, atomics. |
| CUDA C Best Practices Guide | Coalescing, SoA, alignment, bank conflicts, occupancy, pinned transfers, effective bandwidth, determinism. |
| GPU-Accelerated Deep Learning | Profiling workflow, fusion, occupancy, memory access, streams, reproducibility; training-only sections were excluded. |
| CUDA by Example (both extracts) | Practical kernels, launch geometry, streams, shared/constant memory, timing and debugging. |
| Mastering CUDA Python Programming | Numba CUDA kernels, local/register state, scans/reductions, streams, graphs, precision boundaries. |
| GPU Parallel Program Development Using CUDA | Thread/block mapping, shared memory, reductions, streams, synchronization, occupancy. |
| Python for Algorithmic Trading Cookbook | Parquet/Polars/DuckDB, causal rolling windows, event simulation, walk-forward validation, costs. |
| Handbook of AI and Big Data Applications in Investments | Data leakage controls, reproducibility, execution costs, GPU use, review discipline. |
| Hands-On AI Trading with Python, QuantConnect, and AWS | Iterative smoke testing, time-series splits, costs, logging, regression testing. |
| Wei et al. 2025 | Vectorization, sparse workload chunking, stable initialization, and explicit invalid-state handling; runtime numbers are not transferable. |

## Evidence Limits

The references provide principles and implementation patterns, not measured
performance for this exact Smart Fib workload. All speed claims must be
verified on this RTX 3060 with the repository's real dates, contract layout,
fees, and parity gates.

## Web Sources Consulted

Official documentation consulted during the Numba/CUDA investigation:

- Numba CUDA overview: <https://numba.readthedocs.io/en/stable/cuda/overview.html>
  explains that the built-in CUDA target is deprecated in favor of the NVIDIA
  `numba-cuda` package and that CUDA Toolkit/NVVM is required.
- NVIDIA Numba-CUDA installation: <https://nvidia.github.io/numba-cuda/user/installation.html>
  documents `pip install "numba-cuda[cu12]"`, CUDA 12/13 support, and toolkit
  path discovery.
- NVIDIA Numba-CUDA kernel guide: <https://nvidia.github.io/numba-cuda/user/kernels.html>
  documents one-thread-per-series kernel launches, grid bounds checks, global
  memory, local memory, and block-size selection.
- Numba CUDA minor-version compatibility:
  <https://numba.readthedocs.io/en/stable/cuda/minor_version_compatibility.html>
  explicitly notes that CUDA minor-version compatibility is not supported on
  Windows, so a matching toolkit path must be used rather than Linux MVC
  workarounds.
- PyTorch `torch.compile` guide:
  <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>
  confirms that compile can fuse/reduce Python overhead but graph breaks and
  dynamic control flow must be measured and validated; it is not a substitute
  for a real custom recurrent CUDA kernel.
- Local Optimus runbook:
  `artifacts/f6_hybrid/OPTIMIZED_GPU_BACKTEST.md`, especially sections 4 and 5,
  documents the verified `59.3x` scalar-readback fix, batching rules, parity
  gates, and prohibitions.
- Local GPU pipeline guide:
  `GPU_BACKTEST_PIPELINE_GUIDE.md`, especially sections 20-23, documents
  resident VRAM, 3D batching, matrix exits, causal checks, and the high-level
  suite speedup claims.

## Latest GPU-Only Search

The validated full search used `smart_fib_optimus_grid_gpu.py` with resident
float64 variant caches, `B=100`, matrix-first exits, and no CPU signal rebuild:

- Non-WF: `675` configurations, `25.894s` wall, `2.827s` CUDA grid time.
- Maximum-net champion: `+21,130.25` points, `190.23` DD points.
- Expanding WFO: `45.671s` wall, `7,554` stitched OOS trades,
  `+14,067.15` OOS points, `190.23` DD points.
- The exact parameter and fold artifacts are recorded in
  `SMART_FIB_OPTIMUS_RESULTS_2020_2026.md`.

The aborted `smart_fib_optimus_numba.py` CPU replay was removed. Numba-CUDA is
still retained for the future GPU state-machine kernel, but the proven
backtest result uses the runbook's 3D resident-GPU engine as required.
