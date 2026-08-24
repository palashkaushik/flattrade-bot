# Smart Fib Optimus Results

## Run

- Data: `2020-01-01` through `2026-05-05`.
- Available sessions: `1,574`.
- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM, CUDA 12.1.
- Tensor contract: `(B,N,T)` with `B=100`, `N=1,574`, `T=375`, `C=72`.
- Exact CPU/GPU parity: three dates, both stop levels, zero trade-count difference,
  maximum `0.05` point tolerance.
- Smart Fib entry contract was fixed during GPU optimization: combined stream,
  min span `15`, touch buffer `0`, setup age `45`, target `0.29`, threshold `10`,
  fallback `0.0`.
- GPU-safe search axis: stop extension `1.155` or `1.25`.

## Requested Zone/Exit Smoke

This measured smoke used the first five available sessions,
`2020-01-01` through `2020-01-07`, five staged zone-aware variants, and 30
unique configurations. It is not full-period validation, not walk-forward
selection, and must not be used as 2020-2026 performance evidence.

- Zones represented: `(0.618,1.0)`, `(0.618,0.786)`, `(0.786,1.0)`,
  `(0.5,0.786)`, `(0.705,0.886)`.
- Selected exit axes: target `0.29`, fallback `0.0`, thresholds `5/10/15`,
  stops `1.155/1.25`.
- Full bounded axes accepted by the CLI: targets `0.0/0.236/0.29/0.382/0.5`,
  fallback `0.0/0.236` constrained by `fallback <= target`, and stops
  `1.13/1.155/1.25/1.272/1.382/1.618`.
- Dynamic 10-point fallback remains enabled only for primary target `0.29`.
- Exactly three parity dates were checked for every selected target/fallback/
  threshold/stop combination; all 30 comparisons passed.

| Rank | Zone-aware variant | Target | Fallback | Threshold | Stop | Trades | Win rate | Net points | Net Rs | DD points | Fees Rs | Score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `s1k12d4_span15_age45_buf0p5_z0p5-0p786` | 0.29 | 0.0 | 5 | 1.25 | 13 | 69.23% | +55.30 | +3,518.87 | 19.78 | 75.63 | +51.34 |
| 2 | `s1k12d4_span15_age45_buf0p5_z0p5-0p786` | 0.29 | 0.0 | 10 | 1.25 | 12 | 75.00% | +48.10 | +3,056.84 | 13.54 | 69.66 | +45.39 |
| 3 | `s1k12d4_span15_age45_buf0p5_z0p5-0p786` | 0.29 | 0.0 | 15 | 1.25 | 12 | 75.00% | +48.10 | +3,056.84 | 13.54 | 69.66 | +45.39 |
| 4 | `s1k9d3_span10_age30_buf0p5_z0p618-0p786` | 0.29 | 0.0 | 10 | 1.155 | 20 | 60.00% | +38.25 | +2,343.17 | 20.29 | 143.08 | +34.19 |
| 5 | `s1k9d3_span10_age30_buf0p5_z0p618-0p786` | 0.29 | 0.0 | 15 | 1.155 | 20 | 60.00% | +38.25 | +2,343.17 | 20.29 | 143.08 | +34.19 |

### Smoke GPU Evidence

- Device: NVIDIA GeForce RTX 3060, 12 GB VRAM, CUDA 12.1.
- Total wall time: `35.610s`; CPU preparation: `30.760s`.
- Grid CUDA time: `2,310.480ms`; grid wall time: `2.320s`.
- Parity CUDA time: `2,237.844ms`; parity covered exactly three dates.
- Batch size: `100`; allocator: `cudaMallocAsync`; peak allocation: `6.67MB`;
  peak reserved: `32.00MB`.
- Exit axes were evaluated through the existing matrix-first CUDA engine; no
  CPU trial loop called `process_day`.

## Non-Walk-Forward

The 3,000-trial run produced only two unique configurations because the other
Smart Fib rules were intentionally fixed for parity.

| Rank | Stop | Trades | Win rate | Net points | Net Rs | Max DD points | PF | Fees Rs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.155 | 6,372 | 46.06% | +7,057.05 | +391,213.44 | 465.25 | 1.4219 | 67,494.78 |
| 2 | 1.25 | 6,061 | 48.89% | +6,128.70 | +334,074.44 | 471.50 | 1.3248 | 64,291.06 |

Repeated Optuna trials with the same stop value are not separate parameter
configurations and are excluded from this table.

## Prior Additional-Parameter Grid Probe

The earlier bounded probe used the first five available sessions in
`2020-01-01..2020-01-07`, four staged signal variants, `B=100`, and all six
exit combinations (`threshold=5/10/15`, `stop=1.155/1.25`). It evaluated 24
unique configurations. The ranking score was
`net_points - 0.20 * max_drawdown_points`, with a minimum-trade guard of one.

| Rank | Variant | S1 | Span/Age/Buffer | Threshold | Stop | Trades | WR | Net points | Net Rs | DD points | PF | Fees Rs | Score |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `s1k9d3_span10_age30_buf0p5` | 9,3 | 10/30/0.5 | 15 | 1.155 | 21 | 61.90% | +57.25 | +3,577.66 | 24.09 | 7.6052 | 143.59 | +52.43 |
| 2 | `s1k9d3_span10_age30_buf0p5` | 9,3 | 10/30/0.5 | 10 | 1.155 | 21 | 61.90% | +55.30 | +3,451.04 | 22.15 | 7.3715 | 143.46 | +50.87 |
| 3 | `s1k9d3_span10_age30_buf0p5` | 9,3 | 10/30/0.5 | 5 | 1.155 | 21 | 61.90% | +54.50 | +3,399.09 | 21.35 | 7.2756 | 143.41 | +50.23 |
| 4 | `s1k9d3_span10_age30_buf0p5` | 9,3 | 10/30/0.5 | 15 | 1.25 | 21 | 57.14% | +53.90 | +3,360.13 | 24.09 | 7.2036 | 143.37 | +49.08 |
| 5 | `s1k9d3_span10_age30_buf0p5` | 9,3 | 10/30/0.5 | 10 | 1.25 | 21 | 57.14% | +51.95 | +3,233.51 | 22.15 | 6.9699 | 143.24 | +47.52 |

All four selected variants passed CPU/GPU parity on exactly
`2020-01-01`, `2020-01-02`, and `2020-01-03`, for every one of the six
threshold/stop pairs. Trade counts and wins matched exactly; net points, net
rupees, fees, and per-day drawdown stayed within the strict `0.05` tolerance.
The baseline was `S1=(12,3)`, `min_span=15`, `setup_max_age=45`, and
`touch_buffer=0.0`.

### Probe Timing

| Variant | CPU prep seconds | Parity CUDA ms | Grid CUDA ms | Grid wall seconds |
|---|---:|---:|---:|---:|
| `s1k12d3_span15_age45_buf0` | 4.116 | 536.321 | 323.910 | 0.326 |
| `s1k9d3_span10_age30_buf0p5` | 4.338 | 364.984 | 354.352 | 0.356 |
| `s1k14d3_span20_age60_buf1` | 4.747 | 430.186 | 461.540 | 0.464 |
| `s1k12d4_span15_age45_buf0p5` | 4.929 | 480.552 | 481.081 | 0.484 |
| **Total** | **18.129** | **1,812.043** | **1,620.883** | **1.629** |

GPU evidence: NVIDIA GeForce RTX 3060, CUDA 12.1, `cudaMallocAsync`, pinned
host transfer, `torch.inference_mode`, and `B=100`. Peak allocation was 7.17
MB with 32.00 MB reserved for this five-day probe.

This is a bounded in-sample probe, not a 2020-2026 result or walk-forward
selection. The full signal axes contain 108 combinations per zone and 540
zone-aware CPU variants (`5*4*3*3*3`). The full exit axes contain 162 valid
combinations (`9` target/fallback pairs x `3` thresholds x `6` stops), or
87,480 combined signal/exit configurations. Signal extraction remains the
one-time CPU oracle; only repeated target/fallback/threshold/stop exit
evaluation is GPU-generated. The expanded grid is intentionally not run by
default and still requires data coverage checks, the five-day smoke test, and
three-date parity for every selected variant.

## Walk-Forward

Each fold selected its stop only from the preceding training window. Validation
results were then stitched without future parameter selection.

| Validation year | Selected stop | Trades | Win rate | OOS net points | OOS net Rs | Max DD points | PF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 1.25 | 932 | 47.85% | +898.60 | +49,381.55 | 369.63 | 1.3408 |
| 2022 | 1.155 | 1,035 | 45.41% | +742.30 | +36,883.37 | 465.25 | 1.2390 |
| 2023 | 1.155 | 836 | 45.57% | +911.00 | +52,753.93 | 235.16 | 1.5286 |
| 2024 | 1.155 | 1,054 | 48.39% | +1,758.80 | +101,953.50 | 290.67 | 1.6148 |
| 2025 | 1.155 | 1,100 | 46.18% | +963.05 | +50,536.35 | 396.22 | 1.3019 |
| 2026 through May 5 | 1.155 | 356 | 46.07% | +451.70 | +24,154.69 | 285.03 | 1.3416 |
| **Stitched OOS** | — | — | — | **+5,725.45** | **+315,663.39** | — | — |

## Full-Window Float64 Grid (810 configs)

Bounded full-window grid over all `1,574` sessions (`2020-01-01`..`2026-05-05`),
float64 precision (cache `smart_fib_grid_cache_full_float64`), GPU CUDA evaluation.
Result file: `smart_fib_optimus_grid_gpu_full_staged_float64.json`.

- Configs evaluated: `810`; unique configurations: `810`.
- Mode: `bounded_full_window`; execution contract: real option OHLC, dynamic
  CE/PE ATM±50/100, one global position, unlimited sequential re-entry, 1-point
  slippage, lot `65`, fees enabled.
- GPU evidence: RTX 3060 12 GB, CUDA 12.1, `cudaMallocAsync`; peak allocated
  `6,452.6 MB`, peak reserved `7,072.0 MB`.
- Timing: total wall `761.325s`; CPU prep `680.836s`; grid CUDA
  `3,202.811ms`; grid wall `3.302s`; parity CUDA `4,052.438ms`.
- Parity: `exactly_three_dates`, `all_variants_passed = true`. The lead config's
  full-window GPU numbers **exactly match** the prior full CPU audit
  (`11,060` trades / `+21,130.25` points / `+Rs1,255,575.51`), confirming
  full-window parity for the winning configuration.

### Lead configuration

| Field | Value |
|---|---|
| Variant | `s1k12d4_span15_age45_buf0p5_z0p5-0p786` |
| Exit | `threshold5 | target0.5 | fallback0 | stop1.13` |
| Trades | `11,060` |
| Wins | `8,322` |
| Win rate | `75.24%` |
| Net points | `+21,130.25` |
| Net Rs | `+1,255,575.51` |
| Max DD points | `190.23` |
| Max DD Rs | `12,365.17` |
| Profit factor | `5.043` |
| Fees Rs | `117,890.74` |
| Composite score | `21,092.204` |

### Top-five ranking artifact (caution)

All five `top_five` entries collapse to **identical** trades/points/RS because
they differ only in `threshold` (`5/10/15`) and `fallback` (`0/0.236`), which are
**inert at `target = 0.5`**. The grid's genuinely distinct winners are hidden
behind this ranking artifact — they live in the S1-period / zone / stop axes, not
in threshold/fallback. A second-stage grid must disable the `target=0.5`
collapse (e.g. fix `threshold` and `fallback`, sweep `stop`/`zone`/`S1`) to
surface real differentiation. The `target=0.5` (unlimited-profit) regime is the
"cash machine": ~75% win rate, PF ~5, at the cost of large drawdown (190 pts).

## Custom 3D GPU Grid And Expanding Walk-Forward (2020-2026)

This requested run used the resident-tensor 3D CUDA engine, not the aborted CPU
Numba replay. The full float64 variant caches matched the desktop data-root
identity, so full-run CPU preparation was `0.0s`.

Artifacts:

- Non-WF: `artifacts/f6_hybrid/smart_fib_optimus_grid_3d_full_2020_2026.json`
- WFO: `artifacts/f6_hybrid/smart_fib_optimus_wf_gpu_2020_2026_exact2.json`
- Parameters: `artifacts/f6_hybrid/smart_fib_optimus_top_strategy_params_2020_2026.json`

Custom axes:

- Five staged signal variants.
- Targets: `0.0/0.236/0.29/0.382/0.5`.
- Fallbacks: `0.0/0.236`, constrained by `fallback <= target`.
- Thresholds: `5/10/15` points.
- Stops: `1.13/1.155/1.25/1.382/1.618`.
- Unique configurations: `675`; batch size `100`; all staged variants passed
  three-date parity.
- Full run wall time: `25.894s`; CUDA grid time `2.827s`; parity CUDA time
  `3.560s`; peak allocation `5,658.32 MB`.

### Non-WF maximum-net champion

| Signal variant | Target | Fallback | Threshold | Stop | Trades | WR | Net points | Net Rs | DD points | PF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `s1k12d4_span15_age45_buf0p5_z0p5-0p786` | 0.5 | 0.0 | 5 | 1.13 | 11,060 | 75.24% | **+21,130.25** | **+1,255,575.51** | 190.23 | 5.043 |

Threshold and fallback are inert at target `0.5`; the canonical parameter file
fixes them to threshold `5` and fallback `0.0` rather than treating duplicate
rows as separate winners.

### Lower-drawdown full-window candidate

With a minimum guard of `1,000` trades, the lowest-DD candidate was:

| Signal variant | Target | Fallback | Threshold | Stop | Trades | Net points | Net Rs | DD points | PF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `s1k14d3_span20_age60_buf1_z0p786-1` | 0.5 | 0.0 | 5 | 1.13 | 2,621 | +3,615.15 | +206,878.83 | **152.48** | 1.6045 |

### Expanding walk-forward OOS

| Validation window | Signal | Exit | Trades | Net points | Net Rs | DD points |
|---|---|---|---:|---:|---:|---:|
| 2022 | `s1k12d4_span15_age45_buf0p5_z0p5-0p786` | target 0.5 / stop 1.13 | 1,647 | +3,175.60 | +187,862.00 | 83.00 |
| 2023 | same | target 0.5 / stop 1.13 | 1,511 | +2,962.30 | +180,533.41 | 69.56 |
| 2024 | same | target 0.5 / stop 1.13 | 1,878 | +3,673.15 | +216,467.25 | 190.23 |
| 2025-01-01..2026-05-05 | same | target 0.5 / stop 1.13 | 2,518 | +4,256.10 | +247,411.16 | 129.00 |
| **Stitched OOS** | — | — | **7,554** | **+14,067.15** | **+832,273.82** | **190.23** |

WFO wall time was `45.671s` with cached tensors. The WFO result is the more
defensible strategy file; the full-window maximum is in-sample evidence and
must not be promoted live without further robustness checks.

## Fixed All-In Cost Rerun: Rs40 Per Trade

This rerun replaces the detailed brokerage/tax fee formula with the requested
fixed approximation of `Rs40` per completed trade, representing brokerage plus
taxes/charges. Slippage remains unchanged. The cost override was applied inside
both the GPU engine and CPU parity oracle before ranking.

Artifacts:

- Non-WF: `artifacts/f6_hybrid/smart_fib_optimus_grid_3d_full_2020_2026_fixed40.json`
- WFO: `artifacts/f6_hybrid/smart_fib_optimus_wf_gpu_2020_2026_fixed40.json`
- Parameters: `artifacts/f6_hybrid/smart_fib_optimus_top_strategy_params_2020_2026_fixed40.json`

### Non-WF top five distinct contenders

| Rank | Stop | Trades | WR | Net points | Net Rs | DD points | PF | Fixed costs Rs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.13 | 11,046 | 72.49% | **+21,077.05** | **+928,168.25** | 188.53 | 3.4031 | 441,840.00 |
| 2 | 1.155 | 11,003 | 72.85% | +20,921.40 | +919,771.00 | **185.18** | 3.3199 | 440,120.00 |
| 3 | 1.25 | 10,858 | 74.07% | +20,157.20 | +875,898.00 | 223.01 | 2.8951 | 434,320.00 |
| 4 | 1.382 | 10,640 | 75.23% | +19,376.95 | +833,901.75 | 298.06 | 2.6099 | 425,600.00 |
| 5 | 1.618 | 10,236 | 76.94% | +18,286.90 | +779,208.50 | 488.67 | 2.2262 | 409,440.00 |

All five use signal variant
`s1k12d4_span15_age45_buf0p5_z0p5-0p786` and target `0.5`. The lower-DD
candidate is stop `1.155`; the maximum-net candidate is stop `1.13`.

### Fixed-cost expanding WFO OOS

| Validation window | Stop | Trades | WR | Net points | Net Rs | DD points |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 1.13 | 1,647 | 72.25% | +3,175.60 | +140,534.00 | 98.29 |
| 2023 | 1.13 | 1,511 | 75.78% | +2,962.30 | +132,109.50 | 80.13 |
| 2024 | 1.13 | 1,874 | 70.97% | +3,661.10 | +163,011.50 | 185.00 |
| 2025-01-01..2026-05-05 | 1.13 | 2,508 | 69.62% | +4,214.95 | +173,651.75 | 157.21 |
| **Stitched OOS** | — | **7,540** | **71.76%** | **+14,013.95** | **+609,306.75** | **185.00** |

Fixed-cost WFO wall time was `44.198s`; all variant caches were loaded from the
matching desktop data-root identity and all four folds selected the same signal
variant and exit stop.

## Fine Sweep Around Champion (Fixed Rs40 Cost)

The fine sweep explored the previously ungridded exit region on the champion
signal variant `s1k12d4_span15_age45_buf0p5_z0p5-0p786`:
targets `0.382/0.5/0.618` x stops `1.05..1.30` step `0.025` (fallback `0`,
threshold `5`, 33 configs). The old grid capped targets at `0.5` and stops at
`1.13`, so this region was unexplored.

Artifacts: `artifacts/f6_hybrid/smart_fib_finesweep_fixed40.json` and
`artifacts/f6_hybrid/smart_fib_optimus_top_strategy_params_2020_2026_finesweep_fixed40.json`.

**New champion: `target 0.618, stop 1.05`** (tight 5% stop, wide 61.8% target).

### Non-WF improvement vs prior champion

| Config | Trades | WR | Net points | Net Rs | DD points | PF |
|---|---:|---:|---:|---:|---:|---:|
| Prior: target 0.5, stop 1.13 | 11,046 | 72.49% | +21,077.05 | +928,168.25 | 188.53 | 3.4031 |
| **New: target 0.618, stop 1.05** | **12,039** | **71.39%** | **+24,995.30** | **+1,143,134.50** | **163.50** | **10.1503** |

Gain: `+18.6%` net points, `+23.2%` net Rs, `-13.3%` drawdown vs prior champion.
Fixed costs at Rs40/trade: `481,560.00`.

### Fine-sweep expanding WFO OOS (all folds selected target 0.618, stop 1.05)

| Validation window | Trades | WR | Net points | Net Rs | DD points |
|---|---:|---:|---:|---:|---:|
| 2022 | 1,801 | 69.63% | +3,725.65 | +170,127.25 | 53.43 |
| 2023 | 1,611 | 76.91% | +3,407.40 | +157,041.00 | 50.82 |
| 2024 | 2,045 | 68.75% | +4,278.70 | +196,315.50 | 163.50 |
| 2025-01-01..2026-05-05 | 2,716 | 69.40% | +5,646.70 | +258,395.50 | 98.60 |
| **Stitched OOS** | **8,173** | **70.77%** | **+17,058.45** | **+781,879.25** | **163.50** |

WFO stitched gain vs prior (target 0.5/stop 1.13): `+21.7%` net points,
`+28.3%` net Rs, `-11.6%` drawdown. Wall time: `15.985s`. Fixed-cost fees OOS:
`326,920.00`.

Caveat: stop `1.05` allows only a 5% adverse move before exit; the profile
(tight stop, wide target) is what drives the high PF. Slippage is unchanged
(1 pt/leg).

## Wide-Target Sweep (Fixed Rs40 Cost) — TP-Widening Test

User asked to widen the TP to capture bigger points per trade. Sweep: targets
`0.618/0.786/1.0/1.272/1.618/2.0` x stops `1.05/1.13/1.272/1.382/1.618` (30
configs, fallback 0, threshold 5, champion signal, fixed Rs40). Wall `15.426s`.

Artifacts: `artifacts/f6_hybrid/smart_fib_wide_target_sweep_fixed40.json`,
`artifacts/f6_hybrid/smart_fib_optimus_top_strategy_params_2020_2026_wide_target_fixed40.json`.

### ⚠️ Mechanism finding — the exit axes are nearly inert

A 5-day CPU-oracle probe (`artifacts/f6_hybrid/probe_wide_target_exits.py`)
shows that the target/stop levels barely control the realized exit:

- For `target >= 1.0` the fib level lands on the **wrong side** of the entry
  (e.g., PE on a `low_to_high` setup: target = low + 1.272 x span is ABOVE the
  current price, so "bar low <= target" is trivially true) -> TP triggers on
  the next bar, every trade exits at the next bar's close -> **12 of the 30
  configs are byte-identical** (12,385 trades, +25,775.95 pts).
- For `0.618/0.786` the level is inside the span and typically reached within
  1-2 bars; the exit fills at the bar close +/- 1 pt slippage -> the realized
  capture is a ~1-bar move (~1-3 pts) regardless of level.
- Stops almost never fire (0 stops in the 5-day probe across all configs).
- **Conclusion: in this engine "wider TP" cannot produce larger points per
  trade — the knob is effectively `exit at next bar's close`.** A genuine
  wide-TP needs an engine exit-model change (close-based multi-bar TP,
  partial TP + trail, or trailing stop), which is a parity-affecting change.

### Non-WF results (full window) — new champion `target 0.786 / stop 1.13`

| Rank | Target | Stop | Trades | WR | Net pts | Net Rs | DD pts | PF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.786 | 1.13 | 12,380 | 70.93% | **+25,799.35** | **+1,181,757.75** | **127.21** | 18.7351 |
| 2 | 0.786 | 1.05 | 12,383 | 70.90% | +25,796.80 | +1,181,472.00 | 127.21 | 18.7450 |
| 3 | 1.0+ (any stop) | — | 12,385 | 70.84% | +25,775.95 | +1,180,036.75 | 127.21 | 18.7680 |
| 4 | 0.786 | 1.272 | 12,377 | 70.93% | +25,748.45 | +1,178,569.25 | 127.21 | 18.1925 |
| 5 | 0.786 | 1.382 | 12,374 | 70.94% | +25,726.35 | +1,177,252.75 | 127.21 | 18.0308 |

(All `target >= 1.0` rows are the same degenerate next-bar-exit behavior and
occupy ranks 3-19 with identical metrics.) The 0.618 family now ranks 26-30:
the prior champion (0.618/1.05) is +24,995.30 pts / DD 163.50 — the 0.786/1.13
row beats it by `+3.2%` net pts, `+3.4%` net Rs, and `-22.2%` drawdown, but
this gain comes from exit timing, not from wider per-trade capture.

### WFO OOS — all folds selected `target 0.786`

| Validation | Selected stop | Trades | WR | Net pts | Net Rs | DD pts |
|---|---|---:|---:|---:|---:|---:|
| 2022 | 1.618 | 1,834 | 68.97% | +3,619.35 | +161,897.75 | 71.23 |
| 2023 | 1.13 | 1,665 | 77.54% | +3,554.30 | +164,429.50 | 49.62 |
| 2024 | 1.05 | 2,091 | 67.72% | +4,357.40 | +199,591.00 | 127.21 |
| 2025-01-01..2026-05-05 | 1.13 | 2,792 | 68.91% | +5,872.20 | +270,013.00 | 81.38 |
| **Stitched OOS** | — | **8,382** | **70.34%** | **+17,403.25** | **+795,931.25** | **127.21** |

Stitched vs prior champion (0.618/1.05): `+2.0%` pts, `+1.8%` Rs, `-22.2%`
drawdown. Fees OOS: Rs 335,280 (8,382 trades x 40).

**Recommendation:** adopt `target 0.786 / stop 1.13` as the new fixed-cost
champion (modest gain, materially better drawdown), but the user's goal of
larger points per trade is **not achievable through this exit grid** — it
requires the exit-model engine change (Option C in the brainstorm), which
should be prototyped and parity-checked before any further sweeps.

## Timing

- Full non-WF preprocessing and initial transfer: `146.557s`.
- Full non-WF CUDA evaluation: `14.558s` for `3,002` evaluations including parity.
- WFO cached preprocessing and transfer: `2.507s`.
- WFO CUDA evaluation: `83.826s` for `18,008` evaluations.
- Peak GPU allocation: approximately `2.88 GB`; peak reserved: approximately
  `3.23 GB`.

## Artifacts

- Non-WF JSON: `artifacts/f6_hybrid/smart_fib_optimus_gpu_full_3000.json`
- WFO JSON: `artifacts/f6_hybrid/smart_fib_optimus_gpu_wfo_2020_2026-05-05.json`
- Resident tensor cache: `artifacts/f6_hybrid/smart_fib_gpu_tensor_cache_2020-01-01_2026-05-05.npz`
- GPU engine: `artifacts/f6_hybrid/smart_fib_optimus_gpu.py`
- Data workflow: `HISTORICAL_DATA_DOWNLOAD_GUIDE.md`
