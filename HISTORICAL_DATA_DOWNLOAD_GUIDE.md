# Historical Data Download Guide

This guide is the repeatable, read-only workflow for downloading NIFTY index
and option history, building a local cache, validating the data, and running a
Smart Fib replay or backtest.

The commands below assume PowerShell from the repository root:

```powershell
$py = "C:\Users\user\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
```

Use another Python executable only if it has the repository dependencies
installed.

## Data Sources

Use one of these sources deliberately. Do not mix their formats in the same
cache.

| Source | Use | Location or API |
|---|---|---|
| Local archive | Multi-year optimizer/backtest input | `C:\Users\user\Desktop\nifty50 data` |
| Ammu/Upstox archive and fetch guide | Index/futures history, current gaps, and fetch templates | `C:\Websites\ammu\HISTORICAL_DATA_GUIDE.md` |
| Flattrade TPSeries | Recent or missing historical sessions | Read-only `TPSeries` API |
| Flattrade SearchScrip | Historical option token resolution | Read-only `SearchScrip` API |

The local archive uses formats such as:

```text
index/NIFTY 50_minute.csv
nifty_options/YYYY/M/nifty_options_DD_MM_YYYY.csv
```

The local index file uses `date,open,high,low,close,volume`. Option files use
`date,time,symbol,open,high,low,close,oi,volume`.

The broker cache uses broker timestamps in `DD-MM-YYYY HH:MM:SS` format and
stores contracts under keys such as `CE:24350` and `PE:24400`.

For documentation and public web research, Crawl4AI is installed globally as
an LLM-friendly local crawler. It is a web extraction tool, not a replacement
for broker candle APIs:

```powershell
crwl crawl https://example.com -o markdown
```

Use `crawl4ai-setup` after reinstalling or changing Python environments. Keep
market candles sourced from the documented broker/API or local CSV pipeline;
use Crawl4AI for public documentation, source discovery, or permitted web
pages only.

The Ammu guide is the reference for the alternate Upstox-backed source. Read
it before using that source because it records endpoint limits, token rules,
chunking requirements, and known coverage gaps:

```text
C:\Websites\ammu\HISTORICAL_DATA_GUIDE.md
```

Its current source policy is:

- Upstox Historical Candle V3 is the preferred route for NIFTY index history
  and currently active futures.
- Upstox V2 is a fallback and supports only its documented intervals.
- Expired futures/options require an Upstox Plus plan; a free token cannot
  backfill those contracts.
- Weekly chunks are the safe default for 1-minute requests.
- URL-encode instrument keys and resolve them through the search API; never
  guess stale keys.
- Upstox index volume is expected to be zero.

Use the Ammu-local files and fetch templates when they are the best available
source, but keep their provenance separate from the Desktop archive and
Flattrade caches. Do not silently merge them into one continuous series.

## Fast Smart Fib Workflow

### 1. Download the target sessions

The dedicated downloader fetches:

- NIFTY spot token `26000` from `NSE`.
- Historical option contracts from `NFO`.
- The target option session plus the preceding calendar day for option-state warmup.
- Dynamic ATM, first-ITM, and second-ITM candidates at each trigger minute.
- The exact historical expiry, not the currently listed expiry.

The spot request covers the target date. For the first replay date, run the
downloader once for the previous session too so the 5m index state has a
warmup cache.

Run it into a separate cache so the legacy replay cache is not overwritten:

```powershell
& $py artifacts/download_smart_fib_aug_options.py `
  --dates 2026-08-12 2026-08-13 2026-08-14 `
  --cache-dir artifacts/flattrade_day_cache_smart_fib
```

If the first target date needs an index warmup session, download that session
into the same cache as well:

```powershell
& $py artifacts/download_smart_fib_aug_options.py `
  --dates 2026-08-11 `
  --cache-dir artifacts/flattrade_day_cache_smart_fib
```

The downloader is read-only. It uses the existing automated Flattrade login
and does not call an order endpoint.

### 2. Validate the download and tally trades

Run the reconciliation script:

```powershell
& $py artifacts/f6_hybrid/tally_smart_fib_cache.py `
  --cache-dir artifacts/flattrade_day_cache_smart_fib `
  --dates 2026-08-12 2026-08-13 2026-08-14 `
  --output artifacts/f6_hybrid/smart_fib_aug_12_14_tally.md
```

Every reported trade must have both its entry and exit minute present in the
downloaded option contract rows. The report includes:

- Spot row count.
- Option contract count and option row count.
- Trades by date, timeframe, and stop level.
- Entry/exit symbol and timestamps.
- TP/SL reason, points, and net rupees.
- Matched versus unmatched trade rows.

For the Aug 12-14, 2026 download, the expected validation shape was:

| Date | Spot rows | Contracts | Option rows |
|---|---:|---:|---:|
| 2026-08-12 | 376 | 14 | 10,500 |
| 2026-08-13 | 376 | 10 | 7,500 |
| 2026-08-14 | 376 | 10 | 7,500 |

These counts are a sanity check, not a universal requirement. Holidays,
partial sessions, missing contracts, and API corrections can change them.

### 3. Run a Smart Fib smoke test

For local multi-year data, explicitly pass the data root. Do not rely on the
optimizer's default path when the active archive is on the Desktop:

```powershell
& $py artifacts/f6_hybrid/smart_fib_optimizer.py `
  --data-root "C:\Users\user\Desktop\nifty50 data" `
  --start 2020-01-01 `
  --end 2026-05-05 `
  --smoke `
  --smoke-days 5
```

Before any full backtest or sweep, confirm:

- The loader finds overlapping index and option days.
- There are trades on the five-day smoke window.
- Trade counts are plausible, not zero and not explosively high.
- Net P/L is not an accidental 100% capital loss.
- The first smoke result is compared with a known reference replay.

### 4. Run the bounded optimizer

The optimizer refuses an unbounded run by default. Start with a small explicit
trial count only after smoke validation:

```powershell
& $py artifacts/f6_hybrid/smart_fib_optimizer.py `
  --data-root "C:\Users\user\Desktop\nifty50 data" `
  --start 2020-01-01 `
  --end 2026-05-05 `
  --trials 50 `
  --allow-expensive
```

The current parameter schema covers target, stop, fallback target, option
point threshold, minimum span, touch buffer, setup age, and timeframe. The
execution rules remain fixed: actual option contracts, one open position,
slippage, lot size, and causal replay.

The current `smart_fib_optimizer.py` is the correctness-first historical
adapter. Its Optuna loop replays days through the Smart Fib core; it must not
be described as the fused GPU Optimus engine until the Smart Fib event tensors
have passed the same three-day parity and regression checks as
`optimized_gpu_backtest.py`.

### GPU-first Smart Fib Optimus

The GPU evaluator requires CUDA by default, runs the CPU adapter and event
extraction once, validates exactly three dates against the current CPU core, and
then keeps actual option OHLC and event metadata resident on the GPU:

```powershell
& $py artifacts/f6_hybrid/smart_fib_optimus_gpu.py `
  --data-root "C:\Users\user\Desktop\nifty50 data" `
  --start 2020-01-01 `
  --end 2026-05-05 `
  --prep-workers 8 `
  --smoke `
  --batch-size 100 `
  --output artifacts/f6_hybrid/smart_fib_optimus_gpu_smoke.json
```

The evaluator uses fixed `T=375` minute slots and a padded fixed contract axis
`C`. Optuna suggestions and ask/tell orchestration remain CPU-side; batched
trial exit simulation, source-specific Fib checks, dynamic fallback-target
checks, actual option P/L, fees, and re-entry state run on CUDA. The current
Smart Fib entry stream is intentionally not re-tensorized: combined events are
extracted once with `min_span=15`, zero touch buffer, and 45-minute setup age.
The target `0.29`, 10-point threshold, fallback `0.0`, and stop choices
`1.155/1.25` remain the execution contract; only the stop choice is optimized.

`--allow-cpu` is an explicit parity/debug escape hatch and is not a silent
fallback. Do not use it for a sweep. The evaluator refuses unbounded runs and
refuses to overwrite an existing output path.

### Bounded Smart Fib zone and exit grid

`smart_fib_optimus_grid_gpu.py` adds an explicit finite search contract while
keeping signal extraction causal and CPU-once-per-variant/day. The signal axes
are:

- Fibonacci zones: `(0.618, 1.0)`, `(0.618, 0.786)`, `(0.786, 1.0)`,
  `(0.5, 0.786)`, `(0.705, 0.886)`.
- S1 variants: `(9,3)`, `(12,3)`, `(14,3)`, `(12,4)`.
- `min_span`: `10`, `15`, `20`; setup age: `30`, `45`, `60`; touch buffer:
  `0`, `0.5`, `1.0`.
- Targets: `0.0`, `0.236`, `0.29`, `0.382`, `0.5`.
- Fallback targets: `0.0`, `0.236`, only when `fallback <= target`.
- Thresholds: `5`, `10`, `15`; stops: `1.13`, `1.155`, `1.25`, `1.272`,
  `1.382`, `1.618`.

The default smoke remains staged: five zone-aware signal variants, target
`0.29`, fallback `0.0`, thresholds `5/10/15`, and stops `1.155/1.25`.
All target/fallback/stop values outside the explicit axes are rejected. The
dynamic 10-point fallback is enabled only for primary target `0.29`, matching
the historical behavior; other targets do not activate that fallback rule.

Run the bounded smoke as follows:

```powershell
& $py artifacts/f6_hybrid/smart_fib_optimus_grid_gpu.py `
  --data-root "C:\Websites\ammu" `
  --start 2020-01-01 --end 2026-05-05 `
  --smoke --max-variants 5 `
  --targets 0.29 --fallback-targets 0.0 `
  --thresholds 5 10 15 --stops 1.155 1.25 `
  --batch-size 100 --prep-workers 8 `
  --output artifacts/f6_hybrid/smart_fib_optimus_grid_gpu_smoke.json
```

Smoke output is a five-day sanity check, not full-period validation or
walk-forward evidence. Do not treat its ranking as a 2020-2026 strategy
selection.

The remaining full requested grid command is intentionally not run by the
smoke workflow:

```powershell
& $py artifacts/f6_hybrid/smart_fib_optimus_grid_gpu.py `
  --data-root "C:\Websites\ammu" `
  --start 2020-01-01 --end 2026-05-05 `
  --max-variants 540 --allow-expensive `
  --targets 0.0 0.236 0.29 0.382 0.5 `
  --fallback-targets 0.0 0.236 `
  --thresholds 5 10 15 `
  --stops 1.13 1.155 1.25 1.272 1.382 1.618 `
  --batch-size 100 --prep-workers 8 `
  --tensor-cache-dir artifacts/f6_hybrid/smart_fib_grid_tensor_cache `
  --output artifacts/f6_hybrid/smart_fib_optimus_grid_gpu_full.json
```

This is 540 zone-aware signal variants and 162 valid exit combinations before
data coverage, cache, and trade guards. It is the expanded full run, not a
smoke test.

Use `--trials N --allow-expensive` for bounded non-WF optimization, or add
`--wfo` for the annual train-only 2021-2026 folds (the 2026 validation fold
ends at `2026-05-05`).

For rolling walk-forward selection with the GPU engine, use the same explicit
trial count. Each annual fold selects parameters on its train window and
evaluates only the next validation year; the final report stitches validation
trades without reusing future data:

```powershell
& $py artifacts/f6_hybrid/smart_fib_optimus_gpu.py `
  --data-root "C:\Users\user\Desktop\nifty50 data" `
  --start 2020-01-01 `
  --end 2026-05-05 `
  --trials 10 `
  --batch-size 100 `
  --prep-workers 8 `
  --tensor-cache artifacts/f6_hybrid/smart_fib_gpu_tensor_cache_2020-01-01_2026-05-05.npz `
  --wfo `
  --allow-expensive
```

The completed full-period commands use the same cache and differ only in
mode:

```powershell
& $py artifacts/f6_hybrid/smart_fib_optimus_gpu.py `
  --data-root "C:\Users\user\Desktop\nifty50 data" `
  --start 2020-01-01 --end 2026-05-05 `
  --trials 3000 --batch-size 100 --prep-workers 8 `
  --tensor-cache artifacts/f6_hybrid/smart_fib_gpu_tensor_cache_2020-01-01_2026-05-05.npz `
  --allow-expensive `
  --output artifacts/f6_hybrid/smart_fib_optimus_gpu_full_3000.json
```

`C:\Users\user\Desktop\opti` is a research-reference folder, not a market
data source. Its CUDA, AI-trading, and columnar-data PDFs informed the
implementation. The applied techniques are Polars projection/filter parsing,
fixed resident tensors, pinned host-to-device transfer, Optuna ask/tell
batches, and matrix first-hit exits. The PDF claims are not treated as
backtest results.

### Additional-Parameter GPU Grid

`smart_fib_optimus_grid_gpu.py` searches signal variants on the CPU once and
then evaluates every selected target/fallback/threshold/stop combination on the existing
matrix-first CUDA exit engine. The logical option tensor for each variant is
fixed as `(N,T,C)` and is exposed to the engine as a zero-copy `(N,C,T)` view.
The default is a bounded five-variant staged probe, not the full grid:

```powershell
& $py artifacts/f6_hybrid/smart_fib_optimus_grid_gpu.py `
  --data-root "C:\Users\user\Desktop\nifty50 data" `
  --start 2020-01-01 --end 2020-01-07 `
  --smoke --max-variants 5 `
  --targets 0.29 --fallback-targets 0.0 `
  --thresholds 5 10 15 --stops 1.155 1.25 `
  --prep-workers 2 --batch-size 100 `
  --output artifacts/f6_hybrid/smart_fib_optimus_grid_gpu_probe.json
```

The signal axes are:

| Axis | Values |
|---|---|
| Fibonacci zone pair | `(0.618,1.0)`, `(0.618,0.786)`, `(0.786,1.0)`, `(0.5,0.786)`, `(0.705,0.886)` |
| S1 `(k_period,d_period)` | `(9,3)`, `(12,3)` baseline, `(14,3)`, `(12,4)` |
| `min_span` | `10`, `15`, `20` |
| `setup_max_age` | `30`, `45`, `60` minutes |
| `touch_buffer` | `0.0`, `0.5`, `1.0` |
| GPU target | `0.0`, `0.236`, `0.29`, `0.382`, `0.5` |
| GPU fallback target | `0.0`, `0.236`, constrained by `fallback <= target` |
| GPU option-point threshold | `5`, `10`, `15` |
| GPU stop extension | `1.13`, `1.155`, `1.25`, `1.272`, `1.382`, `1.618` |

The signal axes contain 540 possible zone-aware CPU variants and the default
exit axes contain six GPU configurations per variant. Use an explicit bounded
variant list such as `--variants baseline 9:3:10:30:0.5:0.618:0.786`, where the
fields are `s1_k:s1_d:min_span:setup_max_age:touch_buffer[:zone_start:zone_end]`.
The baseline is always included. `--max-variants` selects the deterministic
staged shortlist when `--variants` is omitted; runs above five variants and all
non-smoke runs require `--allow-expensive`.

The CPU phase uses the Polars index/option adapter and preparation workers. It
calls `extract_day_events` once per selected variant/day, freezes actual option
OHLC and event metadata, and never calls `process_day` inside the GPU grid.
Only the repeated exit evaluation is GPU-generated. Parity is mandatory for
the baseline and every selected variant on exactly the first three available
dates, across every selected target/fallback/threshold/stop combination; trade
counts are exact and points/net-Rs/drawdown use the existing strict `0.05`
tolerance. A parity failure aborts the run. The existing full tensor cache is
reused for the unchanged baseline when it matches; optional
`--tensor-cache-dir` caches use a zone-aware variant identity in every cache
key.

The grid probe is not a walk-forward result. It selects signal variants and
exit settings on the supplied window, so its top five are research candidates,
not an out-of-sample claim. The full 540-variant CPU search and expanded exit
grid are intentionally not the default and should be run only after the
five-day probe, parity, data coverage, and trade-count checks pass. Ranking is
`net_points - 0.20 * max_drawdown_points` with a minimum-trade guard; fees,
slippage, actual option rows, and the one-position replay remain enabled.

## Contract Selection

Smart Fib tracks these candidates at every trigger minute after rounding spot
to the nearest 50-point strike:

| Side | Candidates |
|---|---|
| CE | ATM, ATM-50, ATM-100 |
| PE | ATM, ATM+50, ATM+100 |

The correct historical weekly contract must be resolved before downloading
candles. For Aug 12-14, 2026 the verified expiry was `18AUG26`.

The downloader follows this order:

1. Search the exact expiry, strike, and side.
2. Verify the returned `tsym` or `dname` contains the expected historical expiry.
3. Use a generic strike search only as a fallback.
4. Reject the result if its expiry is not the requested historical expiry.
5. Fetch the target day and warmup day for the resolved token.

Never assume that a generic query such as `NIFTY 24400 PE` returns the
historical series. It can return a later currently listed expiry.

## Warmup and Time Rules

- Use Asia/Kolkata market time.
- Keep rows chronological and deduplicate by broker timestamp.
- Fetch prior-session candles for stateful UT, Heikin-Ashi, LinReg, and S1 state.
- For a Monday or post-holiday session, use the previous available trading day,
  not blindly the previous calendar date if the API does not return rows.
- Do not use a future candle to complete or confirm a past signal.
- Keep the 5m index filter causal: build full raw 5m candles, convert to
  Heikin-Ashi, then apply UT and LinReg using confirmed state.
- Keep option entries and exits on actual downloaded option bars.

## Existing Commands

### Download only NIFTY spot for Aug 12-14

This older helper downloads spot data only. It does not download options and is
not sufficient for Smart Fib replay by itself:

```powershell
& $py fetch_aug_days.py
```

It writes:

```text
artifacts/f6_hybrid/nifty_spot_aug12_14_2026.json
```

### Generic single-day cache

This command supports the older Flattrade replay workflow:

```powershell
& $py artifacts/cache_flattrade_day.py `
  --date 2026-08-14 `
  --cache-dir artifacts/flattrade_day_cache `
  --itm 2
```

It should not be used as the Smart Fib downloader because it tracks one
100-point ITM depth per side and may replace an existing cache snapshot. Use
`download_smart_fib_aug_options.py` for Smart Fib.

### Offline legacy replay

The following command reads an existing legacy cache and runs the older F6
signal replay. It is not the Smart Fib tally:

```powershell
& $py artifacts/replay_flattrade_signals.py `
  --date 2026-08-14 `
  --cache-dir artifacts/flattrade_day_cache `
  --offline
```

## Do

- Read `AGENTS.md` and `graphify-out/GRAPH_REPORT.md` before changing data or
  strategy code.
- Use a dedicated cache directory for each data contract or strategy version.
- Pass an explicit `--data-root` for Desktop historical archives.
- Resolve the exact historical expiry before fetching option candles.
- Fetch warmup data before replaying stateful indicators.
- Preserve actual option symbols and timestamps in the cache.
- Validate spot rows, contract rows, date ranges, and missing rows before trading
  simulation.
- Run the five-day smoke test before a full historical run.
- Run a short parity test before changing or optimizing a GPU evaluator.
- Keep transaction costs and slippage enabled for the final report.
- Keep non-walk-forward and walk-forward results separate.
- Use train-only parameter selection for every walk-forward fold.
- Save the command, data root, date range, expiry mapping, parameter set, and
  output path with every result.
- Run `python run_graphify.py` after modifying Python files.

## Do Not

- Do not print, commit, or store `.env` credentials, API secrets, TOTP keys,
  cookies, or session tokens.
- Do not use an order, quote, or live-trading endpoint for historical downloads.
- Do not accept a generic option search result without verifying its expiry.
- Do not mix `C:\Websites\ammu` and `C:\Users\user\Desktop\nifty50 data`
  without documenting which source supplied each file.
- Do not overwrite `artifacts/flattrade_day_cache` when testing Smart Fib.
- Do not treat a spot-only JSON file as complete option history.
- Do not fill missing option candles with future prices or synthetic candles.
- Do not use current-day forming 5m state as confirmed historical bias.
- Do not run a full multi-year sweep before smoke and parity checks pass.
- Do not rank configurations by net profit alone; report maximum drawdown,
  trade count, win rate, fees, and profit factor too.
- Do not call a static post-period mask a true walk-forward test.
- Do not claim a trade is matched unless both entry and exit rows exist in the
  downloaded contract data.
- Do not stage unrelated worktree changes when committing the downloader or
  documentation.

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Login fails | Expired credentials, TOTP, or broker session | Re-run the read-only login; never paste credentials into a command |
| No option contract found | Wrong expiry or strike not listed | Check the exact historical expiry and broker security master |
| Rows are empty | Wrong token, exchange, or date window | Verify `NSE` for spot, `NFO` for options, and epoch bounds |
| Spot rows exist but no trades | Missing warmup or incomplete candidate contracts | Download prior-session warmup and all Smart Fib strike candidates |
| Trade symbol has no entry/exit row | Cache was mixed or partially written | Delete only the failed dedicated cache and re-download it |
| Date count is unexpectedly low | Holiday, partial session, or archive gap | Report missing trading dates; do not silently treat them as zero-signal days |
| Optimizer reports no overlapping days | Wrong `--data-root` or filename layout | Check `index/NIFTY 50_minute.csv` and nested option filenames |
| Large result changes after a code edit | Lookahead, timeframe alignment, or cache change | Run the smoke, three-day parity, and regression guard before continuing |

## Related Files

- `artifacts/download_smart_fib_aug_options.py` - exact Smart Fib broker download.
- `artifacts/f6_hybrid/tally_smart_fib_cache.py` - cache validation and trade tally.
- `artifacts/f6_hybrid/marni_fib_core_combo_cache.py` - Smart Fib signal engine.
- `artifacts/f6_hybrid/smart_fib_optimizer.py` - historical optimizer adapter.
- `artifacts/f6_hybrid/smart_fib_optimus_grid_gpu.py` - bounded signal-variant GPU grid.
- `artifacts/flattrade_day_cache.py` - compressed cache format.
- `artifacts/replay_flattrade_signals.py` - legacy F6 replay and generic fetch helpers.
- `artifacts/f6_hybrid/test_optimus_regression.py` - Optimus accuracy and speed guard.
- `SMART_FIB_STRATEGY.md` - Smart Fib rules and execution assumptions.
