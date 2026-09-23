# FLATTRADE BOT — Agent Rules

## Graphify Knowledge Graph (MANDATORY — Read Before Raw Files)

This project has a graphify knowledge graph at `graphify-out/`.

**Rules — follow in this order before reading any raw source file:**

1. **Start with the graph report:** Read `graphify-out/GRAPH_REPORT.md` first.
   It lists god nodes (highest centrality), critical bridges, and community structure.
   This tells you which files matter most before you touch anything.

2. **Navigate via graph.json:** Query `graphify-out/graph.json` to find nodes,
   edges, and relationships. 395 nodes · 714 edges · AST-extracted.

3. **Use the interactive graph:** `graphify-out/graph.html` — open in browser
   for visual navigation of module dependencies.

4. **Only then read raw files** — if the graph doesn't answer your question,
   read the specific source file. Do NOT bulk-read the whole codebase.

5. **Keep the graph current:** After modifying any `.py` file, run:
   ```
   python run_graphify.py
   ```
   Or just `git commit` — the post-commit hook rebuilds automatically.

## God Nodes (Most Critical — Touch With Care)

| NODE | CENTRALITY | ROLE |
|:---|:---:|:---|
| `patterns_candle` | 0.124 | Core data structure — everything depends on Candle |
| `divergence_divergenceengine` | 0.102 | Central signal generator |
| `backtest_5y_optimized` | 0.094 | Reference engine all backtests import from |
| `backtest_5y_optimized_timeframetracker` | 0.084 | Per-TF state machine |
| `patterns_bullishpinbardetector` | 0.081 | Entry trigger logic |
| `stochastic_incrementalstochastic` | 0.056 | Core indicator |

## Project Overview

**Strategy (LIVE):** Pocket Money — Nifty 50 options intraday on official 10s bars
(FLAG/SUPER stochastic triggers on 2nd ITM strikes, 15m UT Bot + LinReg white-line
index filter, SL/TP = ±7 pts). **Verified congruent** with the backtest reference
(9/9 trades exact on Aug 12-13 2026). See `POCKET_MONEY_STRATEGY.md` and
`BACKTEST_LEDGER.md` → Pocket Money section.

**Strategy (research):** Nifty 50 options intraday — 4-timeframe concurrent engine
(1m, 2m, 3m, 5m), Stochastic divergence + BullishPinBar (Super/Flag/Reversal).
Best Result: Trailing SL + S1 turn-up = **+₹828,890 over 5 years (2020-2024)**.
**Ledger:** See `BACKTEST_LEDGER.md` for all strategy test results indexed.

---

## Smoke Test (MANDATORY — Run Before Every Backtest)

**NEVER launch a full 5-year backtest without first running a smoke test.**
A smoke test validates the engine on 5 days and takes < 30 seconds.
It catches implementation bugs before wasting 20-40 minutes on a full run.

### Smoke Test Checklist

Before running any new or modified backtest script:

1. **Run on 5 days only** — slice `days = days[:5]` temporarily
2. **Verify trade count** — should be 1-10 trades per day (0 is suspicious, >30 is a bug)
3. **Verify win rate** — should be 35-50% for ATR strategies, 35-40% for Trailing SL
4. **Verify net profit sign** — not necessarily positive, but not -100% of capital
5. **Cross-check against known baseline** — compare a known strategy's 5-day smoke output with the reference engine (`backtest_unlimited_profit.py`)
6. **Restore `days` to full range** before the real run

### Quick Smoke Test Template

```python
# Add this TEMPORARILY at the top of main() for smoke testing:
days = days[:5]   # ← SMOKE TEST: remove before full run
print("=== SMOKE TEST — 5 DAYS ONLY ===")
```

### Known Reference Values (5-day sanity check)

When running ANY strategy on the FIRST 5 days of 2020 with standard settings:
- Expected trades: 15-40
- Expected win rate: 30-50%
- A win rate of 0% or >80% means the trigger/exit logic is broken
- A trade count of 0 means signal detection is broken

### Common Bugs This Catches

| BUG | SYMPTOM | ROOT CAUSE |
|:---|:---:|:---|
| pmtrig tuple mismatch | WR = 10-20% instead of 35-45% | Different tuple ordering vs reference |
| Missing ATR warmup | All trades use fallback tf_sl | ATR not warmed up before session |
| Wrong exit logic | WR = 0% | SL check using wrong variable |
| No signals fired | 0 trades | Import from wrong tracker class |

---

## Nifty 50 Option Master Parquet (CANONICAL DATA)

**ALL strategies MUST read option data from the single canonical parquet (v2):**

```
C:\Users\user\Desktop\nifty50 data\nifty50_options_master_v2.parquet
```

Non-synced working copy (use this path in code — OneDrive stalls reads on
the Desktop path, see GPU guide §16 BUG #5):

```
C:\Users\user\AppData\Local\Temp\opencode\data\nifty50_options_master_v2.parquet
```

- This is the **only** option parquet any strategy may use. The old
  `nifty50_options_master.parquet` (v1) is now **DEPRECATED** — it hard-coded
  the Tue/Thu expiry rule and therefore DROPPED genuine holiday-shifted
  weekly contracts (Feb 25–Mar 2 `02MAR26` Monday expiry, Mar 25–30
  `30MAR26` Monday, **Apr 8–13 `13APR26` Monday**). The old
  `cache_marni_opt.parquet` is also banned (wrong-expiry "malicious" rows).
- Built by `rebuild_parquet_v2.py`. Schema: `day, minute, symbol, strike,
  side, open, high, low, close, exp_date` (1-minute bars). `exp_date` =
  parsed contract expiry date (new in v2).
- `minute` = minute-of-day (e.g. 555 = 09:15). Symbols are `NIFTY<DDMMMYY><STRIKE><CE|PE>`.
- **1,526 trading days** (v1 had 1,510 — 16 recovered), 55.16M rows.
- Underlying INDEX 1m (for bias/Elder) lives at
  `C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv`
  (note: Flattrade TPSeries index history starts ~Jun 2022 only).

### Expiry-Day Rule (TRUE rule — supersedes the old Tue/Thu table)

The per-day contract in v2 is chosen as: **the expiry token present in that
day's source CSV with the smallest expiry date ≥ trading day, at most 9
calendar days ahead** (fallback: expiry == trading day). This correctly
captures holiday-shifted Monday expiries that the old Thu→Tue rule missed:

| PERIOD | NOMINAL WEEKLY EXPIRY DAY | EXCEPTIONS SEEN IN DATA |
|:---|:---|:---|
| 2020-01-01 → 2025-08-28 | Thursday | (verify per-day from `exp_date` column) |
| 2025-09-01 → 2026-08-28 | Tuesday | **Monday expiries**: 02MAR26, 30MAR26, 13APR26 (holiday shifts, e.g. Apr 14 2026 = Ambedkar Jayanti) |

- Engines must NOT re-filter by expiry — v2 already contains ONLY the
  correct nearest-weekly contract per trading day.
- 2nd ITM strike per day: **CE = ATM−100, PE = ATM+100** (ATM from index spot 09:15 open).
- **Data-trust note (Sep 2026 audit):** April 2026's high premiums (2nd-ITM
  200–450 vs Aug's 80–200) and −620 index gap days were verified bar-for-bar
  against the independent source CSVs — genuine April tape, not a data bug.
  April 2026 was a high-IV gap regime; do not "fix" it.

---

## Security

- `.env` contains live API credentials — **never commit it** (excluded in `.gitignore`)
- `.env.example` contains only placeholders — safe to commit
- Rotate Flattrade API key/secret if ever exposed

## VPS Access Discipline (MANDATORY — box is memory-starved)

The live VPS (65.2.50.227, Lightsail 512MB plan) has **412Mi RAM total, NO swap,
~130Mi free during session**. It lives at the edge of OOM every market session
(168 oom-killer invocations logged Sep 10–17 2026).

**Incident 2026-09-17 10:07 IST:** 4 parallel SSH sessions tipped the box over —
the kernel OOM-killed the live trading bot mid-session (10s downtime, flat so
no money at risk, auto-recovered via systemd + cap rehydration). Never repeat this.

Hard rules for ANY VPS contact:
1. **Single lightweight reads only** — ONE ssh session at a time, sequential,
   never parallel. One command per call; wait for it to finish.
2. **Reads are cheap, compute is not** — `tail`, `grep -c`, small `grep` on
   today's log only. NEVER run python3, `journalctl` dumps, full-log greps,
   or anything that allocates on the box.
3. **Analyze locally, not on the box** — for anything beyond a one-line check,
   `scp` the log file off the VPS and process it on this machine.
4. **Reads only, always** — no writes, no deploys, no restarts, no `systemctl`
   state changes without the user's explicit go-ahead, on or off session.
5. **Market hours (09:15–15:30 IST) = minimum contact** — if a check can wait
   until after 15:15 EOD exit, it waits.

## Flattrade API — MANDATORY REFERENCE

**ANY Flattrade-related problem (auth, orders, quotes, WebSocket, rate limits,
data fetch, error messages): read `flattrade documentation.md` FIRST** — it is
the complete official Pi API v2.0 docs (saved 2026-09-04), including the
WebSocket handshake protocol, all REST endpoints, rate limits, and the
changelog. Do not guess API behavior; the docs file is the source of truth.

Critical facts (details in the docs file):
- REST base: `https://piconnect.flattrade.in/PiConnectAPI/`
- WS: `wss://piconnect.flattrade.in/PiConnectWSAPI/` — REQUIRES a connect
  handshake `{"t":"a","uid","actid","source":"API","accesstoken"}` and
  expects `{"t":"ak","s":"Ok"}` BEFORE any subscription works (subscribing
  without it silently yields zero ticks — the Sep-4 stuck-prices bug)
- Heartbeat: TEXT message `{"t":"h"}` every 30s (not WS protocol pings)
- MKT prctyp is rejected for API orders (`ALGO_CHK`) — use aggressive LMT
- API rate limit: 40/sec, 200/min (shared with quotes — REST polling 6+
  instruments every second gets throttled: "Order Recieved NNN in a current
  minute exceeds Limit 120")
- SearchScrip query format: `NIFTY 24000 CE` (spaced)
- Token exchange: POST `https://authapi.flattrade.in/trade/apitoken` with
  `api_secret` = SHA-256(api_key + request_code + api_secret), from the
  registered IP; single login session (a new login invalidates the old
  token machine-wide)

## Certified Sweep Pipeline (MANDATORY — Supersedes All Prior Backtesting)

**From 2026-09-05 onward, ANY strategy validation, parameter sweep, or backtest
claim MUST be produced by the certified sweep pipeline. Read
`CERTIFIED_SWEEP_BLUEPRINT.md` FIRST — it contains the full architecture,
the fix ledger (14 bugs), the optimization inventory, and step-by-step
instructions for sweeping ANY general strategy (RSI-based, stoch-based, or
otherwise) through the same pipeline.**

Files (the ONLY sanctioned backtest toolchain):

| FILE | PURPOSE |
|:---|:---|
| `CERTIFIED_SWEEP_BLUEPRINT.md` | **Read this first** — architecture, fixes, do's/don'ts, how to sweep any strategy |
| `lh_sweep_build2.py` | Phase 1: event-stream recorder (raw indicator values; `--stochset` variants) |
| `lh_sweep_replay_cpu.py` | Phase 2: njit CPU twin — the executable strategy spec |
| `lh_sweep_cuda.py` | Phase 3a: CUDA sweep v1 (fixed thresholds; 48/48 parity-certified) |
| `lh_sweep_cuda2.py` | Phase 3b: CUDA sweep v2 — sweeps thresholds + indicator periods from raw recordings |
| `lh_parity_gate.py` | 48-check parity gate vs the live-validated certified harness |
| `lh_cuda_sweep_results.csv` | Certified ledger: 2,940-config sweep (2026-09-05) |
| `lh_cuda_sweep2_results.csv` | Certified ledger: extended-parameter sweep |

Hard rules:
1. **Parity gate before any sweep report** — the CUDA kernel must reproduce
   the CPU twin exactly (48/48-style) on the reference window, every time the
   kernel or recordings change.
2. **Independent ground truth required** — the certification chain must end at
   live-bot logs or the live-validated harness. Self-parity of a shared engine
   core is invalid (the dyn-engine conviction).
3. **Top-K configs re-verified on the CPU twin** before publication.
4. **All §41–§44 dyn-derived numbers are convicted/unverified** —
   `gpu_sim_last_hope.py` and `dyn_strike_engine.py` are RESEARCH-GRADE only,
   never cite them as truth. The certified ledger supersedes BACKTEST_LEDGER
   entries for this family.
5. **Fees always in** (`(exit−entry)×65 − 45`): the 2026-09-05 certified sweep
   proved the whole 2nd-ITM bounce family is gross-profitable but
   fee-destroyed (all 2,940 configs negative over 2020–2026; §44 itself
   −₹821,080).
6. **No config goes live** without positive certified net AND a walk-forward
   pass on this pipeline.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
