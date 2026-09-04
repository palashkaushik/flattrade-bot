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

**ALL strategies MUST read option data from the single canonical parquet:**

```
C:\Users\user\Desktop\nifty50 data\nifty50_options_master.parquet
```

- This is the **only** option parquet any strategy may use. The old
  `cache_marni_opt.parquet` is deprecated/removed (it held wrong-expiry
  "malicious" rows) — do NOT point engines at it.
- Built by `build_canonical_parquet.py`. Schema: `day, minute, symbol,
  strike, side, open, high, low, close` (1-minute bars).
- `minute` = minute-of-day (e.g. 555 = 09:15). Symbols are `NIFTY<DDMMMYY><STRIKE><CE|PE>`.
- Underlying INDEX 1m (for bias/Elder) lives at
  `C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv`.

### Expiry-Day Rule (NSE/SEBI, web-verified)
Nifty 50 option expiry day changed exactly once in 2020–2026:

| PERIOD | WEEKLY EXPIRY DAY |
|:---|:---|
| 2020-01-01 → 2025-08-28 | **Thursday** of the week |
| 2025-09-01 → today | **Tuesday** of the week |

- Effective: end of trading 28-Aug-2025; new contracts from 1-Sep-2025 use Tuesday
  (SEBI mandate limiting index-expiry days to Tue/Thu; NSE→Tue, BSE→Thu).
- The canonical parquet keeps ONLY the **correct weekly contract per trading day**
  (Thu pre-change, Tue post-change). Wrong-contract rows (e.g. a monthly expiry used
  on a weekly-strategy day) were dropped. Engines must NOT re-filter by expiry.
- 2nd ITM strike per day: **CE = ATM−100, PE = ATM+100** (ATM from index spot 09:15 open).

---

## Security

- `.env` contains live API credentials — **never commit it** (excluded in `.gitignore`)
- `.env.example` contains only placeholders — safe to commit
- Rotate Flattrade API key/secret if ever exposed

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

## Key Files

| FILE | PURPOSE |
|:---|:---|
| `flattrade_bot/main.py` | Live trading entry point (Pocket Money 10s) |
| `flattrade_bot/strategies/pocket_money.py` | Live Pocket Money engine (10s bars, stochastics, 15m filter, gating) |
| `POCKET_MONEY_STRATEGY.md` | Official Pocket Money strategy rules + live mapping |
| `artifacts/f6_hybrid/pocket_money_backtest.py` | Backtest reference (`process_day`, `PocketHTFFilter`) |
| `backtest_5y_optimized.py` | Core backtest engine (reference) |
| `backtest_unlimited_profit.py` | Unlimited profit variant (pmtrig reference) |
| `BACKTEST_LEDGER.md` | All strategy test results |
| `nifty50_options_master.parquet` | **Canonical** Nifty 50 option 1m bars (see Data section) |
| `build_canonical_parquet.py` | Rebuilds the canonical parquet (correct weekly expiry per day) |
| `graphify-out/GRAPH_REPORT.md` | Knowledge graph audit report |
| `.env` | Live credentials (local only) |
