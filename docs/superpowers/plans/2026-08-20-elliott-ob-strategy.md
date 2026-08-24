# Implementation Plan — Elliott OB Strategy (EW-OB)

**Date:** 2026-08-20
**Source of truth:** `docs/superpowers/specs/2026-08-20-elliott-ob-strategy-design.md` (approved)
**Replaces:** F6 Champion + Marny family (structurally unprofitable, WR 33–37% vs 44.4% BE)

## Goal

Build and backtest a new intraday NIFTY options strategy: lenient 5-wave Elliott impulse
count on 1m index bars + multi-timeframe single-candle order blocks (1m/2m/3m/5m).
Entry = first 1m bar after the correction's B wave whose range pulls back into an
unmitigated OB (index direction decides CE/PE, 2nd-ITM nearest weekly). Exit = index
SL `3×ATR(10)` / TP `+60 pts`, P&L settled on the option. Full 7y run (2020-01-01 →
2026-08-20, ~1,588 days) with smoke-test congruency anchors first.

## Reference interfaces (already in repo)

| Interface | File | Use |
|:---|:---|:---|
| `load_spot()` → `{day: {min, open, high, low, close}}` | `opt_futures_quad.py:669` | Index 1m spot 2015→2026-05-15 |
| `extend_with_august(opt_map, spot_all)` | `artifacts/f6_hybrid/f6_mtf_7y_runner.py:41` | Merge Desktop Aug options + `ammu/data/2026-08-*/nifty50_index_1m_*.csv` spot |
| `option_day_files(start, end)` / `cached_option` / `make_slice` / `bar_at` / `latest_value` | `opt_futures_quad.py:339,116,139,154,354` | Option 1m bars + 2nd-ITM strike lookup |
| `trade_cost(entry, exit, broker)` / `apply_costs` | `backtest_walkforward_fees.py:76,89` | Fee model (1 pt slippage/side, STT/exch/SEBI/stamp/GST, ₹0 brokerage default) |
| Constants | `opt_futures_quad.py:55-57` | `LOT_SIZE=65`, `CE_OFFSET=-100`, `PE_OFFSET=+100`, sessions |
| `pytest 9.1.1` | repo `tests/` | Unit tests |

Data paths:
- `C:\Websites\ammu\index\NIFTY 50_minute.csv` — index 1m (1,048,738 rows, ends 2026-05-15)
- `C:\Websites\ammu\data\2026-08-*\{nifty50_index_1m_*.csv, nifty_options_1m_*.csv}` — Aug spot/options
- `C:\Users\user\Desktop\nifty50 data\nifty_options\2026\8\nifty_options_DD_MM_2026.csv` — Aug option bars (vendor format)
- `C:\Websites\ammu\nifty_options\**\nifty_options_*.csv` — archive option bars

## Module layout

```
artifacts/ew_ob/
  ew_ob_engine.py      # wave detector (session-continuous), OB detector per TF,
                       # entry/exit logic, option settlement  → all pure/logic
  ew_ob_runner.py      # CLI: load data, run engine over days, report + JSON output
tests/
  test_ew_ob_wave_detect.py   # unit: wave state machine transitions + conditions
  test_ew_ob_ob_detect.py     # unit: pattern A/B, mitigation, partial consumption
  test_ew_ob_entry_exit.py    # unit: armed→pullback→entry, SL/TP ordering
  test_ew_ob_aug20_anchors.py # integration: real Aug 20 data → 2 congruency trades
```

The engine is **sequential over the concatenated index stream** (state carries across
sessions — this is a core spec requirement). No per-day parallelism for the signal pass;
option settlement is resolved inline via cached day files.

## Data loading (`ew_ob_runner.py`)

1. `spot_all = load_spot()` (index 1m, includes 2015→2026-05-15).
2. `opt_map = option_day_files("2020-01-01", "2026-08-20")` (AMMU archive).
3. `spot_all, opt_map = extend_with_august(opt_map, spot_all)` (Aug Desktop options + Aug 1m spot).
4. `days = sorted(d for d in spot_all if "2020-01-01" <= d <= "2026-08-20" and d in opt_map)`.
5. Concatenate the days' spot arrays into one continuous stream:
   `bar` = `(day, min, open, high, low, close)`, sorted by `(day, min)`.
   Last bar of day N is immediately followed by first bar of day N+1 for wave counting.
6. Per-day, also build resampled 2m/3m/5m OHLC bar lists (aggregate 1m by minute bucket,
   per day only — no cross-session resample bars).

## Core algorithms

### A. Wave detector — session-continuous state machine

Operates on the full concatenated 1m stream. Candle color: `green` if `close >= open`,
else `red`.

State:
- `pos` — current wave position: `0`=looking for W1 start, `1..5`=W1..W5, `6`=A, `7`=B.
- `wave_start_idx`, `candles_in_wave` (0,1,2) — current 3-candle window.
- `colors_consumed` — the 3 colors of the current window (for matching).
- Per completed wave: `peak` = max(high), `trough` = min(low) of its 3 candles.
- `impulse_origin_idx` — index of bar immediately before W1's first candle (its `low` for
  bullish, `high` for bearish).
- `impulse` = None | dict(direction, w1..w5 peaks/troughs, origin, start_idx, w5_end_idx).
- `armed` — True after B completes following a valid impulse.

Expected color pattern per position (bullish-form; pattern is fixed, direction is decided
by condition 1):
- W1 `[R,G,R]`, W2 `[G,R,G]`, W3 `[R,G,R]`, W4 `[G,R,G]`, W5 `[R,G,R]`, A `[G,R,G]`, B `[R,G,R]`.

Transitions (evaluated once per new bar):
1. Append candle color to a rolling buffer. Maintain `window_start` = start of the
   current candidate 3-candle window (advances by 1 each bar).
2. If the last 3 colors `[c1,c2,c3]` match the pattern required at `pos`:
   - record the window as the completed wave (peak/trough from the 3 bars);
   - if `pos == 0` (no W1 yet): this window becomes W1; set `impulse_origin_idx = window_start - 1`
     if that bar exists (else this W1 is invalid → keep scanning);
   - advance `pos += 1`; reset `window_start` to current bar index + 1 (waves are
     non-overlapping 3-candle windows).
3. If the last 3 colors do **not** match: if `pos > 0`, reset `pos = 0` and drop the
   partial impulse; continue scanning for a new W1 (`window_start` advances by 1).
4. On completing W5 (`pos` 5→6), evaluate **Condition 1**:
   - bullish: `W5.peak > W1.peak and W5.peak > W3.peak and origin.low < W2.trough and origin.low < W4.trough`
   - bearish: `W5.trough < W1.trough and W5.trough < W3.trough and origin.high > W2.peak and origin.high > W4.peak`
   - If it passes: store `impulse = {direction, ..., start_idx, w5_end_idx}`. If it fails,
     reset `pos = 0`.
5. Complete A (`pos` 6→7). Complete B (`pos` 7→8): if `impulse` is set → `armed = True`,
   `candidate_obs = OBs formed on any TF between impulse.start_idx and impulse.w5_end_idx`
   (snapshot at W5 completion, refreshed if a new impulse arms). Then reset `pos = 0`
   (C is ignored; scanning continues for the next W1).
6. If a new impulse arms while a position is open: update `candidate_obs`; entry still
   deferred until flat (one position at a time).

Design decisions locked (from spec): waves are exactly 3 consecutive candles; condition 1
is the ONLY validation on the impulse; C wave is ignored entirely.

### B. OB detector (per TF: 1m, 2m, 3m, 5m)

On each TF's per-day bar list, for every bar X:
- **Pattern A:** `X.low < (X-1).low` and `(X+1).high > X.high` → OB zone `[X.low, X.high]`.
- **Pattern B:** `X.high > (X-1).high` and `(X+1).low < X.low` → OB zone `[X.low, X.high]`.

OB record: `{tf, zone=[lo, hi], formed_idx, formed_bar, untouched_top=hi, dead=False}`.
The pattern itself has no direction — direction comes only from the impulse.

Mitigation / partial consumption (per spec: "never traded through the full zone; only
untouched portion reusable; one use per OB portion"):
- A subsequent bar whose `low < lo` (breaks below the entire zone) → `dead = True`.
  (Mirror: `high > hi` also marks dead for the bearish side.)
- A bar that trades inside the zone from above (`lo <= bar.low <= untouched_top`) reduces
  `untouched_top = min(untouched_top, bar.low)` — the portion above the touched level is
  consumed.
- Entry pulls INTO the usable portion: bullish requires `untouched_top - TOL <= bar.low`
  (i.e., reaches the untouched portion or within tolerance of its top) and `bar.low >= lo`;
  after entry, consume: `untouched_top = min(untouched_top, entry_bar_low)`.
- OBs persist across sessions in a registry (no daily reset) but are only *entered* in the
  current session.

### C. Entry (per spec §4)

When `armed` and flat:
- For each candidate OB (formed during the impulse window, any TF, not dead, usable
  portion remaining), check the current 1m bar:
  - bullish (up impulse): `bar.low <= untouched_top + TOL and bar.low >= lo` → pullback
    into zone from above (near-edge tolerance default `TOL = 0.5`).
  - bearish (down impulse): `bar.high >= untouched_top - TOL and bar.high <= hi`.
- First qualifying bar in the current session (`SESSION_START=555` (09:15) … `SESSION_END=930`
  (15:30)) → **enter at that bar's close**.
- Strike: `spot_px = latest_value(spot, minute)`; `atm = round(spot_px/50)*50`;
  `strike = atm + (CE_OFFSET if up else PE_OFFSET)`; side `CE`/`PE` per impulse direction.
- Option entry premium = option bar `close` at entry minute (nearest prior bar if missing).
- Consume the OB portion; mark armed handled for this trade.

### D. Exit (per spec §5, index-based)

- `SL = entry_close ∓ 3 × ATR(10)_index` (below for CE, above for PE).
- `TP = entry_close ± 60` index pts.
- Each subsequent 1m bar: if `bar.high >= TP` → TP hit; if `bar.low <= SL` → SL hit.
  Same-bar both → SL first (conservative; documented decision).
- On the hit bar, exit at that bar's close; option exit premium = option bar close at the
  exit minute. `P&L = (exit_prem - entry_prem) × 65 - fees`.
- Force-flat at end of day (15:30 bar) if still open.
- ATR(10) = simple mean of the last 10 true-ranges on the 1m index stream up to the entry
  bar (warmup = 10 bars before first use; else skip that entry).

### E. Multi-trade

One position at a time. Every valid armed setup is traded sequentially (no daily cap, no
daily loss limit). New entries deferred while a position is open.

## Fees & P&L

Reuse the model in `backtest_walkforward_fees.py` (`trade_cost`): slippage 1 pt/side,
STT 0.0625% (sell), exchange 0.035% (both), SEBI 0.0001%, stamp 0.003% (buy), GST 18% on
charges, brokerage 0 (default). Report per-trade `pts`, `pts_net`, `fee`, `rs_net` and a
summary (trades, WR, net pts, net ₹, PF, MaxDD, per-year table).

## Runner CLI

```
python artifacts/ew_ob/ew_ob_runner.py --smoke            # Aug 19-20 + first 5 days of 2020
python artifacts/ew_ob/ew_ob_runner.py --full             # all days 2020-01-01..2026-08-20
python artifacts/ew_ob/ew_ob_runner.py --full --tol 0.0 --atr-mult 3.0 --tp 60
python artifacts/ew_ob/ew_ob_runner.py --sweep            # tol [0,0.25,0.5,0.75,1.0] × atr-mult [2,3,4] × tp [40,60,80]
```

Output: console table + `artifacts/ew_ob/results_<variant>.json` (overwrite on full run;
AGENTS.md requires deleting the previous JSON before `--full`).

## Implementation phases

### Phase 0 — engine core (unit-test driven)
1. `ew_ob_engine.py`: `WaveDetector` state machine per §A. Pure bar consumption.
2. `ew_ob_engine.py`: `OrderBlockTracker` per TF per §B, plus the session-persistent registry.
3. `ew_ob_engine.py`: `entry()` + `exit()` + `ATR10` + option settlement per §C/D/E.
4. `tests/test_ew_ob_wave_detect.py`, `test_ew_ob_ob_detect.py`, `test_ew_ob_entry_exit.py`:
   synthetic inputs covering: valid 5-wave impulse both directions; W1 needs a prior bar;
   pattern-break resets; condition-1 pass/fail; OB pattern A and B; through-trade kill;
   partial consumption; entry pullback; same-bar SL-first; EOD flatten.
5. `python -m pytest tests/ -k ew_ob` — green.

### Phase 1 — runner + data
6. `ew_ob_runner.py`: load + concatenate data (§Data loading), wire engine, report/summarize,
   write JSON. Windows guard: `if __name__ == "__main__":` (no Pool needed — sequential).
7. `python artifacts/ew_ob/ew_ob_runner.py --smoke` → verify Aug 19–20.

### Phase 2 — smoke test (MANDATORY, AGENTS.md)
8. **Congruency anchors (Aug 20, 2026):** assert exactly the expected trades fire:
   - Trade 1: **CE** entry in the **~11:15–11:18** window — OB1 = 09:52–09:54 purple box,
     zone `24,196.00–24,199.97`; touch bar low `24,200.1` (≤ zone top + 0.5) → entry at the
     11:15 bar close (both 11:15 and 11:18 are valid; low-touch fires the first qualifying bar).
   - Trade 2: **CE** entry in the **~12:24–12:25** window — OB2 = 11:40–11:42 blue box, zone
     `24,208.71–24,213.24`; touch bar 12:25 low `24,209.95` (inside zone) → entry at 12:25 close.
   - Sanity: 1–10 trades/day, WR 35–50%, net sign check.
9. Also run the 5-day 2020 reference: 15–40 trades, WR 30–50%.
10. If anchors DON'T reproduce → debug against the decoded drawing data
    (Impulse 1: origin 09:33@24,185.03 → W5 end 10:16@24,227.01; correction A 10:17, B 10:23,
    C end 10:37; Impulse 2: 11:26 → 11:56@24,221.18). Fix wave-window alignment or
    tolerance before any full run.

### Phase 3 — full run + sweep
11. `python artifacts/ew_ob/ew_ob_runner.py --full` → 1,588 days. Report trades/WR/net ₹/PF/MaxDD.
12. `python artifacts/ew_ob/ew_ob_runner.py --sweep` → pick best stable tolerance/ATR/TP.
13. Record results in `BACKTEST_LEDGER.md`; run `python run_graphify.py` after the `.py` edits.

## Definition of done
- Unit tests green (`pytest tests/ -k ew_ob`).
- Smoke passes: Aug 20 produces the 2 congruency trades at the anchor windows; sanity
  ranges hold; 2020 5-day reference sane.
- Full run completes with a per-year report + JSON artifact.
- Results appended to `BACKTEST_LEDGER.md`; knowledge graph rebuilt.

## Risks / open decisions
- **Wave window alignment:** strict non-overlapping 3-candle windows may not align with the
  drawing's vertices on real data; smoke anchors are the arbiter. If needed, allow
  overlapping-candidate windows with `window_start` advance-by-1 (keep first match).
- **C-wave role:** ignored per spec; if a C-based re-arm is later wanted it's additive.
- **Same-bar SL/TP:** SL-first (conservative).
- **Option bar gaps:** use nearest prior option bar; skip trade if no option slice exists.
- **ATR(10):** simple mean of last 10 TRs (Wilder variant is a sweep axis if needed).