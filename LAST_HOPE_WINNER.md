# Last Hope GPU Engine — Net-Points (Max-Profit) Winner

> **Engine:** `run_7y_v4_master.py` (Master V4) + `gpu_sim_last_hope.py` (GPU sim)
> **Question answered:** "What single configuration maximizes net profit (₹) across the 7-year Last Hope sweep, and exactly how do you build it?"
> **Answer (base):** the **B-kind, bias-OFF** config `arm_window=10, atr_sl=True, atr_mult=1.5, atr_period=10` — **₹1,775,684 net, 61.1% win rate, 24,761 trades (2020-01-01 → 2026-08-27, 1,512 days).**
> **Answer (research-improved):** adding a **breakeven stop** (`be_trigger=0.70, be_buffer=1.0`) to the base lifts it to **₹1,800,483 net, 62.24% win rate, 24,990 trades**, with *lower* max drawdown (₹16,246 vs ₹17,913) and *higher* PF / Sharpe / Sortino / Calmar. See §10.
> **Answer (touch-buffer optimized — NEW BEST):** tightening the SR-bounce touch buffer to **0.0** (candle must touch/pierce the S/R level, no gap tolerance) lifts it to **₹2,108,703 net, 63.89% win rate, 24,198 trades**, with *even lower* max drawdown (₹9,303) and *higher* PF / Sharpe / Sortino / Calmar. See §11.
> **Companion results files:** `sweep_biasoff_ratios.csv` (base, 608-config bias-OFF sweep), `sweep_research.csv` (research-improvement grid, 864 configs), `sweep_touch_buffer.csv` (touch-buffer sweep, 28 configs), and `sweep_touch_fixed.csv` (fixed SL/TP sweep, 84 configs). See also `FIXED_TP_SL_WINNER.md` for the risk-adjusted alternative (fixed SL=7 / TP=15, PF 3.0).

---

## 1. Headline Result

| METRIC | VALUE |
|:---|---:|
| **Net Profit (₹)** | **+2,108,703.23** |
| Gross Points (Σ exit−entry) | 46,236 (≈1.91 pts/trade) |
| Win Rate | **63.89%** |
| Total Trades | 24,198 |
| Avg Trades / Day | 16.04 |
| Max Drawdown (₹) | 9,302.62 |
| Profit Factor | 1.390 |
| Payoff (avg W / avg L) | 0.835 |
| Expectancy (₹/trade) | 87.14 |
| Sharpe (daily, annualized) | 9.055 |
| Sortino (daily, annualized) | 23.602 |
| Pareto share (top-20% profit) | 0.517 |
| Calmar (net / maxDD) | 226.68 |
| Avg SL distance (pts) | 8.329 |
| Avg TP distance (pts) | 8.464 |
| Lot size | 65 | Fee (₹/trade) | 45 |

Rank vs all tested configurations: **#1 by net profit**, #1 by Sharpe, #1 by Sortino, #1 by Calmar, #1 by expectancy. Strictly dominates the prior research winner (₹1,800,483) on every metric: net +₹308,220 (+17%), WR +1.65pp, maxDD −43%, PF +0.080, Sharpe +1.46, Sortino +6.26, Calmar +115.8.

---

## 2. The Exact Winning Configuration

```python
dict(
    kind='B',          # ATR SL == TP (SL/TP distance = min(ATR*mult, TP_PTS))
    sl=15, tp=15,      # ignored when atr_sl=True (capped at TP_PTS=15)
    arm_window=10,     # flag/super must fire within 10 bars of arming
    use_elder=False,   # index Elder impulse gate OFF
    use_rsi=False,     # option-chart RSI gate OFF
    reversal=False,    # cross-side reversal entries OFF
    atr_sl=True,       # use ATR-based stop distance
    atr_mult=1.5,      # SL/TP distance = min(ATR(atr_period) * 1.5, 15)
    atr_period=10,     # ATR lookback = 10 bars (on the option chart)
    cap=0,             # no daily P&L cap (positions exit only at SL/TP)
    use_bias=False,    # 15m Marni-Fib bias OFF (the single biggest lever)
    # --- research improvement: breakeven stop (web-derived) ---
    be_trigger=0.70,   # after price reaches 0.70*SL-distance above entry, harden the stop
    be_buffer=1.0,     # ...move SL to entry + 1 pt (bank a small profit, beat noise)
    tp_frac=1.0,       # full distance TP (do NOT shrink target)
    # --- touch-buffer optimization (sweep-derived) ---
    touch_buffer=0.0,  # SR-bounce: candle must TOUCH/pierce the S/R level (no gap tolerance)
    entry_start=0,     # entry window: full session (window gating hurt net)
    entry_end=345,
    max_bars=0,        # staleness exit OFF (also hurt net)
)
```

**One-line thesis:** drop the directional bias entirely, widen the stop to a volatility-scaled ~9 pts (ATR×1.5, capped at 15), tighten the SR-bounce touch to zero tolerance (candle must actually touch/pierce the level), and let the raw Flag+Super stochastic triggers + option-chart SR-bounce speak for themselves. Bias-OFF beats every bias-ON variant, and touch_buffer=0.0 beats every larger buffer (monotonic, no inflection).

---

## 3. Files & Data

| ITEM | PATH |
|:---|:---|
| Master engine (builds all state) | `C:\Websites\FLATTRADE BOT\run_7y_v4_master.py` |
| GPU simulation core | `C:\Websites\FLATTRADE BOT\gpu_sim_last_hope.py` |
| Sweep generator | `C:\Websites\FLATTRADE BOT\gpu_sweep_batch.py` (`gen_grid()`) |
| Enriched-ratios sweep (produced the base winner) | `C:\Websites\FLATTRADE BOT\gpu_sweep_ratios.py` |
| Results CSV (base, 608-config) | `C:\Websites\FLATTRADE BOT\sweep_biasoff_ratios.csv` |
| Research-improvement sweep (breakeven grid, 864-config) | `C:\Websites\FLATTRADE BOT\gpu_sweep_research.py` |
| Results CSV (research grid) | `C:\Websites\FLATTRADE BOT\sweep_research.csv` |
| Touch-buffer sweep (28-config) | `C:\Websites\FLATTRADE BOT\gpu_sweep_touch.py` |
| Results CSV (touch-buffer grid) | `C:\Websites\FLATTRADE BOT\sweep_touch_buffer.csv` |
| Fixed SL/TP variant doc | `C:\Websites\FLATTRADE BOT\FIXED_TP_SL_WINNER.md` |
| Fixed SL/TP sweep (84-config) | `C:\Websites\FLATTRADE BOT\sweep_touch_fixed.csv` |
| Canonical option parquet | `C:\Users\user\Desktop\nifty50 data\nifty50_options_master.parquet` |
| Working parquet copy (engine reads this) | `C:\Users\user\AppData\Local\Temp\opencode\data\nifty50_options_master.parquet` |
| Index 1m CSV (for bias / Elder / RSI / LinReg) | `C:\Users\user\AppData\Local\Temp\opencode\data\NIFTY 50_minute.csv` |
| Bias builder (also decomposes bias into LR / UT modes) | `C:\Websites\FLATTRADE BOT\bias15m.py` |

> **Data caveat:** the engine reads the parquet from `AppData\Local\Temp\opencode\data`. Windows Defender AV scan on the 472 MB parquet stalls the import 90–290 s. Launch via a detached process and poll (see §7), never interactively with a short timeout. The canonical source lives on the Desktop.

---

## 4. Engine Constants (from `run_7y_v4_master.py`)

| CONSTANT | VALUE | MEANING |
|:---|---:|:---|
| `S1_K, S1_D` | 12, 3 | Fast stochastic lookback / smoothing |
| `S3_K, S3_D` | 40, 4 | Slow stochastic #3 |
| `S4_K, S4_D` | 50, 10 | Slowest stochastic (macro) |
| `ARM_S1` | 25.0 | Arming threshold: S1 ≤ 25 arms flag & super |
| `M6_S4` | 79.5 | Flag (M6) entry: S4 ≥ 79.5 |
| `M6_S1` | 79.5 | Flag (M6) entry: S1 < 79.5 |
| `REV_EMBED` | 14 | Reversal: S4 embedded ≤ 20.5 for ≥14 bars |
| `SL_PTS` | 7.0 | Default fixed stop (unused when `atr_sl=True`) |
| `TP_PTS` | 15.0 | Hard cap on SL/TP distance (engine clamps `min(dist, 15)`) |
| `LOT, FEE` | 65, 45 | Contract size / flat fee per trade |
| `SESSION_START, SESSION_END` | 555, 900 | 09:15 → 15:00 IST |
| `T1` | 345 | bars per session |
| `RSI_CE_HI / RSI_PE_LO` | 60.0 / 40.0 | RSI entry gates (OFF in winner) |
| `USE_BIAS` (default) | True | 15m Marni-Fib bias (winner overrides → OFF) |

The stochastic series `pe_s1/s3/s4`, `ce_s1/s3/s4` on the option chart, plus `pe_super_full`, `pe_m6_full`, `pe_rev_on` (and CE mirrors) are precomputed in the master import and loaded as GPU tensors.

---

## 5. Entry Logic (exactly as implemented)

**Arming** (`gpu_sim_last_hope.py:336-348`): when `S1 ≤ ARM_S1 (25)` on a flat bar, both the Flag and Super "armed" flags are set and timestamped (`pe_arm_t`). The armed state expires after `arm_window` bars (10 for the winner).

**Trigger signals** (precomputed in master, `run_7y_v4_master.py:386-413`):
- **Flag / M6:** `pe_m6_full = (S4 ≥ 79.5) & (S1 < 79.5)` on the option chart, gated by the arm window.
- **Super:** `pe_super_full = (S3 < 25) & (S4 < 25) & (S1 < 25) & (rising > 0.5)` (S1 turn-up), gated by arm window.
- **Reversal:** `pe_rev_on = sustain_runs(S4 ≤ 20.5) ≥ 14 & super` → opposite-side entry. OFF in winner (`reversal=False`).

**SR bounce required** (`gpu_sim_last_hope.py:165,370,400`): `bounce_pe` / `bounce_ce` — the bar must touch an option-chart support/resistance level from the **S/R suite** (CPR BC / Pivot / TC, Camarilla H3 / L3, EMA20, EMA200, VWAP) and bounce, verified on both the main chart **and** combined lower-timeframe (TF) low/close. The `touch_buffer` parameter controls the gap tolerance: **0.0** means the candle must actually touch/pierce the S/R level (no buffer). Without an SR bounce, no entry.

**PE entry** (`gpu_sim_last_hope.py:370-394`):
```
pe_cand = (pe_m6 OR pe_super OR pe_rev_sig)
          AND flat (no position)
          AND not daily-cap-hit
          AND bounce_pe
          AND NOT elder-block            (elder OFF → no-op)
          AND (bias == -1 OR bias OFF)   (bias OFF → no-op)
          AND NOT in no-trade-window     (window filter OFF in winner)
          AND NOT in VWAP↔ST zone        (st-zone filter OFF in winner)
          AND (NOT use_rsi OR rsi < 40)  (rsi OFF → no-op)
entry PE at close; SL = close − dist; TP = close + dist; side = PE
```

**CE entry** (`gpu_sim_last_hope.py:396-420`) is the mirror: `ce_cand = (ce_m6 OR ce_super OR ce_rev_sig) AND flat AND bounce_ce AND gates`, with `bias == +1 OR bias OFF` for the bullish side. Only **one position at a time** (`pos_side == 0` guard).

> The winner has **all auxiliary filters OFF** (`use_elder=False`, `use_rsi=False`, `use_bias=False`, no-trade window OFF, ST-zone OFF, reversal OFF). The only active gating is: Flag/Super arm window + option-chart SR bounce. This is the cleanest, most-parameter-light configuration in the entire sweep.

---

## 6. Exit Logic (exactly as implemented)

**Per-bar, SL has priority over TP** (`gpu_sim_last_hope.py:302-312`):
```
sl_hit = (PE & pe_low ≤ sl_price) | (CE & ce_low ≤ sl_price)
tp_hit = (PE & pe_high ≥ tp_price) | (CE & ce_high ≥ tp_price)
do_sl  = in_pos & sl_hit
do_tp  = in_pos & ~sl_hit & tp_hit     # TP only if SL did not also hit
pnl    = (exit − entry) * 65 − 45
```
If both SL and TP could trigger on the same bar, SL wins (conservative).

**Stop distance** (`gpu_sim_last_hope.py:384-389`):
```python
# When atr_sl=True (winner):
atr_dist = min(ATR(option chart, period=10) * atr_mult(1.5), TP_PTS(15))
sl_dist = atr_dist    # symmetric
tp_dist = atr_dist    # symmetric
SL = entry - sl_dist
TP = entry + tp_dist * tp_frac   # tp_frac=1.0 → full distance
```
Because `atr_sl=True`, the `sl=15/tp=15` fields are ignored and the distance is purely ATR-driven, capped at 15 pts. Empirical average distance ≈ **8.33 pts SL / 8.46 pts TP** (ATR×1.5 lands near ~9 pts on most bars; a few bars hit the 15-pt cap).

**End-of-session close:** only fires for **uncapped** configs (`cap != 0`). The winner has `cap=0`, so positions **only ever exit at SL or TP** — there is no EOD forced close and no "run to session end" behaviour.

**Breakeven stop (research improvement, `gpu_sim_last_hope.py:350-359`):** once a position's high reaches `entry + be_trigger * SL_distance` (default off; winner uses `0.70`), the SL is hardened to `entry + be_buffer` (winner uses `1.0` pt) and never moves again (`be_done` latch). This converts the tail of would-be full-stop losers into small banked winners, lifting both net and WR without shrinking the TP target. `be_trigger` must be *late* (well inside the noise band) — the web research explicitly warns that a BE at 1R (full distance) sits in the noise and *hurts* expectancy; a later trigger (0.70 here) plus a +1pt buffer is what works.

---

## 7. How to Reproduce

**Minimal single-config run** (drop into a `.py` next to the engine files):
```python
import sys, time
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
import run_7y_v4_master as M          # imports & builds all GPU tensors (slow: parquet AV scan)
import gpu_sim_last_hope as G

cfg = dict(kind='B', sl=15, tp=15, arm_window=10, use_elder=False, use_rsi=False,
            atr_sl=True, atr_mult=1.5, atr_period=10, reversal=False, cap=0, use_bias=False,
            be_trigger=0.70, be_buffer=1.0, tp_frac=1.0, touch_buffer=0.0,
            entry_start=0, entry_end=345, max_bars=0)

t0 = time.time()
trades = G.gpu_sim_batch([cfg])[0]    # B=1 batch
r = M._metrics_from_trades(trades)
print(f"trades={r['trades']} wr={r['wr']:.4f} net_rs={r['net_rs']:.2f} max_dd={r['max_dd']:.2f}")
print(f"elapsed={time.time()-t0:.0f}s")
```
Expected: `trades=24198 wr=0.6389 net_rs=2108703.23 max_dd=9302.62` (matches `sweep_touch_buffer.csv` row BE buf=0.0).

**Launch under the reliable detached pattern** (the parquet import hangs 90–290 s under AV scan; do not use a short interactive timeout):
```powershell
$proc = Start-Process python -ArgumentList "-u","repro_winner.py" `
    -RedirectStandardOutput "winner_out.log" -RedirectStandardError "winner_err.log" `
    -NoNewWindow -PassThru
for ($i=0; $i -lt 22; $i++) { Start-Sleep 30; if ($proc.HasExited) { break } }
Get-Content winner_out.log
```

**Regenerate the whole 608-config bias-OFF sweep** (enriched ratios):
```powershell
Start-Process python -ArgumentList "-u","gpu_sweep_ratios.py" `
    -RedirectStandardOutput "ratios_out.log" -RedirectStandardError "ratios_err.log" `
    -NoNewWindow -PassThru
# poll ~22 × 30s; result -> sweep_biasoff_ratios.csv
```
`gpu_sweep_ratios.py` reuses `gen_grid()` from `gpu_sweep_batch.py` (608 configs) and adds `use_bias=False` to every config before calling `G.gpu_sim_batch`.

---

## 8. Why This Wins (analysis)

1. **Bias is the biggest drag.** Across every bias mode tested (combined EMA bias ₹0.77M, LinReg-plot bias ₹1.03M, UT-colour bias ₹1.10M), all are beaten by **bias-OFF ₹1.78M**. The 15m Marni-Fib bias filter throws away too many valid SR-bounce setups; raw stochastic + SR-bounce is a higher-quality unconditional signal.
2. **Volatility-scaled stop beats fixed 7-pt stop.** The original 7-pt stop bled ~62% of trades to SL; scaling to ~9 pts (ATR×1.5) recovers far more winners. Because SL and TP are symmetric, the payoff ratio is ~0.84 (losers slightly larger) yet WR is 64% — the high hit-rate carries the edge.
3. **SR-bounce gating is the quality core.** Every entry must touch a known option-chart S/R level and bounce, which keeps the trade count high (24.2k) but the loss rate low.
4. **All auxiliary filters hurt.** Midday no-trade window (11:30–13:30) and VWAP↔Supertrend "zone" filter were each swept with bias on/off and **reduced net profit in every case** (midday is profitable; skipping it loses edge). They are OFF in the winner.

---

## 9. Caveats / Honest Limits

- **In-sample.** This is a full-window (2020→2026) optimization pick, not walk-forward. The 7-year net is the highest of all tested configurations but has **not** been validated OOS. The sweep was purely P&L-maximizing; PF is ~1.39, so the edge is moderate per-trade (expectancy ₹87.14).
- **Cost model.** ₹45 flat fee/lot, **no slippage, no liquidity constraint**. Realistic fills (1 pt/leg slippage + STT/exchange/SEBI as in other ledger sections) would lower net materially.
- **`cap=0` (uncapped).** No daily-loss or daily-profit circuit breaker. Drawdown is small in aggregate (₹9.3k max) but a live account would want a daily halt.
- **Engine path.** CUDA-graph (`_sim_loop`) is disabled (segfaults on torch 2.5.1+cu121/Windows); only the eager `_eager_sim_core` path is used. Same-path parity verified in prior sessions.
- **Fixed SL/TP alternative.** A separate sweep (`FIXED_TP_SL_WINNER.md`) found that fixed SL=7 / TP=15 with the same breakeven + touch_buffer settings yields PF 3.0 (vs 1.39) but lower net (₹1.67M vs ₹2.11M). Choose based on risk tolerance: ATR for max profit, fixed SL/TP for max risk-adjusted returns.
- **Next step (per user direction).** Any **trailing-stop** variant must be swept in a **separate grid**, kept away from this non-trailing sweep, so the green (consistent) configs are not contaminated by trailing's inconsistent trade counts.

---

## 10. Research Improvements — Web-Derived Breakeven Stop

**Directive:** deep web research to find techniques that raise net points *and* win rate, implement them as engine parameters, and re-sweep. Four prior + two confirmatory searches yielded a consistent, implementable thesis for **long-options buyers**:

1. **Breakeven stop after a real move** — move the stop to entry (or entry+buffer) once price has traveled a meaningful fraction of the way to target. Research (TradeAlgo) shows this is the legitimate free-roll; but a BE at **1R sits inside the noise band and *hurts* expectancy**, so the trigger must be *late* (we use 0.70 of the SL distance) with a small **buffer** to beat per-bar noise. This is the only knob that raised *both* net and WR simultaneously.
2. **50%-of-target / 50%-rule take-profit** — close at a fraction of the distance. Implemented as `tp_frac` (0.5 / 0.7). It spikes WR (up to 75%) but **collapses net** (target halved → ₹397K), so it is rejected for a net-points objective.
3. **Time-of-day entry gating** (skip the open / skip the 12:30–1:30 lunch / no entries after ~2:45 PM). Implemented as `entry_start` / `entry_end` bar windows. **Counter-productive here** — the SR-bounce + stochastic triggers already self-filter; every restricted-window config scored *below* the full-session baseline. The generic "avoid the open" lore does not hold for this signal set.
4. **Staleness exit** (close if no directional move in N bars). Implemented as `max_bars`. Also **hurt net** in this sweep (the base already exits fast at SL/TP).

**Implementation (all default OFF in `_eager_sim_core`, so the 608 base sweep is untouched):**
- `be_trigger` (float, 0=off): fraction of SL-distance at which to harden the stop.
- `be_buffer` (float): pts above entry the hardened stop sits at.
- `tp_frac` (float, 1.0=full): scales the TP distance (50%-rule).
- `entry_start` / `entry_end` (bar index): entry-window gate.
- `max_bars` (int, 0=off): staleness exit.

**Re-sweep:** `gpu_sweep_research.py` — 864-config grid (entry_start × entry_end × max_bars × be_trigger × be_buffer × tp_frac) on the base winner, 7y, bias OFF. Completed in ~69 s, 0 errors.

**Result — global max net in the grid:**

| KNOB | VALUE |
|:---|:---:|
| be_trigger | **0.70** |
| be_buffer | **1.0** |
| tp_frac | 1.0 |
| entry_start / entry_end | 0 / 345 (full session) |
| max_bars | 0 (off) |
| **Net Profit** | **+1,800,482.81** |
| **Win Rate** | **62.24%** |
| Trades | 24,990 |
| Max DD | 16,245.94 |
| PF / Sharpe / Sortino / Calmar | 1.310 / 7.591 / 17.345 / 110.83 |

**Conclusion:** the breakeven stop is the single research-derived lever that improves the strategy on every axis (net +1.4%, WR +1.1pp, maxDD −9%, all risk ratios up). The other three web-suggested levers (50%-rule target, entry-window, staleness exit) were implemented and tested but **rejected** for this strategy — they each lower net profit. Trailing-stop remains a **separate, unswept** grid per user direction.

---

## 11. Touch-Buffer Optimization (NEW BEST)

**Directive:** parameterize the SR-bounce touch buffer (previously hardcoded at 1.0) and sweep across fine-grained values to find the optimum.

**Implementation (`gpu_sim_last_hope.py`):**
- `_build_bounce` signature: `def _build_bounce(..., buf=1.0)` — body uses `sr + buf` (lines 155, 160).
- Precomputed stacks: `TOUCH_BUFFERS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]` (14 values). `bounce_pe_stack` / `bounce_ce_stack` shape `(14, D, T1)`.
- Per-config selection in `_eager_sim_core`: `touch_buffer` param → nearest-match index → `bounce_pe_sel[:, :, t]` / `bounce_ce_sel[:, :, t]`.
- Float precision fix: nearest-match lookup instead of `TOUCH_BUFFERS.index(float(v))`.

**Sweep:** `gpu_sweep_touch.py` — 28 configs (14 buffers × 2 modes: FLAT + BE). Completed in ~4s.

**Results — monotonic, no inflection (smaller = strictly better):**

| Mode | buf | net_rs | WR% | Trades | Max DD | PF | Sharpe | Sortino | Calmar |
|:---|:---:|---:|:---:|---:|---:|:---:|---:|---:|---:|
| FLAT | 0.0 | 2,084,584 | 63% | 23,987 | 9,978 | 1.375 | 8.86 | 22.72 | 208.9 |
| **BE** | **0.0** | **2,108,703** | **64%** | **24,198** | **9,303** | **1.390** | **9.05** | **23.60** | **226.7** |
| FLAT | 0.1 | 2,051,186 | 62% | 24,061 | 10,770 | 1.366 | 8.71 | 22.07 | 190.5 |
| BE | 0.5 | 1,946,181 | 63% | 24,600 | 11,862 | 1.347 | 8.30 | 20.20 | 164.1 |
| BE | 1.0 | 1,800,483 | 62% | 24,990 | 16,246 | 1.310 | 7.59 | 17.35 | 110.8 |
| BE | 5.0 | 1,117,608 | 60% | 26,996 | 39,412 | 1.165 | 4.49 | 8.37 | 28.4 |

**Key finding:** touch_buffer=0.0 (strict touch/pierce, no gap) beats the prior champion (buf=1.0) by +₹308K (+17%), halves the drawdown, and nearly doubles the Calmar ratio. The monotonic trend with no floor suggests this is at or near the true optimum for this parameter.

---

## 12. Implementation Reference (for agents)

This section provides everything needed to reproduce, modify, or extend the strategy without reading the engine source.

### 12.1 Architecture

```
run_7y_v4_master.py          gpu_sim_last_hope.py
┌─────────────────────┐      ┌──────────────────────────┐
│ Builds ALL state:   │      │ Imports M (triggers build)│
│  - stochastic series│      │                          │
│  - ATR              │──────│  Static tensors (CE/PE)  │
│  - bounce stacks    │      │  _build_bounce()         │
│  - Elder/RSI/bias   │      │  _eager_sim_core()       │
│  - SR levels        │      │  gpu_sim_batch()         │
│  - parquet → GPU    │      │                          │
└─────────────────────┘      └──────────────────────────┘
         ↓                               ↓
   M.pe_c, M.pe_h, ...          G.gpu_sim_batch([cfg])
   M.trading_days               → list of trade-lists
   M._metrics_from_trades()     → dict of metrics
```

### 12.2 Tensor Shapes

All tensors are `(D, T1)` unless noted. `D` = number of trading days (1509), `T1` = bars per session (345, 09:15–15:00 IST).

| Tensor | Shape | Dtype | Description |
|:---|:---:|:---|:---|
| `pe_c, pe_h, pe_l` | `(D, T1)` | float32 | PE option chart OHLC (close/high/low) |
| `ce_c, ce_h, ce_l` | `(D, T1)` | float32 | CE option chart OHLC |
| `pe_s1, pe_s3, pe_s4` | `(D, T1)` | float32 | PE stochastic series (fast/slow/slowest) |
| `ce_s1, ce_s3, ce_s4` | `(D, T1)` | float32 | CE stochastic series |
| `pe_super_full` | `(D, T1)` | bool | PE super trigger (S1 turn-up) |
| `pe_m6_full` | `(D, T1)` | bool | PE flag/M6 trigger |
| `pe_rev_on` | `(D, T1)` | bool | PE reversal signal |
| `pe_atr` | `(D, T1)` | float32 | PE ATR(10) on option chart |
| `pe_ema20, pe_ema200, pe_vwap` | `(D, T1)` | float32 | PE indicators for SR suite |
| `bounce_pe_stack` | `(14, D, T1)` | bool | Precomputed bounce for each touch buffer |
| `bounce_ce_stack` | `(14, D, T1)` | bool | Precomputed bounce for each touch buffer |
| `elder_state` | `(D, T1)` | int8 | Elder impulse: -1=red, 0=blue, 1=green |
| `rsi_mat` | `(D, T1)` | float32 | RSI(14) on 3m underlying |
| `bias_lr_grid` | `(D, T1)` | int8 | LinReg bias: -1=bear, 0=neutral, +1=bull |

### 12.3 GPU Batch API

```python
import gpu_sim_last_hope as G
import run_7y_v4_master as M

# Single config
trades = G.gpu_sim_batch([cfg])[0]   # returns list of trade-tuples

# Multiple configs
trade_lists = G.gpu_sim_batch([cfg1, cfg2, cfg3])
# trade_lists[i] = list of trades for config i

# Each trade tuple:
(date, side, kind, entry_price, exit_price, pnl_rupees)
#  date:     str 'YYYY-MM-DD'
#  side:     str 'PE' or 'CE'
#  kind:     str 'SL' or 'TP'
#  entry:    float (option premium at entry)
#  exit:     float (option premium at exit)
#  pnl:      float (exit - entry) * 65 - 45  (in rupees)

# Metrics from trades:
r = M._metrics_from_trades(trades)
# r = {trades, wr, net_rs, net_pts, avg_sl, avg_tp, avg_trades_day, max_dd}
```

**DO NOT use the legacy `gpu_sim()` wrapper** — it lacks params for `kind`, `use_bias`, `be_trigger`, `touch_buffer`, `entry_start`, etc. Always use `gpu_sim_batch`.

### 12.4 Parameter Dictionary (all keys)

```python
cfg = dict(
    # --- core ---
    kind='B',              # 'B' = ATR SL==TP, 'A' = legacy mode
    sl=15,                 # fixed SL pts (used when atr_sl=False)
    tp=15,                 # fixed TP pts (used when atr_sl=False)
    arm_window=10,         # bars: flag/super must fire within N bars of arming
    cap=0,                 # daily P&L cap (0 = uncapped)

    # --- stochastic filters ---
    use_elder=False,       # index Elder impulse gate
    use_rsi=False,         # option-chart RSI gate
    reversal=False,        # cross-side reversal entries

    # --- ATR stop ---
    atr_sl=True,           # True = ATR-based SL/TP; False = fixed sl/tp
    atr_mult=1.5,          # SL/TP dist = min(ATR * mult, TP_PTS)
    atr_period=10,         # ATR lookback on option chart

    # --- bias ---
    use_bias=False,        # 15m Marni-Fib bias filter
    bias_mode='ema',       # 'ema' | 'lr' | 'ut'

    # --- breakeven stop ---
    be_trigger=0.70,       # fraction of SL distance to trigger BE (0=off)
    be_buffer=1.0,         # pts above entry to harden SL

    # --- touch buffer ---
    touch_buffer=0.0,      # SR bounce gap tolerance (0.0 = strict touch)

    # --- research (all default OFF) ---
    tp_frac=1.0,           # scales TP distance (0.5 = 50%-rule)
    entry_start=0,         # first bar index for entries
    entry_end=345,         # last bar index for entries
    max_bars=0,            # staleness exit (0=off)

    # --- zone filters (all OFF in winner) ---
    use_st_zone=False,     # VWAP↔Supertrend zone filter
    nt_start=-1,           # no-trade window start bar
    nt_end=-1,             # no-trade window end bar
)
```

### 12.5 How Parameters Flow Through the Engine

In `_eager_sim_core` (line 209+), each param becomes a `(B,)` tensor indexed per-bar:

```python
# Parameter → tensor conversion (lines 212-248)
sl_b       = torch.tensor([p['sl'] for p in params_list])        # (B,)
atr_sl_b   = torch.tensor([p['atr_sl'] for p in params_list])    # (B,) bool
be_trigger_b = torch.tensor([p['be_trigger'] for p in ...])      # (B,)
touch_buf_b  = torch.tensor([p['touch_buffer'] for p in ...])    # (B,)

# Touch buffer → nearest-match index into TOUCH_BUFFERS (lines 249-254)
tb_idx = [min(range(14), key=lambda i: abs(TOUCH_BUFFERS[i] - v))
          for v in touch_buf_b.tolist()]
bounce_pe_sel = bounce_pe_stack[tb_idx]   # (B, D, T1)
```

**Stop distance calculation (lines 384-389):**
```python
# When atr_sl=True:
atr_dist = min(ATR[d,t] * atr_mult, TP_PTS)
sl_dist = atr_dist
tp_dist = atr_dist

# When atr_sl=False:
sl_dist = min(sl_param, TP_PTS)
tp_dist = min(tp_param, TP_PTS)

# Both paths:
SL = entry - sl_dist
TP = entry + tp_dist * tp_frac
```

### 12.6 How to Add a New Parameter

1. **Add default to the param dict** in `_eager_sim_core` (line ~212):
   ```python
   my_new_b = torch.tensor([float(p.get('my_new', 0.0)) for p in params_list], device=DEVICE)
   ```

2. **Use it in the sim loop** (lines 301-420):
   ```python
   # Example: conditional logic using my_new_b
   condition = my_new_b[:, None] > 0  # (B, 1) broadcasts with (B, D)
   ```

3. **Document it** in §12.4 above.

4. **Add to sweep script** as a grid dimension:
   ```python
   configs = [dict(base, my_new=v) for v in [0.0, 0.5, 1.0]]
   ```

### 12.7 Data Pipeline

```
Raw data (Desktop):
  nifty50_options_master.parquet   → option OHLC (day, minute, symbol, strike, side, OHLCP)
  NIFTY 50_minute.csv              → index 1m (for Elder, RSI, bias, LinReg)

run_7y_v4_master.py (import):
  1. Reads parquet → filters to correct weekly expiry per day
  2. Builds stochastic series (S1/S3/S4) for PE and CE
  3. Builds ATR(10) for PE and CE
  4. Builds bounce stacks (_build_bounce × 14 buffers)
  5. Builds Elder impulse, RSI(14), bias grids from index CSV
  6. Builds SR levels (CPR, Camarilla, EMA20/200, VWAP) from option chart
  7. Loads everything as GPU tensors

gpu_sim_last_hope.py (import):
  1. Imports M → triggers step 1-7 above (slow: 90-290s AV scan)
  2. Copies tensors to local vars (ce_c, pe_h, etc.)
  3. Precomputes Supertrend, combined-TF close/low
  4. Precomputes bounce_pe_stack, bounce_ce_stack (14 buffers)
  5. Ready for gpu_sim_batch() calls
```

### 12.8 Common Pitfalls

| PITFALL | FIX |
|:---|:---|
| Using `gpu_sim()` instead of `gpu_sim_batch()` | Legacy wrapper lacks params for be_trigger, touch_buffer, etc. Always use `gpu_sim_batch([cfg])`. |
| Running interactively with short timeout | Parquet import takes 90-290s. Use `Start-Process` + polling (§7). |
| Modifying `_sim_loop` (CUDA graph path) | Disabled (segfaults on torch 2.5.1+cu121/Windows). Only `_eager_sim_core` works. |
| Using `M.N_DAYS` | Does not exist. Use `M.D` for day count (line 109 of run_7y_v4_master.py). |
| `TOUCH_BUFFERS.index(float(v))` | Fails for sub-0.5 floats due to precision. Use nearest-match: `min(range(14), key=lambda i: abs(TOUCH_BUFFERS[i] - v))`. |
| Expecting `touch_buffer` in `_sim_loop` | `_sim_loop` (disabled) still uses hardcoded `bounce_pe[None, :, t]` (buf=1.0). Only `_eager_sim_core` supports the parameter. |

### 12.9 Quick Validation Checklist

Before claiming a new config works, verify:

1. **Import succeeds:** `python -c "import gpu_sim_last_hope as G; print('OK')"`
2. **Single config runs:** `G.gpu_sim_batch([cfg])[0]` returns a list of tuples
3. **Trade tuple format:** `(str, str, str, float, float, float)` — date, side, kind, entry, exit, pnl
4. **PnL formula:** `(exit - entry) * 65 - 45` matches `t[5]`
5. **Metrics match:** `M._metrics_from_trades(trades)` gives `net_rs` = sum of `t[5]`
6. **Smoke test:** buf=1.0 BE should give net_rs ≈ 1,800,482.81 (known baseline)
7. **Parity:** modified engine still produces exact same output for unchanged configs
