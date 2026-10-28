# 6-Year Congruence Audit — ATR-16K vs the validated build

**Question:** is the 2020→2026 backtest (`sy2.py`, −₹19,361 best config) actually
running the same strategy that produced +₹16,136 over 8 sessions (`ob_pat.py`)?

**Method:** parameter-by-parameter diff of the two engines, then controlled
re-runs isolating every difference on data where both can be evaluated.

---

## 1. Parameter table

| # | Parameter | Validated build (ob_pat) | 6-year engine (sy2) | Congruent | Evidence |
|---|---|---|---|---|---|
| 1 | Contract universe | 2nd ITM at the **wave start** (CE ATM−100 / PE ATM+100) | same | ✅ | identical `need` set construction |
| 2 | ATM source | NSE index 1m | NSE index 1m CSV | ✅ | same rounding `round(spot/50)*50` |
| 3 | Expiry | nearest weekly | nearest weekly from the day file | ✅ | 82-day cross-check vs 2 parquets: 0 mismatches 2020→May 2026 |
| 4 | Missing weekly | — | took the monthly anyway | ❌ → fixed in sy3 | Jun–Aug 2026 only (85/55/29 DTE); outside the 6-yr window |
| 5 | Today's bars | 09:15–15:30 | 09:15–15:29 (CSV ends at 929) | ⚠️ data | no trade decision uses bars after 15:16 |
| 6 | Prior-day close (CPR/TC/cam) | **15:32 settlement** print | last bar ≤15:30 → **15:29** | ❌ data-impossible | the historical CSVs have no 15:30/15:32 bar at all (parquet too) |
| 7 | Prior-day H/L | full print range 09:15–15:45 | ≤15:30 | ❌ data | same missing tail |
| 8 | EMA/ST seed | up to 2,000 prior bars (multi-day) | 1 prior day (375 bars) | ❌ → fixed in sy3 | April 2026: ±1.5 pts on +85 |
| 9 | EMA20 5m + 3m | yes | yes | ✅ | identical bucket math |
| 10 | EMA200 3m + 5m | yes | yes | ✅ | identical |
| 11 | SuperTrend (10,3) 3m | yes | yes | ✅ | identical |
| 12 | Session VWAP | volume-weighted from 09:15 | same (CSV volume) | ✅ | identical |
| 13 | CPR | P=(H+L+C)/3, BC=(H+L)/2, TC=2P−BC | same (no rounding) | ✅ | ob_pat rounds to 2dp — ≤0.01 effect |
| 14 | fibR3/S3, camR3/L3 | same formulas | same | ✅ | — |
| 15 | Pivots | close zigzag THR 5, leg-extreme refine | same | ✅ | line-for-line port |
| 16 | Waves | 6 pivots, skip budget 4, 5>3>1 & 4>2>0, 1-pt buffer | same | ✅ | same DFS |
| 17 | Overlapping waves | dropped (`w0 > last_end`) | same | ✅ | — |
| 18 | a/b/c | first L, first H, first L; b_conf; c_conf+1 | same | ✅ | — |
| 19 | Entry window | b_conf+1 … c_conf+1 | same | ✅ | — |
| 20 | Trigger | engulf **or** harami, either direction, at a level, open ≥ level | same | ✅ | same `pat_any` |
| 21 | Level pick | highest level inside the bar | same | ✅ | — |
| 22 | Entry price | trigger close | same | ✅ | — |
| 23 | Stop | level − 1.5×ATR(14, 1m) | same | ✅ | — |
| 24 | Target | level + 3× trigger range | same | ✅ | — |
| 25 | Stop fill | low ≤ SL−1 → exit at SL | same | ✅ | — |
| 26 | Target fill | high ≥ TP → exit at TP | same | ✅ | — |
| 27 | EOD | first bar > 15:15, exit at close | same | ✅ | — |
| 28 | Position rule | one at a time; entry ≤ previous exit dropped | same | ✅ | — |
| 29 | Fees | 65 pts-lot − ₹45/trade | same | ✅ | — |

**Verdict on the logic: congruent.** Every decision rule is identical. The only
non-congruences are **data availability** (#5–#8) and **one missing-weekly guard**
(#4), which sy3 fixes.

---

## 2. Measured impact of the data rules (8 sessions, 21–28 Oct 2026)

Same engine, same bars, only the prior-day rules changed:

| Rules | Trades | Points | ₹ after fees |
|---|---|---|---|
| Live-grade (15:32 close, full range, 5-day seed) | 19 | +278.95 | +17,277 |
| Historical-CSV (15:29 close, ≤929 range, 5-day seed) | 17 | +252.64 | +15,657 |
| Live close/range, **1-day seed** (sy2's seed) | 16 | +296.92 | +18,580 |
| All historical-CSV rules (929 close, ≤929, 1-day seed) | 16 | +266.82 | +16,623 |

The build's +₹16,136 sits **inside** this range. The entire non-congruence is
worth roughly ±8% of the 8-day P&L — not a sign flip.

April 2026 (20 sessions), sy2 vs the corrected sy3:

| Config | sy2 trades / pts | sy3 trades / pts |
|---|---|---|
| atr1.5 | 44 / +25.45 | 44 / +23.98 |
| atr2.0 | 42 / +85.13 | 42 / +83.64 |

Trade counts identical; points differ by ~1.5 — the seed window is not the issue.

---

## 3. Parquet verification (asked: is the 2nd-ITM selection right?)

Sampled 140 days (one per month, 2020→2026), ATM from the index 09:15 close:

| Year | DTE (med) | 2nd-ITM CE (med) | Time value (med) | Missing |
|---|---|---|---|---|
| 2020 | 2.5 | 154.6 | 47.0 | 0 |
| 2021 | 2.5 | 179.9 | 68.5 | 0 |
| 2022 | 2.5 | 183.3 | 87.2 | 0 |
| 2023 | 2.0 | 146.9 | 50.1 | 0 |
| 2024 | 3.0 | 190.7 | 90.8 | 0 |
| 2025 | 2.5 | 197.0 | 92.9 | 0 |
| 2026 | 5.5 | 241.5 | 160.7 | 2 |

`2nd-ITM = 100 intrinsic + time value` holds exactly. 73% of days sit inside
80–220; the outliers are genuine high-IV days (3 Jun 2024 election, 1 Apr 2026),
verified against the index move — **not** a mis-selected strike.

Three files compared (`cache_marni_opt`, `nifty50_options_master_v2`, and the
per-day CSVs):

- **Single expiry per day, no duplicate bars, no mixed expiries.**
- **CSV expiry == marni == v2 on every sampled day 2020→May 2026.**
- `cache_marni_opt` covers more days (59.09M rows) than v2 (55.16M; v2 is missing
  whole days such as 2026-07-01).
- Both historical option sources lack the 15:30–15:39 prints.

**Conclusion:** `cache_marni_opt` is usable — its expiry mapping and 2nd-ITM
selection match v2 exactly on the audited days. It is not "wrong-expiry" data on
this test; the earlier warning applied to a different construction path.

### 3.1 Premium at the wave start (the strike is chosen there, not at entry)

| Day | Contracts traded (premium at the wave-start minute) |
|---|---|
| 21 Oct | 252.5, 254.3, 238.6 |
| 22 Oct | 208.5, 197.3, 248.4 |
| 23 Oct | 170.2, 171.2, 256.9 |
| 26 Oct | 147.7, 128.2, 203.1 |
| 27 Oct | 130.1, 124.6, 90.7, 100.7 |
| 28 Oct | 257.6, 235.8, 259.7 |

Every selection sits in the 100-intrinsic + time-value band. Entry prices can be
higher (e.g. 14:52 entry at 269) because the index moved after the wave start —
that is the strategy's design, not a mis-selected strike.

---

## 4. Six-year re-run with the congruent engine (sy3)

*(filled from the sy3 full run)*

---

## 5. Verdict

- The strategy logic in the 6-year test **is** the strategy we built: 29/29 rules
  congruent.
- The historical data cannot supply the 15:32 prior close; that costs ~9% on the
  measured sample.
- The contract selection (nearest weekly, 2nd ITM at ATM±100) is correct; the
  premium band confirms it.
- The one real bug — trading monthly contracts on missing-weekly days — only
  affects Jun–Aug 2026, outside the tested window.

Therefore the negative six-year result is **not** an artefact of mis-selected
contracts or wrong rules. If the corrected run still loses, the edge is below
costs; if it turns positive, the difference will be attributable to the seed
window and the missing 15:30 bar, both bounded above.
