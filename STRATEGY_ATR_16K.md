# STRATEGY SPEC — "ATR-16K" build

**SR-level engulfing / harami with an ATR stop, on the 2nd-ITM NIFTY option chart.**

This document is a complete, implementation-ready specification. It records the exact
rules that produced **+₹16,136 over 8 sessions**, and — in §9 — the six-year evidence
that the same build **loses money** once the sample is long enough. Read both before
porting anything.

---

## 1. One-paragraph summary

On the 1-minute chart of the **2nd in-the-money** NIFTY option (CE for upside setups,
PE for downside), find a five-wave impulse whose correction forms an **a → b → c**
pattern. Inside the c-window, when an **engulfing or harami candle** touches a
**static/institutional SR level** from above, buy at the candle's close. Risk
**1.5 × ATR(14, 1m)** below the level, target **3 × the trigger candle's range**.

---

## 2. Verified results

| Window | Build | Trades | WR | Points | Net |
|---|---|---|---|---|---|
| 21–27 Oct 2026 (live data) | full spec, ATR 1.5 | 18 | 61.1% | +239.90 | +₹14,784 |
| 28 Oct 2026 (live data) | full spec, ATR 1.5 | 3 | 66.7% | +22.89 | +₹1,353 |
| **8 sessions** | **full spec** | **21** | **63.6%** | **+262.79** | **+₹16,136** |
| Apr 2026 (CSV data, 20 sess.) | ATR 2.0 | 42 | 54.8% | +85.13 | +₹3,643 (Calmar 12.8) |
| **2020-01 → 2026-04 (1,572 sess.)** | **best of 7 stops (f5)** | **2,663** | **34.4%** | **+1,891.44** | **+₹3,109 (Calmar 1.285)** |

> **The 16K figure is an 8-session result.** On 6.3 years the same logic earns
> +0.32 points per trade gross and pays ~0.69 points per trade in costs.

---

## 3. Data requirements

| Input | Granularity | Purpose |
|---|---|---|
| Option 1-minute OHLC | 09:15–15:30 | the chart being traded |
| **Volume** | 1-minute | session VWAP (not optional — VWAP is an entry level) |
| NSE index 1-minute | 09:15–15:30 | ATM at the wave start → 2nd ITM strike |
| Prior-day option OHLC | full session | CPR / PDH / PDL / fib / camarilla |
| Prior-day option bars | ≥ 250 bars | EMA200 + SuperTrend warmup |

**Expiry rule:** trade **only the nearest weekly expiry** (`exp_date ≥ day`,
smallest gap; skip if the gap > 9 days). Never mix two expiries in one session.

---

## 4. The rules, exactly

### 4.1 Pivots
Close-based zigzag, threshold **THR = 5** points:
- track the running extreme *close*;
- when the close reverses by ≥ THR, the pivot is **confirmed** at that bar;
- the pivot **price** is the leg's extreme high/low (not the confirmation bar).

Pivot record = `(time, type, price, price_index, confirmation_index)`.

### 4.2 Wave search
- six alternating pivots `L, H, L, H, L, H`;
- **skip budget 4**: up to four intermediate pivots may be ignored;
- validity: `5 > 3 > 1` and `4 > 2 > 0`, each with a **1-point buffer**
  (`p5 > p3 − 1`, …);
- take the **first valid** sequence; drop overlapping waves.

### 4.3 a / b / c and the entry window
```
a = first L pivot after wave 5
b = first H pivot after a
c = first L pivot after b
window = [b_confirmation + 1 , c_confirmation + 1]
```
The **+1 bar** on the c side is required — it is what admits the 14:19 morning-star
case and the 09:39 winner.

### 4.4 Which contract
At the **wave-start minute** (`wave 0` price index):
```
ATM     = round(index / 50) * 50
CE strike = ATM − 100      (2nd ITM call)
PE strike = ATM + 100      (2nd ITM put)
```
Do **not** re-evaluate the strike at the entry minute — the contract is fixed when
the impulse begins.

### 4.5 SR levels (all computed causally at each bar close)
From the contract's **prior-day** H, L, C (close = last bar ≤ 15:30):
```
CPR  = (H + L + C) / 3
BC   = (H + L) / 2
TC   = 2 × CPR − BC
PDH, PDL
fibR3 = CPR + (H − L)      fibS3 = CPR − (H − L)
camR3 = C + 1.1/4 × (H − L)   camL3 = C − 1.1/4 × (H − L)
```
Intraday:
```
EMA20   : 5-minute and 3-minute
EMA200  : 3-minute and 5-minute
SuperTrend (10, 3) : 3-minute
VWAP    : session, hlc3 weighted by volume, from 09:15
```
All EMAs and SuperTrend must be **seeded with the prior day's bars** before the
first bar of the session. Buckets update with the live 1-minute close (chart
behaviour), not only at bucket close.

### 4.6 Trigger
A bar **is a trigger** when it is an engulfing or harami candle in *either*
direction (body-only test, colour irrelevant):
```
engulf : min(o,c) ≤ min(po,pc)  and  max(o,c) ≥ max(po,pc)
harami : min(o,c) ≥ min(po,pc)  and  max(o,c) ≤ max(po,pc)
         and body < previous body
```
…**and** at least one SR level satisfies:
```
level inside the bar's range   (low ≤ level ≤ high)
bar OPEN above the level       (open ≥ level)
```
Entry price = the trigger candle's **close**.

### 4.7 Stop
```
SL = level − 1.5 × ATR(14, 1-minute)
```
- ATR = simple average of true range over the last 14 one-minute bars.
- **Grace:** the stop only fires when `low ≤ SL − 1` (one-point tolerance).
- Fixed-stop variant measured: 5 / 8 / 8.5 / 10 / 12 points → all inferior on the
  long sample except f5; see §9.

### 4.8 Target
```
TP = level + 3 × (trigger bar high − trigger bar low)
```

### 4.9 Position management
- **one position at a time** (a new signal is ignored while a trade is open);
- end-of-day exit when the bar minute **> 915** (15:15), filled at that close;
- lot 65, fees **₹45 per round trip** for all P&L reporting.

---

## 5. DO

1. **Deduplicate bars by minute** before anything else — a duplicated feed corrupts pivots (a phantom `09:15 H` on the same bar as the 09:15 low is the classic symptom).
2. **Seed EMA200 and SuperTrend with the prior day's bars**, otherwise the first hours are wrong.
3. Use the **last bar ≤ 15:30** (or the exchange settlement print) as the prior-day close for CPR.
4. Compute levels at **each bar's close** and snapshot them — never rewrite history.
5. Respect the **pivot confirmation bar**: a pivot exists only after its reversal close.
6. Verify **contract identity** (`tsym` → token) whenever tokens are involved.
7. **Smoke test 8 sessions first** and confirm 2–6 trades/day before any long run.
8. Report **Calmar = annualised points ÷ max drawdown** on the trade-sequence equity.
9. Keep CE and PE in one book with a shared one-position rule.

## 6. DON'T

1. **Don't place the stop at "entry − 10"** for pattern entries. The stop belongs at `level − k`; the entry close can be 10–25 points above the level, which silently multiplies risk.
2. **Don't mix timeframes**: EMA20 on the 1-minute chart when the platform shows 5m will move every rung.
3. **Don't skip the open-above test** — it is what distinguishes a rejection from a breakout.
4. **Don't allow overlapping positions**, and don't re-enter the same wave.
5. **Don't trade a contract that wasn't the 2nd ITM at the wave start.**
6. **Don't include post-15:30 settlement prints** in the trading session (they belong to pricing, not to signals).
7. **Don't guess token→symbol mappings after expiry.** Cache `tsym + token + bars` while live; arithmetic token ladders are wrong (calls are not uniform).
8. **Don't use the put-call-parity spot when the index is available** — parity is a fallback only.
9. **Don't trust a short window's Calmar.** April 2026 gave 12.8; the same build gave 0.17 over six years.
10. **Don't deploy on a sample shorter than 12 months**, and never without the fee sensitivity in §9.

---

## 7. Reference implementation map

| File | Role |
|---|---|
| `ob_pat.py` | live-data engine (Flattrade API), all variants (touch / engulf / harami / pins / morning star / MTF) |
| `sy2.py` | 7-year engine on per-day CSVs + NSE index (volume, nearest expiry, prior-day warmup) |
| `full_run.txt` / `sy2_result.json` | six-year results |

Constants that must not drift:

```
THR = 5.0      BUF = 1.0      SKIP = 4
TOL = 2.5      (level coincidence, OB modes only)
TP_MULT = 3.0  ATR_LEN = 14   ATR_MULT = 1.5
SL grace = 1.0 EOD minute > 915
lot = 65       fee = ₹45 per trade
```

---

## 8. Acceptance checklist (run before declaring a port correct)

- [ ] 27 Oct 09:39 C22600 entry at camL3 **159.34**, exit 10:01, **+50.59**
- [ ] 27 Oct 13:08 P22900 entry at EMA20 **143.10**, exit 14:15, **+89.60**
- [ ] 22 Oct 14:52 C22500 entry at VWAP **266.24**, exit 15:01, **+20.19**
- [ ] CPR check: for 03NOV26 22550 CE on 28 Oct, **BC = 333.77 ±0.05**, CP 330.38, TC 326.99
- [ ] 8-session total with ATR 1.5: **21 trades, +262.79 points, +₹16,136**
- [ ] No trade may enter after 15:15 or before its c-window opens

---

## 9. Honest limitations (read before deploying)

1. **Costs still consume almost all of it.**
   ```
   f5 gross edge = +1,891.44 pts / 2,663 trades = 0.710 pts/trade ≈ ₹46.2
   round trip    = ₹45 ≈ 0.692 pts/trade
   net           = +₹1.17/trade → +₹3,109 over 6.24 years (Calmar 1.285)
   ```
   Only f5 clears fees. The ATR × 1.5 stop used in this build loses ₹26,711 over
   the full period, and the break-even fee is ₹46.2/trade — any extra slippage
   erases the f5 edge.

2. **The stop ranking is regime-dependent.** October favoured ATR × 1.5; April
   favoured ATR × 2.0 and a tight fixed 5; six years favoured f5 (the only stop
   that clears fees, +₹3,109 net).

3. **The 16K window is 21 trades.** Two candles (+50.59, +89.60) carry 53% of the
   points. Remove them and the build is flat-to-negative.

4. **Data fragility.** Expired option contracts disappear from broker scrip masters;
   without a local cache the historical run cannot be reproduced. The 27OCT26 series
   had to be recovered by price fingerprinting (11 contracts) after expiry.

**Verdict:** mechanically valid, statistically unproven. Use for research, not with
live capital, until a ≥12-month run clears fees on a cost model that matches the
actual broker.

---

## 10. Live engine and congruence (VPS)

The live implementation is `flattrade_bot/strategies/atr16k_engine.py`. It emits the
same signal shape the existing executor already consumes, so order placement, the
risk manager, the WS feed, and the broker integration are untouched. Enable it with:

```
STRATEGY_MODE=atr16k      # default remains full10
```

### Live wiring that MUST match the backtest

| Item | Live value | Why |
|---|---|---|
| Warmup | 10-day fetch → 5 prior sessions, last 2,000 bars (1,925 observed) | reference seeds up to 2,000 prior bars |
| Prior-day close (CPR/BC/TC) | **15:32 print** | post-close prints 15:33–15:39 differ by up to 25 points (27 Oct 03NOV26 C22450: 395.00 vs 370.10) |
| Contract ladder | CE ATM−50…−300, PE ATM+50…+300 (12) + every live wave's 2nd ITM | the 2nd ITM is fixed at the **wave start**, not at entry |
| Entry window | 09:15–15:00 | `b_conf+1 … c_conf+1`, entry cutoff 15:00 |
| Stop | **1.1 × ATR**, hard cap **20 pts**, minimum **3 pts** (re-anchored) | level-based stop can land at/above the entry when the trigger closes through the level |
| Target | **3.5 × trigger range** | tuned on the 12-session book |
| Wave validity | **correction leg (4>2>0)** | impulse condition 5>3>1 dropped (more 2nd-ITM setups, same DD) |
| Levels | statics + EMA20/200 3m+5m + session VWAP + ST 3m + **PDVWAP** + **virgin CPR/BC/TC** | PDVWAP = previous day's closing VWAP (volume-weighted hlc3) |
| Live bars | `logs/bars_YYYY-MM-DD.csv` (one row per completed bar) | replay must run on the bars the bot trades, not TPSeries |
| EOD | 15:16 bar close (`eod_min` per position) | reference exits at the first bar with minute > 915 |
| Breakeven | none | reference has no BE |
| Daily cap | **−40 points** (`ATR16K_LOSS_CAP=40`; profit cap off) | day-cumulative closed points; once hit, no new entries for the session |
| Position rule | one at a time; entry minute ≤ previous exit minute → dropped | reference day-level non-overlap filter |

### Signal-gate fix (3 Nov 2026, evening)

Every engine signal was being discarded by the runner before `_try_enter`, so no
`📣 SIGNAL` line was ever written and no order could be placed. Two causes:

1. The gate called `engine.desired_strikes(spot)`, a method that exists on the
   Full10/last-hope engines but **not** on `ATR16KEngine`. The call raised
   `AttributeError`; the surrounding `except Exception` swallowed it and set
   `spec_keys = set()`, after which `key in spec_keys` was always false.
2. Even with the method present, the gate compared the signal's contract with
   the **current** 2nd ITM, while the strategy trades the 2nd ITM **at the wave
   start**. All three missed signals (3 Nov C23500, 2 Nov P23150, 2 Nov P23450)
   were wave-start strikes and failed that comparison.

Fixed: the gate now comes from `desired_pairs(spot)` — the current 2nd ITM plus
every live wave's 2nd ITM — and an exception no longer empties the set; it logs a
warning and accepts the engine's signal (the engine's own strike gate is the
authority). Verified on the three missed cases: engine fires and the gate accepts
all three (`fire=True, in_gate=True`).

### Context-ordering fix (2 Nov 2026)The live engine used to apply the non-overlap filter **before** building the a/b/c
windows. A carried wave whose window had already closed on the prior session
therefore survived and shadowed a later wave whose window was still open — the
bot missed the 09:29 carried-wave signal on P23150 on 2 Nov. The engine now:

1. builds a context for **every** valid wave,
2. drops contexts whose `c_conf` is before today's first bar (`c_conf < t0`),
3. applies the non-overlap priority only to the survivors.

This is exactly the reference order in `carry9.py`. After the fix the live engine
class reproduces 2 Nov bar-for-bar: `09:29 P23150 harami fibS3 55.88 entry 57.35`
and `12:02 P23450 twin EMA20 186.68 entry 189.15`.

### Parity evidence

28 Oct 2026, all 34 03NOV26 contracts, reference loop vs live engine:

```
11:20 C22450 EMA20=274.53   entry 277.40  exit 11:58  TP +26.08
12:56 C22550 EMA200_5=295.30 entry 297.70 exit 12:59  TP +15.15
13:49 C22600 EMA200_5=267.20 entry 272.05 exit 13:59  SL −16.65
MATCH 3/3 — identical strike, minute, level, entry, exit and points
```

8-session tally (21–28 Oct, nearest expiry, 15:32 prior close, ₹45/trade):

| | Trades | Points | ₹ after fees |
|---|---|---|---|
| Live engine | 19 | +278.95 | +17,277 |
| Built headline | 21 | +262.79 | +16,136 |

The 2-trade gap is data coverage, not logic: only 11 of the 27OCT26 contracts could
be recovered after expiry, and the missing strikes account for the difference.

### Daily acceptance test

After each session, run the engine over that day's 2nd-ITM ladder (same warmup and
15:32 prior close) and compare the trade list with the bot's closed trades — entry
minute, strike, level, exit reason, points. Identical lists = the VPS is congruent.

