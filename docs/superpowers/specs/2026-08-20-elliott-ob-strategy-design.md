# Elliott OB Strategy — Design Spec

**Date:** 2026-08-20
**Status:** Approved design
**Author:** User + opencode

## Summary

A new intraday NIFTY options strategy combining a lenient 5-wave Elliott impulse count
(1-minute bars only) with multi-timeframe single-candle order blocks (1m/2m/3m/5m).
Entries fire on the 1m chart when price pulls back into an unmitigated order block after
the correction's B wave. Direction (CE vs PE) is determined **only** by the impulse
direction. SL = 3×ATR(10) of the index; TP = +60 index points. P&L settles on the
2nd-ITM nearest-weekly option.

This replaces the F6 Champion + Marny family, which was proven structurally unprofitable
(WR ~33–37% vs 44.4% breakeven) across the full 7y dataset.

## Signal Pipeline (all on 1m)

### 1. Impulse count (lenient 5-wave)

Five consecutive 3-candle "waves" with alternating color patterns:

- Wave 1: red, green, red
- Wave 2: green, red, green
- Wave 3: red, green, red
- Wave 4: green, red, green
- Wave 5: red, green, red

Each wave = 3 consecutive 1m candles. **Only two validation conditions:**

- **Condition 1 (bullish):** wave-5 peak > wave-1 peak AND wave-5 peak > wave-3 peak;
  the impulse origin (low before wave 1) < wave-2 low AND < wave-4 low.
- **Condition 1 (bearish, mirrored):** wave-5 trough < wave-1 trough AND < wave-3 trough;
  the impulse origin (high before wave 1) > wave-2 high AND > wave-4 high.

"Peak" of a wave = max(high of its 3 candles). "Trough" = min(low of its 3 candles).

**Session continuity:** the impulse count does NOT reset at session boundaries. A
5-wave impulse that began forming in a prior session continues counting into the current
session, provided the alternating 3-candle pattern and the two conditions are still met.
This means an impulse whose origin was yesterday (or earlier) can complete today and arm
a setup for today's entry. The full 1m history (not just the current session) is used for
wave detection; only entries are restricted to the current session's trading window.

### 2. Order blocks (per TF: 1m, 2m, 3m, 5m)

Either single-candle pattern forms an order block (zone = candle X's range):

- **Pattern A:** candle X breaks the prior candle's LOW; the next candle (X+1) breaks
  candle X's HIGH.
- **Pattern B:** candle X breaks the prior candle's HIGH; the next candle (X+1) breaks
  candle X's LOW.

The pattern itself carries **no direction**. Bullish/bearish comes only from the impulse.

**Unmitigated definition:** the OB is valid only if price has never traded *through* the
full OB zone since formation (a touch/retest inside the zone is allowed). Only the
untouched portion of a partially-mitigated zone is reusable.

### 3. Correction (ABC)

After wave 5 completes, the alternating 3-candle pattern continues:

- A = next wave after W5 (green/red/green for bullish)
- B = the wave after A
- After **B completes**, the setup is armed.

The C wave is ignored — it can take any shape; no condition is applied to it.

### 4. Entry

First 1m bar (after B completes) whose range **pulls back into** an unmitigated OB zone
(from the impulse side — price returns toward the level) → **buy at that bar's close.**

- Near-edge tolerance: a low within ~0.5 pt of the OB's far edge counts as a touch
  (verified against Aug 20 anchors: 24,200.1 vs OB top 24,199.97 → valid touch).
- **Direction:** up-impulse → buy CE; down-impulse → buy PE. The option is the 2nd-ITM
  nearest-weekly strike for that direction.
- Entry price = the 1m bar's close (index level); option premium from the matching
  2nd-ITM option bar.

### 5. Exit (index-based)

- **SL:** entry − 3 × ATR(10) of the index at the entry bar close (bullish); entry +
  3 × ATR(10) (bearish).
- **TP:** entry + 60 index points (bullish); entry − 60 index points (bearish).
- Whichever is hit first on subsequent 1m bars; premium P&L read from the option bars at
  the exit bar.

## Trading Rules

- Window: full session 09:15–15:30 IST, every trading day.
- Impulse waves may span sessions (prior-session impulse continuation is valid); entries
  only fire in the current session.
- Multiple setups per day — every valid armed setup is traded.
- No daily loss cap.
- One use per OB portion: once a trade resolves, that portion is consumed.

## Data

- Reuse existing 1m data loaders (`opt_futures_quad.py`) + Desktop/ammu Aug extension
  pattern already used in `f6_mtf_7y_runner.py`.
- Full range: 2020-01-01 → 2026-08-20 (1,588 days, includes Aug extension).
- Option P&L on 2nd-ITM nearest-weekly CE/PE (existing option dataset).

## Testing

1. **Smoke test first** (MANDATORY per AGENTS.md): run Aug 19–20 only.
   - **Congruency anchors (Aug 20, 2026):** trades must fire in the ~11:15–11:18 window
     (OB1 = 09:53 purple box, zone 24,196.00–24,199.97, touch bar L=24,200.1) and the
     ~12:24–12:25 window (OB2 = 11:40–11:42 blue box, zone 24,208.71–24,213.24, touch
     bar L=24,209.95). 1–10 trades/day; WR 35–50% sanity; sign check.
   - 5-day reference: 15–40 trades expected on the first 5 days of 2020.
2. **Full run:** all 1,588 days. Report trades, WR, net ₹, PF, MaxDD per variant.
3. Sweep tolerance (0–1 pt) and ATR multiplier if needed.

## Open Questions (resolved)

- TF scope: waves counted on 1m only; OBs from 1m/2m/3m/5m. ✅
- Touch = pullback into OB from impulse side. ✅
- Both 11:15 and 11:18 are valid entries (low-touch rule fires at first qualifying bar).
  ✅
- OB direction: none intrinsic; impulse determines CE/PE. ✅
- Impulse continuation across sessions is valid; entries only in the current session. ✅
- SL/TP index-based; option only for P&L settlement. ✅
- Full session; multiple setups; no daily cap. ✅