# Pocket Money Strategy — Official Rules

**Name:** Pocket Money Strategy (PM)
**Style:** NIFTY 50 options intraday scalping — long option premium only
**Official timeframe:** **10 seconds** (TradingView). The historical backtest runs on a 1-minute proxy (finest archived resolution); the live bot builds true 10-second bars from ~1-second quote polls.

---

## 1. Indicators

### 1.1 Option chart — 4 stochastics (on EVERY option chart)
| Stochastic | %K | %D |
|:---|:---:|:---:|
| S1 (fast) | 9 | 3 |
| S2 (medium) | 14 | 3 |
| S3 (slow) | 40 | 4 |
| S4 (trend) | 60 | 10 |

Stochastics are computed on the option's **own chart** (its own high/low/close), never on the index.

### 1.2 Index 15-minute filter (strike/side selection only)
Computed on NIFTY 50 spot 1-minute bars aggregated to 15-minute bars. The chart this filter replicates is a **Heikin-Ashi chart** (TradingView chart type 8) with UT Bot + Humble LinReg overlays, so:

- **Bars are clock-aligned to TradingView:** the 15m bar stamped 09:15 covers minutes 555..569, 09:30 covers 570..584, etc. The broker's flat 09:14 pre-open placeholder row is skipped (`minute < 555`), and each bucket is committed at the bucket boundary — never by row count. This removes a 1-minute shift that previously skewed every bar vs the chart.
- **Each 15m bar is Heikin-Ashi converted first** (HA state is continuous across days, exactly like the chart): `ha_close=(O+H+L+C)/4`, `ha_open=(prev_ha_open+prev_ha_close)/2` (first bar `(O+C)/2`), `ha_high=max(H,ha_open,ha_close)`, `ha_low=min(L,ha_open,ha_close)`.
- **UT Bot:** key = 1.0, ATR period = 10, source = **HA close** (green when HA close > trailing stop, red when HA close < trailing stop).
- **LinReg Candles:** `linreg(value, 11, 0)` applied per HA OHLC component; **white line = sma(linreg(ha_close,11), 11)**.
- Filter state for the last completed 15m bar:

| Condition | Allowed side |
|:---|:---|
| UT **green** AND 15m HA close **>** white line | **CE** only |
| UT **red** AND 15m HA close **<** white line | **PE** only |
| anything else (mixed / no signal) | **No entry** |

- Before today's first 15m bar completes, the **forming bar**'s live state drives the side (TradingView-style), so the filter answers from the open.

The index is used **only** for strike selection and this side filter — never for entry signals.

---

## 2. Entry Triggers (option chart, 10s bars)

### FLAG
S1 %D touches **20.5** from the neutral zone — S1 **crosses from > 20.5 down to ≤ 20.5** — **while S4 %D ≥ 79.5**. **No divergence requirement** — a FLAG fires on the plain setup.

### SUPER
S1 %D **crosses back above 20** (previous bar ≤ 20, current bar > 20) **AND** a **bullish trough divergence** is confirmed at that same bar. The crossing is what makes the current trough "fully formed" — the divergence is looked for exactly at this crossing, never before.

### Live evaluation (10s bars, bar-close semantics)
Stochastics are evaluated on the **completed 10s bar** — the moment the bar closes. The divergence engine is synchronous with the trigger: it is fed the same committed S1/S2 the trigger reads, so no intra-bar (forming-tick) evaluation is done and the per-tick low wobble cannot create spurious divergence troughs. This matches the backtest proxy, which evaluates on completed 1m bars. One signal per 10s bar epoch.

### Divergence gate (SUPER only)
SUPER additionally requires a confirmed **bullish trough divergence** on the option's own chart:

- A **trough** is formed when S1 %D turns up after a decline (the declining leg has bottomed). The trough is only **fully formed / confirmed once S1 %D crosses above 20** — that crossing is when the divergence is checked.
- **Bullish divergence:** the current confirmed trough's price is **lower** than the previous confirmed trough's price **and** the current trough's S1 **or** S2 is **higher** than the previous trough's S1/S2 (price makes a lower low while momentum makes a higher low — whichever of S1/S2 shows it counts).
- The divergence is assessed on **S1 OR S2** — if either shows the higher-low momentum, the divergence is valid. The side of the chart that shows the divergence decides the entry side (divergence on the CE chart → filter must say CE; divergence on the PE chart → filter must say PE).
- Implemented in `flattrade_bot/indicators/divergence.py` (`DivergenceEngine`, turn-up-trough mode + S1-crosses-20 confirmation, `divergence_confirmed_at_last_update()`); the legacy pivot-based bearish-peak mode is retained for reversal-exit research scripts only.

### Entry rules
1. Trigger side must equal the 15m filter side (FLAG fires on a CE chart → filter must say CE, etc.).
2. **Divergence required for SUPER only** — FLAG entries need no divergence.
3. Trade only the **2nd ITM strike**: **CE at ATM − 100, PE at ATM + 100**, where ATM = round(spot / 50) × 50. The spec strikes are watched with warm stochastics from the session open; the ATM ± 50 pair is tracked as **rollover watch** so the new spec pair is always warm when spot crosses a band.
4. **One position at a time** — entry only when flat.
5. Entry price = close of the completed 10s trigger bar (for SUPER, the bar on which S1 crosses above 20 with the divergence confirmed).
6. **No new trades after 15:00.**
7. Position held until SL / TP / EOD — no manual exits, no averaging.

---

## 3. Exits (10s bars)

| Exit | Rule | Priority |
|:---|:---|:---|
| **Stop loss** | LTP ≤ entry − **7.0** premium points | highest — if SL and TP touch in the same bar, SL wins |
| **Take profit** | LTP ≥ entry + **7.0** premium points | |
| **EOD** | Close at the **15:00** bar close at market | |

Both sides are LONG option positions (long CE premium / long PE premium) — premium rising is favorable for either, so SL = entry − 7 and TP = entry + 7 on **both** sides.

---

## 4. Session & Risk

- Trading window: **09:20 – 15:00 IST** (entries 09:20–14:59, EOD exit 15:00).
- **4 consecutive losses → trading blocked for the rest of the day.** No daily ₹ loss cap (matches backtest).
- One lot (65) per trade, MIS product.
- **Not used:** no pin bars, no ATR exits, no reversal mode, no hedging, no second position.

---

## 5. Live Implementation Mapping

| Concept | Backtest (1m proxy) | Live bot |
|:---|:---|:---|
| Option bars | 1m archive | **10s bars built from ~1s LTP polls** |
| Stochastic warmup | prior day's 1m rows (`fprev`) | prior day's 1m rows as seed (converges to true 10s values after ~12 min) |
| Index filter warmup | 12 prior days | 12 prior days + today's rows up to now (mid-session restarts reconstruct today's chain exactly; the API's flat future placeholder rows for today are dropped so the chain never runs ahead of the clock) |
| Trigger check | on each completed 1m bar | **each completed 10s bar** (bar-close semantics, matching the backtest; the forming bar is peeked only for the setup monitor) |
| Tracked strikes | full window | spec pair (ATM±100) + rollover watch (ATM±50) = 4 contracts |
| SL / TP | bar H/L intrabar | polled LTP, immediate market exit |

## 6. Notifications (Discord)

- Trade **open** embed: side, strike, entry, SL / TP, trigger (FLAG / SUPER), filter side.
- Trade **close** embed: reason (SL / TP / EOD), points, ₹ P&L, duration.
- **EOD summary**: trades, win rate, net ₹ P&L, consecutive losses.
- No setup-ping spam — alerts fire only on actual events.