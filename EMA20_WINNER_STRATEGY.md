# EMA20 WINNER STRATEGY — §43 Plateau Champion (LIVE on VPS)

> **This is the single source of truth for the strategy currently trading on the
> AWS Lightsail VPS.** Any agent (AI or human) must be able to rebuild the live
> bot from this file alone and be certain it takes the EXACT trades the
> backtest takes. Congruence is not aspirational — it is verified
> (trade-level causal parity, 43+ tests, 9/9 exact-trade days vs the reference
> engine).

---

## 1. IDENTITY

| | |
|:---|:---|
| **Name** | Last Hope Winner — EMA20 Gate (Plateau Champion) |
| **Ledger ref** | `BACKTEST_LEDGER.md` → §43 (Pareto/pairs sweeps) |
| **Instrument** | Nifty 50 options, 2nd-ITM weekly (CE = ATM−100, PE = ATM+100) |
| **ATM anchor** | Index spot at 09:15 open, rounded to 50 |
| **Weekly expiry** | Nearest expiry ≥ today (Tuesday weeklies post-2025-09-01; Thursday before) |
| **Bars** | Official 1-minute option OHLC, 09:15–15:00 IST (345 bars) |
| **Position size** | 1 lot = 65 qty (Nifty), fee ₹45 round-trip |
| **Backtest 7y** | Net **+₹2,832,706** · WR **78.5%** · trades **19,701** · maxDD **₹1,963** · worst day **−₹1,963** · Calmar **1,443** |
| **April 2026 sample** | 248 trades, 201W/47L (81.0%), **+₹53,635**, 16/16 winning days (`april_2026_trades.csv`) |

## 2. THE STRATEGY (exact rules)

### 2.1 Contract selection
1. At 09:15, read index spot open → `ATM = round(spot / 50) * 50`
2. Trade the **2nd-ITM** pair: `CE strike = ATM − 100`, `PE strike = ATM + 100`
   (fixed for the day; rollover watch on ±50 strikes is execution plumbing only)
3. Contract = current weekly expiry. On the expiry Tuesday itself, trade the
   **expiring** weekly (do NOT jump to next week).

### 2.2 Indicators (on the OPTION's own 1m chart, seeded)
- **EMA20** — incremental EMA, alpha = 2/21
- **ATR(10)** — EMA-smoothed true range, alpha = 2/11
- **Multi-TF stochastics** on 1m/2m/3m/5m bars, clock-aligned buckets:
  - S1 (%K=12, %D=3) fast · S3 (%K=40, %D=4) slow · S4 (%K=50, %D=10) macro
- **SEEDED warmup (critical)**: before 09:15, replay the prior day's **last
  300 one-minute bars** through EMA/ATR/TF-trackers. ATR is never cold.
  VWAP resets at the day boundary (session-only).

### 2.3 Arming
- A completed 1m bar with **S1(1m) ≤ 25.0** arms both FLAG and SUPER setups
- Arming expires after **ARM_WINDOW = 10 bars**; arming only while flat
- Arming resets on every position open/close and every new day (day-cold)

### 2.4 Triggers (evaluated on every completed 1m bar, any TF)
- **FLAG (M6)**: on a TF whose `S4 ≥ 79.5` and `S1 < 79.5`
- **SUPER**: on a TF where `S3 < 25` AND `S4 < 25` AND `S1 < 25` AND S1 rising
  (S1 > previous TF-completed S1)
- Trigger valid only while armed (within 10 bars of the arming bar)

### 2.5 THE GATE — EMA20 bounce/rejection (the only gate)
Entry is allowed **only** when the option price rejects the option's own EMA20:

```
TOUCH:   bar.low  <= EMA20 + 0.0        (touch_buffer = 0.0, strict)
RECLAIM: bar.close >= EMA20 − 0.5
```
- The 1m bar may satisfy it directly, OR any completed 2m/3m/5m bucket
  (its low/close vs the same EMA20 value at that bar)
- **EMA20 is the ONLY trading level.** CPR, Camarilla, PDH/PDL, Fibonacci,
  PrevVWAP, VirginCPR, EMA200, VWAP are DISPLAY-ONLY (dashboard /
  TradingView verification). Never let them gate a trade — verified strictly
  dilutive (−₹105K…−₹349K each in paired sweeps; more levels = more entries
  with worse win rate).

### 2.6 Entry
- Signal bar close = entry price (market order at next tick)
- Entry window: 09:15–15:00, no entries while any position is open

### 2.7 Risk geometry (per trade)
```
dist = clamp(ATR(10) × 1.0, floor 2.0, cap 15.0)      [§43: multiplier 1.0]
SL   = entry − dist
TP   = entry + dist
BE trigger: when LTP/high ≥ entry + 0.60 × dist  →  SL hardens to entry + 1.0 (never retreats)
SL priority over TP on the same bar/tick
```
- BE geometry re-bases on the ACTUAL fill price (not the signal close)
- Exits at market (`force_mkt=True`) — limit sells get rejected on fast moves

### 2.8 Filters — ALL OFF (do not turn on)
Bias (Marni-Fib 15m): **OFF** (halves profit, ₹2.40M→₹1.07M) · Elder: OFF ·
RSI: OFF · Reversal: OFF · ST-Zone: OFF · Entry-time windows: OFF (all day)

---

## 3. LIVE WIRING (flattrade_bot) — the congruent implementation

| File | Role |
|:---|:---|
| `flattrade_bot/strategies/last_hope_winner.py` | Strategy engine (constants above; `sr_levels={}` gate, EMA20 injected live; `display_levels` = CPR/Cam/Fib display-only) |
| `flattrade_bot/last_hope_main.py` | Live orchestrator: 300-bar seeded warmup, WS-first ticks, day rollover, force_mkt exits, funds-fallback 2nd→1st ITM |
| `flattrade_bot/execution.py` | `check_exit(dry_run)` SL/BE/TP logic, MKT exits |
| `flattrade_bot/broker/history.py` | `search_option_token` (nearest weekly ≥ today), TPSeries warmup bars |

**Congruence contract (live MUST equal backtest):**
1. TF buckets close only on clock boundaries: `(minute + 1 − 555) % tf == 0`
2. Seeded warmup: prior day's final 300 bars BEFORE today's first signal
3. EMA20/ATR update on every COMPLETED 1m bar, before the gate check
4. BE trigger uses fill price, ratio 0.60, buffer +1.0, one-way ratchet
5. Arming: day-cold, position-open-close-cold, 10-bar expiry
6. Same-bar SL/TP: SL wins
7. EOD: nothing survives 15:00; next day re-seeds fresh

### DO
- Run the 47-test suite after ANY change: `python -m pytest tests/test_last_hope_winner_suite.py tests/test_last_hope_comprehensive_v2.py`
- Verify a fresh deployment with a known day (e.g. 2026-04-01: expect 17 trades, +₹4,783 per `april_2026_trades.csv`)
- Keep exits MKT (`force_mkt=True`)
- Keep the seed at exactly 300 prior-day bars
- Round ATR dist to 2 decimals; clamp [2.0, 15.0]
- If 2nd-ITM margin-rejected → fallback 1st-ITM (CE ATM−50 / PE ATM+50) for that buy only

### DON'T
- **Never** add any static level to the trading gate (CPR/Cam/PDHL/Fib/PrevVWAP/VirginCPR/EMA200/VWAP)
- **Never** turn on Bias/Elder/RSI/Reversal/ST-Zone
- **Never** change ATR_MULT to 1.5 / BE to 0.50 / ARM_WINDOW to 15 (that is the OLD fragile-peak §42 config — isolated −20% neighbor drop; do not deploy)
- Never let arming survive a position cycle or a day rollover
- Never exit with limit orders on SL/TP/EOD
- Never warm up with fewer than 300 prior-day bars (cold ATR → wrong SL/TP)
- Never trade a non-weekly (monthly) contract

---

## 4. VPS OPERATIONS (AWS Lightsail, ip-172-26-8-101)

- Dashboard: `tmux attach -t bot` (Rich live screen)
- Logs: `journalctl -u flattrade-bot -f`
- Discord: `/trading start|stop|restart|status|logs`
- Timers: start 09:05 / stop 15:15 IST Mon-Fri (systemd)
- Deploy: push to `release` branch → `git pull` on VPS → `sudo systemctl restart flattrade-bot`

## 5. ROLLBACK — restore the previous (§42) strategy anytime

The previous champion is preserved in git history. To switch back:

```bash
cd /home/ubuntu/FLATTRADE_BOT
git log --oneline -- flattrade_bot/strategies/last_hope_winner.py   # find the last §42 commit
git checkout <§42-commit> -- flattrade_bot/strategies/last_hope_winner.py flattrade_bot/last_hope_main.py
python -m pytest tests/ -q            # expect the §42-era tests to pass; §43 tests will fail (expected)
sudo systemctl restart flattrade-bot
```
§42 constants for reference: `ARM_WINDOW=15, ATR_MULT=1.5, BE_TRIGGER_RATIO=0.50, tb=0.0`,
gate = 10-level suite (CPR BC/Pivot/TC, Cam H3/L3, PDH, PDL, EMA20, EMA200, VWAP),
seeded warmup, 7y net +₹2.07M WF-OOS / ₹3.54M in-sample-peak, WR 75.9%.
**Note:** §42 is the fragile-peak config (−20% neighbor drop) — rollback only if
§43 live-vs-backtest diverges, not for P&L reasons alone.

## 6. VERIFICATION ARTIFACTS (regression anchors)

| Artifact | Expected |
|:---|:---|
| `april_2026_trades.csv` | 248 trades, 201W/47L, +₹53,635.49, no losing day |
| `aug_18_20_trades.csv` | (Aug 18-20 2026 unavailable in canonical data — days missing from parquet; do not treat 0-trades as a regression) |
| `parity_metrics_only.py` | PARITY OK (net/trades/WR/maxDD bit-identical) |
| `pairs_parity_report.txt` | ALL PASS (trade-level causal parity, 10 rows/6 gates) |
| Sweep sources | `gpu_sweep_final_pareto.py`, `gpu_sweep_pairs_v2.py`, `gpu_sweep_sr_expanded.py` |

**April 2026 daily reference (live P&L sanity check):**

| Day | Net ₹ | Day | Net ₹ |
|:---|---:|:---|---:|
| 04-01 | +4,782.78 | 04-21 | +1,186.25 |
| 04-02 | +5,051.67 | 04-22 | +3,840.10 |
| 04-06 | +3,555.34 | 04-23 | +1,790.28 |
| 04-07 | +4,035.38 | 04-24 | +2,677.48 |
| 04-15 | +1,678.44 | 04-27 | +3,361.52 |
| 04-16 | +4,512.93 | 04-28 | +2,448.42 |
| 04-17 | +2,631.95 | 04-29 | +1,899.36 |
| 04-20 | +5,161.86 | 04-30 | +5,021.72 |

## 7. CHANGE LOG

| § | Change | Date |
|:---|:---|:---|
| 41 | Seeded indicators, max-net arm15/x1.5/tb0.0/be0.5 | Aug 2026 |
| 42 | WF-validated champion deployed to VPS (arm15/x1.5/be0.5) | Aug 2026 |
| **43** | **EMA20-only gate, plateau champion (arm10/x1.0/be0.6), Pareto+parity verified, deployed** | **Sep 2026** |
