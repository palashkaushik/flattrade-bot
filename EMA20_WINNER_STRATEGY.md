# EMA20 WINNER STRATEGY — §44 Dynamic-Strike Champion (LIVE on VPS)

> **This is the single source of truth for the strategy currently trading on the
> AWS Lightsail VPS.** Any agent (AI or human) must be able to rebuild the live
> bot from this file alone and be certain it takes the EXACT trades the
> backtest takes. Congruence is not aspirational — it is verified
> (trade-level causal parity, 47 tests, three-way static == replay == dyn
> agreement on 2025-09-08).

---

## 1. IDENTITY

| | |
|:---|:---|
| **Name** | Last Hope Winner — Dynamic-Strike EMA20 Gate (§44 Champion) |
| **Ledger ref** | `BACKTEST_LEDGER.md` → §44 (dynamic-strike dual-gate sweep) |
| **Instrument** | Nifty 50 options, 2nd-ITM weekly (CE = ATM−100, PE = ATM+100) |
| **ATM anchor** | **DYNAMIC: current index spot at each trade time, rounded to 50** (NOT pinned at 09:15) |
| **Weekly expiry** | Nearest expiry ≥ today (Tuesday weeklies post-2025-09-01; Thursday before) |
| **Bars** | Official 1-minute option OHLC, 09:15–15:00 IST (345 bars) |
| **Position size** | 1 lot = 65 qty (Nifty), fee ₹45 round-trip |
| **Backtest 7y** | Net **+₹3,623,562** · WR **90.2%** · trades **16,491** · maxDD **₹1,504** · worst day **−₹1,476** · Calmar **2,408.7** |
| **Engine fix** | `be_done` reset on entry (BE now fires on EVERY trade, not once/day) — all prior §41-§43 numbers were run pre-fix |

### §44 vs §43 (corrected, post-fix)

| | §43 (static-strike, pre-fix) | §43 (static, post-fix) | **§44 (dyn, post-fix)** |
|:---|---:|---:|---:|
| Net | ₹2,832,706 | ₹2,799,385 | **₹3,623,562** |
| Win rate | 78.5% | 84.4% | **90.2%** |
| MaxDD | ₹1,963 | ₹1,996 | **₹1,504** |
| Calmar | 1,443 | 1,401 | **2,409** |
| Trades | 19,701 | 20,347 | **16,491** |

## 2. THE STRATEGY (exact rules)

### 2.1 Contract selection (DYNAMIC — the §44 core change)
1. At **every signal evaluation**, read current index spot →
   `ATM = round(spot / 50) * 50`
2. Trade the **2nd-ITM** pair at that moment: `CE = ATM − 100`, `PE = ATM + 100`
3. As the index moves, the selected strike **moves with it** (±50 rollover
   watch pairs resolve the new strikes before they're needed — execution
   plumbing, not a strategy rule)
4. Contract = current weekly expiry. On the expiry Tuesday itself, trade the
   **expiring** weekly (do NOT jump to next week).

### 2.2 Indicators (on the OPTION's own 1m chart, seeded)
- **EMA20** — incremental EMA, alpha = 2/21
- **ATR(10)** — EMA-smoothed true range, alpha = 2/11
- **Multi-TF stochastics** on 1m/2m/3m/5m bars, clock-aligned buckets:
  - S1 (%K=12, %D=3) fast · S3 (%K=40, %D=4) slow · S4 (%K=50, %D=10) macro
- **SEEDED warmup (critical)**: before 09:15, replay the prior day's **last
  300 one-minute bars** through EMA/ATR/TF-trackers — **of the SAME token**
  (prior-day contract of the same strike). ATR is never cold.
  VWAP resets at the day boundary (session-only).

### 2.3 Arming
- A completed 1m bar with **S1(1m) ≤ 25.0** arms both FLAG and SUPER setups
- Arming expires after **ARM_WINDOW = 10 bars**; arming only while flat
- Arming resets on every position open/close and every new day (day-cold)
- **Per-side arming**: PE and CE arm/disarm independently (a CE trade never
  clears the PE arming state and vice versa)

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
  dilutive in paired sweeps on BOTH static and dynamic engines.

### 2.6 Entry
- Signal bar close = entry price (market order at next tick)
- Entry window: 09:15–15:00, no entries while any position is open
- **Strike re-selected at entry time** from current spot (see 2.1)

### 2.7 Risk geometry (per trade)
```
dist = clamp(ATR(10) × 1.5, floor 2.0, cap 15.0)      [§44: multiplier 1.5]
SL   = entry − dist
TP   = entry + dist
BE trigger: when LTP/high ≥ entry + 0.40 × dist  →  SL hardens to entry + 1.0 (never retreats)
BE ratchet resets on EVERY new entry (be_done = false at entry)
SL priority over TP on the same bar/tick
```
- BE geometry re-bases on the ACTUAL fill price (not the signal close)
- Exits at market (`force_mkt=True`) — limit sells get rejected on fast moves

### 2.8 Filters — ALL OFF (do not turn on)
Bias (Marni-Fib 15m): **OFF** (halves profit) · Elder: OFF ·
RSI: OFF · Reversal: OFF · ST-Zone: OFF · Entry-time windows: OFF (all day)

---

## 3. LIVE WIRING (flattrade_bot) — the congruent implementation

| File | Role |
|:---|:---|
| `flattrade_bot/strategies/last_hope_winner.py` | Strategy engine (§44 constants; `sr_levels={}` gate, EMA20 injected live; `display_levels` = CPR/Cam/Fib display-only) |
| `flattrade_bot/last_hope_main.py` | Live orchestrator: dynamic strike re-resolution each tick (`desired_strikes(spot)`), 300-bar seeded warmup, WS-first ticks, day rollover, force_mkt exits, funds-fallback 2nd→1st ITM |
| `flattrade_bot/execution.py` | `check_exit(dry_run)` SL/BE/TP logic, MKT exits |
| `flattrade_bot/broker/history.py` | `search_option_token` (nearest weekly ≥ today), TPSeries warmup bars |

**Congruence contract (live MUST equal backtest):**
1. TF buckets close only on clock boundaries: `(minute + 1 − 555) % tf == 0`
2. Seeded warmup: prior day's final 300 bars of the SAME token BEFORE today's first signal
3. EMA20/ATR update on every COMPLETED 1m bar, before the gate check
4. BE trigger uses fill price, ratio 0.40, buffer +1.0, one-way ratchet,
   **resets on every entry**
5. Arming: day-cold, position-open-close-cold, 10-bar expiry, per-side
6. Same-bar SL/TP: SL wins
7. EOD: nothing survives 15:00; next day re-seeds fresh
8. **Strike = dynamic**: at each entry, 2nd-ITM from current spot

### DO
- Run the 47-test suite after ANY change: `python -m pytest tests/test_last_hope_winner_suite.py tests/test_last_hope_comprehensive_v2.py`
- Keep exits MKT (`force_mkt=True`)
- Keep the seed at exactly 300 prior-day bars (same token)
- Round ATR dist to 2 decimals; clamp [2.0, 15.0]
- If 2nd-ITM margin-rejected → fallback 1st-ITM (CE ATM−50 / PE ATM+50) for that buy only

### DON'T
- **Never** pin strikes at 09:15 (that is the §43 engine — ₹800K worse)
- **Never** add any static level to the trading gate (CPR/Cam/PDHL/Fib/PrevVWAP/VirginCPR/EMA200/VWAP)
- **Never** turn on Bias/Elder/RSI/Reversal/ST-Zone
- **Never** deploy the §42 config (ARM_WINDOW=15/ATR×1.5/BE0.50 + 10-level gate — the fragile-peak, −20% neighbor drop)
- Never let arming survive a position cycle or a day rollover
- Never exit with limit orders on SL/TP/EOD
- Never warm up with fewer than 300 prior-day bars (cold ATR → wrong SL/TP)
- Never trade a non-weekly (monthly) contract

### §44 plateau evidence (robustness)
The entire x1.5/be0.4 column is stable across ALL arm values (Calmar
2,191–2,409; net ₹3.48M–₹3.65M): arm5 2,312 · arm10 2,403 · arm15 2,198 ·
arm20 2,191. The champion is NOT a fragile peak. x2.0 configs net more
(₹3.83M) but at 52–169% worse DD (Calmar ≤1,695) — rejected on risk-adjusted
grounds. FULL10 gate loses to EMA20-only everywhere on dynamic strikes
(best ₹2.74M vs ₹3.84M).

---

## 4. VPS OPERATIONS (AWS Lightsail, ip-172-26-8-101)

- Dashboard: `tmux attach -t bot` (Rich live screen)
- Logs: `journalctl -u flattrade-bot -f`
- Discord: `/trading start|stop|restart|status|logs`
- Timers: start 09:05 / stop 15:15 IST Mon-Fri (systemd)
- Deploy: push to `release` branch → `git pull` on VPS → `sudo systemctl restart flattrade-bot`

## 5. ROLLBACK — restore the previous (§43) strategy anytime

The previous champion is preserved in git history. To switch back:

```bash
cd /home/ubuntu/FLATTRADE_BOT
git log --oneline -- flattrade_bot/strategies/last_hope_winner.py   # find the last §43 commit (f44c16a)
git checkout f44c16a -- flattrade_bot/strategies/last_hope_winner.py flattrade_bot/last_hope_main.py
python -m pytest tests/ -q            # expect the §43-era tests to pass; §44 tests will fail (expected)
sudo systemctl restart flattrade-bot
```
§43 constants for reference: `ARM_WINDOW=10, ATR_MULT=1.0, BE_TRIGGER_RATIO=0.60, tb=0.0`,
static 09:15 strikes, EMA20-only gate, seeded warmup, 7y net +₹2.80M (post-fix), WR 84.4%.
**Note:** §43 is strictly dominated by §44 (−₹824K net, worse WR, worse DD) —
rollback only if §44 live-vs-backtest diverges, not for P&L reasons.

## 6. VERIFICATION ARTIFACTS (regression anchors)

| Artifact | Expected |
|:---|:---|
| `dyn_sweep_results.csv` | 256 configs, dual-gate, corrected engine — §44 source |
| `validate_fix.py` output | 2025-09-08: 20 trades trade-for-trade vs replay/dyn |
| `dyn_strike_engine.py` | Dynamic-strike reference engine (TV-parity seeding) |
| `parity_dyn2.py` / `bisect_parity.py` | Trade + mask-level causal parity |
| April 2026 reference | §43-era artifact — superseded by §44 (numbers will differ; do not regression-test against it) |

## 7. CHANGE LOG

| § | Change | Date |
|:---|:---|:---|
| 41 | Seeded indicators, max-net arm15/x1.5/tb0.0/be0.5 | Aug 2026 |
| 42 | WF-validated champion deployed to VPS (arm15/x1.5/be0.5) | Aug 2026 |
| 43 | EMA20-only gate, plateau champion (arm10/x1.0/be0.6), Pareto+parity verified, deployed | Sep 2026 |
| **44** | **Dynamic strikes (2nd-ITM at trade time) + be_done engine fix + x1.5/be0.4 — net ₹3.62M, Calmar 2,409, plateau-stable, deployed** | **Sep 2026** |
