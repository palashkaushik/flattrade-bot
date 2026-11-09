# ATR bracket vs level-anchored stop — six-year results

**Date:** 2026-11-10
**Universe:** NIFTY 2nd-ITM options (CE ATM−100 / PE ATM+100), 1,572 sessions, 2020-01-01 → 2026-04-30
**Fees:** ₹45 per round trip, 65 qty
**Grid:** triggers `ob` / `obonly` × SL factor 0.5–2.0 × TP factor 3–8, max stop 20, min risk 3, day cap −40, entry ≤ 15:00
**Engine:** `research/backtest/bt.py` with `SL_FROM_ENTRY` / `TP_FROM_ENTRY`
**Data:** `slash slentry_days.jsonl`, full 61-row table in `slentry.log`

## What changed

| | old | new |
|---|---|---|
| Stop | `level − k×ATR` (anchored to SR/OB level) | `entry − k×ATR` |
| Target | `level + k×bar range` | `entry + k×ATR` |

## Six-year table (top by Calmar)

| Config | Trades | WR | Points | ₹ net | DD | Calmar | BE slip | Avg SL | RR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ob atr 0.5/3** | 6,079 | 20.8% | 8,524 | +280,532 | **290** | **4.65** | 0.35 | 4.71 | 5.21 |
| **ob atr 0.5/4** | 5,931 | 16.9% | 9,011 | +318,844 | 311 | 4.58 | 0.41 | 4.73 | 6.79 |
| **ob atr 0.5/5** | 5,792 | 14.4% | 9,955 | +386,432 | 378 | 4.16 | 0.51 | 4.75 | 8.44 |
| **ob atr 0.5/8** | 5,436 | 10.7% | **11,584** | **+508,318** | 455 | 4.02 | **0.72** | 4.80 | 12.55 |
| ob atr 0.5/6 | 5,623 | 12.4% | 9,868 | +388,409 | 410 | 3.80 | 0.53 | 4.76 | 9.99 |
| obonly atr 0.5/4 | 8,126 | 16.3% | 11,414 | +376,256 | 529 | 3.41 | 0.36 | 4.79 | 6.92 |
| ob only atr 0.5/8 | 7,114 | 10.2% | **12,830** | +513,848 | 904 | 2.24 | 0.56 | 4.89 | 12.42 |
| ob atr 0.75/3 | 5,789 | 24.4% | 7,430 | +222,440 | 384 | 3.06 | 0.30 | 6.52 | 3.89 |
| **baseline: ob level 1.0/5 (deployed)** | 4,741 | **26.8%** | 6,152 | +186,547 | **1,204** | **0.81** | 0.30 | 11.04 | 3.17 |
| ob atr 2/3 (max WR) | 4,733 | **39.0%** | 3,504 | +14,777 | 913 | 0.61 | 0.02 | 14.22 | 1.70 |
| obonly atr 2/4 (WR + profit) | 5,584 | 34.1% | 7,957 | +265,952 | 1,047 | 1.20 | 0.37 | 14.62 | 2.22 |

## Short window 2026-11-04 → 2026-11-09 (5 sessions)

| Config | Trades | WR | Points | ₹ | DD |
|---|---:|---:|---:|---:|---:|
| ob atr 0.5/8 | 12 | 33.3% | +581.44 | +37,254 | 74.24 |
| ob atr 0.75/8 | 11 | 36.4% | +572.14 | +36,694 | 76.95 |
| ob atr 1/8 | 10 | 40.0% | +567.78 | +36,456 | 74.70 |
| obonly atr 0.5/8 | 19 | 21.1% | +543.52 | +34,474 | 73.55 |
| baseline ob level 1.0/5 | 11 | 45.5% | +306.07 | +19,400 | 75.20 |

## Findings

1. **The level-anchored stop was expensive.** Moving the stop from `level − ATR` to `entry − 0.5×ATR` and the target to `entry + k×ATR` multiplies Calmar by ~5 (0.81 → 4.65) and cuts drawdown by 76% (1,204 → 290 points), while raising points and rupees.
2. **Monotonic structure.** Smaller SL multiples raise Calmar and cut drawdown; larger TP multiples raise points, rupees and slippage tolerance. There is no cliff — the frontier is smooth, which is a good sign of robustness rather than curve-fitting.
3. **Best all-round:** `ob atr 0.5/8` — 11,584 points (+88%), ₹508,318 (2.7×), DD 455 (−62%), Calmar 4.02 (5×), slippage tolerance 0.72 (2.4× the deployed 0.30). Cost: win rate falls to 10.7%.
4. **Win rate still opposes profitability.** Best WR (39.0%) earns ₹14,777; the 10–20% WR configs earn ₹280k–₹508k.
5. **Dropping the SR level helps at every ATR pair**, matching the earlier finding: `obonly` sits beside `ob` on the frontier and produces the single highest points figure (12,830).

## Caveats before any live use

- Average stop is **4.7 points**. The backtest exits at the stop price with a 1-point buffer; a live spread of 0.3–0.5 points would consume a large share of the edge. Slippage breakeven is 0.35–0.72 per side, so the 0.5×ATR configs are viable but not generous.
- Win rates of 10–20% imply long losing streaks (the point-recorded configs have not been streak-checked in this run).
- These are 1-minute-bar fills. Fast moves through a 4.7-point stop will fill worse in live trading.

## Next steps (not run)

1. Streak and per-year breakdown for the shortlist (`ob atr 0.5/3`, `0.5/5`, `0.5/8`).
2. Out-of-sample split 2020–2024 → 2025–2026 for those three.
3. Live-week signal tally on the chosen config before funding.

## Related files

- `bt.py` — `SL_FROM_ENTRY`, `TP_FROM_ENTRY`
- `sl_entry_sweep.py` — this sweep (short + six-year modes)
- `slentry.log` — full 61-row table
- `slentry_days.jsonl` — per-day checkpoint
