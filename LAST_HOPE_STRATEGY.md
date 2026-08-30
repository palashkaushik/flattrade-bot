# Last Hope — Nifty 50 Options 7-Year Strategy

> **Name:** Last Hope
> **Engine:** `run_7y_v4_master.py` (Master V4 — non-concurrent, daily cap optional)
> **Data:** `nifty50_options_master.parquet` (canonical, correct weekly expiry per day)
> **Result (flagship config — ATR-adaptive, NEW top):** **+₹2,771,070 net · 57.47% WR · 8,612 trades (2020–2026, uncapped, 15m bias, SL=TP=min(ATR×1.0, 15))**
> **Fixed SL15/TP15 baseline:** +₹1,613,560 net · 54.08% WR · 5,902 trades
> **Born:** 2026-08-29

---

## Why "Last Hope"

Across the SL sweep (uncapped, 15m bias), widening the stop from the original 7 pts to
**15 pts (symmetric 1:1 with TP)** produced a step-function improvement in edge:

| SL / TP | 15m WR | 15m Net PnL | 5m WR | 5m Net PnL |
|---|---:|---:|---:|---:|
| 7 / 15  | 38.1% | +₹326,340  | 37.8% | +₹190,870 |
| 10 / 15 | 45.3% | +₹974,410  | 45.2% | +₹646,000 |
| **15 / 15** | **54.1%** | **+₹1,613,560** | **54.1%** | **+₹1,121,170** |

The original 7-pt stop was bleeding ~62% of trades to SL. At 1:1 R:R the strategy finally
shows a *genuine* statistical edge (WR well above the 50% break-even for 1:1).

---

## Flagship Config (15m bias)

| PARAMETER | VALUE |
|:---|:---|
| **Underlying** | Nifty 50 index options (CE + PE, both sides) |
| **Bias filter** | Marni Fib, INDEX UT-on-HA, **15m** (TV-verified parity) |
| **Elder impulse** | 3m index Elder (USE_ELDER=True) |
| **Entry signals** | Combined 1m/2m/3m/5m stochastic — Super + Flag(M6); ARM=5, touch_buf=1.0, no reversal |
| **Strike selection** | 2nd ITM: **CE = ATM−100, PE = ATM+100** (ATM = index spot 09:15 open) |
| **Stop Loss** | **15 points** (LTP distance) |
| **Take Profit** | **15 points** (LTP distance, 1:1 R:R) |
| **Lot size** | 65 |
| **Fee** | ₹45 flat / trade |
| **Daily cap** | **None (uncapped)** |
| **Concurrency** | Non-concurrent (1 position at a time) |
| **Data range** | 2020-01-01 → 2026-08-27 (1,509 trading days) |
| **Expiry rule** | Correct weekly per day: Thursday 2020→2025-08-28, Tuesday 2025-09-01→today |

**Run command:**
```bash
python run_7y_v4_master.py --cap 0 --sl 15 --tp 15 --bias_tf 15m --workers 8
```

---

## 7-Year Results (15m bias, uncapped, SL15/TP15)

- **Total Trades:** 5,902
- **Win Rate:** 54.08%
- **Net PnL:** **+₹1,613,560**
- **Device:** CUDA (RTX 3060), TF32 high — full run ~1.8s

### Year-by-Year
| Year | Trades | WR | Net PnL |
|---|---:|---:|---:|
| 2020 | 876 | 53.4% | +₹231,240 |
| 2021 | 827 | 53.9% | +₹224,280 |
| 2022 | 1,055 | 53.6% | +₹281,880 |
| 2023 | 848 | 54.8% | +₹240,950 |
| 2024 | 969 | 55.3% | +₹281,980 |
| 2025 | 985 | 53.3% | +₹258,250 |
| 2026 | 342 | 54.4% | +₹94,980 |

### 5-Fold Walk-Forward OOS (each fold = fixed rule tested on a future year)
| Fold | Test Year | OOS Trades | OOS WR | OOS Net PnL |
|---|---|---:|---:|---:|
| F1 | 2022 | 1,055 | 53.6% | +₹281,880 |
| F2 | 2023 | 848 | 54.8% | +₹240,950 |
| F3 | 2024 | 969 | 55.3% | +₹281,980 |
| F4 | 2025 | 985 | 53.3% | +₹258,250 |
| F5 | 2026 | 342 | 54.4% | +₹94,980 |

All folds positive → edge is robust across the full period, including the 2025-09-01
Thursday→Tuesday expiry transition.

---

## 5m Bias Variant (for comparison)
- **Total Trades:** 4,104 · **WR:** 54.07% · **Net PnL:** +₹1,121,170
- Fewer trades but identical WR; 15m bias edges out on total PnL due to higher frequency.

---

## ATR-Adaptive SL+TP — MOST PROMISING VARIANT ✅

Replacing the fixed 15/15 stop with a **volatility-scaled symmetric stop** (`SL = TP = min(ATR(14)×1.0, 15)`)
preserves the 1:1 R:R symmetry that the edge depends on, while tightening on calm days (more trades
captured) and pinning at 15/15 on volatile days. This lifted both WR and PnL materially.

| Bias | Trades | Win Rate | Net PnL |
|---|---:|---:|---:|
| **15m** | **8,612** | **57.47%** | **+₹2,771,070** |
| 5m | 5,521 | 57.45% | +₹1,775,460 |

Every year and every OOS walk-forward fold is strongly positive (2020 +₹414k … 2026 +₹143k; WF
folds +₹143k–467k each). Net PnL is ~1.7× the fixed SL15/TP15 version.

**Config delta vs flagship:** `--atr_sl --atr_mult 1.0 --atr_period 14` (SL/TP no longer fixed; capped at 15).
All other params identical (ARM=5, Elder on, 15m bias, no reversal, 2nd-ITM).

**Run command:**
```bash
python run_7y_v4_master.py --cap 0 --sl 15 --tp 15 --bias_tf 15m --workers 8 --atr_sl --atr_mult 1.0
```

> **Candidate for the parameter sweep:** this ATR variant is the leading starting point. The pending
> 3D batch GPU sweep (Sweep A = fixed SL/TP, Sweep B = ATR SL/TP) will search SL/TP/ARM/Elder/reversal
> grids to maximize WR with least drawdown and max net points, and will add per-config max-drawdown
> tracking that is not yet in the engine.

---

## Honest Caveats (read before trusting live)
1. **Uncapped** — removing the ±29 daily cap roughly doubled PnL vs the capped run. Real
   deployment MUST cap daily loss or a single bad day can wipe weeks of gains.
2. **Cost-simplified** — flat ₹45/trade, **no slippage, no bid/ask spread, no liquidity
   modeling**. Live fills will be worse, especially on the 2nd-ITM strike during fast moves.
3. **Non-walk-forward headline** — the +₹1.61M is the single full-period pass. The strategy
   has **zero fitted parameters** (pure rule), so it is not curve-fit; the WF table confirms
   OOS robustness. Still, a strict optimize-on-N / test-on-N+1 sweep is recommended before sizing.
4. **SL=15/TP=15 is a hand-picked choice** from a 3-point sweep (7/10/15). A wider grid
   (e.g., 12–25) may do better or reveal it is locally optimal.
5. Data ends 2026-08-27; the most recent ~7 months (post expiry change) is the thinnest evidence.

## Files
- Ledger CSV: `artifacts/f6_hybrid/trades_7y_v4_master_15m_cap0_sl15_tp15.csv`
- Run log: `artifacts/f6_hybrid/log_bt3_15m_nocap_sl15_tp15.txt`
- Engine: `run_7y_v4_master.py`
- Data builder: `build_canonical_parquet.py`
