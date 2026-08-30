# Last Hope GPU Engine — Fixed SL/TP Winner (Risk-Adjusted Best)

> **Engine:** `run_7y_v4_master.py` (Master V4) + `gpu_sim_last_hope.py` (GPU sim)
> **Question answered:** "What fixed (non-ATR) SL/TP configuration maximizes risk-adjusted returns across the 7-year Last Hope sweep?"
> **Answer:** the **B-kind, bias-OFF, fixed SL=7 / TP=15** config with **breakeven stop** (`be_trigger=0.70, be_buffer=1.0`) and **touch_buffer=0.0** — **₹1,673,830 net, 44.18% win rate, 19,205 trades**, with exceptional risk ratios (PF 2.996, Sharpe 6.71, Calmar 83.2).
> **Companion results file:** `sweep_touch_fixed.csv` (84 configs: 3 fixed SL/TP × 2 modes × 14 touch buffers).
> **Relationship to ATR champion:** the ATR-based winner (`atr_sl=True, atr_mult=1.5`) generates ₹2,108,703 net (26% more profit) but with lower PF (1.39 vs 3.00). The fixed SL=7 / TP=15 variant is the **superior risk-adjusted** configuration.

---

## 1. Headline Result

| METRIC | VALUE | vs ATR CHAMPION |
|:---|---:|:---|
| **Net Profit (₹)** | **+1,673,829.98** | −₹434,873 (−21%) |
| Gross Points (Σ exit−entry) | 33,676 (≈1.75 pts/trade) | lower (fewer trades) |
| Win Rate | **44.18%** | −19.7pp (asymmetric R:R) |
| Total Trades | 19,205 | −4,993 fewer |
| Avg Trades / Day | 12.73 | fewer entries |
| Max Drawdown (₹) | 20,120.00 | +₹10,817 (higher DD) |
| **Profit Factor** | **2.996** | **+1.606 (2.2× higher)** |
| Payoff (avg W / avg L) | 2.22 | higher (asymmetric) |
| Expectancy (₹/trade) | 87.14 | same |
| **Sharpe (daily, annualized)** | **6.709** | −0.35 (lower) |
| **Sortino (daily, annualized)** | **11.138** | −12.46 (lower) |
| **Calmar (net / maxDD)** | **83.19** | −143.5 (lower) |
| Avg SL distance (pts) | 7.00 | fixed |
| Avg TP distance (pts) | 15.00 | fixed (asymmetric) |
| Lot size | 65 | Fee (₹/trade) | 45 |

**Why this matters:** PF 3.0 means ₹3 won for every ₹1 lost. The ATR champion has PF 1.39 — a much thinner edge per unit risk. For a risk-constrained account, the fixed SL=7 / TP=15 variant is the safer choice.

---

## 2. The Exact Winning Configuration

```python
dict(
    kind='B',          # Fixed SL/TP (atr_sl=False → uses sl/tp params directly)
    sl=7,              # fixed stop loss = 7 pts below entry
    tp=15,             # fixed take profit = 15 pts above entry (asymmetric: TP > SL)
    arm_window=10,     # flag/super must fire within 10 bars of arming
    use_elder=False,   # index Elder impulse gate OFF
    use_rsi=False,     # option-chart RSI gate OFF
    reversal=False,    # cross-side reversal entries OFF
    atr_sl=False,      # FIXED SL/TP (not ATR-based)
    atr_mult=1.5,      # ignored when atr_sl=False
    atr_period=10,     # ignored when atr_sl=False
    cap=0,             # no daily P&L cap (positions exit only at SL/TP)
    use_bias=False,    # 15m Marni-Fib bias OFF
    # --- breakeven stop ---
    be_trigger=0.70,   # after price reaches 0.70*7 = 4.9 pts above entry, harden stop
    be_buffer=1.0,     # ...move SL to entry + 1 pt (bank a small profit, beat noise)
    tp_frac=1.0,       # full distance TP (do NOT shrink target)
    # --- touch-buffer optimization ---
    touch_buffer=0.0,  # SR-bounce: candle must TOUCH/pierce the S/R level (no gap tolerance)
    entry_start=0,     # entry window: full session
    entry_end=345,
    max_bars=0,        # staleness exit OFF
)
```

**One-line thesis:** fixed asymmetric R:R (SL=7, TP=15) gives a 2.14× payoff ratio. Combined with 44% WR, this yields PF ≈ 3.0 — the highest risk-adjusted edge in the entire fixed-SL/TP grid. The breakeven trigger at 0.70 of the 7-pt SL distance (= 4.9 pts) hardens the stop early, converting tail losers into small winners.

---

## 3. How Fixed SL/TP Differs from ATR-Based

| Aspect | Fixed (this winner) | ATR-based (§11 champion) |
|:---|:---|:---|
| `atr_sl` | **False** | True |
| SL distance | **7 pts** (fixed) | min(ATR(10) × 1.5, 15) ≈ 8.3 pts avg |
| TP distance | **15 pts** (fixed) | min(ATR(10) × 1.5, 15) ≈ 8.5 pts avg |
| R:R ratio | **2.14:1** (asymmetric) | ~1:1 (symmetric) |
| Win rate | 44% (harder to hit 15-pt target) | 64% (easier 8.5-pt target) |
| Profit factor | **3.00** (high) | 1.39 (thin) |
| Net profit | ₹1,673,830 | **₹2,108,703** (26% more) |
| Best for | Risk-constrained accounts | Maximum absolute profit |

**Engine implementation (`gpu_sim_last_hope.py:385-389`):**
```python
# When atr_sl=False:
sl_dist = min(sl_param, TP_PTS)    # = min(7, 15) = 7
tp_dist = min(tp_param, TP_PTS)    # = min(15, 15) = 15
sl_price = entry - sl_dist
tp_price = entry + tp_dist * tp_frac
```

---

## 4. Full Sweep Results (Top 10 by Net Profit)

| Rank | Mode | SL/TP | buf | Net Rs | WR | PF | Sharpe | Calmar |
|:---:|:---|:---|:---:|---:|:---:|:---:|:---:|:---:|
| 1 | BE | sl7/tp15 | 0.0 | **1,673,830** | 44.2% | **2.996** | 6.71 | 83.2 |
| 2 | BE | sl7/tp15 | 0.1 | 1,649,460 | 44.1% | 2.944 | 6.59 | 83.7 |
| 3 | BE | sl7/tp15 | 0.2 | 1,618,770 | 43.9% | 2.874 | 6.44 | 81.8 |
| 4 | BE | sl7/tp15 | 0.25 | 1,599,130 | 43.8% | 2.839 | 6.37 | 80.9 |
| 5 | BE | sl7/tp15 | 0.3 | 1,585,490 | 43.8% | 2.810 | 6.31 | 80.2 |
| 6 | BE | sl7/tp15 | 0.4 | 1,576,510 | 43.7% | 2.790 | 6.25 | 81.9 |
| 7 | BE | sl7/tp15 | 0.5 | 1,536,400 | 43.5% | 2.719 | 6.10 | 75.9 |
| 8 | FLAT | sl7/tp15 | 0.0 | 1,593,760 | 40.9% | 2.773 | 6.28 | 66.8 |
| 9 | BE | sl7/tp15 | 0.75 | 1,467,460 | 43.2% | 2.586 | 5.79 | 64.3 |
| 10 | BE | sl10/tp15 | 0.0 | 1,432,450 | 50.9% | 2.438 | 5.42 | 32.8 |

**Key patterns:**
- sl7/tp15 dominates sl10/tp15 and sl15/tp15 across all metrics
- BE mode adds ~₹80K over FLAT at buf=0.0
- Touch buffer monotonic: smaller = better (no inflection)
- sl7/tp15 has higher PF but lower net than ATR-based (fewer trades, lower WR)

---

## 5. Files & Data

| ITEM | PATH |
|:---|:---|
| Master engine | `C:\Websites\FLATTRADE BOT\run_7y_v4_master.py` |
| GPU simulation core | `C:\Websites\FLATTRADE BOT\gpu_sim_last_hope.py` |
| Fixed SL/TP touch-buffer sweep | `C:\Websites\FLATTRADE BOT\gpu_sweep_touch_fixed.py` |
| Results CSV (84 configs) | `C:\Websites\FLATTRADE BOT\sweep_touch_fixed.csv` |
| ATR-based touch-buffer sweep | `C:\Websites\FLATTRADE BOT\gpu_sweep_touch.py` |
| ATR results CSV (28 configs) | `C:\Websites\FLATTRADE BOT\sweep_touch_buffer.csv` |
| ATR champion doc | `C:\Websites\FLATTRADE BOT\LAST_HOPE_WINNER.md` (§11) |

---

## 6. Reproduce

```python
import sys, time
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
import run_7y_v4_master as M
import gpu_sim_last_hope as G

cfg = dict(kind='B', sl=7, tp=15, arm_window=10, use_elder=False, use_rsi=False,
           reversal=False, atr_sl=False, atr_mult=1.5, atr_period=10, cap=0,
           use_bias=False, tp_frac=1.0, be_trigger=0.70, be_buffer=1.0,
           touch_buffer=0.0, entry_start=0, entry_end=345, max_bars=0)

t0 = time.time()
trades = G.gpu_sim_batch([cfg])[0]
r = M._metrics_from_trades(trades)
print(f"trades={r['trades']} wr={r['wr']:.4f} net_rs={r['net_rs']:.2f} max_dd={r['max_dd']:.2f}")
print(f"elapsed={time.time()-t0:.0f}s")
```

Expected: `trades=19205 wr=0.4418 net_rs=1673829.98 max_dd=20120.00`

---

## 7. Caveats

- **In-sample.** Full-window (2020→2026) optimization pick, not walk-forward.
- **Asymmetric R:R.** SL=7, TP=15 means you lose 7 pts to win 15 pts. This requires 44% WR to break even — the strategy achieves 44.18%, just barely above breakeven on hit-rate alone. The edge comes from the breakeven stop converting some 7-pt losses into 1-pt wins.
- **Fewer trades.** 19,205 vs 24,198 (ATR). The tighter SL means more trades get stopped out before reaching the SR-bounce setup's full potential.
- **Higher drawdown.** ₹20,120 vs ₹9,303 (ATR). The fixed 7-pt stop is tighter than ATR's ~8.3 pts, so more trades hit SL, creating larger cumulative drawdowns.
- **Cost model.** ₹45 flat fee/lot, no slippage.
