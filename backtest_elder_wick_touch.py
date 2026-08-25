"""Elder Wick Touch Strategy — 5-Year Backtest (2020-2024).

NEW SUPREME STRATEGY:
  - Signal: Any candle wick (upper/lower shadow) touches S/R level within buffer.
  - Elder Impulse Gate: GREEN/BLUE for CE (LONG), RED/BLUE for PE (SHORT).
  - Index chart signals → Options chart execution (2nd ITM).
  - Same 3-Tier S/R Hierarchy as Combined Supreme.
  - Same exits: SL/TP/Trailing SL on spot index points.
  - Fee: ₹45/trade deducted.
"""

from __future__ import annotations

import concurrent.futures
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──
ROOT = Path(r"C:\Websites\FLATTRADE BOT")
DESKTOP_DATA = Path(r"C:\Users\user\Desktop\nifty50 data")
OPT_DIR = DESKTOP_DATA / "nifty_options"
IDX_FILE = DESKTOP_DATA / "index" / "NIFTY 50_minute.csv"

LOT_SIZE = 65
FEE_PER_TRADE = 45.0
WORKERS = 8
SYM_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")


# ═══════════════════════════════════════════════════════════════════════
# INCREMENTAL ELDER IMPULSE (standalone for backtest, no imports needed)
# ═══════════════════════════════════════════════════════════════════════

class _EMA:
    def __init__(self, period):
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self.value = None

    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            self.value = v * self.alpha + self.value * (1 - self.alpha)
        return self.value


class ElderImpulse:
    """EMA(13) slope + MACD(12,26,9) histogram slope."""

    def __init__(self):
        self.ema13 = _EMA(13)
        self.ema12 = _EMA(12)
        self.ema26 = _EMA(26)
        self.macd_ema9 = _EMA(9)
        self.prev_ema13 = None
        self.prev_hist = None
        self.color = "blue"

    def update(self, close: float) -> str:
        e13 = self.ema13.update(close)
        e12 = self.ema12.update(close)
        e26 = self.ema26.update(close)
        macd_line = e12 - e26
        signal_line = self.macd_ema9.update(macd_line)
        histogram = macd_line - signal_line  # TRUE MACD histogram
        color = "blue"
        if self.prev_ema13 is not None and self.prev_hist is not None:
            if e13 > self.prev_ema13 and histogram > self.prev_hist:
                color = "green"
            elif e13 < self.prev_ema13 and histogram < self.prev_hist:
                color = "red"
        self.prev_ema13 = e13
        self.prev_hist = histogram
        self.color = color
        return color


def elder_allows(color: str, side: str) -> bool:
    """GREEN/BLUE→CE allowed, RED/BLUE→PE allowed (permissive mode)."""
    if side == "CE":
        return color in ("green", "blue")
    else:
        return color in ("red", "blue")


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING (reused from Direct Option Chart backtest)
# ═══════════════════════════════════════════════════════════════════════

def extract_strike_from_sym(sym: str):
    m = SYM_RE.match(sym)
    if not m:
        return None
    return (int(m.group(2)), m.group(3))


def parse_option_file(fpath: Path):
    try:
        df = pd.read_parquet(fpath) if fpath.suffix == ".parquet" else pd.read_csv(fpath)
        if df.empty or "close" not in df.columns or "symbol" not in df.columns:
            return None
        d_str = str(df["date"].iloc[0])
        dt = pd.to_datetime(d_str)
        day_key = dt.strftime("%Y-%m-%d")

        df["datetime"] = pd.to_datetime(df["date"])
        df["minute"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
        return (day_key, df)
    except Exception:
        return None


def compute_supertrend(highs, lows, closes, period=10, multiplier=3.0):
    n = len(closes)
    hl2 = (highs + lows) / 2.0
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
    atr = np.convolve(tr, np.ones(period) / period, mode="full")[:n]
    st = np.full(n, np.nan)
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper_band = np.copy(upper_basic)
    lower_band = np.copy(lower_basic)
    for i in range(period, n):
        if lower_basic[i] > lower_band[i - 1] or closes[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i - 1]
        if upper_basic[i] < upper_band[i - 1] or closes[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i - 1]
    direction = np.ones(n)
    for i in range(period, n):
        if direction[i - 1] == 1:
            if closes[i] < lower_band[i]:
                direction[i] = -1
                st[i] = upper_band[i]
            else:
                direction[i] = 1
                st[i] = lower_band[i]
        else:
            if closes[i] > upper_band[i]:
                direction[i] = 1
                st[i] = lower_band[i]
            else:
                direction[i] = -1
                st[i] = upper_band[i]
    return st, direction


# ═══════════════════════════════════════════════════════════════════════
# ELDER WICK TOUCH SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def run_elder_wick_touch(
    days: List[str],
    spot_by_day: Dict[str, Dict[int, float]],
    day_opt: Dict[str, pd.DataFrame],
    daily_levels: Dict[str, Any],
    virgin_cprs_by_day: Dict[str, Any],
    sl_pts: float = 7.0,
    tp_pts: float = 14.0,
    trail_trigger_pts: float = 6.0,
    trail_step_pts: float = 2.0,
    buffer_atr_mult: float = 0.30,
    buffer_min_pts: float = 3.0,
) -> Tuple[List[Dict], Dict]:
    trades = []

    for day in days:
        if day not in spot_by_day or day not in day_opt:
            continue

        spot_dict = spot_by_day[day]
        opt_df = day_opt[day]
        if opt_df.empty or len(spot_dict) < 50:
            continue

        # Parse option symbols
        sym_map = {}
        for s in opt_df["symbol"].unique():
            res = extract_strike_from_sym(s)
            if res:
                sym_map[res] = s

        # Build 3m spot bars
        spot_mins = sorted(list(spot_dict.keys()))
        spot_closes = [spot_dict[m] for m in spot_mins]
        spot_df = pd.DataFrame({"minute": spot_mins, "close": spot_closes})
        spot_df["bar_3m_idx"] = (spot_df["minute"] - 555) // 3

        agg_3m = spot_df.groupby("bar_3m_idx").agg(
            open=("close", "first"),
            high=("close", "max"),
            low=("close", "min"),
            close=("close", "last"),
            minute_start=("minute", "first"),
        ).to_dict("records")

        if len(agg_3m) < 15:
            continue

        # SuperTrend + VWAP for chop filter
        highs_3m = np.array([b["high"] for b in agg_3m])
        lows_3m = np.array([b["low"] for b in agg_3m])
        closes_3m = np.array([b["close"] for b in agg_3m])
        st_vals, _ = compute_supertrend(highs_3m, lows_3m, closes_3m, 10, 3.0)

        # 15m bars for macro trend gate
        spot_df["bar_15m_idx"] = (spot_df["minute"] - 555) // 15
        agg_15m = spot_df.groupby("bar_15m_idx").agg(close=("close", "last")).to_dict("records")

        # Daily S/R levels
        dl = daily_levels.get(day, {})
        virgins = virgin_cprs_by_day.get(day, [])
        op_h = agg_3m[0]["high"] if agg_3m else 24200.0
        op_l = agg_3m[0]["low"] if agg_3m else 24100.0
        touch_budget = {}

        # Elder Impulse (reset daily)
        elder = ElderImpulse()
        cooldown_until = 0  # Bar index cooldown

        for b_idx in range(1, len(agg_3m)):
            bar = agg_3m[b_idx]
            t_min = bar["minute_start"]

            # Session filter: 09:18 to 15:00
            if t_min < 558 or t_min > 900:
                continue

            spot_px = bar["close"]
            bar_high = bar["high"]
            bar_low = bar["low"]
            bar_open = bar["open"]

            # Update Elder on each 3m bar
            elder_color = elder.update(spot_px)

            # Cooldown: no trade for 3 bars after last trade
            if b_idx < cooldown_until:
                continue

            # VWAP (simple running mean)
            vwap = float(np.mean(closes_3m[:b_idx + 1]))
            st_val = st_vals[b_idx] if not np.isnan(st_vals[b_idx]) else spot_px

            # Chop Corridor Filter
            chop_hi = max(st_val, vwap)
            chop_lo = min(st_val, vwap)
            if chop_lo <= spot_px <= chop_hi:
                continue

            # 15m Macro Trend Gate
            b15_idx = min(b_idx // 5, len(agg_15m) - 1)
            is_bull_15m = agg_15m[b15_idx]["close"] >= vwap

            # ATR(5) on spot
            past_trs = []
            for k in range(max(1, b_idx - 5), b_idx + 1):
                tr = max(
                    highs_3m[k] - lows_3m[k],
                    abs(highs_3m[k] - closes_3m[k - 1]),
                    abs(lows_3m[k] - closes_3m[k - 1]),
                )
                past_trs.append(tr)
            atr5 = float(np.mean(past_trs)) if past_trs else 10.0

            # Dynamic buffer
            buffer = max(buffer_atr_mult * atr5, buffer_min_pts)

            # EMA proxies
            ema20_5m = float(np.mean(closes_3m[max(0, b_idx - 10):b_idx + 1]))
            ema200_5m = float(np.mean(closes_3m[max(0, b_idx - 40):b_idx + 1]))
            ema20_3m = float(np.mean(closes_3m[max(0, b_idx - 6):b_idx + 1]))

            # S/R candidates
            candidates = [
                ("Virgin CPR Pivot", virgins[0][0] if virgins else None, 1, True),
                ("Virgin CPR Top", virgins[0][1] if virgins else None, 1, True),
                ("Virgin CPR Bot", virgins[0][2] if virgins else None, 1, True),
                ("Camarilla H3", dl.get("cam_h3"), 1, False),
                ("Camarilla L3", dl.get("cam_l3"), 1, False),
                ("Daily CPR Pivot", dl.get("cpr_p"), 1, False),
                ("Daily CPR Top", dl.get("cpr_top"), 1, False),
                ("Daily CPR Bot", dl.get("cpr_bot"), 1, False),
                ("Daily VWAP", vwap, 1, False),
                ("5m EMA 20", ema20_5m, 1, False),
                ("5m EMA 200", ema200_5m, 1, False),
                ("Opening 3m High", op_h, 2, False),
                ("Opening 3m Low", op_l, 2, False),
                ("3m EMA 20", ema20_3m, 2, False),
                ("Prev Day High", dl.get("pdh"), 2, False),
                ("Prev Day Low", dl.get("pdl"), 2, False),
                ("Fibonacci H3", dl.get("fib_h3"), 3, False),
                ("Fibonacci L3", dl.get("fib_l3"), 3, False),
                ("Camarilla H4", dl.get("cam_h4"), 3, False),
                ("Camarilla L4", dl.get("cam_l4"), 3, False),
            ]

            for lvl_name, lvl_px, tier, is_v in candidates:
                if lvl_px is None:
                    continue
                if touch_budget.get(lvl_name, 0) >= 2:
                    continue

                # ── WICK TOUCH DETECTION ──
                # Lower wick touches S/R (support bounce → LONG/CE)
                lower_wick = min(bar_open, spot_px) - bar_low  # Lower shadow size
                lower_touch = abs(bar_low - lvl_px) <= buffer and lower_wick > 0

                # Upper wick touches S/R (resistance rejection → SHORT/PE)
                upper_wick = bar_high - max(bar_open, spot_px)  # Upper shadow size
                upper_touch = abs(bar_high - lvl_px) <= buffer and upper_wick > 0

                # ── LONG SETUP (CE): lower wick touched level, close bounced (within buffer OK) ──
                if lower_touch and spot_px >= (lvl_px - buffer) and is_bull_15m:
                    if elder_allows(elder_color, "CE"):
                        score = 40 + (25 if is_v else 20 if tier == 1 else 10 if tier == 2 else 5)
                        if is_bull_15m:
                            score += 25
                        if score >= 50:
                            atm = int(round(spot_px / 50.0) * 50)
                            ce_strike = atm - 100
                            opt_sym = sym_map.get((ce_strike, "CE"))
                            if not opt_sym:
                                continue

                            # CAUSAL: Signal at bar close → enter on NEXT minute's open
                            # Options are 1-min bars; spot signal on 3m bar close
                            next_bar_min = t_min + 1
                            c_df = opt_df[opt_df["symbol"] == opt_sym].sort_values("minute")
                            c_bars = c_df[c_df["minute"] >= next_bar_min].to_dict("records")
                            if not c_bars:
                                continue

                            entry_px = c_bars[0]["open"]  # Next bar's open (causal)
                            sl_px = entry_px - sl_pts
                            tp_px = entry_px + tp_pts
                            peak = entry_px
                            trail_active = False
                            cur_sl = sl_px
                            exit_px = entry_px
                            won = False

                            for fb in c_bars[1:]:
                                if fb["high"] > peak:
                                    peak = fb["high"]
                                    if (peak - entry_px) >= trail_trigger_pts:
                                        trail_active = True
                                if trail_active:
                                    new_sl = peak - trail_step_pts
                                    if new_sl > cur_sl:
                                        cur_sl = new_sl
                                if fb["low"] <= cur_sl:
                                    exit_px = cur_sl
                                    won = exit_px > entry_px
                                    break
                                if fb["high"] >= tp_px:
                                    exit_px = tp_px
                                    won = True
                                    break
                                if fb["minute"] >= 900:
                                    exit_px = fb["close"]
                                    won = exit_px > entry_px
                                    break
                            else:
                                exit_px = c_bars[-1]["close"]
                                won = exit_px > entry_px

                            pnl_pts = exit_px - entry_px
                            net_rs = (pnl_pts * LOT_SIZE) - FEE_PER_TRADE
                            trades.append({
                                "day": day, "direction": "LONG", "level": lvl_name,
                                "elder": elder_color, "entry": entry_px, "exit": exit_px,
                                "pnl_pts": pnl_pts, "net_rs": net_rs, "win": won,
                                "year": int(day[:4]),
                            })
                            touch_budget[lvl_name] = touch_budget.get(lvl_name, 0) + 1
                            cooldown_until = b_idx + 3
                            break

                # ── SHORT SETUP (PE): upper wick touched level, close rejected (within buffer OK) ──
                if upper_touch and spot_px <= (lvl_px + buffer) and not is_bull_15m:
                    if elder_allows(elder_color, "PE"):
                        score = 40 + (25 if is_v else 20 if tier == 1 else 10 if tier == 2 else 5)
                        if not is_bull_15m:
                            score += 25
                        if score >= 50:
                            atm = int(round(spot_px / 50.0) * 50)
                            pe_strike = atm + 100
                            opt_sym = sym_map.get((pe_strike, "PE"))
                            if not opt_sym:
                                continue

                            # CAUSAL: Signal at bar close → enter on NEXT minute's open
                            next_bar_min = t_min + 1
                            c_df = opt_df[opt_df["symbol"] == opt_sym].sort_values("minute")
                            c_bars = c_df[c_df["minute"] >= next_bar_min].to_dict("records")
                            if not c_bars:
                                continue

                            entry_px = c_bars[0]["open"]  # Next bar's open (causal)
                            sl_px = entry_px - sl_pts
                            tp_px = entry_px + tp_pts
                            peak = entry_px
                            trail_active = False
                            cur_sl = sl_px
                            exit_px = entry_px
                            won = False

                            for fb in c_bars[1:]:
                                if fb["high"] > peak:
                                    peak = fb["high"]
                                    if (peak - entry_px) >= trail_trigger_pts:
                                        trail_active = True
                                if trail_active:
                                    new_sl = peak - trail_step_pts
                                    if new_sl > cur_sl:
                                        cur_sl = new_sl
                                if fb["low"] <= cur_sl:
                                    exit_px = cur_sl
                                    won = exit_px > entry_px
                                    break
                                if fb["high"] >= tp_px:
                                    exit_px = tp_px
                                    won = True
                                    break
                                if fb["minute"] >= 900:
                                    exit_px = fb["close"]
                                    won = exit_px > entry_px
                                    break
                            else:
                                exit_px = c_bars[-1]["close"]
                                won = exit_px > entry_px

                            pnl_pts = exit_px - entry_px
                            net_rs = (pnl_pts * LOT_SIZE) - FEE_PER_TRADE
                            trades.append({
                                "day": day, "direction": "SHORT", "level": lvl_name,
                                "elder": elder_color, "entry": entry_px, "exit": exit_px,
                                "pnl_pts": pnl_pts, "net_rs": net_rs, "win": won,
                                "year": int(day[:4]),
                            })
                            touch_budget[lvl_name] = touch_budget.get(lvl_name, 0) + 1
                            cooldown_until = b_idx + 3
                            break

    if not trades:
        return [], {"trades": 0, "win_rate": 0.0, "net_profit": 0.0, "pf": 0.0, "calmar": 0.0}

    df_tr = pd.DataFrame(trades)
    n_tr = len(df_tr)
    wr = float(df_tr["win"].mean() * 100)
    net_p = float(df_tr["net_rs"].sum())
    gw = float(df_tr[df_tr["net_rs"] > 0]["net_rs"].sum())
    gl = float(abs(df_tr[df_tr["net_rs"] < 0]["net_rs"].sum()))
    pf = round(gw / gl, 2) if gl > 0 else 999.0
    cum = df_tr["net_rs"].cumsum()
    peak_cum = np.maximum.accumulate(cum)
    dd = peak_cum - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 1.0
    calmar = round(net_p / max(max_dd, 100.0), 1)

    return trades, {
        "trades": n_tr, "win_rate": wr, "net_profit": net_p,
        "pf": pf, "max_dd": max_dd, "calmar": calmar,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print(" 🔮 ELDER WICK TOUCH STRATEGY — 5-YEAR BACKTEST (2020-2024)")
    print("=" * 90)

    # 1. Load Spot Index
    print("\n[1] Loading Spot Index...")
    df_spot = pd.read_csv(IDX_FILE)
    df_spot["datetime"] = pd.to_datetime(df_spot["date"])
    df_spot["day_str"] = df_spot["datetime"].dt.strftime("%Y-%m-%d")
    df_spot["minute"] = df_spot["datetime"].dt.hour * 60 + df_spot["datetime"].dt.minute
    spot_by_day = {}
    for d, grp in df_spot.groupby("day_str"):
        spot_by_day[d] = dict(zip(grp["minute"], grp["close"]))

    # 2. Compute Daily S/R Levels
    print("[2] Computing S/R Levels...")
    daily_stats = df_spot.groupby("day_str").agg(
        high=("close", "max"), low=("close", "min"), close=("close", "last"),
    ).reset_index()

    all_days = list(daily_stats["day_str"].unique())
    daily_levels = {}
    virgin_cprs_by_day = {}
    history = []

    # Precompute daily high/low ranges for Virgin CPR check (avoid O(n²) DataFrame filtering)
    daily_ranges = {}
    for d, grp in df_spot.groupby("day_str"):
        daily_ranges[d] = (grp["close"].min(), grp["close"].max())

    for i in range(1, len(all_days)):
        prev_d = all_days[i - 1]
        cur_d = all_days[i]
        p = daily_stats[daily_stats["day_str"] == prev_d].iloc[0]
        ph, pl, pc = p["high"], p["low"], p["close"]

        pivot = (ph + pl + pc) / 3.0
        bc = (ph + pl) / 2.0
        tc = (pivot - bc) + pivot
        cpr_top, cpr_bot = max(tc, bc), min(tc, bc)
        rng = ph - pl

        daily_levels[cur_d] = {
            "pdh": ph, "pdl": pl, "pdc": pc,
            "cpr_p": pivot, "cpr_top": cpr_top, "cpr_bot": cpr_bot,
            "cam_h3": pc + rng * (1.1 / 4.0), "cam_l3": pc - rng * (1.1 / 4.0),
            "cam_h4": pc + rng * (1.1 / 2.0), "cam_l4": pc - rng * (1.1 / 2.0),
            "fib_h3": pivot + rng, "fib_l3": pivot - rng,
        }

        history.append((pivot, cpr_top, cpr_bot, prev_d))
        active_v = []
        if cur_d in daily_ranges:
            d_low, d_high = daily_ranges[cur_d]
            for vp, vtc, vbc, vday in history[:-1]:
                if not (d_low <= vtc and d_high >= vbc):
                    active_v.append((vp, vtc, vbc, vday))
        virgin_cprs_by_day[cur_d] = active_v

    # 3. Load Options Data
    print("[3] Loading Options CSVs...")
    opt_files = sorted(list(OPT_DIR.glob("**/*.csv")) + list(OPT_DIR.glob("**/*.parquet")))
    print(f"    Found {len(opt_files)} option files.")
    day_opt = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(parse_option_file, opt_files))
    for r in results:
        if r is not None:
            day_opt[r[0]] = r[1]

    common_days = sorted(set(day_opt.keys()) & set(spot_by_day.keys()))
    print(f"    ✅ {len(common_days)} valid trading days ({common_days[0]} to {common_days[-1]})")

    # 4. SMOKE TEST (5 Days)
    print("\n--- 🔥 SMOKE TEST (5 Days) ---")
    smoke_days = common_days[:5]
    _, s_sum = run_elder_wick_touch(smoke_days, spot_by_day, day_opt, daily_levels, virgin_cprs_by_day)
    print(f"Smoke: Trades={s_sum['trades']}, WR={s_sum['win_rate']:.1f}%, Profit=Rs {s_sum['net_profit']:,.2f}")
    assert s_sum["trades"] > 0, "❌ Smoke test failed: 0 trades!"
    print("✅ Smoke test passed!\n")

    # 5. FULL 5-YEAR BACKTEST
    print("Running Full 5-Year Elder Wick Touch Simulation...")
    t0 = time.time()
    tr_full, sum_full = run_elder_wick_touch(common_days, spot_by_day, day_opt, daily_levels, virgin_cprs_by_day)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s.\n")

    # 6. RESULTS
    print("=" * 100)
    print(f"{'ELDER WICK TOUCH STRATEGY':<55} | {'TRADES':<7} | {'WIN RATE':<9} | {'PROFIT (Rs)':<16} | {'PF':<5} | {'CALMAR':<7}")
    print("-" * 100)
    print(f"{'Full Period NWF':<55} | {sum_full['trades']:<7} | {sum_full['win_rate']:<8.1f}% | Rs {sum_full['net_profit']:>12,.2f} | {sum_full['pf']:<5.2f} | {sum_full['calmar']:<7.1f}")
    print("=" * 100)

    df_all = pd.DataFrame(tr_full)
    years = sorted(df_all["year"].unique())

    print(f"\n{'YEAR':<6} | {'TRADES':<7} | {'WIN RATE':<9} | {'NET PROFIT (Rs)':<16} | {'PF':<13} | {'GREEN DAYS':<10}")
    print("-" * 75)
    for y in years:
        sub = df_all[df_all["year"] == y]
        n = len(sub)
        wr = float(sub["win"].mean() * 100)
        net = float(sub["net_rs"].sum())
        gw = float(sub[sub["net_rs"] > 0]["net_rs"].sum())
        gl = float(abs(sub[sub["net_rs"] < 0]["net_rs"].sum()))
        pf = round(gw / gl, 2) if gl > 0 else 999.0
        dp = sub.groupby("day")["net_rs"].sum()
        gd = float((dp > 0).sum() / max(1, len(dp)) * 100)
        print(f"{y:<6} | {n:<7} | {wr:<8.1f}% | Rs {net:>12,.2f} | {pf:<13.2f} | {gd:<9.1f}%")
    print("-" * 75)
    print(f"{'TOTAL':<6} | {sum_full['trades']:<7} | {sum_full['win_rate']:<8.1f}% | Rs {sum_full['net_profit']:>12,.2f} | {sum_full['pf']:<13.2f}")
    print("=" * 75)

    # Elder color distribution
    print(f"\n{'ELDER COLOR':<12} | {'TRADES':<7} | {'WIN RATE':<9} | {'NET PROFIT':<14}")
    print("-" * 50)
    for color in ["green", "blue", "red"]:
        sub = df_all[df_all["elder"] == color]
        if len(sub) > 0:
            print(f"{color.upper():<12} | {len(sub):<7} | {float(sub['win'].mean()*100):<8.1f}% | Rs {float(sub['net_rs'].sum()):>10,.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
