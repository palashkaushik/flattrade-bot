"""Dow Jones Global Macro Filter Engine for APEX RUNNER.

Quantitative Framework:
  1. Computes Previous Day US Dow Jones Industrial Average (DJIA) Close-to-Close Return.
  2. Classifies Day into:
     - DOW_BULLISH: Dow Return >= +0.30% (Prioritizes CE Runners, Suppresses Morning PE Counter-Trend)
     - DOW_BEARISH: Dow Return <= -0.30% (Prioritizes PE Runners, Suppresses Morning CE Counter-Trend)
     - DOW_NEUTRAL: |Dow Return| < 0.30% (Standard Bidirectional Scalp)
  3. Evaluates Mandatory 5-Day Smoke Test & Full 2024–2025 Overlapping Dataset.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_5y_optimized import SYM_RE, latest_spot, option_files
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.test_super_only_aug import extend_with_august, ParamStoch, IncrementalATR, bslice, to_hhmm, LOT_SIZE, FEE
from artifacts.f6_hybrid.compare_rules_1_and_2 import load_full_ohlc_spot
import grid_optimize_f6_atr as grid


def load_dow_sentiment_map() -> dict[str, dict]:
    dow_file = Path("C:/Users/user/Desktop/nifty50 data/DowJones1m.csv")
    if not dow_file.exists():
        print(f"[Warning] Dow file not found at {dow_file}")
        return {}

    df = pd.read_csv(dow_file, usecols=["time", "close"])
    df["utc_dt"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["utc_dt"].dt.date.astype(str)

    # Daily close per US trading day
    day_close = df.groupby("date")["close"].last().to_dict()
    us_days_sorted = sorted(day_close.keys())

    # Map each US date's return
    us_returns = {}
    for i in range(1, len(us_days_sorted)):
        d_curr = us_days_sorted[i]
        d_prev = us_days_sorted[i - 1]
        c_curr = day_close[d_curr]
        c_prev = day_close[d_prev]
        pct = ((c_curr - c_prev) / c_prev) * 100.0
        us_returns[d_curr] = {"close": c_curr, "prev_close": c_prev, "pct": pct}

    # Map to Nifty Trading Days (Uses most recent US trading day before Nifty day)
    nifty_dow_map = {}
    for nifty_day in pd.date_range("2024-01-01", "2026-08-31").strftime("%Y-%m-%d"):
        prior_us = [d for d in us_days_sorted if d < nifty_day]
        if prior_us:
            latest_us_day = prior_us[-1]
            ret_info = us_returns.get(latest_us_day)
            if ret_info:
                pct = ret_info["pct"]
                if pct >= 0.30:
                    bias = "BULLISH"
                elif pct <= -0.30:
                    bias = "BEARISH"
                else:
                    bias = "NEUTRAL"
                nifty_dow_map[nifty_day] = {
                    "us_day": latest_us_day,
                    "dow_pct": pct,
                    "bias": bias,
                }
            else:
                nifty_dow_map[nifty_day] = {"bias": "NEUTRAL", "dow_pct": 0.0}
        else:
            nifty_dow_map[nifty_day] = {"bias": "NEUTRAL", "dow_pct": 0.0}

    return nifty_dow_map


class ApexRunnerDetector:
    def __init__(self):
        self.stoch = ParamStoch()
        self.prev_s1 = None
        self._fired = False

    def push(self, c: Candle):
        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
        is_super = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
        is_flag = s4 is not None and s1 is not None and s4 >= 79.5 and s1 <= 20.5
        s1_turn_up = self.prev_s1 is not None and s1 is not None and s1 > self.prev_s1

        cond = (is_super or is_flag) and s1_turn_up
        trig = False
        if cond and not self._fired:
            trig = True
            self._fired = True
        if not cond:
            self._fired = False

        self.prev_s1 = s1
        return trig, "SUPER" if is_super else "FLAG", c.close


def simulate_apex_dow_day(
    day: str,
    opt_map: dict,
    all_cal: list,
    cal_idx: dict,
    spot_all: dict,
    dow_info: dict,
    use_dow_filter: bool = False,
    initial_sl_pts: float = 6.0,
    lock_trigger_pts: float = 12.0,
    locked_profit_pts: float = 10.0,
    trail_dist_pts: float = 4.0,
    hard_tp_pts: float = 20.0,
    start_minute: int = 570,  # 09:30 AM
    end_minute: int = 900,    # 03:00 PM
):
    fpath = opt_map.get(day)
    fprev = opt_map.get(all_cal[cal_idx[day] - 1]) if cal_idx.get(day, 0) > 0 else ""
    gc = grid.cached_day(str(fpath))
    if not gc:
        return []

    spot = spot_all.get(day)
    sp0 = latest_spot(spot, 555) or latest_spot(spot, 560)
    if sp0 is None:
        return []
    atm0 = int(round(sp0 / 50) * 50)
    target_strikes = set(range(atm0 - 250, atm0 + 300, 50))

    prefix = "NIFTY"
    for s in gc.keys():
        if (m := SYM_RE.match(s)):
            prefix = m.group(1)
            break

    def filtered(data):
        return {sym: g for sym, g in data.items()
                if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}

    gu = filtered(gc)
    gp = filtered(grid.cached_day(str(fprev))) if fprev else {}

    trk = {}
    for sym, g in gp.items():
        trk[sym] = ApexRunnerDetector()
        for i in range(len(g["min"])):
            c = Candle(open=g["open"][i], high=g["high"][i], low=g["low"][i], close=g["close"][i], minute=g["min"][i])
            trk[sym].push(c)

    pmtrig = {}
    slices = {}
    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = ApexRunnerDetector()
        t = trk[sym]
        slices[sym] = g
        mm2 = SYM_RE.match(sym)
        if not mm2:
            continue
        sv, side = int(mm2.group(2)), mm2.group(3)
        for i in range(len(g["min"])):
            m = g["min"][i]
            c = Candle(open=g["open"][i], high=g["high"][i], low=g["low"][i], close=g["close"][i], minute=m)
            trig, stype, px = t.push(c)
            if trig:
                pmtrig.setdefault(m, []).append((side, sv, sym, c.close, stype))

    def ainfo(side, m):
        spx = latest_spot(spot, m)
        if spx is None:
            return None
        atm = int(round(spx / 50) * 50)
        stk = atm + (-100 if side == "CE" else 100)
        sym = f"{prefix}{stk}{side}"
        sl = slices.get(sym)
        return (sym, sl, stk) if sl is not None else None

    dow_bias = dow_info.get("bias", "NEUTRAL") if use_dow_filter else "NEUTRAL"

    trades = []
    pos = None

    for minute in range(560, 931):
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c)
                pos["duration_min"] += 1
                if h > pos["peak_px"]:
                    pos["peak_px"] = float(h)

                gain = pos["peak_px"] - pos["entry"]

                # High-Yield Lock at +12 pts -> Lock +10 pts
                if gain >= lock_trigger_pts:
                    locked_sl = pos["entry"] + locked_profit_pts
                    if locked_sl > pos["sl"]:
                        pos["sl"] = round(locked_sl, 2)
                        pos["is_locked"] = True

                # Chandelier Trail after locking
                if pos["is_locked"]:
                    trail_sl = pos["peak_px"] - trail_dist_pts
                    if trail_sl > pos["sl"]:
                        pos["sl"] = round(trail_sl, 2)
                        pos["is_trailing"] = True

                ex, rsn = None, ""
                if l <= pos["sl"] and h >= pos["tp"]:
                    ex, rsn = pos["sl"], "PROFIT_LOCK" if pos.get("is_locked") else "SL"
                elif h >= pos["tp"]:
                    ex, rsn = pos["tp"], "BIG_TP"
                elif l <= pos["sl"]:
                    ex, rsn = pos["sl"], "PROFIT_LOCK" if pos.get("is_locked") else "SL"

                if ex is not None:
                    pts = round(ex - pos["entry"], 2)
                    rs_net = round(pts * LOT_SIZE - FEE, 2)
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": ex, "pts": pts, "rs_net": rs_net, "fee": FEE,
                        "reason": rsn, "duration_min": pos["duration_min"], "stype": pos["stype"],
                    })
                    pos = None

        if minute >= end_minute and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            rs_net = round(pts * LOT_SIZE - FEE, 2)
            trades.append({
                "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                "exit": pos["last_px"], "pts": pts, "rs_net": rs_net, "fee": FEE,
                "reason": "EOD", "duration_min": pos["duration_min"], "stype": pos["stype"],
            })
            pos = None
            break

        if pos is not None or minute < start_minute or minute >= end_minute:
            continue

        for (sig_side, sig_stk, sig_sym, c_px, stype) in pmtrig.get(minute, []):
            # Apply Dow Macro Filter during the morning session (09:30 to 11:30 AM)
            if use_dow_filter and minute < 690:  # Before 11:30 AM
                if dow_bias == "BULLISH" and sig_side == "PE":
                    continue  # Suppress counter-trend PE against strong US bull momentum
                elif dow_bias == "BEARISH" and sig_side == "CE":
                    continue  # Suppress counter-trend CE against strong US bear momentum

            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                bar = bslice(ai[1], minute)
                if bar:
                    ep = float(bar[3])
                    pos = {
                        "entry": ep,
                        "sl": round(ep - initial_sl_pts, 2),
                        "tp": round(ep + hard_tp_pts, 2),
                        "side": sig_side, "symbol": ai[0], "entry_min": minute,
                        "last_px": ep, "peak_px": ep, "slice": ai[1],
                        "duration_min": 0, "is_locked": False, "is_trailing": False,
                        "stype": stype,
                    }
                    break

    return trades


def run_dow_experiment(days_subset: list, title: str):
    spot_all = load_full_ohlc_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    grid.GLOBAL_SPOT = spot_all
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}
    dow_map = load_dow_sentiment_map()

    configs = [
        {"name": "1. APEX RUNNER BASELINE (No Dow Filter)", "use_dow": False},
        {"name": "2. APEX RUNNER + DOW MACRO FILTER (Overnight US Bias)", "use_dow": True},
    ]

    print("=" * 145)
    print(f"EXPERIMENT: {title.upper()} ({len(days_subset)} Trading Days)")
    print("=" * 145)

    results_table = []

    for cfg in configs:
        all_trs = []
        for d in days_subset:
            dow_info = dow_map.get(d, {"bias": "NEUTRAL", "dow_pct": 0.0})
            trs = simulate_apex_dow_day(
                d, opt_map, all_cal, cal_idx, spot_all, dow_info,
                use_dow_filter=cfg["use_dow"],
            )
            all_trs.extend(trs)

        n_t = len(all_trs)
        wins = [t for t in all_trs if t["rs_net"] > 0]
        losses = [t for t in all_trs if t["rs_net"] <= 0]
        net_rs = sum(t["rs_net"] for t in all_trs)
        net_pts = sum(t["pts"] for t in all_trs)
        wr = len(wins) / n_t * 100 if n_t > 0 else 0.0
        win_tot = sum(t["rs_net"] for t in wins)
        loss_tot = abs(sum(t["rs_net"] for t in losses))
        pf = win_tot / loss_tot if loss_tot > 0 else (99.0 if win_tot > 0 else 0.0)

        eq = np.cumsum([t["rs_net"] for t in all_trs]) if all_trs else np.array([0.0])
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0

        wins_pts = [t["pts"] for t in wins]
        loss_pts = [t["pts"] for t in losses]
        avg_win = float(np.mean(wins_pts)) if wins_pts else 0.0
        avg_loss = float(np.mean(loss_pts)) if loss_pts else 0.0

        results_table.append({
            "name": cfg["name"],
            "trades": n_t,
            "win_rate": wr,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "net_points": net_pts,
            "net_rs": net_rs,
            "profit_factor": pf,
            "max_drawdown": max_dd,
            "avg_trade_rs": net_rs / n_t if n_t > 0 else 0.0,
            "trades_list": all_trs,
        })

    print(f"\n{'Configuration':60s} | {'Trades':7s} | {'Win Rate':9s} | {'Avg Win':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s}")
    print("-" * 155)
    for r in results_table:
        print(f"{r['name']:60s} | {r['trades']:7d} | {r['win_rate']:7.1f}% | +{r['avg_win']:5.2f} pt | {r['net_points']:+10.2f} | Rs {r['net_rs']:+12.2f} | {r['profit_factor']:6.3f} | Rs {r['max_drawdown']:9.2f}")

    return results_table


def main():
    spot_all = load_full_ohlc_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    all_cal = sorted(opt_map.keys())

    # Filter to 2024–2025 where Dow data is available (Dow dataset: Jan 2024 - Dec 2025)
    dow_available_days = [d for d in all_cal if d.startswith("2024") or d.startswith("2025")]

    # 1. Mandatory Smoke Test (5 Days)
    smoke_days = dow_available_days[:5]
    print("\n" + "#" * 45 + " PART 1: MANDATORY 5-DAY SMOKE TEST (JAN 2024) " + "#" * 45)
    run_dow_experiment(smoke_days, "5-Day Mandatory Smoke Test (Jan 2024)")

    # 2. Full 2-Year Dow Overlapping Test (2024–2025, 498 Days)
    print("\n" + "#" * 45 + " PART 2: FULL 2-YEAR DOW JONES OVERLAP (2024–2025) " + "#" * 45)
    run_dow_experiment(dow_available_days, "2-Year Dow Overlap (2024–2025: 498 Days)")


if __name__ == "__main__":
    main()
