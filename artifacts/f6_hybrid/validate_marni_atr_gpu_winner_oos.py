"""
MARNI ATR DYNAMIC VOLATILITY ENGINE — OUT-OF-SAMPLE VALIDATION OF GPU WINNER
============================================================================
Winner Configuration from 100-Trial GPU Optuna Study:
  - atr_period: 16
  - atr_sl_mult: 1.2
  - atr_tp_mult: 5.5 (R:R = 4.58 : 1)
  - s1_spec: (7, 4)
  - s4_spec: (55, 8)
  - s4_ob: 77.5
  - s1_os: 25.0
Dataset:
  - In-Sample (Training): 2020-01-01 to 2023-12-31 (4 Years)
  - Out-of-Sample (Untouched Test): 2024-01-01 to 2026-05-05 (2.5 Years)
Friction: ₹15 Brokerage + 0.50 pt Slippage (1.0 pt round-trip) + STT + GST + SEBI
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from flattrade_bot.indicators.divergence import DivergenceEngine

LOT_SIZE = 65
SESSION_START = 560
SESSION_END = 900
DAY_LAST = 930
DAILY_MAX_LOSS_RS = -2000.0
DAILY_MAX_LOSS_PTS = DAILY_MAX_LOSS_RS / LOT_SIZE
CONSECUTIVE_LOSS_LIMIT = 4
SYMBOL_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")
GLOBAL_SPOT = {}

def init_worker(spot):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot
    source.GLOBAL_CACHE = {}

class IncrementalATR:
    def __init__(self, period=16):
        self.period = period
        self._buf = deque(maxlen=period)
        self.atr = None
        self.prev_close = None
        self._n = 0

    def update(self, h, l, c):
        tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close)) if self.prev_close else h - l
        self._buf.append(tr)
        self._n += 1
        self.prev_close = c
        if self._n < self.period:
            self.atr = None
        elif self._n == self.period:
            self.atr = sum(self._buf) / self.period
        else:
            self.atr = (self.atr * (self.period - 1) + tr) / self.period
        return self.atr

class TFTracker:
    def __init__(self, lb, tf_sl, tf_tp, s1_spec, s2_spec, s3_spec, s4_spec, atr_period, s4_ob, s4_os, s1_os):
        self.lb = lb
        self.tf_sl = tf_sl
        self.tf_tp = tf_tp
        self.s1 = IncrementalStochastic(*s1_spec)
        self.s2 = IncrementalStochastic(*s2_spec)
        self.s3 = IncrementalStochastic(*s3_spec)
        self.s4 = IncrementalStochastic(*s4_spec)
        self.div = DivergenceEngine()
        self.atr = IncrementalATR(atr_period)
        self.hist = []
        self.setup = False
        self.stype = ""
        self.prev_s1 = None
        self.s4_emb = 0
        self.s4_ob = s4_ob
        self.s4_os = s4_os
        self.s1_os = s1_os

    def push(self, c: Candle):
        self.hist.append(c)
        if len(self.hist) > 60:
            self.hist.pop(0)
        s1 = self.s1.push(c.high, c.low, c.close)
        s2 = self.s2.push(c.high, c.low, c.close)
        s3 = self.s3.push(c.high, c.low, c.close)
        s4 = self.s4.push(c.high, c.low, c.close)
        atr_val = self.atr.update(c.high, c.low, c.close)

        if s4 is not None:
            self.s4_emb = self.s4_emb + 1 if s4 <= self.s4_os else 0
        emb = self.s4_emb > 25
        self.div.update(c.close, s1)
        bull_div = self.div.has_bullish_trough_divergence()

        is_flag = s4 is not None and s1 is not None and s4 >= self.s4_ob and s1 <= self.s1_os
        is_super = all(v is not None and v <= self.s4_os for v in (s1, s2, s3, s4))
        if (is_flag or is_super) and bull_div:
            self.setup = True
            self.stype = "super" if is_super else "flag"

        is_rev = emb and self.stype == "super"
        triggered = False

        if self.setup and len(self.hist) >= 2:
            if BullishPinBarDetector.check_vicinity_breakout(self.hist, self.lb):
                triggered = True
                self.setup = False

        self.prev_s1 = s1
        return triggered, is_rev, self.stype, c.close, atr_val

class MTFTracker:
    def __init__(self, s1_spec, s2_spec, s3_spec, s4_spec, atr_period, s4_ob, s4_os, s1_os, tf_specs):
        self.trackers = {
            tf: TFTracker(spec[1], spec[2], spec[3], s1_spec, s2_spec, s3_spec, s4_spec, atr_period, s4_ob, s4_os, s1_os)
            for tf, spec in tf_specs.items()
        }
        self.bufs = {tf: [] for tf in tf_specs}
        self.tf_specs = tf_specs

    def push_1m(self, c1m: Candle):
        out = []
        for tf, spec in self.tf_specs.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf]) == spec[0]:
                buf = self.bufs[tf]
                ctf = Candle(
                    open=buf[0].open,
                    high=max(x.high for x in buf),
                    low=min(x.low for x in buf),
                    close=buf[-1].close,
                    minute=buf[-1].minute,
                )
                self.bufs[tf] = []
                trig, is_rev, stype, px, atr_val = self.trackers[tf].push(ctf)
                if trig:
                    out.append((tf, is_rev, stype, px, atr_val, spec[2], spec[3]))
        return out

def option_rows(frame, groups, symbol):
    indexes = groups.get(symbol)
    if indexes is None:
        return []
    rows = frame.iloc[indexes].sort_values("time")
    res = []
    for _, row in rows.iterrows():
        t_str = str(row["time"])
        parts = t_str.split(":")
        m = int(parts[0]) * 60 + int(parts[1])
        res.append({
            "time": t_str,
            "minute": m,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        })
    return res

def process_day(task):
    (
        day,
        opt_file,
        prev_opt_file,
        params,
        include_fees,
    ) = task

    spot = GLOBAL_SPOT.get(day)
    if spot is None or len(spot["min"]) < 50:
        return []

    current = source.cached_option(str(opt_file))
    if current is None:
        return []
    frame, groups, prefix = current

    sp0 = float(spot["close"][0])
    atm0 = int(round(sp0 / 50.0) * 50)
    target_strikes = set(range(atm0 - 250, atm0 + 300, 50))

    s1_spec = (params["s1_k"], params["s1_d"])
    s2_spec = (12, 3)
    s3_spec = (35, 4)
    s4_spec = (params["s4_k"], params["s4_d"])
    atr_period = params["atr_period"]
    atr_sl_mult = params["atr_sl_mult"]
    atr_tp_mult = params["atr_tp_mult"]
    s4_ob = params["s4_ob"]
    s4_os = 20.5
    s1_os = params["s1_os"]

    tf_specs = {
        "1m": (1, 10, 6.0, 30.0),
        "2m": (2, 5, 10.0, 15.0),
        "3m": (3, 4, 8.0, 25.0),
        "5m": (5, 3, 10.0, 35.0),
    }

    # Warm up trackers
    trk = {}
    if prev_opt_file:
        prev_res = source.cached_option(str(prev_opt_file))
        if prev_res:
            p_frame, p_groups, _ = prev_res
            for sym in p_groups:
                m = SYMBOL_RE.match(sym)
                if not m or int(m.group(2)) not in target_strikes:
                    continue
                p_rows = option_rows(p_frame, p_groups, sym)
                trk[sym] = MTFTracker(s1_spec, s2_spec, s3_spec, s4_spec, atr_period, s4_ob, s4_os, s1_os, tf_specs)
                for r in p_rows:
                    trk[sym].push_1m(Candle(r["open"], r["high"], r["low"], r["close"], minute=r["minute"]))

    bars = {}
    for sym in groups:
        m = SYMBOL_RE.match(sym)
        if not m or int(m.group(2)) not in target_strikes:
            continue
        if sym not in trk:
            trk[sym] = MTFTracker(s1_spec, s2_spec, s3_spec, s4_spec, atr_period, s4_ob, s4_os, s1_os, tf_specs)
        r_list = option_rows(frame, groups, sym)
        bars[sym] = {r["minute"]: r for r in r_list}

    pmtrig = defaultdict(list)
    for m in range(SESSION_START, DAY_LAST + 1):
        for sym, b_map in bars.items():
            if m not in b_map:
                continue
            r = b_map[m]
            c = Candle(r["open"], r["high"], r["low"], r["close"], minute=m)
            trigs = trk[sym].push_1m(c)
            if trigs and m >= SESSION_START:
                match = SYMBOL_RE.match(sym)
                if match:
                    stk = int(match.group(2))
                    side = match.group(3)
                    for (tf, is_rev, stype, px, atr_val, sl_pts, tp_pts) in trigs:
                        pmtrig[m].append((side, stk, sym, px, is_rev, tf, sl_pts, tp_pts, atr_val))

    def get_latest_spot(minute):
        idx = 0
        for i, m_val in enumerate(spot["min"]):
            if int(m_val) <= minute:
                idx = i
            else:
                break
        return float(spot["close"][idx])

    pos = None
    trades = []
    dpnl = 0.0
    closs = 0
    shut = False

    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None:
            sym = pos["symbol"]
            b_map = bars.get(sym, {})
            if minute in b_map:
                b = b_map[minute]
                o, h, l, c = b["open"], b["high"], b["low"], b["close"]
                pos["last_px"] = c
                pos["duration_min"] += 1

                # Daily max loss check
                unrealized_rs = (c - pos["entry"]) * LOT_SIZE
                if dpnl * LOT_SIZE + unrealized_rs <= DAILY_MAX_LOSS_RS:
                    pts = round(c - pos["entry"], 2)
                    fee = trade_cost(pos["entry"], c, BROKERAGE_PER_ORDER) if include_fees else 0.0
                    rs_net = round(pts * LOT_SIZE - fee, 2)
                    trades.append({
                        "date": day,
                        "entry_min": pos["entry_min"],
                        "exit_min": minute,
                        "side": pos["side"],
                        "symbol": pos["symbol"],
                        "entry": pos["entry"],
                        "exit": c,
                        "points": pts,
                        "fee": fee,
                        "rs_net": rs_net,
                        "reason": "SHUTDOWN_LOSS",
                        "duration_min": pos["duration_min"],
                        "is_rev": pos["is_rev"],
                        "tf": pos["tf"],
                    })
                    dpnl += pts
                    pos = None
                    shut = True
                    continue

                ex, rsn = None, ""
                if h >= pos["tgt"] and l <= pos["sl"]:
                    ex, rsn = pos["sl"], "SL"
                elif h >= pos["tgt"]:
                    ex, rsn = pos["tgt"], "TP"
                elif l <= pos["sl"]:
                    ex, rsn = pos["sl"], "SL"

                if ex is None:
                    t1 = trk.get(pos["symbol"])
                    if t1:
                        t1m = t1.trackers["1m"]
                        t1m.div.update(c, t1m.prev_s1)
                        if t1m.div.has_bearish_peak_divergence():
                            ex, rsn = c, "BEARISH_PEAK_REVERSAL"

                if ex is not None:
                    slip = SLIPPAGE_PTS if include_fees else 0.0
                    entry_f = pos["entry"] + slip
                    exit_f = ex - slip
                    pts = round(exit_f - entry_f, 2)
                    fee = trade_cost(entry_f, exit_f, BROKERAGE_PER_ORDER) if include_fees else 0.0
                    rs_net = round(pts * LOT_SIZE - fee, 2)

                    trades.append({
                        "date": day,
                        "entry_min": pos["entry_min"],
                        "exit_min": minute,
                        "side": pos["side"],
                        "symbol": pos["symbol"],
                        "entry": entry_f,
                        "exit": exit_f,
                        "points": pts,
                        "fee": fee,
                        "rs_net": rs_net,
                        "reason": rsn,
                        "duration_min": pos["duration_min"],
                        "is_rev": pos["is_rev"],
                        "tf": pos["tf"],
                    })
                    dpnl += pts
                    closs = closs + 1 if pts <= 0 else 0
                    if closs >= CONSECUTIVE_LOSS_LIMIT or dpnl <= DAILY_MAX_LOSS_PTS:
                        shut = True
                    pos = None

            if minute >= SESSION_END and pos is not None:
                slip = SLIPPAGE_PTS if include_fees else 0.0
                entry_f = pos["entry"] + slip
                exit_f = pos["last_px"] - slip
                pts = round(exit_f - entry_f, 2)
                fee = trade_cost(entry_f, exit_f, BROKERAGE_PER_ORDER) if include_fees else 0.0
                rs_net = round(pts * LOT_SIZE - fee, 2)

                trades.append({
                    "date": day,
                    "entry_min": pos["entry_min"],
                    "exit_min": minute,
                    "side": pos["side"],
                    "symbol": pos["symbol"],
                    "entry": entry_f,
                    "exit": exit_f,
                    "points": pts,
                    "fee": fee,
                    "rs_net": rs_net,
                    "reason": "EOD",
                    "duration_min": pos["duration_min"],
                    "is_rev": pos["is_rev"],
                    "tf": pos["tf"],
                })
                dpnl += pts
                pos = None
                break

        if pos is not None or shut or minute >= SESSION_END:
            continue

        for (sig_side, sig_stk, sig_sym, c_px, is_rev, tf, sl_pts, tp_pts, atr_val) in pmtrig.get(minute, []):
            sp_cur = get_latest_spot(minute)
            atm_cur = int(round(sp_cur / 50.0) * 50)
            target_stk = atm_cur - 100 if sig_side == "CE" else atm_cur + 100

            if target_stk == sig_stk and pos is None:
                asym = sig_sym
                as2 = sig_side
                if is_rev:
                    as2 = "PE" if sig_side == "CE" else "CE"
                    rev_stk = atm_cur - 100 if as2 == "CE" else atm_cur + 100
                    asym = f"{prefix}{rev_stk}{as2}"

                if asym in bars and minute in bars[asym]:
                    b_bar = bars[asym][minute]
                    ep = b_bar["close"]
                    if atr_val and atr_val > 0.5:
                        sl_use = atr_val * atr_sl_mult
                        tp_use = atr_val * atr_tp_mult
                    else:
                        sl_use = sl_pts
                        tp_use = tp_pts

                    pos = {
                        "side": as2,
                        "symbol": asym,
                        "entry": ep,
                        "sl": ep - sl_use,
                        "tgt": ep + tp_use,
                        "entry_min": minute,
                        "last_px": ep,
                        "duration_min": 0,
                        "is_rev": is_rev,
                        "tf": tf,
                    }
                    break

    return trades

def compute_stats(trades: list[dict], days_count: int) -> dict:
    wins = [t for t in trades if t["rs_net"] > 0]
    losses = [t for t in trades if t["rs_net"] <= 0]
    loss_total = abs(sum(t["rs_net"] for t in losses))
    win_total = sum(t["rs_net"] for t in wins)
    net_rs = sum(t["rs_net"] for t in trades)
    net_pts = sum(t["points"] for t in trades)
    wr = round(len(wins) / len(trades) * 100, 2) if trades else 0.0
    pf = round(win_total / loss_total, 4) if loss_total else (float("inf") if win_total else 0.0)
    fees = round(sum(t["fee"] for t in trades), 2)
    avg_trades = round(len(trades) / days_count, 3) if days_count else 0.0

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in sorted(trades, key=lambda x: (x["date"], x["entry_min"])):
        equity += t["rs_net"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "net_rs": round(net_rs, 2),
        "net_points": round(net_pts, 2),
        "profit_factor": pf,
        "max_drawdown_rs": round(max_dd, 2),
        "fees_rs": fees,
        "avg_trades_per_day": avg_trades,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    params = {
        "atr_period": 16,
        "atr_sl_mult": 1.2,
        "atr_tp_mult": 5.5,
        "s1_k": 7, "s1_d": 4,
        "s4_k": 55, "s4_d": 8,
        "s4_ob": 77.5, "s1_os": 25.0,
    }

    print(f"\n{'='*120}")
    print("MARNI ATR OPTUNA GPU WINNER — OUT-OF-SAMPLE MULTI-YEAR VALIDATION (2020 - 2026)")
    print(f"Optimal Parameters: ATR Period=16, SL=x1.2, TP=x5.5 (R:R = 4.58:1), S1=(7,4), S4=(55,8)")
    print(f"{'='*120}")

    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    previous = {day: max((c for c in all_days if c < day), default="") for day in all_days}

    # Split into In-Sample (2020-2023) and Out-of-Sample (2024-2026)
    is_days = [d for d in all_days if d < "2024-01-01"]
    oos_days = [d for d in all_days if d >= "2024-01-01"]

    tasks_all = [(d, opt_map[d], opt_map.get(previous[d], ""), params, True) for d in all_days]

    all_trades = []
    with Pool(processes=args.workers, initializer=init_worker, initargs=(spot_all,)) as pool:
        for res in pool.imap_unordered(process_day, tasks_all, chunksize=1):
            all_trades.extend(res)

    is_trades = [t for t in all_trades if t["date"] < "2024-01-01"]
    oos_trades = [t for t in all_trades if t["date"] >= "2024-01-01"]

    st_total = compute_stats(all_trades, len(all_days))
    st_is = compute_stats(is_trades, len(is_days))
    st_oos = compute_stats(oos_trades, len(oos_days))

    print(f"\n{'Segment':20s} | {'Days':6s} | {'Trades':8s} | {'Win Rate':9s} | {'Profit Factor':14s} | {'Max DD (Rs)':14s} | {'Net Realized P&L (Rs)':22s}")
    print("-" * 105)
    print(f"{'In-Sample (2020-23)':20s} | {len(is_days):6d} | {st_is['trades']:8d} | {st_is['win_rate']:8.1f}% | {st_is['profit_factor']:13.2f} | Rs {st_is['max_drawdown_rs']:11,.2f} | Rs {st_is['net_rs']:+19,.2f}")
    print(f"{'Out-of-Sample (24-26)':20s} | {len(oos_days):6d} | {st_oos['trades']:8d} | {st_oos['win_rate']:8.1f}% | {st_oos['profit_factor']:13.2f} | Rs {st_oos['max_drawdown_rs']:11,.2f} | Rs {st_oos['net_rs']:+19,.2f}")
    print("-" * 105)
    print(f"{'7-YEAR FULL TOTAL':20s} | {len(all_days):6d} | {st_total['trades']:8d} | {st_total['win_rate']:8.1f}% | {st_total['profit_factor']:13.2f} | Rs {st_total['max_drawdown_rs']:11,.2f} | Rs {st_total['net_rs']:+19,.2f}")
    print("-" * 105)

    # Yearly breakdown
    by_year = defaultdict(list)
    for t in all_trades:
        by_year[t["date"][:4]].append(t)

    print(f"\nYEAR-BY-YEAR PERFORMANCE BREAKDOWN:")
    print(f"{'Year':6s} | {'Trades':8s} | {'Win Rate':9s} | {'Points':12s} | {'Profit Factor':14s} | {'Max DD (Rs)':14s} | {'Net Realized P&L (Rs)':22s}")
    print("-" * 95)
    for y in sorted(by_year.keys()):
        y_trades = by_year[y]
        yst = compute_stats(y_trades, len(set(t["date"] for t in y_trades)))
        print(f"{y:6s} | {yst['trades']:8d} | {yst['win_rate']:8.1f}% | {yst['net_points']:+11.2f}p | {yst['profit_factor']:13.2f} | Rs {yst['max_drawdown_rs']:11,.2f} | Rs {yst['net_rs']:+19,.2f}")
    print("-" * 95)

if __name__ == "__main__":
    main()
