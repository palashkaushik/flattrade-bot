"""
MARNI VSA ENGINE — MULTI-YEAR BACKTEST ENGINE (2020 - 2026)
============================================================
Strategy Name: Marni VSA Engine
Author: Pair Programmed with User & Antigravity
Specifications:
  1. Mirrored 1m 3-Phase Impulse Waves on NIFTY Index (Span >= 20.0 pts)
  2. Fibonacci Retracement Discount Pocket [0.618 - 0.786]
  3. Vincent Kott Volume Spike Analysis (VSA_MS) Pine Script 1-to-1 Trigger
  4. 15-Minute HTF Trend Gate (Heikin-Ashi + 11-period LinReg Plot + 15m UT Bot)
  5. Single Execution per Impulse Wave (Strict Causal State Machine)
  6. Targets: 0.786 Retest (TP) | Stop Loss: 1.155 Extension (SL) | EOD at 15:00
  7. Real-world Friction: ₹15/order Brokerage + 0.50 pt/trade Slippage
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.causal_live_parity_research import IncrementalATR

AMMU = Path(r"C:\Websites\ammu")
FUT_DIR = AMMU / "nifty_fut"
LOT_SIZE = 65
SESSION_START = 555
SESSION_END = 900
DAY_LAST = 930
CONSECUTIVE_LOSS_LIMIT = 4
SYMBOL_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")
GLOBAL_FUT_MAP = {}

def init_worker(fut_map):
    global GLOBAL_FUT_MAP
    GLOBAL_FUT_MAP = fut_map
    source.GLOBAL_CACHE = {}

class UTBotState:
    """Causal translation of UT Bot Pine logic."""
    def __init__(self, key: float = 1.0, period: int = 10):
        self.key = key
        self.atr = IncrementalATR(period)
        self.trailing_stop = 0.0
        self.previous_source = None
        self.position = 0

    def update(self, candle: Candle, source_close: float | None = None) -> str:
        source_price = candle.close if source_close is None else source_close
        atr = self.atr.update(candle.high, candle.low, candle.close)
        previous_source = self.previous_source
        self.previous_source = source_price

        if previous_source is None or atr is None or atr == 0.0:
            return "none"

        n_loss = self.key * atr
        if source_price > self.trailing_stop:
            proposed = source_price - n_loss
            self.trailing_stop = max(self.trailing_stop, proposed) if previous_source > self.trailing_stop else proposed
        else:
            proposed = source_price + n_loss
            self.trailing_stop = min(self.trailing_stop, proposed) if previous_source < self.trailing_stop else proposed

        if previous_source <= self.trailing_stop and source_price > self.trailing_stop:
            self.position = 1
        elif previous_source >= self.trailing_stop and source_price < self.trailing_stop:
            self.position = -1

        return "green" if self.position == 1 else ("red" if self.position == -1 else "none")

class StrictHTFBiasState:
    """15-Minute Heikin-Ashi + 11-period Linear Regression SMA Plot State."""
    def __init__(self, period: int = 15, linreg_len: int = 11, ut_key: float = 1.0, ut_period: int = 10):
        self.period = period
        self.linreg_len = linreg_len
        self.buf: List[Candle] = []
        self.ha_prev: Optional[Candle] = None
        self.ha_closes: deque[float] = deque(maxlen=linreg_len + 5)
        self.ut = UTBotState(key=ut_key, period=ut_period)
        self.ut_color = "none"
        self.current_linreg_plot: Optional[float] = None
        self.current_ha_close: Optional[float] = None

    def _calc_linreg(self) -> Optional[float]:
        if len(self.ha_closes) < self.linreg_len:
            return None
        closes = list(self.ha_closes)[-self.linreg_len:]
        x = list(range(self.linreg_len))
        x_mean = (self.linreg_len - 1) / 2.0
        y_mean = sum(closes) / float(self.linreg_len)
        denom = sum((xi - x_mean) ** 2 for xi in x)
        if denom == 0.0:
            return y_mean
        numer = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, closes))
        slope = numer / denom
        intercept = y_mean - slope * x_mean
        return intercept + slope * (self.linreg_len - 1)

    def update_1m(self, c: Candle) -> None:
        self.buf.append(c)
        if len(self.buf) == self.period:
            agg_open = self.buf[0].open
            agg_high = max(b.high for b in self.buf)
            agg_low = min(b.low for b in self.buf)
            agg_close = self.buf[-1].close
            self.buf.clear()

            if self.ha_prev is None:
                ha_open = (agg_open + agg_close) / 2.0
                ha_close = (agg_open + agg_high + agg_low + agg_close) / 4.0
                ha_high = agg_high
                ha_low = agg_low
            else:
                ha_close = (agg_open + agg_high + agg_low + agg_close) / 4.0
                ha_open = (self.ha_prev.open + self.ha_prev.close) / 2.0
                ha_high = max(agg_high, ha_open, ha_close)
                ha_low = min(agg_low, ha_open, ha_close)

            ha_c = Candle(ha_open, ha_high, ha_low, ha_close, minute=c.minute)
            self.ha_prev = ha_c
            self.ha_closes.append(ha_close)
            self.current_ha_close = ha_close
            self.current_linreg_plot = self._calc_linreg()
            self.ut_color = self.ut.update(ha_c)

    def snapshot(self) -> dict:
        return {
            "linreg_plot": self.current_linreg_plot,
            "ut_color": self.ut_color,
            "ha_close": self.current_ha_close,
        }

class PineVSAState:
    """Vincent Kott VSA_MS Pine Script 1-to-1 Translation."""
    def __init__(self, short_lb: int = 4, med_lb: int = 20, long_lb: int = 100):
        self.short_lb = short_lb
        self.med_lb = med_lb
        self.long_lb = long_lb
        self.history: List[float] = []

    def update(self, delta_vol: float) -> str:
        self.history.append(delta_vol)
        n = len(self.history)
        if delta_vol <= 0 or n < 2:
            return "white"
        
        h_short = max(self.history[max(0, n - self.short_lb): n])
        h_med = max(self.history[max(0, n - self.med_lb): n])
        h_long = max(self.history[max(0, n - self.long_lb): n])

        if delta_vol == h_long and n >= 20:
            return "blue"
        elif delta_vol == h_med and n >= 5:
            return "purple"
        elif delta_vol == h_short and n >= 2:
            return "red"
        else:
            return "white"

def load_fut_day(p: Path) -> List[dict]:
    df = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
    res = []
    for _, r in df.iterrows():
        t_str = str(r["time"])
        parts = t_str.split(":")
        m = int(parts[0]) * 60 + int(parts[1])
        res.append({
            "minute": m,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r.get("volume", 0.0)),
        })
    return res

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
        prev_day,
        min_span,
        include_fees,
    ) = task

    fut_path = GLOBAL_FUT_MAP.get(day)
    if fut_path is None or not fut_path.exists():
        return []

    fut_rows = load_fut_day(fut_path)
    if len(fut_rows) < 50:
        return []

    # Load option data
    current = source.cached_option(str(opt_file))
    if current is None:
        return []
    frame, groups, prefix = current

    # Find active strikes for the day
    active_keys = set()
    for r in fut_rows:
        m = r["minute"]
        if SESSION_START <= m <= DAY_LAST:
            atm = int(round(r["close"] / 50.0) * 50)
            active_keys.add(("CE", atm - 100))
            active_keys.add(("PE", atm + 100))

    bars = {}
    symbol_by_key = {}

    for sym in groups:
        m = SYMBOL_RE.match(sym)
        if not m:
            continue
        strike = int(m.group(2))
        side = m.group(3)
        key = (side, strike)
        if key not in active_keys:
            continue

        symbol_by_key[key] = sym
        r_list = option_rows(frame, groups, sym)
        bars[key] = {r["minute"]: r for r in r_list}

    # Warmup 15m HTF bias on previous day futures
    htf = StrictHTFBiasState(period=15, linreg_len=11)
    ut_1m = UTBotState(key=1.0, period=10)
    vsa = PineVSAState(short_lb=4, med_lb=20, long_lb=100)

    if prev_day:
        prev_fut_path = GLOBAL_FUT_MAP.get(prev_day)
        if prev_fut_path and prev_fut_path.exists():
            prev_rows = load_fut_day(prev_fut_path)
            for prow in prev_rows:
                pc = Candle(prow["open"], prow["high"], prow["low"], prow["close"], minute=prow["minute"])
                htf.update_1m(pc)
                ut_1m.update(pc)
                vsa.update(prow.get("volume", 0.0))

    history = []
    ce_setups = []
    pe_setups = []
    events = []

    open_high = fut_rows[0]["high"]
    open_low = fut_rows[0]["low"]

    for idx in range(len(fut_rows)):
        row = fut_rows[idx]
        m = row["minute"]
        c = Candle(row["open"], row["high"], row["low"], row["close"], minute=m)
        
        htf.update_1m(c)
        htf_snap = htf.snapshot()
        col = ut_1m.update(c)
        history.append((c, col))

        # Incremental VSA on 1m chart
        d_vol = row["volume"] if idx == 0 else max(0.0, row["volume"] - fut_rows[idx-1]["volume"])
        vsa_col = vsa.update(d_vol)

        # Check 09:15 open anchor drop setup at 09:32
        if m == 572:
            trough_low = min(history[i][0].low for i in range(len(history)))
            span = open_high - trough_low
            if span >= min_span:
                pe_setups.append({
                    "origin_high": open_high,
                    "peak_low": trough_low,
                    "span": span,
                    "triggered": False,
                })

        # 3-Phase Bullish (CE): 1 Red -> >= 5 Green -> 1 Red
        if col == "red" and len(history) >= 7:
            g_count = 0
            k = len(history) - 2
            while k >= 0 and history[k][1] == "green":
                g_count += 1
                k -= 1
            if g_count >= 5 and k >= 0 and history[k][1] == "red":
                pat = [history[i][0] for i in range(k, len(history))]
                pk = max(p.high for p in pat)
                orig = min(p.low for p in pat)
                sp = pk - orig
                if sp >= min_span:
                    ce_setups.append({
                        "peak_high": pk,
                        "origin_low": orig,
                        "span": sp,
                        "triggered": False,
                    })

        # 3-Phase Bearish (PE): 1 Green -> >= 5 Red -> 1 Green
        if col == "green" and len(history) >= 7:
            r_count = 0
            k = len(history) - 2
            while k >= 0 and history[k][1] == "red":
                r_count += 1
                k -= 1
            if r_count >= 5 and k >= 0 and history[k][1] == "green":
                pat = [history[i][0] for i in range(k, len(history))]
                orig = max(p.high for p in pat)
                pk = min(p.low for p in pat)
                sp = orig - pk
                if sp >= min_span:
                    pe_setups.append({
                        "origin_high": orig,
                        "peak_low": pk,
                        "span": sp,
                        "triggered": False,
                    })

        # Determine ATM strike
        atm = int(round(c.close / 50.0) * 50)
        ce_strike = atm - 100
        pe_strike = atm + 100
        ce_key = ("CE", ce_strike)
        pe_key = ("PE", pe_strike)

        # Check CE Touches in Pocket [0.618 - 0.786]
        valid_ce = []
        for s in ce_setups:
            if s.get("triggered", False):
                continue
            pk, orig, sp = s["peak_high"], s["origin_low"], s["span"]
            if c.low < orig - 0.25 * sp:
                continue
            f618 = pk - 0.618 * sp
            f786 = pk - 0.786 * sp
            in_zone = (c.low <= f618 + 0.5) and (c.high >= f786 - 0.5)

            linreg_p = htf_snap.get("linreg_plot")
            ut_col = htf_snap.get("ut_color")

            if in_zone and vsa_col in ("red", "purple", "blue"):
                if linreg_p is not None and c.close > linreg_p and ut_col == "green":
                    events.append({
                        "minute": m,
                        "side": "CE",
                        "strike": ce_strike,
                        "symbol": symbol_by_key.get(ce_key, f"NIFTY{ce_strike}CE"),
                        "span": sp,
                        "vsa_color": vsa_col,
                    })
                    s["triggered"] = True
                    continue
            valid_ce.append(s)
        ce_setups = valid_ce

        # Check PE Touches in Pocket [0.618 - 0.786]
        valid_pe = []
        for s in pe_setups:
            if s.get("triggered", False):
                continue
            orig, pk, sp = s["origin_high"], s["peak_low"], s["span"]
            if c.high > orig + 0.25 * sp:
                continue
            f618 = pk + 0.618 * sp
            f786 = pk + 0.786 * sp
            in_zone = (c.high >= f618 - 0.5) and (c.low <= f786 + 0.5)

            linreg_p = htf_snap.get("linreg_plot")
            ut_col = htf_snap.get("ut_color")

            if in_zone and vsa_col in ("red", "purple", "blue"):
                if linreg_p is not None and c.close < linreg_p and (ut_col == "red" or m <= 600):
                    events.append({
                        "minute": m,
                        "side": "PE",
                        "strike": pe_strike,
                        "symbol": symbol_by_key.get(pe_key, f"NIFTY{pe_strike}PE"),
                        "span": sp,
                        "vsa_color": vsa_col,
                    })
                    s["triggered"] = True
                    continue
            valid_pe.append(s)
        pe_setups = valid_pe

    # Simulate trades
    trades = []
    consecutive_losses = 0

    for ev in events:
        m = ev["minute"]
        if m < SESSION_START or m >= SESSION_END:
            continue
        if consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            continue

        side = ev["side"]
        strike = ev["strike"]
        key = (side, strike)

        if key not in bars or m not in bars[key]:
            continue

        opt_entry = bars[key][m]["close"]
        opt_span = ev["span"] * 0.5
        tp_price = opt_entry + (opt_span * 0.786)
        sl_price = opt_entry - (opt_span * 0.369)

        exit_fill, exit_m, rsn = None, None, ""
        for bar_m in range(m + 1, DAY_LAST + 1):
            if bar_m not in bars[key]:
                continue
            b = bars[key][bar_m]
            h, l, cl = b["high"], b["low"], b["close"]

            if l <= sl_price and h >= tp_price:
                exit_fill, exit_m, rsn = sl_price, bar_m, "SL"
                break
            elif h >= tp_price:
                exit_fill, exit_m, rsn = tp_price, bar_m, "TP"
                break
            elif l <= sl_price:
                exit_fill, exit_m, rsn = sl_price, bar_m, "SL"
                break
            elif bar_m >= SESSION_END:
                exit_fill, exit_m, rsn = cl, bar_m, "EOD"
                break

        if exit_fill is not None:
            slip = SLIPPAGE_PTS if include_fees else 0.0
            entry_f = opt_entry + slip
            exit_f = exit_fill - slip
            pts = round(exit_f - entry_f, 2)
            fee = trade_cost(entry_f, exit_f, BROKERAGE_PER_ORDER) if include_fees else 0.0
            rs_net = round(pts * LOT_SIZE - fee, 2)

            if rs_net > 0:
                consecutive_losses = 0
            else:
                consecutive_losses += 1

            trades.append({
                "date": day,
                "entry_min": m,
                "exit_min": exit_m,
                "side": side,
                "strike": strike,
                "symbol": ev["symbol"],
                "span": ev["span"],
                "vsa_color": ev["vsa_color"],
                "entry": entry_f,
                "exit": exit_f,
                "tp": tp_price,
                "sl": sl_price,
                "reason": rsn,
                "points": pts,
                "fee": fee,
                "rs_net": rs_net,
            })

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
    parser = argparse.ArgumentParser(description="Marni VSA Multi-Year Backtest Engine")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-10-31")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Run 5-day smoke test only")
    parser.add_argument("--no-fees", action="store_true", help="Disable brokerage & slippage fees")
    parser.add_argument("--min-span", type=float, default=20.0, help="Minimum impulse span in points (default: 20.0)")
    parser.add_argument("--output", default="artifacts/f6_hybrid/marni_vsa_7y_results.json")
    args = parser.parse_args()

    include_fees = not args.no_fees
    print(f"\n{'='*100}")
    print(f"MARNI VSA ENGINE — MULTI-YEAR BACKTEST (2020 - 2024/2026)")
    print(f"Date Range: {args.start} to {args.end} | Min Span: {args.min_span} pts | Fees: {'YES' if include_fees else 'NO'}")
    print(f"{'='*100}")

    # Build futures map
    fut_map = {}
    for p in FUT_DIR.rglob("*"):
        if p.is_file() and p.name.startswith("nifty_fut_"):
            m = re.search(r"nifty_fut_(\d{2})_(\d{2})_(\d{4})", p.name)
            if m:
                d, mo, y = m.group(1), m.group(2), m.group(3)
                date_str = f"{y}-{mo}-{d}"
                fut_map[date_str] = p

    opt_map = source.option_day_files(args.start, args.end)
    all_days = sorted(set(opt_map.keys()) & set(fut_map.keys()))
    days = all_days[:5] if args.smoke else all_days

    print(f"Running on {len(days)} trading days {'(SMOKE TEST — 5 DAYS ONLY)' if args.smoke else ''}...")

    previous = {day: max((c for c in all_days if c < day), default="") for day in days}
    tasks = [
        (
            day,
            opt_map[day],
            previous[day],
            args.min_span,
            include_fees,
        )
        for day in days
    ]

    t0 = time.time()
    all_trades = []

    if args.smoke or args.workers == 1:
        init_worker(fut_map)
        for task in tasks:
            day_trades = process_day(task)
            all_trades.extend(day_trades)
    else:
        with Pool(processes=args.workers, initializer=init_worker, initargs=(fut_map,)) as pool:
            for day_trades in pool.imap_unordered(process_day, tasks, chunksize=1):
                all_trades.extend(day_trades)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.2f}s ({len(days)/elapsed:.1f} days/sec).")

    # Overall stats
    st = compute_stats(all_trades, len(days))

    print(f"\n{'='*100}")
    print(f"OVERALL PERFORMANCE METRICS ({args.start[:4]} - {args.end[:4]})")
    print(f"{'='*100}")
    print(f"Total Trading Days:       {len(days):,d}")
    print(f"Total Qualified Trades:   {st['trades']:,d} ({st['avg_trades_per_day']:.2f} trades/day)")
    print(f"Winning Trades:           {st['wins']:,d} / {st['trades']:,d} ({st['win_rate']:.2f}% Win Rate)")
    print(f"Total Option Points:      {st['net_points']:+,.2f} pts")
    print(f"Profit Factor:            {st['profit_factor']:.2f}")
    print(f"Brokerage & Slippage:     Rs {st['fees_rs']:,.2f}")
    print(f"Max Drawdown:             Rs {st['max_drawdown_rs']:,.2f}")
    print(f"NET REALIZED PROFIT:      Rs {st['net_rs']:+,.2f}")
    print(f"{'='*100}\n")

    # Year by year breakdown
    by_year = defaultdict(list)
    for t in all_trades:
        by_year[t["date"][:4]].append(t)

    print(f"{'Year':6s} | {'Days':6s} | {'Trades':8s} | {'Win Rate':9s} | {'Points':12s} | {'Profit Factor':14s} | {'Max DD (Rs)':14s} | {'Net Realized P&L (Rs)':22s}")
    print("-" * 105)
    for y in sorted(by_year.keys()):
        y_trades = by_year[y]
        y_days = len(set(t["date"] for t in y_trades))
        yst = compute_stats(y_trades, y_days)
        print(f"{y:6s} | {y_days:6d} | {yst['trades']:8d} | {yst['win_rate']:8.1f}% | {yst['net_points']:+11.2f}p | {yst['profit_factor']:13.2f} | Rs {yst['max_drawdown_rs']:11,.2f} | Rs {yst['net_rs']:+19,.2f}")

    print("-" * 105)

    # Save detailed JSON report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"stats": st, "trades": all_trades}, f, indent=2)
    print(f"\nDetailed JSON report saved to: {out_path}")

if __name__ == "__main__":
    main()
