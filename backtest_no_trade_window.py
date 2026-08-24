"""Backtest: No-Trade Window 11:30 AM - 1:30 PM IST

Tests these 4 strategies:
  #1  Trailing SL (+5/+10), Fixed cap  | Pin bar | S1=(9,3)   | MaxProfit=+Rs1,950
  #2  ATR x2.0/x4.0, Unlimited        | Pin bar | S1=(9,3)   | MaxProfit=UNLIMITED
  #3  S1=(12,3) + ATR x2.0/x4.0       | Pin bar | S1=(12,3)  | MaxProfit=+Rs1,950
  #4  S1=(7,3)  + ATR x2.0/x4.0       | Pin bar | S1=(7,3)   | MaxProfit=+Rs1,950

No-Trade Rule: New entries BLOCKED from 11:30 AM - 1:30 PM IST.
               Existing positions continue (SL/TP still managed normally).
Daily Max Loss  = Rs 2,000 (all strategies)
"""

import re
import time
from pathlib import Path
from collections import deque
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from backtest_5y_optimized import (
    load_spot, option_files, SYM_RE, to_minutes,
    latest_spot, summarize, print_yearly_breakdown,
)
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from flattrade_bot.indicators.divergence import DivergenceEngine

# ── Global constants ────────────────────────────────────────────────
LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_MAX_LOSS_RS  = -2000.0
DAILY_MAX_LOSS_PTS = DAILY_MAX_LOSS_RS / LOT_SIZE     # -30.77 pts
CONSECUTIVE_LOSS_LIMIT = 6
ATR_PERIOD = 14
TRAIL_STEP_PTS   = 10.0
TRAIL_AMOUNT_PTS = 5.0

# ── No-trade window ─────────────────────────────────────────────────
NO_TRADE_START = 690   # 11:30 AM IST (11*60 + 30)
NO_TRADE_END   = 810   # 01:30 PM IST (13*60 + 30)

# ── TF specs: (bar_size_1m, lookback, default_sl, default_tp) ───────
TF_SPECS = {
    "1m": (1, 10, 6.0,  30.0),
    "2m": (2,  5, 10.0, 15.0),
    "3m": (3,  4, 8.0,  25.0),
    "5m": (5,  3, 10.0, 35.0),
}

GLOBAL_SPOT   = {}
GLOBAL_CONFIG = {}

def init_worker_local(sd, cfg):
    global GLOBAL_SPOT, GLOBAL_CONFIG
    GLOBAL_SPOT = sd
    GLOBAL_CONFIG = cfg


# ── Indicators ───────────────────────────────────────────────────────
class IncrementalATR:
    def __init__(self, period=14):
        self.period = period; self._buf = deque(maxlen=period)
        self.atr = None; self.prev_close = None; self._n = 0
    def update(self, h, l, c):
        tr = max(h-l, abs(h-self.prev_close), abs(l-self.prev_close)) if self.prev_close else h-l
        self._buf.append(tr); self._n += 1; self.prev_close = c
        if self._n < self.period:    self.atr = None
        elif self._n == self.period: self.atr = sum(self._buf) / self.period
        else:                        self.atr = (self.atr*(self.period-1)+tr) / self.period
        return self.atr


class TFTracker:
    """Single-timeframe pin bar trigger with custom stochastic params."""
    def __init__(self, lb, tf_sl, tf_tp, s1_k, s1_d, s2_k, s2_d, s3_k, s3_d, s4_k, s4_d):
        self.lb = lb; self.tf_sl = tf_sl; self.tf_tp = tf_tp
        self.s1 = IncrementalStochastic(s1_k, s1_d)
        self.s2 = IncrementalStochastic(s2_k, s2_d)
        self.s3 = IncrementalStochastic(s3_k, s3_d)
        self.s4 = IncrementalStochastic(s4_k, s4_d)
        self.div = DivergenceEngine()
        self.atr = IncrementalATR(ATR_PERIOD)
        self.hist = []; self.setup = False; self.stype = ""; self.s4_emb = 0

    def push(self, c: Candle):
        self.hist.append(c)
        if len(self.hist) > 60: self.hist.pop(0)
        s1 = self.s1.push(c.high, c.low, c.close)
        s2 = self.s2.push(c.high, c.low, c.close)
        s3 = self.s3.push(c.high, c.low, c.close)
        s4 = self.s4.push(c.high, c.low, c.close)
        atr_val = self.atr.update(c.high, c.low, c.close)
        if s4 is not None:
            self.s4_emb = self.s4_emb + 1 if s4 <= 20.0 else 0
        emb = self.s4_emb > 25
        self.div.update(c.close, s1)
        bull_div = self.div.has_bullish_trough_divergence()
        is_flag  = s4 is not None and s1 is not None and s4 >= 79.5 and s1 <= 20.5
        is_super = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
        if (is_flag or is_super) and bull_div:
            self.setup = True; self.stype = "super" if is_super else "flag"
        is_rev = emb and self.stype == "super"
        triggered = False
        if self.setup and len(self.hist) >= 2:
            if BullishPinBarDetector.check_vicinity_breakout(self.hist, self.lb):
                triggered = True; self.setup = False
        return triggered, is_rev, self.stype, c.close, atr_val


class MTFTracker:
    """Multi-timeframe tracker using specified stochastic params."""
    def __init__(self, s1_spec, s2_spec, s3_spec, s4_spec):
        k1, d1 = s1_spec; k2, d2 = s2_spec; k3, d3 = s3_spec; k4, d4 = s4_spec
        self.trackers = {
            tf: TFTracker(spec[1], spec[2], spec[3], k1,d1, k2,d2, k3,d3, k4,d4)
            for tf, spec in TF_SPECS.items()
        }
        self.bufs = {tf: [] for tf in TF_SPECS}

    def push_1m(self, c1m: Candle):
        out = []
        for tf, spec in TF_SPECS.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf]) == spec[0]:
                buf = self.bufs[tf]; self.bufs[tf] = []
                ctf = Candle(open=buf[0].open, high=max(x.high for x in buf),
                             low=min(x.low for x in buf), close=buf[-1].close, minute=buf[-1].minute)
                trig, is_rev, stype, px, atr_val = self.trackers[tf].push(ctf)
                if trig:
                    out.append((tf, is_rev, stype, px, atr_val, spec[2], spec[3]))
        return out


# ── Per-day simulation ───────────────────────────────────────────────
def process_day(args):
    day, fpath, fprev = args
    cfg   = GLOBAL_CONFIG
    spot  = GLOBAL_SPOT.get(day)
    mode  = cfg["mode"]
    atr_sl_mult = cfg["atr_sl"]
    atr_tp_mult = cfg["atr_tp"]
    daily_max_profit_pts = cfg["max_profit_pts"]
    s1_spec = cfg["s1"]; s2_spec = cfg["s2"]; s3_spec = cfg["s3"]; s4_spec = cfg["s4"]

    if spot is None or not fpath: return []
    fp = Path(fpath)
    if not fp.exists(): return []
    sp0 = latest_spot(spot, 555) or latest_spot(spot, 560)
    if sp0 is None: return []
    atm0 = int(round(sp0 / 50) * 50)
    target_strikes = set(range(atm0-250, atm0+300, 50))
    try:
        dfc = pd.read_csv(fp, usecols=["time","symbol","open","high","low","close"], engine="c")
    except: return []
    if dfc.empty: return []
    fsym = dfc["symbol"].iloc[0]; mm = SYM_RE.match(fsym)
    if not mm: return []
    prefix = mm.group(1)
    dfc["min"] = np.array([to_minutes(t) for t in dfc["time"]])
    gc = {sym: g for sym, g in dfc.groupby("symbol")
          if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}

    # Warm up on previous day
    gp = {}
    if fprev and Path(fprev).exists():
        try:
            dfp = pd.read_csv(fprev, usecols=["time","symbol","open","high","low","close"], engine="c")
            if not dfp.empty:
                dfp["min"] = np.array([to_minutes(t) for t in dfp["time"]])
                gp = {sym: g for sym, g in dfp.groupby("symbol")
                      if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}
        except: pass

    # Build multi-TF tracker
    trk = {}
    for sym, g in gp.items():
        trk[sym] = MTFTracker(s1_spec, s2_spec, s3_spec, s4_spec)
        mn, op, hi, lo, cl = (g[c].to_numpy() for c in ["min","open","high","low","close"])
        for i in range(len(mn)):
            trk[sym].push_1m(Candle(open=op[i], high=hi[i], low=lo[i], close=cl[i], minute=mn[i]))

    # Collect trigger events and build slices for today
    pmtrig = {}; slices = {}
    for sym, g in gc.items():
        if sym not in trk:
            trk[sym] = MTFTracker(s1_spec, s2_spec, s3_spec, s4_spec)
        t = trk[sym]
        mn, op, hi, lo, cl = (g[c].to_numpy() for c in ["min","open","high","low","close"])
        slices[sym] = {"min": mn, "open": op, "high": hi, "low": lo, "close": cl}
        mm2 = SYM_RE.match(sym)
        if not mm2: continue
        sv, side = int(mm2.group(2)), mm2.group(3)
        for i in range(len(mn)):
            m = mn[i]
            for item in t.push_1m(Candle(open=op[i], high=hi[i], low=lo[i], close=cl[i], minute=m)):
                pmtrig.setdefault(m, []).append((side, sv, sym) + item)

    def bslice(sl, m):
        idx = np.searchsorted(sl["min"], m)
        if idx < len(sl["min"]) and sl["min"][idx] == m:
            return sl["open"][idx], sl["high"][idx], sl["low"][idx], sl["close"][idx]
        return None

    def ainfo(side, m):
        spx = latest_spot(spot, m)
        if spx is None: return None
        atm = int(round(spx / 50) * 50)
        stk = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        sym = f"{prefix}{stk}{side}"
        sl  = slices.get(sym)
        return (sym, sl, stk) if sl is not None else None

    trades = []; pos = None; dpnl = 0.0; closs = 0; shut = False

    for minute in range(SESSION_START, DAY_LAST+1):
        in_no_trade = NO_TRADE_START <= minute <= NO_TRADE_END

        # ── Manage open position ──────────────────────────────────
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c); pos["duration_min"] += 1

                # Update trailing stop
                if mode == "trailing":
                    gain = c - pos["entry"]
                    steps = int(gain / TRAIL_STEP_PTS)
                    if steps > pos.get("trail_steps", 0):
                        pos["sl"] += (steps - pos["trail_steps"]) * TRAIL_AMOUNT_PTS
                        pos["trail_steps"] = steps

                # Emergency shutdown check
                if dpnl * LOT_SIZE + (c - pos["entry"]) * LOT_SIZE <= DAILY_MAX_LOSS_RS:
                    pts = round(c - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"],
                        "exit_min": minute, "side": pos["side"], "symbol": pos["symbol"],
                        "entry": pos["entry"], "exit": c, "pts": pts,
                        "rs": round(pts*LOT_SIZE), "reason": "SHUTDOWN_LOSS",
                        "duration_min": pos["duration_min"], "tf": pos["tf"]})
                    dpnl += pts; pos = None; shut = True; continue

                # Check SL/TP
                ex = None; rsn = ""
                has_tgt = pos.get("tgt") is not None
                if has_tgt and h >= pos["tgt"] and l <= pos["sl"]: ex, rsn = pos["sl"], "SL"
                elif has_tgt and h >= pos["tgt"]:                  ex, rsn = pos["tgt"], "TP"
                elif l <= pos["sl"]:                               ex, rsn = pos["sl"], "SL"

                if ex is not None:
                    pts = round(ex - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"],
                        "exit_min": minute, "side": pos["side"], "symbol": pos["symbol"],
                        "entry": pos["entry"], "exit": ex, "pts": pts,
                        "rs": round(pts*LOT_SIZE), "reason": rsn,
                        "duration_min": pos["duration_min"], "tf": pos["tf"]})
                    dpnl += pts; closs = closs+1 if pts <= 0 else 0
                    if closs >= CONSECUTIVE_LOSS_LIMIT or dpnl <= DAILY_MAX_LOSS_PTS: shut = True
                    pos = None

        # EOD close
        if minute >= SESSION_END and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            trades.append({"date": day, "entry_min": pos["entry_min"],
                "exit_min": minute, "side": pos["side"], "symbol": pos["symbol"],
                "entry": pos["entry"], "exit": pos["last_px"], "pts": pts,
                "rs": round(pts*LOT_SIZE), "reason": "EOD",
                "duration_min": pos["duration_min"], "tf": pos["tf"]})
            dpnl += pts; pos = None; break

        # ── Skip entries: occupied / shutdown / EOD / no-trade window ──
        if pos is not None or shut or minute >= SESSION_END or in_no_trade:
            continue

        # Profit cap check (if applicable)
        if daily_max_profit_pts != float("inf") and dpnl >= daily_max_profit_pts:
            continue

        # ── Try to enter ──────────────────────────────────────────
        for item in pmtrig.get(minute, []):
            sig_side, sig_stk, sig_sym, tf, is_rev, stype, c_px, atr_val, tf_sl, tf_tp = item
            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                if is_rev:
                    as2 = "PE" if sig_side == "CE" else "CE"
                    ai2 = ainfo(as2, minute)
                    if ai2 is None: continue
                    asym, asl, _ = ai2
                else:
                    as2 = sig_side; asym = sig_sym; asl = ai[1]
                bar = bslice(asl, minute)
                if bar:
                    ep = float(bar[3])
                    if mode == "atr":
                        sl_p = atr_val*atr_sl_mult if (atr_val and atr_val > 0.5) else tf_sl
                        tp_p = atr_val*atr_tp_mult if (atr_val and atr_val > 0.5) else tf_tp
                        pos = {"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                               "sl":ep-sl_p,"tgt":ep+tp_p,"entry_min":minute,
                               "last_px":ep,"duration_min":0,"tf":tf}
                    else:  # trailing
                        pos = {"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                               "sl":ep-tf_sl,"tgt":None,"entry_min":minute,
                               "last_px":ep,"duration_min":0,"tf":tf,"trail_steps":0}
                    break
    return trades


# ── Strategy definitions ─────────────────────────────────────────────
STRATEGIES = [
    {
        "label": "Trailing SL (+5/+10) Fixed Cap [Pin Bar, S1=(9,3)]",
        "mode": "trailing", "atr_sl": 0.0, "atr_tp": 0.0,
        "max_profit_pts": 30.0,
        "s1": (9,3), "s2": (14,3), "s3": (40,4), "s4": (60,10),
    },
    {
        "label": "ATR x2.0/x4.0 Unlimited [Pin Bar, S1=(9,3)]",
        "mode": "atr", "atr_sl": 2.0, "atr_tp": 4.0,
        "max_profit_pts": float("inf"),
        "s1": (9,3), "s2": (14,3), "s3": (40,4), "s4": (60,10),
    },
    {
        "label": "S1=(12,3) + ATR x2.0/x4.0 [Pin Bar]",
        "mode": "atr", "atr_sl": 2.0, "atr_tp": 4.0,
        "max_profit_pts": 30.0,
        "s1": (12,3), "s2": (14,3), "s3": (40,4), "s4": (60,10),
    },
    {
        "label": "S1=(7,3)  + ATR x2.0/x4.0 [Pin Bar]",
        "mode": "atr", "atr_sl": 2.0, "atr_tp": 4.0,
        "max_profit_pts": 30.0,
        "s1": (7,3), "s2": (12,3), "s3": (21,4), "s4": (50,10),
    },
]

OLD_RESULTS = {
    "Trailing SL (+5/+10) Fixed Cap [Pin Bar, S1=(9,3)]": (6248,  736001),
    "ATR x2.0/x4.0 Unlimited [Pin Bar, S1=(9,3)]":        (6080,  701533),
    "S1=(12,3) + ATR x2.0/x4.0 [Pin Bar]":                (5277,  644933),
    "S1=(7,3)  + ATR x2.0/x4.0 [Pin Bar]":                (4487,  601381),
}


def main():
    spot_all = load_spot()
    files    = option_files("2020-01-01", "2024-12-31")
    days     = sorted(set(files.keys()) & set(spot_all.keys()))
    print(f"Loaded {len(days)} trading days.")
    print(f"NO-TRADE WINDOW : 11:30 AM - 1:30 PM IST (minutes {NO_TRADE_START}-{NO_TRADE_END})")
    print(f"Daily MaxLoss   : Rs {abs(DAILY_MAX_LOSS_RS):,.0f}")
    print(f"Daily MaxProfit : Strategy-dependent (see labels)\n", flush=True)

    all_results = []
    for cfg in STRATEGIES:
        label = cfg["label"]
        print(f"Running [{label}]...", flush=True)
        tasks = [(day, str(files[day]),
                  str(files[days[i-1]]) if i > 0 else "")
                 for i, day in enumerate(days)]
        t0 = time.time()
        all_trades = []
        with Pool(processes=min(cpu_count(), 8),
                  initializer=init_worker_local,
                  initargs=(spot_all, cfg)) as pool:
            for res in pool.map(process_day, tasks):
                all_trades.extend(res)
        st = summarize(all_trades)
        elapsed = time.time() - t0
        print(f"  Trades:{st['trades']} | WR:{st['wr']:.1f}% | "
              f"Net Pts:{st['pts']:+.2f} | Net: Rs {st['rs']:+,d} | "
              f"PF:{st['pf']:.2f} | {elapsed:.0f}s", flush=True)
        print_yearly_breakdown(all_trades)
        all_results.append((label, st, all_trades))

    # ── Final comparison ─────────────────────────────────────────
    w = 130
    print(f"\n{'='*w}")
    print(f"FINAL: No-Trade Window 11:30 AM - 1:30 PM  vs  Unrestricted  (2020-2024)")
    print(f"{'='*w}")
    print(f"{'STRATEGY':50s} | {'OLD TRADES':9} | {'OLD PROFIT':12} || "
          f"{'NEW TRADES':9} | {'NEW PROFIT':12} | {'PF':5} | CHANGE")
    print(f"{'-'*w}")
    for label, st, _ in all_results:
        old = OLD_RESULTS.get(label, (0, 0))
        delta = st['rs'] - old[1]
        chg = f"+Rs {delta:+,d}" if delta >= 0 else f"-Rs {abs(delta):,d}"
        print(f"{label:50s} | {old[0]:9,d} | Rs {old[1]:+10,d} || "
              f"{st['trades']:9,d} | Rs {st['rs']:+10,d} | {st['pf']:5.2f} | {chg}")
    print(f"\nNO-TRADE WINDOW: entries blocked {NO_TRADE_START//60}:{NO_TRADE_START%60:02d} - "
          f"{NO_TRADE_END//60}:{NO_TRADE_END%60:02d} IST | existing positions managed normally")


if __name__ == "__main__":
    main()
