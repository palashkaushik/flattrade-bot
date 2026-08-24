"""F6 Per-Timeframe Analysis

Tests F6 (flag no-div: S4>=80 + S1<=20 → immediate entry) signals
independently on each timeframe (1m, 2m, 3m, 5m).

Base strategy: ATR×2/×4 Unlimited (best baseline for F6).

Runs 5 scenarios:
  ALL  - All TFs, both F6 and standard pin bar (full strategy) — reference
  F6-1m  - Only F6 signals from 1m timeframe
  F6-2m  - Only F6 signals from 2m timeframe
  F6-3m  - Only F6 signals from 3m timeframe
  F6-5m  - Only F6 signals from 5m timeframe

Also shows F6-ONLY (all TFs) to isolate F6 contribution.
"""

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

SMOKE_TEST = False   # ← set True to validate on 5 days

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
CONSECUTIVE_LOSS_LIMIT = 6
ATR_PERIOD = 14
ATR_SL_MULT = 2.0
ATR_TP_MULT = 4.0

TF_SPECS = {
    "1m": (1, 10, 6.0,  30.0),
    "2m": (2,  5, 10.0, 15.0),
    "3m": (3,  4, 8.0,  25.0),
    "5m": (5,  3, 10.0, 35.0),
}

GLOBAL_SPOT   = {}
GLOBAL_CONFIG = {}

def init_worker(sd, cfg):
    global GLOBAL_SPOT, GLOBAL_CONFIG
    GLOBAL_SPOT = sd; GLOBAL_CONFIG = cfg


class IncrementalATR:
    def __init__(self, period=14):
        self.period = period; self._buf = deque(maxlen=period)
        self.atr = None; self.prev_close = None; self._n = 0
    def update(self, h, l, c):
        tr = max(h-l, abs(h-self.prev_close), abs(l-self.prev_close)) if self.prev_close else h-l
        self._buf.append(tr); self._n += 1; self.prev_close = c
        if self._n < self.period:    self.atr = None
        elif self._n == self.period: self.atr = sum(self._buf)/self.period
        else:                        self.atr = (self.atr*(self.period-1)+tr)/self.period
        return self.atr


class TFTracker:
    def __init__(self, lb):
        self.lb = lb
        self.s1 = IncrementalStochastic(9, 3)
        self.s2 = IncrementalStochastic(14, 3)
        self.s3 = IncrementalStochastic(40, 4)
        self.s4 = IncrementalStochastic(60, 10)
        self.div = DivergenceEngine()
        self.hist = []; self.setup = False; self.stype = ""
        self.prev_s1 = None; self.s4_emb = 0
        self.atr = IncrementalATR(ATR_PERIOD)
    def push(self, c: Candle):
        self.hist.append(c)
        if len(self.hist) > 40: self.hist.pop(0)
        s1 = self.s1.push(c.high, c.low, c.close)
        s2 = self.s2.push(c.high, c.low, c.close)
        s3 = self.s3.push(c.high, c.low, c.close)
        s4 = self.s4.push(c.high, c.low, c.close)
        atr_val = self.atr.update(c.high, c.low, c.close)
        self.prev_s1 = s1
        if s4 is not None: self.s4_emb = self.s4_emb+1 if s4 <= 20 else 0
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


class FlagNoDivScanner:
    def __init__(self):
        self.s1 = IncrementalStochastic(9, 3)
        self.s4 = IncrementalStochastic(60, 10)
        self._fired = False
    def push(self, h, l, c):
        s1v = self.s1.push(h, l, c)
        s4v = self.s4.push(h, l, c)
        if s1v is None or s4v is None: return False
        flag = s4v >= 79.5 and s1v <= 20.5
        if flag and not self._fired:
            self._fired = True; return True
        if not flag: self._fired = False
        return False


class MTFTracker:
    def __init__(self, allow_pinbar_tfs, allow_f6_tfs):
        """
        allow_pinbar_tfs: set of TFs for standard pin bar signals
        allow_f6_tfs:     set of TFs for F6 immediate flag signals
        """
        self.trackers = {tf: TFTracker(spec[1]) for tf, spec in TF_SPECS.items()}
        self.f6scans  = {tf: FlagNoDivScanner() for tf in TF_SPECS}
        self.bufs = {tf: [] for tf in TF_SPECS}
        self.allow_pinbar_tfs = allow_pinbar_tfs
        self.allow_f6_tfs     = allow_f6_tfs

    def push_1m(self, c1m: Candle):
        out = []
        for tf, spec in TF_SPECS.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf]) == spec[0]:
                buf = self.bufs[tf]; self.bufs[tf] = []
                ctf = Candle(open=buf[0].open, high=max(x.high for x in buf),
                             low=min(x.low for x in buf), close=buf[-1].close,
                             minute=buf[-1].minute)
                trig, is_rev, stype, px, atr_val = self.trackers[tf].push(ctf)
                if trig and tf in self.allow_pinbar_tfs:
                    out.append((tf, is_rev, "pinbar_"+stype, px, atr_val))
                if tf in self.allow_f6_tfs:
                    if self.f6scans[tf].push(ctf.high, ctf.low, ctf.close):
                        out.append((tf, False, "f6_flag", ctf.close, atr_val))
        return out


def process_day(args):
    day, fpath, fprev = args
    cfg = GLOBAL_CONFIG
    spot = GLOBAL_SPOT.get(day)
    allow_pinbar_tfs = cfg["allow_pinbar_tfs"]
    allow_f6_tfs     = cfg["allow_f6_tfs"]

    if spot is None or not fpath: return []
    fp = Path(fpath)
    if not fp.exists(): return []
    sp0 = latest_spot(spot, 555) or latest_spot(spot, 560)
    if sp0 is None: return []
    atm0 = int(round(sp0/50)*50)
    target_strikes = set(range(atm0-250, atm0+300, 50))
    try:
        dfc = pd.read_csv(fp, usecols=["time","symbol","open","high","low","close"], engine="c")
    except: return []
    if dfc.empty: return []
    fsym = dfc["symbol"].iloc[0]; mm = SYM_RE.match(fsym)
    if not mm: return []
    prefix = mm.group(1)
    dfc["min"] = np.array([to_minutes(t) for t in dfc["time"]])
    gc = {sym:g for sym,g in dfc.groupby("symbol")
          if (m:=SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}
    gp = {}
    if fprev and Path(fprev).exists():
        try:
            dfp = pd.read_csv(fprev, usecols=["time","symbol","open","high","low","close"], engine="c")
            if not dfp.empty:
                dfp["min"] = np.array([to_minutes(t) for t in dfp["time"]])
                gp = {sym:g for sym,g in dfp.groupby("symbol")
                      if (m:=SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}
        except: pass

    trk = {}
    for sym, g in gp.items():
        trk[sym] = MTFTracker(allow_pinbar_tfs, allow_f6_tfs)
        mn, op, hi, lo, cl = (g[c].to_numpy() for c in ["min","open","high","low","close"])
        for i in range(len(mn)):
            trk[sym].push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=mn[i]))

    pmtrig = {}; slices = {}
    for sym, g in gc.items():
        if sym not in trk: trk[sym] = MTFTracker(allow_pinbar_tfs, allow_f6_tfs)
        t = trk[sym]
        mn, op, hi, lo, cl = (g[c].to_numpy() for c in ["min","open","high","low","close"])
        slices[sym] = {"min": mn, "open": op, "high": hi, "low": lo, "close": cl}
        mm2 = SYM_RE.match(sym)
        if not mm2: continue
        sv, side = int(mm2.group(2)), mm2.group(3)
        for i in range(len(mn)):
            m = mn[i]
            for (tf, is_rev, stype, px, atr_val) in t.push_1m(
                    Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=m)):
                pmtrig.setdefault(m, []).append(
                    (side, sv, sym, px, is_rev, tf, stype,
                     TF_SPECS[tf][2], TF_SPECS[tf][3], atr_val))

    def bslice(sl, m):
        idx = np.searchsorted(sl["min"], m)
        if idx < len(sl["min"]) and sl["min"][idx] == m:
            return sl["open"][idx], sl["high"][idx], sl["low"][idx], sl["close"][idx]
        return None
    def ainfo(side, m):
        spx = latest_spot(spot, m)
        if spx is None: return None
        atm = int(round(spx/50)*50)
        stk = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        sym = f"{prefix}{stk}{side}"; sl = slices.get(sym)
        return (sym, sl, stk) if sl is not None else None

    trades = []; pos = None; dpnl = 0.0; closs = 0; shut = False
    for minute in range(SESSION_START, DAY_LAST+1):
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held; pos["last_px"] = float(c); pos["duration_min"] += 1
                if dpnl*LOT_SIZE + (c-pos["entry"])*LOT_SIZE <= -2000.0:
                    pts = round(c-pos["entry"], 2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":c,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"SHUTDOWN_LOSS",
                        "duration_min":pos["duration_min"],"tf":pos["tf"],"stype":pos["stype"]})
                    dpnl += pts; pos = None; shut = True; continue
                ex, rsn = None, ""
                if h >= pos["tgt"] and l <= pos["sl"]: ex, rsn = pos["sl"], "SL"
                elif h >= pos["tgt"]: ex, rsn = pos["tgt"], "TP"
                elif l <= pos["sl"]: ex, rsn = pos["sl"], "SL"
                if ex is None:
                    t1 = trk.get(pos["symbol"])
                    if t1:
                        t1m = t1.trackers["1m"]
                        t1m.div.update(c, t1m.prev_s1)
                        if t1m.div.has_bearish_peak_divergence(): ex, rsn = c, "BEARISH_PEAK_REVERSAL"
                if ex is not None:
                    pts = round(ex-pos["entry"], 2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":ex,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":rsn,
                        "duration_min":pos["duration_min"],"tf":pos["tf"],"stype":pos["stype"]})
                    dpnl += pts; closs = closs+1 if pts <= 0 else 0
                    if closs >= CONSECUTIVE_LOSS_LIMIT or dpnl <= -2000.0/LOT_SIZE: shut = True
                    pos = None
        if minute >= SESSION_END and pos is not None:
            pts = round(pos["last_px"]-pos["entry"], 2)
            trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":pos["last_px"],
                "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"EOD",
                "duration_min":pos["duration_min"],"tf":pos["tf"],"stype":pos["stype"]})
            dpnl += pts; pos = None; break
        if pos is not None or shut or minute >= SESSION_END: continue

        for (sig_side, sig_stk, sig_sym, c_px, is_rev, tf, stype, sl_pts, tp_pts, atr_val) \
                in pmtrig.get(minute, []):
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
                    if atr_val and atr_val > 0.5:
                        sl_use = atr_val * ATR_SL_MULT
                        tp_use = atr_val * ATR_TP_MULT
                    else:
                        sl_use = sl_pts; tp_use = tp_pts
                    pos = {"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                           "sl":ep-sl_use,"tgt":ep+tp_use,
                           "entry_min":minute,"last_px":ep,"duration_min":0,
                           "tf":tf,"stype":stype}
                    break
    return trades


def tf_breakdown(all_trades):
    """Per-TF per-signal-type breakdown."""
    from collections import defaultdict
    groups = defaultdict(list)
    for t in all_trades:
        key = (t["tf"], "F6" if t["stype"] == "f6_flag" else "PinBar")
        groups[key].append(t)
    rows = []
    for (tf, sig), trades in sorted(groups.items()):
        wins = [t for t in trades if t["pts"] > 0]
        total_rs = sum(t["rs"] for t in trades)
        total_pts = sum(t["pts"] for t in trades)
        wr = 100*len(wins)/len(trades) if trades else 0
        avg_pts = total_pts/len(trades) if trades else 0
        rows.append((tf, sig, len(trades), wr, total_pts, total_rs, avg_pts))
    return rows


def run_scenario(label, cfg, spot_all, files, days):
    print(f"\nRunning [{label}]...", flush=True)
    tasks = [(day, str(files[day]), str(files[days[i-1]]) if i > 0 else "")
             for i, day in enumerate(days)]
    t0 = time.time()
    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker,
              initargs=(spot_all, cfg)) as pool:
        for res in pool.map(process_day, tasks):
            all_trades.extend(res)
    st = summarize(all_trades)
    elapsed = time.time() - t0
    print(f"  Trades:{st['trades']:6,d} | WR:{st['wr']:.1f}% | "
          f"Rs:{st['rs']:+,d} | PF:{st['pf']:.2f} | {elapsed:.0f}s", flush=True)
    return st, all_trades


def main():
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))
    if SMOKE_TEST:
        days = days[:5]
        print(f"=== SMOKE TEST — {len(days)} DAYS ONLY ===")
    print(f"Loaded {len(days)} trading days.")
    print("F6 Per-Timeframe Analysis — ATR x2/x4 Unlimited base\n", flush=True)

    ALL_TFS = set(TF_SPECS.keys())

    scenarios = [
        # (label, allow_pinbar_tfs, allow_f6_tfs)
        ("Full Strategy (PinBar + F6, all TFs)",  ALL_TFS, ALL_TFS),
        ("F6-ONLY (all TFs, no PinBar)",          set(),   ALL_TFS),
        ("F6-ONLY 1m",                             set(),   {"1m"}),
        ("F6-ONLY 2m",                             set(),   {"2m"}),
        ("F6-ONLY 3m",                             set(),   {"3m"}),
        ("F6-ONLY 5m",                             set(),   {"5m"}),
        ("PinBar-ONLY (no F6, all TFs)",           ALL_TFS, set()),
    ]

    all_results = []
    for label, pinbar_tfs, f6_tfs in scenarios:
        cfg = {"allow_pinbar_tfs": pinbar_tfs, "allow_f6_tfs": f6_tfs}
        st, trades = run_scenario(label, cfg, spot_all, files, days)
        all_results.append((label, st, trades))

    # Summary table
    w = 95
    print(f"\n{'='*w}")
    print("F6 PER-TIMEFRAME ANALYSIS  (ATR x2.0/x4.0 | Unlimited | 2020-2024)")
    print(f"{'='*w}")
    print(f"{'SCENARIO':44s} | {'TRADES':>7} | {'WR%':>6} | {'NET PROFIT':>13} | {'PF':>5} | {'AVG PTS':>8}")
    print(f"{'-'*w}")
    for label, st, _ in all_results:
        trades_list = [t for r in all_results if r[0] == label for t in r[2]]
        avg_pts = sum(t["pts"] for t in trades_list)/len(trades_list) if trades_list else 0
        print(f"{label:44s} | {st['trades']:7,d} | {st['wr']:6.1f}% | "
              f"Rs {st['rs']:+10,d} | {st['pf']:5.2f} | {avg_pts:+8.2f}")

    # Per-TF breakdown for Full Strategy
    print(f"\n{'='*w}")
    print("PER-TF BREAKDOWN — Full Strategy (PinBar + F6)")
    print(f"{'='*w}")
    print(f"{'TF':>4} | {'SIGNAL':>8} | {'TRADES':>7} | {'WR%':>6} | "
          f"{'TOTAL PTS':>10} | {'NET RS':>12} | {'AVG PTS/TRD':>12}")
    print(f"{'-'*w}")
    full_trades = all_results[0][2]
    rows = tf_breakdown(full_trades)
    tf_totals = {}
    for tf, sig, cnt, wr, total_pts, total_rs, avg_pts in rows:
        print(f"{tf:>4} | {sig:>8} | {cnt:7,d} | {wr:6.1f}% | "
              f"{total_pts:+10.2f} | Rs {total_rs:+10,d} | {avg_pts:+12.2f}")
        if tf not in tf_totals: tf_totals[tf] = {"cnt":0,"wins":0,"pts":0,"rs":0}
        tf_totals[tf]["cnt"] += cnt
        tf_totals[tf]["wins"] += int(wr*cnt/100)
        tf_totals[tf]["pts"] += total_pts
        tf_totals[tf]["rs"]  += total_rs
    print(f"{'-'*w}")
    print("  TF TOTALS:")
    for tf in ["1m","2m","3m","5m"]:
        if tf in tf_totals:
            d = tf_totals[tf]
            wr = 100*d["wins"]/d["cnt"] if d["cnt"] else 0
            avg = d["pts"]/d["cnt"] if d["cnt"] else 0
            print(f"{tf:>4} | {'TOTAL':>8} | {d['cnt']:7,d} | {wr:6.1f}% | "
                  f"{d['pts']:+10.2f} | Rs {d['rs']:+10,d} | {avg:+12.2f}")

    # Per-TF breakdown for F6-ONLY
    print(f"\n{'='*w}")
    print("PER-TF BREAKDOWN — F6-ONLY (no PinBar)")
    print(f"{'='*w}")
    print(f"{'TF':>4} | {'TRADES':>7} | {'WR%':>6} | {'TOTAL PTS':>10} | "
          f"{'NET RS':>12} | {'AVG PTS/TRD':>12}")
    print(f"{'-'*w}")
    f6_only_trades = all_results[1][2]
    for tf in ["1m","2m","3m","5m"]:
        tf_trades = [t for t in f6_only_trades if t["tf"] == tf]
        if not tf_trades: continue
        wins = sum(1 for t in tf_trades if t["pts"] > 0)
        total_pts = sum(t["pts"] for t in tf_trades)
        total_rs  = sum(t["rs"]  for t in tf_trades)
        wr = 100*wins/len(tf_trades)
        avg = total_pts/len(tf_trades)
        print(f"{tf:>4} | {len(tf_trades):7,d} | {wr:6.1f}% | "
              f"{total_pts:+10.2f} | Rs {total_rs:+10,d} | {avg:+12.2f}")

    # Yearly for full strategy
    print(f"\n{'='*w}")
    print("YEARLY — Full Strategy (PinBar + F6)")
    print_yearly_breakdown(full_trades)


if __name__ == "__main__":
    main()
