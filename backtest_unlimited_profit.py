"""All SL/TP Strategies Backtest — Daily Max Loss = Rs 2,000 | Daily Max Profit = UNLIMITED.

Tests all 8 strategies:
  1. Trailing SL (+5 pts per +10 pts gain)
  2. ATR(14) SL x2.0 / TP x4.0
  3. ATR(14) SL x2.0 / TP x3.0
  4. ATR(14) SL x1.5 / TP x3.0
  5. Baseline (Fixed SL/TP per TF)
  6. ATR(14) SL x1.0 / TP x3.0
  7. ATR(14) SL x1.5 / TP x2.0
  8. ATR(14) SL x1.0 / TP x2.0
"""

import re, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count
from collections import deque

import numpy as np
import pandas as pd

from backtest_5y_optimized import load_spot, option_files, SYM_RE, to_minutes, latest_spot, TimeframeTracker, summarize, print_yearly_breakdown
from flattrade_bot.indicators.patterns import Candle

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930

# ── KEY CHANGE: Daily Max Loss = Rs 2,000 | Max Profit = UNLIMITED ──
DAILY_MAX_LOSS_RS   = -2000.0
DAILY_MAX_LOSS_PTS  = DAILY_MAX_LOSS_RS / LOT_SIZE   # = -30.77 pts
DAILY_MAX_PROFIT_PTS = float("inf")                  # UNLIMITED

CONSECUTIVE_LOSS_LIMIT = 6
ATR_PERIOD = 14
TRAIL_STEP_PTS  = 10.0
TRAIL_AMOUNT_PTS = 5.0

TF_SPECS = {
    "1m": (1, 10, 6.0, 30.0),
    "2m": (2,  5, 10.0, 15.0),
    "3m": (3,  4, 8.0, 25.0),
    "5m": (5,  3, 10.0, 35.0),
}

GLOBAL_SPOT = {}
def init_worker_local(sd): global GLOBAL_SPOT; GLOBAL_SPOT = sd


class IncrementalATR:
    def __init__(self, period=14):
        self.period = period; self._buf = deque(maxlen=period)
        self.atr = None; self.prev_close = None; self._n = 0
    def update(self, high, low, close):
        tr = max(high-low, abs(high-self.prev_close), abs(low-self.prev_close)) if self.prev_close else high-low
        self._buf.append(tr); self._n += 1; self.prev_close = close
        if self._n < self.period:    self.atr = None
        elif self._n == self.period: self.atr = sum(self._buf)/self.period
        else:                        self.atr = (self.atr*(self.period-1)+tr)/self.period
        return self.atr


class MultiTimeframeTracker:
    def __init__(self):
        self.trackers = {tf: TimeframeTracker(tf, max_lookback=spec[1]) for tf,spec in TF_SPECS.items()}
        self.atrs  = {tf: IncrementalATR(ATR_PERIOD) for tf in TF_SPECS}
        self.bufs  = {tf: [] for tf in TF_SPECS}
    def push_1m(self, c1m: Candle):
        out = []
        for tf, spec in TF_SPECS.items():
            tf_m = spec[0]; self.bufs[tf].append(c1m)
            if len(self.bufs[tf]) == tf_m:
                buf = self.bufs[tf]
                ctf = Candle(open=buf[0].open, high=max(x.high for x in buf),
                             low=min(x.low for x in buf), close=buf[-1].close, minute=buf[-1].minute)
                self.bufs[tf] = []
                atr_val = self.atrs[tf].update(ctf.high, ctf.low, ctf.close)
                trig,is_rev,stype,px = self.trackers[tf].push(ctf)
                if trig: out.append((tf, is_rev, stype, px, atr_val))
        return out


def process_day(args):
    day, fpath, fprev, mode, atr_sl_mult, atr_tp_mult = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not fpath: return []
    fp = Path(fpath)
    if not fp.exists(): return []
    sp0 = latest_spot(spot,555) or latest_spot(spot,560)
    if sp0 is None: return []
    atm0 = int(round(sp0/50)*50)
    target_strikes = set(range(atm0-250, atm0+300, 50))
    try:
        dfc = pd.read_csv(fp, usecols=["time","symbol","open","high","low","close"], engine="c")
    except: return []
    if dfc.empty: return []
    fsym=dfc["symbol"].iloc[0]; mm=SYM_RE.match(fsym)
    if not mm: return []
    prefix=mm.group(1)
    dfc["min"]=np.array([to_minutes(t) for t in dfc["time"]])
    gc={sym:g for sym,g in dfc.groupby("symbol") if (m:=SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}
    gp={}
    if fprev and Path(fprev).exists():
        try:
            dfp=pd.read_csv(fprev, usecols=["time","symbol","open","high","low","close"], engine="c")
            if not dfp.empty:
                dfp["min"]=np.array([to_minutes(t) for t in dfp["time"]])
                gp={sym:g for sym,g in dfp.groupby("symbol") if (m:=SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}
        except: pass
    trk={}
    for sym,g in gp.items():
        trk[sym]=MultiTimeframeTracker()
        mn,op,hi,lo,cl=g["min"].to_numpy(),g["open"].to_numpy(),g["high"].to_numpy(),g["low"].to_numpy(),g["close"].to_numpy()
        for i in range(len(mn)): trk[sym].push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=mn[i]))
    pmtrig={}; slices={}
    for sym,g in gc.items():
        if sym not in trk: trk[sym]=MultiTimeframeTracker()
        t=trk[sym]
        mn,op,hi,lo,cl=g["min"].to_numpy(),g["open"].to_numpy(),g["high"].to_numpy(),g["low"].to_numpy(),g["close"].to_numpy()
        slices[sym]={"min":mn,"open":op,"high":hi,"low":lo,"close":cl}
        mm2=SYM_RE.match(sym)
        if not mm2: continue
        sv,side=int(mm2.group(2)),mm2.group(3)
        for i in range(len(mn)):
            m=mn[i]
            for (tf,is_rev,stype,px,atr_val) in t.push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=m)):
                pmtrig.setdefault(m,[]).append((side,sv,sym,px,is_rev,tf,TF_SPECS[tf][2],TF_SPECS[tf][3],atr_val))

    trades=[]; pos=None; dpnl=0.0; closs=0; shut=False
    def bslice(sl,m):
        idx=np.searchsorted(sl["min"],m)
        if idx<len(sl["min"]) and sl["min"][idx]==m: return sl["open"][idx],sl["high"][idx],sl["low"][idx],sl["close"][idx]
        return None
    def ainfo(side,m):
        spx=latest_spot(spot,m)
        if spx is None: return None
        atm=int(round(spx/50)*50); stk=atm+(CE_OFFSET if side=="CE" else PE_OFFSET)
        sym=f"{prefix}{stk}{side}"; sl=slices.get(sym)
        return (sym,sl,stk) if sl is not None else None

    for minute in range(SESSION_START, DAY_LAST+1):
        if pos is not None:
            held=bslice(pos["slice"],minute)
            if held:
                o,h,l,c=held; pos["last_px"]=float(c); pos["duration_min"]+=1
                # Trailing SL update
                if mode=="trailing":
                    gain=c-pos["entry"]
                    steps=int(gain/TRAIL_STEP_PTS)
                    if steps>pos["trail_steps"]:
                        pos["sl"]+=(steps-pos["trail_steps"])*TRAIL_AMOUNT_PTS
                        pos["trail_steps"]=steps
                # Daily max loss check (Rs 2,000)
                unrealized_rs=(c-pos["entry"])*LOT_SIZE
                if dpnl*LOT_SIZE + unrealized_rs <= DAILY_MAX_LOSS_RS:
                    pts=round(c-pos["entry"],2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":c,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"SHUTDOWN_LOSS",
                        "duration_min":pos["duration_min"],"is_rev":pos["is_rev"],"tf":pos["tf"]})
                    dpnl+=pts; pos=None; shut=True; continue
                ex,rsn=None,""
                has_tgt=pos.get("tgt") is not None
                if has_tgt and h>=pos["tgt"] and l<=pos["sl"]: ex,rsn=pos["sl"],"SL"
                elif has_tgt and h>=pos["tgt"]: ex,rsn=pos["tgt"],"TP"
                elif l<=pos["sl"]: ex,rsn=pos["sl"],"SL"
                if ex is None:
                    t1=trk.get(pos["symbol"])
                    if t1:
                        t1m=t1.trackers["1m"]; t1m.divergence.update(c,t1m.prev_s1)
                        if t1m.divergence.has_bearish_peak_divergence(): ex,rsn=c,"BEARISH_PEAK_REVERSAL"
                if ex is not None:
                    pts=round(ex-pos["entry"],2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":ex,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":rsn,
                        "duration_min":pos["duration_min"],"is_rev":pos["is_rev"],"tf":pos["tf"]})
                    dpnl+=pts; closs=closs+1 if pts<=0 else 0
                    # NO max profit shutdown — unlimited upside!
                    if closs>=CONSECUTIVE_LOSS_LIMIT or dpnl<=DAILY_MAX_LOSS_PTS: shut=True
                    pos=None
            if minute>=SESSION_END and pos is not None:
                pts=round(pos["last_px"]-pos["entry"],2)
                trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                    "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":pos["last_px"],
                    "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"EOD",
                    "duration_min":pos["duration_min"],"is_rev":pos["is_rev"],"tf":pos["tf"]})
                dpnl+=pts; pos=None; break
        if pos is not None or shut or minute>=SESSION_END: continue
        for (sig_side,sig_stk,sig_sym,c_px,is_rev,tf,sl_pts,tp_pts,atr_val) in pmtrig.get(minute,[]):
            ai=ainfo(sig_side,minute)
            if ai and ai[2]==sig_stk and pos is None:
                if is_rev:
                    as2="PE" if sig_side=="CE" else "CE"; ai2=ainfo(as2,minute)
                    if ai2 is None: continue
                    asym,asl,_=ai2
                else:
                    as2=sig_side; asym=sig_sym; asl=ai[1]
                bar=bslice(asl,minute)
                if bar:
                    ep=float(bar[3])
                    if mode=="atr":
                        if atr_val and atr_val>0.5:
                            sl_use=atr_val*atr_sl_mult; tp_use=atr_val*atr_tp_mult
                        else:
                            sl_use=sl_pts; tp_use=tp_pts
                        pos={"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                             "sl":ep-sl_use,"tgt":ep+tp_use,
                             "entry_min":minute,"last_px":ep,"duration_min":0,"is_rev":is_rev,"tf":tf}
                    elif mode=="trailing":
                        pos={"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                             "sl":ep-sl_pts,"tgt":None,
                             "entry_min":minute,"last_px":ep,"duration_min":0,"is_rev":is_rev,"tf":tf,"trail_steps":0}
                    else:
                        pos={"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                             "sl":ep-sl_pts,"tgt":ep+tp_pts,
                             "entry_min":minute,"last_px":ep,"duration_min":0,"is_rev":is_rev,"tf":tf}
                    break
    return trades


STRATEGIES = [
    ("trailing",  0,   0,   "Trailing SL (+5pts / +10pts gain, No TP)"),
    ("atr",       2.0, 4.0, "ATR(14) SL x2.0 / TP x4.0"),
    ("atr",       2.0, 3.0, "ATR(14) SL x2.0 / TP x3.0"),
    ("atr",       1.5, 3.0, "ATR(14) SL x1.5 / TP x3.0"),
    ("baseline",  0,   0,   "Baseline (Fixed SL/TP per TF)"),
    ("atr",       1.0, 3.0, "ATR(14) SL x1.0 / TP x3.0"),
    ("atr",       1.5, 2.0, "ATR(14) SL x1.5 / TP x2.0"),
    ("atr",       1.0, 2.0, "ATR(14) SL x1.0 / TP x2.0"),
]


def run_strategy(mode, atr_sl, atr_tp, label, days, files, spot_all):
    tasks=[(day, str(files[day]), str(files[days[i-1]]) if i>0 else "", mode, atr_sl, atr_tp)
           for i,day in enumerate(days)]
    all_trades=[]
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,)) as pool:
        for res in pool.map(process_day, tasks): all_trades.extend(res)
    st=summarize(all_trades)
    print(f"\n{'='*115}")
    print(f"RESULTS (Daily MaxLoss=Rs 2,000 | MaxProfit=UNLIMITED): {label.upper()}")
    print(f"{'='*115}")
    print(f"Trades: {st['trades']} | WR: {st['wr']:.1f}% | Net Pts: {st['pts']:+.2f} | Net Profit: Rs {st['rs']:+,d} | PF: {st['pf']:.2f}")
    print_yearly_breakdown(all_trades)
    return st


def main():
    spot_all=load_spot()
    files=option_files("2020-01-01","2024-12-31")
    days=sorted(set(files.keys()) & set(spot_all.keys()))
    print(f"Loaded {len(days)} trading days.")
    print(f"Daily Max Loss: Rs {DAILY_MAX_LOSS_RS:,.0f} ({DAILY_MAX_LOSS_PTS:.2f} pts) | Max Profit: UNLIMITED", flush=True)

    results=[]
    for mode, atr_sl, atr_tp, label in STRATEGIES:
        print(f"\nRunning: {label}...", flush=True)
        st=run_strategy(mode, atr_sl, atr_tp, label, days, files, spot_all)
        results.append((label, st))

    print(f"\n{'='*125}")
    print(f"FINAL COMPARISON: Daily MaxLoss=Rs 2,000 | MaxProfit=UNLIMITED (2020-2024)")
    print(f"{'='*125}")
    print(f"{'STRATEGY':45s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':12s} | {'NET PROFIT':14s} | PF")
    print(f"{'-'*125}")
    for label, st in sorted(results, key=lambda x: x[1]["rs"], reverse=True):
        pf_s=f"{st['pf']:.2f}" if st['pf']!=float('inf') else "INF"
        print(f"{label:45s} | {st['trades']:7d} | {st['wr']:8.1f}% | {st['pts']:+12.2f} | Rs {st['rs']:+12,d} | {pf_s}")

if __name__=="__main__":
    main()
