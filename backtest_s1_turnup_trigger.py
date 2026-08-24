"""Full Strategy Leaderboard Backtest  New Trigger Rule:
  SUPER SIGNAL: Fires when S1 turns up (S1_current > S1_previous) after setup conditions.
  FLAG SIGNAL : Unchanged  still uses BullishPinBar vicinity breakout.

Tests all top configurations:
  1. Trailing SL,          S1=(9,3)   Baseline comparison
  2. S1=(12,3) + ATR2/4             Best net profit
  3. S1=(7,3)  + ATR2/4             Best quality/WR
  4. ATR2/4,             S1=(9,3)   ATR only
  5. S1=(12,3) Fixed SL/TP            Stoch-only best
  6. Baseline Fixed SL/TP, S1=(9,3)   Original baseline
"""

import time
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from collections import deque

import numpy as np
import pandas as pd

from backtest_5y_optimized import load_spot, option_files, SYM_RE, to_minutes, latest_spot, summarize, print_yearly_breakdown
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from flattrade_bot.indicators.divergence import DivergenceEngine

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_MAX_PROFIT_PTS = 30.0
DAILY_MAX_LOSS_PTS   = -30.0
CONSECUTIVE_LOSS_LIMIT = 6
ATR_PERIOD = 14
TRAIL_STEP_PTS  = 10.0
TRAIL_AMOUNT_PTS = 5.0

TF_SPECS = {
    "1m": (1, 10, 6.0, 30.0),
    "2m": (2,  5, 10.0, 15.0),
    "3m": (3,  4, 8.0,  25.0),
    "5m": (5,  3, 10.0, 35.0),
}

GLOBAL_CONFIG = {}
GLOBAL_SPOT   = {}

def init_worker_local(sd, cfg):
    global GLOBAL_SPOT, GLOBAL_CONFIG
    GLOBAL_SPOT   = sd
    GLOBAL_CONFIG = cfg


#  Incremental ATR 
class IncrementalATR:
    def __init__(self, period=14):
        self.period=period; self._buf=deque(maxlen=period)
        self.atr=None; self.prev_close=None; self._n=0
    def update(self, h, l, c):
        tr=max(h-l, abs(h-self.prev_close), abs(l-self.prev_close)) if self.prev_close else h-l
        self._buf.append(tr); self._n+=1; self.prev_close=c
        if self._n<self.period:    self.atr=None
        elif self._n==self.period: self.atr=sum(self._buf)/self.period
        else:                      self.atr=(self.atr*(self.period-1)+tr)/self.period
        return self.atr


#  Per-timeframe tracker with NEW trigger rule 
class TFTracker:
    def __init__(self, lb, tf_sl, tf_tp, s1_spec, s2_spec, s3_spec, s4_spec):
        self.lb=lb; self.tf_sl=tf_sl; self.tf_tp=tf_tp
        self.s1=IncrementalStochastic(*s1_spec)
        self.s2=IncrementalStochastic(*s2_spec)
        self.s3=IncrementalStochastic(*s3_spec)
        self.s4=IncrementalStochastic(*s4_spec)
        self.div=DivergenceEngine(); self.atr=IncrementalATR(ATR_PERIOD)
        self.hist=[]; self.setup=False; self.stype=""
        self.prev_s1=None; self.s4_emb=0

    def push(self, c: Candle):
        self.hist.append(c)
        if len(self.hist)>60: self.hist.pop(0)
        s1=self.s1.push(c.high, c.low, c.close)
        s2=self.s2.push(c.high, c.low, c.close)
        s3=self.s3.push(c.high, c.low, c.close)
        s4=self.s4.push(c.high, c.low, c.close)
        atr_val=self.atr.update(c.high, c.low, c.close)

        if s4 is not None: self.s4_emb=self.s4_emb+1 if s4<=20 else 0
        emb=self.s4_emb>25
        self.div.update(c.close, s1); bull_div=self.div.has_bullish_trough_divergence()

        is_flag =s4 is not None and s1 is not None and s4>=79.5 and s1<=20.5
        is_super=all(v is not None and v<=20.5 for v in (s1,s2,s3,s4))
        if (is_flag or is_super) and bull_div:
            self.setup=True
            self.stype="super" if is_super else "flag"

        is_rev=emb and self.stype=="super"
        triggered=False

        if self.setup and len(self.hist)>=2:
            if self.stype=="super":
                #  NEW: S1 turning up triggers the super signal 
                if s1 is not None and self.prev_s1 is not None and s1 > self.prev_s1:
                    triggered=True; self.setup=False
            else:
                # Flag signal: unchanged  pin bar vicinity breakout
                if BullishPinBarDetector.check_vicinity_breakout(self.hist, self.lb):
                    triggered=True; self.setup=False

        self.prev_s1=s1   # Update AFTER trigger check
        return triggered, is_rev, self.stype, c.close, atr_val


#  Multi-timeframe tracker 
class MTFTracker:
    def __init__(self, s1_spec, s2_spec, s3_spec, s4_spec):
        self.trackers={
            tf: TFTracker(spec[1], spec[2], spec[3], s1_spec, s2_spec, s3_spec, s4_spec)
            for tf,spec in TF_SPECS.items()
        }
        self.bufs={tf:[] for tf in TF_SPECS}

    def push_1m(self, c1m: Candle):
        out=[]
        for tf,spec in TF_SPECS.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf])==spec[0]:
                buf=self.bufs[tf]
                ctf=Candle(open=buf[0].open,high=max(x.high for x in buf),
                           low=min(x.low for x in buf),close=buf[-1].close,minute=buf[-1].minute)
                self.bufs[tf]=[]
                trig,is_rev,stype,px,atr_val=self.trackers[tf].push(ctf)
                if trig: out.append((tf,is_rev,stype,px,atr_val,spec[2],spec[3]))
        return out


#  Per-day worker 
def process_day(args):
    day, fpath, fprev = args
    cfg    = GLOBAL_CONFIG
    spot   = GLOBAL_SPOT.get(day)
    mode   = cfg["mode"]
    atr_sl = cfg["atr_sl"]
    atr_tp = cfg["atr_tp"]
    s1_spec,s2_spec,s3_spec,s4_spec = cfg["s1"],cfg["s2"],cfg["s3"],cfg["s4"]

    if spot is None or not fpath: return []
    fp=Path(fpath)
    if not fp.exists(): return []
    sp0=latest_spot(spot,555) or latest_spot(spot,560)
    if sp0 is None: return []
    atm0=int(round(sp0/50)*50)
    target_strikes=set(range(atm0-250, atm0+300, 50))
    try:
        dfc=pd.read_csv(fp, usecols=["time","symbol","open","high","low","close"], engine="c")
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
    def make_tracker(): return MTFTracker(s1_spec,s2_spec,s3_spec,s4_spec)
    for sym,g in gp.items():
        trk[sym]=make_tracker()
        mn,op,hi,lo,cl=g["min"].to_numpy(),g["open"].to_numpy(),g["high"].to_numpy(),g["low"].to_numpy(),g["close"].to_numpy()
        for i in range(len(mn)): trk[sym].push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=mn[i]))

    pmtrig={}; slices={}
    for sym,g in gc.items():
        if sym not in trk: trk[sym]=make_tracker()
        t=trk[sym]
        mn,op,hi,lo,cl=g["min"].to_numpy(),g["open"].to_numpy(),g["high"].to_numpy(),g["low"].to_numpy(),g["close"].to_numpy()
        slices[sym]={"min":mn,"open":op,"high":hi,"low":lo,"close":cl}
        mm2=SYM_RE.match(sym)
        if not mm2: continue
        sv,side=int(mm2.group(2)),mm2.group(3)
        for i in range(len(mn)):
            m=mn[i]
            for item in t.push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=m)):
                pmtrig.setdefault(m,[]).append((side,sv,sym)+item)

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
                    if steps>pos.get("trail_steps",0):
                        pos["sl"]+=(steps-pos["trail_steps"])*TRAIL_AMOUNT_PTS
                        pos["trail_steps"]=steps
                if dpnl+(c-pos["entry"])<=DAILY_MAX_LOSS_PTS:
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
                        t1m=t1.trackers["1m"]; t1m.div.update(c,t1m.prev_s1)
                        if t1m.div.has_bearish_peak_divergence(): ex,rsn=c,"BEARISH_PEAK_REVERSAL"
                if ex is not None:
                    pts=round(ex-pos["entry"],2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":ex,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":rsn,
                        "duration_min":pos["duration_min"],"is_rev":pos["is_rev"],"tf":pos["tf"]})
                    dpnl+=pts; closs=closs+1 if pts<=0 else 0
                    if dpnl>=DAILY_MAX_PROFIT_PTS or closs>=CONSECUTIVE_LOSS_LIMIT or dpnl<=DAILY_MAX_LOSS_PTS: shut=True
                    pos=None
            if minute>=SESSION_END and pos is not None:
                pts=round(pos["last_px"]-pos["entry"],2)
                trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                    "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":pos["last_px"],
                    "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"EOD",
                    "duration_min":pos["duration_min"],"is_rev":pos["is_rev"],"tf":pos["tf"]})
                dpnl+=pts; pos=None; break
        if pos is not None or shut or minute>=SESSION_END: continue
        for item in pmtrig.get(minute,[]):
            sig_side,sig_stk,sig_sym,tf,is_rev,stype,c_px,atr_val,tf_sl,tf_tp=item
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
                        sl_p=atr_val*atr_sl if atr_val and atr_val>0.5 else tf_sl
                        tp_p=atr_val*atr_tp if atr_val and atr_val>0.5 else tf_tp
                        pos={"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                             "sl":ep-sl_p,"tgt":ep+tp_p,"entry_min":minute,
                             "last_px":ep,"duration_min":0,"is_rev":is_rev,"tf":tf}
                    elif mode=="trailing":
                        pos={"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                             "sl":ep-tf_sl,"tgt":None,"entry_min":minute,
                             "last_px":ep,"duration_min":0,"is_rev":is_rev,"tf":tf,"trail_steps":0}
                    else:
                        pos={"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                             "sl":ep-tf_sl,"tgt":ep+tf_tp,"entry_min":minute,
                             "last_px":ep,"duration_min":0,"is_rev":is_rev,"tf":tf}
                    break
    return trades


def run_config(cfg, label, days, files, spot_all):
    tasks=[(day, str(files[day]), str(files[days[i-1]]) if i>0 else "") for i,day in enumerate(days)]
    all_trades=[]
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,cfg)) as pool:
        for res in pool.map(process_day, tasks): all_trades.extend(res)
    st=summarize(all_trades)
    print(f"\n{'='*115}")
    print(f"[S1-TURNUP TRIGGER] {label}")
    print(f"{'='*115}")
    print(f"Trades: {st['trades']} | WR: {st['wr']:.1f}% | Net Pts: {st['pts']:+.2f} | Net Profit: Rs {st['rs']:+,d} | PF: {st['pf']:.2f}")
    print_yearly_breakdown(all_trades)
    return st


#  Strategy configurations 
STRATEGIES = [
    {"label":"Trailing SL +5/+10, S1=(9,3)",      "mode":"trailing","atr_sl":0,"atr_tp":0,
     "s1":(9,3),"s2":(14,3),"s3":(40,4),"s4":(60,10)},
    {"label":"S1=(12,3) + ATR2.0/4.0",          "mode":"atr",    "atr_sl":2.0,"atr_tp":4.0,
     "s1":(12,3),"s2":(14,3),"s3":(40,4),"s4":(60,10)},
    {"label":"S1=(7,3)  + ATR2.0/4.0",          "mode":"atr",    "atr_sl":2.0,"atr_tp":4.0,
     "s1":(7,3),"s2":(12,3),"s3":(21,4),"s4":(50,10)},
    {"label":"ATR2.0/4.0, S1=(9,3)",            "mode":"atr",    "atr_sl":2.0,"atr_tp":4.0,
     "s1":(9,3),"s2":(14,3),"s3":(40,4),"s4":(60,10)},
    {"label":"S1=(12,3) Fixed SL/TP",             "mode":"fixed",  "atr_sl":0,"atr_tp":0,
     "s1":(12,3),"s2":(14,3),"s3":(40,4),"s4":(60,10)},
    {"label":"Baseline Fixed SL/TP, S1=(9,3)",     "mode":"fixed",  "atr_sl":0,"atr_tp":0,
     "s1":(9,3),"s2":(14,3),"s3":(40,4),"s4":(60,10)},
]

OLD_RESULTS = {
    "Trailing SL +5/+10, S1=(9,3)": (6248, 36.6, 11323.10, 736001, 1.57),
    "S1=(12,3) + ATR2.0/4.0":    (5277, 45.9,  9922.04, 644933, 1.45),
    "S1=(7,3)  + ATR2.0/4.0":    (4487, 47.2,  9252.02, 601381, 1.48),
    "ATR2.0/4.0, S1=(9,3)":      (4843, 45.7,  8611.31, 559735, 1.41),
    "S1=(12,3) Fixed SL/TP":        (7122, 39.5,  5777.15, 375515, 1.25),
    "Baseline Fixed SL/TP, S1=(9,3)":(6467,39.4,  5109.30, 332105, 1.24),
}


def main():
    spot_all=load_spot()
    files=option_files("2020-01-01","2024-12-31")
    days=sorted(set(files.keys()) & set(spot_all.keys()))
    print(f"Loaded {len(days)} trading days.")
    print("*** NEW TRIGGER: Super Signal fires on S1 turn-up | Flag Signal keeps PinBar rule ***", flush=True)

    new_results=[]
    for cfg in STRATEGIES:
        label=cfg["label"]
        print(f"\nRunning [{label}]...", flush=True)
        t0=time.time(); st=run_config(cfg, label, days, files, spot_all)
        print(f"  Done in {time.time()-t0:.0f}s", flush=True)
        new_results.append((label, st))

    print(f"\n{'='*135}")
    print("FINAL COMPARISON: Old PinBar Trigger  vs  New S1-TurnUp Trigger (2020-2024, 5 Years)")
    print(f"{'='*135}")
    print(f"{'STRATEGY':40s} | {'OLD TRADES':9s} | {'OLD NET RS':12s} | {'OLD PF':6s} || {'NEW TRADES':9s} | {'NEW NET RS':12s} | {'NEW PF':6s} | {'CHANGE'}")
    print(f"{'-'*135}")
    for label, st in new_results:
        old=OLD_RESULTS.get(label, (0,0,0,0,0))
        delta=st['rs']-old[3]; arrow="" if delta>=0 else ""
        pf_s=f"{st['pf']:.2f}"
        print(f"{label:40s} | {old[0]:9d} | Rs {old[3]:+10,d} | {old[4]:6.2f} || {st['trades']:9d} | Rs {st['rs']:+10,d} | {pf_s:6s} | {arrow} {delta:+,d}")


if __name__=="__main__":
    main()
