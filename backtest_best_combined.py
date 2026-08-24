"""Combined Best Settings Backtest:
  - S1=(12,3), S2=(14,3), S3=(40,4), S4=(60,10)  [Best Stochastic Params]
  - ATR(14) SL x2.0 | TP x4.0                    [Best ATR Exit]
"""

import re, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
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
ATR_SL_MULT = 2.0
ATR_TP_MULT = 4.0
S1_SPEC = (12, 3)
S2_SPEC = (14, 3)
S3_SPEC = (40, 4)
S4_SPEC = (60, 10)

TF_SPECS = {
    "1m": (1, 10),
    "2m": (2,  5),
    "3m": (3,  4),
    "5m": (5,  3),
}

GLOBAL_SPOT = {}
def init_worker_local(sd): 
    global GLOBAL_SPOT; GLOBAL_SPOT = sd


class IncrementalATR:
    def __init__(self, period=14):
        self.period = period
        self._buf = deque(maxlen=period)
        self.atr = None
        self.prev_close = None
        self._n = 0

    def update(self, high, low, close):
        tr = max(high-low, abs(high-self.prev_close), abs(low-self.prev_close)) if self.prev_close else high-low
        self._buf.append(tr); self._n += 1; self.prev_close = close
        if self._n < self.period:   self.atr = None
        elif self._n == self.period: self.atr = sum(self._buf)/self.period
        else:                        self.atr = (self.atr*(self.period-1)+tr)/self.period
        return self.atr


class CustomStoch:
    def __init__(self):
        self.s1 = IncrementalStochastic(*S1_SPEC)
        self.s2 = IncrementalStochastic(*S2_SPEC)
        self.s3 = IncrementalStochastic(*S3_SPEC)
        self.s4 = IncrementalStochastic(*S4_SPEC)
    def push(self, h, l, c):
        return {"s1d": self.s1.push(h,l,c), "s2d": self.s2.push(h,l,c),
                "s3d": self.s3.push(h,l,c), "s4d": self.s4.push(h,l,c)}


class TFTracker:
    def __init__(self, lb):
        self.lb = lb; self.stoch = CustomStoch(); self.div = DivergenceEngine()
        self.hist: List[Candle] = []; self.setup = False; self.stype = ""
        self.prev_s1 = None; self.s4_emb = 0; self.atr = IncrementalATR(ATR_PERIOD)
    def push(self, c: Candle):
        self.hist.append(c)
        if len(self.hist) > 40: self.hist.pop(0)
        sv = self.stoch.push(c.high, c.low, c.close)
        s1,s2,s3,s4 = sv["s1d"],sv["s2d"],sv["s3d"],sv["s4d"]
        atr_val = self.atr.update(c.high, c.low, c.close)
        self.prev_s1 = s1
        if s4 is not None: self.s4_emb = self.s4_emb+1 if s4<=20 else 0
        emb = self.s4_emb > 25
        self.div.update(c.close, s1)
        bull_div = self.div.has_bullish_trough_divergence()
        is_flag  = s4 is not None and s1 is not None and s4>=79.5 and s1<=20.5
        is_super = all(v is not None and v<=20.5 for v in (s1,s2,s3,s4))
        if (is_flag or is_super) and bull_div:
            self.setup = True; self.stype = "super" if is_super else "flag"
        is_rev = emb and self.stype=="super"
        triggered = False
        if self.setup and len(self.hist)>=2:
            if BullishPinBarDetector.check_vicinity_breakout(self.hist, self.lb):
                triggered = True; self.setup = False
        return triggered, is_rev, self.stype, c.close, atr_val


class MTFTracker:
    def __init__(self):
        self.trackers = {tf: TFTracker(lb) for tf,(tf_m,lb) in TF_SPECS.items()}
        self.bufs = {tf: [] for tf in TF_SPECS}
    def push_1m(self, c1m: Candle):
        out = []
        for tf,(tf_m,lb) in TF_SPECS.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf])==tf_m:
                buf = self.bufs[tf]
                ctf = Candle(open=buf[0].open, high=max(x.high for x in buf),
                             low=min(x.low for x in buf), close=buf[-1].close, minute=buf[-1].minute)
                self.bufs[tf] = []
                trig,is_rev,stype,px,atr_val = self.trackers[tf].push(ctf)
                if trig: out.append((tf,is_rev,stype,px,atr_val))
        return out


def process_day(args):
    day, fpath, fprev = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not fpath: return []
    fp = Path(fpath)
    if not fp.exists(): return []
    sp0 = latest_spot(spot,555) or latest_spot(spot,560)
    if sp0 is None: return []
    atm0 = int(round(sp0/50)*50)
    strikes = set(range(atm0-250, atm0+300, 50))
    try:
        dfc = pd.read_csv(fp, usecols=["time","symbol","open","high","low","close"], engine="c")
    except: return []
    if dfc.empty: return []
    fsym = dfc["symbol"].iloc[0]; mm = SYM_RE.match(fsym)
    if not mm: return []
    prefix = mm.group(1)
    dfc["min"] = np.array([to_minutes(t) for t in dfc["time"]])
    gc = {sym:g for sym,g in dfc.groupby("symbol") if (m:=SYM_RE.match(sym)) and int(m.group(2)) in strikes}
    gp = {}
    if fprev and Path(fprev).exists():
        try:
            dfp = pd.read_csv(fprev, usecols=["time","symbol","open","high","low","close"], engine="c")
            if not dfp.empty:
                dfp["min"] = np.array([to_minutes(t) for t in dfp["time"]])
                gp = {sym:g for sym,g in dfp.groupby("symbol") if (m:=SYM_RE.match(sym)) and int(m.group(2)) in strikes}
        except: pass
    trk = {}
    for sym,g in gp.items():
        trk[sym]=MTFTracker()
        mn,op,hi,lo,cl = g["min"].to_numpy(),g["open"].to_numpy(),g["high"].to_numpy(),g["low"].to_numpy(),g["close"].to_numpy()
        for i in range(len(mn)): trk[sym].push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=mn[i]))
    pmtrig = {}; slices = {}
    for sym,g in gc.items():
        if sym not in trk: trk[sym]=MTFTracker()
        t = trk[sym]
        mn,op,hi,lo,cl = g["min"].to_numpy(),g["open"].to_numpy(),g["high"].to_numpy(),g["low"].to_numpy(),g["close"].to_numpy()
        slices[sym]={"min":mn,"open":op,"high":hi,"low":lo,"close":cl}
        mm2=SYM_RE.match(sym)
        if not mm2: continue
        sv,side=int(mm2.group(2)),mm2.group(3)
        for i in range(len(mn)):
            m=mn[i]
            for (tf,is_rev,stype,px,atr_val) in t.push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=m)):
                pmtrig.setdefault(m,[]).append((side,sv,sym,px,is_rev,tf,atr_val))

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
                if dpnl+(c-pos["entry"])<=DAILY_MAX_LOSS_PTS:
                    pts=round(c-pos["entry"],2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":c,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"SHUTDOWN_LOSS",
                        "duration_min":pos["duration_min"],"is_rev":pos["is_rev"],"tf":pos["tf"]})
                    dpnl+=pts; pos=None; shut=True; continue
                ex,rsn=None,""
                if h>=pos["tgt"] and l<=pos["sl"]: ex,rsn=pos["sl"],"SL"
                elif h>=pos["tgt"]: ex,rsn=pos["tgt"],"TP"
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
        for (sig_side,sig_stk,sig_sym,c_px,is_rev,tf,atr_val) in pmtrig.get(minute,[]):
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
                    if atr_val and atr_val>0.5:
                        sl_pts=atr_val*ATR_SL_MULT; tp_pts=atr_val*ATR_TP_MULT
                    else:
                        # fallback to tf defaults
                        tf_defaults={"1m":(6.0,30.0),"2m":(10.0,15.0),"3m":(8.0,25.0),"5m":(10.0,35.0)}
                        sl_pts,tp_pts=tf_defaults.get(tf,(8.0,25.0))
                    pos={"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                         "sl":ep-sl_pts,"tgt":ep+tp_pts,
                         "entry_min":minute,"last_px":ep,"duration_min":0,"is_rev":is_rev,"tf":tf}
                    break
    return trades


def main():
    spot_all = load_spot()
    files = option_files("2020-01-01","2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))
    tasks = [(day, str(files[day]), str(files[days[i-1]]) if i>0 else "") for i,day in enumerate(days)]
    print(f"Loaded {len(days)} trading days.")
    print(f"Settings: S1={S1_SPEC} S2={S2_SPEC} S3={S3_SPEC} S4={S4_SPEC} | ATR({ATR_PERIOD}) SL x{ATR_SL_MULT} TP x{ATR_TP_MULT}", flush=True)
    t0 = time.time()
    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,)) as pool:
        for res in pool.map(process_day, tasks): all_trades.extend(res)
    elapsed = time.time()-t0
    st = summarize(all_trades)
    print(f"\n[OK] COMPLETED IN {elapsed:.2f}s across {len(days)} days!")
    print(f"\n{'='*115}")
    print(f"5-YEAR COMBINED BACKTEST: S1={S1_SPEC} + ATR({ATR_PERIOD}) SL x{ATR_SL_MULT} / TP x{ATR_TP_MULT}")
    print(f"{'='*115}")
    print(f"Total Trades : {st['trades']}")
    print(f"Win Rate     : {st['wr']:.1f}%")
    print(f"Net Points   : {st['pts']:+.2f} pts")
    print(f"Net Profit   : Rs {st['rs']:+,d}")
    pf = f"{st['pf']:.2f}" if st['pf']!=float('inf') else "INF"
    print(f"Profit Factor: {pf}")
    print_yearly_breakdown(all_trades)
    print(f"\nBreakdown by Timeframe:")
    df=pd.DataFrame(all_trades)
    print(f"{'TF':6s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'NET RS':14s} | PF")
    print("-"*70)
    for tf,g in df.groupby("tf"):
        s=summarize(g.to_dict("records"))
        pfs=f"{s['pf']:.2f}" if s['pf']!=float('inf') else "INF"
        print(f"{tf:6s} | {s['trades']:7d} | {s['wr']:8.1f}% | {s['pts']:+10.2f} | Rs {s['rs']:+12,d} | {pfs}")

if __name__=="__main__":
    main()
