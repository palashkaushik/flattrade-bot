"""6-Filter Win Rate Study — CORRECTED (smoke-tested against reference)

Baseline: Trailing SL Unlimited Pin Bar S1=(9,3) = Rs +739,193

SMOKE TEST: Set SMOKE_TEST=True to validate on 5 days before full run.
Reference engine: backtest_unlimited_profit.py (pmtrig format + bearish-peak exit)

Filters tested independently vs baseline:
  F0  Baseline (no extra filter) — must reproduce Rs +739,193
  F1  Pin Bar Quality (wick >= 4pts, wick >= 2x body, close in top 35% of range)
  F2  Power Hours Only (9:30-11:30 AM + 1:30-2:45 PM entries only)
  F3  15-Minute Spot EMA-21 Alignment (CE only when spot > 15m EMA21)
  F4  Spot 5m RSI <= 40 for CE / >= 60 for PE at signal time
  F5  Setup Freshness (tracker bar count <= 10 since setup detected)
  F6  Flag No-Divergence: S4>=80 AND S1<=20 -> immediate entry, no pin bar/div
"""

import time
from pathlib import Path
from collections import deque
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from backtest_5y_optimized import (
    load_spot, option_files, SYM_RE, to_minutes,
    latest_spot, TimeframeTracker, summarize, print_yearly_breakdown,
)
from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.indicators.stochastic import IncrementalStochastic, QuadStochastics
from flattrade_bot.indicators.divergence import DivergenceEngine

# ── Config ────────────────────────────────────────────────────────────
SMOKE_TEST = False   # ← set True to validate 5 days before full run

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_MAX_LOSS_RS  = -2000.0
DAILY_MAX_LOSS_PTS = DAILY_MAX_LOSS_RS / LOT_SIZE
CONSECUTIVE_LOSS_LIMIT = 6
ATR_PERIOD = 14
TRAIL_STEP_PTS    = 10.0
TRAIL_AMOUNT_PTS  = 5.0

# Power Hours (F2)
POWER_1_START, POWER_1_END = 570, 690   # 9:30–11:30
POWER_2_START, POWER_2_END = 810, 885   # 1:30–2:45

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


# ── ATR indicator (identical to backtest_unlimited_profit.py) ─────────
class IncrementalATR:
    def __init__(self, period=14):
        self.period=period; self._buf=deque(maxlen=period)
        self.atr=None; self.prev_close=None; self._n=0
    def update(self, h, l, c):
        tr=max(h-l,abs(h-self.prev_close),abs(l-self.prev_close)) if self.prev_close else h-l
        self._buf.append(tr); self._n+=1; self.prev_close=c
        if self._n<self.period:    self.atr=None
        elif self._n==self.period: self.atr=sum(self._buf)/self.period
        else:                      self.atr=(self.atr*(self.period-1)+tr)/self.period
        return self.atr


# ── Spot indicator helpers (for F3 EMA and F4 RSI) ───────────────────
class IncrementalEMA:
    def __init__(self, period):
        self.k=2/(period+1); self.ema=None
    def push(self, v):
        self.ema=v if self.ema is None else v*self.k+self.ema*(1-self.k)
        return self.ema

class IncrementalRSI:
    def __init__(self, period=14):
        self.period=period; self._n=0
        self.avg_gain=0.0; self.avg_loss=0.0; self.prev=None; self.rsi=None
    def push(self, close):
        if self.prev is None: self.prev=close; return None
        delta=close-self.prev; self.prev=close; self._n+=1
        gain=max(delta,0); loss=max(-delta,0)
        k=self.period
        self.avg_gain=(self.avg_gain*(k-1)+gain)/k
        self.avg_loss=(self.avg_loss*(k-1)+loss)/k
        if self._n<k: return None
        self.rsi=100.0 if self.avg_loss==0 else 100-100/(1+self.avg_gain/self.avg_loss)
        return self.rsi

def build_spot_indicators(spot_day):
    """Per-minute lookup for 15m EMA21 and 5m RSI14."""
    if spot_day is None: return {}, {}
    mins=spot_day["min"]; closes=spot_day["close"]
    # 5m RSI
    rsi_calc=IncrementalRSI(14); rbuf=[]; rsi5_map={}
    for m,c in zip(mins,closes):
        rbuf.append(c)
        if len(rbuf)==5: rsi5_map[m]=rsi_calc.push(rbuf[-1]); rbuf=[]
    last_r=None
    for m in range(SESSION_START,DAY_LAST+1):
        v=rsi5_map.get(m)
        if v is not None: last_r=v
        rsi5_map[m]=last_r
    # 15m EMA21
    ema21=IncrementalEMA(21); ebuf=[]; ema15_map={}
    for m,c in zip(mins,closes):
        ebuf.append(c)
        if len(ebuf)==15: ema15_map[m]=ema21.push(ebuf[-1]); ebuf=[]
    last_e=None
    for m in range(SESSION_START,DAY_LAST+1):
        v=ema15_map.get(m)
        if v is not None: last_e=v
        ema15_map[m]=last_e
    return rsi5_map,ema15_map


# ── F6: Flag No-Divergence scanner (runs alongside normal tracker) ───
class FlagNoDivScanner:
    """Emits immediate trigger when S4>=79.5 AND S1<=20.5, no divergence/pin bar."""
    def __init__(self):
        self.s1=IncrementalStochastic(9,3)
        self.s4=IncrementalStochastic(60,10)
        self._fired_this_setup=False   # one trade per flag episode
    def push(self, h, l, c):
        s1_val=self.s1.push(h,l,c)
        s4_val=self.s4.push(h,l,c)
        if s1_val is None or s4_val is None: return False
        is_flag=s4_val>=79.5 and s1_val<=20.5
        if is_flag and not self._fired_this_setup:
            self._fired_this_setup=True
            return True
        if not is_flag:
            self._fired_this_setup=False
        return False


class MultiTFTracker:
    """Standard tracker (identical to backtest_unlimited_profit.py MultiTimeframeTracker)."""
    def __init__(self, flag_nodiv=False):
        self.trackers={tf:TimeframeTracker(tf,spec[1]) for tf,spec in TF_SPECS.items()}
        self.atrs={tf:IncrementalATR(ATR_PERIOD) for tf in TF_SPECS}
        self.bufs={tf:[] for tf in TF_SPECS}
        # F6 scanners per TF
        self.flag_scanners={tf:FlagNoDivScanner() for tf in TF_SPECS} if flag_nodiv else {}
        self.flag_nodiv=flag_nodiv
    def push_1m(self, c1m: Candle):
        out=[]
        for tf,spec in TF_SPECS.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf])==spec[0]:
                buf=self.bufs[tf]; self.bufs[tf]=[]
                ctf=Candle(open=buf[0].open,high=max(x.high for x in buf),
                           low=min(x.low for x in buf),close=buf[-1].close,minute=buf[-1].minute)
                atr_val=self.atrs[tf].update(ctf.high,ctf.low,ctf.close)
                # Standard pin bar + divergence trigger
                trig,is_rev,stype,px=self.trackers[tf].push(ctf)
                if trig: out.append((tf,is_rev,stype,px,atr_val))
                # F6: Flag no-divergence immediate trigger
                if self.flag_nodiv and tf in self.flag_scanners:
                    if self.flag_scanners[tf].push(ctf.high,ctf.low,ctf.close):
                        out.append((tf,False,"flag_nodiv",ctf.close,atr_val))
        return out


# ── Process one day ───────────────────────────────────────────────────
def process_day(args):
    day,fpath,fprev=args
    cfg=GLOBAL_CONFIG; spot=GLOBAL_SPOT.get(day)
    filter_id=cfg["filter_id"]
    flag_nodiv=(filter_id=="F6")

    if spot is None or not fpath: return []
    fp=Path(fpath)
    if not fp.exists(): return []
    sp0=latest_spot(spot,555) or latest_spot(spot,560)
    if sp0 is None: return []
    atm0=int(round(sp0/50)*50)
    target_strikes=set(range(atm0-250,atm0+300,50))
    try:
        dfc=pd.read_csv(fp,usecols=["time","symbol","open","high","low","close"],engine="c")
    except: return []
    if dfc.empty: return []
    fsym=dfc["symbol"].iloc[0]; mm=SYM_RE.match(fsym)
    if not mm: return []
    prefix=mm.group(1)
    dfc["min"]=np.array([to_minutes(t) for t in dfc["time"]])
    gc={sym:g for sym,g in dfc.groupby("symbol")
        if (m:=SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}
    gp={}
    if fprev and Path(fprev).exists():
        try:
            dfp=pd.read_csv(fprev,usecols=["time","symbol","open","high","low","close"],engine="c")
            if not dfp.empty:
                dfp["min"]=np.array([to_minutes(t) for t in dfp["time"]])
                gp={sym:g for sym,g in dfp.groupby("symbol")
                    if (m:=SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}
        except: pass

    # Build spot indicators for F3/F4
    rsi5_map,ema15_map={},{}
    if filter_id in ("F3","F4"):
        rsi5_map,ema15_map=build_spot_indicators(spot)

    trk={}
    for sym,g in gp.items():
        trk[sym]=MultiTFTracker(flag_nodiv=flag_nodiv)
        mn,op,hi,lo,cl=(g[c].to_numpy() for c in ["min","open","high","low","close"])
        for i in range(len(mn)): trk[sym].push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=mn[i]))

    pmtrig={}; slices={}
    for sym,g in gc.items():
        if sym not in trk: trk[sym]=MultiTFTracker(flag_nodiv=flag_nodiv)
        t=trk[sym]
        mn,op,hi,lo,cl=(g[c].to_numpy() for c in ["min","open","high","low","close"])
        slices[sym]={"min":mn,"open":op,"high":hi,"low":lo,"close":cl}
        mm2=SYM_RE.match(sym)
        if not mm2: continue
        sv,side=int(mm2.group(2)),mm2.group(3)
        for i in range(len(mn)):
            m=mn[i]
            # ── IDENTICAL tuple format to backtest_unlimited_profit.py line 128 ──
            for (tf,is_rev,stype,px,atr_val) in t.push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=m)):
                pmtrig.setdefault(m,[]).append((side,sv,sym,px,is_rev,tf,TF_SPECS[tf][2],TF_SPECS[tf][3],atr_val))

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

    trades=[]; pos=None; dpnl=0.0; closs=0; shut=False
    for minute in range(SESSION_START,DAY_LAST+1):
        if pos is not None:
            held=bslice(pos["slice"],minute)
            if held:
                o,h,l,c=held; pos["last_px"]=float(c); pos["duration_min"]+=1
                # Trailing SL update
                gain=c-pos["entry"]; steps=int(gain/TRAIL_STEP_PTS)
                if steps>pos["trail_steps"]:
                    pos["sl"]+=(steps-pos["trail_steps"])*TRAIL_AMOUNT_PTS
                    pos["trail_steps"]=steps
                # Daily max loss check
                if dpnl*LOT_SIZE+(c-pos["entry"])*LOT_SIZE<=DAILY_MAX_LOSS_RS:
                    pts=round(c-pos["entry"],2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":c,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"SHUTDOWN_LOSS",
                        "duration_min":pos["duration_min"],"tf":pos["tf"]})
                    dpnl+=pts; pos=None; shut=True; continue
                ex,rsn=None,""
                if l<=pos["sl"]: ex,rsn=pos["sl"],"SL"
                # Bearish peak divergence exit (from reference engine)
                if ex is None:
                    t1=trk.get(pos["symbol"])
                    if t1:
                        t1m=t1.trackers["1m"]
                        t1m.divergence.update(c,t1m.prev_s1)
                        if t1m.divergence.has_bearish_peak_divergence(): ex,rsn=c,"BEARISH_PEAK_REVERSAL"
                if ex is not None:
                    pts=round(ex-pos["entry"],2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":ex,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":rsn,
                        "duration_min":pos["duration_min"],"tf":pos["tf"]})
                    dpnl+=pts; closs=closs+1 if pts<=0 else 0
                    if closs>=CONSECUTIVE_LOSS_LIMIT or dpnl<=DAILY_MAX_LOSS_PTS: shut=True
                    pos=None
        if minute>=SESSION_END and pos is not None:
            pts=round(pos["last_px"]-pos["entry"],2)
            trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":pos["last_px"],
                "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"EOD",
                "duration_min":pos["duration_min"],"tf":pos["tf"]})
            dpnl+=pts; pos=None; break
        if pos is not None or shut or minute>=SESSION_END: continue

        # ── F2: Power Hours gate ─────────────────────────────────────
        if filter_id=="F2":
            in_power=POWER_1_START<=minute<=POWER_1_END or POWER_2_START<=minute<=POWER_2_END
            if not in_power: continue

        for (sig_side,sig_stk,sig_sym,c_px,is_rev,tf,sl_pts,tp_pts,atr_val) in pmtrig.get(minute,[]):
            ai=ainfo(sig_side,minute)
            if ai and ai[2]==sig_stk and pos is None:
                # ── F1: Pin Bar Quality ──────────────────────────────
                if filter_id=="F1":
                    bar_now=bslice(ai[1],minute) if ai[1] is not None else None
                    if bar_now:
                        _,bh,bl,bc=bar_now
                        wick=bc-bl; full_range=bh-bl
                        body=abs(bc-c_px)  # approximate body
                        if wick<4.0: continue
                        if full_range>0 and wick<2*body: continue
                        if full_range>0 and (bc-bl)/full_range<0.60: continue

                # ── F3: 15m Spot EMA-21 Alignment ───────────────────
                if filter_id=="F3":
                    ema_val=ema15_map.get(minute)
                    spx=latest_spot(spot,minute)
                    if ema_val and spx:
                        if sig_side=="CE" and spx<ema_val: continue
                        if sig_side=="PE" and spx>ema_val: continue

                # ── F4: Spot 5m RSI oversold/overbought ─────────────
                if filter_id=="F4":
                    rsi_val=rsi5_map.get(minute)
                    if rsi_val is not None:
                        if sig_side=="CE" and rsi_val>40: continue
                        if sig_side=="PE" and rsi_val<60: continue

                if is_rev:
                    as2="PE" if sig_side=="CE" else "CE"; ai2=ainfo(as2,minute)
                    if ai2 is None: continue
                    asym,asl,_=ai2
                else:
                    as2=sig_side; asym=sig_sym; asl=ai[1]
                bar=bslice(asl,minute)
                if bar:
                    ep=float(bar[3])
                    pos={"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                         "sl":ep-sl_pts,"tgt":None,
                         "entry_min":minute,"last_px":ep,"duration_min":0,"tf":tf,"trail_steps":0}
                    break
    return trades


# ── Filter configs ────────────────────────────────────────────────────
FILTERS = [
    {"filter_id":"F0","label":"F0: Baseline (no filter)                  — expect Rs +739,193"},
    {"filter_id":"F1","label":"F1: Pin Bar Quality (wick>=4, wick>=2xbody, close top 35%)"},
    {"filter_id":"F2","label":"F2: Power Hours Only (9:30-11:30 + 1:30-2:45)"},
    {"filter_id":"F3","label":"F3: 15m Spot EMA-21 Alignment"},
    {"filter_id":"F4","label":"F4: Spot 5m RSI <= 40 CE / >= 60 PE"},
    {"filter_id":"F5","label":"F5: (Setup Freshness — placeholder, same as F0)"},
    {"filter_id":"F6","label":"F6: Flag No-Div (S4>=80 + S1<=20 -> immediate entry)"},
]
BASELINE_RS = 739193


def main():
    spot_all=load_spot()
    files=option_files("2020-01-01","2024-12-31")
    days=sorted(set(files.keys()) & set(spot_all.keys()))
    if SMOKE_TEST:
        days=days[:5]
        print(f"=== SMOKE TEST — {len(days)} DAYS ONLY ===")
    print(f"Loaded {len(days)} trading days. Running 7 filters...\n",flush=True)

    results=[]
    for cfg in FILTERS:
        label=cfg["label"]
        print(f"Running [{label}]...",flush=True)
        tasks=[(day,str(files[day]),str(files[days[i-1]]) if i>0 else "") for i,day in enumerate(days)]
        t0=time.time(); all_trades=[]
        with Pool(processes=min(cpu_count(), 8),initializer=init_worker,initargs=(spot_all,cfg)) as pool:
            for res in pool.map(process_day,tasks): all_trades.extend(res)
        st=summarize(all_trades); elapsed=time.time()-t0
        delta=st["rs"]-BASELINE_RS
        print(f"  Trades:{st['trades']:5,d} | WR:{st['wr']:5.1f}% | "
              f"Rs:{st['rs']:+,d} | PF:{st['pf']:.2f} | vs baseline:{delta:+,d} | {elapsed:.0f}s",flush=True)
        results.append((cfg["filter_id"],label,st,all_trades))

    w=130
    print(f"\n{'='*w}")
    print(f"6-FILTER COMPARISON (Trailing SL Unlimited | 2020-2024)")
    print(f"{'='*w}")
    print(f"{'FILTER':60s} | {'TRADES':7} | {'WR%':6} | {'NET PROFIT':12} | {'PF':5} | CHANGE vs BASELINE")
    print(f"{'-'*w}")
    for fid,label,st,_ in results:
        delta=st["rs"]-BASELINE_RS
        chg=f"+Rs {delta:+,d}" if delta>=0 else f" Rs {delta:,d}"
        print(f"{label:60s} | {st['trades']:7,d} | {st['wr']:5.1f}% | Rs {st['rs']:+10,d} | {st['pf']:5.2f} | {chg}")

    print(f"\n{'='*w}")
    print("YEARLY BREAKDOWN — filters that BEAT BASELINE")
    for fid,label,st,all_trades in results:
        if st["rs"]>BASELINE_RS:
            print(f"\n>>> {label}")
            print_yearly_breakdown(all_trades)

    if not results:
        print("No filter beat the baseline.")
    print(f"\nBaseline = Rs +{BASELINE_RS:,}  |  Check CHANGE column for improvements.")


if __name__=="__main__":
    main()
