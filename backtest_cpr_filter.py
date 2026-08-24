"""Backtest: CPR Width as Trade FILTER (not multiplier)

Key insight from backtest_cpr_adaptive.py:
  - CPR 0-15 pts   (306 days): WR 32.8%, Net +Rs 23,108  PROFITABLE
  - CPR 15-90 pts  (765 days): All losing
  - CPR 90-105 pts (32 days):  WR 37.5%, Net +Rs 13,825  PROFITABLE

This test:
  F-A: Trade ONLY on very narrow CPR days (< 15 pts)   [306 days/yr → 25% of days]
  F-B: Trade ONLY on very narrow CPR days (< 30 pts)   [591 days → 49% of days]
  F-C: SKIP medium CPR (15-90 pts), trade 0-15 + >90   [338 days → 28% of days]
  F-D: Trade only on wide CPR days (> 90 pts)          [84 days → 7% of days]

All use standard ATR x2.0/x4.0 | Unlimited Profit | Daily MaxLoss Rs 2,000
Baseline: ATR x2.0/x4.0 Unlimited = Rs +701,533 (6,080 trades, 45.9% WR)
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

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_MAX_LOSS_RS  = -2000.0
DAILY_MAX_LOSS_PTS = DAILY_MAX_LOSS_RS / LOT_SIZE
CONSECUTIVE_LOSS_LIMIT = 6
ATR_PERIOD = 14
ATR_SL_MULT = 2.0
ATR_TP_MULT = 4.0

DATA_DIR  = Path("C:/Websites/ammu")
SPOT_PATH = DATA_DIR / "index" / "NIFTY 50_minute.csv"

GLOBAL_SPOT   = {}
GLOBAL_CPR    = {}
GLOBAL_CONFIG = {}

TF_SPECS = {
    "1m": (1, 10, 6.0,  30.0),
    "2m": (2,  5, 10.0, 15.0),
    "3m": (3,  4, 8.0,  25.0),
    "5m": (5,  3, 10.0, 35.0),
}

def init_worker_cpr(sd, cpr_map, cfg):
    global GLOBAL_SPOT, GLOBAL_CPR, GLOBAL_CONFIG
    GLOBAL_SPOT = sd; GLOBAL_CPR = cpr_map; GLOBAL_CONFIG = cfg


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


class MTFTrackerATR:
    def __init__(self):
        self.trackers={tf:TimeframeTracker(tf,spec[1]) for tf,spec in TF_SPECS.items()}
        self.atrs={tf:IncrementalATR(ATR_PERIOD) for tf in TF_SPECS}
        self.bufs={tf:[] for tf in TF_SPECS}
    def push_1m(self, c1m: Candle):
        out=[]
        for tf,spec in TF_SPECS.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf])==spec[0]:
                buf=self.bufs[tf]; self.bufs[tf]=[]
                ctf=Candle(open=buf[0].open,high=max(x.high for x in buf),
                           low=min(x.low for x in buf),close=buf[-1].close,minute=buf[-1].minute)
                atr_val=self.atrs[tf].update(ctf.high,ctf.low,ctf.close)
                trig,is_rev,stype,px=self.trackers[tf].push(ctf)
                if trig: out.append((tf,is_rev,stype,px,atr_val,spec[2],spec[3]))
        return out


def calculate_cpr_width(high, low, close):
    pivot=(high+low+close)/3; bc=(high+low)/2; tc=2*pivot-bc
    return abs(tc-bc)

def load_daily_ohlc():
    df=pd.read_csv(SPOT_PATH,parse_dates=["date"],engine="c")
    df["day"]=df["date"].dt.strftime("%Y-%m-%d")
    return df.groupby("day").agg(high=("high","max"),low=("low","min"),close=("close","last")).reset_index()

def build_cpr_map(daily_ohlc, days):
    daily_ohlc=daily_ohlc.sort_values("day").reset_index(drop=True)
    day_to_idx={row["day"]:i for i,row in daily_ohlc.iterrows()}
    cpr_map={}
    for day in days:
        idx=day_to_idx.get(day)
        if idx is None or idx==0: cpr_map[day]=None; continue
        prev=daily_ohlc.iloc[idx-1]
        cpr_map[day]=calculate_cpr_width(prev["high"],prev["low"],prev["close"])
    return cpr_map


def process_day(args):
    day,fpath,fprev=args
    cfg=GLOBAL_CONFIG; spot=GLOBAL_SPOT.get(day); cpr_w=GLOBAL_CPR.get(day)
    filter_id=cfg["filter_id"]

    # ── CPR filter: skip the day if it doesn't match the filter ──────
    if cpr_w is None:
        cpr_w=40.0  # default to moderate if no data
    if filter_id=="FA" and cpr_w >= 15:  return []   # only < 15
    if filter_id=="FB" and cpr_w >= 30:  return []   # only < 30
    if filter_id=="FC" and 15<=cpr_w<=90: return []   # skip 15-90
    if filter_id=="FD" and cpr_w <= 90:  return []   # only > 90

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

    trk={}
    for sym,g in gp.items():
        trk[sym]=MTFTrackerATR()
        mn,op,hi,lo,cl=(g[c].to_numpy() for c in ["min","open","high","low","close"])
        for i in range(len(mn)):
            trk[sym].push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=mn[i]))

    pmtrig={}; slices={}
    for sym,g in gc.items():
        if sym not in trk: trk[sym]=MTFTrackerATR()
        t=trk[sym]
        mn,op,hi,lo,cl=(g[c].to_numpy() for c in ["min","open","high","low","close"])
        slices[sym]={"min":mn,"open":op,"high":hi,"low":lo,"close":cl}
        mm2=SYM_RE.match(sym)
        if not mm2: continue
        sv,side=int(mm2.group(2)),mm2.group(3)
        for i in range(len(mn)):
            m=mn[i]
            for item in t.push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=m)):
                pmtrig.setdefault(m,[]).append((side,sv,sym)+item)

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
                if dpnl*LOT_SIZE+(c-pos["entry"])*LOT_SIZE<=DAILY_MAX_LOSS_RS:
                    pts=round(c-pos["entry"],2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,"side":pos["side"],
                        "symbol":pos["symbol"],"entry":pos["entry"],"exit":c,"pts":pts,"rs":round(pts*LOT_SIZE),
                        "reason":"SHUTDOWN_LOSS","duration_min":pos["duration_min"],"tf":pos["tf"],"cpr_w":cpr_w})
                    dpnl+=pts; pos=None; shut=True; continue
                ex=None; rsn=""
                has_tgt=pos.get("tgt") is not None
                if has_tgt and h>=pos["tgt"] and l<=pos["sl"]: ex,rsn=pos["sl"],"SL"
                elif has_tgt and h>=pos["tgt"]: ex,rsn=pos["tgt"],"TP"
                elif l<=pos["sl"]: ex,rsn=pos["sl"],"SL"
                if ex is not None:
                    pts=round(ex-pos["entry"],2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,"side":pos["side"],
                        "symbol":pos["symbol"],"entry":pos["entry"],"exit":ex,"pts":pts,"rs":round(pts*LOT_SIZE),
                        "reason":rsn,"duration_min":pos["duration_min"],"tf":pos["tf"],"cpr_w":cpr_w})
                    dpnl+=pts; closs=closs+1 if pts<=0 else 0
                    if closs>=CONSECUTIVE_LOSS_LIMIT or dpnl<=DAILY_MAX_LOSS_PTS: shut=True
                    pos=None
        if minute>=SESSION_END and pos is not None:
            pts=round(pos["last_px"]-pos["entry"],2)
            trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,"side":pos["side"],
                "symbol":pos["symbol"],"entry":pos["entry"],"exit":pos["last_px"],"pts":pts,"rs":round(pts*LOT_SIZE),
                "reason":"EOD","duration_min":pos["duration_min"],"tf":pos["tf"],"cpr_w":cpr_w})
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
                    sl_p=atr_val*ATR_SL_MULT if (atr_val and atr_val>0.5) else tf_sl
                    tp_p=atr_val*ATR_TP_MULT if (atr_val and atr_val>0.5) else tf_tp
                    pos={"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                         "sl":ep-sl_p,"tgt":ep+tp_p,"entry_min":minute,
                         "last_px":ep,"duration_min":0,"tf":tf,"cpr_w":cpr_w}
                    break
    return trades


FILTERS = [
    {"filter_id":"FA","label":"F-A: CPR < 15 pts  ONLY  (very narrow, trending)",      "days_pct":"~25%"},
    {"filter_id":"FB","label":"F-B: CPR < 30 pts  ONLY  (narrow, trending)",           "days_pct":"~49%"},
    {"filter_id":"FC","label":"F-C: Skip 15-90 pts (trade 0-15 + >90 only)",           "days_pct":"~32%"},
    {"filter_id":"FD","label":"F-D: CPR > 90 pts  ONLY  (extreme wide, contrarian)",   "days_pct":"~7%"},
]
BASELINE_RS = 701533


def main():
    spot_all   = load_spot()
    daily_ohlc = load_daily_ohlc()
    files      = option_files("2020-01-01","2024-12-31")
    days       = sorted(set(files.keys()) & set(spot_all.keys()))
    cpr_map    = build_cpr_map(daily_ohlc, days)
    print(f"Loaded {len(days)} trading days.")
    print(f"ATR x{ATR_SL_MULT}/x{ATR_TP_MULT} | Unlimited Profit | MaxLoss Rs 2,000")
    print(f"Baseline (all days): Rs +{BASELINE_RS:,}\n", flush=True)

    results=[]
    for cfg in FILTERS:
        label=cfg["label"]
        print(f"Running [{label}]...",flush=True)
        tasks=[(day,str(files[day]),str(files[days[i-1]]) if i>0 else "") for i,day in enumerate(days)]
        t0=time.time(); all_trades=[]
        with Pool(processes=min(cpu_count(), 8),initializer=init_worker_cpr,initargs=(spot_all,cpr_map,cfg)) as pool:
            for res in pool.map(process_day,tasks): all_trades.extend(res)
        st=summarize(all_trades); elapsed=time.time()-t0
        delta=st["rs"]-BASELINE_RS
        print(f"  Days:{sum(1 for d in days if process_is_active(cpr_map[d],cfg['filter_id']))} | "
              f"Trades:{st['trades']:,} | WR:{st['wr']:.1f}% | "
              f"Rs:{st['rs']:+,d} | PF:{st['pf']:.2f} | vs baseline:{delta:+,d} | {elapsed:.0f}s",flush=True)
        print_yearly_breakdown(all_trades)
        results.append((cfg["filter_id"],label,cfg["days_pct"],st,all_trades))

    w=130
    print(f"\n{'='*w}")
    print(f"CPR FILTER COMPARISON  vs  Unrestricted ATR×2.0/×4.0 (Rs +{BASELINE_RS:,})")
    print(f"{'='*w}")
    print(f"{'FILTER':55s} | {'DAYS':5} | {'TRADES':7} | {'WR%':6} | {'NET PROFIT':12} | {'PF':5} | CHANGE")
    print(f"{'-'*w}")
    for fid,label,dpct,st,_ in results:
        delta=st['rs']-BASELINE_RS
        active=[d for d in days if process_is_active(cpr_map[d],fid)]
        print(f"{label:55s} | {len(active):5d} | {st['trades']:7,d} | {st['wr']:5.1f}% | "
              f"Rs {st['rs']:+10,d} | {st['pf']:5.2f} | {delta:+,d}")

def process_is_active(cpr_w, filter_id):
    if cpr_w is None: cpr_w=40.0
    if filter_id=="FA": return cpr_w < 15
    if filter_id=="FB": return cpr_w < 30
    if filter_id=="FC": return not (15<=cpr_w<=90)
    if filter_id=="FD": return cpr_w > 90
    return True

if __name__=="__main__":
    main()
