"""Daily Win/Loss Analysis for Full Strategy Leaderboard.
Settings: Daily MaxLoss = Rs 2,000 | MaxProfit = UNLIMITED

Computes per-strategy:
  - Winning Days (daily P&L > 0)
  - Losing Days  (daily P&L < 0)
  - Breakeven Days (daily P&L = 0)
  - Max single day gain/loss
  - Average winning/losing day
"""

import time
from pathlib import Path
from typing import List
from multiprocessing import Pool, cpu_count
from collections import deque, defaultdict

import numpy as np
import pandas as pd

from backtest_5y_optimized import load_spot, option_files, SYM_RE, to_minutes, latest_spot, TimeframeTracker, summarize
from flattrade_bot.indicators.patterns import Candle

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_MAX_LOSS_RS    = -2000.0
DAILY_MAX_LOSS_PTS   = DAILY_MAX_LOSS_RS / LOT_SIZE
DAILY_MAX_PROFIT_PTS = float("inf")
CONSECUTIVE_LOSS_LIMIT = 6
ATR_PERIOD = 14
TRAIL_STEP_PTS   = 10.0
TRAIL_AMOUNT_PTS = 5.0

TF_SPECS = {
    "1m": (1, 10, 6.0, 30.0),
    "2m": (2,  5, 10.0, 15.0),
    "3m": (3,  4, 8.0,  25.0),
    "5m": (5,  3, 10.0, 35.0),
}

GLOBAL_SPOT   = {}
GLOBAL_CONFIG = {}

def init_worker_local(sd, cfg):
    global GLOBAL_SPOT, GLOBAL_CONFIG
    GLOBAL_SPOT = sd; GLOBAL_CONFIG = cfg


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


class MTFTracker:
    def __init__(self):
        self.trackers={tf:TimeframeTracker(tf, max_lookback=spec[1]) for tf,spec in TF_SPECS.items()}
        self.atrs={tf:IncrementalATR(ATR_PERIOD) for tf in TF_SPECS}
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
                atr_val=self.atrs[tf].update(ctf.high, ctf.low, ctf.close)
                trig,is_rev,stype,px=self.trackers[tf].push(ctf)
                if trig: out.append((tf,is_rev,stype,px,atr_val,spec[2],spec[3]))
        return out


def process_day(args):
    day, fpath, fprev = args
    cfg=GLOBAL_CONFIG; spot=GLOBAL_SPOT.get(day)
    mode=cfg["mode"]; atr_sl=cfg["atr_sl"]; atr_tp=cfg["atr_tp"]
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
    for sym,g in gp.items():
        trk[sym]=MTFTracker()
        mn,op,hi,lo,cl=g["min"].to_numpy(),g["open"].to_numpy(),g["high"].to_numpy(),g["low"].to_numpy(),g["close"].to_numpy()
        for i in range(len(mn)): trk[sym].push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=mn[i]))
    pmtrig={}; slices={}
    for sym,g in gc.items():
        if sym not in trk: trk[sym]=MTFTracker()
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
                if mode=="trailing":
                    gain=c-pos["entry"]
                    steps=int(gain/TRAIL_STEP_PTS)
                    if steps>pos.get("trail_steps",0):
                        pos["sl"]+=(steps-pos["trail_steps"])*TRAIL_AMOUNT_PTS; pos["trail_steps"]=steps
                if dpnl*LOT_SIZE+(c-pos["entry"])*LOT_SIZE<=DAILY_MAX_LOSS_RS:
                    pts=round(c-pos["entry"],2)
                    trades.append({"date":day,"pts":pts,"rs":round(pts*LOT_SIZE),"reason":"SHUTDOWN_LOSS"})
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
                    trades.append({"date":day,"pts":pts,"rs":round(pts*LOT_SIZE),"reason":rsn})
                    dpnl+=pts; closs=closs+1 if pts<=0 else 0
                    if closs>=CONSECUTIVE_LOSS_LIMIT or dpnl<=DAILY_MAX_LOSS_PTS: shut=True
                    pos=None
            if minute>=SESSION_END and pos is not None:
                pts=round(pos["last_px"]-pos["entry"],2)
                trades.append({"date":day,"pts":pts,"rs":round(pts*LOT_SIZE),"reason":"EOD"})
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
                             "sl":ep-sl_p,"tgt":ep+tp_p,"entry_min":minute,"last_px":ep,"duration_min":0,"is_rev":is_rev,"tf":tf}
                    elif mode=="trailing":
                        pos={"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                             "sl":ep-tf_sl,"tgt":None,"entry_min":minute,"last_px":ep,"duration_min":0,"is_rev":is_rev,"tf":tf,"trail_steps":0}
                    else:
                        pos={"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                             "sl":ep-tf_sl,"tgt":ep+tp_p if mode=="atr" else ep+tf_tp,"entry_min":minute,"last_px":ep,"duration_min":0,"is_rev":is_rev,"tf":tf}
                    break
    return trades


def daily_stats(trades, total_days):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    daily = df.groupby("date")["rs"].sum()
    win_days  = (daily > 0).sum()
    loss_days = (daily < 0).sum()
    be_days   = (daily == 0).sum()
    no_trade_days = total_days - len(daily)
    win_amounts  = daily[daily > 0]
    loss_amounts = daily[daily < 0]
    return {
        "win_days":    win_days,
        "loss_days":   loss_days,
        "be_days":     be_days,
        "no_trade":    no_trade_days,
        "total_days":  total_days,
        "avg_win_day": win_amounts.mean() if len(win_amounts) else 0,
        "avg_loss_day":loss_amounts.mean() if len(loss_amounts) else 0,
        "max_win_day": win_amounts.max()  if len(win_amounts) else 0,
        "max_loss_day":loss_amounts.min() if len(loss_amounts) else 0,
        "day_wr":      100 * win_days / (win_days + loss_days) if (win_days+loss_days) > 0 else 0,
        "net_rs":      int(daily.sum()),
        "trades":      len(df),
    }


STRATEGIES = [
    {"label":"Trailing SL +5/+10 (No TP)",    "mode":"trailing","atr_sl":0,"atr_tp":0},
    {"label":"ATR(14) SL x2.0 / TP x4.0",     "mode":"atr",    "atr_sl":2.0,"atr_tp":4.0},
    {"label":"ATR(14) SL x2.0 / TP x3.0",     "mode":"atr",    "atr_sl":2.0,"atr_tp":3.0},
    {"label":"ATR(14) SL x1.5 / TP x3.0",     "mode":"atr",    "atr_sl":1.5,"atr_tp":3.0},
    {"label":"Baseline Fixed SL/TP",           "mode":"fixed",  "atr_sl":0,"atr_tp":0},
    {"label":"ATR(14) SL x1.0 / TP x3.0",     "mode":"atr",    "atr_sl":1.0,"atr_tp":3.0},
    {"label":"ATR(14) SL x1.5 / TP x2.0",     "mode":"atr",    "atr_sl":1.5,"atr_tp":2.0},
    {"label":"ATR(14) SL x1.0 / TP x2.0",     "mode":"atr",    "atr_sl":1.0,"atr_tp":2.0},
]


def main():
    spot_all=load_spot()
    files=option_files("2020-01-01","2024-12-31")
    days=sorted(set(files.keys()) & set(spot_all.keys()))
    total_days=len(days)
    print(f"Loaded {total_days} trading days.")
    print(f"Daily MaxLoss=Rs {abs(DAILY_MAX_LOSS_RS):,.0f} | MaxProfit=UNLIMITED\n", flush=True)

    all_results=[]
    for cfg in STRATEGIES:
        label=cfg["label"]
        print(f"Running: {label}...", flush=True)
        tasks=[(day,str(files[day]),str(files[days[i-1]]) if i>0 else "") for i,day in enumerate(days)]
        all_trades=[]
        with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,cfg)) as pool:
            for res in pool.map(process_day, tasks): all_trades.extend(res)
        ds=daily_stats(all_trades, total_days)
        all_results.append((label, ds))
        print(f"  -> WinDays:{ds['win_days']} LossDays:{ds['loss_days']} | Day WR:{ds['day_wr']:.1f}% | Net: Rs {ds['net_rs']:+,d}", flush=True)

    # ── Summary Table ──────────────────────────────────────────────
    print(f"\n{'='*145}")
    print(f"DAILY WIN/LOSS ANALYSIS | MaxLoss=Rs 2,000 | MaxProfit=UNLIMITED | 2020-2024 ({total_days} trading days)")
    print(f"{'='*145}")
    print(f"{'STRATEGY':36s} | {'TRADES':7s} | {'WIN DAYS':9s} | {'LOSS DAYS':10s} | {'BE/NONE':8s} | {'DAY WR%':8s} | {'AVG WIN DAY':12s} | {'AVG LOSS DAY':13s} | {'MAX WIN':10s} | {'MAX LOSS':10s} | NET PROFIT")
    print(f"{'-'*145}")
    for label, ds in sorted(all_results, key=lambda x: x[1]["net_rs"], reverse=True):
        other=ds['be_days']+ds['no_trade']
        print(f"{label:36s} | {ds['trades']:7d} | {ds['win_days']:9d} | {ds['loss_days']:10d} | {other:8d} | {ds['day_wr']:7.1f}% | Rs {ds['avg_win_day']:+9,.0f} | Rs {ds['avg_loss_day']:+10,.0f} | Rs {ds['max_win_day']:+8,.0f} | Rs {ds['max_loss_day']:+8,.0f} | Rs {ds['net_rs']:+,d}")

    # ── Yearly win/loss day breakdown for top 2 ──────────────────
    print(f"\n{'='*100}")
    print(f"YEARLY WIN/LOSS DAY BREAKDOWN — Top 2 Strategies")
    print(f"{'='*100}")
    for cfg in STRATEGIES[:2]:
        label=cfg["label"]
        tasks=[(day,str(files[day]),str(files[days[i-1]]) if i>0 else "") for i,day in enumerate(days)]
        all_trades=[]
        with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,cfg)) as pool:
            for res in pool.map(process_day, tasks): all_trades.extend(res)
        if not all_trades: continue
        df=pd.DataFrame(all_trades)
        df["year"]=df["date"].str[:4]
        daily=df.groupby("date")["rs"].sum().reset_index()
        daily["year"]=daily["date"].str[:4]
        print(f"\n{label}")
        print(f"{'YEAR':6s} | {'WIN DAYS':9s} | {'LOSS DAYS':10s} | {'DAY WR%':8s} | {'AVG WIN':10s} | {'AVG LOSS':10s} | {'BEST DAY':10s} | {'WORST DAY':10s} | NET PROFIT")
        print(f"{'-'*100}")
        for yr, g in daily.groupby("year"):
            wd=(g["rs"]>0).sum(); ld=(g["rs"]<0).sum()
            awd=g.loc[g["rs"]>0,"rs"].mean() if wd else 0
            ald=g.loc[g["rs"]<0,"rs"].mean() if ld else 0
            best=g["rs"].max(); worst=g["rs"].min()
            dwr=100*wd/(wd+ld) if (wd+ld)>0 else 0
            net=int(g["rs"].sum())
            print(f"{yr:6s} | {wd:9d} | {ld:10d} | {dwr:7.1f}% | Rs {awd:+8,.0f} | Rs {ald:+8,.0f} | Rs {best:+8,d} | Rs {worst:+8,d} | Rs {net:+,d}")


if __name__=="__main__":
    main()
