"""Faithful CPU replay of the chosen Optimus config (1.9/day @ 71% WR, 5m HA UT Bot)
on the Flattrade-fetched Nifty spot 1m candles for Aug 12-14, 2026.

Replicates cross_strategy_ensemble_gpu.evaluate_ensemble_batch math exactly:
  - 15 fixed components (stochastic votes, confirm_k=1)
  - 5m Heikin-Ashi + UT Bot(key=1.0,period=10) + LinReg(11) trend gate
  - ATR(29) 1m SL/TP, moneyness=0.6, re-entry, daily caps 5/-60 pts (x65)
Read-only; no orders. Output: trade list per date.
"""
import json, numpy as np
from datetime import datetime, timedelta

LOT, FEE, SLIP = 65, 30, 1.0
BASE_SESSION_START = 5
BSE = 345

# -- FIXED 15 components (from optimus_hft_cash_machine_sweep.py COMPONENTS) --
def C(tf, s1k, s4k, s4ob, s1os):
    return {"timeframe": tf, "s1_k": s1k, "s4_k": s4k, "s4_ob": s4ob, "s1_os": s1os}
COMPONENTS = [
    C(1,30,70,70.0,40.0), C(2,30,70,70.0,40.0), C(3,30,70,70.0,40.0),
    C(5,30,70,70.0,40.0), C(3,30,70,70.0,40.0), C(1,16,80,77.5,17.5),
    C(1,7,60,80.0,25.0), C(1,12,50,79.5,25.0), C(1,9,70,79.5,25.0),
    C(1,11,75,75.0,25.0), C(1,24,95,77.5,17.5), C(1,9,20,72.0,28.0),
    C(1,5,20,68.0,32.0), C(1,12,30,75.0,25.0), C(1,7,30,70.0,30.0),
]
# chosen ens params (top1, trend_filter=5)
ENS = {"timeframe":1,"atr_p":29,"sl_m":4.0,"tp_m":6.0,"daily_loss_pts":5,
       "daily_profit_pts":60,"moneyness":0.6,"sess_start_off":30,"sess_end_off":30,
       "sess_end":BSE-30,"confirm_k":1,"reentry":True,"trend_filter":5}

# -- load candles --
import os as _os
_rawpath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "nifty_spot_aug12_14_2026.json")
raw = json.load(open(_rawpath))
days = {}
for r in raw:
    dt = datetime.strptime(r["time"], "%d-%m-%Y %H:%M:%S")
    b = dt.hour*60 + dt.minute - 555
    if not (0 <= b < 375):
        continue
    d = dt.strftime("%Y-%m-%d")
    days.setdefault(d, {})[b] = (float(r["open"]),float(r["high"]),float(r["low"]),float(r["close"]))
dates = sorted(days)
N = len(dates)
O = np.zeros((N,375)); H = np.zeros((N,375)); L = np.zeros((N,375)); C_ = np.zeros((N,375))
for i,d in enumerate(dates):
    for b,(o,h,l,c) in days[d].items():
        O[i,b],H[i,b],L[i,b],C_[i,b]=o,h,l,c
print("Loaded days:", dates)

# -- TF aggregate (replicate-pad) --
def aggregate_tf(k):
    if k == 1:
        pc = np.roll(C_,1,axis=1); pc[:,0]=C_[:,0]
        tr = np.maximum(np.maximum(H-L, np.abs(H-pc)), np.abs(L-pc))
        return H,L,C_,tr
    pad = (k - 375 % k) % k
    def rpad(x):
        if pad==0: return x
        return np.concatenate([x, np.repeat(x[:,-1:], pad, axis=1)], axis=1)
    ho=rpad(H).reshape(N,-1,k); lo=rpad(L).reshape(N,-1,k); co=rpad(C_).reshape(N,-1,k)
    hh=ho.max(2); ll=lo.min(2); cc=co[:,:,-1]
    pc=np.concatenate([cc[:, :1], cc[:, :-1]], axis=1)
    tr=np.maximum(np.maximum(hh-ll, np.abs(hh-pc)), np.abs(ll-pc))
    return hh,ll,cc,tr

TF={k:aggregate_tf(k) for k in (1,2,3,5)}

def roll_max(x,p):
    N_,L_=x.shape; out=np.empty_like(x)
    for i in range(L_):
        out[:,i]=x[:,max(0,i-p+1):i+1].max(1)
    return out
def roll_min(x,p): return -roll_max(-x,p)
def roll_mean(x,p):
    N_,L_=x.shape; out=np.empty_like(x)
    for i in range(L_):
        out[:,i]=x[:,max(0,i-p+1):i+1].mean(1)
    return out

def stoch_1m(tf,period):
    h,l,c,_=TF[tf]; n=TF[tf][0].shape[1]
    maxh=roll_max(h,period); minl=roll_min(l,period)
    s=((c-minl)/(maxh-minl).clip(min=1e-6))*100.0
    if tf==1: return s
    out=np.empty((N,375))
    for i in range(n):
        a=i*tf; out[:,a:min(a+tf,375)]=s[:,i:i+1]
    return out
def atr_1m(tf,period):
    _,_,_,tr=TF[tf]; n=tr.shape[1]
    a=roll_mean(tr,period)
    if tf==1: return a
    out=np.empty((N,375))
    for i in range(n):
        st=i*tf; out[:,st:min(st+tf,375)]=a[:,i:i+1]
    return out

# -- component votes --
S1=[stoch_1m(c["timeframe"],c["s1_k"]) for c in COMPONENTS]
S4=[stoch_1m(c["timeframe"],c["s4_k"]) for c in COMPONENTS]
comp_vw=np.zeros(375,dtype=bool); comp_vw[BASE_SESSION_START+5:300]=True  # component sess [10:300)
ce_vote=np.zeros((N,375)); pe_vote=np.zeros((N,375))
for ci,c in enumerate(COMPONENTS):
    s1,s4=S1[ci],S4[ci]; ob=c["s4_ob"]; os_=c["s1_os"]
    ce=(s4>=ob)&(s1<=os_)&comp_vw
    pe=(s4<=(100-ob))&(s1>=(100-os_))&comp_vw
    ce_vote+=ce; pe_vote+=pe
ck=ENS["confirm_k"]
ce_ens=ce_vote>=ck; pe_ens=pe_vote>=ck

# -- 5m HA UT Bot trend filter --
htf=5; n_h=375//htf
ho=H.reshape(N,n_h,htf); lo=L.reshape(N,n_h,htf); co=C_.reshape(N,n_h,htf)
hopen=ho[:,:,0]; hhigh=ho.max(2); hlow=lo.min(2); hclose=co[:,:,-1]
ha_o=np.zeros((N,n_h)); ha_c=np.zeros((N,n_h))
ha_o[:,0]=(hopen[:,0]+hclose[:,0])/2; ha_c[:,0]=(hopen[:,0]+hhigh[:,0]+hlow[:,0]+hclose[:,0])/4
for i in range(1,n_h):
    ha_o[:,i]=(ha_o[:,i-1]+ha_c[:,i-1])/2
    ha_c[:,i]=(hopen[:,i]+hhigh[:,i]+hlow[:,i]+hclose[:,i])/4
prev=hclose[:,0].copy(); tr=np.zeros((N,n_h)); tr[:,0]=np.maximum(np.maximum(hhigh[:,0]-hlow[:,0],np.abs(hhigh[:,0]-prev)),np.abs(hlow[:,0]-prev))
atr=np.zeros((N,n_h)); run=tr[:,0].copy(); atr[:,0]=run
for i in range(1,n_h):
    pc=hclose[:,i-1]
    t=tr[:,i]=np.maximum(np.maximum(hhigh[:,i]-hlow[:,i],np.abs(hhigh[:,i]-pc)),np.abs(hlow[:,i]-pc))
    if i<10: run=(run*i+t)/(i+1)
    else: run=(run*9+t)/10
    atr[:,i]=run
trailing=np.zeros(N); prevsrc=np.zeros(N); pos=np.zeros(N)
color=np.zeros((N,n_h),dtype=int)
for i in range(n_h):
    src=hclose[:,i]; loss=1.0*atr[:,i]; pstop=trailing.copy(); psrc=prevsrc.copy()
    new=np.where((src>pstop)&(psrc>pstop),np.maximum(pstop,src-loss),
                 np.where((src<pstop)&(psrc<pstop),np.minimum(pstop,src+loss),
                          np.where(src>pstop,src-loss,src+loss)))
    up=(psrc<pstop)&(src>pstop); dn=(psrc>pstop)&(src<pstop)
    p=np.where(up,1,np.where(dn,-1,pos)); color[:,i]=np.where(p==1,1,np.where(p==-1,-1,0))
    trailing,prevsrc,pos=new,src,p
lin=np.zeros((N,n_h))
for i in range(n_h): lin[:,i]=ha_c[:,max(0,i-10):i+1].mean(1)
bull=(ha_c>lin)&(color==1); bear=(ha_c<lin)&(color==-1)
bull1=np.repeat(bull,htf,axis=1)[:,:375]; bear1=np.repeat(bear,htf,axis=1)[:,:375]
ce_ens=ce_ens&bull1; pe_ens=pe_ens&bear1

# diagnostics
for n,d_ in enumerate(dates):
    nb=days[d_]; 
    print(f"[diag] {d_}: bars={len(nb)} close[{C_[n].min():.0f}-{C_[n].max():.0f}] "
          f"ce_votes={int(ce_vote[n].sum())} pe_votes={int(pe_vote[n].sum())} "
          f"ce_ens(trend)={int(ce_ens[n].sum())} pe_ens(trend)={int(pe_ens[n].sum())} "
          f"bull_bars={int(bull1[n].sum())} bear_bars={int(bear1[n].sum())}", flush=True)


# -- SL/TP from ens ATR(29) 1m --
ATR=atr_1m(ENS["timeframe"],ENS["atr_p"])
off=(ENS["moneyness"]-0.5)*2*ATR
ce_sl=C_-off-ATR*ENS["sl_m"]; ce_tp=C_-off+ATR*ENS["tp_m"]
pe_sl=C_+off+ATR*ENS["sl_m"]; pe_tp=C_+off-ATR*ENS["tp_m"]
sess_end=ENS["sess_end"]; eod=min(sess_end-1,374)
DL=ENS["daily_loss_pts"]*LOT; DP=ENS["daily_profit_pts"]*LOT

def bt(t, direction):
    trades=[]
    for n in range(N):
        avail=0
        for bar in range(375):
            if direction=="CE":
                fire=ce_ens[n,bar]&(avail<=bar)
            else:
                fire=pe_ens[n,bar]&(avail<=bar)
            if not fire: continue
            if bar+1>=sess_end: continue
            ci=np.arange(bar+1,min(sess_end,375))
            fh=H[n,ci]; fl=L[n,ci]
            if direction=="CE":
                slp=ce_sl[n,bar]; tpp=ce_tp[n,bar]
                hit_sl=fl<=slp; hit_tp=fh>=tpp
                entry_eff=C_[n,bar]+0.5*SLIP; ex_sl=slp-0.5*SLIP; ex_tp=tpp-0.5*SLIP
                ex_eod=C_[n,eod]-0.5*SLIP
            else:
                slp=pe_sl[n,bar]; tpp=pe_tp[n,bar]
                hit_sl=fh>=slp; hit_tp=fl<=tpp
                entry_eff=C_[n,bar]-0.5*SLIP; ex_sl=slp+0.5*SLIP; ex_tp=tpp+0.5*SLIP
                ex_eod=C_[n,eod]+0.5*SLIP
            sl_any=hit_sl.any(); tp_any=hit_tp.any()
            sl_f=ci[hit_sl.argmax()] if sl_any else 999999
            tp_f=ci[hit_tp.argmax()] if tp_any else 999999
            if sl_any and sl_f<=tp_f:
                exb=sl_f; pxp=ex_sl; reason="SL"
            elif tp_any:
                exb=tp_f; pxp=ex_tp; reason="TP"
            else:
                exb=eod; pxp=ex_eod; reason="EOD"
            raw=(pxp-entry_eff)*0.5 if direction=="CE" else (entry_eff-pxp)*0.5
            pnl=raw*LOT-FEE
            trades.append((n,bar,exb,direction,C_[n,bar],slp,tpp,pxp,reason,pnl))
            avail=bar+1+(exb-bar)
    return trades

allt=bt(0,"CE")+bt(0,"PE")
# finalize daily caps
allt.sort(key=lambda x:(x[0],x[1]))
# HONEST accounting: the cap-triggering trade DID happen (loss is real), then day halts.
kept=[]; last=None; cum=0.0; stopped=False
for t in allt:
    n=t[0]
    if n!=last: last=n; cum=0.0; stopped=False
    if stopped: continue
    new_cum=cum+t[9]
    if (new_cum<-DL) or (new_cum)>DP:
        kept.append(t); stopped=True; cum=new_cum; continue   # count real loss/win, then halt
    cum=new_cum; kept.append(t)
kept.sort(key=lambda x:(x[0],x[1]))

def fmt(b):
    tot=9*60+15+b; return f"{tot//60:02d}:{tot%60:02d}"
print(f"\n{'='*92}\nTRADES  |  config: 5m HA UT Bot, sl_m=4 tp_m=6 ATR29 mon=0.6, caps {ENS['daily_loss_pts']}/{-ENS['daily_profit_pts']} pts x{LOT}\n{'='*92}")
for n,d_ in enumerate(dates):
    dt=[t for t in kept if t[0]==n]
    uncapped=len([t for t in allt if t[0]==n])
    print(f"\n-- {d_}  (kept trades={len(dt)}, raw signals simulated={uncapped}) --")
    daynet=0.0; wins=0
    for t in dt:
        daynet+=t[9]
        if t[9]>0: wins+=1
        print(f"  {fmt(t[1])} {t[3]:>2}  entry={t[4]:.2f}  SL={t[5]:.2f}  TP={t[6]:.2f}  "
              f"exit@{fmt(t[2])} {t[8]:<3} px={t[7]:.2f}  PnL=Rs{t[9]:+.0f}")
    wr=wins/len(dt)*100 if dt else 0
    print(f"   day net Rs{daynet:+.0f}  WR {wr:.0f}%")
tot=sum(t[9] for t in kept)
print(f"\nTOTAL trades={len(kept)}  net Rs{tot:+.0f}")


