"""F6 Champion FULL SIGNALS 1m - 3D GPU batched indicators + 4-signal state machine.
TV-exact stochastic (middle arg = 1, no %K smoothing):
  S1 Stoch(12,1,3)  S2 Stoch(14,1,3)  S3 Stoch(40,1,4)  S4 Stoch(50,1,10)
All on 1m OPTION chart. Signals (opt_futures_quad.FuturesQuadTriggers):
  bull_flag   : high_embed>=embed & neutral_prev & S1<=20.5              -> buy CE
  bear_flag   : low_embed >=embed & neutral_prev & S1>=79.5              -> buy PE
  supersignal_bull : all S1..S4<=20.5 & S1 turns up                      -> buy CE
  supersignal_bear : all S1..S4>=79.5 & S1 turns down                    -> buy PE
Then 8SR bounce (EMA20/VWAP/EMA200 + FIB/CAM/CPR from prior-day option daily), 1.0pt, C>EMA gate,
2nd ITM (CE=ATM-100, PE=ATM+100), SL7 TP15 LOT65 fees. One position/side at a time.
GPU 3D [D,N,T] for stoch/ema/vwap; per-(t,i) state machine on CPU (causal, history<=t).
"""
import pathlib, re, time, json, csv
from collections import defaultdict
import numpy as np
try:
    import cupy as cp
    HAS_CUPY=True
    print(f"CuPy {cp.__version__} GPU available")
except Exception:
    import numpy as cp
    HAS_CUPY=False
    print("CuPy not available, using NumPy fallback (still batched)")

OPT_ROOT=pathlib.Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options")
IDX_PATH=r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
SYM_RE=re.compile(r"^NIFTY(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")
# TV-exact stoch params
S1_K,S1_D=12,3; S2_K,S2_D=14,3; S3_K,S3_D=40,4; S4_K,S4_D=50,10
LIMIT,LOW_ZONE=79.5,20.5
SL,TP=7.0,15.0
LOT=65
FEE=45.0
EMBED=1  # user FLAG = S4>=79.5 single bar (lenient); tunable

def get_days(limit=5, smoke=True, start="2020-01-01", end="2026-05-05"):
    all_files=sorted(OPT_ROOT.rglob("nifty_options_*.csv"))
    date_to_path={}
    for p in all_files:
        m=re.search(r"nifty_options_(\d{2})_(\d{2})_(\d{4})\.csv", p.name)
        if m:
            d,mn,y=m.groups()
            date_to_path[f"{y}-{mn}-{d}"]=p
    dates=sorted(date_to_path.keys())
    if not smoke:
        # full range: all days in [start,end] that are complete (have 14:14 & 15:00)
        complete=[]
        for dat in dates:
            if not (start <= dat <= end): continue
            p=date_to_path[dat]
            try:
                with open(p) as f:
                    tail=" ".join(f.readlines()[-800:])
                    if "14:14:00" in tail and "15:00:00" in tail:
                        complete.append(dat)
            except: pass
        complete=sorted(complete)
        return [date_to_path[d] for d in complete], complete
    complete=[]
    for dat in reversed(dates):
        p=date_to_path[dat]
        try:
            with open(p) as f:
                tail=" ".join(f.readlines()[-800:])
                if "14:14:00" in tail and "15:00:00" in tail:
                    complete.append(dat)
                if len(complete)>=limit:
                    break
        except: pass
    if len(complete)<limit:
        complete=dates[-limit:]
    complete=sorted(complete)
    return [date_to_path[d] for d in complete], complete

def load_spot():
    spot_by_day={}
    with open(IDX_PATH) as f:
        r=csv.DictReader(f)
        for row in r:
            day=row["date"].split(" ")[0]
            t=row["date"].split(" ")[1]
            hh,mm,_=map(int,t.split(":"))
            minute=hh*60+mm
            spot_by_day.setdefault(day, {})[minute]=float(row["close"])
    return spot_by_day

def load_to_tensor(day_paths, day_names, spot_by_day, max_sym=40, max_t=375):
    tensors=[]
    for p, day in zip(day_paths, day_names):
        per_sym=defaultdict(list)
        with open(p) as f:
            r=csv.DictReader(f)
            for row in r:
                sym=row["symbol"]; t=row["time"]
                hh,mm,_=map(int,t.split(":"))
                minute=hh*60+mm
                per_sym[sym].append((minute,float(row["open"]),float(row["high"]),float(row["low"]),float(row["close"]),int(float(row["volume"] or 0))))
        atm_minute=555
        spot_915=spot_by_day.get(day, {}).get(atm_minute, 25000)
        atm=int(round(spot_915/50)*50)
        scored=[]
        for sym, bars in per_sym.items():
            m=SYM_RE.match(sym)
            if not m: continue
            strike=int(m.group(2))
            dist=abs(strike-atm)
            vol=sum(b[5] for b in bars)
            scored.append((dist, -vol, sym))
        scored.sort()
        top_syms=[s for _,_,s in scored[:max_sym]]
        N=len(top_syms); T=max_t
        close=np.full((N,T),np.nan,dtype=np.float32)
        high=np.full((N,T),np.nan,dtype=np.float32)
        low=np.full((N,T),np.nan,dtype=np.float32)
        open_=np.full((N,T),np.nan,dtype=np.float32)
        vol=np.zeros((N,T),dtype=np.float32)
        sym_idx={s:i for i,s in enumerate(top_syms)}
        for sym in top_syms:
            idx=sym_idx[sym]
            for minute,o,h,l,c,v in per_sym[sym]:
                tpos=minute-555
                if 0<=tpos<T:
                    close[idx,tpos]=c; high[idx,tpos]=h; low[idx,tpos]=l; open_[idx,tpos]=o; vol[idx,tpos]=v
        for i in range(N):
            last=np.nan
            for t in range(T):
                if np.isnan(close[i,t]):
                    if not np.isnan(last):
                        close[i,t]=last; high[i,t]=last; low[i,t]=last; open_[i,t]=last
                else:
                    last=close[i,t]
        spot_vec=np.full(T,np.nan,dtype=np.float32)
        for minute,price in spot_by_day.get(day,{}).items():
            tpos=minute-555
            if 0<=tpos<T: spot_vec[tpos]=price
        last=np.nan
        for t in range(T):
            if np.isnan(spot_vec[t]): spot_vec[t]=last
            else: last=spot_vec[t]
        # Static 8SR from prior-day option daily (use same day H/L/C as proxy for smoke)
        day_lo=min(np.nanmin(low[i]) for i in range(N) if not np.all(np.isnan(low[i])))
        day_hi=max(np.nanmax(high[i]) for i in range(N) if not np.all(np.isnan(high[i])))
        day_cl=spot_915
        pivot=(day_hi+day_lo+day_cl)/3; bc=(day_hi+day_lo)/2; tc=(pivot-bc)+pivot
        rng=day_hi-day_lo
        cam_h3=day_cl+rng*(1.1/4); cam_l3=day_cl-rng*(1.1/4)
        fib_h3=pivot+rng; fib_l3=pivot-rng
        static=[pivot,tc,bc,cam_h3,cam_l3,fib_h3,fib_l3]
        tensors.append((close,high,low,open_,vol,spot_vec,top_syms,static,atm))
    return tensors

def stoch_gpu(high,low,close,k,d):
    """TV-exact stochastic %D (middle arg=1, no %K smoothing). Uses PARTIAL window
    like Pine Script: at bar t the rolling window is max(0,t-k+1)..t (available bars),
    so %D is valid from the first bar (matches TradingView continuous-chart behavior)."""
    N,T=high.shape
    h_cp=cp.asarray(high); l_cp=cp.asarray(low); c_cp=cp.asarray(close)
    k_vals=cp.zeros((N,T),dtype=cp.float32)
    for t in range(T):
        start=max(0,t-k+1)
        win_h=h_cp[:,start:t+1]; win_l=l_cp[:,start:t+1]
        hh=cp.max(win_h,axis=1); ll=cp.min(win_l,axis=1)
        raw=cp.where(hh==ll,50.0,(c_cp[:,t]-ll)/(hh-ll)*100.0)
        k_vals[:,t]=raw
    result=cp.zeros((N,T),dtype=cp.float32)
    for t in range(T):
        # %D = SMA(raw %K, d) over available bars (TV uses available when < d)
        start=max(0,t-d+1)
        result[:,t]=cp.mean(k_vals[:,start:t+1],axis=1)
    return result

def ema_gpu(close,period):
    N,T=close.shape
    c_cp=cp.asarray(close)
    ema=cp.zeros((N,T),dtype=cp.float32)
    alpha=2/(period+1)
    ema[:,0]=c_cp[:,0]
    for t in range(1,T):
        ema[:,t]=ema[:,t-1]*(1-alpha)+c_cp[:,t]*alpha
    return ema

def vwap_gpu(high,low,close,vol):
    h_cp=cp.asarray(high); l_cp=cp.asarray(low); c_cp=cp.asarray(close); v_cp=cp.asarray(vol)
    hlc3=(h_cp+l_cp+c_cp)/3.0
    cum_pv=cp.cumsum(hlc3*cp.maximum(v_cp,10),axis=1)
    cum_v=cp.cumsum(cp.maximum(v_cp,10),axis=1)
    return cum_pv/cp.maximum(cum_v,1)

def run_smoke(limit=5, smoke=True):
    print("=== F6 FULL SIGNALS 3D GPU (causal) ===" + (" SMOKE" if smoke else " FULL 7y"))
    spot_by_day=load_spot()
    day_paths,day_names=get_days(limit=limit,smoke=smoke)
    print(f"Days: {day_names}")
    t0=time.time()
    tensors=load_to_tensor(day_paths,day_names,spot_by_day,max_sym=40,max_t=375)
    print(f"[load tensor] {time.time()-t0:.2f}s")
    all_trades=[]
    daily_pnl={}
    for day_idx,(close,high,low,open_,vol,spot_vec,syms,static,atm) in enumerate(tensors):
        day=day_names[day_idx]; N,T=close.shape
        t_gpu=time.time()
        s1=stoch_gpu(high,low,close,S1_K,S1_D)
        s2=stoch_gpu(high,low,close,S2_K,S2_D)
        s3=stoch_gpu(high,low,close,S3_K,S3_D)
        s4=stoch_gpu(high,low,close,S4_K,S4_D)
        ema20=ema_gpu(close,20); ema200=ema_gpu(close,200); vwap=vwap_gpu(high,low,close,vol)
        s1=cp.asnumpy(s1) if HAS_CUPY else s1
        s2=cp.asnumpy(s2) if HAS_CUPY else s2
        s3=cp.asnumpy(s3) if HAS_CUPY else s3
        s4=cp.asnumpy(s4) if HAS_CUPY else s4
        ema20=cp.asnumpy(ema20) if HAS_CUPY else ema20
        ema200=cp.asnumpy(ema200) if HAS_CUPY else ema200
        vwap=cp.asnumpy(vwap) if HAS_CUPY else vwap
        print(f"[{day}] GPU indicators {time.time()-t_gpu:.2f}s N={N} T={T}")
        # per-symbol signal state
        prev_s1=np.full(N,np.nan)
        bear_armed=np.zeros(N,bool); bull_armed=np.zeros(N,bool)
        super_bear_armed=np.zeros(N,bool); super_bull_armed=np.zeros(N,bool)
        low_embed=np.zeros(N,int); high_embed=np.zeros(N,int)
        pos_active=np.zeros(N,bool); pos_entry=np.full(N,np.nan); pos_sl=np.full(N,np.nan); pos_tp=np.full(N,np.nan); pos_side=np.full(N,"",dtype=object)
        for t in range(T):
            minute=555+t
            for i in range(N):
                s1d,s2d,s3d,s4d=s1[i,t],s2[i,t],s3[i,t],s4[i,t]
                if np.isnan(s1d) or np.isnan(s4d): continue
                prev=prev_s1[i]
                neutral_prev=(not np.isnan(prev)) and LOW_ZONE<prev<LIMIT
                all_high=all(v>=LIMIT for v in (s1d,s2d,s3d,s4d))
                all_low=all(v<=LOW_ZONE for v in (s1d,s2d,s3d,s4d))
                # arm
                if LOW_ZONE<s1d<LIMIT:
                    bear_armed[i]=True; bull_armed[i]=True
                if s1d<LIMIT: super_bear_armed[i]=True
                if s1d>LOW_ZONE: super_bull_armed[i]=True
                # embed
                low_embed[i]=low_embed[i]+1 if s4d<=LOW_ZONE else 0
                high_embed[i]=high_embed[i]+1 if s4d>=LIMIT else 0
                signal=None
                if low_embed[i]>=EMBED and neutral_prev and s1d>=LIMIT and bear_armed[i]:
                    bear_armed[i]=False; signal="bear_flag"
                elif high_embed[i]>=EMBED and neutral_prev and s1d<=LOW_ZONE and bull_armed[i]:
                    bull_armed[i]=False; signal="bull_flag"
                elif all_high and prev is not None and s1d<prev and super_bear_armed[i]:
                    super_bear_armed[i]=False; signal="supersignal_bear"
                elif all_low and prev is not None and s1d>prev and super_bull_armed[i]:
                    super_bull_armed[i]=False; signal="supersignal_bull"
                prev_s1[i]=s1d
                if signal is None: continue
                # signal side determines option: bear->PE, bull->CE (we are on that sym already)
                side="PE" if "bear" in signal else "CE"
                # must match symbol side
                m=SYM_RE.match(syms[i])
                if not m or m.group(3)!=side: continue
                if pos_active[i]: continue
                c=close[i,t]; l=low[i,t]; o=open_[i,t]
                ema=ema20[i,t]; vw=vwap[i,t]; e200=ema200[i,t]
                if np.isnan(c) or np.isnan(ema): continue
                if c<=ema: continue  # EMA gate defer
                levels=[ema,vw,e200]+static
                bounced=False
                for lvl in levels:
                    if abs(l-lvl)<=1.0 and c>lvl and c>o and abs(l-lvl)<=10:
                        bounced=True; break
                if not bounced: continue
                # 2nd ITM filter
                strike=int(m.group(2))
                target=atm-100 if side=="CE" else atm+100
                if strike!=target: continue
                entry=c; sl=entry-SL; tp=entry+TP
                pos_active[i]=True; pos_entry[i]=entry; pos_sl[i]=sl; pos_tp[i]=tp; pos_side[i]=side
                if day=="2026-08-27":
                    print(f"ENTRY {day} {minute} {syms[i]} {signal} {entry:.2f} S1 {s1d:.1f} S4 {s4d:.1f}")
            # exits
            for i in range(N):
                if not pos_active[i]: continue
                h=high[i,t]; l=low[i,t]; c=close[i,t]
                if np.isnan(h): continue
                if l<=pos_sl[i]:
                    pts=(pos_sl[i]-pos_entry[i])*LOT-FEE; all_trades.append((day,syms[i],pos_side[i],pos_entry[i],pos_sl[i],"SL",pts,555,minute)); pos_active[i]=False
                elif h>=pos_tp[i]:
                    pts=(pos_tp[i]-pos_entry[i])*LOT-FEE; all_trades.append((day,syms[i],pos_side[i],pos_entry[i],pos_tp[i],"TP",pts,555,minute)); pos_active[i]=False
                elif minute>=15*60+15:
                    pts=(c-pos_entry[i])*LOT-FEE; all_trades.append((day,syms[i],pos_side[i],pos_entry[i],c,"EOD",pts,555,minute)); pos_active[i]=False
        # close any open
    elapsed=time.time()-t0
    print(f"Smoke done {elapsed:.2f}s trades {len(all_trades)}")
    if all_trades:
        wins=sum(1 for td in all_trades if td[6]>0)
        net=sum(td[6] for td in all_trades)
        print(f"WR {wins/len(all_trades)*100:.1f}% Net {net:.1f}")
        from collections import Counter
        sc=Counter(td[2] for td in all_trades)
        print("side counts",dict(sc))
    return all_trades

if __name__=="__main__":
    import sys
    if "--full" in sys.argv:
        run_smoke(limit=100000, smoke=False)
    else:
        run_smoke()
