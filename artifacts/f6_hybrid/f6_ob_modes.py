"""OB modes — test one by one."""
import re
from pathlib import Path
import pandas as pd
import numpy as np
import numba

OPT_ROOT=Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options")
CSV_INDEX=r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
CACHE=Path(r"C:\Users\user\Desktop\nifty50 data\cache_marni_opt.parquet")
S1_K=12; S1_D=3; S4_K=50; S4_D=10; S4_OB=79.5; S1_OS=25.0; SL=10; TP=15
SYM_RE=re.compile(r"^NIFTY\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$")

@numba.njit
def stoch_nb(highs,lows,closes,n,k,d,out):
    for i in range(n):
        if i < k-1: out[i]=np.nan; continue
        hh=highs[i]; ll=lows[i]
        for j in range(i-k+1,i+1):
            if highs[j]>hh: hh=highs[j]
            if lows[j]<ll: ll=lows[j]
        kval=50.0 if hh==ll else (closes[i]-ll)/(hh-ll)*100.0
        if i < k-1+d-1: out[i]=np.nan
        else:
            s=0.0
            for j in range(i-d+1,i+1):
                hh2=highs[j]; ll2=lows[j]
                for t in range(j-k+1,j+1):
                    if highs[t]>hh2: hh2=highs[t]
                    if lows[t]<ll2: ll2=lows[t]
                kj=50.0 if hh2==ll2 else (closes[j]-ll2)/(hh2-ll2)*100.0
                s+=kj
            out[i]=s/d
    return out

def load_index():
    df=pd.read_csv(CSV_INDEX, skiprows=1, names=["date","open","high","low","close","volume"])
    df["dt"]=pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df["minute"]=df["dt"].dt.hour*60+df["dt"].dt.minute
    df["day"]=df["dt"].dt.strftime("%Y-%m-%d")
    return df.sort_values("dt").reset_index(drop=True)

def option_file_for_day(day):
    y,m,d=day.split("-")
    for cand in [OPT_ROOT / y / str(int(m)) / f"nifty_options_{d}_{m}_{y}.csv",
                 OPT_ROOT / y / m / f"nifty_options_{d}_{m}_{y}.csv"]:
        if cand.exists(): return cand
    pats=list(OPT_ROOT.rglob(f"nifty_options_{d}_{m}_{y}.csv"))
    return pats[0] if pats else None

# Build index 15m OBs (daily, HA+LinReg swing) for OB mode 1
def build_15m_obs(df, avail):
    # Use same BiasComputer logic but extract swing highs/lows where HA+LinReg flips
    # Simplified: 15m OB = 15m bar's high/low where bias flips and hasn't been retested
    # For now, define OB as 15m bar where bullish->bearish or vice versa, high/low of that bar
    # Keep untested if price hasn't closed beyond its 50%
    import pandas as pd
    # Build 15m bars for each day
    obs_per_day={}
    for day in avail:
        day_df=df[df["day"]==day].sort_values("minute")
        # Build 15m bars 555,570,585...
        bars=[]
        for slot in range(555, 916, 15):
            sub=day_df[(day_df["minute"]>=slot) & (day_df["minute"]<slot+15)]
            if len(sub)==0: continue
            bars.append({"open":sub.iloc[0]["open"],"high":sub["high"].max(),"low":sub["low"].min(),"close":sub.iloc[-1]["close"],"minute":slot})
        # HA + LinReg quick
        # HA
        ha_o=None; ha_c=None; closes=[]; sig=[]
        obs=[]
        for b in bars:
            ha_c=(b["open"]+b["high"]+b["low"]+b["close"])/4.0
            ha_o=(b["open"]+b["close"])/2.0 if ha_o is None else (ha_o+ha_c)/2.0
            # LinReg would need closes, skip for now, use HA close vs open as proxy for swing
            # Swing high/low if HA flips
            # Keep closes for LinReg
            closes.append(ha_c)
            # OB as bar high/low where HA body flips
            # For simplicity, OB = bar high/low when HA close crosses HA open
            # We'll keep all bars as potential OBs, filter later by untouched
            obs.append((b["low"], b["high"], b["minute"]))
        obs_per_day[day]=obs
    return obs_per_day

def process_day_ob_mode(day, df_idx, obs_15m, mode):
    # mode 1: 15m OB, 2: fib OB, 3: side-specific
    import pandas as pd, polars as pl
    # ATM
    day_idx=df_idx[df_idx["day"]==day]
    if day_idx.empty: return []
    spot_915=day_idx[day_idx["minute"]==555]["close"].values[0] if len(day_idx[day_idx["minute"]==555]) else day_idx.iloc[0]["close"]
    atm=int(round(float(spot_915)/50)*50)
    p=option_file_for_day(day)
    if not p or not p.exists(): return []
    # Load option day via polars or pandas
    import polars as pl
    try:
        df=pl.read_csv(str(p), columns=["time","symbol","open","high","low","close"])
        df=df.filter(pl.col("time").str.contains(r"^\d{2}:\d{2}:\d{2}$"))
        df=df.with_columns((pl.col("time").str.slice(0,2).cast(pl.Int32)*60 + pl.col("time").str.slice(3,2).cast(pl.Int32)).alias("minute"))
        df=df.filter((pl.col("minute")>=555) & (pl.col("minute")<=915))
        pdf=df.to_pandas()
    except: return []
    trades=[]
    # For OB mode 1, need prior daily 15m OBs that are still untouched
    # Build global untouched lists per day (we will keep per symbol? For mode1, OB is index 15m, global)
    # For mode 2, fib OB per symbol
    # For mode 3, side-specific OB per symbol
    # Simplify: maintain per-symbol untouched list, but mode1 uses index 15m OBs (global)
    # We will handle per mode
    # Preload warmup for stoch
    prev_day=(pd.to_datetime(day)-pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for _ in range(10):
        y,m,d=prev_day.split("-")
        cand1=OPT_ROOT / y / str(int(m)) / f"nifty_options_{d}_{m}_{y}.csv"
        cand2=OPT_ROOT / y / m / f"nifty_options_{d}_{m}_{y}.csv"
        if cand1.exists() or cand2.exists(): break
        prev_day=(pd.to_datetime(prev_day)-pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else: prev_day=None
    warmup_map={}
    if prev_day:
        y,m,d=prev_day.split("-")
        pp=None
        for cand in [OPT_ROOT / y / str(int(m)) / f"nifty_options_{d}_{m}_{y}.csv",
                     OPT_ROOT / y / m / f"nifty_options_{d}_{m}_{y}.csv"]:
            if cand.exists(): pp=cand; break
        if pp and pp.exists():
            try:
                df_prev=pl.read_csv(str(pp), columns=["time","symbol","open","high","low","close"])
                df_prev=df_prev.with_columns((pl.col("time").str.slice(0,2).cast(pl.Int32)*60 + pl.col("time").str.slice(3,2).cast(pl.Int32)).alias("minute"))
                df_prev=df_prev.filter((pl.col("minute")>=555) & (pl.col("minute")<=915))
                pdf_prev=df_prev.to_pandas()
                for sym2, g2 in pdf_prev.groupby("symbol"):
                    warmup_map[sym2]=g2.sort_values("minute").tail(60)
            except: pass
    # For mode 1, prepare prior 15m OBs that are still untouched (global)
    # For simplicity, use last 5 days of 15m OBs
    untouched_15m=[]
    if mode==1:
        # collect last 5 days of 15m OBs that haven't been mitigated
        # For now, take obs_15m for prior days
        # Find prior 5 avail days
        all_days_sorted=sorted(obs_15m.keys())
        try: idx=all_days_sorted.index(day)
        except: idx=0
        for pd_ in all_days_sorted[max(0,idx-5):idx]:
            for ob_low, ob_high, _ in obs_15m[pd_]:
                # check if still untouched (price hasn't closed beyond 50%)
                # For now, keep all
                untouched_15m.append((ob_low, ob_high))
    for sym, g in pdf.groupby("symbol"):
        # filter to ATM±250
        m=SYM_RE.match(sym)
        if not m: continue
        strike=int(m.group(1))
        if abs(strike-atm)>250: continue
        side=m.group(2)
        g=g.sort_values("minute")
        n=len(g)
        if n<60: continue
        highs=g["high"].to_numpy(dtype=np.float64)
        lows=g["low"].to_numpy(dtype=np.float64)
        closes=g["close"].to_numpy(dtype=np.float64)
        minutes=g["minute"].to_numpy(dtype=np.int64)
        if sym in warmup_map:
            wg=warmup_map[sym]
            wh=wg["high"].to_numpy(dtype=np.float64)
            wl=wg["low"].to_numpy(dtype=np.float64)
            wc=wg["close"].to_numpy(dtype=np.float64)
            highs=np.concatenate([wh, highs]); lows=np.concatenate([wl, lows]); closes=np.concatenate([wc, closes])
            wm=wg["minute"].to_numpy(dtype=np.int64) - 1440
            minutes=np.concatenate([wm, minutes])
            n=len(highs); warm_len=len(wh)
        else: warm_len=0
        s1_out=np.empty(n, dtype=np.float64); s4_out=np.empty(n, dtype=np.float64)
        stoch_nb(highs, lows, closes, n, S1_K, S1_D, s1_out)
        stoch_nb(highs, lows, closes, n, S4_K, S4_D, s4_out)
        # OB structures per mode
        # Mode 2: fib OB per symbol (prior fib range untouched)
        # Mode 3: side-specific OB per symbol
        untouched_fib=[]  # for mode2
        untouched_side_ce=[]; untouched_side_pe=[]
        pos=None
        for i in range(warm_len, n):
            if np.isnan(s1_out[i]) or np.isnan(s4_out[i]): continue
            v1=s1_out[i]; v4=s4_out[i]
            is_flag = v4>=S4_OB and v1<=S1_OS
            # Apply OB filter per mode
            if is_flag:
                if mode==1:
                    # 15m OB: close within any prior 15m OB ±1
                    c_tmp=float(closes[i])
                    if untouched_15m:
                        if not any(lo-1 <= c_tmp <= hi+1 for lo,hi in untouched_15m):
                            is_flag=False
                    else:
                        # no prior OB, allow first
                        pass
                elif mode==2:
                    # fib OB: prior fib range [bottom,peak] untouched (price hasn't closed beyond 50%)
                    # Check if entry would be within any untouched fib OB
                    # For now, approximate: prior fib OB is last trade's entry range
                    # Use untouched_fib list
                    c_tmp=float(closes[i])
                    if untouched_fib:
                        # need to compute entry for this signal to check
                        # For PE, entry = bottom+0.786*rng, for CE peak-0.786*rng
                        # But we don't have peak/bottom yet for this signal (it's F6, not Marnie)
                        # For F6, OB is signal bar's high/low, so entry is close, check close within OB
                        if not any(lo-1 <= c_tmp <= hi+1 for lo,hi in untouched_fib):
                            is_flag=False
                        # Also check 50% untouched: price hasn't closed beyond 50% of OB
                        # For simplicity, if OB's 50% has been breached, remove it
                        # We handle mitigation below
                    # mitigation: if current bar touches an OB beyond 50%, remove it
                    # (handled below)
                elif mode==3:
                    c_tmp=float(closes[i])
                    lst = untouched_side_ce if side=="CE" else untouched_side_pe
                    if lst:
                        if not any(lo-1 <= c_tmp <= hi+1 for lo,hi in lst):
                            is_flag=False
            # Mitigate OBs that are touched by current bar (even without signal)
            cur_h=float(highs[i]); cur_l=float(lows[i])
            if mode==1 and untouched_15m:
                untouched_15m=[(lo,hi) for lo,hi in untouched_15m if not (cur_l <= hi+1 and cur_h >= lo-1)]
            if mode==2 and untouched_fib:
                # remove fib OBs that are touched beyond 50%
                new=[]
                for lo,hi in untouched_fib:
                    mid=(lo+hi)/2.0
                    # if price closes beyond 50%, it's touched
                    if not (cur_l <= hi and cur_h >= lo): # simple overlap
                        new.append((lo,hi))
                    else:
                        # if close beyond 50%, consider touched
                        c_tmp=float(closes[i])
                        if side=="CE":
                            # for CE, 50% is mid, if close > mid, touched
                            if c_tmp <= mid+1:
                                new.append((lo,hi))
                        else:
                            if c_tmp >= mid-1:
                                new.append((lo,hi))
                untouched_fib=new
            if mode==3:
                lst = untouched_side_ce if side=="CE" else untouched_side_pe
                if lst:
                    new=[]
                    for lo,hi in lst:
                        if not (cur_l <= hi+1 and cur_h >= lo-1):
                            new.append((lo,hi))
                    if side=="CE": untouched_side_ce=new
                    else: untouched_side_pe=new
            # exits
            if pos is not None:
                h=float(g.iloc[i-warm_len]["high"]) if warm_len==0 else float(highs[i])
                l=float(g.iloc[i-warm_len]["low"]) if warm_len==0 else float(lows[i])
                c=float(g.iloc[i-warm_len]["close"]) if warm_len==0 else float(closes[i])
                minute=int(minutes[i])
                if l <= pos["sl"]:
                    pts=pos["sl"]-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"side":side,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":pos["sl"],"pts":pts,"reason":"SL"})
                    pos=None
                    # Add this trade's OB for future (mode2)
                    if mode==2:
                        untouched_fib.append((pos["sl"] if False else 0,0)) # placeholder
                elif h >= pos["tp"]:
                    pts=pos["tp"]-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":pos["tp"],"pts":pts,"reason":"TP"})
                    pos=None
                elif minute>=915:
                    pts=c-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD"})
                    pos=None
                if pos is not None and minute>=915:
                    pts=c-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD"})
                    pos=None
            if pos is None and is_flag:
                c=float(closes[i])
                minute=int(minutes[i])
                pos={"entry":c,"entry_min":minute,"sl":c-SL,"tp":c+TP}
                # Add OB for future
                if mode==1:
                    # 15m OB is not per F6 bar, so not added here
                    pass
                elif mode==2:
                    # Add this entry's fib-like OB (entry ±7)
                    untouched_fib.append((c-7, c+7))
                elif mode==3:
                    if side=="CE":
                        untouched_side_ce.append((float(lows[i]), float(highs[i])))
                    else:
                        untouched_side_pe.append((float(lows[i]), float(highs[i])))
        if pos is not None:
            c=float(closes[-1])
            minute=int(minutes[-1])
            pts=c-pos["entry"]
            trades.append({"day":day,"symbol":sym,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD"})
    return trades
