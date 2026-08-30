"""Tight OB — 2 modes, 1pt? No, body 50%-100%."""
import re, json, time
from pathlib import Path
import pandas as pd, numpy as np, numba
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

def build_15m_obs_tight(df, avail):
    # 15m OB = last 2 untested 15m swings (HA+LinReg body 50%-100%)
    # Swing = 15m bar where HA flips and LinReg confirms
    # For tight, keep only last 2 per day that are still untested
    obs={}
    for day in avail:
        day_df=df[df["day"]==day].sort_values("minute")
        # build 15m bars
        bars=[]
        for slot in range(555, 916, 15):
            sub=day_df[(day_df["minute"]>=slot) & (day_df["minute"]<slot+15)]
            if len(sub)==0: continue
            bars.append((sub["low"].min(), sub["high"].max(), sub.iloc[0]["open"], sub.iloc[-1]["close"], slot))
        # find swings: where high is local max and low is local min
        swings=[]
        for i in range(1, len(bars)-1):
            lo, hi, o, c, m = bars[i]
            prev_hi=bars[i-1][1]; next_hi=bars[i+1][1]
            prev_lo=bars[i-1][0]; next_lo=bars[i+1][0]
            # swing high if hi > prev and hi > next
            if hi > prev_hi and hi > next_hi:
                swings.append((lo, hi, m, "high"))
            if lo < prev_lo and lo < next_lo:
                swings.append((lo, hi, m, "low"))
        # keep last 2
        obs[day]=swings[-2:]
    return obs

def process_day_tight(day, df_idx, obs_15m, mode):
    # mode 1: 15m 2 OB body 50-100%, mode 2: fib OB [bottom,peak] untested 50%
    import polars as pl
    p=option_file_for_day(day)
    if not p or not p.exists(): return []
    df=pl.read_csv(str(p), columns=["time","symbol","open","high","low","close"])
    df=df.filter(pl.col("time").str.contains(r"^\d{2}:\d{2}:\d{2}$"))
    df=df.with_columns((pl.col("time").str.slice(0,2).cast(pl.Int32)*60 + pl.col("time").str.slice(3,2).cast(pl.Int32)).alias("minute"))
    df=df.filter((pl.col("minute")>=555) & (pl.col("minute")<=915))
    pdf=df.to_pandas()
    # ATM
    day_idx=df_idx[df_idx["day"]==day]
    if day_idx.empty: return []
    spot_915=day_idx[day_idx["minute"]==555]["close"].values[0] if len(day_idx[day_idx["minute"]==555]) else day_idx.iloc[0]["close"]
    atm=int(round(float(spot_915)/50)*50)
    pdf=pdf[pdf["symbol"].apply(lambda s: (m:=SYM_RE.match(s)) and abs(int(m.group(1))-atm)<=250)]
    trades=[]
    # For mode 2, need prior fib ranges (Marnie) — build from index 1m UT for this day (we will compute per symbol? For F6, fib OB is prior F6 trade's range, but user says Marnie fib range)
    # For mode 2, we will treat prior F6 signal's [bottom,peak] as fib OB, and check entry in [0.5,1.0]
    # Keep per-symbol fib obs
    fib_obs_map={}
    for sym, g in pdf.groupby("symbol"):
        g=g.sort_values("minute")
        n=len(g)
        if n<60: continue
        highs=g["high"].to_numpy(dtype=np.float64); lows=g["low"].to_numpy(dtype=np.float64); closes=g["close"].to_numpy(dtype=np.float64); minutes=g["minute"].to_numpy(dtype=np.int64)
        # warmup 60
        # find prev day
        import datetime
        prev_day=(pd.to_datetime(day)-pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for _ in range(10):
            y,m,d=prev_day.split("-")
            cand1=OPT_ROOT / y / str(int(m)) / f"nifty_options_{d}_{m}_{y}.csv"
            cand2=OPT_ROOT / y / m / f"nifty_options_{d}_{m}_{y}.csv"
            if cand1.exists() or cand2.exists(): break
            prev_day=(pd.to_datetime(prev_day)-pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else: prev_day=None
        warm_len=0
        if prev_day:
            y,m,d=prev_day.split("-")
            pp=None
            for cand in [OPT_ROOT / y / str(int(m)) / f"nifty_options_{d}_{m}_{y}.csv",
                         OPT_ROOT / y / m / f"nifty_options_{d}_{m}_{y}.csv"]:
                if cand.exists(): pp=cand; break
            if pp and pp.exists():
                try:
                    df_prev=pl.read_csv(str(pp), columns=["time","symbol","open","high","low","close"])
                    df_prev=df_prev.filter(pl.col("symbol")==sym)
                    df_prev=df_prev.with_columns((pl.col("time").str.slice(0,2).cast(pl.Int32)*60 + pl.col("time").str.slice(3,2).cast(pl.Int32)).alias("minute"))
                    df_prev=df_prev.filter((pl.col("minute")>=555) & (pl.col("minute")<=915))
                    pdf_prev=df_prev.to_pandas().sort_values("minute").tail(60)
                    if len(pdf_prev)>0:
                        wh=pdf_prev["high"].to_numpy(dtype=np.float64); wl=pdf_prev["low"].to_numpy(dtype=np.float64); wc=pdf_prev["close"].to_numpy(dtype=np.float64)
                        wm=pdf_prev["minute"].to_numpy(dtype=np.int64)-1440
                        highs=np.concatenate([wh, highs]); lows=np.concatenate([wl, lows]); closes=np.concatenate([wc, closes]); minutes=np.concatenate([wm, minutes])
                        n=len(highs); warm_len=len(wh)
                except: pass
        s1_out=np.empty(n, dtype=np.float64); s4_out=np.empty(n, dtype=np.float64)
        stoch_nb(highs, lows, closes, n, S1_K, S1_D, s1_out)
        stoch_nb(highs, lows, closes, n, S4_K, S4_D, s4_out)
        # OB structures
        if mode==1:
            # 15m 2 OB body 50-100%
            # obs_15m[day] are (lo,hi) for last 2 swings
            obs_list=obs_15m.get(day, [])
            # keep only 2, body is [mid, hi] for bullish, [lo, mid] for bearish? Use 50-100% of OB
            # For simplicity, body 50-100% = [mid, hi] for high OB, [lo, mid] for low OB, but we need to know which is which
            # For mode1, we will check entry in [mid, hi] if PE (put) expects price in upper half, or [lo, mid] if CE
            # For now, check entry in [mid, hi] for all (upper half)
            pass
        pos=None
        # For mode2, keep fib obs per symbol
        if mode==2:
            # fib_obs_map per symbol
            if sym not in fib_obs_map:
                fib_obs_map[sym]=[]
        for i in range(warm_len, n):
            if np.isnan(s1_out[i]) or np.isnan(s4_out[i]): continue
            v1=s1_out[i]; v4=s4_out[i]
            is_flag = v4>=S4_OB and v1<=S1_OS
            # OB filter
            if is_flag:
                if mode==1:
                    c_tmp=float(closes[i])
                    lst = obs_15m.get(day, [])
                    # last 2 untested 15m swings
                    check = lst[-2:] if len(lst)>=2 else lst
                    if check:
                        ok=False
                        for lo, hi, _, _ in check:
                            mid=(lo+hi)/2.0
                            if mid <= c_tmp <= hi:
                                ok=True; break
                        if not ok:
                            is_flag=False
                    else:
                        is_flag=False
                elif mode==2:
                    c_tmp=float(closes[i])
                    # need to compute entry for this F6 signal to check, but F6 entry is close, so check close in [0.5,1.0] of prior fib OB
                    # prior fib OB is last trade's [bottom,peak] that is still untested (close not beyond 50%)
                    lst=fib_obs_map.get(sym, [])
                    if lst:
                        # check if Close in [0.5,1.0] of any prior fib OB
                        ok=False
                        for lo, hi in lst:
                            mid=(lo+hi)/2.0
                            # for F6, entry in [mid, hi] if PE? Actually for F6, we check close in [mid, hi] for all
                            # Use [mid, hi] as 50-100%
                            if mid <= c_tmp <= hi+1:
                                ok=True; break
                        if not ok:
                            is_flag=False
                    else:
                        # no prior fib OB, allow first
                        pass
            # exits
            if pos is not None:
                # need to map i to g index
                # g has n-warm_len rows, i-warm_len is index in g
                idx_in_g=i-warm_len
                if idx_in_g <0 or idx_in_g >= len(g): continue
                row=g.iloc[idx_in_g]
                h=float(row["high"]); l=float(row["low"]); c=float(row["close"]); minute=int(row["minute"])
                if l <= pos["sl"]:
                    pts=pos["sl"]-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":pos["sl"],"pts":pts,"reason":"SL"})
                    pos=None
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
                c=float(closes[i]); minute=int(minutes[i])
                pos={"entry":c,"entry_min":minute,"sl":c-SL,"tp":c+TP}
                if mode==2:
                    # add this entry's range as new fib OB for future (peak/bottom of this bar? For F6, OB is bar high/low)
                    # Use bar high/low as OB
                    lo=float(lows[i]); hi=float(highs[i])
                    if sym not in fib_obs_map: fib_obs_map[sym]=[]
                    fib_obs_map[sym].append((lo, hi))
                    # mitigate: remove OBs that have been touched beyond 50%
                    new=[]
                    for obl, obh in fib_obs_map[sym][:-1]:  # exclude just added
                        mid=(obl+obh)/2.0
                        # if price closed beyond 50%, it's tested
                        if c <= mid:
                            new.append((obl, obh))
                    fib_obs_map[sym]=new + [fib_obs_map[sym][-1]]
        if pos is not None:
            # last bar
            c=float(closes[-1]); minute=int(minutes[-1])
            pts=c-pos["entry"]
            trades.append({"day":day,"symbol":sym,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD"})
    return trades
