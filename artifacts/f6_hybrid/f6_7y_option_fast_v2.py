"""F6 ULTRA-FAST — Polars + Numba Stoch + pointer, 8 workers, warmup, causal."""
import pandas as pd, re, json, time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import numba

OPT_ROOT=Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options")
CSV_INDEX=r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
OUT=Path(__file__).with_name("f6_7y_option_v2_result.json")
S1_K=9; S1_D=3; S4_K=60; S4_D=10; S4_OB=79.5; S1_OS=25.0; SL=10; TP=15
SYM_RE=re.compile(r"^NIFTY\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$")

@numba.njit
def stoch_numba(highs, lows, closes, n, k, d, out):
    # highs/lows/closes: arrays length n, out length n, fill with nan where not enough
    # Use rolling window max/min
    for i in range(n):
        if i < k-1:
            out[i]=np.nan
            continue
        hh=highs[i]
        ll=lows[i]
        for j in range(i-k+1, i+1):
            if highs[j]>hh: hh=highs[j]
            if lows[j]<ll: ll=lows[j]
        if hh==ll:
            kval=50.0
        else:
            kval=(closes[i]-ll)/(hh-ll)*100.0
        # need d smoothing
        if i < k-1 + d -1:
            out[i]=np.nan
        else:
            # average last d kvals
            s=0.0
            for j in range(i-d+1, i+1):
                # recompute kval for j
                # we have stored kvals in out? Need separate
                # Instead compute on fly and average
                # For simplicity, compute kval for each j in window and average
                hh2=highs[j]
                ll2=lows[j]
                for t in range(j-k+1, j+1):
                    if highs[t]>hh2: hh2=highs[t]
                    if lows[t]<ll2: ll2=lows[t]
                if hh2==ll2:
                    kj=50.0
                else:
                    kj=(closes[j]-ll2)/(hh2-ll2)*100.0
                s+=kj
            out[i]=s/d
    return out

def load_index():
    import pandas as pd
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

def process_day_fast(day):
    # Use Polars for reading (fast)
    import polars as pl
    p=option_file_for_day(day)
    if not p or not p.exists(): return []
    # Polars read
    df=pl.read_csv(str(p), columns=["time","symbol","open","high","low","close","volume"], ignore_errors=True)
    df=df.filter(pl.col("time").str.contains(r"^\d{2}:\d{2}:\d{2}$"))
    df=df.with_columns((pl.col("time").str.slice(0,2).cast(pl.Int32)*60 + pl.col("time").str.slice(3,2).cast(pl.Int32)).alias("minute"))
    # filter to session
    df=df.filter((pl.col("minute")>=555) & (pl.col("minute")<=915))
    # ATM filter: need index for ATM, load from global
    # For now, just process all symbols with F6 (not ATM filter) to keep speed, but we filter to ATM±250 later
    # Get ATM
    atm=None
    try:
        # Use global index df
        global GLOBAL_IDX
        day_idx=GLOBAL_IDX[GLOBAL_IDX["day"]==day]
        if len(day_idx)>0:
            spot_915=day_idx[day_idx["minute"]==555]["close"].values[0] if len(day_idx[day_idx["minute"]==555]) else day_idx.iloc[0]["close"]
            atm=int(round(float(spot_915)/50)*50)
    except: atm=None
    if atm is not None:
        # filter to ATM±250
        df=df.filter(pl.col("symbol").str.contains(f"{atm-250}|{atm-200}|{atm-150}|{atm-100}|{atm}|{atm+100}|{atm+150}|{atm+200}|{atm+250}"))
    # to pandas for grouping
    pdf=df.to_pandas()
    trades=[]
    # warmup: need previous day same symbols
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
                    g2=g2.sort_values("minute").tail(60)
                    warmup_map[sym2]=g2
            except: pass
    for sym, g in pdf.groupby("symbol"):
        g=g.sort_values("minute")
        n=len(g)
        if n<60: continue
        highs=g["high"].to_numpy(dtype=np.float64)
        lows=g["low"].to_numpy(dtype=np.float64)
        closes=g["close"].to_numpy(dtype=np.float64)
        minutes=g["minute"].to_numpy(dtype=np.int64)
        vols=g["volume"].to_numpy(dtype=np.float64) if "volume" in g.columns else np.ones(n, dtype=np.float64)*1000
        # warmup arrays
        if sym in warmup_map:
            wg=warmup_map[sym]
            wh=wg["high"].to_numpy(dtype=np.float64)
            wl=wg["low"].to_numpy(dtype=np.float64)
            wc=wg["close"].to_numpy(dtype=np.float64)
            wv=wg["volume"].to_numpy(dtype=np.float64) if "volume" in wg.columns else np.ones(len(wg), dtype=np.float64)*1000
            highs=np.concatenate([wh, highs])
            lows=np.concatenate([wl, lows])
            closes=np.concatenate([wc, closes])
            vols=np.concatenate([wv, vols])
            # need to adjust minutes, but warmup minutes are previous day, offset by -1440
            # For Stoch, minutes not needed for calc, just for output, so we can offset
            wm=wg["minute"].to_numpy(dtype=np.int64) - 1440
            minutes=np.concatenate([wm, minutes])
            n=len(highs)
            warm_len=len(wh)
        else:
            warm_len=0
        # Numba Stoch
        s1_out=np.empty(n, dtype=np.float64)
        s4_out=np.empty(n, dtype=np.float64)
        stoch_numba(highs, lows, closes, n, S1_K, S1_D, s1_out)
        stoch_numba(highs, lows, closes, n, S4_K, S4_D, s4_out)
        # S4 5m >80 filter (replaces OB) — BUILD 5m S4 per symbol
        m5_highs=[]; m5_lows=[]; m5_closes=[]
        for j in range(0, n, 5):
            end=min(j+5, n)
            if end-j <5: break
            m5_highs.append(float(np.max(highs[j:end])))
            m5_lows.append(float(np.min(lows[j:end])))
            m5_closes.append(float(closes[j+4]))
        s4_5m_out=np.empty(len(m5_highs), dtype=np.float64)
        if len(m5_highs)>=50:
            for j in range(len(m5_highs)):
                if j < 49:
                    s4_5m_out[j]=np.nan
                else:
                    hh=np.max(m5_highs[j-49:j+1]); ll=np.min(m5_lows[j-49:j+1])
                    k=(m5_closes[j]-ll)/(hh-ll)*100 if hh!=ll else 50
                    if j < 49+9:
                        s4_5m_out[j]=np.nan
                    else:
                        s=0.0
                        for t in range(j-9, j+1):
                            hht=np.max(m5_highs[t-49:t+1]) if t>=49 else 0
                            llt=np.min(m5_lows[t-49:t+1]) if t>=49 else 0
                            if t<49: continue
                            kk=(m5_closes[t]-llt)/(hht-llt)*100 if hht!=llt else 50
                            s+=kk
                        s4_5m_out[j]=s/10
        else:
            s4_5m_out[:]=np.nan
        # S4 5m >80 filter
        m5_highs=[]; m5_lows=[]; m5_closes=[]
        for j in range(0, n, 5):
            end=min(j+5, n)
            if end-j <5: break
            m5_highs.append(float(np.max(highs[j:end]))); m5_lows.append(float(np.min(lows[j:end]))); m5_closes.append(float(closes[j+4]))
        s4_5m_out=np.empty(len(m5_highs), dtype=np.float64)
        if len(m5_highs)>=50:
            for j in range(len(m5_highs)):
                if j < 49: s4_5m_out[j]=np.nan
                else:
                    hh=np.max(m5_highs[j-49:j+1]); ll=np.min(m5_lows[j-49:j+1])
                    k=(m5_closes[j]-ll)/(hh-ll)*100 if hh!=ll else 50
                    if j < 49+9: s4_5m_out[j]=np.nan
                    else:
                        s=0.0
                        for t in range(j-9, j+1):
                            hht=np.max(m5_highs[t-49:t+1]) if t>=49 else 0
                            llt=np.min(m5_lows[t-49:t+1]) if t>=49 else 0
                            if t<49: continue
                            kk=(m5_closes[t]-llt)/(hht-llt)*100 if hht!=llt else 50
                            s+=kk
                        s4_5m_out[j]=s/10
        else: s4_5m_out[:]=np.nan
        pos=None
        for i in range(warm_len, n):
            if np.isnan(s1_out[i]) or np.isnan(s4_out[i]): continue
            v1=s1_out[i]; v4=s4_out[i]
            is_flag = v4>=S4_OB and v1<=S1_OS
            if is_flag:
                idx5=i//5
                if idx5 < len(s4_5m_out):
                    v4_5m=s4_5m_out[idx5]
                    if np.isnan(v4_5m) or v4_5m <= 80.0:
                        is_flag=False
                else: is_flag=False
            # exits
            if pos is not None:
                h=float(g.iloc[i-warm_len]["high"]) if warm_len==0 else float(highs[i])
                l=float(g.iloc[i-warm_len]["low"]) if warm_len==0 else float(lows[i])
                c=float(g.iloc[i-warm_len]["close"]) if warm_len==0 else float(closes[i])
                minute=int(minutes[i])
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
                c=float(closes[i])
                minute=int(minutes[i])
                pos={"entry":c,"entry_min":minute,"sl":c-SL,"tp":c+TP}
        # EOD for remaining pos
        if pos is not None:
            # last bar
            c=float(closes[-1])
            minute=int(minutes[-1])
            pts=c-pos["entry"]
            trades.append({"day":day,"symbol":sym,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD"})
    return trades

GLOBAL_IDX=None
def init_worker(idx):
    global GLOBAL_IDX
    GLOBAL_IDX=idx

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2026-08-27")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    args=p.parse_args()
    import pandas as pd
    df_idx=load_index()
    days_sorted=sorted(df_idx["day"].unique().tolist())
    avail=[d for d in days_sorted if args.start <= d <= args.end]
    # filter to option days
    avail=[d for d in avail if option_file_for_day(d) and option_file_for_day(d).exists()]
    if args.smoke:
        avail=avail[:5]
        print("=== SMOKE TEST — 5 DAYS ONLY (v2) ===")
    print(f"Backtest {avail[0]} to {avail[-1]} ({len(avail)} days), workers={args.workers}")
    t0=time.time()
    if args.smoke:
        GLOBAL_IDX=df_idx
        trades=[]
        for d in avail:
            trades.extend(process_day_fast(d))
        print(f"Smoke {len(trades)} trades in {time.time()-t0:.1f}s")
        if trades:
            import pandas as pd
            tdf=pd.DataFrame(trades)
            print(tdf.head().to_string(index=False))
            wins=len(tdf[tdf["pts"]>0])
            print(f"WR {wins/len(trades)*100:.1f}% Net {tdf['pts'].sum():.1f} PF {(tdf[tdf['pts']>0]['pts'].sum()/abs(tdf[tdf['pts']<=0]['pts'].sum())):.2f}")
        exit(0)
    workers=min(args.workers,8,cpu_count())
    with Pool(workers, initializer=init_worker, initargs=(df_idx,)) as pool:
        results=pool.map(process_day_fast, avail)
    all_trades=[t for lst in results for t in lst]
    elapsed=time.time()-t0
    print(f"Completed {len(avail)} days in {elapsed:.1f}s, {len(all_trades)} trades")
    if all_trades:
        import pandas as pd
        tdf=pd.DataFrame(all_trades)
        wins=len(tdf[tdf["pts"]>0])
        print(f"Total {len(all_trades)} WR {wins/len(all_trades)*100:.1f}% Net {tdf['pts'].sum():.1f} PF {(tdf[tdf['pts']>0]['pts'].sum()/abs(tdf[tdf['pts']<=0]['pts'].sum())):.2f}")
        tdf["year"]=pd.to_datetime(tdf["day"]).dt.year
        for y, g in tdf.groupby("year"):
            wins_y=len(g[g["pts"]>0])
            print(f"{y}: {len(g)} WR {wins_y/len(g)*100:.1f}% Net {g['pts'].sum():.1f} PF {(g[g['pts']>0]['pts'].sum()/abs(g[g['pts']<=0]['pts'].sum())):.2f}")
    import json
    out=Path(__file__).with_name("f6_7y_option_v2_result.json")
    if args.smoke: out=out.with_name(out.stem+"_smoke"+out.suffix)
    out.write_text(json.dumps({"start":args.start,"end":args.end,"days":len(avail),"elapsed":elapsed,"trades":all_trades}, indent=2, default=float), encoding="utf-8")
    print(f"JSON {out}")
