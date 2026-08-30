"""F6+SUPER+REVERSAL — clean, 60-bar warmup, causal, 8 workers."""
import pandas as pd, re, json, time
from pathlib import Path
from multiprocessing import Pool, cpu_count
OPT_ROOT=Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options")
CSV_INDEX=r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
OUT=Path(__file__).with_name("f6_super_reversal_clean_result.json")
S1_K=9; S1_D=3; S2_K=14; S2_D=3; S3_K=40; S3_D=4; S4_K=60; S4_D=10; S4_OB=79.5; S1_OS=25.0; SL=10; TP=15
SYM_RE=re.compile(r"^NIFTY\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$")

class Stoch:
    def __init__(self,k,d): self.k=k; self.d=d; self.highs=[]; self.lows=[]; self.closes=[]; self.k_vals=[]
    def push(self,h,l,c):
        self.highs.append(h); self.lows.append(l); self.closes.append(c)
        if len(self.highs)>self.k: self.highs.pop(0); self.lows.pop(0); self.closes.pop(0)
        if len(self.highs)<self.k: return None
        hh=max(self.highs); ll=min(self.lows)
        kval=(c-ll)/(hh-ll)*100 if hh!=ll else 50
        self.k_vals.append(kval)
        if len(self.k_vals)>self.d: self.k_vals.pop(0)
        if len(self.k_vals)<self.d: return None
        return sum(self.k_vals)/self.d

class Divergence:
    def __init__(self): self.price=[]; self.s1=[]; self.troughs=[]
    def update(self, close, s1):
        if s1 is None: return
        self.price.append(close); self.s1.append(s1)
        if len(self.price)>40: self.price.pop(0); self.s1.pop(0)
        if len(self.s1)>=3 and self.s1[-2] < self.s1[-3] and self.s1[-1] > self.s1[-2] and self.s1[-2] <= 20:
            self.troughs.append((self.price[-2], self.s1[-2]))
            if len(self.troughs)>5: self.troughs.pop(0)
    def has_bullish(self):
        if len(self.troughs)<2: return False
        p1,s1=self.troughs[-2]; p2,s2=self.troughs[-1]
        return p2 < p1 and s2 > s1

def load_index():
    try:
        df=pd.read_csv(CSV_INDEX)
        if "date" not in df.columns:
            df=pd.read_csv(CSV_INDEX, skiprows=1, names=["date","open","high","low","close","volume","oi"])
    except:
        df=pd.read_csv(CSV_INDEX, skiprows=1, names=["date","open","high","low","close","volume","oi"])
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

def process_day(day, mode):
    df_idx=GLOBAL_IDX
    day_idx=df_idx[df_idx["day"]==day]
    if day_idx.empty: return []
    spot_915=day_idx[day_idx["minute"]==555]["close"].values[0] if len(day_idx[day_idx["minute"]==555]) else day_idx.iloc[0]["close"]
    atm=int(round(float(spot_915)/50)*50)
    p=option_file_for_day(day)
    if not p or not p.exists(): return []
    import polars as pl
    df=pl.read_csv(str(p), columns=["time","symbol","open","high","low","close"])
    df=df.filter(pl.col("time").str.contains(r"^\d{2}:\d{2}:\d{2}$"))
    df=df.with_columns((pl.col("time").str.slice(0,2).cast(pl.Int32)*60 + pl.col("time").str.slice(3,2).cast(pl.Int32)).alias("minute"))
    df=df.filter((pl.col("minute")>=555) & (pl.col("minute")<=915))
    pdf=df.to_pandas()
    pdf=pdf[pdf["symbol"].apply(lambda s: (mm:=SYM_RE.match(s)) and abs(int(mm.group(1))-atm)<=250)]
    trades=[]
    prev_day=(pd.to_datetime(day)-pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for _ in range(10):
        y2,m2,d2=prev_day.split("-")
        cand1=OPT_ROOT / y2 / str(int(m2)) / f"nifty_options_{d2}_{m2}_{y2}.csv"
        cand2=OPT_ROOT / y2 / m2 / f"nifty_options_{d2}_{m2}_{y2}.csv"
        if cand1.exists() or cand2.exists(): break
        prev_day=(pd.to_datetime(prev_day)-pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else: prev_day=None
    warmup_map={}
    if prev_day:
        y2,m2,d2=prev_day.split("-")
        pp=None
        for cand in [OPT_ROOT / y2 / str(int(m2)) / f"nifty_options_{d2}_{m2}_{y2}.csv",
                     OPT_ROOT / y2 / m2 / f"nifty_options_{d2}_{m2}_{y2}.csv"]:
            if cand.exists(): pp=cand; break
        if pp and pp.exists():
            try:
                df_prev=pl.read_csv(str(pp), columns=["time","symbol","open","high","low","close"])
                df_prev=df_prev.filter(pl.col("time").str.contains(r"^\d{2}:\d{2}:\d{2}$"))
                df_prev=df_prev.with_columns((pl.col("time").str.slice(0,2).cast(pl.Int32)*60 + pl.col("time").str.slice(3,2).cast(pl.Int32)).alias("minute"))
                df_prev=df_prev.filter((pl.col("minute")>=555) & (pl.col("minute")<=915))
                pdf_prev=df_prev.to_pandas()
                for sym2, g2 in pdf_prev.groupby("symbol"):
                    warmup_map[sym2]=g2.sort_values("minute").tail(60)
            except: pass
    for sym, g in pdf.groupby("symbol"):
        g=g.sort_values("minute")
        n=len(g)
        if n<60: continue
        stoch_s1=Stoch(S1_K,S1_D); stoch_s2=Stoch(S2_K,S2_D); stoch_s3=Stoch(S3_K,S3_D); stoch_s4=Stoch(S4_K,S4_D)
        div=Divergence()
        if sym in warmup_map:
            wg=warmup_map[sym]
            for _, r2 in wg.iterrows():
                stoch_s1.push(float(r2["high"]), float(r2["low"]), float(r2["close"]))
                stoch_s2.push(float(r2["high"]), float(r2["low"]), float(r2["close"]))
                stoch_s3.push(float(r2["high"]), float(r2["low"]), float(r2["close"]))
                stoch_s4.push(float(r2["high"]), float(r2["low"]), float(r2["close"]))
                div.update(float(r2["close"]), stoch_s1.k_vals[-1] if stoch_s1.k_vals else None)
        # 3m/5m S4
        m3_highs=[]; m3_lows=[]; m3_closes=[]
        for j in range(0, len(g), 3):
            sub=g.iloc[j:j+3]
            if len(sub)<3: break
            m3_highs.append(sub["high"].max()); m3_lows.append(sub["low"].min()); m3_closes.append(sub.iloc[-1]["close"])
        m5_highs=[]; m5_lows=[]; m5_closes=[]
        for j in range(0, len(g), 5):
            sub=g.iloc[j:j+5]
            if len(sub)<5: break
            m5_highs.append(sub["high"].max()); m5_lows.append(sub["low"].min()); m5_closes.append(sub.iloc[-1]["close"])
        stoch_s4_3m=Stoch(S4_K,S4_D); stoch_s4_5m=Stoch(S4_K,S4_D)
        for hv,lv,cv in zip(m3_highs[:10], m3_lows[:10], m3_closes[:10]):
            stoch_s4_3m.push(float(hv),float(lv),float(cv))
        for hv,lv,cv in zip(m5_highs[:10], m5_lows[:10], m5_closes[:10]):
            stoch_s4_5m.push(float(hv),float(lv),float(cv))
        pos=None; s4_embedded=0
        for idx, (_,r) in enumerate(g.iterrows()):
            minute=int(r["minute"]); h=float(r["high"]); l=float(r["low"]); c=float(r["close"])
            v1=stoch_s1.push(h,l,c); v2=stoch_s2.push(h,l,c); v3=stoch_s3.push(h,l,c); v4=stoch_s4.push(h,l,c)
            if idx % 3 == 2:
                sub=g.iloc[max(0,idx-2):idx+1]
                stoch_s4_3m.push(float(sub["high"].max()), float(sub["low"].min()), float(sub.iloc[-1]["close"]))
            if idx % 5 == 4:
                sub=g.iloc[max(0,idx-4):idx+1]
                stoch_s4_5m.push(float(sub["high"].max()), float(sub["low"].min()), float(sub.iloc[-1]["close"]))
            if v4 is not None:
                if v4 <= 20: s4_embedded+=1
                else: s4_embedded=0
            div.update(c, v1)
            has_bull=div.has_bullish()
            is_flag = v4 is not None and v1 is not None and v4>=S4_OB and v1<=S1_OS
            is_super = all(v is not None and v<=20.5 for v in (v1,v2,v3,v4)) and has_bull
            super_trigger=False
            if is_super and len(stoch_s1.k_vals)>=2 and stoch_s1.k_vals[-2]<=20 and stoch_s1.k_vals[-1]>20:
                super_trigger=True
            flag_trigger=is_flag
            triggered=False; setup_type=""; is_rev=False
            if flag_trigger:
                triggered=True; setup_type="flag"
            elif super_trigger:
                triggered=True; setup_type="super"
            if s4_embedded>25 and setup_type=="super":
                is_rev=True
            if mode==1 and triggered:
                v4_3m=stoch_s4_3m.k_vals[-1] if stoch_s4_3m.k_vals else None
                if v4_3m is None or v4_3m <=80:
                    triggered=False
            if mode==2 and triggered:
                v4_5m=stoch_s4_5m.k_vals[-1] if stoch_s4_5m.k_vals else None
                if v4_5m is None or v4_5m <=80:
                    triggered=False
            if pos is not None:
                if r["low"] <= pos["sl"]:
                    pts=pos["sl"]-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"side":pos["side"],"strike":pos["strike"],"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":pos["sl"],"pts":pts,"reason":"SL","setup":pos["setup"]})
                    pos=None
                elif r["high"] >= pos["tp"]:
                    pts=pos["tp"]-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"side":pos["side"],"strike":pos["strike"],"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":pos["tp"],"pts":pts,"reason":"TP"})
                    pos=None
                elif minute>=915:
                    pts=c-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"side":pos["side"],"strike":pos["strike"],"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD"})
                    pos=None
                if pos is not None and minute>=915:
                    pts=c-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"side":pos["side"],"strike":pos["strike"],"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD"})
                    pos=None
            if pos is None and triggered:
                m_match=SYM_RE.match(sym)
                opt_side=m_match.group(2)
                actual_side = ("PE" if opt_side=="CE" else "CE") if is_rev else opt_side
                day_idx=GLOBAL_IDX[GLOBAL_IDX["day"]==day]
                spot_915=day_idx[day_idx["minute"]==555]["close"].values[0] if len(day_idx[day_idx["minute"]==555]) else day_idx.iloc[0]["close"]
                atm2=int(round(float(spot_915)/50)*50)
                strike=atm2-100 if actual_side=="CE" else atm2+100
                found=None
                for s2 in pdf["symbol"].unique():
                    m2=SYM_RE.match(s2)
                    if m2 and int(m2.group(1))==strike and m2.group(2)==actual_side:
                        found=s2; break
                if found is None: continue
                actual_row=pdf[(pdf["symbol"]==found) & (pdf["minute"]==minute)]
                if actual_row.empty: continue
                actual_close=float(actual_row.iloc[0]["close"])
                pos={"entry":actual_close,"entry_min":minute,"sl":actual_close-SL,"tp":actual_close+TP,"side":actual_side,"strike":strike,"symbol":found,"setup":setup_type,"is_rev":is_rev}
        if pos is not None:
            c=float(g.iloc[-1]["close"]); minute=int(g.iloc[-1]["minute"])
            pts=c-pos["entry"]
            trades.append({"day":day,"symbol":sym,"side":pos["side"],"strike":pos["strike"],"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD","setup":pos["setup"]})
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
    p.add_argument("--mode", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args=p.parse_args()
    import pandas as pd
    df_idx=load_index()
    days_sorted=sorted(df_idx["day"].unique().tolist())
    avail=[d for d in days_sorted if args.start <= d <= args.end]
    avail=[d for d in avail if option_file_for_day(d) and option_file_for_day(d).exists()]
    if args.smoke:
        avail=avail[:5]
        print("=== SMOKE TEST ===")
    print(f"Backtest {avail[0]} to {avail[-1]} ({len(avail)} days), mode {args.mode}, workers={args.workers}")
    import time
    t0=time.time()
    if args.smoke:
        GLOBAL_IDX=df_idx
        trades=[]
        for d in avail:
            trades.extend(process_day(d, args.mode))
        print(f"Smoke {len(trades)} trades in {time.time()-t0:.1f}s")
        if trades:
            import pandas as pd
            tdf=pd.DataFrame(trades)
            print(tdf.head().to_string(index=False))
            wins=len(tdf[tdf["pts"]>0])
            print(f"WR {wins/len(trades)*100:.1f}% Net {tdf['pts'].sum():.1f} PF {(tdf[tdf['pts']>0]['pts'].sum()/abs(tdf[tdf['pts']<=0]['pts'].sum())):.2f}")
            print(tdf["setup"].value_counts().to_string())
            print(tdf["reason"].value_counts().to_string())
        exit(0)
    from multiprocessing import Pool, cpu_count
    workers=min(args.workers,8,cpu_count())
    with Pool(workers, initializer=init_worker, initargs=(df_idx,)) as pool:
        results=pool.starmap(process_day, [(d, args.mode) for d in avail])
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
        print(tdf["setup"].value_counts().to_string())
        print(tdf["reason"].value_counts().to_string())
    import json
    out=Path(__file__).with_name(f"f6_super_reversal_clean_mode{args.mode}_result.json")
    out.write_text(json.dumps({"start":args.start,"end":args.end,"days":len(avail),"elapsed":elapsed,"mode":args.mode,"trades":all_trades}, indent=2, default=float), encoding="utf-8")
    print(f"JSON {out}")
