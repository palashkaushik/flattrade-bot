"""F6 7Y on OPTION chart — proper 60-bar warmup, SL10 TP15, 8 workers, causal."""
import pandas as pd, re, json, time
from pathlib import Path
from multiprocessing import Pool, cpu_count

DAY="2026-08-27"
OPT_ROOT=Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options")
CSV_INDEX=r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
OUT=Path(__file__).with_name("f6_7y_option_result.json")

S1_K=12; S1_D=3; S4_K=50; S4_D=10; S4_OB=79.5; S1_OS=25.0; SL=10; TP=15
SYM_RE=re.compile(r"^NIFTY\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$")

class Stoch:
    def __init__(self,k,d):
        self.k=k; self.d=d; self.highs=[]; self.lows=[]; self.closes=[]; self.k_vals=[]
    def push(self,h,l,c):
        self.highs.append(h); self.lows.append(l); self.closes.append(c)
        if len(self.highs)>self.k: self.highs.pop(0); self.lows.pop(0); self.closes.pop(0)
        if len(self.highs)<self.k: return None
        hh=max(self.highs); ll=min(self.lows)
        k=(c-ll)/(hh-ll)*100 if hh!=ll else 50
        self.k_vals.append(k)
        if len(self.k_vals)>self.d: self.k_vals.pop(0)
        if len(self.k_vals)<self.d: return None
        return sum(self.k_vals)/self.d

def load_index():
    df=pd.read_csv(CSV_INDEX, skiprows=1, names=["date","open","high","low","close","volume"])
    df["dt"]=pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df["minute"]=df["dt"].apply(lambda d: d.hour*60+d.minute)
    df["day"]=df["dt"].dt.strftime("%Y-%m-%d")
    return df.sort_values("dt").reset_index(drop=True)

def option_file_for_day(day):
    y,m,d=day.split("-")
    for cand in [OPT_ROOT / y / str(int(m)) / f"nifty_options_{d}_{m}_{y}.csv",
                 OPT_ROOT / y / m / f"nifty_options_{d}_{m}_{y}.csv"]:
        if cand.exists(): return cand
    pats=list(OPT_ROOT.rglob(f"nifty_options_{d}_{m}_{y}.csv"))
    return pats[0] if pats else None

def process_day(day):
    # ATM
    df_idx=GLOBAL_IDX
    today_idx=df_idx[df_idx["day"]==day]
    if today_idx.empty: return []
    spot_915=today_idx[today_idx["minute"]==555]["close"].values[0] if len(today_idx[today_idx["minute"]==555]) else today_idx.iloc[0]["close"]
    atm=int(round(spot_915/50)*50)
    strikes=set(range(atm-250, atm+300, 50))
    p=option_file_for_day(day)
    if not p or not p.exists(): return []
    df=pd.read_csv(str(p), usecols=["time","symbol","open","high","low","close"])
    df["minute"]=df["time"].apply(lambda t: int(t.split(":")[0])*60+int(t.split(":")[1]))
    df=df[df["symbol"].apply(lambda s: (m:=SYM_RE.match(s)) and int(m.group(1)) in strikes)]
    # warmup map
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
                df_prev=pd.read_csv(str(pp), usecols=["time","symbol","open","high","low","close"])
                df_prev["minute"]=df_prev["time"].apply(lambda t: int(t.split(":")[0])*60+int(t.split(":")[1]))
                for sym2, g2 in df_prev.groupby("symbol"):
                    g2=g2.sort_values("minute").tail(60)
                    warmup_map[sym2]=g2
            except: pass
    trades=[]
    for sym, g in df.groupby("symbol"):
        m=SYM_RE.match(sym)
        if not m: continue
        strike=int(m.group(1)); side=m.group(2)
        g=g.sort_values("minute")
        s1=Stoch(S1_K,S1_D); s4=Stoch(S4_K,S4_D)
        if sym in warmup_map:
            for _, r2 in warmup_map[sym].iterrows():
                s1.push(float(r2["high"]), float(r2["low"]), float(r2["close"]))
                s4.push(float(r2["high"]), float(r2["low"]), float(r2["close"]))
        pos=None
        for _,r in g.iterrows():
            minute=int(r["minute"])
            if not (555 <= minute <= 915): continue
            h=float(r["high"]); l=float(r["low"]); c=float(r["close"])
            v1=s1.push(h,l,c); v4=s4.push(h,l,c)
            is_flag=v4 is not None and v1 is not None and v4>=S4_OB and v1<=S1_OS
            if pos is not None:
                if r["low"] <= pos["sl"]:
                    pts=pos["sl"]-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"side":side,"strike":strike,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":pos["sl"],"pts":pts,"reason":"SL"})
                    pos=None
                elif r["high"] >= pos["tp"]:
                    pts=pos["tp"]-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"side":side,"strike":strike,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":pos["tp"],"pts":pts,"reason":"TP"})
                    pos=None
                elif minute>=915:
                    pts=c-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"side":side,"strike":strike,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD"})
                    pos=None
                if pos is not None and minute>=915:
                    pts=c-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"side":side,"strike":strike,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD"})
                    pos=None
            if pos is None and is_flag:
                pos={"entry":c,"entry_min":minute,"sl":c-SL,"tp":c+TP}
        # EOD for remaining pos
        if pos is not None:
            # get last close
            last=g.iloc[-1]
            c=float(last["close"]); minute=int(last["minute"])
            pts=c-pos["entry"]
            trades.append({"day":day,"symbol":sym,"side":side,"strike":strike,"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD"})
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
    df_idx=load_index()
    days_sorted=sorted(df_idx["day"].unique().tolist())
    avail=[d for d in days_sorted if args.start <= d <= args.end]
    # filter to days with option file
    avail=[d for d in avail if option_file_for_day(d) and option_file_for_day(d).exists()]
    if args.smoke:
        avail=avail[:5]
        print("=== SMOKE TEST — 5 DAYS ONLY ===")
    print(f"Backtest {avail[0]} to {avail[-1]} ({len(avail)} days), workers={args.workers}")
    t0=time.time()
    # smoke check
    if args.smoke:
        # quick smoke: process 5 days
        GLOBAL_IDX=df_idx
        trades=[]
        for d in avail:
            trades.extend(process_day(d))
        print(f"Smoke {len(trades)} trades in {time.time()-t0:.1f}s")
        # summary
        if trades:
            import pandas as pd
            tdf=pd.DataFrame(trades)
            print(tdf.head().to_string(index=False))
            wins=len(tdf[tdf["pts"]>0])
            print(f"WR {wins/len(trades)*100:.1f}% Net {tdf['pts'].sum():.1f} PF {(tdf[tdf['pts']>0]['pts'].sum()/abs(tdf[tdf['pts']<=0]['pts'].sum())):.2f}")
        exit(0)
    # full 7Y with 8 workers
    workers=min(args.workers,8,cpu_count())
    with Pool(workers, initializer=init_worker, initargs=(df_idx,)) as pool:
        results=pool.map(process_day, avail)
    all_trades=[t for lst in results for t in lst]
    elapsed=time.time()-t0
    print(f"Completed {len(avail)} days in {elapsed:.1f}s, {len(all_trades)} trades")
    if all_trades:
        import pandas as pd
        tdf=pd.DataFrame(all_trades)
        wins=len(tdf[tdf["pts"]>0])
        print(f"Total {len(all_trades)} WR {wins/len(all_trades)*100:.1f}% Net {tdf['pts'].sum():.1f} PF {(tdf[tdf['pts']>0]['pts'].sum()/abs(tdf[tdf['pts']<=0]['pts'].sum())):.2f}")
        # yearly
        tdf["year"]=pd.to_datetime(tdf["day"]).dt.year
        for y, g in tdf.groupby("year"):
            wins_y=len(g[g["pts"]>0])
            print(f"{y}: {len(g)} WR {wins_y/len(g)*100:.1f}% Net {g['pts'].sum():.1f} PF {(g[g['pts']>0]['pts'].sum()/abs(g[g['pts']<=0]['pts'].sum())):.2f}")
    # export
    out=OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    payload={"start":args.start,"end":args.end,"days":len(avail),"elapsed":elapsed,"trades":all_trades}
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"JSON {out}")
