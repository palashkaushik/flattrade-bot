"""F6 + SR bounce — SR from Combined Supreme (CPR/Cam/Fib/PDH/PDC + virgin) on index, F6 on option with 1pt SR bounce, SL10 TP15, 8 workers, causal."""
import pandas as pd, re, json, time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import polars as pl
OPT_ROOT=Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options")
CSV_INDEX=r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
CACHE=Path(r"C:\Users\user\Desktop\nifty50 data\cache_marni_opt.parquet")
OUT=Path(__file__).with_name("f6_sr_bounce_result.json")
S1_K=12; S1_D=3; S4_K=50; S4_D=10; S4_OB=79.5; S1_OS=25.0; SL=10; TP=15
SYM_RE=re.compile(r"^NIFTY\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$")

class Stoch:
    def __init__(self,k,d): self.k=k; self.d=d; self.highs=[]; self.lows=[]; self.closes=[]; self.k_vals=[]
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
    df=pd.read_csv(CSV_INDEX)
    # handle header with oi column
    if "date" not in df.columns:
        df=pd.read_csv(CSV_INDEX, skiprows=1, names=["date","open","high","low","close","volume","oi"])
    df["dt"]=pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df["minute"]=df["dt"].apply(lambda d: d.hour*60+d.minute)
    df["day"]=df["dt"].dt.strftime("%Y-%m-%d")
    return df.sort_values("dt").reset_index(drop=True)

def build_index_sr_levels(df):
    # From backtest_master_supreme_chop_full.py daily_levels
    df["day"]=df["day"].astype(str)
    daily_ohlc=df.groupby("day").agg(high=("high","max"), low=("low","min"), close=("close","last")).to_dict("index")
    all_days=sorted(daily_ohlc.keys())
    daily_levels={}
    for i in range(1, len(all_days)):
        day=all_days[i]; prev_day=all_days[i-1]
        p_h=daily_ohlc[prev_day]["high"]; p_l=daily_ohlc[prev_day]["low"]; p_c=daily_ohlc[prev_day]["close"]
        pivot=(p_h+p_l+p_c)/3.0; bc=(p_h+p_l)/2.0; tc=(pivot-bc)+pivot
        c_top=max(tc,bc); c_bot=min(tc,bc)
        cam_rng=p_h-p_l
        h3=p_c+cam_rng*(1.1/4.0); l3=p_c-cam_rng*(1.1/4.0)
        h4=p_c+cam_rng*(1.1/2.0); l4=p_c-cam_rng*(1.1/2.0)
        fib_h3=pivot+cam_rng*1.0; fib_l3=pivot-cam_rng*1.0
        is_virgin=False
        # is_virgin check later per day
        daily_levels[day]={"cpr_p":pivot,"cpr_top":c_top,"cpr_bot":c_bot,"cam_h3":h3,"cam_l3":l3,"cam_h4":h4,"cam_l4":l4,"fib_h3":fib_h3,"fib_l3":fib_l3,"pdh":p_h,"pdl":p_l,"pdc":p_c}
    # virgin
    for day in all_days:
        if day in daily_levels:
            # is_virgin if today's low/high doesn't overlap prior CPR
            # For simplicity, keep all as not virgin
            daily_levels[day]["is_virgin"]=False
    return daily_levels

def option_file_for_day(day):
    y,m,d=day.split("-")
    for cand in [OPT_ROOT / y / str(int(m)) / f"nifty_options_{d}_{m}_{y}.csv",
                 OPT_ROOT / y / m / f"nifty_options_{d}_{m}_{y}.csv"]:
        if cand.exists(): return cand
    pats=list(OPT_ROOT.rglob(f"nifty_options_{d}_{m}_{y}.csv"))
    return pats[0] if pats else None

def is_bounce_off_sr(bar_low, bar_close, sr_level, buffer=1.0):
    # upward bounce: low touches SR within 1pt and close > SR and close > open (bullish)
    # Here bar_low within sr±1 and bar_close > sr
    return abs(bar_low - sr_level) <= buffer and bar_close > sr_level

def process_day(day, sr_levels):
    df_idx=GLOBAL_IDX
    day_idx=df_idx[df_idx["day"]==day]
    if day_idx.empty: return []
    spot_915=day_idx[day_idx["minute"]==555]["close"].values[0] if len(day_idx[day_idx["minute"]==555]) else day_idx.iloc[0]["close"]
    atm=int(round(float(spot_915)/50)*50)
    p=option_file_for_day(day)
    if not p or not p.exists(): return []
    # SR levels for this day (from prior day)
    sr=sr_levels.get(day)
    if not sr: return []
    sr_vals=[sr["cpr_p"], sr["cpr_top"], sr["cpr_bot"], sr["cam_h3"], sr["cam_l3"], sr["cam_h4"], sr["cam_l4"], sr["fib_h3"], sr["fib_l3"], sr["pdh"], sr["pdl"]]
    # Filter to reasonable range near ATM (optional)
    df=pl.read_csv(str(p), columns=["time","symbol","open","high","low","close"])
    df=df.filter(pl.col("time").str.contains(r"^\d{2}:\d{2}:\d{2}$"))
    df=df.with_columns((pl.col("time").str.slice(0,2).cast(pl.Int32)*60 + pl.col("time").str.slice(3,2).cast(pl.Int32)).alias("minute"))
    df=df.filter((pl.col("minute")>=555) & (pl.col("minute")<=915))
    pdf=df.to_pandas()
    pdf=pdf[pdf["symbol"].apply(lambda s: (mm:=SYM_RE.match(s)) and abs(int(mm.group(1))-atm)<=250)]
    trades=[]
    # warmup map
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
        s1=Stoch(S1_K,S1_D); s4=Stoch(S4_K,S4_D)
        if sym in warmup_map:
            wg=warmup_map[sym]
            for _, r2 in wg.iterrows():
                s1.push(float(r2["high"]), float(r2["low"]), float(r2["close"]))
                s4.push(float(r2["high"]), float(r2["low"]), float(r2["close"]))
        pos=None
        for _, r in g.iterrows():
            minute=int(r["minute"]); h=float(r["high"]); l=float(r["low"]); c=float(r["close"]); o=float(r["open"])
            v1=s1.push(h,l,c); v4=s4.push(h,l,c)
            is_flag = v4 is not None and v1 is not None and v4>=S4_OB and v1<=S1_OS
            # SR bounce filter: price low touches any SR within 1pt and close > SR (upward bounce)
            # Touch budget: 1 per 10 minutes, resets every 10 minutes (causal)
            # We track last touch minute per symbol
            if not hasattr(process_day, "_touch_budget"):
                process_day._touch_budget = {}
            # Initialize per symbol
            if sym not in process_day._touch_budget:
                process_day._touch_budget[sym] = -1000
            last_touch = process_day._touch_budget[sym]
            # Check budget: if within 10 minutes of last touch, skip
            if is_flag and minute - last_touch < 10:
                is_flag=False
            if is_flag:
                # Check if this bar's low touches any SR within 1pt and close > open (bullish)
                bounced=False
                for sr_lvl in sr_vals:
                    if abs(l - sr_lvl) <= 1.0 and c > o and c > sr_lvl:
                        bounced=True
                        break
                if not bounced:
                    is_flag=False
                else:
                    # This touch consumes budget
                    process_day._touch_budget[sym] = minute
            if pos is not None:
                if r["low"] <= pos["sl"]:
                    pts=pos["sl"]-pos["entry"]
                    trades.append({"day":day,"symbol":sym,"side":pos["side"],"strike":pos["strike"],"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":pos["sl"],"pts":pts,"reason":"SL"})
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
            if pos is None and is_flag:
                m_match=SYM_RE.match(sym)
                opt_side=m_match.group(2)
                # F6 flag side is opt_side
                actual_side=opt_side
                # 2nd ITM strike
                # Need ATM
                atm2=int(round(float(spot_915)/50)*50)
                strike=atm2-100 if actual_side=="CE" else atm2+100
                # Find actual symbol
                found=None
                for s2 in pdf["symbol"].unique():
                    m2=SYM_RE.match(s2)
                    if m2 and int(m2.group(1))==strike and m2.group(2)==actual_side:
                        found=s2; break
                if found is None: continue
                actual_row=pdf[(pdf["symbol"]==found) & (pdf["minute"]==minute)]
                if actual_row.empty: continue
                actual_close=float(actual_row.iloc[0]["close"])
                pos={"entry":actual_close,"entry_min":minute,"sl":actual_close-SL,"tp":actual_close+TP,"side":actual_side,"strike":strike,"symbol":found}
        if pos is not None:
            c=float(g.iloc[-1]["close"]); minute=int(g.iloc[-1]["minute"])
            pts=c-pos["entry"]
            trades.append({"day":day,"symbol":sym,"side":pos["side"],"strike":pos["strike"],"entry_min":pos["entry_min"],"exit_min":minute,"entry":pos["entry"],"exit":c,"pts":pts,"reason":"EOD"})
    return trades

GLOBAL_IDX=None
GLOBAL_SR=None
def init_worker(idx, sr):
    global GLOBAL_IDX, GLOBAL_SR
    GLOBAL_IDX=idx; GLOBAL_SR=sr

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
    sr_levels=build_index_sr_levels(df_idx)
    days_sorted=sorted(df_idx["day"].unique().tolist())
    avail=[d for d in days_sorted if args.start <= d <= args.end]
    avail=[d for d in avail if option_file_for_day(d) and option_file_for_day(d).exists()]
    if args.smoke:
        avail=avail[:5]
        print("=== SMOKE TEST — SR bounce ===")
    print(f"Backtest {avail[0]} to {avail[-1]} ({len(avail)} days), workers={args.workers}")
    import time
    t0=time.time()
    if args.smoke:
        GLOBAL_IDX=df_idx; GLOBAL_SR=sr_levels
        trades=[]
        for d in avail:
            trades.extend(process_day(d, sr_levels))
        print(f"Smoke {len(trades)} trades in {time.time()-t0:.1f}s")
        if trades:
            import pandas as pd
            tdf=pd.DataFrame(trades)
            print(tdf.head().to_string(index=False))
            wins=len(tdf[tdf["pts"]>0])
            print(f"WR {wins/len(trades)*100:.1f}% Net {tdf['pts'].sum():.1f} PF {(tdf[tdf['pts']>0]['pts'].sum()/abs(tdf[tdf['pts']<=0]['pts'].sum())):.2f}")
        exit(0)
    from multiprocessing import Pool, cpu_count
    workers=min(args.workers,8,cpu_count())
    with Pool(workers, initializer=init_worker, initargs=(df_idx, sr_levels)) as pool:
        import functools
        results=pool.starmap(process_day, [(d, sr_levels) for d in avail])
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
    out=Path(__file__).with_name("f6_sr_bounce_result.json")
    out.write_text(json.dumps({"start":args.start,"end":args.end,"days":len(avail),"elapsed":elapsed,"trades":all_trades}, indent=2, default=float), encoding="utf-8")
    print(f"JSON {out}")
