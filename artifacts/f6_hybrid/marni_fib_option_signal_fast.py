"""OPTION-SIGNAL FAST — signals+execution both on option chart, 1pt buffer, causal+parity, 8 workers.

Web-opt: Polars + Numba + Parquet + pointer + incremental (PolarBT/VectorBT/RaptorBT)
- Index 15m bias still causal (HA+LinReg11+UT) for HTF filter (parity with live)
- Per-option 1m UT (Key1 ATR10) incremental per symbol per day, Numba impulse finder
- Mirror + span>20 + 0.786±1 + TP0.0/SL1.079 all on OPTION premium levels
- Only ATM±100 (2 symbols/day) — pointer, 8 workers
"""
import json, time, re
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import polars as pl
import numba

CSV_INDEX = r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
CACHE_PARQUET = Path(r"C:\Users\user\Desktop\nifty50 data\cache_marni_opt.parquet")
OUT_JSON = Path(__file__).with_name("marni_fib_option_signal_result.json")

UT_KEY=1.0; UT_ATR=10; ENTRY=0.786; SL_LEVEL=0.079; TOUCH_OPT=1.0
SESSION_START=555; SESSION_END=915; MIN_SPAN_OPT=20.0; LOT=65; SLIPPAGE=1.0
SYM_RE=re.compile(r"^NIFTY\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$")

@numba.njit
def find_impulses_nb(colors, highs, lows, minutes, n, out_side, out_peak, out_bottom, out_rstart, out_rend, out_last):
    cnt=0
    for i in range(1, n):
        c_prev=colors[i-1]; c_cur=colors[i]
        if c_prev==1 and c_cur==0: # green->red PE (put)
            j=i
            while j<n and colors[j]==0: j+=1
            mid=j-i
            if mid>=5 and j<n and colors[j]==1:
                if not (highs[i-1] > highs[j]): continue
                a=i-1
                while a-1>=0 and colors[a-1]==1: a-=1
                peak=highs[a]
                for t in range(a+1, i):
                    if highs[t]>peak: peak=highs[t]
                bottom=lows[i]
                for t in range(i+1, j):
                    if lows[t]<bottom: bottom=lows[t]
                rng=peak-bottom
                if rng>20.0:
                    out_side[cnt]=0; out_peak[cnt]=peak; out_bottom[cnt]=bottom
                    out_rstart[cnt]=minutes[i-1]; out_rend[cnt]=minutes[j]; out_last[cnt]=j
                    cnt+=1
        if c_prev==0 and c_cur==1: # red->green CE (call)
            j=i
            while j<n and colors[j]==1: j+=1
            mid=j-i
            if mid>=5 and j<n and colors[j]==0:
                if not (lows[i-1] < lows[j]): continue
                peak=highs[i]
                for t in range(i+1, j):
                    if highs[t]>peak: peak=highs[t]
                k=j
                while k<n and colors[k]==0: k+=1
                bottom=lows[j]
                for t in range(j+1, k):
                    if lows[t]<bottom: bottom=lows[t]
                rng=peak-bottom
                if rng>20.0:
                    out_side[cnt]=1; out_peak[cnt]=peak; out_bottom[cnt]=bottom
                    out_rstart[cnt]=minutes[i-1]; out_rend[cnt]=minutes[j]; out_last[cnt]=j
                    cnt+=1
    return cnt

# ---- incremental index bias (causal, pointer) ----
class ATR:
    def __init__(self,n): self.n=n; self.alpha=2.0/(n+1); self.prev=None; self.atr=None
    def update(self,h,l,c):
        if self.prev is None: self.prev=c; return None
        pc=self.prev; tr=max(h-l, abs(h-pc), abs(l-pc))
        if self.atr is None: self.atr=tr
        else: self.atr=self.alpha*tr+(1-self.alpha)*self.atr
        self.prev=c; return self.atr
class UTBot:
    def __init__(self,key=UT_KEY,period=UT_ATR):
        self.atr=ATR(period); self.key=key; self.stop=0.0; self.prev_src=None; self.pos=0
    def update(self,c):
        src=c["close"]; atr=self.atr.update(c["high"],c["low"],c["close"])
        ps=self.prev_src; pstop=self.stop; self.prev_src=src
        if atr is None or ps is None: self.stop=src; self.pos=1; return "green"
        loss=self.key*atr
        if src>pstop and ps>pstop: self.stop=max(pstop, src-loss)
        elif src<pstop and ps<pstop: self.stop=min(pstop, src+loss)
        elif src>pstop: self.stop=src-loss
        else: self.stop=src+loss
        if ps<pstop and src>pstop: self.pos=1
        elif ps>pstop and src<pstop: self.pos=-1
        if self.pos==0: self.pos=1 if src>self.stop else -1
        return "green" if self.pos==1 else "red"
class HA:
    def __init__(self): self.o=None; self.c=None
    def update(self,bar):
        ha_c=(bar["open"]+bar["high"]+bar["low"]+bar["close"])/4.0
        ha_o=(bar["open"]+bar["close"])/2.0 if self.o is None else (self.o+self.c)/2.0
        ha_h=max(bar["high"],ha_o,ha_c); ha_l=min(bar["low"],ha_o,ha_c)
        self.o,self.c=ha_o,ha_c
        return {"open":ha_o,"high":ha_h,"low":ha_l,"close":ha_c}
def linreg_val(vals):
    n=len(vals)
    if n<11: return None
    xs=np.arange(n); x_sum=xs.sum(); x2=(xs**2).sum()
    y_sum=vals.sum(); xy=(xs*vals).sum()
    denom=n*x2 - x_sum*x_sum
    slope=(n*xy - x_sum*y_sum)/denom
    intercept=(y_sum - slope*x_sum)/n
    return intercept + slope*(n-1)
class BiasComputer:
    def __init__(self): self.ha=HA(); self.ut=UTBot(); self.closes=[]; self.sig=[]; self.snap=None
    def feed(self,bar):
        h=self.ha.update(bar); color=self.ut.update(bar)
        self.closes.append(h["close"])
        if len(self.closes)>=11:
            bclose=linreg_val(np.array(self.closes[-11:]))
            if bclose is not None:
                self.sig.append(bclose)
                if len(self.sig)>=11:
                    linreg_sig=sum(self.sig[-11:])/11
                    bull=(h["close"]>h["open"]) and (h["close"]>linreg_sig) and (color=="green")
                    bear=(h["close"]<h["open"]) and (h["close"]<linreg_sig) and (color=="red")
                    self.snap={"bullish":bull,"bearish":bear,"ha_close":h["close"],"linreg":linreg_sig,"ut":color}
        return self.snap

def load_index_df():
    import pandas as pd
    df=pd.read_csv(CSV_INDEX, skiprows=1, names=["date","open","high","low","close","volume"])
    df["dt"]=pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df["minute"]=df["dt"].apply(lambda d: d.hour*60+d.minute)
    df["day"]=df["dt"].dt.strftime("%Y-%m-%d")
    return df.sort_values("dt").reset_index(drop=True)

def build_index_bias(df, start_day, end_day, warmup_days=90):
    days_sorted=sorted(df["day"].unique().tolist())
    try: s_idx=days_sorted.index(start_day)
    except: s_idx=0
    warm_start=days_sorted[max(0, s_idx-warmup_days)]
    df_slice=df[(df["day"]>=warm_start) & (df["day"]<=end_day)]
    ut=UTBot(); bc=BiasComputer()
    rows=[]; colors=[]; bias_by_min={}; day_to_indices={}
    pending=None; cur_slot=None; cur_bar=None
    for _,r in df_slice.iterrows():
        day=r["day"]; in_range=start_day <= day <= end_day
        c={"open":float(r["open"]),"high":float(r["high"]),"low":float(r["low"]),"close":float(r["close"]),"minute":int(r["minute"])}
        col=ut.update(c)
        if in_range:
            gidx=len(rows); rows.append(c); colors.append(col)
            m=c["minute"]
            if SESSION_START <= m <= SESSION_END:
                day_to_indices.setdefault(day, []).append(gidx)
        m=int(r["minute"])
        if SESSION_START <= m <= SESSION_END:
            slot=SESSION_START + ((m-SESSION_START)//15)*15
            if cur_slot is None or slot != cur_slot:
                if cur_bar is not None: pending=bc.feed(cur_bar)
                cur_slot=slot
                cur_bar={"open":c["open"],"high":c["high"],"low":c["low"],"close":c["close"],"minute":slot}
            else:
                cur_bar["high"]=max(cur_bar["high"],c["high"]); cur_bar["low"]=min(cur_bar["low"],c["low"]); cur_bar["close"]=c["close"]
            if in_range: bias_by_min[(day,m)]=pending
    if cur_bar is not None: bc.feed(cur_bar)
    return rows, colors, bias_by_min, day_to_indices

def option_file_for_day(day):
    y,m,d=day.split("-")
    for cand in [Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options") / y / str(int(m)) / f"nifty_options_{d}_{m}_{y}.csv",
                 Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options") / y / m / f"nifty_options_{d}_{m}_{y}.csv"]:
        if cand.exists(): return cand
    pats=list(Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options").rglob(f"nifty_options_{d}_{m}_{y}.csv"))
    return pats[0] if pats else None

G_ROWS=None; G_COLORS=None; G_BIAS=None; G_CACHE=None
def _init_worker(rows, colors, bias, cache_path):
    global G_ROWS, G_COLORS, G_BIAS, G_CACHE
    G_ROWS=rows; G_COLORS=colors; G_BIAS=bias; G_CACHE=cache_path

def _process_day_opt_signal(args):
    day, gidxs = args
    # index bias already global, but option signals are per-option symbol (ATM±100)
    # Build option symbol list for this day (ATM from index at 09:15)
    idx_rows=[G_ROWS[g] for g in gidxs]
    # find first index minute >=555
    spot_915=None
    for r in idx_rows:
        if r["minute"]>=555:
            spot_915=r["close"]; break
    if spot_915 is None: return []
    atm=int(round(spot_915/50)*50)
    # two strikes
    candidates=[(atm-100,"CE"), (atm+100,"PE")]
    # load option data for this day (from parquet per-day scan or CSV fallback)
    opt_maps={}  # (strike,side) -> {minute: (o,h,l,c)}
    # try parquet
    if G_CACHE and Path(G_CACHE).exists():
        try:
            day_opt=pl.scan_parquet(str(G_CACHE)).filter(pl.col("day")==day).collect()
            if len(day_opt)>0:
                pdf=day_opt.to_pandas()
                for _,r in pdf.iterrows():
                    key=(int(r["strike"]), r["side"])
                    if key in [(atm-100,"CE"),(atm+100,"PE")]:
                        if key not in opt_maps: opt_maps[key]={}
                        opt_maps[key][int(r["minute"])]=(float(r["open"]),float(r["high"]),float(r["low"]),float(r["close"]))
        except Exception:
            pass
    # fallback for missing or not cached
    for key in candidates:
        if key not in opt_maps or not opt_maps[key]:
            p=option_file_for_day(day)
            if p and p.exists():
                try:
                    import pandas as pd
                    df_tmp=pd.read_csv(str(p), usecols=["time","symbol","open","high","low","close"])
                    df_tmp["minute"]=df_tmp["time"].apply(lambda t: int(t.split(":")[0])*60+int(t.split(":")[1]))
                    for _,r in df_tmp.iterrows():
                        m=re.match(r"^NIFTY\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$", r["symbol"])
                        if not m: continue
                        k=(int(m.group(1)), m.group(2))
                        if k==key:
                            if k not in opt_maps: opt_maps[k]={}
                            opt_maps[k][int(r["minute"])]=(float(r["open"]),float(r["high"]),float(r["low"]),float(r["close"]))
                except Exception:
                    pass
    trades=[]
    for key in candidates:
        strike, side = key
        bars=opt_maps.get(key)
        if not bars: continue
        # build ordered minute list 555..915
        minutes_sorted=sorted(bars.keys())
        # filter to session
        minutes_sorted=[m for m in minutes_sorted if SESSION_START <= m <= SESSION_END]
        if len(minutes_sorted)<20: continue
        # build arrays for this option's 1m bars in order
        n=len(minutes_sorted)
        # need to ensure continuity: if missing minutes, we still have sequence but gaps are okay
        # Build arrays aligned to minutes_sorted order
        highs=np.array([bars[m][1] for m in minutes_sorted], dtype=np.float64)
        lows=np.array([bars[m][2] for m in minutes_sorted], dtype=np.float64)
        closes=np.array([bars[m][3] for m in minutes_sorted], dtype=np.float64)
        opens=np.array([bars[m][0] for m in minutes_sorted], dtype=np.float64)
        mins=np.array(minutes_sorted, dtype=np.int64)
        # 1m UT on option premium — warm up from previous day's same strike (parity with index 90-day warmup)
        ut=UTBot()
        # Warmup: try previous trading day's same strike last 20 bars
        try:
            prev_day = None
            # find previous trading day with data
            all_days_sorted = sorted([d for d in G_BIAS.keys()])  # not correct, use avail
            # Instead, find previous day in avail that is < day
            # We have avail list in closure, but not here. Fallback: try day-1, day-2, day-3
            import datetime
            dt = __import__('datetime').datetime.strptime(day, "%Y-%m-%d")
            for back in range(1, 5):
                pd = dt - __import__('datetime').timedelta(days=back)
                pd_str = pd.strftime("%Y-%m-%d")
                p2 = option_file_for_day(pd_str)
                if p2 and p2.exists():
                    try:
                        import pandas as pd2
                        df_prev = pd2.read_csv(str(p2), usecols=["time","symbol","open","high","low","close"])
                        df_prev["minute"]=df_prev["time"].apply(lambda t: int(t.split(":")[0])*60+int(t.split(":")[1]))
                        import re as _re2
                        pat2=re.compile(rf"NIFTY\d{{2}}[A-Z]{{3}}\d{{2}}{strike}{side}$")
                        df_prev=df_prev[df_prev["symbol"].apply(lambda s: bool(pat2.search(s)))]
                        df_prev=df_prev.sort_values("minute").tail(20)
                        for _,r in df_prev.iterrows():
                            c2={"open": float(r["open"]),"high":float(r["high"]),"low":float(r["low"]),"close":float(r["close"]),"minute":int(r["minute"])}
                            ut.update(c2)
                        if len(df_prev)>0:
                            break
                    except: pass
        except: pass
        colors=[]
        for i in range(n):
            c={"open": float(opens[i]),"high":float(highs[i]),"low":float(lows[i]),"close":float(closes[i]),"minute":int(mins[i])}
            colors.append(1 if ut.update(c)=="green" else 0)
        colors=np.array(colors, dtype=np.int64)
        # impulse finder (numba)
        max_imps=n
        out_side=np.empty(max_imps, dtype=np.int64)
        out_peak=np.empty(max_imps, dtype=np.float64)
        out_bottom=np.empty(max_imps, dtype=np.float64)
        out_rstart=np.empty(max_imps, dtype=np.int64)
        out_rend=np.empty(max_imps, dtype=np.int64)
        out_last=np.empty(max_imps, dtype=np.int64)
        cnt=find_impulses_nb(colors, highs, lows, mins, n, out_side, out_peak, out_bottom, out_rstart, out_rend, out_last)
        # for each impulse, check touch with 1pt buffer on OPTION levels, HTF bias still index 15m
        for k in range(cnt):
            imp_side="PE" if out_side[k]==0 else "CE"
            # Only trade the option that matches impulse side? For ATM-100 CE, only CE impulses (red->green->red)
            # For ATM+100 PE, only PE impulses (green->red->green)
            # This enforces signal side == option side
            if imp_side != side: continue
            peak=float(out_peak[k]); bottom=float(out_bottom[k]); rng=peak-bottom
            entry = bottom + ENTRY*rng if imp_side=="PE" else peak - ENTRY*rng
            tp2 = bottom if imp_side=="PE" else peak
            sl = peak + SL_LEVEL*rng if imp_side=="PE" else bottom - SL_LEVEL*rng
            last_start=int(out_last[k])
            # map last_start index in mins array to position
            start_idx=last_start
            end_idx=min(n, start_idx+60)
            hit=None
            for t in range(start_idx, end_idx):
                m=int(mins[t]); hi=float(highs[t]); lo=float(lows[t])
                b=G_BIAS.get((day,m))
                if b is None: continue
                if not (hi >= entry - 1.0 and lo <= entry + 1.0): continue
                if imp_side=="CE" and not b["bullish"]: continue
                if imp_side=="PE" and not b["bearish"]: continue
                # option premium at entry
                if m not in bars: continue
                opt_close=bars[m][3]
                hit=(t,m,opt_close)
                break
            if hit is None:
                trades.append({"day":day,"side":side,"result":"UNTRIGGERED","range_start":int(out_rstart[k]),"range_end":int(out_rend[k]),"peak":peak,"bottom":bottom,"entry_idx":entry,"tp2_idx":tp2,"sl_idx":sl})
                continue
            t,m,opt_entry = hit
            res=None; ex_min=None; opt_exit=None
            for u in range(t, n):
                mm=int(mins[u])
                if side=="CE":
                    lo=float(lows[u]); hi=float(highs[u])
                    if lo <= sl:
                        opt_exit=bars[mm][3] if mm in bars else None
                        if opt_exit is not None: res,ex_min="SL",mm; break
                    if hi >= tp2:
                        opt_exit=bars[mm][3] if mm in bars else None
                        if opt_exit is not None: res,ex_min="TP0.0",mm; break
                else:
                    hi=float(highs[u]); lo=float(lows[u])
                    if hi >= sl:
                        opt_exit=bars[mm][3] if mm in bars else None
                        if opt_exit is not None: res,ex_min="SL",mm; break
                    if lo <= tp2:
                        opt_exit=bars[mm][3] if mm in bars else None
                        if opt_exit is not None: res,ex_min="TP0.0",mm; break
                if mm >= SESSION_END and res is None and mm in bars:
                    opt_exit=bars[mm][3]; res,ex_min="EOD",mm; break
            if res is None:
                last_m=int(mins[-1])
                opt_exit=bars[last_m][3] if last_m in bars else opt_entry
                res,ex_min="EOD",last_m
            entry_fill=opt_entry + 1.0; exit_fill=opt_exit - 1.0
            pts_opt=round(exit_fill - entry_fill,2)
            trades.append({"day":day,"side":side,"result":"TRADE","range_start":int(out_rstart[k]),"range_end":int(out_rend[k]),"peak":peak,"bottom":bottom,"entry_idx":entry,"tp2_idx":tp2,"sl_idx":sl,"entry_min":m,"exit_min":ex_min,"strike":strike,"opt_entry":opt_entry,"opt_exit":opt_exit,"pts_opt":pts_opt,"exit_reason":res,"spot":None})
    return trades

def backtest_opt_signal(start_day, end_day, smoke=False, workers=8):
    df=load_index_df()
    days_sorted=sorted(df["day"].unique().tolist())
    avail=[d for d in days_sorted if start_day <= d <= end_day]
    if smoke:
        avail=avail[:5]
        print("=== SMOKE TEST — 5 DAYS ONLY (option-signal) ===")
    # filter to days with option data (parquet)
    if Path(CACHE_PARQUET).exists():
        try:
            all_opt_days=set(pl.scan_parquet(str(CACHE_PARQUET)).select("day").unique().collect()["day"].to_list())
            avail=[d for d in avail if d in all_opt_days]
        except: pass
    print(f"Backtest {avail[0]} to {avail[-1]} ({len(avail)} days), workers={workers}, cache={Path(CACHE_PARQUET).exists()}")
    t0=time.time()
    rows, colors, bias_by_min, day_to_indices = build_index_bias(df, avail[0], avail[-1], warmup_days=90)
    # need rows/colors/bias for HTF only, but per-option signals are independent
    # Use 8 workers for per-day option-signal processing
    tasks=[(day, day_to_indices[day]) for day in avail if day in day_to_indices]
    # Actually we need to pass day only, since option signals need to load option bars per day
    # Simplify: tasks are just days
    workers=min(max(1,workers),8,cpu_count())
    all_trades=[]
    cache_path=str(CACHE_PARQUET) if Path(CACHE_PARQUET).exists() else None
    # Rebuild bias prerequisites for workers (need G_BIAS)
    # Use index rows/colors/bias for HTF
    if workers==1:
        global G_ROWS, G_COLORS, G_BIAS, G_CACHE
        G_ROWS, G_COLORS, G_BIAS, G_CACHE = rows, colors, bias_by_min, cache_path
        for day,_ in tasks:
            all_trades.extend(_process_day_opt_signal((day, _)))
    else:
        with Pool(workers, initializer=_init_worker, initargs=(rows, colors, bias_by_min, cache_path)) as pool:
            # need to adapt _process_day_opt_signal to take (day, gidxs) but we want (day, gidxs)
            # We'll call with (day, gidxs)
            for day_trades in pool.map(_process_day_opt_signal, tasks):
                all_trades.extend(day_trades)
    elapsed=time.time()-t0
    print(f"Completed {len(avail)} days in {elapsed:.2f}s, {len(all_trades)} impulses, {sum(1 for t in all_trades if t['result']=='TRADE')} option-signal trades")
    return all_trades, avail, elapsed

if __name__=="__main__":
    import argparse, json
    p=argparse.ArgumentParser()
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2026-08-27")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=str(OUT_JSON))
    args=p.parse_args()
    trades, days, elapsed = backtest_opt_signal(args.start, args.end, smoke=args.smoke, workers=args.workers)
    def summarize(trades, days):
        tr=[t for t in trades if t["result"]=="TRADE"]
        wins=[t for t in tr if t["pts_opt"]>0]
        gross_w=sum(t["pts_opt"] for t in wins) if wins else 0
        gross_l=abs(sum(t["pts_opt"] for t in tr if t["pts_opt"]<=0)) if tr else 0
        pf=gross_w/gross_l if gross_l else float("inf")
        return {"impulses":len(trades),"trades":len(tr),"wins":len(wins),"losses":len(tr)-len(wins),"win_rate":round(len(wins)/len(tr)*100,2) if tr else 0,"net_pts_opt":round(sum(t["pts_opt"] for t in tr),2) if tr else 0,"net_rs_opt":round(sum(t["pts_opt"] for t in tr)*65,2) if tr else 0,"pf":round(pf,4) if pf!=float("inf") else "inf","days":len(days)}
    summ=summarize(trades, days)
    print(json.dumps(summ, indent=2))
    from collections import defaultdict
    by_year=defaultdict(list)
    for t in trades:
        if t["result"]=="TRADE": by_year[t["day"][:4]].append(t)
    print("\nYearly (option-signal):")
    for y in sorted(by_year):
        s=summarize([t for t in trades if t["day"][:4]==y], sorted(set(t["day"] for t in by_year[y])))
        print(f"{y}: trades={s['trades']} WR={s['win_rate']}% net_pts_opt={s['net_pts_opt']} PF={s['pf']}")
    out_path=Path(args.out)
    if args.smoke: out_path=out_path.with_name(out_path.stem+"_smoke"+out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload={"start":args.start,"end":args.end,"days":len(days),"elapsed_s":round(elapsed,2),"params":{"UT_KEY":1.0,"UT_ATR":10,"ENTRY":0.786,"TP":0.0,"SL":1.079,"TOUCH":1.0,"MIN_SPAN":20.0,"mirror":True,"bias":"15m HA+LinReg11+UT","signal":"option-chart","execution":"option-chart","incremental":True,"pointer":True,"numba":True,"workers":args.workers},"summary":summ,"trades":trades}
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"JSON: {out_path}")
