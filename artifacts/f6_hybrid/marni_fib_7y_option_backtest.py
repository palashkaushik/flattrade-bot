"""Marnie Fib 7Y — REAL OPTION PREMIUMS (index signals, option fills).

Signals: same as marni_fib_7y_backtest.py (UT no-blue, 15m HA+LinReg11+UT, middle>=5, mirror, span>20, 0.786±3)
Execution: index 0.786 touch -> buy ATM±100 option at its premium (close) at touch minute,
           exit when INDEX hits TP0.0/SL/EOD -> sell option at its premium at exit minute.
P&L = (exit_premium - entry_premium) for long CE/PE, fees/slippage optional.
Causal, pointer arrays, 8 workers, parity with live.
"""
import json, time, re
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

CSV_INDEX = r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
OPT_ROOT = Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options")
OUT_JSON = Path(__file__).with_name("marni_fib_7y_option_result.json")

UT_KEY=1.0; UT_ATR=10; ENTRY=0.786; TP_LEVEL=0.29; SL_LEVEL=0.079; TOUCH=3.0
SESSION_START=555; SESSION_END=915; MIN_SPAN=20.0; LOT=65
SLIPPAGE=1.0  # per side, matches ledger policy

SYM_RE=re.compile(r"^NIFTY\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$")

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

def load_index():
    df=pd.read_csv(CSV_INDEX, skiprows=1, names=["date","open","high","low","close","volume"])
    df["dt"]=pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df["minute"]=df["dt"].apply(lambda d: d.hour*60+d.minute)
    df["day"]=df["dt"].dt.strftime("%Y-%m-%d")
    return df.sort_values("dt").reset_index(drop=True)

def build_index_state(df, start_day, end_day):
    ut=UTBot(); bc=BiasComputer()
    rows=[]; colors=[]; bias_by_min={}
    day_to_indices={}
    pending=None; cur_slot=None; cur_bar=None
    for _,r in df.iterrows():
        day=r["day"]; in_range = start_day <= day <= end_day
        c={"open":r["open"],"high":r["high"],"low":r["low"],"close":r["close"],"minute":int(r["minute"])}
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
                cur_bar["high"]=max(cur_bar["high"], c["high"]); cur_bar["low"]=min(cur_bar["low"], c["low"]); cur_bar["close"]=c["close"]
            if in_range: bias_by_min[(day,m)]=pending
    if cur_bar is not None: bc.feed(cur_bar)
    return rows, colors, bias_by_min, day_to_indices

def find_impulses_for_day(seq):
    N=len(seq); imps=[]
    for i in range(1,N):
        c_prev,c_cur=seq[i-1][1],seq[i][1]
        if c_prev=="green" and c_cur=="red":
            j=i
            while j<N and seq[j][1]=="red": j+=1
            if j-i>=5 and j<N and seq[j][1]=="green":
                if not (seq[i-1][2] > seq[j][2]): continue
                a=i-1
                while a-1>=0 and seq[a-1][1]=="green": a-=1
                peak=max(seq[t][2] for t in range(a,i))
                bottom=min(seq[t][3] for t in range(i,j))
                rng=peak-bottom
                if rng>MIN_SPAN:
                    imps.append({"side":"PE","peak":peak,"bottom":bottom,"span":rng,"range_start":seq[i-1][4],"range_end":seq[j][4],"last_start":j})
        if c_prev=="red" and c_cur=="green":
            j=i
            while j<N and seq[j][1]=="green": j+=1
            if j-i>=5 and j<N and seq[j][1]=="red":
                if not (seq[i-1][3] < seq[j][3]): continue
                a=i-1
                while a-1>=0 and seq[a-1][1]=="red": a-=1
                peak=max(seq[t][2] for t in range(i,j))
                k=j
                while k<N and seq[k][1]=="red": k+=1
                bottom=min(seq[t][3] for t in range(j,k))
                rng=peak-bottom
                if rng>MIN_SPAN:
                    imps.append({"side":"CE","peak":peak,"bottom":bottom,"span":rng,"range_start":seq[i-1][4],"range_end":seq[j][4],"last_start":j})
    imps.sort(key=lambda x: (x["range_start"], 0 if x["side"]=="PE" else 1))
    return imps

# ---- option helpers (pointer) ----
def option_file_for_day(day):
    # day = YYYY-MM-DD -> file nifty_options_DD_MM_YYYY.csv under OPT_ROOT/YYYY/M
    y,m,d = day.split("-")
    # try m without leading zero and with
    for cand in [OPT_ROOT / y / str(int(m)) / f"nifty_options_{d}_{m}_{y}.csv",
                 OPT_ROOT / y / m / f"nifty_options_{d}_{m}_{y}.csv"]:
        if cand.exists():
            return cand
    # also try single digit day
    cand = OPT_ROOT / y / str(int(m)) / f"nifty_options_{int(d):02d}_{int(m):02d}_{y}.csv"
    if cand.exists(): return cand
    # fallback glob
    pats = list(OPT_ROOT.rglob(f"nifty_options_{d}_{m}_{y}.csv"))
    if pats: return pats[0]
    pats = list(OPT_ROOT.rglob(f"nifty_options_{int(d):02d}_{int(m):02d}_{y}.csv"))
    if pats: return pats[0]
    return None

def load_option_day(day):
    p=option_file_for_day(day)
    if p is None or not p.exists():
        return None
    try:
        df=pd.read_csv(p, usecols=["time","symbol","open","high","low","close"], engine="c")
    except Exception:
        return None
    if df.empty: return None
    # minute
    df["minute"]=df["time"].apply(lambda t: int(t.split(":")[0])*60+int(t.split(":")[1]))
    # index by (symbol, minute) -> row
    # build dict {(strike, side): {minute: (open,high,low,close)}}
    out={}
    for _,r in df.iterrows():
        sym=r["symbol"]; mt=SYM_RE.match(sym)
        if not mt: continue
        strike=int(mt.group(1)); side=mt.group(2)
        key=(strike, side)
        if key not in out: out[key]={}
        out[key][int(r["minute"])]=(float(r["open"]),float(r["high"]),float(r["low"]),float(r["close"]))
    return out

# globals for workers
G_ROWS=None; G_COLORS=None; G_BIAS=None
def _init_worker(rows, colors, bias):
    global G_ROWS, G_COLORS, G_BIAS
    G_ROWS=rows; G_COLORS=colors; G_BIAS=bias

def _process_day_task(args):
    day, gidxs = args
    seq=[(g, G_COLORS[g], G_ROWS[g]["high"], G_ROWS[g]["low"], G_ROWS[g]["minute"]) for g in gidxs]
    imps=find_impulses_for_day(seq)
    # load option day
    opt_map=load_option_day(day)
    trades=[]
    # need index close per minute for ATM
    idx_min_to_close={ r["minute"]: r["close"] for r in [G_ROWS[g] for g in gidxs] }
    # also need index high/low per minute for exit triggers (we have seq)
    seq_min_to_hl={ m:(hi,lo) for _,_,hi,lo,m in seq }
    for imp in imps:
        peak,bottom,rng,side = imp["peak"], imp["bottom"], imp["span"], imp["side"]
        entry_idx = bottom + ENTRY*rng if side=="PE" else peak - ENTRY*rng
        tp_idx = bottom if side=="PE" else peak
        sl_idx = peak + SL_LEVEL*rng if side=="PE" else bottom - SL_LEVEL*rng
        tp1_idx = bottom + TP_LEVEL*rng if side=="PE" else peak - TP_LEVEL*rng
        start=imp["last_start"]
        end=min(len(seq), start+60)
        hit=None
        for t in range(start, end):
            _,_,hi,lo,m = seq[t]
            b=G_BIAS.get((day,m))
            if b is None: continue
            if not (hi >= entry_idx - TOUCH and lo <= entry_idx + TOUCH): continue
            if side=="CE" and not b["bullish"]: continue
            if side=="PE" and not b["bearish"]: continue
            # need option strike at this minute
            spot=idx_min_to_close.get(m)
            if spot is None: continue
            atm=int(round(spot/50.0)*50)
            strike= atm -100 if side=="CE" else atm+100
            key=(strike, side)
            if opt_map is None or key not in opt_map or m not in opt_map[key]:
                # no option data for this minute -> skip this touch, try next minute
                continue
            opt_close=opt_map[key][m][3]
            hit=(t,m,spot,strike,opt_close,key)
            break
        if hit is None:
            # Check if there was any index touch at all (even without option) to label bias vs no-touch - keep as UNTRIGGERED
            trades.append({"day":day,"side":side,"result":"UNTRIGGERED","range_start":imp["range_start"],"range_end":imp["range_end"],
                           "peak":peak,"bottom":bottom,"span":rng,"entry_idx":entry_idx,"tp1_idx":tp1_idx,"tp2_idx":tp_idx,"sl_idx":sl_idx,
                           "entry_min":None,"exit_min":None,"opt_entry":None,"opt_exit":None,"pts_opt":None})
            continue
        t,m,spot,strike,opt_entry,_key = hit
        # scan option/index for exit (index triggers, option fills)
        res=None; ex_min=None; opt_exit=None
        # need to scan forward in time order
        for u in range(t, len(seq)):
            _,_,hi,lo,mm = seq[u]
            # index trigger
            if side=="CE":
                if lo <= sl_idx:  # SL hit on index
                    # need option premium at mm
                    km=(strike, side)
                    if opt_map and km in opt_map and mm in opt_map[km]:
                        opt_exit=opt_map[km][mm][3]  # close
                        res,ex_min="SL",mm
                        break
                if hi >= tp_idx:
                    km=(strike, side)
                    if opt_map and km in opt_map and mm in opt_map[km]:
                        opt_exit=opt_map[km][mm][3]
                        res,ex_min="TP0.0",mm
                        break
            else:
                if hi >= sl_idx:
                    km=(strike, side)
                    if opt_map and km in opt_map and mm in opt_map[km]:
                        opt_exit=opt_map[km][mm][3]
                        res,ex_min="SL",mm
                        break
                if lo <= tp_idx:
                    km=(strike, side)
                    if opt_map and km in opt_map and mm in opt_map[km]:
                        opt_exit=opt_map[km][mm][3]
                        res,ex_min="TP0.0",mm
                        break
            if mm >= SESSION_END and res is None:
                km=(strike, side)
                if opt_map and km in opt_map and mm in opt_map[km]:
                    opt_exit=opt_map[km][mm][3]
                    res,ex_min="EOD",mm
                    break
        if res is None:
            # EOD at last available option bar
            last_m=seq[-1][4]
            km=(strike, side)
            if opt_map and km in opt_map and last_m in opt_map[km]:
                opt_exit=opt_map[km][last_m][3]
                res,ex_min="EOD",last_m
            else:
                # fallback to last seq close? use index close as proxy
                opt_exit=opt_entry
                res,ex_min="EOD",last_m
        # P&L on option premium (long) with slippage
        # buy at opt_entry + slippage, sell at opt_exit - slippage? For long, slippage against you both sides
        entry_fill=opt_entry + SLIPPAGE
        exit_fill=opt_exit - SLIPPAGE
        pts_opt = round(exit_fill - entry_fill, 2)  # long
        # also keep index pts for reference
        pts_idx = (tp_idx - entry_idx) if res.startswith("TP") else (sl_idx - entry_idx) if side=="CE" else (entry_idx - tp_idx) if res.startswith("TP") else (entry_idx - sl_idx)
        # but store actual index move from entry to exit index level
        exit_idx_val = tp_idx if res.startswith("TP") else sl_idx if res=="SL" else seq[-1][2]  # approx
        pts_idx_real = (exit_idx_val - entry_idx) if side=="CE" else (entry_idx - exit_idx_val)
        trades.append({"day":day,"side":side,"result":"TRADE","range_start":imp["range_start"],"range_end":imp["range_end"],
                       "peak":peak,"bottom":bottom,"span":rng,"entry_idx":entry_idx,"tp1_idx":tp1_idx,"tp2_idx":tp_idx,"sl_idx":sl_idx,
                       "entry_min":m,"exit_min":ex_min,"strike":strike,"opt_entry":opt_entry,"opt_exit":opt_exit,
                       "pts_opt":pts_opt,"pts_idx":round(pts_idx,2),"exit_reason":res,"spot":spot})
    return trades

def backtest(start_day, end_day, smoke=False, workers=8):
    df=load_index()
    days_sorted=sorted(df["day"].unique())
    avail=[d for d in days_sorted if start_day <= d <= end_day]
    # filter to days that have option file (optional)
    avail_with_opt=[d for d in avail if option_file_for_day(d) is not None]
    if smoke:
        avail=avail_with_opt[:5]
        print("=== SMOKE TEST — 5 DAYS ONLY (option) ===")
    else:
        avail=avail_with_opt
    print(f"Backtest {avail[0]} to {avail[-1]} ({len(avail)} days, {len(days_sorted)} total), warmup from {days_sorted[0]}, workers={workers}")
    t0=time.time()
    rows, colors, bias_by_min, day_to_indices = build_index_state(df, avail[0], avail[-1])
    # filter avail to those with indices
    tasks=[(day, day_to_indices[day]) for day in avail if day in day_to_indices]
    workers=min(max(1,workers),8,cpu_count())
    all_trades=[]
    if workers==1:
        for day,gidxs in tasks:
            seq=[(g, colors[g], rows[g]["high"], rows[g]["low"], rows[g]["minute"]) for g in gidxs]
            imps=find_impulses_for_day(seq)
            # reuse _process_day_task logic but single threaded - call directly
            # need opt_map inside
            all_trades.extend(_process_day_task((day,gidxs)) if False else [])
            # instead call via global hack
            pass
        # fallback single: use pool-less loop via function that loads opt
        import types
        # simple loop
        for day,gidxs in tasks:
            seq=[(g, colors[g], rows[g]["high"], rows[g]["low"], rows[g]["minute"]) for g in gidxs]
            imps=find_impulses_for_day(seq)
            # need to set globals for helper
            global G_ROWS, G_COLORS, G_BIAS
            G_ROWS, G_COLORS, G_BIAS = rows, colors, bias_by_min
            all_trades.extend(_process_day_task((day,gidxs)))
    else:
        with Pool(workers, initializer=_init_worker, initargs=(rows, colors, bias_by_min)) as pool:
            for day_trades in pool.map(_process_day_task, tasks):
                all_trades.extend(day_trades)
    elapsed=time.time()-t0
    print(f"Completed {len(avail)} days in {elapsed:.2f}s, {len(all_trades)} impulses, {sum(1 for t in all_trades if t['result']=='TRADE')} option trades")
    return all_trades, avail, elapsed

def summarize(all_trades, days):
    trades=[t for t in all_trades if t["result"]=="TRADE"]
    wins=[t for t in trades if t["pts_opt"] is not None and t["pts_opt"]>0]
    losses=[t for t in trades if t["pts_opt"] is not None and t["pts_opt"]<=0]
    gross_w=sum(t["pts_opt"] for t in wins) if wins else 0
    gross_l=abs(sum(t["pts_opt"] for t in losses)) if losses else 0
    pf=gross_w/gross_l if gross_l else float("inf")
    return {
        "impulses": len(all_trades),
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins)/len(trades)*100,2) if trades else 0,
        "net_pts_opt": round(sum(t["pts_opt"] for t in trades if t["pts_opt"] is not None),2),
        "net_rs_opt": round(sum(t["pts_opt"] for t in trades if t["pts_opt"] is not None)*LOT),
        "pf": round(pf,4) if pf!=float("inf") else "inf",
        "days": len(days),
    }

if __name__=="__main__":
    import argparse, json
    p=argparse.ArgumentParser()
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2026-08-27")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=str(OUT_JSON))
    args=p.parse_args()
    trades, days, elapsed = backtest(args.start, args.end, smoke=args.smoke, workers=args.workers)
    summ=summarize(trades, days)
    print(json.dumps(summ, indent=2))
    from collections import defaultdict
    by_year=defaultdict(list)
    for t in trades:
        if t["result"]=="TRADE": by_year[t["day"][:4]].append(t)
    print("\nYearly (option premium):")
    for y in sorted(by_year):
        s=summarize([t for t in trades if t["day"][:4]==y], sorted(set(t["day"] for t in by_year[y])))
        print(f"{y}: trades={s['trades']} WR={s['win_rate']}% net_pts_opt={s['net_pts_opt']} PF={s['pf']}")
    out_path=Path(args.out)
    if args.smoke: out_path=out_path.with_name(out_path.stem+"_smoke"+out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload={"start":args.start,"end":args.end,"days":len(days),"elapsed_s":round(elapsed,2),
             "params":{"UT_KEY":UT_KEY,"UT_ATR":UT_ATR,"ENTRY":ENTRY,"TP":0.0,"SL":1.079,"TOUCH":TOUCH,"MIN_SPAN":MIN_SPAN,"mirror":True,"bias":"15m HA+LinReg11+UT","incremental":True,"pointer":True,"causal":True,"execution":"real_option_premium","slippage":SLIPPAGE},
             "summary":summ,"trades":trades}
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"JSON: {out_path}")
    import csv
    csv_path=out_path.with_suffix(".csv")
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["day","side","result","range_start","range_end","peak","bottom","entry_idx","tp2_idx","sl_idx","entry_min","exit_min","strike","opt_entry","opt_exit","pts_opt","reason"])
        for t in trades:
            w.writerow([t.get("day"),t.get("side"),t.get("result"),t.get("range_start"),t.get("range_end"),
                        round(t.get("peak",0),2) if t.get("peak") else "", round(t.get("bottom",0),2) if t.get("bottom") else "",
                        round(t.get("entry_idx",0),2) if t.get("entry_idx") else "", round(t.get("tp2_idx",0),2) if t.get("tp2_idx") else "", round(t.get("sl_idx",0),2) if t.get("sl_idx") else "",
                        t.get("entry_min"),t.get("exit_min"),t.get("strike"),t.get("opt_entry"),t.get("opt_exit"),t.get("pts_opt"),t.get("exit_reason","")])
    print(f"CSV: {csv_path}")
    # parity 2026-08-27
    live=[t for t in trades if t["day"]=="2026-08-27" and t["result"]=="TRADE"]
    print(f"\nParity 2026-08-27 option trades: {len(live)}")
    for t in live:
        print(f"  {t['side']} {t['range_start']}-{t['range_end']} idx {t['entry_idx']:.2f} opt {t['opt_entry']:.2f}->{t['opt_exit']:.2f} pts_opt {t['pts_opt']:.2f} {t['exit_reason']}")
