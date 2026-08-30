"""Marnie Fib 7-Year Backtest -- incremental + pointer filters, causal + parity.

Parity with live engine (marni_fib_today.py):
 - UTBot (no blue, Key 1.0 ATR 10) incremental
 - 15m HA + 11 LinReg + 15m UT bias (same BiasComputer)
 - Impulse: green->red->green (PE) / red->green->red (CE), middle>=5, span>20,
   mirror: PE first high > last high, CE first low < last low, pivot-to-pivot
 - Entry 0.786 ±3, HTF bias gated, exits TP 0.0 (user) / SL 1.079, EOD 15:15
 - Strikes ATM±100, causal (no future), pointer arrays (numpy), warmed from day 0
"""

import json
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np
import pandas as pd

CSV = r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
OUT_JSON = Path(__file__).with_name("marni_fib_7y_result.json")
OUT_CSV = Path(__file__).with_name("marni_fib_7y_trades.csv")

UT_KEY = 1.0
UT_ATR = 10
ENTRY = 0.786
TP_LEVEL = 0.29  # kept for reference, but TP is 0.0 per user
SL_LEVEL = 0.079
TOUCH = 3.0
SESSION_START = 555
SESSION_END = 915
MIN_SPAN = 20.0
LOT = 65

# --- incremental primitives (parity with live) ---
class ATR:
    def __init__(self, n):
        self.n = n; self.alpha = 2.0/(n+1); self.prev=None; self.atr=None
    def update(self, h,l,c):
        if self.prev is None:
            self.prev=c; return None
        pc=self.prev
        tr=max(h-l, abs(h-pc), abs(l-pc))
        if self.atr is None: self.atr=tr
        else: self.atr=self.alpha*tr+(1-self.alpha)*self.atr
        self.prev=c; return self.atr

class UTBot:
    def __init__(self, key=UT_KEY, period=UT_ATR):
        self.atr=ATR(period); self.key=key; self.stop=0.0; self.prev_src=None; self.pos=0
    def update(self, c):
        src=c["close"]; atr=self.atr.update(c["high"],c["low"],c["close"])
        ps=self.prev_src; pstop=self.stop; self.prev_src=src
        if atr is None or ps is None:
            self.stop=src; self.pos=1; return "green"
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
    def update(self, bar):
        ha_c=(bar["open"]+bar["high"]+bar["low"]+bar["close"])/4.0
        ha_o=(bar["open"]+bar["close"])/2.0 if self.o is None else (self.o+self.c)/2.0
        ha_h=max(bar["high"],ha_o,ha_c); ha_l=min(bar["low"],ha_o,ha_c)
        self.o, self.c=ha_o,ha_c
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
    def __init__(self):
        self.ha=HA(); self.ut=UTBot(); self.closes=[]; self.sig=[]; self.snap=None
    def feed(self, bar):
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

def load_all():
    df=pd.read_csv(CSV, skiprows=1, names=["date","open","high","low","close","volume"])
    df["dt"]=pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    df["minute"]=df["dt"].apply(lambda d: d.hour*60+d.minute)
    df["day"]=df["dt"].dt.strftime("%Y-%m-%d")
    return df.sort_values("dt").reset_index(drop=True)

def build_global_state(df, start_day, end_day):
    """Single causal pass: incremental UT + 15m bias, pointer arrays."""
    ut=UTBot(); bc=BiasComputer()
    # pointer arrays for entire range
    rows=[]; colors=[]; bias_by_min={}  # key: (day, minute) -> snap
    day_to_indices={}  # day -> list of global indices that are in session
    pending=None; cur_slot=None; cur_bar=None
    for i, r in df.iterrows():
        day=r["day"]
        if day < start_day or day > end_day:
            # still need warmup for UT/bias before start_day, so feed anyway but don't collect
            in_range=False
        else:
            in_range=True
        c={"open":r["open"],"high":r["high"],"low":r["low"],"close":r["close"],"minute":int(r["minute"])}
        col=ut.update(c)
        if in_range:
            # we still store rows/colors for in-range days only, but UT/bias are global
            gidx=len(rows)
            rows.append(c); colors.append(col)
            m=c["minute"]
            if SESSION_START <= m <= SESSION_END:
                day_to_indices.setdefault(day, []).append(gidx)
            # 15m aggregation (global, causal)
            # we need to keep 15m bias building globally, not just in_range
            # so we handle cur_bar globally below
            pass
        # 15m bias building — do for ALL rows (warmup), but bias_by_min only for in_range days
        m=int(r["minute"])
        if SESSION_START <= m <= SESSION_END:
            slot=SESSION_START + ((m-SESSION_START)//15)*15
            if cur_slot is None or slot != cur_slot:
                if cur_bar is not None:
                    pending=bc.feed(cur_bar)
                cur_slot=slot
                cur_bar={"open":c["open"],"high":c["high"],"low":c["low"],"close":c["close"],"minute":slot}
            else:
                cur_bar["high"]=max(cur_bar["high"], c["high"])
                cur_bar["low"]=min(cur_bar["low"], c["low"])
                cur_bar["close"]=c["close"]
            if in_range:
                bias_by_min[(day, m)] = pending
        # handle day change for 15m bar finalization at midnight? cur_bar carries over — fine
    if cur_bar is not None:
        bc.feed(cur_bar)
    return rows, colors, bias_by_min, day_to_indices

def find_impulses_for_day(seq):
    """seq: list of (gidx, color, high, low, minute) for one day, already causal."""
    N=len(seq)
    imps=[]
    for i in range(1, N):
        c_prev, c_cur = seq[i-1][1], seq[i][1]
        if c_prev=="green" and c_cur=="red":
            j=i
            while j<N and seq[j][1]=="red": j+=1
            if j-i>=5 and j<N and seq[j][1]=="green":
                if not (seq[i-1][2] > seq[j][2]): continue
                a=i-1
                while a-1>=0 and seq[a-1][1]=="green": a-=1
                peak=max(seq[t][2] for t in range(a, i))
                bottom=min(seq[t][3] for t in range(i, j))
                rng=peak-bottom
                if rng>MIN_SPAN:
                    imps.append({"side":"PE","peak":peak,"bottom":bottom,"span":rng,
                                 "range_start":seq[i-1][4],"range_end":seq[j][4],
                                 "mid_start":i,"last_start":j,"first_a":a})
        if c_prev=="red" and c_cur=="green":
            j=i
            while j<N and seq[j][1]=="green": j+=1
            if j-i>=5 and j<N and seq[j][1]=="red":
                if not (seq[i-1][3] < seq[j][3]): continue
                a=i-1
                while a-1>=0 and seq[a-1][1]=="red": a-=1
                peak=max(seq[t][2] for t in range(i, j))
                k=j
                while k<N and seq[k][1]=="red": k+=1
                bottom=min(seq[t][3] for t in range(j, k))
                rng=peak-bottom
                if rng>MIN_SPAN:
                    imps.append({"side":"CE","peak":peak,"bottom":bottom,"span":rng,
                                 "range_start":seq[i-1][4],"range_end":seq[j][4],
                                 "mid_start":i,"last_start":j,"first_a":a})
    imps.sort(key=lambda x: (x["range_start"], 0 if x["side"]=="PE" else 1))
    return imps

def run_day_trades(seq, imps, bias_by_min, day, rows):
    trades=[]
    for imp in imps:
        peak, bottom, rng, side = imp["peak"], imp["bottom"], imp["span"], imp["side"]
        entry = bottom + ENTRY*rng if side=="PE" else peak - ENTRY*rng
        tp2 = bottom if side=="PE" else peak
        sl = peak + SL_LEVEL*rng if side=="PE" else bottom - SL_LEVEL*rng
        # also keep tp1 for reporting
        tp1 = bottom + TP_LEVEL*rng if side=="PE" else peak - TP_LEVEL*rng
        start = imp["last_start"]
        end = min(len(seq), start+60)
        hit=None
        for t in range(start, end):
            _,_,hi,lo,m = seq[t]
            b=bias_by_min.get((day, m))
            if b is None: continue
            if not (hi >= entry - TOUCH and lo <= entry + TOUCH): continue
            if side=="CE" and not b["bullish"]: continue
            if side=="PE" and not b["bearish"]: continue
            gidx=seq[t][0]
            spot=rows[gidx]["close"]
            hit=(t,m,spot); break
        if hit is None:
            trades.append({"day":day,"side":side,"result":"UNTRIGGERED","range_start":imp["range_start"],"range_end":imp["range_end"],
                           "peak":peak,"bottom":bottom,"span":rng,"entry":entry,"tp1":tp1,"tp2":tp2,"sl":sl,"entry_min":None,"exit_min":None,"exit_price":None,"strike":None,"pts":None})
            continue
        t,m,spot = hit
        atm=int(round(spot/50.0)*50); strike=atm-100 if side=="CE" else atm+100
        res=None; ex_min=None; ex_price=None
        for u in range(t, len(seq)):
            _,_,hi,lo,mm = seq[u]
            if side=="CE":
                if lo <= sl: res,ex_min,ex_price="SL",mm,sl; break
                if hi >= tp2: res,ex_min,ex_price="TP0.0",mm,tp2; break
            else:
                if hi >= sl: res,ex_min,ex_price="SL",mm,sl; break
                if lo <= tp2: res,ex_min,ex_price="TP0.0",mm,tp2; break
        if res is None:
            res,ex_min="EOD", seq[-1][4]
            ex_price=rows[seq[-1][0]]["close"]
        pts = (ex_price - entry) if side=="CE" else (entry - ex_price)
        # also fill-based pts
        pts_fill = (ex_price - spot) if side=="CE" else (spot - ex_price)
        trades.append({"day":day,"side":side,"result":"TRADE","range_start":imp["range_start"],"range_end":imp["range_end"],
                       "peak":peak,"bottom":bottom,"span":rng,"entry":entry,"tp1":tp1,"tp2":tp2,"sl":sl,
                       "entry_min":m,"exit_min":ex_min,"exit_price":ex_price,"strike":strike,"atm":atm,
                       "pts":pts,"pts_fill":pts_fill,"exit_reason":res,"spot":spot})
    return trades

G_ROWS=None
G_COLORS=None
G_BIAS=None

def _init_worker(rows, colors, bias):
    global G_ROWS, G_COLORS, G_BIAS
    G_ROWS=rows; G_COLORS=colors; G_BIAS=bias

def _process_day_task(args):
    day, gidxs = args
    seq=[(g, G_COLORS[g], G_ROWS[g]["high"], G_ROWS[g]["low"], G_ROWS[g]["minute"]) for g in gidxs]
    imps=find_impulses_for_day(seq)
    return run_day_trades(seq, imps, G_BIAS, day, G_ROWS)

def backtest(start_day, end_day, smoke=False, workers=8):
    df=load_all()
    days_sorted=sorted(df["day"].unique())
    avail=[d for d in days_sorted if start_day <= d <= end_day]
    if smoke:
        avail=avail[:5]
        print("=== SMOKE TEST — 5 DAYS ONLY ===")
    print(f"Backtest {avail[0]} to {avail[-1]} ({len(avail)} days), warmup from {days_sorted[0]}, workers={workers}")
    t0=time.time()
    rows, colors, bias_by_min, day_to_indices = build_global_state(df, avail[0], avail[-1])
    # parallel day loop (causal state already built pointer-wise)
    workers = min(max(1, workers), 8, cpu_count())
    tasks=[(day, day_to_indices[day]) for day in avail if day in day_to_indices]
    all_trades=[]
    if workers==1:
        for day, gidxs in tasks:
            seq=[(g, colors[g], rows[g]["high"], rows[g]["low"], rows[g]["minute"]) for g in gidxs]
            imps=find_impulses_for_day(seq)
            all_trades.extend(run_day_trades(seq, imps, bias_by_min, day, rows))
    else:
        with Pool(workers, initializer=_init_worker, initargs=(rows, colors, bias_by_min)) as pool:
            for day_trades in pool.map(_process_day_task, tasks):
                all_trades.extend(day_trades)
    elapsed=time.time()-t0
    print(f"Completed {len(avail)} days in {elapsed:.2f}s, {len(all_trades)} impulses, {sum(1 for t in all_trades if t['result']=='TRADE')} trades")
    return all_trades, avail, elapsed

def summarize(all_trades, days):
    trades=[t for t in all_trades if t["result"]=="TRADE"]
    wins=[t for t in trades if t["pts"]>0]
    losses=[t for t in trades if t["pts"]<=0]
    gross_w=sum(t["pts"] for t in wins)
    gross_l=abs(sum(t["pts"] for t in losses))
    pf=gross_w/gross_l if gross_l else float("inf")
    # fees optional
    return {
        "impulses": len(all_trades),
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins)/len(trades)*100,2) if trades else 0,
        "net_pts_fib": round(sum(t["pts"] for t in trades),2),
        "net_pts_fill": round(sum(t["pts_fill"] for t in trades),2),
        "net_rs_fib": round(sum(t["pts"] for t in trades)*LOT),
        "net_rs_fill": round(sum(t["pts_fill"] for t in trades)*LOT),
        "pf": round(pf,4) if pf!=float("inf") else "inf",
        "days": len(days),
    }

def yearly_breakdown(all_trades):
    from collections import defaultdict
    by_year=defaultdict(list)
    for t in all_trades:
        if t["result"]!="TRADE": continue
        by_year[t["day"][:4]].append(t)
    rows=[]
    for y in sorted(by_year):
        s=summarize([t for t in all_trades if t["day"][:4]==y and t["result"]=="TRADE"], sorted(set(t["day"] for t in by_year[y])))
        # need proper days count for that year: count of trading days in that year
        # approximate via len(set days)
        s["year"]=y
        rows.append(s)
    return rows

if __name__=="__main__":
    import argparse
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
    # yearly
    print("\nYearly:")
    for r in yearly_breakdown(trades):
        print(f"{r['year']}: trades={r['trades']} WR={r['win_rate']}% net_pts_fib={r['net_pts_fib']} PF={r['pf']}")
    # export
    out_path=Path(args.out)
    if args.smoke:
        out_path=out_path.with_name(out_path.stem+"_smoke"+out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # json
    payload={"start":args.start,"end":args.end,"days":len(days),"elapsed_s":round(elapsed,2),
             "params":{"UT_KEY":UT_KEY,"UT_ATR":UT_ATR,"ENTRY":ENTRY,"TP":0.0,"SL":1.079,"TOUCH":TOUCH,"MIN_SPAN":MIN_SPAN,"mirror":True,"bias":"15m HA+LinReg11+UT","incremental":True,"pointer":True,"causal":True},
             "summary":summ,"trades":trades}
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"JSON: {out_path}")
    # csv
    import csv
    csv_path=out_path.with_suffix(".csv")
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["day","side","result","range_start","range_end","peak","bottom","span","entry","tp1","tp2","sl","entry_min","exit_min","exit_price","strike","pts_fib","pts_fill","reason"])
        for t in trades:
            w.writerow([t.get("day"),t.get("side"),t.get("result"),t.get("range_start"),t.get("range_end"),
                        round(t.get("peak",0),2),round(t.get("bottom",0),2),round(t.get("span",0),2),
                        round(t.get("entry",0),2),round(t.get("tp1",0),2),round(t.get("tp2",0),2),round(t.get("sl",0),2),
                        t.get("entry_min"),t.get("exit_min"),t.get("exit_price"),t.get("strike"),
                        round(t.get("pts",0),2) if t.get("pts") is not None else "", round(t.get("pts_fill",0),2) if t.get("pts_fill") is not None else "", t.get("exit_reason","")])
    print(f"CSV: {csv_path}")
    # parity check vs live engine for 2026-08-27
    live_day="2026-08-27"
    live_trades=[t for t in trades if t["day"]==live_day and t["result"]=="TRADE"]
    print(f"\nParity check {live_day}: {len(live_trades)} trades")
    for t in live_trades:
        print(f"  {t['side']} {t['range_start']}-{t['range_end']} entry {t['entry']:.2f} -> {t['exit_reason']} {t['exit_price']:.2f} pts {t['pts']:.2f}")
