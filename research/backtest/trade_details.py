"""Trade geometry for the current and previous deployed rules.

current  = ob trigger, SL 1.0xATR, TP 5.0xrange, max stop 20, min 3, cap -40
previous = pattern trigger (engulf/harami/twin), same risk parameters

Modes:
  python trade_details.py short   -> 2026-11-04 .. 2026-11-09 from data/ CSVs
  python trade_details.py         -> six-year cache, day-level parallel
"""
import json
import os
import statistics
import sys
import threading
from datetime import datetime
from multiprocessing import Pool

import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sixyear_par as S  # noqa: E402
import bt  # noqa: E402

CHECK = os.path.join(HERE, "trade_details.jsonl")
WORKERS = int(os.environ.get("SY_WORKERS", "10"))
CAP = 40.0

CONFIGS = [
    ("current ob 1.0/5/20", "ob", 1.0, 5.0, 20.0),
    ("previous pattern 1.0/5/20", "pattern", 1.0, 5.0, 20.0),
]


def take(cfg, ctx):
    """Deployed-style evaluation: one signal per wave, global dedupe, day cap."""
    _name, trig, sl, tp, mx = cfg
    bt.TRIGGER_MODE = trig
    bt.SL_ATR, bt.TP_MULT, bt.MAX_SL = sl, tp, mx
    bt.MIN_RISK, bt.ENTRY_END = 3.0, 900
    sigs = []
    for c in ctx:
        sigs.extend(bt.evaluate([c], cap=float("inf")))
    kept, last = [], -1
    for s in sorted(sigs, key=lambda x: x["j"]):
        if s["j"] <= last:
            continue
        kept.append(s)
        last = s["exit"]
    out, cum = [], 0.0
    for s in kept:
        if cum <= -CAP:
            break
        out.append(s)
        cum += s["pts"]
    return out


def rows_of(trades):
    return [(t["entry"] - t["sl"], t["tp"] - t["entry"]) for t in trades]


def summarise(name, trades):
    rows = rows_of(trades)
    if not rows:
        print(f"{name}: no trades")
        return
    sl = [r[0] for r in rows]
    tp = [r[1] for r in rows]
    rr = [r[1] / r[0] for r in rows if r[0] > 0]
    print(f"\n{name}: {len(rows)} trades")
    print(f"  avg SL {statistics.mean(sl):6.2f}  median {statistics.median(sl):6.2f}  "
          f"min {min(sl):.2f}  max {max(sl):.2f}")
    print(f"  avg TP {statistics.mean(tp):6.2f}  median {statistics.median(tp):6.2f}  "
          f"min {min(tp):.2f}  max {max(tp):.2f}")
    print(f"  avg RR {statistics.mean(rr):6.2f}  median {statistics.median(rr):6.2f}  "
          f"aggregate {statistics.mean(tp)/statistics.mean(sl):.2f}")


def nearest_map(data):
    import re
    exp = re.compile(r"^NIFTY(\d{2}[A-Z]{3}\d{2})")
    exps = set()
    for s in data.df.symbol.unique():
        m = exp.match(s)
        if m:
            try:
                exps.add(datetime.strptime(m.group(1), "%d%b%y").date())
            except ValueError:
                pass
    out = {}
    for d in data.days:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        later = sorted(e for e in exps if e >= dt)
        e = later[0] if later else (sorted(exps)[-1] if exps else None)
        out[d] = f"NIFTY{e:%d%b%y}".upper() if e else ""
    return out


def short():
    data = bt.Data()
    days = [d for d in data.days if "2026-11-04" <= d <= "2026-11-09"]
    nm = nearest_map(data)
    print(f"short window: {days}")
    for cfg in CONFIGS:
        allt = []
        per_day = []
        for d in days:
            t = take(cfg, bt.precompute(data, d, nm.get(d, "")))
            allt.extend(t)
            per_day.append((d, t))
        print(f"\n--- {cfg[0]} ---")
        for d, t in per_day:
            print(f"  {d}: {len(t):2d} tr {sum(x['pts'] for x in t):+8.2f}  " +
                  " ".join(f"[SL {x['entry']-x['sl']:.1f} TP {x['tp']-x['entry']:.1f}]" for x in t))
        summarise(cfg[0], allt)


def worker(task):
    day, prior = task
    d0 = prior[0] if prior else day
    tbl = pq.read_table(S.CACHE, filters=[("date", ">=", d0), ("date", "<=", day)])
    df = tbl.to_pandas(self_destruct=True)
    if df.empty:
        return day, {}
    df["date"] = pd.Categorical(df["date"], categories=sorted(df["date"].unique()), ordered=True)
    df["symbol"] = df["symbol"].astype("category")
    if day not in set(df.date.unique()):
        return day, {}
    data = S.FrameData(df, S.SPOT)
    ddf = df[df.date == day]
    exps = {S.expiry(s) for s in ddf.symbol.unique()}
    exps.discard(None)
    dd = datetime.strptime(day, "%Y-%m-%d").date()
    later = sorted(e for e in exps if e >= dd)
    nearest = f"NIFTY{later[0]:%d%b%y}".upper() if later else ""
    ctx = bt.precompute(data, day, nearest) if nearest else []
    res = {}
    for cfg in CONFIGS:
        res[cfg[0]] = [{"j": s["j"], "entry": s["entry"], "sl": s["sl"], "tp": s["tp"],
                        "pts": s["pts"]} for s in take(cfg, ctx)]
    return day, res


def load():
    done = {}
    if os.path.exists(CHECK):
        with open(CHECK, encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                done[o["day"]] = o["books"]
    return done


def long():
    days = S.all_days()
    done = load()
    todo = [(d, days[max(0, i - S.BUF_DAYS):i]) for i, d in enumerate(days) if d not in done]
    print(f"{len(done)} days done, {len(todo)} to run", flush=True)
    job = S.cap_cpu(S.CPU_CAP_PCT)
    stop_evt = threading.Event()
    wd = None
    if job is None:
        wd = threading.Thread(target=S._watchdog, args=(stop_evt,), daemon=True)
        wd.start()
    t0 = datetime.now()
    try:
        with Pool(WORKERS) as pool:
            for day, books in pool.imap_unordered(worker, todo, chunksize=1):
                done[day] = books
                with open(CHECK, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"day": day, "books": books}) + "\n")
    finally:
        stop_evt.set()
        if wd is not None:
            wd.join(timeout=3)
    print(f"done in {(datetime.now()-t0).seconds}s", flush=True)
    print(f"\n=== six-year geometry, {len(done)} sessions ===")
    for cfg in CONFIGS:
        allt = [t for d in sorted(done) for t in done[d].get(cfg[0], [])]
        summarise(cfg[0], allt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "short":
        short()
    else:
        long()
