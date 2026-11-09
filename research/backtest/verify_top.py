"""Exact re-verification of selected sweep configs (full trade lists, per-year).

Usage: python verify_top.py sweep_top.json
sweep_top.json: [{"name": "...", "trigger": "ob", "sl": 1.0, "tp": 5.0,
                  "mx": 20.0, "cap": 40.0}, ...]
Writes verify_top_days.jsonl (resume-safe) and prints exact metrics.
"""
import json
import os
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

CHECK = os.path.join(HERE, "verify_top_days.jsonl")
WORKERS = int(os.environ.get("SY_WORKERS", "10"))
QTY, FEE = 65, 45
YEARS = 6.33

SPEC = json.load(open(os.path.join(HERE, sys.argv[1]), encoding="utf-8"))


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
    bt.MIN_RISK, bt.ENTRY_END = 3.0, 900
    res = {}
    for cfg in SPEC:
        bt.TRIGGER_MODE = cfg["trigger"]
        bt.SL_ATR, bt.TP_MULT, bt.MAX_SL = cfg["sl"], cfg["tp"], cfg["mx"]
        sigs = []
        for c in ctx:
            sigs.extend(bt.evaluate([c], cap=float("inf")))
        kept, last = [], -1
        for s in sorted(sigs, key=lambda x: x["j"]):
            if s["j"] <= last:
                continue
            kept.append(s)
            last = s["exit"]
        cap = float(cfg["cap"])
        out, cum = [], 0.0
        for s in kept:
            if cum <= -cap:
                break
            out.append(s)
            cum += s["pts"]
        res[cfg["name"]] = [(s["j"], s["pts"], s["exit"]) for s in out]
    return day, res


def load():
    done = {}
    if os.path.exists(CHECK):
        with open(CHECK, encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                done[o["day"]] = o["books"]
    return done


def main():
    days = S.all_days()
    done = load()
    todo = [(d, days[max(0, i - S.BUF_DAYS):i])
            for i, d in enumerate(days) if d not in done]
    print(f"{len(done)} days done, {len(todo)} to run, {len(SPEC)} configs", flush=True)
    job = S.cap_cpu(S.CPU_CAP_PCT)
    stop_evt = threading.Event()
    wd = None
    if job is None:
        wd = threading.Thread(target=S._watchdog, args=(stop_evt,), daemon=True)
        wd.start()
    t0 = datetime.now()
    try:
        with Pool(WORKERS) as pool:
            for day, res in pool.imap_unordered(worker, todo, chunksize=1):
                done[day] = res
                with open(CHECK, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"day": day, "books": res}) + "\n")
    finally:
        stop_evt.set()
        if wd is not None:
            wd.join(timeout=3)
    print(f"done in {(datetime.now()-t0).seconds}s", flush=True)

    names = [c["name"] for c in SPEC]
    sd = sorted(done)
    print(f"\n=== exact verification over {len(sd)} sessions ===")
    for name in names:
        seq = [(d, p) for d in sd for (_, p, _ee) in done[d].get(name, [])]
        eq = peak = dd = 0.0
        streak = worst = 0
        wins = 0
        per_year = {}
        for d, p in seq:
            eq += p
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
            if p > 0:
                wins += 1
                streak = 0
            else:
                streak += 1
                worst = max(worst, streak)
            y = per_year.setdefault(d[:4], [0.0, 0, 0])
            y[0] += p
            y[1] += 1
            y[2] += 1 if p > 0 else 0
        rs = eq * QTY - len(seq) * FEE
        cal = (eq / YEARS) / dd if dd else 0.0
        be = rs / (2 * len(seq) * QTY) if seq else 0.0
        print(f"\n{name}: tr {len(seq)} WR {100*wins/max(len(seq),1):.1f}% pts {eq:.2f} "
              f"Rs {rs:+,.0f} DD {dd:.2f} Cal {cal:.2f} streak {worst} BE {be:.2f}")
        for y in sorted(per_year):
            e, n, w = per_year[y]
            print(f"   {y} tr {n:4d} WR {100*w/max(n,1):5.1f}% pts {e:9.2f}")


if __name__ == "__main__":
    main()
