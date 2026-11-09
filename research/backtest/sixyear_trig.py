"""Six-year trigger comparison: live pattern bundle vs engulf / harami / twin vs OB.

Same day-level parallelism, checkpointing and CPU watchdog as sixyear_par.py.
Each config carries its own TRIGGER_MODE, so one day's precompute serves all.
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

CONTROL = os.environ.get("SIXTRIG_CONTROL", "0") == "1"
CHECK = os.path.join(HERE, "sixyear_days_trig_ctrl.jsonl" if CONTROL
                     else "sixyear_days_trig.jsonl")
WORKERS = int(os.environ.get("SY_WORKERS", "10"))

# (name, trigger mode, SL_ATR, TP_MULT, MAX_SL)
CONFIGS = [
    ("pattern (live pre-OB)", "pattern", 1.0, 5.0, 20),
    ("engulf only", "engulf", 1.0, 5.0, 20),
    ("harami only", "harami", 1.0, 5.0, 20),
    ("twin only", "twin", 1.0, 5.0, 20),
]
if CONTROL:
    CONFIGS.append(("ob control", "ob", 1.0, 5.0, 20))


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
    for name, mode, sl, tp, mx in CONFIGS:
        bt.TRIGGER_MODE = mode
        bt.SL_ATR, bt.TP_MULT, bt.MAX_SL = sl, tp, mx
        bt.MIN_RISK, bt.ENTRY_END, bt.DAY_CAP = 3, 900, 40
        sigs = []
        for c in ctx:
            sigs.extend(bt.evaluate([c], cap=float("inf")))
        kept, last = [], -1
        for s in sorted(sigs, key=lambda x: x["j"]):
            if s["j"] <= last:
                continue
            kept.append(s)
            last = s["exit"]
        capped, cum = [], 0.0
        for s in kept:
            if cum <= -40:
                break
            capped.append(s)
            cum += s["pts"]
        res[name] = [(s["j"], s["pts"]) for s in capped]
    return day, res


def load_check():
    done = {}
    if os.path.exists(CHECK):
        with open(CHECK, encoding="utf-8") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                    done[o["day"]] = {k: [tuple(x) for x in v] for k, v in o["books"].items()}
                except Exception:
                    pass
    return done


def main(limit=None):
    days = S.all_days()
    if limit:
        days = days[:limit]
    done = load_check()
    tasks = [(d, days[max(0, i - S.BUF_DAYS):i]) for i, d in enumerate(days) if d not in done]
    print(f"checkpoint: {len(done)} days done, {len(tasks)} to run", flush=True)
    print(f"{len(CONFIGS)} configs x {len(tasks)} days, {WORKERS} workers", flush=True)
    job = S.cap_cpu(S.CPU_CAP_PCT)
    print(f"cpu cap: {S.CPU_CAP_PCT:.0f}% via job object" if job
          else "cpu cap: job object failed, using watchdog", flush=True)
    stop_evt = threading.Event()
    wd = None
    if job is None:
        wd = threading.Thread(target=S._watchdog, args=(stop_evt,), daemon=True)
        wd.start()
    t0 = datetime.now()
    try:
        with Pool(WORKERS) as pool:
            for day, res in pool.imap_unordered(worker, tasks, chunksize=1):
                done[day] = res
                with open(CHECK, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"day": day, "books": {k: [[j, p] for j, p in v]
                                                               for k, v in res.items()}}) + "\n")
    finally:
        stop_evt.set()
        if wd is not None:
            wd.join(timeout=3)
    print(f"done in {(datetime.now()-t0).seconds}s", flush=True)
    print(f"\n=== {len(done)} sessions (trigger comparison, SL1.0 TP5 cap20) ===")
    for name, _, _, _, _ in CONFIGS:
        seq = [(d, p) for d in sorted(done) for _, p in done[d].get(name, [])]
        eq, dd, worst = 0.0, 0.0, 0
        peak = streak = 0.0
        for _, p in seq:
            eq += p
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
            streak = streak + 1 if p < 0 else 0
            worst = max(worst, int(streak))
        wins = sum(1 for _, p in seq if p > 0)
        rs = eq * 65 - len(seq) * 45
        print(f"{name:22s} tr {len(seq):5d} WR {100*wins/max(len(seq),1):5.1f}% "
              f"pts {eq:9.2f} Rs {rs:+10,} DD {dd:8.2f} streak {worst}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
