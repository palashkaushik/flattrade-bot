"""Comprehensive combinatorial sweep of the OB / pattern trigger family.

Grid: 5 triggers x 7 SL_ATR x 8 TP_MULT x 6 MAX_SL = 1,680 evaluations per day.
Day-cap variants (30/40/60) are derived free from each evaluation, giving
5,040 published configs. Precompute runs once per day; evaluate is ~1 ms/config.

Per day we store, per config: net points, trade count, wins, and the intraday
path extremes (min/max relative to day start) so exact WR/points and a tight
drawdown bound can be reconstructed. Top configs are re-run exactly afterwards.

Checkpoint: one .npz per day under sweep_days/ (resume by skipping files).
"""
import os
import sys
import threading
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sixyear_par as S  # noqa: E402
import bt  # noqa: E402

OUT = os.path.join(HERE, "sweep_days")
WORKERS = int(os.environ.get("SY_WORKERS", "10"))

TRIGGERS = ["ob", "obonly", "pattern", "pat2", "union"]
SLS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
TPS = [2.5, 3.0, 3.6, 4.0, 5.0, 6.0, 8.0, 10.0]
MAXSLS = [10.0, 12.0, 16.0, 20.0, 24.0, 30.0]
CAPS = [30.0, 40.0, 60.0]

BASE = [(t, sl, tp, mx) for t in TRIGGERS for sl in SLS for tp in TPS for mx in MAXSLS]
KEYS = [f"{t}|{sl}|{tp}|{mx}|{int(cap)}" for (t, sl, tp, mx) in BASE for cap in CAPS]


def worker(task):
    day, prior = task
    d0 = prior[0] if prior else day
    tbl = pq.read_table(S.CACHE, filters=[("date", ">=", d0), ("date", "<=", day)])
    df = tbl.to_pandas(self_destruct=True)
    n = len(KEYS)
    empty = (np.zeros(n, np.float32), np.zeros(n, np.int32), np.zeros(n, np.int32),
             np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32))
    if df.empty:
        return day, empty
    df["date"] = pd.Categorical(df["date"], categories=sorted(df["date"].unique()), ordered=True)
    df["symbol"] = df["symbol"].astype("category")
    if day not in set(df.date.unique()):
        return day, empty
    data = S.FrameData(df, S.SPOT)
    ddf = df[df.date == day]
    exps = {S.expiry(s) for s in ddf.symbol.unique()}
    exps.discard(None)
    dd = datetime.strptime(day, "%Y-%m-%d").date()
    later = sorted(e for e in exps if e >= dd)
    nearest = f"NIFTY{later[0]:%d%b%y}".upper() if later else ""
    ctx = bt.precompute(data, day, nearest) if nearest else []

    net = np.zeros(n, np.float32)
    cnt = np.zeros(n, np.int32)
    win = np.zeros(n, np.int32)
    lo = np.zeros(n, np.float32)
    hi = np.zeros(n, np.float32)
    ddi = np.zeros(n, np.float32)
    bt.MIN_RISK, bt.ENTRY_END, bt.DAY_CAP = 3.0, 900, 40.0
    k = 0
    for (trig, sl, tp, mx) in BASE:
        bt.TRIGGER_MODE = trig
        bt.SL_ATR, bt.TP_MULT, bt.MAX_SL = sl, tp, mx
        sigs = []
        for c in ctx:
            sigs.extend(bt.evaluate([c], cap=float("inf")))
        kept, last = [], -1
        for s in sorted(sigs, key=lambda x: x["j"]):
            if s["j"] <= last:
                continue
            kept.append(s)
            last = s["exit"]
        for cap in CAPS:
            cum = 0.0
            pts = []
            for s in kept:
                if cum <= -cap:
                    break
                pts.append(s["pts"])
                cum += s["pts"]
            if pts:
                arr = np.asarray(pts, np.float32)
                eq = np.cumsum(arr)
                path = np.concatenate((np.zeros(1, np.float32), eq))
                peak = np.maximum.accumulate(path)
                net[k] = float(eq[-1])
                cnt[k] = len(arr)
                win[k] = int((arr > 0).sum())
                lo[k] = float(eq.min())
                hi[k] = float(eq.max())
                ddi[k] = float(np.max(peak - path))
            k += 1
    return day, (net, cnt, win, lo, hi, ddi)


def main(limit=None):
    os.makedirs(OUT, exist_ok=True)
    days = S.all_days()
    if limit:
        days = days[:limit]
    todo = [(d, days[max(0, i - S.BUF_DAYS):i])
            for i, d in enumerate(days) if not os.path.exists(os.path.join(OUT, f"{d}.npz"))]
    print(f"{len(KEYS)} configs; {len(todo)} of {len(days)} days to run", flush=True)
    if not todo:
        return
    job = S.cap_cpu(S.CPU_CAP_PCT)
    print(f"cpu cap: {S.CPU_CAP_PCT:.0f}% via job object" if job
          else "cpu cap: job object failed, using watchdog", flush=True)
    stop_evt = threading.Event()
    wd = None
    if job is None:
        wd = threading.Thread(target=S._watchdog, args=(stop_evt,), daemon=True)
        wd.start()
    t0 = datetime.now()
    done = 0
    try:
        with Pool(WORKERS) as pool:
            for day, (net, cnt, win, lo, hi, ddi) in pool.imap_unordered(worker, todo, chunksize=1):
                np.savez_compressed(os.path.join(OUT, f"{day}.npz"),
                                    net=net, cnt=cnt, win=win, lo=lo, hi=hi, ddi=ddi)
                done += 1
                if done % 100 == 0:
                    el = (datetime.now() - t0).seconds
                    print(f"  {done}/{len(todo)} days  {el}s  "
                          f"eta {int(el / done * (len(todo) - done))}s", flush=True)
    finally:
        stop_evt.set()
        if wd is not None:
            wd.join(timeout=3)
    print(f"done in {(datetime.now()-t0).seconds}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
