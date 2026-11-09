"""Measure precompute vs per-config evaluate cost on representative days."""
import os
import sys
import time
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sixyear_par as S  # noqa: E402
import bt  # noqa: E402

DAYS = ["2020-03-26", "2023-06-15", "2025-09-18"]


def build(day, prior):
    d0 = prior[0] if prior else day
    tbl = pq.read_table(S.CACHE, filters=[("date", ">=", d0), ("date", "<=", day)])
    df = tbl.to_pandas(self_destruct=True)
    df["date"] = pd.Categorical(df["date"], categories=sorted(df["date"].unique()), ordered=True)
    df["symbol"] = df["symbol"].astype("category")
    data = S.FrameData(df, S.SPOT)
    ddf = df[df.date == day]
    exps = {S.expiry(s) for s in ddf.symbol.unique()}
    exps.discard(None)
    dd = datetime.strptime(day, "%Y-%m-%d").date()
    later = sorted(e for e in exps if e >= dd)
    nearest = f"NIFTY{later[0]:%d%b%y}".upper() if later else ""
    return data, nearest


def main():
    days = S.all_days()
    idx = {d: i for i, d in enumerate(days)}
    bt.TRIGGER_MODE = "ob"
    bt.SL_ATR, bt.TP_MULT, bt.MAX_SL = 1.0, 5.0, 20
    bt.MIN_RISK, bt.ENTRY_END, bt.DAY_CAP = 3, 900, 40

    for day in DAYS:
        if day not in idx:
            continue
        i = idx[day]
        prior = days[max(0, i - S.BUF_DAYS):i]
        t0 = time.time()
        data, nearest = build(day, prior)
        t1 = time.time()
        ctx = bt.precompute(data, day, nearest)
        t2 = time.time()
        for _ in range(5):
            for c in ctx:
                bt.evaluate([c], cap=float("inf"))
        t3 = time.time()
        print(f"{day}  load {t1-t0:5.2f}s  precompute {t2-t1:6.2f}s  "
              f"evaluate {((t3-t2)/5):6.2f}s/config  contexts {len(ctx)}", flush=True)


if __name__ == "__main__":
    main()
