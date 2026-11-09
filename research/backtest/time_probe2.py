"""Precise cost split: precompute vs per-config evaluate, on normal and extreme days."""
import os
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sixyear_par as S  # noqa: E402
import bt  # noqa: E402


def build(day, prior):
    from datetime import datetime
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

    for day in ["2020-03-26", "2024-06-04", "2025-09-18", "2026-04-29"]:
        if day not in idx:
            print(day, "not in data")
            continue
        i = idx[day]
        prior = days[max(0, i - S.BUF_DAYS):i]
        data, nearest = build(day, prior)
        t = time.time()
        ctx = bt.precompute(data, day, nearest)
        pc = time.time() - t
        t = time.time()
        n = 0
        for c in ctx:
            n += len(bt.evaluate([c], cap=float("inf")))
        ev = time.time() - t
        print(f"{day}  precompute {pc:6.2f}s  evaluate {ev:7.4f}s  "
              f"signals {n:3d}  contexts {len(ctx)}", flush=True)


if __name__ == "__main__":
    main()
