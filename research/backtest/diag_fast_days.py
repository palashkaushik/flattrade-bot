import pandas as pd

import diff_all as D
import sixyear_fast as F
from sixyear import files

NAME = {"cal SL0.9 TP8": "cal", "wr  SL0.9 TP4 nocap": "wr",
        "dep SL1.0 TP5": "dep", "SL1.2 TP3.6": "t36"}
days = F.all_days()[:40]
allf = files()[:40]
res = F.process_chunk((days, []))
fast = {d: r for d, r in res}
frames = []
bad = 0
for i, (day, path) in enumerate(allf):
    frames.append(D.load_csv(day, path))
    ref = D.day_book(pd.concat(frames[-13:], ignore_index=True), day)
    f = fast.get(day, {})
    fsum = {NAME[k]: (len(v), round(sum(p for _, p in v), 2)) for k, v in f.items()}
    if fsum != ref:
        bad += 1
        print(day, "REF", ref, "FAST", fsum)
print("days", len(allf), "mismatches", bad)
import collections
reft = collections.Counter(); fastt = collections.Counter()
for i, (day, path) in enumerate(allf):
    ref = D.day_book(pd.concat(frames[:i+1][-13:], ignore_index=True), day)
    f = fast.get(day, {})
    for k, (n2, p2) in ref.items():
        reft[k] += n2
    for k, v in f.items():
        fastt[NAME[k]] += len(v)
print("reference trades", dict(reft))
print("fast trades     ", dict(fastt))


