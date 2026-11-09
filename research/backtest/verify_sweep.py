"""Parity check: sweep control rows must match the finished six-year checkpoints."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from sweep_grid import KEYS  # noqa: E402


def load(name):
    out = {}
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            out[o["day"]] = o["books"]
    return out


main = load("sixyear_days.jsonl")
obonly = load("sixyear_days_obonly.jsonl")

rows = [
    ("ob|1.0|5.0|20.0|40", main, "dep SL1.0 TP5"),
    ("obonly|1.0|5.0|20.0|40", obonly, "ob only (no SR)"),
]

ok = True
for key, ref, refkey in rows:
    if key not in KEYS:
        print(f"MISSING {key} in sweep grid")
        ok = False
        continue
    i = KEYS.index(key)
    for day in sorted(os.listdir(os.path.join(HERE, "sweep_days"))):
        if not day.endswith(".npz"):
            continue
        d = day[:-4]
        if d not in ref or refkey not in ref[d]:
            continue
        z = np.load(os.path.join(HERE, "sweep_days", day))
        cnt = int(z["cnt"][i]); net = float(z["net"][i]); win = int(z["win"][i])
        pts = [p for _, p in ref[d][refkey]]
        same = (cnt == len(pts) and abs(net - sum(pts)) < 0.02
                and win == sum(1 for p in pts if p > 0))
        ok = ok and same
        print(f"{key:28s} {d}  sweep tr {cnt} pts {net:8.2f} win {win} | "
              f"ref tr {len(pts)} pts {sum(pts):8.2f} win {sum(1 for p in pts if p>0)}  match={same}")
print("PARITY OK" if ok else "PARITY FAILED")
