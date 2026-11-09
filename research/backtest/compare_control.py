"""Compare the smoke-run OB control against the finished six-year checkpoint."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    path = os.path.join(HERE, name)
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            out[o["day"]] = o["books"]
    return out


main = load("sixyear_days.jsonl")
ctrl = load("sixyear_days_trig_ctrl.jsonl")
ok = True
for day in sorted(ctrl):
    a = [tuple(x) for x in main[day]["dep SL1.0 TP5"]]
    b = [tuple(x) for x in ctrl[day]["ob control"]]
    same = a == b
    ok = ok and same
    print(f"{day}  main {len(a):2d} trades  control {len(b):2d}  match={same}")
print("CONGRUENT" if ok else "MISMATCH")
