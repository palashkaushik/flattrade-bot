"""Slippage sensitivity: how much execution cost the six-year edge can absorb."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "sixyear_days.jsonl")
QTY, FEE = 65, 45
NAMES = ["cal SL0.9 TP8", "wr  SL0.9 TP4 nocap", "dep SL1.0 TP5", "SL1.2 TP3.6"]

done = {}
with open(CHECK, encoding="utf-8") as fh:
    for line in fh:
        o = json.loads(line)
        done[o["day"]] = o["books"]
days = sorted(done)

hdr = "config".ljust(22) + "  tr".rjust(5) + "      pts" + "  edge/tr"
for s in (0.0, 0.25, 0.5, 0.75, 1.0):
    hdr += f"   s={s:.2f}"
print(hdr)
for name in NAMES:
    pts = sum(p for d in days for _, p in done[d].get(name, []))
    tr = sum(len(done[d].get(name, [])) for d in days)
    row = f"{name:22s}{tr:6d}{pts:9.2f}{pts/tr:9.2f}"
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        rs = (pts - 2 * tr * s) * QTY - tr * FEE
        row += f"{rs:+10,.0f}"
    print(row)
