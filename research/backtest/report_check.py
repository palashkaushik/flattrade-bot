"""Generic six-year report: per-year stats, streaks, drawdown and slippage table.

Usage:  python report_check.py [checkpoint.jsonl]
Defaults to sixyear_days.jsonl. Reads every config name present in the file.
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
QTY, FEE = 65, 45


def load(path):
    done = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                o = json.loads(line)
                done[o["day"]] = o["books"]
            except Exception:
                pass
    return done


def stats(seq):
    eq = peak = dd = 0.0
    streak = worst = 0
    for _, p in seq:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
        streak = streak + 1 if p < 0 else 0
        worst = max(worst, streak)
    wins = sum(1 for _, p in seq if p > 0)
    return eq, dd, worst, wins


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "sixyear_days.jsonl")
    done = load(path)
    days = sorted(done)
    names = []
    for d in days:
        for k in done[d]:
            if k not in names:
                names.append(k)
    print(f"{path}\n{len(days)} sessions  {days[0]} .. {days[-1]}\n")

    for name in names:
        seq = [(d, p) for d in days for _, p in done[d].get(name, [])]
        if not seq:
            continue
        eq, dd, worst, wins = stats(seq)
        rs = eq * QTY - len(seq) * FEE
        be = rs / (2 * len(seq) * QTY) if len(seq) else 0.0
        print(f"=== {name}")
        print(f"  tr {len(seq):5d}  WR {100*wins/len(seq):5.1f}%  pts {eq:9.2f}  Rs {rs:+10,}  "
              f"DD {dd:8.2f}  streak {worst}  breakeven slip/side {be:.2f}")
        per_year = defaultdict(list)
        for d, p in seq:
            per_year[d[:4]].append((d, p))
        for y in sorted(per_year):
            s = per_year[y]
            e, ddd, w, wy = stats(s)
            print(f"    {y}  tr {len(s):4d}  WR {100*wy/len(s):5.1f}%  pts {e:9.2f}  "
                  f"DD {ddd:8.2f}  streak {w}")
        print()


if __name__ == "__main__":
    main()
