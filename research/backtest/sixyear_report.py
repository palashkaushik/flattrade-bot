"""Six-year verdict: per-year stats, DD, streaks and outlier concentration."""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "sixyear_days.jsonl")

CONFIGS = [
    ("cal SL0.9 TP8", 0.9, 8.0),
    ("wr  SL0.9 TP4 nocap", 0.9, 4.0),
    ("dep SL1.0 TP5", 1.0, 5.0),
    ("SL1.2 TP3.6", 1.2, 3.6),
]
QTY, FEE = 65, 45


def load():
    done = {}
    with open(CHECK, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            done[o["day"]] = o["books"]
    return done


def curve(seq):
    """seq = [(day, pts)] -> trades list with equity and max DD."""
    eq = 0.0
    peak = 0.0
    dd = 0.0
    streak = worst = 0
    for _, p in seq:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
        if p < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return eq, dd, worst


def main():
    done = load()
    days = sorted(done)
    print(f"{len(days)} sessions  {days[0]} .. {days[-1]}\n")

    for name, _, _ in CONFIGS:
        seq = []
        per_year = defaultdict(list)
        for d in days:
            for j, p in done[d].get(name, []):
                seq.append((d, p))
                per_year[d[:4]].append((d, p))
        eq, dd, worst = curve(seq)
        wins = sum(1 for _, p in seq if p > 0)
        top10 = sum(sorted((p for _, p in seq), reverse=True)[:10])
        rs = eq * QTY - len(seq) * FEE
        print(f"=== {name}")
        print(f"  trades {len(seq):5d}  WR {100*wins/len(seq):5.1f}%  pts {eq:9.2f}  "
              f"Rs {rs:+10,}  DD {dd:8.2f}  worst losing streak {worst}")
        print(f"  top-10 trades = {top10:8.2f} pts ({100*top10/eq if eq else 0:5.1f}% of total)")
        for y in sorted(per_year):
            s = per_year[y]
            e, ddd, w = curve(s)
            wy = sum(1 for _, p in s if p > 0)
            print(f"    {y}  tr {len(s):4d}  WR {100*wy/len(s):5.1f}%  pts {e:9.2f}  "
                  f"DD {ddd:8.2f}  streak {w}")
        print()


if __name__ == "__main__":
    main()
