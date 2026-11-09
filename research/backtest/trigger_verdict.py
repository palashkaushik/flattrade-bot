"""Head-to-head: OB vs the pre-OB pattern bundle and its parts, risk-adjusted."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
QTY, FEE = 65, 45
YEARS = 6.33


def load(name):
    out = {}
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            out[o["day"]] = o["books"]
    return out


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


main = load("sixyear_days.jsonl")
trig = load("sixyear_days_trig.jsonl")

books = [
    ("OB (deployed)", main, "dep SL1.0 TP5"),
    ("pattern (live pre-OB)", trig, "pattern (live pre-OB)"),
    ("harami only", trig, "harami only"),
    ("engulf only", trig, "engulf only"),
    ("twin only", trig, "twin only"),
]

print(f"{'book':24s}{'tr':>6s}{'WR':>7s}{'pts':>9s}{'Rs':>11s}{'DD':>9s}{'Calmar':>8s}{'BE slip':>9s}")
for name, src, key in books:
    seq = [(d, p) for d in sorted(src) for _, p in src[d].get(key, [])]
    eq, dd, worst, wins = stats(seq)
    rs = eq * QTY - len(seq) * FEE
    cal = (eq / YEARS) / dd if dd else 0.0
    be = rs / (2 * len(seq) * QTY) if len(seq) else 0.0
    print(f"{name:24s}{len(seq):6d}{100*wins/len(seq):6.1f}%{eq:9.1f}{rs:+11,.0f}{dd:9.1f}{cal:8.2f}{be:9.2f}")
