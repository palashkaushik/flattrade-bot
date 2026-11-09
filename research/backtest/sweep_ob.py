"""Combinatorial sweep for the order-block trigger (full waves).

Precomputes once per level set, then re-evaluates the cached contexts for each
parameter combination (SL/TP/stop floor/cap/cutoff are evaluate-time only).
"""
import bt
from run import nearest_map

DAYS = ["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]
rows = []

data = bt.Data()
nm = nearest_map(data)
DAYS = [d for d in DAYS if d in data.days]

CACHE = {}
for emas in (False, True):
    bt.EXTRA_EMAS = emas
    bt.WAVE_MODE = "full"
    bt.TRIGGER_MODE = "ob"
    CACHE[emas] = [(d, bt.precompute(data, d, nm[d])) for d in DAYS]


def book(emas):
    sigs = []
    for d, cs in CACHE[emas]:
        day_sigs = []
        for c in cs:
            day_sigs.extend(bt.evaluate([c], cap=float("inf")))
        kept, last = [], -1
        for s in sorted(day_sigs, key=lambda x: x["j"]):
            if s["j"] <= last:
                continue
            kept.append(s); last = s["exit"]
        if bt.DAY_CAP != float("inf"):
            out, cum = [], 0.0
            for s in kept:
                if cum <= -bt.DAY_CAP:
                    break
                out.append(s); cum += s["pts"]
            kept = out
        sigs.append((d, kept))
    return sigs


def run(tag, emas, sl, tp, mx, mr, end, cap):
    bt.SL_ATR, bt.TP_MULT = sl, tp
    bt.MAX_SL, bt.MIN_RISK = mx, mr
    bt.ENTRY_END, bt.DAY_CAP = end, cap
    s = bt.summarise(book(emas))
    rows.append((tag, s))
    return s


def show(tag, s):
    print(f"{tag:44s} tr {s['trades']:3d} WR {s['wr']:5.1f} pts {s['pts']:8.2f} "
          f"Rs {s['rs']:+7,} DD {s['dd']:6.2f} Cal {s['calmar']}")


print("== stage 1: SL x TP (3m/5m, max 20, min 3, end 900, cap 40) ==")
best = None
for sl in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75):
    for tp in (2.5, 3.0, 3.5, 4.0, 5.0):
        s = run(f"SL {sl} TP {tp}", False, sl, tp, 20, 3, 900, 40)
        if best is None or (s["calmar"] or 0) > (best[1]["calmar"] or 0):
            best = ((sl, tp), s)
(sl0, tp0), _ = best
show(f"best SL/TP {sl0}/{tp0}", best[1])

print("\n== stage 2: MAX_SL x MIN_RISK at best SL/TP ==")
best2 = None
for mx in (15, 20, 25, 30, 99):
    for mr in (2, 3, 4, 5):
        s = run(f"max {mx} min {mr}", False, sl0, tp0, mx, mr, 900, 40)
        if best2 is None or (s["calmar"] or 0) > (best2[1]["calmar"] or 0):
            best2 = ((mx, mr), s)
(mx0, mr0), _ = best2
show(f"best max/min {mx0}/{mr0}", best2[1])

print("\n== stage 3: cutoff x day cap ==")
best3 = None
for end in (840, 870, 900, 915):
    for cap in (30, 40, 60, float("inf")):
        s = run(f"end {end} cap {cap}", False, sl0, tp0, mx0, mr0, end, cap)
        if best3 is None or (s["calmar"] or 0) > (best3[1]["calmar"] or 0):
            best3 = ((end, cap), s)
(e0, c0), _ = best3
show(f"best end/cap {e0}/{c0}", best3[1])

print("\n== stage 4: level set ==")
for emas in (False, True):
    s = run(f"emas {emas}", emas, sl0, tp0, mx0, mr0, e0, c0)
    show(f"level set {'1m/2m' if emas else '3m/5m'}", s)

print("\n== TOP 10 by Calmar ==")
for tag, s in sorted(rows, key=lambda z: -(z[1]["calmar"] or 0))[:10]:
    show(tag, s)
print("\n== TOP 5 by points ==")
for tag, s in sorted(rows, key=lambda z: -z[1]["pts"])[:5]:
    show(tag, s)
