"""Report the best configuration by Calmar, by win rate and by net points."""
import bt
from run import nearest_map

DAYS = ["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]
data = bt.Data()
nm = nearest_map(data)
DAYS = [d for d in DAYS if d in data.days]
bt.EXTRA_EMAS = False
bt.WAVE_MODE = "full"
bt.TRIGGER_MODE = "ob"
CACHE = [(d, bt.precompute(data, d, nm[d])) for d in DAYS]


def book():
    out = []
    for d, cs in CACHE:
        sigs = []
        for c in cs:
            sigs.extend(bt.evaluate([c], cap=float("inf")))
        kept, last = [], -1
        for s in sorted(sigs, key=lambda x: x["j"]):
            if s["j"] <= last:
                continue
            kept.append(s); last = s["exit"]
        if bt.DAY_CAP != float("inf"):
            capped, cum = [], 0.0
            for s in kept:
                if cum <= -bt.DAY_CAP:
                    break
                capped.append(s); cum += s["pts"]
            kept = capped
        out.append((d, kept))
    return out


rows = []          # (sl, tp, mx, mr, end, cap, stats)


def run(sl, tp, mx, mr, end, cap):
    bt.SL_ATR, bt.TP_MULT = sl, tp
    bt.MAX_SL, bt.MIN_RISK = mx, mr
    bt.ENTRY_END, bt.DAY_CAP = end, cap
    s = bt.summarise(book())
    rows.append((sl, tp, mx, mr, end, cap, s))
    return s


for sl in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5):
    for tp in (3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0):
        run(sl, tp, 20, 3, 900, 40)

best_cal = max(rows, key=lambda z: z[6]["calmar"] or 0)
best_wr = max(rows, key=lambda z: (z[6]["wr"], z[6]["pts"]))
best_pts = max(rows, key=lambda z: z[6]["pts"])

refine = []
for base in (best_cal, best_wr, best_pts):
    sl, tp = base[0], base[1]
    for mx in (20, 25, 99):
        for end in (900, 915):
            for cap in (40, float("inf")):
                refine.append(run(sl, tp, mx, 3, end, cap))

allrows = rows


def show(tag, z):
    sl, tp, mx, mr, end, cap, s = z
    print(f"{tag}: SL {sl} TP {tp} maxSL {mx} minRisk {mr} end {end} cap {cap}")
    print(f"   {s['trades']} trades | WR {s['wr']}% | {s['pts']:+.2f} pts | "
          f"Rs {s['rs']:+,} | DD {s['dd']:.2f} | Calmar {s['calmar']}")


print("\n== best by CALMAR ==")
show("CALMAR", max(allrows, key=lambda z: z[6]["calmar"] or 0))
print("\n== best by WIN RATE ==")
show("WINRATE", max(allrows, key=lambda z: (z[6]["wr"], z[6]["pts"])))
print("\n== best by NET POINTS ==")
show("POINTS", max(allrows, key=lambda z: z[6]["pts"]))

print("\n== deployed reference ==")
show("deployed", (1.0, 5.0, 20, 3, 900, 40, [z for z in rows
                                              if z[0] == 1.0 and z[1] == 5.0][0][6]))
