"""Extend the OB sweep beyond the grid edge (with the day cap applied)."""
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


rows = []
for sl in (0.6, 0.8, 0.9, 1.0, 1.1):
    for tp in (4.0, 5.0, 6.0, 7.0, 8.0):
        bt.SL_ATR, bt.TP_MULT = sl, tp
        bt.MAX_SL, bt.MIN_RISK, bt.ENTRY_END, bt.DAY_CAP = 20, 3, 900, 40
        s = bt.summarise(book())
        rows.append((sl, tp, s))
        print(f"SL {sl:<4} TP {tp:<4} tr {s['trades']:3d} WR {s['wr']:5.1f} "
              f"pts {s['pts']:8.2f} DD {s['dd']:6.2f} Cal {s['calmar']}")

best = max(rows, key=lambda z: (z[2]["calmar"] or 0))
bt.SL_ATR, bt.TP_MULT = best[0], best[1]
print(f"\nbest {best[0]}/{best[1]}: {best[2]}")
for d, kept in book():
    print(f"  {d}: {len(kept):2d} tr {sum(t['pts'] for t in kept):+8.2f}")
    for t in kept:
        print(f"     {t['j']//60:02d}:{t['j']%60:02d} {t['sym']} {t['nm']}={t['entry']} "
              f"risk {t['risk']:.2f} -> {t['why']} {t['pts']:+.2f}")
