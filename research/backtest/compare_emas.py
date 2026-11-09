"""Level-set comparison with and without 1m/2m EMA20+EMA200."""
import bt
from run import nearest_map

DAYS = ["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]

for extra in (False, True):
    bt.EXTRA_EMAS = extra
    data = bt.Data()
    nm = nearest_map(data)
    days = [d for d in DAYS if d in data.days]
    all_days = bt.run_days(data, days, nm)
    s = bt.summarise(all_days)
    label = "+1m/2m EMA20+200" if extra else "current levels (live)"
    print(f"\n=== {label} ===")
    for d, kept in all_days:
        pts = sum(t["pts"] for t in kept)
        print(f"  {d}: {len(kept):2d} tr {pts:+8.2f}")
        for t in kept:
            print(f"     {t['j']//60:02d}:{t['j']%60:02d} {t['sym']} {t['nm']}={t['entry']} "
                  f"risk {t['risk']:.2f} -> {t['why']} {t['pts']:+.2f}")
    print(f"  TOTAL {s['trades']} trades | WR {s['wr']}% | {s['pts']:+.2f} pts | "
          f"Rs {s['rs']:+,} | DD {s['dd']:.2f} | Calmar {s['calmar']}")
