"""Previous (full) vs current (correction-only) wave detection, same data."""
import bt
from run import nearest_map

DAYS = ["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]

for mode in ("full", "correction"):
    bt.WAVE_MODE = mode
    data = bt.Data()
    nm = nearest_map(data)
    days = [d for d in DAYS if d in data.days]
    all_days = bt.run_days(data, days, nm)
    s = bt.summarise(all_days)
    print(f"\n=== wave mode: {mode} (previous={mode == 'full'}) ===")
    for d, kept in all_days:
        pts = sum(t["pts"] for t in kept)
        print(f"  {d}: {len(kept):2d} tr {pts:+8.2f}")
        for t in kept:
            print(f"     {t['j']//60:02d}:{t['j']%60:02d} {t['sym']} {t['nm']}="
                  f"{t['entry']} risk {t['risk']:.2f} -> {t['why']} {t['pts']:+.2f}")
    print(f"  TOTAL {s['trades']} trades | WR {s['wr']}% | {s['pts']:+.2f} pts | "
          f"Rs {s['rs']:+,} | DD {s['dd']:.2f} | Calmar {s['calmar']}")
