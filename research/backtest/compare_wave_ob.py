"""Sensitive (correction) vs full wave detection, with the order-block trigger."""
import bt
from run import nearest_map

DAYS = ["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]
bt.EXTRA_EMAS = False

for wm in ("full", "correction"):
    for trig in ("pattern", "ob"):
        bt.WAVE_MODE = wm
        bt.TRIGGER_MODE = trig
        data = bt.Data()
        nm = nearest_map(data)
        days = [d for d in DAYS if d in data.days]
        all_days = bt.run_days(data, days, nm)
        s = bt.summarise(all_days)
        print(f"{wm:10s} {trig:8s} tr {s['trades']:3d} WR {s['wr']:5.1f} "
              f"pts {s['pts']:8.2f} Rs {s['rs']:+7,} DD {s['dd']:6.2f} Cal {s['calmar']}")
        if wm == "correction" and trig == "ob":
            for d, kept in all_days:
                print(f"    {d}: {len(kept):2d} tr {sum(t['pts'] for t in kept):+8.2f}")
                for t in kept:
                    print(f"       {t['j']//60:02d}:{t['j']%60:02d} {t['sym']} "
                          f"{t['nm']}={t['entry']} risk {t['risk']:.2f} -> {t['why']} {t['pts']:+.2f}")
