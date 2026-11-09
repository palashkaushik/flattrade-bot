"""3 triggers x level set with/without 1m+2m EMA20/200."""
import bt
from run import nearest_map

DAYS = ["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]

for extra in (False, True):
    for mode in ("pattern", "ob", "union"):
        bt.EXTRA_EMAS = extra
        bt.TRIGGER_MODE = mode
        data = bt.Data()
        nm = nearest_map(data)
        days = [d for d in DAYS if d in data.days]
        all_days = bt.run_days(data, days, nm)
        s = bt.summarise(all_days)
        lab = "1m2m" if extra else "3m5m"
        print(f"{lab:5s} {mode:8s} tr {s['trades']:3d} WR {s['wr']:5.1f} "
              f"pts {s['pts']:8.2f} Rs {s['rs']:+7,} DD {s['dd']:6.2f} Cal {s['calmar']}")
