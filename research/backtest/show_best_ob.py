"""Trade list for the Calmar/points winner: SL 0.9, TP 8.0, maxSL 20, cap 40."""
import bt
from run import nearest_map

DAYS = ["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]
bt.WAVE_MODE = "full"
bt.TRIGGER_MODE = "ob"
bt.EXTRA_EMAS = False
bt.SL_ATR, bt.TP_MULT = 0.9, 8.0
bt.MAX_SL, bt.MIN_RISK, bt.ENTRY_END, bt.DAY_CAP = 20, 3, 900, 40

data = bt.Data()
nm = nearest_map(data)
DAYS = [d for d in DAYS if d in data.days]
all_days = bt.run_days(data, DAYS, nm)

risks, tps = [], []
print(f"{'day':10s} {'time':5s} {'contract':12s} {'level':10s} {'entry':>8s} {'SL':>8s} "
      f"{'TP':>8s} {'risk':>6s} {'tp':>7s}  result")
for d, kept in all_days:
    for t in kept:
        risks.append(t["risk"]); tps.append(t["tp_pts"])
        print(f"{d:10s} {t['j']//60:02d}:{t['j']%60:02d} {t['sym'][9:]:12s} "
              f"{t['nm']:10s} {t['entry']:8.2f} {t['sl']:8.2f} {t['tp']:8.2f} "
              f"{t['risk']:6.2f} {t['tp_pts']:7.2f}  {t['why']} {t['pts']:+.2f}")

n = len(risks)
print(f"\n{n} trades | avg SL {sum(risks)/n:.2f} | avg TP {sum(tps)/n:.2f}")
print(f"SL min/med/max {min(risks):.2f}/{sorted(risks)[n//2]:.2f}/{max(risks):.2f}")
print(f"TP min/med/max {min(tps):.2f}/{sorted(tps)[n//2]:.2f}/{max(tps):.2f}")
print(f"total {sum(t['pts'] for _, k in all_days for t in k):+.2f} pts")
