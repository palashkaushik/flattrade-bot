"""Deployed OB config: per-trade SL/TP distances and averages."""
import bt
from run import nearest_map

DAYS = ["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]
bt.WAVE_MODE = "full"
bt.TRIGGER_MODE = "ob"
bt.EXTRA_EMAS = False
bt.SL_ATR, bt.TP_MULT = 1.0, 5.0
bt.MAX_SL, bt.MIN_RISK, bt.ENTRY_END, bt.DAY_CAP = 20, 3, 900, 40

data = bt.Data()
nm = nearest_map(data)
DAYS = [d for d in DAYS if d in data.days]
all_days = bt.run_days(data, DAYS, nm)

risks, tps = [], []
print(f"{'day':10s} {'time':5s} {'symbol':24s} {'entry':>8s} {'SL':>8s} {'TP':>8s} "
      f"{'risk':>6s} {'tp pts':>7s} {'result':>8s}")
for d, kept in all_days:
    for t in kept:
        risks.append(t["risk"])
        tps.append(t["tp_pts"])
        print(f"{d:10s} {t['j']//60:02d}:{t['j']%60:02d} {t['sym']:24s} "
              f"{t['entry']:8.2f} {t['sl']:8.2f} {t['tp']:8.2f} "
              f"{t['risk']:6.2f} {t['tp_pts']:7.2f} {t['why']:>3s} {t['pts']:+7.2f}")

n = len(risks)
print(f"\ntrades {n}")
print(f"SL points : min {min(risks):.2f}  avg {sum(risks)/n:.2f}  max {max(risks):.2f}")
print(f"TP points : min {min(tps):.2f}  avg {sum(tps)/n:.2f}  max {max(tps):.2f}")
print(f"avg reward:risk = {(sum(tps)/n) / (sum(risks)/n):.2f} : 1")
