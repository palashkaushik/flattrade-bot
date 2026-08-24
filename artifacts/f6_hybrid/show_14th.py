import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import artifacts.f6_hybrid.run_marni_idxbias_12_13_14 as r

for mode in ("idxbias", "idxbias_index"):
    trades = r.run_mode(mode)
    aug14 = [t for t in trades if t["date"] == "2026-08-14"]
    print("=" * 90)
    print(f"{mode.upper()}  --  AUGUST 14 trades ({len(aug14)})")
    print("=" * 90)
    hdr = f"{'Time':<8}{'TF':<5}{'Side':<5}{'Strike':<8}{'ATM':<7}{'Entry':<9}{'Exit':<9}{'R':<4}{'Pts':<8}{'Net':<9}Reason"
    print(hdr)
    print("-" * 90)
    for t in aug14:
        tf = t.get("timeframe", "-")
        side = t.get("side", "-")
        strike = t.get("strike", 0)
        atm = t.get("atm", 0)
        print(
            f"{r.fmt_min(t['entry_min']):<8}{tf:<5}{side:<5}{strike:<8}{atm:<7}"
            f"{t['entry']:<9.2f}{t['exit']:<9.2f}{t['reason']:<4}{t['points']:<8.2f}{t['rs_net']:<9.2f}{t['reason']}"
        )
    print()
    w = sum(1 for t in aug14 if t["points"] > 0)
    print(f"  Aug14 {mode}: {len(aug14)} sig | {w} win | {100*w/len(aug14):.1f}%")

print("=" * 90)
ti = r.run_mode("idxbias_index")
ti_sorted = sorted(ti, key=lambda t: t["entry_min"])
t14 = ti_sorted[13]
print("IDXBIAS_INDEX 14th trade in chronological order:")
for k, v in t14.items():
    print(f"  {k}: {v}")
