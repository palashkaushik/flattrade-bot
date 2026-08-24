"""Tally the Aug 19-20 smoke trades inside the extended 7y run JSON."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path("artifacts/f6_hybrid")


def main():
    data = json.loads((OUT_DIR / "f6_mtf_7y_all4xaug_variantA_nonwf.json").read_text(encoding="utf-8"))
    trs = data["all_trades"]
    print(f"total trades: {len(trs)} | net_rs={data['net_rs']:+,.2f} | WR={data['win_rate']}%")
    for t in sorted(trs, key=lambda x: (x["date"], x["entry_min"])):
        if t["date"] >= "2026-08-01":
            em, xm = t["entry_min"], t["exit_min"]
            print(f"{t['date']} {em//60:02d}:{em%60:02d}->{xm//60:02d}:{xm%60:02d} "
                  f"{t['side']} {t['symbol']} {t['stype']:>10} tf={t.get('tf','?'):>2} "
                  f"entry={t['entry']:.2f} exit={t['exit']:.2f} ({t['reason']}) "
                  f"pts={t['points']:+.2f} rs={t['rs_net']:+,.2f}")
    aug = [t for t in trs if t["date"] >= "2026-08-01"]
    if aug:
        print(f"\nAUGUST total: {len(aug)} trades, net_rs={sum(t['rs_net'] for t in aug):+,.2f}")


if __name__ == "__main__":
    main()