"""Diagnose the 7y negative: yearly / side / stype / TF / win-loss breakdowns."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path("artifacts/f6_hybrid")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def show(path, label):
    data = load(path)
    trades = data.get("all_trades", [])
    print(f"\n{'='*100}\n{label} — {len(trades)} trades\n{'='*100}")
    if not trades:
        return

    def grp(key, fn):
        agg = defaultdict(lambda: [0, 0.0])
        for t in trades:
            k = fn(t)
            agg[k][0] += 1
            agg[k][1] += t["rs_net"]
        print(f"--- by {key} ---")
        for k, (n, rs) in sorted(agg.items()):
            print(f"  {k!s:>6}: n={n:5d}  net_rs={rs:>12,.2f}  avg={rs/n:>9,.2f}")

    grp("year", lambda t: t["date"][:4])
    grp("side", lambda t: t["side"])
    grp("stype", lambda t: t["stype"])
    grp("tf", lambda t: t.get("tf", "?"))
    grp("reason", lambda t: t["reason"])

    wins = [t for t in trades if t["rs_net"] > 0]
    losses = [t for t in trades if t["rs_net"] <= 0]
    print(f"\n  wins: {len(wins)}  avg={sum(t['rs_net'] for t in wins)/max(len(wins),1):,.2f}")
    print(f"  losses: {len(losses)}  avg={sum(t['rs_net'] for t in losses)/max(len(losses),1):,.2f}")

    # filter gate: how many days had trades vs total
    days_with_trades = len({t["date"] for t in trades})
    print(f"  days with trades: {days_with_trades}")


if __name__ == "__main__":
    show(OUT_DIR / "f6_mtf_7y_variantA_nonwf.json", "ALL 4 TF variant A (non-WF)")
    show(OUT_DIR / "f6_mtf_7y_1m_variantA_nonwf.json", "1m ONLY variant A (non-WF)")
    show(OUT_DIR / "f6_mtf_7y_2m_variantA_nonwf.json", "2m ONLY variant A (non-WF)")
    show(OUT_DIR / "f6_mtf_7y_3m_variantA_nonwf.json", "3m ONLY variant A (non-WF)")
    show(OUT_DIR / "f6_mtf_7y_5m_variantA_nonwf.json", "5m ONLY variant A (non-WF)")