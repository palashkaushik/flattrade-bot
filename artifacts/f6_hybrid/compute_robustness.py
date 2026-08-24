"""Robustness verification for the Optimus HFT sweep.

Reads the per-pool IS/OOS CSV produced by optimus_hft_cash_machine_sweep.py and
reports the overfitting checks recommended by the web research:

  * Walk-Forward Correlation (WFC): Pearson/Spearman corr of IS net vs OOS net
    across the consistency pool. High (>0.6) => genuine edge, not overfit.
  * OOS positive rate: fraction of pool candidates still profitable OOS.
  * Walk-Forward Efficiency (WFE): median OOS/IS net ratio (positive-IS only).
  * Top-K stability: of the top-20 IS candidates, how many survive OOS (net>0, WR>=floor).

Usage:
  python compute_robustness.py <trend> <use_vol>
  python compute_robustness.py 15 1        # vol gate ON
  python compute_robustness.py 15 0        # ablation (vol gate OFF)
"""
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_pool(trend, use_vol):
    p = HERE / f"optimus_pool_oos_trend{trend}_vol{use_vol}.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            rows.append({
                "trial": int(r["trial"]),
                "is_net": float(r["is_net"]),
                "oos_net": float(r["oos_net"]),
                "is_wr": float(r["is_wr"]),
                "oos_wr": float(r["oos_wr"]),
            })
    return rows


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sx = sy = 0.0
    for a, b in zip(x, y):
        d = a - mx
        e = b - my
        sxy += d * e
        sx += d * d
        sy += e * e
    if sx == 0 or sy == 0:
        return float("nan")
    return sxy / math.sqrt(sx * sy)


def spearman(x, y):
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    return pearson(rank(x), rank(y))


def main():
    trend = sys.argv[1] if len(sys.argv) > 1 else "15"
    use_vol = sys.argv[2] if len(sys.argv) > 2 else "1"
    rows = load_pool(trend, use_vol)
    is_net = [r["is_net"] for r in rows]
    oos_net = [r["oos_net"] for r in rows]
    is_wr = [r["is_wr"] for r in rows]
    oos_wr = [r["oos_wr"] for r in rows]

    n = len(rows)
    r_p = pearson(is_net, oos_net)
    r_s = spearman(is_net, oos_net)
    oos_pos = sum(1 for v in oos_net if v > 0)
    wfe = [b / a for a, b in zip(is_net, oos_net) if a > 0]
    wfe_med = sorted(wfe)[len(wfe) // 2] if wfe else float("nan")

    top = sorted(rows, key=lambda r: r["is_net"], reverse=True)[:20]
    top_oos_pos = sum(1 for r in top if r["oos_net"] > 0)
    top_oos_wr_ok = sum(1 for r in top if r["oos_net"] > 0 and r["oos_wr"] >= 55.0)

    champ = max(rows, key=lambda r: r["is_net"])
    print(f"\n===== ROBUSTNESS  [trend={trend} vol={use_vol}]  pool={n} =====")
    print(f"  Walk-Forward Correlation (Pearson) IS vs OOS net : {r_p:.3f}")
    print(f"  Walk-Forward Correlation (Spearman)             : {r_s:.3f}")
    print(f"  OOS positive rate (pool)                        : {oos_pos}/{n} = {100*oos_pos/n:.1f}%")
    print(f"  Walk-Forward Efficiency (median OOS/IS)         : {wfe_med:.3f}")
    print(f"  Top-20 IS candidates surviving OOS (net>0)      : {top_oos_pos}/20")
    print(f"  Top-20 IS candidates OOS net>0 AND WR>=55%      : {top_oos_wr_ok}/20")
    print(f"  Champion IS net=Rs{champ['is_net']:+,.0f}  OOS net=Rs{champ['oos_net']:+,.0f}  "
          f"OOS WR={champ['oos_wr']:.1f}%")

    verdict = "ROBUST (genuine edge)" if (r_p > 0.6 and oos_pos / n > 0.7) else "OVERFIT RISK — re-check"
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
