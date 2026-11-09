"""Rank the sweep: exact WR/points/fees, tight drawdown bound, Pareto frontier.

Reads sweep_days/*.npz in day order. Memory-light: one day at a time.
Usage: python sweep_rank.py [min_trades] [top_n]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sweep_grid import KEYS, TRIGGERS, SLS, TPS, MAXSLS, CAPS  # noqa: E402

QTY, FEE = 65, 45
YEARS = 6.33


def main():
    min_tr = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    days = sorted(f[:-4] for f in os.listdir(os.path.join(HERE, "sweep_days"))
                  if f.endswith(".npz"))
    n = len(KEYS)
    E = np.zeros(n, np.float64)      # cumulative points before day
    P = np.zeros(n, np.float64)      # running peak
    DD = np.zeros(n, np.float64)     # drawdown bound
    NTR = np.zeros(n, np.int64)
    NW = np.zeros(n, np.int64)
    for d in days:
        z = np.load(os.path.join(HERE, "sweep_days", f"{d}.npz"))
        net, cnt, win, lo, hi, ddi = (z[k].astype(np.float64)
                                      for k in ("net", "cnt", "win", "lo", "hi", "ddi"))
        has = cnt > 0
        P_prev = P.copy()
        P = np.maximum(P, E + np.where(has, hi, 0.0))
        DD = np.maximum(DD, np.where(has, np.maximum(P_prev - (E + lo), ddi), 0.0))
        E = E + net
        NTR += cnt.astype(np.int64)
        NW += win.astype(np.int64)

    pts = E
    wr = np.divide(NW, NTR, out=np.zeros(n), where=NTR > 0) * 100.0
    rs = pts * QTY - NTR * FEE
    cal = np.where(DD > 0, (pts / YEARS) / DD, 0.0)
    be = np.divide(rs, 2 * NTR * QTY, out=np.zeros(n), where=NTR > 0)

    # control lines
    def show(key):
        if key in KEYS:
            i = KEYS.index(key)
            print(f"  {key:34s} tr {NTR[i]:5d} WR {wr[i]:5.1f}% pts {pts[i]:9.2f} "
                  f"Rs {rs[i]:+10,.0f} DD {DD[i]:8.2f} Cal {cal[i]:6.2f} BE {be[i]:5.2f}")
    print(f"{len(days)} days, {n} configs, min_trades {min_tr}")
    print("controls:")
    show("ob|1.0|5.0|20.0|40")
    show("obonly|1.0|5.0|20.0|40")

    ok = NTR >= min_tr
    idx = np.where(ok)[0]

    def table(title, order, k):
        print(f"\n=== {title} ({len(idx)} eligible)")
        print(f"{'config':34s} {'tr':>5s} {'WR':>6s} {'pts':>9s} {'Rs':>11s} {'DD':>8s} {'Cal':>6s} {'BE':>5s}")
        for i in order[:k]:
            print(f"{KEYS[i]:34s} {NTR[i]:5d} {wr[i]:6.1f} {pts[i]:9.2f} "
                  f"{rs[i]:+11,.0f} {DD[i]:8.2f} {cal[i]:6.2f} {be[i]:5.2f}")

    table("top Calmar", idx[np.argsort(-cal[idx])], topn)
    table("top points", idx[np.argsort(-pts[idx])], min(topn, 10))
    table("top win rate", idx[np.argsort(-wr[idx])], min(topn, 10))
    table("top slippage tolerance", idx[np.argsort(-be[idx])], min(topn, 10))

    # Pareto frontier on (WR, pts, Calmar)
    order = idx[np.argsort(-pts[idx])]
    frontier = []
    best_wr = best_cal = -1e9
    for i in order:
        if wr[i] > best_wr or cal[i] > best_cal:
            frontier.append(i)
            best_wr = max(best_wr, wr[i])
            best_cal = max(best_cal, cal[i])
    table("Pareto (WR x pts x Calmar)", frontier, 25)

    # save csv
    out = os.path.join(HERE, "sweep_metrics.csv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("config,trigger,sl,tp,maxsl,cap,trades,wr,pts,rs,dd,calmar,be\n")
        for i in range(n):
            if NTR[i] == 0:
                continue
            t, sl, tp, mx, cap = KEYS[i].split("|")
            fh.write(f"{KEYS[i]},{t},{sl},{tp},{mx},{cap},{NTR[i]},{wr[i]:.1f},"
                     f"{pts[i]:.2f},{rs[i]:.0f},{DD[i]:.2f},{cal[i]:.3f},{be[i]:.3f}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
