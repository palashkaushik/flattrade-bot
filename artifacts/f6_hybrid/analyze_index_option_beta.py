"""Precise index<->option sensitivity (empirical delta) from cached Aug 12-14 data.

For each option contract we align the Nifty 50 spot (index) and option premium at
1-min resolution and regress the option LEVEL on [index level, minute-of-day]:

    option_price = a + beta * index_price + gamma * minute_of_day

beta = points-of-option per point-of-index (the empirical delta). The time term
soaks up intraday theta decay so beta is not biased low. We only use minutes
where the option actually traded (volume>0) so stale prints don't pollute it.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from artifacts.flattrade_day_cache import load_day_cache

CACHE = Path("artifacts/flattrade_day_cache")
DATES = ["2026-08-12", "2026-08-13", "2026-08-14"]


def minute_key(t):
    return datetime.strptime(t, "%d-%m-%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M")


def to_mod(t):
    dt = datetime.strptime(t, "%d-%m-%Y %H:%M:%S")
    return dt.hour * 60 + dt.minute


def main():
    buckets = {}
    per_contract = []

    for d in DATES:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        cache = load_day_cache(CACHE, dt)
        atm0 = int(round(cache["spot_rows"][0]["close"] / 50.0)) * 50

        # index minute series
        idx = {minute_key(r["time"]): float(r["close"]) for r in cache["spot_rows"]}

        for key, info in cache["contracts"].items():
            side, st = key.split(":", 1)
            strike = int(st)
            xs, ys, ts = [], [], []
            for r in info["rows"]:
                if r["time"].split(" ")[0] != dt.strftime("%d-%m-%Y"):
                    continue
                mk = minute_key(r["time"])
                if mk not in idx:
                    continue
                vol = float(r.get("volume", 0) or 0)
                if vol <= 0:
                    continue
                xs.append(idx[mk])
                ys.append(float(r["close"]))
                ts.append(to_mod(r["time"]))
            if len(xs) < 30:
                continue
            X = np.column_stack([np.ones(len(xs)), np.array(xs), np.array(ts)])
            y = np.array(ys)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            beta = coef[1]
            # R^2
            pred = X @ coef
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            dist = abs(strike - atm0)
            bucket = f"{max(50, (dist // 50) * 50)}"
            per_contract.append((d, side, strike, atm0, dist, beta, r2, len(xs)))
            buckets.setdefault((side, bucket), []).append((np.array(xs), np.array(ys), beta, r2, len(xs)))

    print("=" * 92)
    print("INDEX <-> OPTION SENSITIVITY   beta = option pts per 1 index pt (empirical delta)")
    print("=" * 92)
    print(f"{'Side':<5}{'DistATM':<8}{'nC':<5}{'beta':<8}{'idx +10pts':<12}{'idx -10pts':<12}{'R^2':<6}{'n'}")
    print("-" * 92)
    for (side, bucket), samples in sorted(buckets.items(), key=lambda x: (x[0][0], int(x[0][1]))):
        # pool: beta and r2 weighted by sample count n (s[2]=beta, s[3]=r2, s[4]=n)
        den = sum(s[4] for s in samples)
        if den < 50:
            continue
        beta_pool = sum(s[2] * s[4] for s in samples) / den
        r2_pool = sum(s[3] * s[4] for s in samples) / den
        ncon = len(samples)
        print(f"{side:<5}{bucket:<8}{ncon:<5}{beta_pool:<8.3f}{'+'+format(beta_pool*10,'.2f'):<12}{format(beta_pool*-10,'.2f'):<12}{r2_pool:<6.2f}{int(den)}")

    print("-" * 92)
    print("\nPer-contract (dist<=200 shown):")
    print(f"{'Date':<12}{'Side':<5}{'Strike':<8}{'ATM':<7}{'Dist':<6}{'beta':<8}{'idx10':<8}{'R^2':<6}{'n'}")
    for d, side, strike, atm0, dist, beta, r2, n in sorted(per_contract, key=lambda x: (x[4], x[0])):
        if dist <= 200:
            print(f"{d:<12}{side:<5}{strike:<8}{atm0:<7}{dist:<6}{beta:<8.3f}{beta*10:<8.2f}{r2:<6.2f}{n}")


if __name__ == "__main__":
    main()
