"""Average SL / TP distance + trade frequency stats for a given config.

Usage: py -3.13 sl_tp_stats.py
Runs the full 5Y (2020-2024) with the exact #1 Optuna config
(S1=12,3 S4=50,10 ATR10 SLx3.0 TPx6.0 F6 79.5/25.0 CL=8) and prints:
  - avg / median / p25 / p75 SL and TP distance (points, per trade)
  - avg trades per day, trades by exit reason, by TF
  - avg SL/TP by TF (1m/2m/3m/5m)
"""

import time
from multiprocessing import Pool

import pandas as pd

from backtest_5y_optimized import load_spot, option_files
import grid_optimize_f6_atr as eng

CONFIG = {"s1_k": 12, "s4_k": 50, "atr_period": 10, "atr_sl_mult": 3.0,
          "atr_tp_mult": 6.0, "f6_s4_thresh": 79.5, "f6_s1_thresh": 20.5,
          "consec_loss": 8, "s1_d": 3}


def main():
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))
    print(f"Days: {len(days)}")

    t0 = time.time()
    with Pool(processes=eng.WORKERS, initializer=eng.init_worker_local,
              initargs=(spot_all,)) as pool:
        trades = eng.run_days(pool, CONFIG, days, files, spot_all)
    print(f"Run time: {time.time()-t0:.0f}s | trades: {len(trades)}")

    df = pd.DataFrame(trades)
    df["date"] = df["date"].astype(str)
    st = eng.summarize(trades)
    print(f"\nTotal: {st['trades']} trades | WR {st['wr']:.1f}% | Net {st['rs']:+,d} | PF {st['pf']:.2f}")

    # avg trades per day
    tpd = df.groupby("date").size()
    print(f"\nAvg trades/day: {tpd.mean():.2f} | median {tpd.median():.0f} | "
          f"max {tpd.max()} | days with 0 trades: {(tpd == 0).sum()}")

    def pct(series, name):
        print(f"\n{name}: mean {series.mean():.2f} pts | median {series.median():.2f} | "
              f"p25 {series.quantile(0.25):.2f} | p75 {series.quantile(0.75):.2f} | "
              f"min {series.min():.2f} | max {series.max():.2f}")

    pct(df["sl_pts"], "SL distance")
    pct(df["tp_pts"], "TP distance")

    print("\nBy TF:")
    for tf, g in df.groupby("tf"):
        print(f"  {tf}: n={len(g):4d} | avg SL {g['sl_pts'].mean():6.2f} | avg TP {g['tp_pts'].mean():7.2f} | "
              f"avg pts {g['pts'].mean():7.2f} | WR {100*(g['pts']>0).mean():5.1f}%")

    print("\nBy exit reason:")
    for r, g in df.groupby("reason"):
        print(f"  {r:22s} n={len(g):4d} ({100*len(g)/len(df):5.1f}%) | avg pts {g['pts'].mean():7.2f}")

    print("\nYearly:")
    df["year"] = df["date"].str[:4]
    for y, g in df.groupby("year"):
        print(f"  {y}: n={len(g):4d} | trades/day {len(g)/g['date'].nunique():5.2f} | net {g['rs'].sum():+9,d} | WR {100*(g['pts']>0).mean():5.1f}%")

    print(f"\nAve SL/TP ratio: {df['sl_pts'].mean() / df['tp_pts'].mean():.3f}")


if __name__ == "__main__":
    main()
