import gzip
import json
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import defaultdict

ROOT = Path(r"c:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
import artifacts.f6_hybrid.marni_fib_5y_fast as m5

def main():
    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    previous = {day: max((c for c in all_days if c < day), default="") for day in all_days}

    timeframe_modes = ("1m",)
    target_levels = (0.0, 0.29)
    stop_levels = (1.155, 1.079)

    tasks = [
        (
            day,
            opt_map[day],
            opt_map.get(previous[day], ""),
            timeframe_modes,
            target_levels,
            stop_levels,
            "index",
            True,
            "touch",
            20.0,
        )
        for day in all_days
    ]

    aggregated = defaultdict(list)
    with Pool(processes=8, initializer=m5.init_worker, initargs=(spot_all,)) as pool:
        for res in pool.imap_unordered(m5.process_day, tasks, chunksize=1):
            for k, v in res.items():
                aggregated[k].extend(v)

    print(f"\n{'='*110}")
    print(f"MARNI FIBONACCI 7-YEAR MULTI-YEAR STUDY: TP 0.29 vs TP 0.0 (2020 - 2026)")
    print(f"{'='*110}")
    print(f"{'Configuration':30s} | {'Trades':6s} | {'Win Rate':9s} | {'Net Points':12s} | {'Profit Factor':14s} | {'Max DD (Rs)':14s} | {'Net Realized P&L (Rs)':22s}")
    print("-" * 115)

    for k in sorted(aggregated.keys()):
        trades = aggregated[k]
        st = m5.compute_stats(trades, len(all_days))
        print(f"{k:30s} | {st['trades']:6d} | {st['win_rate']:8.1f}% | {st['net_points']:+11.2f}p | {st['profit_factor']:13.2f} | Rs {st['max_drawdown_rs']:11,.2f} | Rs {st['net_rs']:+19,.2f}")

    print("-" * 115)

if __name__ == "__main__":
    main()
