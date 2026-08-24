import sys
from pathlib import Path

ROOT = Path(r"c:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import opt_futures_quad as source
import artifacts.f6_hybrid.marni_vsa_trailing_sl_7y as m_trail

def run():
    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    previous = {day: max((c for c in all_days if c < day), default="") for day in all_days}

    # We test Fixed Target 0.786 TP + Trailing SL (+10/+5)
    tasks = [
        (
            day,
            opt_map[day],
            previous[day],
            20.0,
            "fixed_plus_trail",
            True,
        )
        for day in all_days
    ]

    all_trades = []
    with Pool(processes=8, initializer=m_trail.init_worker, initargs=(spot_all,)) as pool:
        for day_trades in pool.imap_unordered(m_trail.process_day, tasks, chunksize=1):
            all_trades.extend(day_trades)

    st = m_trail.compute_stats(all_trades, len(all_days))
    print(f"\n{'='*110}")
    print(f"MARNI VSA: FIXED TARGET (0.786 TP) + TRAILING SL (+10/+5) 7-YEAR RESULTS")
    print(f"{'='*110}")
    print(f"Trades: {st['trades']} | WR: {st['win_rate']:.1f}% | Net Points: {st['net_points']:+.2f}p | Net Rs: Rs {st['net_rs']:+,.2f} | PF: {st['profit_factor']:.2f} | Max DD: Rs {st['max_drawdown_rs']:,.2f}")

    by_year = defaultdict(list)
    for t in all_trades:
        by_year[t["date"][:4]].append(t)

    print(f"\n{'Year':6s} | {'Trades':8s} | {'Win Rate':9s} | {'Points':12s} | {'Profit Factor':14s} | {'Max DD (Rs)':14s} | {'Net Realized P&L (Rs)':22s}")
    print("-" * 95)
    for y in sorted(by_year.keys()):
        y_trades = by_year[y]
        yst = m_trail.compute_stats(y_trades, len(set(t["date"] for t in y_trades)))
        print(f"{y:6s} | {yst['trades']:8d} | {yst['win_rate']:8.1f}% | {yst['net_points']:+11.2f}p | {yst['profit_factor']:13.2f} | Rs {yst['max_drawdown_rs']:11,.2f} | Rs {yst['net_rs']:+19,.2f}")

if __name__ == "__main__":
    run()
