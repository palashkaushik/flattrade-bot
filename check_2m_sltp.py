"""Verify 2-Minute Timeframe Performance across SL/TP settings."""

import pandas as pd
from backtest_5y_optimized import load_spot, option_files, init_worker, process_single_day, summarize

def main():
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    tasks = []
    for i in range(len(days)):
        day = days[i]
        curr_file = str(files[day])
        prev_file = str(files[days[i-1]]) if i > 0 else ""
        tasks.append((day, curr_file, prev_file))

    from multiprocessing import Pool, cpu_count
    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker, initargs=(spot_all,)) as pool:
        results = pool.map(process_single_day, tasks)
        for res in results:
            all_trades.extend(res)

    df = pd.DataFrame(all_trades)
    df_2m = df[df["tf"] == "2m"].to_dict("records")
    st_2m = summarize(df_2m)

    print("\n" + "=" * 90)
    print("VERIFICATION: 2-MINUTE TIMEFRAME AT DEFAULT SL = 10 pts, TP = 15 pts")
    print("=" * 90)
    print(f"Total Trades : {st_2m['trades']}")
    print(f"Win Rate     : {st_2m['wr']:.1f}%")
    print(f"Net Points   : {st_2m['pts']:+.2f} pts")
    print(f"Net Profit   : Rs {st_2m['rs']:+,d}")
    print(f"Profit Factor: {st_2m['pf']:.2f}")

if __name__ == "__main__":
    main()
