"""Find exact SL/TP parameters that achieve MAXIMUM NET POINTS for each timeframe."""

import pandas as pd
from grid_search_fast_pointer import precompute_day_signals, simulate_day_fast, load_spot, option_files, init_worker_local, summarize
from itertools import product

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
    print(f"Loading {len(days)} trading days...", flush=True)
    precomputed_days = {}
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,)) as pool:
        results = pool.map(precompute_day_signals, tasks)
        for day_val, data_dict in results:
            if data_dict:
                precomputed_days[day_val] = data_dict

    sl_range = [6.0, 8.0, 10.0, 12.0, 15.0]
    tp_range = [12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0]
    tf_labels = ["1m", "2m", "3m", "5m"]

    grid_results = []
    for tf_label in tf_labels:
        for sl, tp in product(sl_range, tp_range):
            all_trades = []
            for day_val, day_data in precomputed_days.items():
                t_list = simulate_day_fast(day_data, tf_label, sl, tp)
                for tr in t_list:
                    tr["date"] = day_val
                all_trades.extend(t_list)

            st = summarize(all_trades)
            grid_results.append({
                "tf": tf_label, "sl": sl, "tp": tp, "rr": round(tp/sl, 2),
                "trades": st["trades"], "wr": st["wr"], "pts": st["pts"], "rs": st["rs"], "pf": st["pf"]
            })

    df = pd.DataFrame(grid_results)

    print("\n" + "=" * 125)
    print("SL / TP PARAMETERS FOR MAXIMUM NET POINTS IN EACH TIMEFRAME")
    print("=" * 125)
    print(f"{'TIMEFRAME':12s} | {'BEST SL (pts)':13s} | {'BEST TP (pts)':13s} | {'R:R RATIO':9s} | {'TRADES':7s} | {'WIN RATE':9s} | {'MAX NET PTS':12s} | {'NET PROFIT (Rs)':16s} | {'PROFIT FACTOR'}")
    print("-" * 125)

    for tf_name, g in df.groupby("tf"):
        best_row = g.sort_values("pts", ascending=False).iloc[0]
        pf_str = f"{best_row['pf']:.2f}" if best_row['pf'] != float("inf") else "INF"
        print(f"{best_row['tf']:12s} | {best_row['sl']:13.1f} | {best_row['tp']:13.1f} | 1:{best_row['rr']:<7.2f} | {int(best_row['trades']):7d} | {best_row['wr']:8.1f}% | {best_row['pts']:+12.2f} | Rs {int(best_row['rs']):+14,d} | {pf_str:>13s}")

if __name__ == "__main__":
    main()
