"""Direct 5-day comparison: my F0 baseline vs reference engine."""
from backtest_5y_optimized import load_spot, option_files, summarize
import backtest_unlimited_profit as ref
import test_winrate_filters as myf
from multiprocessing import Pool


def run_ref(spot_all, files, days):
    days5 = days[:5]
    print(f"Days: {days5}")
    tasks_ref = [
        (day, str(files[day]), str(files[days[i-1]]) if i > 0 else "",
         "trailing", 1.0, 2.0)
        for i, day in enumerate(days5)
    ]
    all_ref = []
    with Pool(4, initializer=ref.init_worker_local, initargs=(spot_all,)) as pool:
        for res in pool.map(ref.process_day, tasks_ref):
            all_ref.extend(res)
    st = summarize(all_ref)
    print(f"REF  5-day: Trades={st['trades']:3d} | WR={st['wr']:.1f}% | Rs={st['rs']:+,d}")
    return all_ref


def run_mine(spot_all, files, days):
    days5 = days[:5]
    cfg0 = {"filter_id": "F0"}
    tasks_my = [
        (day, str(files[day]), str(files[days[i-1]]) if i > 0 else "")
        for i, day in enumerate(days5)
    ]
    all_my = []
    with Pool(4, initializer=myf.init_worker, initargs=(spot_all, cfg0)) as pool:
        for res in pool.map(myf.process_day, tasks_my):
            all_my.extend(res)
    st = summarize(all_my)
    print(f"MINE 5-day: Trades={st['trades']:3d} | WR={st['wr']:.1f}% | Rs={st['rs']:+,d}")
    return all_my


if __name__ == "__main__":
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    all_ref  = run_ref(spot_all, files, days)
    all_mine = run_mine(spot_all, files, days)

    print(f"\n--- REF trades ({len(all_ref)}) ---")
    for t in sorted(all_ref, key=lambda x: (x['date'], x['entry_min'])):
        print(f"  {t['date']} m={t['entry_min']:4d}->{t['exit_min']:4d} "
              f"{t['side']:2s} entry={t['entry']:.1f} exit={t['exit']:.1f} "
              f"pts={t['pts']:+.1f} [{t['reason']}]")

    print(f"\n--- MY trades ({len(all_mine)}) ---")
    for t in sorted(all_mine, key=lambda x: (x['date'], x['entry_min'])):
        print(f"  {t['date']} m={t['entry_min']:4d}->{t['exit_min']:4d} "
              f"{t['side']:2s} entry={t['entry']:.1f} exit={t['exit']:.1f} "
              f"pts={t['pts']:+.1f} [{t['reason']}]")
