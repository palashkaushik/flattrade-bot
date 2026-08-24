"""Compare recent-period behavior: last 10 trading days of the 7y data (Apr-May 2026)
vs the full-range negative. Uses the SAME engine + params as the smoke (variant A,
all 4 TFs). If the last days are positive, the bleed is old-regime; if negative,
suspect the engine/data path itself."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from artifacts.f6_hybrid.f6_mtf_7y_runner import params_for
from artifacts.f6_hybrid.f6_champion_marny_15m_filter_backtest import run_f6_marny_15m_filter_backtest


def main():
    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    days = sorted(set(opt_map) & set(spot_all))

    for label, subset in [
        ("LAST 5 DAYS", days[-5:]),
        ("MAY 2026", [d for d in days if d.startswith("2026-05")]),
        ("APR 2026", [d for d in days if d.startswith("2026-04")]),
        ("2025 (full)", [d for d in days if d.startswith("2025")]),
        ("2020 (full)", [d for d in days if d.startswith("2020")]),
    ]:
        if not subset:
            print(f"{label}: no days")
            continue
        t0 = time.time()
        r = run_f6_marny_15m_filter_backtest(params_for("A"), subset, workers=8,
                                             spot_all=spot_all, opt_map=opt_map)
        print(f"{label}: n={len(subset):4d} days | trades={r['trades']:5d} "
              f"WR={r['win_rate']}% | net_rs={r['net_rs']:+,.2f} "
              f"pts={r['net_points']:+,.2f} | PF={r['profit_factor']} "
              f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()