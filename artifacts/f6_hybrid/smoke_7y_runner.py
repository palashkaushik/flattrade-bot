"""5-day smoke of the 7y runner (both risk variants)."""
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
    days = sorted(set(opt_map) & set(spot_all))[:5]
    for v in ("A", "B"):
        t0 = time.time()
        r = run_f6_marny_15m_filter_backtest(params_for(v), days, workers=4,
                                             spot_all=spot_all, opt_map=opt_map)
        print(f"variant {v}: trades={r['trades']} WR={r['win_rate']}% "
              f"net_rs={r['net_rs']:+,.2f} pts={r['net_points']:+,.2f} "
              f"PF={r['profit_factor']} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()