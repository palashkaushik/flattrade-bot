"""Sweep theta_offset and time_stop_min across the extended 7y dataset (through 08-20).
Reports a compact table so we can find the combination that best fights theta decay."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from artifacts.f6_hybrid.f6_mtf_7y_runner import extend_with_august, params_for
from artifacts.f6_hybrid.f6_champion_marny_15m_filter_backtest import run_f6_marny_15m_filter_backtest


def main():
    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    days = sorted(set(opt_map) & set(spot_all))
    print(f"days: {days[0]}..{days[-1]} | n={len(days)}", flush=True)

    rows = []
    for to in [0.0, 1.0, 2.0, 3.0, 5.0]:
        for ts in [None, 90, 120, 150]:
            params = params_for("A", None, theta_offset=to, time_stop_min=ts)
            t0 = time.time()
            r = run_f6_marny_15m_filter_backtest(params, days, workers=8,
                                                 spot_all=spot_all, opt_map=opt_map)
            rows.append((to, ts, r["trades"], r["win_rate"], r["net_rs"], r["net_points"],
                         r["profit_factor"], r["max_drawdown_rs"], time.time() - t0))
            print(f"to={to:>3} ts={str(ts):>5} | n={r['trades']:5d} "
                  f"WR={r['win_rate']:5.2f}% net={r['net_rs']:>+13,.2f} "
                  f"pts={r['net_points']:>+10,.2f} PF={r['profit_factor']:.3f} "
                  f"MaxDD={r['max_drawdown_rs']:>12,.0f} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== SORTED BY NET RS ===")
    for to, ts, n, wr, nrs, npts, pf, mdd, dt in sorted(rows, key=lambda x: -x[5]):
        print(f"to={to:>3} ts={str(ts):>5} | n={n:5d} WR={wr:5.2f}% "
              f"net={nrs:>+13,.2f} pts={npts:>+10,.2f} PF={pf:.3f} MaxDD={mdd:>12,.0f}")


if __name__ == "__main__":
    main()