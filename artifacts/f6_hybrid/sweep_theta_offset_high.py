"""Extend the theta_offset sweep higher (to=7..12) — the trend is monotonic, find where it peaks."""
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
    for to in [6.0, 7.0, 8.0, 10.0, 12.0]:
        params = params_for("A", None, theta_offset=to, time_stop_min=None)
        t0 = time.time()
        r = run_f6_marny_15m_filter_backtest(params, days, workers=8,
                                             spot_all=spot_all, opt_map=opt_map)
        rows.append((to, r["trades"], r["win_rate"], r["net_rs"], r["net_points"],
                     r["profit_factor"], r["max_drawdown_rs"], time.time() - t0))
        print(f"to={to:>3} | n={r['trades']:5d} WR={r['win_rate']:5.2f}% "
              f"net={r['net_rs']:>+13,.2f} pts={r['net_points']:>+10,.2f} "
              f"PF={r['profit_factor']:.3f} MaxDD={r['max_drawdown_rs']:>12,.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("\n=== SORTED BY NET RS ===")
    for to, n, wr, nrs, npts, pf, mdd, dt in sorted(rows, key=lambda x: -x[4]):
        print(f"to={to:>3} | n={n:5d} WR={wr:5.2f}% net={nrs:>+13,.2f} "
              f"pts={npts:>+10,.2f} PF={pf:.3f} MaxDD={mdd:>12,.0f}")


if __name__ == "__main__":
    main()