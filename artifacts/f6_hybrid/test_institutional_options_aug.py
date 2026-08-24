"""Compare Institutional Strategy with and without 15m HTF Filter on August 18, 19, 20."""

from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.run_institutional_high_conviction_aug import (
    simulate_institutional_day, extend_with_august, load_full_ohlc_spot, option_files, to_hhmm,
)
import grid_optimize_f6_atr as grid

def main():
    spot_all = load_full_ohlc_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    grid.GLOBAL_SPOT = spot_all
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}

    for use_htf, label in [(True, "WITH 15m HTF Trend Filter"), (False, "PURE OPTIONS PRICE ACTION (NO HTF)")]:
        print("=" * 125)
        print(f">>> CONFIGURATION: {label} (09:30 AM Start, 15m Theta Cut, Max 3 Tr/Day, 15m Cooldown)")
        print("=" * 125)
        all_trs = []
        for day in ["2026-08-18", "2026-08-19", "2026-08-20"]:
            trs = simulate_institutional_day(
                day, opt_map, all_cal, cal_idx, spot_all,
                use_htf=use_htf, start_minute=570, max_trades_day=3, cooldown_min=15,
                sl_mult=2.0, be_trig_mult=1.25, trail_trig_mult=1.5, trail_dist_mult=0.5,
                time_stop_min=15, tp_mult=3.5,
            )
            all_trs.extend(trs)
            print(f"\n* Date: {day} (Trades: {len(trs)})")
            for i, t in enumerate(trs, 1):
                t_str = f"{to_hhmm(t['entry_min'])}->{to_hhmm(t['exit_min'])}"
                print(f"  {i}. {t_str:11s} | {t['symbol']:18s} | {t['side']:4s} | Entry: {t['entry']:7.2f} | Exit: {t['exit']:7.2f} | Pts: {t['pts']:+6.2f} | Net Rs: Rs {t['rs_net']:+8.2f} | {t['reason']:12s}")

        wins = [t for t in all_trs if t["rs_net"] > 0]
        losses = [t for t in all_trs if t["rs_net"] <= 0]
        net_rs = sum(t["rs_net"] for t in all_trs)
        net_pts = sum(t["pts"] for t in all_trs)
        wr = len(wins) / len(all_trs) * 100 if all_trs else 0.0

        print("\n" + "-" * 80)
        print(f"3-DAY RESULT: {len(all_trs)} trades | {len(wins)} Wins, {len(losses)} Losses | Win Rate: {wr:.1f}% | Net Pts: {net_pts:+.2f} pts | Net PnL: Rs {net_rs:+,.2f}")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    main()
