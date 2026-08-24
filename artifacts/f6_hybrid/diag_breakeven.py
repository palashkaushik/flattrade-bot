"""Breakeven math + exit-reason breakdown for a full year."""
import sys
from collections import defaultdict
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
    days = [d for d in sorted(set(opt_map) & set(spot_all)) if d.startswith("2024")]
    r = run_f6_marny_15m_filter_backtest(params_for("A"), days, workers=8,
                                         spot_all=spot_all, opt_map=opt_map)
    trs = r["all_trades"]
    print(f"2024: {len(trs)} trades | net={r['net_rs']:+,.2f} | WR={r['win_rate']}%")

    by_reason = defaultdict(lambda: [0, 0.0])
    for t in trs:
        by_reason[t["reason"]][0] += 1
        by_reason[t["reason"]][1] += t["rs_net"]
    print("\nby exit reason:")
    for k, (n, rs) in sorted(by_reason.items()):
        print(f"  {k:>6}: n={n:5d} net={rs:>12,.2f} avg={rs/max(n,1):>9,.2f}")

    # breakeven WR for 3:5 with costs: need avg_win*WR > avg_loss*(1-WR)
    wins = [t for t in trs if t["rs_net"] > 0]
    losses = [t for t in trs if t["rs_net"] <= 0]
    aw = sum(t["rs_net"] for t in wins) / max(len(wins), 1)
    al = -sum(t["rs_net"] for t in losses) / max(len(losses), 1)
    be_wr = al / (aw + al) * 100
    print(f"\navg_win={aw:.2f} avg_loss={al:.2f} -> breakeven WR={be_wr:.1f}% (actual {r['win_rate']}%)")


if __name__ == "__main__":
    main()