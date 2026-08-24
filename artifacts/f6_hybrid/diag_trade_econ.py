"""Check trade economics across eras: entry premium, ATR, SL/TP distance vs 2pt slippage."""
import sys
import time
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
    days = sorted(set(opt_map) & set(spot_all))

    eras = {
        "2020": [d for d in days if d.startswith("2020")],
        "2022": [d for d in days if d.startswith("2022")],
        "2024": [d for d in days if d.startswith("2024")],
        "2026-may": [d for d in days if d.startswith("2026-05")],
    }
    for label, subset in eras.items():
        r = run_f6_marny_15m_filter_backtest(params_for("A"), subset, workers=8,
                                             spot_all=spot_all, opt_map=opt_map)
        trs = r["all_trades"]
        if not trs:
            print(f"{label}: no trades")
            continue
        # entry stored already includes +1 slip; sl/tp not persisted -> recompute
        entries = [t["entry"] - 1.0 for t in trs]  # raw entry px before slip
        wins = [t for t in trs if t["rs_net"] > 0]
        losses = [t for t in trs if t["rs_net"] <= 0]
        aw = sum(t["rs_net"] for t in wins) / max(len(wins), 1)
        al = sum(t["rs_net"] for t in losses) / max(len(losses), 1)
        print(f"{label}: n={len(trs):4d} trades | avg_entry(px)={sum(entries)/len(entries):7.2f} "
              f"| min={min(entries):6.2f} max={max(entries):7.2f} "
              f"| avg_win={aw:9.2f} avg_loss={al:9.2f} | WR={r['win_rate']}% "
              f"| net_rs={r['net_rs']:+,.2f}")


if __name__ == "__main__":
    main()