"""Smoke run of F6 Champion 12/50 + Marny 15m Option Filter on 2026-08-19 and 2026-08-20.

Injects:
  - spot (index 1m) from C:\\Websites\\ammu\\data\\2026-08-{19,20}\\nifty50_index_1m_2026-08-{19,20}.csv
  - option day files from C:\\Users\\user\\Desktop\\nifty50 data\\nifty_options\\2026\\8\\

Runs process_day_f6_marny_15m_filter exactly like the full engine, prints the
trade list + summary for manual verification against the live trading day.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.f6_champion_marny_15m_filter_backtest import (
    CHAMPION_12_50,
    run_f6_marny_15m_filter_backtest,
)

DESKTOP_OPTS = Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options\2026\8")
AMMU_DATA = Path(r"C:\Websites\ammu\data")
DAYS = ["2026-08-19", "2026-08-20"]


def load_index_spot():
    """Parse the fresh 1m index CSV into load_spot()'s {day: arrays} format.

    The file has a stray extra column; parse the first 5 fields as raw text and
    rebuild timestamp/minute manually to stay robust.
    """
    out = {}
    for day in DAYS:
        p = AMMU_DATA / day / f"nifty50_index_1m_{day}.csv"
        rows = []
        with open(p) as fh:
            header = fh.readline().strip().split(",")
            t_col = header.index("timestamp")
            for line in fh:
                fields = line.strip().split(",")
                if len(fields) <= t_col:
                    continue
                ts = fields[t_col]
                try:
                    o = float(fields[t_col + 1])
                    h = float(fields[t_col + 2])
                    l = float(fields[t_col + 3])
                    c = float(fields[t_col + 4])
                except (ValueError, IndexError):
                    continue
                dt = datetime.fromisoformat(ts)
                rows.append((dt.strftime("%Y-%m-%d"), dt.hour * 60 + dt.minute, o, h, l, c))
        if not rows:
            continue
        arr = np.array(rows, dtype=object)
        df = pd.DataFrame(rows, columns=["day", "min", "open", "high", "low", "close"])
        for d, g in df.groupby("day"):
            out[d] = {
                "min": g["min"].to_numpy(),
                "open": g["open"].to_numpy(),
                "high": g["high"].to_numpy(),
                "low": g["low"].to_numpy(),
                "close": g["close"].to_numpy(),
            }
    return out


def build_opt_map():
    """Map the Desktop day-files (incl. 08-18 warmup) into {day: path}."""
    result = {}
    for p in sorted(DESKTOP_OPTS.glob("nifty_options_*.csv")):
        parts = p.stem.split("_")
        day = f"{parts[4]}-{parts[3]}-{parts[2]}"
        result[day] = p
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", action="append", choices=["1m", "2m", "3m", "5m"],
                    help="Restrict F6 engine TFs (repeatable; default = all 4)")
    args = ap.parse_args()

    spot_all = load_index_spot()
    opt_map = build_opt_map()
    present = sorted(set(DAYS) & set(opt_map) & set(spot_all))
    print("Injected spot days:", sorted(spot_all))
    print("Injected option days:", sorted(opt_map))
    print("Smoke days (overlap):", present)

    params = {
        "f6_params": CHAMPION_12_50,
        "include_fees": True,
        "trail_sl": True,
        "daily_loss_pts": -30.0,
        "no_pinbar": False,
        "s1_turn_up": False,
        "tfs": args.tf,
    }

    t0 = time.time()
    res = run_f6_marny_15m_filter_backtest(params, present, workers=4,
                                           spot_all=spot_all, opt_map=opt_map)
    el = time.time() - t0

    print(f"\nExecution finished in {el:.2f}s")
    print("=" * 130)
    print(f"Overall: Trades={res['trades']} | Win Rate={res['win_rate']}% | "
          f"Net Points={res['net_points']:+,.2f} | Net Rs=Rs {res['net_rs']:+,.2f} | "
          f"PF={res['profit_factor']} | MaxDD=Rs {res['max_drawdown_rs']:,.2f} | Fees=Rs {res['fees_rs']:,.2f}")
    print("=" * 130)

    by_day = {}
    for t in sorted(res["all_trades"], key=lambda x: (x["date"], x["entry_min"])):
        by_day.setdefault(t["date"], []).append(t)

    for day in present:
        trs = by_day.get(day, [])
        print(f"\n--- {day} ({len(trs)} trades) ---")
        for t in trs:
            em, xm = t["entry_min"], t["exit_min"]
            print(f"  {em//60:02d}:{em%60:02d} -> {xm//60:02d}:{xm%60:02d} "
                  f"{t['side']} {t['symbol']} {t['stype']:>12} tf={t.get('tf', '?'):>2} "
                  f"entry={t['entry']:.2f} "
                  f"exit={t['exit']:.2f} ({t['reason']}) pts={t['points']:+.2f} rs={t['rs_net']:+,.2f}")
        day_pts = sum(t["points"] for t in trs)
        day_rs = sum(t["rs_net"] for t in trs)
        print(f"  DAY P/L: {day_pts:+.2f} pts | Rs {day_rs:+,.2f}")


if __name__ == "__main__":
    main()