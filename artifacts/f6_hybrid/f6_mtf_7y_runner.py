"""F6 Champion 12/50 + Marny 15m Option Filter — 7-year backtest driver.

Runs the combined MTF engine (1m/2m/3m/5m simultaneously, one position at a
time, first-setup-wins, deterministic tie-break) in two modes:

  - non-WF: full 2020-2026 single window
  - WF:     fixed-champion walk-forward folds (IS 2020-22 -> OOS 2023, ...)

under two risk variants:
  - A: daily loss -30 pts + 8 consecutive losses (as tested on 08-19/08-20)
  - B: 4 consecutive losses, no daily-loss cap (Pocket Money live style)

Outputs JSON per (mode, variant) under artifacts/f6_hybrid/.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from artifacts.f6_hybrid.f6_champion_marny_15m_filter_backtest import (
    CHAMPION_12_50,
    run_f6_marny_15m_filter_backtest,
)

START = "2020-01-01"
END = "2026-05-05"
OUT_DIR = Path("artifacts/f6_hybrid")

DESKTOP_OPTS = Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options\2026\8")
AMMU_DATA = Path(r"C:\Websites\ammu\data")


def extend_with_august(opt_map: dict, spot_all: dict) -> tuple[dict, dict]:
    """Merge Desktop August option files + ammu August 1m spot into the maps.

    The ammu option archive ends 2026-05-05; the fresh Desktop files cover
    2026-08-03..2026-08-20 (same vendor format). Spot for those days exists
    in ammu/data/2026-08-*/. Returns (opt_map, spot_all) merged.
    """
    opt_map = dict(opt_map)
    for p in sorted(DESKTOP_OPTS.glob("nifty_options_*.csv")):
        parts = p.stem.split("_")
        day = f"{parts[4]}-{parts[3]}-{parts[2]}"
        opt_map[day] = p
    for d in sorted(AMMU_DATA.glob("2026-08-*")):
        day = d.name
        f = d / f"nifty50_index_1m_{day}.csv"
        if not f.exists():
            continue
        rows = []
        with open(f) as fh:
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
                rows.append((dt.hour * 60 + dt.minute, o, h, l, c))
        if not rows:
            continue
        import numpy as np
        arr = np.array(rows)
        spot_all[day] = {
            "min": arr[:, 0].astype(int),
            "open": arr[:, 1],
            "high": arr[:, 2],
            "low": arr[:, 3],
            "close": arr[:, 4],
        }
    return opt_map, spot_all

FOLDS = [
    {"oos_year": "2023"},
    {"oos_year": "2024"},
    {"oos_year": "2025"},
    {"oos_year": "2026"},
]


def params_for(variant: str, tfs=None, theta_offset: float = 0.0,
               time_stop_min: int | None = None) -> dict:
    f6_params = dict(CHAMPION_12_50)
    if variant == "A":
        f6_params["consec_loss"] = 8
        daily_loss_pts = -30.0
    else:
        f6_params["consec_loss"] = 4
        daily_loss_pts = -1_000_000.0  # no daily-loss cap (4-loss stop only)
    return {
        "f6_params": f6_params,
        "include_fees": True,
        "trail_sl": True,
        "daily_loss_pts": daily_loss_pts,
        "no_pinbar": False,
        "s1_turn_up": False,
        "tfs": tfs,  # None = all 4 timeframes simultaneously; ["2m"] = single TF
        "theta_offset": theta_offset,
        "time_stop_min": time_stop_min,
    }


def run_mode_nonwf(spot_all, opt_map, params, workers):
    days = sorted(set(opt_map) & set(spot_all))
    t0 = time.time()
    res = run_f6_marny_15m_filter_backtest(params, days, workers=workers,
                                           spot_all=spot_all, opt_map=opt_map)
    return {
        "mode": "non_walk_forward",
        "days": len(days),
        "trades": res["trades"],
        "wins": res["wins"],
        "losses": res["losses"],
        "win_rate": res["win_rate"],
        "net_rs": res["net_rs"],
        "net_points": res["net_points"],
        "profit_factor": res["profit_factor"],
        "max_drawdown_rs": res["max_drawdown_rs"],
        "fees_rs": res["fees_rs"],
        "all_trades": res["all_trades"],
        "seconds": round(time.time() - t0, 2),
    }


def run_mode_wf(spot_all, opt_map, params, workers):
    all_days = sorted(set(opt_map) & set(spot_all))
    folds_out = []
    stitched = {"trades": 0, "net_rs": 0.0, "net_points": 0.0, "wins": 0}
    for fold in FOLDS:
        oos = [d for d in all_days if d.startswith(fold["oos_year"])]
        if not oos:
            folds_out.append({**fold, "skipped": True})
            continue
        t0 = time.time()
        res = run_f6_marny_15m_filter_backtest(params, oos, workers=workers,
                                               spot_all=spot_all, opt_map=opt_map)
        folds_out.append({
            **fold,
            "oos_days": len(oos),
            "trades": res["trades"],
            "wins": res["wins"],
            "losses": res["losses"],
            "win_rate": res["win_rate"],
            "net_rs": res["net_rs"],
            "net_points": res["net_points"],
            "profit_factor": res["profit_factor"],
            "max_drawdown_rs": res["max_drawdown_rs"],
            "fees_rs": res["fees_rs"],
            "seconds": round(time.time() - t0, 2),
        })
        stitched["trades"] += res["trades"]
        stitched["wins"] += res["wins"]
        stitched["net_rs"] += res["net_rs"]
        stitched["net_points"] += res["net_points"]
    stitched["win_rate"] = round(100.0 * stitched["wins"] / stitched["trades"], 2) if stitched["trades"] else 0.0
    return {
        "mode": "walk_forward",
        "folds": folds_out,
        "stitched_oos": stitched,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variant", choices=["A", "B", "AB"], default="AB")
    ap.add_argument("--mode", choices=["nonwf", "wf", "both"], default="both")
    ap.add_argument("--tf", action="append", choices=["1m", "2m", "3m", "5m", "all"],
                    help="Repeat per individual TF (default: single combined run)")
    ap.add_argument("--extend", action="store_true",
                    help="Merge Desktop Aug 2026 options + ammu Aug spot (run through 08-20)")
    ap.add_argument("--theta-offset", type=float, default=0.0,
                    help="Theta-decay offset in option points: SL widened & TP pulled closer by this amount")
    ap.add_argument("--time-stop", type=int, default=None,
                    help="Exit at market after N minutes if neither SL/TP hit (theta time-stop)")
    args = ap.parse_args()

    spot_all = source.load_spot()
    opt_map = source.option_day_files(START, END)
    if args.extend:
        opt_map, spot_all = extend_with_august(opt_map, spot_all)
    days = sorted(set(opt_map) & set(spot_all))
    print(f"F6 MTF 7Y | {days[0]}..{days[-1]} | {len(days)} days | workers={args.workers} "
          f"| extend={args.extend}", flush=True)

    tf_runs = args.tf if args.tf else ["all"]
    tag_ext = "xaug" if args.extend else ""
    if args.theta_offset or args.time_stop is not None:
        tag_ext += f"_to{args.theta_offset:.0f}_ts{args.time_stop or 0}"

    for tfs_label in tf_runs:
        tfs = None if tfs_label == "all" else [tfs_label]
        tag = ("all4" if tfs is None else tfs_label) + tag_ext
        for variant in ("A", "B"):
            if variant not in args.variant:
                continue
            params = params_for(variant, tfs, theta_offset=args.theta_offset,
                                time_stop_min=args.time_stop)
            print(f"\n=== TF={tfs_label} VARIANT {variant} "
                  f"(consec={params['f6_params']['consec_loss']}, "
                  f"daily_loss={params['daily_loss_pts']}) ===", flush=True)

            if args.mode in ("nonwf", "both"):
                t0 = time.time()
                out = run_mode_nonwf(spot_all, opt_map, params, args.workers)
                out["params"] = params
                path = OUT_DIR / f"f6_mtf_7y_{tag}_variant{variant}_nonwf.json"
                path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
                print(f"[non-WF] trades={out['trades']} WR={out['win_rate']}% "
                      f"net_rs={out['net_rs']:+,.2f} pts={out['net_points']:+,.2f} "
                      f"PF={out['profit_factor']} MaxDD={out['max_drawdown_rs']:,.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

            if args.mode in ("wf", "both"):
                t0 = time.time()
                out = run_mode_wf(spot_all, opt_map, params, args.workers)
                out["params"] = params
                path = OUT_DIR / f"f6_mtf_7y_{tag}_variant{variant}_wf.json"
                path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
                s = out["stitched_oos"]
                print(f"[WF] stitched: trades={s['trades']} WR={s['win_rate']}% "
                      f"net_rs={s['net_rs']:+,.2f} pts={s['net_points']:+,.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                for f in out["folds"]:
                    if not f.get("skipped"):
                        print(f"  {f['oos_year']}: trades={f['trades']} WR={f['win_rate']}% "
                              f"net_rs={f['net_rs']:+,.2f} pts={f['net_points']:+,.2f} "
                              f"PF={f['profit_factor']} ({f['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()