"""EW-OB 7-year backtest driver.

Loads index 1m spot (opt_futures_quad + Aug extension), runs the sequential
Elliott OB engine over all days, and reports trades/summary/JSON.

Usage:
   python artifacts/ew_ob/ew_ob_runner.py --smoke            # Aug 18-20 + first 5 days of 2020
  python artifacts/ew_ob/ew_ob_runner.py --full             # all days 2020-01-01..2026-08-20
  python artifacts/ew_ob/ew_ob_runner.py --full --tol 0.5 --sl-mult 3.0 --tp 60
  python artifacts/ew_ob/ew_ob_runner.py --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import opt_futures_quad as source
from artifacts.ew_ob.ew_ob_engine import (
    Bar,
    EWOBEngine,
    SESSION_START,
    SESSION_END,
)

START = "2020-01-01"
END = "2026-08-20"
OUT_DIR = Path("artifacts/ew_ob")
TFS = (1, 2, 3, 5)


def resample_tf(spot: dict, tf: int):
    """Aggregate a day's 1m spot into tf-minute bars.

    Returns (high, low, gi) arrays aligned by index; gi = global index of the
    LAST constituent 1m bar (when the TF bar is known).
    """
    mins = spot["min"]
    if tf == 1:
        return spot["high"], spot["low"], np.arange(len(mins))
    buckets = mins // tf
    highs, lows, gis = [], [], []
    prev = -1
    for i, b in enumerate(buckets):
        if b != prev:
            highs.append(spot["high"][i])
            lows.append(spot["low"][i])
            gis.append(i)
            prev = b
        else:
            highs[-1] = max(highs[-1], spot["high"][i])
            lows[-1] = min(lows[-1], spot["low"][i])
            gis[-1] = i
    return np.array(highs), np.array(lows), np.array(gis)


def resample_ohlc_tf(spot: dict, tf: int):
    """Aggregate regular-session OHLC bars for a wave detector.

    Bucket alignment resets for each day because this function receives one
    day's spot data. The detector that consumes the returned bars is kept
    alive by the engine, so wave state still carries across sessions.
    """
    mins = spot["min"]
    if tf == 1:
        return (
            np.asarray(spot["open"]), np.asarray(spot["high"]),
            np.asarray(spot["low"]), np.asarray(spot["close"]),
            np.arange(len(mins)), np.asarray(mins),
        )

    buckets = mins // tf
    opens, highs, lows, closes, gis, start_mins = [], [], [], [], [], []
    previous_bucket = None
    for i, bucket in enumerate(buckets):
        if bucket != previous_bucket:
            opens.append(spot["open"][i])
            highs.append(spot["high"][i])
            lows.append(spot["low"][i])
            closes.append(spot["close"][i])
            gis.append(i)
            start_mins.append(mins[i])
            previous_bucket = bucket
        else:
            highs[-1] = max(highs[-1], spot["high"][i])
            lows[-1] = min(lows[-1], spot["low"][i])
            closes[-1] = spot["close"][i]
            gis[-1] = i
    return (
        np.asarray(opens), np.asarray(highs), np.asarray(lows),
        np.asarray(closes), np.asarray(gis), np.asarray(start_mins),
    )


def make_option_resolver(opt_map):
    """Resolver for the engine with an optional fixed option strike."""
    day_cache = {}

    def resolve(day, side, minute, spot_px, strike=None):
        path = opt_map.get(day)
        if path is None:
            return None
        rec = day_cache.get(day)
        if rec is None:
            rec = source.cached_option(str(path))
            day_cache[day] = rec
        if rec is None:
            return None
        df, groups, prefix = rec
        if prefix is None:
            return None
        if strike is None:
            atm = int(round(spot_px / 50) * 50)
            strike = atm + (source.CE_OFFSET if side == "CE" else source.PE_OFFSET)
        sym = f"{prefix}{strike}{side}"
        sl = source.make_slice(df, groups, sym)
        if sl is None:
            return None
        idx = int(np.searchsorted(sl["times"], minute, side="right")) - 1
        if idx < 0:
            return None
        return float(sl["close"][idx])

    return resolve


def _prepare_day_bundle(job):
    """Prepare independent day data in a worker; state feed stays sequential."""
    day, spot, day_start_gi = job
    ob_data = {}
    wave_bars_by_gi = {}
    for tf in TFS:
        highs, lows, gis = resample_tf(spot, tf)
        ob_data[tf] = (highs, lows, gis + day_start_gi)
        opens, highs, lows, closes, ends, start_mins = resample_ohlc_tf(spot, tf)
        for i in range(len(ends)):
            end_gi = int(ends[i]) + day_start_gi
            wave_bars_by_gi.setdefault(end_gi, []).append((
                tf,
                Bar(
                    gi=end_gi,
                    day=day,
                    minute=int(start_mins[i]),
                    open=float(opens[i]),
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                ),
            ))
    return day, day_start_gi, ob_data, wave_bars_by_gi


def run_engine(spot_all, opt_map, days, tol, sl_mult, tp_pts,
               risk_mode="ob_w5", tp_atr_mult=None, workers=1,
               progress=False, option_sl_pts=12.0, option_tp_pts=36.0):
    eng = EWOBEngine(
        tol=tol,
        sl_mult=sl_mult,
        tp_pts=tp_pts,
        risk_mode=risk_mode,
        tp_atr_mult=tp_atr_mult,
        option_sl_pts=option_sl_pts,
        option_tp_pts=option_tp_pts,
    )
    eng.opt_map = opt_map
    eng.resolve_option = make_option_resolver(opt_map)
    jobs = []
    cursor = 0
    for day in days:
        spot = spot_all.get(day)
        if spot is None:
            continue
        jobs.append((day, spot, cursor))
        cursor += sum(1 for minute in spot["min"]
                      if SESSION_START <= int(minute) <= SESSION_END)

    if workers > 1 and len(jobs) > 1:
        if progress:
            print(f"DAY_PREP start workers={workers} days={len(jobs)}", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            bundles = pool.map(_prepare_day_bundle, jobs)
            bundles = list(bundles)
        if progress:
            print("DAY_PREP complete", flush=True)
    else:
        bundles = (_prepare_day_bundle(job) for job in jobs)

    for day_index, (day, day_start_gi, ob_data, wave_bars_by_gi) in enumerate(bundles, 1):
        if progress and (day_index == 1 or day_index % 100 == 0):
            print(f"STATE_FEED day={day_index}/{len(jobs)} date={day}", flush=True)
        spot = spot_all[day]
        for tf, (highs, lows, gis) in ob_data.items():
            eng.obs.feed_tf_bars(tf, highs, lows, gis)
        gi = day_start_gi
        for i in range(len(spot["min"])):
            minute = int(spot["min"][i])
            if minute < SESSION_START or minute > SESSION_END:
                continue
            b = Bar(gi=gi, day=day, minute=minute,
                    open=float(spot["open"][i]), high=float(spot["high"][i]),
                    low=float(spot["low"][i]), close=float(spot["close"][i]))
            eng.feed(b, wave_bars=wave_bars_by_gi.get(gi, []))
            gi += 1
        eng.close_day()
    return eng.trades


def summarize(trades):
    n = len(trades)
    if n == 0:
        return {"trades": 0, "wr": 0.0, "pts": 0.0, "rs": 0.0, "pf": 0.0, "maxdd": 0.0}
    wins = [t for t in trades if t["rs_net"] > 0]
    losses = [t for t in trades if t["rs_net"] <= 0]
    gross_win = sum(t["rs_net"] for t in wins)
    gross_loss = abs(sum(t["rs_net"] for t in losses))
    cum = 0.0
    peak = 0.0
    maxdd = 0.0
    for t in sorted(trades, key=lambda x: (x["date"], x["exit_min"])):
        cum += t["rs_net"]
        peak = max(peak, cum)
        maxdd = min(maxdd, cum - peak)
    return {
        "trades": n,
        "wr": len(wins) / n * 100.0,
        "pts": round(sum(t["pts_net"] for t in trades), 2),
        "rs": round(sum(t["rs_net"] for t in trades), 2),
        "pf": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "maxdd": round(maxdd, 2),
    }


def print_summary(label, st):
    print(f"{label}: Trades {st['trades']} | WR {st['wr']:.1f}% | "
          f"Net {st['pts']:+,.2f} pts / Rs.{st['rs']:+,.2f} | PF {st['pf']:.2f} | MaxDD Rs.{st['maxdd']:,.2f}")


def minute_label(minute):
    if minute is None:
        return "--:--"
    return f"{int(minute) // 60:02d}:{int(minute) % 60:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Aug 18-20 + first 5 days of 2020")
    ap.add_argument("--full", action="store_true", help="all days")
    ap.add_argument("--sweep", action="store_true", help="grid tol x sl-mult x tp")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--sl-mult", dest="sl_mult", type=float, default=3.0)
    ap.add_argument("--tp", dest="tp_pts", type=float, default=60.0)
    ap.add_argument("--risk-mode", choices=("ob_w5", "ob_same_tf", "atr", "option_fixed", "fib"), default="ob_w5")
    ap.add_argument("--tp-atr-mult", type=float, default=None)
    ap.add_argument("--option-sl", type=float, default=12.0)
    ap.add_argument("--option-tp", type=float, default=36.0)
    args = ap.parse_args()

    print("Loading data ...")
    t0 = time.time()
    spot_all = source.load_spot()
    opt_map = source.option_day_files(args.start, args.end)
    opt_map, spot_all = _augment(opt_map, spot_all)
    all_days = sorted(set(spot_all) & set(opt_map))
    days_in_range = [d for d in all_days if args.start <= d <= args.end]
    print(f"spot days {len(spot_all)} | option days {len(opt_map)} | in-range {len(days_in_range)}")
    print(f"load time {time.time()-t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        anchor = [d for d in days_in_range if d >= "2026-08-18"][:3]
        ref = [d for d in days_in_range if d.startswith("2020")][:5]
        for label, days in (("SMOKE Aug 18-20", anchor), ("REFERENCE 2020-01-06", ref)):
            trades = run_engine(
                spot_all, opt_map, days, args.tol, args.sl_mult, args.tp_pts,
                args.risk_mode, args.tp_atr_mult,
                option_sl_pts=args.option_sl, option_tp_pts=args.option_tp,
            )
            st = summarize(trades)
            print_summary(label, st)
            for t in trades:
                print(f"  {t['date']} {t['exit_reason']:>3} {t['side']} tf{t['timeframe']} "
                      f"zero {minute_label(t['wave_zero_minute'])} "
                      f"entry {minute_label(t['entry_min'])}@{t['entry']} "
                      f"exit {minute_label(t['exit_min'])} "
                      f"pts {t['pts_net']:+.1f} rs {t['rs_net']:+.0f}")
            print()
        return

    if args.full:
        trades = run_engine(
            spot_all, opt_map, days_in_range, args.tol, args.sl_mult, args.tp_pts,
            args.risk_mode, args.tp_atr_mult,
            option_sl_pts=args.option_sl, option_tp_pts=args.option_tp,
        )
        st = summarize(trades)
        print_summary("FULL", st)
        # per-year
        years = {}
        for t in trades:
            years.setdefault(t["date"][:4], []).append(t)
        for y in sorted(years):
            s = summarize(years[y])
            print(f"  {y}: {s['trades']} trades WR {s['wr']:.1f}% "
                  f"net Rs.{s['rs']:+,.2f} PF {s['pf']:.2f}")
        out = OUT_DIR / f"results_tol{args.tol}_sl{args.sl_mult}_tp{args.tp_pts}.json"
        with open(out, "w") as fh:
            json.dump({"params": {"tol": args.tol, "sl_mult": args.sl_mult, "tp_pts": args.tp_pts},
                       "summary": st, "trades": trades}, fh, indent=1)
        print(f"wrote {out}")
        return

    if args.sweep:
        for tol in (0.0, 0.25, 0.5, 0.75, 1.0):
            for sm in (2.0, 3.0, 4.0):
                for tp in (40.0, 60.0, 80.0):
                    trades = run_engine(
                        spot_all, opt_map, days_in_range, tol, sm, tp,
                        args.risk_mode, args.tp_atr_mult,
                        option_sl_pts=args.option_sl, option_tp_pts=args.option_tp,
                    )
                    st = summarize(trades)
                    print(f"tol={tol:<4} sl={sm:<4} tp={tp:<4} -> "
                          f"{st['trades']} trades WR {st['wr']:.1f}% "
                          f"net Rs.{st['rs']:+,.2f} PF {st['pf']:.2f}")
        return

    ap.print_help()


def _augment(opt_map, spot_all):
    """Merge Desktop August options + ammu August spot (kept local to avoid
    importing the f6 runner's heavy modules), then fix any descending order."""
    from artifacts.f6_hybrid.f6_mtf_7y_runner import extend_with_august
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    import numpy as np
    for day, sp in spot_all.items():
        if len(sp["min"]) > 1 and sp["min"][0] > sp["min"][-1]:
            idx = np.argsort(sp["min"])
            spot_all[day] = {k: v[idx] for k, v in sp.items()}
    return opt_map, spot_all


if __name__ == "__main__":
    main()
