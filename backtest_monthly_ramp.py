"""Monthly Lot-Ramp Backtest — Optuna Optimized ATR F6 (Champion) + equity-based sizing.

Money-management rules (user-specified):
  - Lot size is constant for the WHOLE month; re-evaluated only at month boundaries.
  - lots = max(1, floor(equity_at_month_start / 40,000))  -> +1 lot per +Rs 40K portfolio.
  - Daily Max Loss = Rs 2,000 / 65 = 30.77 points -> scale-free daily shutdown.
  - Consecutive Loss Stop = 8 (champion CL).
  - Daily Max Profit = UNLIMITED (champion).
  - Starting capital ~Rs 13-27K → 1 lot.

Speed architecture (inherited from grid_optimize_f6_atr.py):
  - ONE persistent multiprocessing Pool; workers cache parsed day files
    (keyed by path, bounded) -> per-day I/O amortized to ~0.
  - Incremental stochastic/ATR/divergence, numpy searchsorted pointer filtering.
  - The engine emits per-lot POINTS (shutdown is point-based), so the trade list
    is lot-size independent: run once, then apply the monthly lot schedule as a
    O(months) post-pass. No per-lot re-simulation.

Usage:
  python backtest_monthly_ramp.py --smoke          # 5-day sanity (AGENTS mandate)
  python backtest_monthly_ramp.py                  # full 5Y (2020-2024)
  python backtest_monthly_ramp.py --start 2023-01-01 --end 2024-12-31
  python backtest_monthly_ramp.py --capital 20000 --increment 40000
"""

import argparse
import json
import sys
import time
from math import floor
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import (
    load_spot, option_files, SYM_RE, to_minutes, latest_spot, summarize,
)

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_LOSS_PTS = -grid.DAILY_LOSS_RS / LOT_SIZE  # shared Rs2,000 reference rule
MARGIN_PER_LOT = 13268.0       # NIFTY option margin from live fire-test rejection
WORKERS = 8

CHAMPION_PARAMS = {
    "s1_k": 12, "s1_d": 3, "s4_k": 50,
    "atr_period": 10, "atr_sl_mult": 3.0, "atr_tp_mult": 6.0,
    "f6_s4_thresh": 79.5, "f6_s1_thresh": 20.5, "consec_loss": 8,
}
ACTIVE_PARAMS = dict(CHAMPION_PARAMS)


def init_worker(spot_dict, params=None):
    """Share spot data + worker-local parse cache via the grid module globals."""
    global ACTIVE_PARAMS
    grid.GLOBAL_SPOT = spot_dict
    grid.GLOBAL_CACHE = {}
    ACTIVE_PARAMS = dict(params or CHAMPION_PARAMS)


def process_day(args):
    """Fork of grid.process_day with point-based daily shutdown.

    All exits/shutdowns are computed in per-lot POINTS, so the returned trade
    list is valid for ANY lot size; the monthly ramp multiplies afterwards.
    """
    day, fpath, fprev = args
    p = ACTIVE_PARAMS
    spot = grid.GLOBAL_SPOT.get(day)
    if spot is None or not fpath:
        return []
    gc = grid.cached_day(fpath)
    if not gc:
        return []
    fsym = next(iter(gc))
    mm = SYM_RE.match(fsym)
    if not mm:
        return []
    prefix = mm.group(1)
    sp0 = latest_spot(spot, 555) or latest_spot(spot, 560)
    if sp0 is None:
        return []
    atm0 = int(round(sp0 / 50) * 50)
    target_strikes = set(range(atm0 - 250, atm0 + 300, 50))

    def filtered(data):
        return {sym: g for sym, g in data.items()
                if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}

    gu = filtered(gc)
    gp = {}
    if fprev:
        dp = grid.cached_day(fprev)
        if dp:
            gp = filtered(dp)

    trk = {}
    for sym, g in gp.items():
        trk[sym] = grid.MTFTracker(p)
        for i in range(len(g["min"])):
            trk[sym].push_1m(grid.Candle(open=g["open"][i], high=g["high"][i],
                                         low=g["low"][i], close=g["close"][i],
                                         minute=g["min"][i]))

    pmtrig = {}
    slices = {}
    allowed_timeframes = resolve_timeframes(p)
    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = grid.MTFTracker(p)
        t = trk[sym]
        slices[sym] = g
        mm2 = SYM_RE.match(sym)
        if not mm2:
            continue
        sv, side = int(mm2.group(2)), mm2.group(3)
        for i in range(len(g["min"])):
            m = g["min"][i]
            for (tf, is_rev, stype, px, atr_val) in t.push_1m(
                    grid.Candle(open=g["open"][i], high=g["high"][i],
                                low=g["low"][i], close=g["close"][i], minute=m)):
                if tf not in allowed_timeframes:
                    continue
                pmtrig.setdefault(m, []).append(
                    (side, sv, sym, px, is_rev, tf,
                     grid.TF_SPECS[tf][2], grid.TF_SPECS[tf][3], atr_val))

    consec_loss = p["consec_loss"]
    sl_mult, tp_mult = p["atr_sl_mult"], p["atr_tp_mult"]

    def bslice(sl, m):
        idx = np.searchsorted(sl["min"], m)
        if idx < len(sl["min"]) and sl["min"][idx] == m:
            return sl["open"][idx], sl["high"][idx], sl["low"][idx], sl["close"][idx]
        return None

    def ainfo(side, m):
        spx = latest_spot(spot, m)
        if spx is None:
            return None
        atm = int(round(spx / 50) * 50)
        stk = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        sym = f"{prefix}{stk}{side}"
        sl = slices.get(sym)
        return (sym, sl, stk) if sl is not None else None

    trades = []
    pos = None
    dpnl = 0.0
    closs = 0
    shut = False
    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c)
                pos["duration_min"] += 1
                if dpnl + (c - pos["entry"]) <= -DAILY_LOSS_PTS:
                    pts = round(c - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"],
                                   "exit_min": minute, "side": pos["side"],
                                   "symbol": pos["symbol"], "entry": pos["entry"],
                                   "exit": c, "pts": pts, "rs": round(pts * LOT_SIZE),
                                   "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                                   "reason": "SHUTDOWN_LOSS",
                                   "duration_min": pos["duration_min"], "tf": pos["tf"]})
                    dpnl += pts
                    pos = None
                    shut = True
                    continue
                ex, rsn = None, ""
                has_tgt = pos.get("tgt") is not None
                if has_tgt and h >= pos["tgt"] and l <= pos["sl"]:
                    ex, rsn = pos["sl"], "SL"
                elif has_tgt and h >= pos["tgt"]:
                    ex, rsn = pos["tgt"], "TP"
                elif l <= pos["sl"]:
                    ex, rsn = pos["sl"], "SL"
                if ex is None:
                    t1 = trk.get(pos["symbol"])
                    if t1:
                        t1m = t1.trackers["1m"]
                        t1m.div.update(c, t1m.prev_s1, low_price=l, high_price=h)
                        if t1m.div.has_bearish_peak_divergence():
                            ex, rsn = c, "BEARISH_PEAK_REVERSAL"
                if ex is not None:
                    pts = round(ex - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"],
                                   "exit_min": minute, "side": pos["side"],
                                   "symbol": pos["symbol"], "entry": pos["entry"],
                                   "exit": ex, "pts": pts, "rs": round(pts * LOT_SIZE),
                                   "reason": rsn, "sl_pts": pos["sl_pts"],
                                   "tp_pts": pos["tp_pts"],
                                   "duration_min": pos["duration_min"], "tf": pos["tf"]})
                    dpnl += pts
                    closs = closs + 1 if pts <= 0 else 0
                    if closs >= consec_loss or dpnl <= -DAILY_LOSS_PTS:
                        shut = True
                    pos = None
        if minute >= SESSION_END and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            trades.append({"date": day, "entry_min": pos["entry_min"],
                           "exit_min": minute, "side": pos["side"],
                           "symbol": pos["symbol"], "entry": pos["entry"],
                           "exit": pos["last_px"], "pts": pts,
                           "rs": round(pts * LOT_SIZE), "sl_pts": pos["sl_pts"],
                           "tp_pts": pos["tp_pts"], "reason": "EOD",
                           "duration_min": pos["duration_min"], "tf": pos["tf"]})
            dpnl += pts
            pos = None
            break
        if pos is not None or shut or minute >= SESSION_END:
            continue

        for (sig_side, sig_stk, sig_sym, c_px, is_rev, tf,
             sl_pts, tp_pts, atr_val) in pmtrig.get(minute, []):
            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                if is_rev:
                    as2 = "PE" if sig_side == "CE" else "CE"
                    ai2 = ainfo(as2, minute)
                    if ai2 is None:
                        continue
                    asym, asl, _ = ai2
                else:
                    as2 = sig_side
                    asym = sig_sym
                    asl = ai[1]
                bar = bslice(asl, minute)
                if bar:
                    ep = float(bar[3])
                    sl_use, tp_use = resolve_exit_points(
                        atr_val,
                        sl_mult,
                        tp_mult,
                        sl_pts,
                        tp_pts,
                        p,
                    )
                    pos = {"side": as2, "symbol": asym, "slice": asl, "entry": ep,
                           "sl": ep - sl_use, "tgt": ep + tp_use,
                           "sl_pts": round(sl_use, 2), "tp_pts": round(tp_use, 2),
                           "entry_min": minute, "last_px": ep,
                           "duration_min": 0, "tf": tf}
                    break
    return trades


def resolve_exit_points(
    atr_val,
    sl_mult: float,
    tp_mult: float,
    fallback_sl: float,
    fallback_tp: float,
    params: dict,
) -> tuple[float, float]:
    """Resolves fixed exits first, then ATR exits, then timeframe fallbacks."""
    fixed_sl = params.get("fixed_sl_points")
    fixed_tp = params.get("fixed_tp_points")
    if fixed_sl is not None and fixed_tp is not None:
        return float(fixed_sl), float(fixed_tp)
    if atr_val and atr_val > 0.5:
        return atr_val * sl_mult, atr_val * tp_mult
    return float(fallback_sl), float(fallback_tp)


def resolve_timeframes(params: dict) -> set[str]:
    """Returns the execution timeframes, defaulting to the full 4-TF set."""
    requested = params.get("timeframes")
    if requested is None:
        return set(grid.TF_SPECS)
    return {str(timeframe) for timeframe in requested}


def run_days(pool, days, files):
    tasks = [(day, str(files[day]), str(files[days[i - 1]]) if i > 0 else "")
             for i, day in enumerate(days)]
    all_trades = []
    for res in pool.map(process_day, tasks):
        all_trades.extend(res)
    return all_trades


def apply_monthly_ramp(trades, start_capital, increment_rs, margin_per_lot):
    """Post-pass: constant lots per month, re-evaluated only at month boundaries.

    lots(month) = max(1, floor(equity_at_month_start / increment_rs))
    capped by usable margin (equity/3 / margin-per-lot).
    """
    if not trades:
        return [], None
    df = pd.DataFrame(trades)
    use_net = "pts_net" in df.columns and "rs_net" in df.columns
    col_pts = "pts_net" if use_net else "pts"
    col_rs = "rs_net" if use_net else None
    df["month"] = df["date"].str[:7]
    months = sorted(df["month"].unique())
    rows = []
    equity = float(start_capital)
    for m in months:
        mdf = df[df["month"] == m]
        pts_sum = float(mdf[col_pts].sum())
        lots_portfolio = max(1, floor(equity / increment_rs))
        lots_margin = floor(equity / 3.0 / margin_per_lot) if margin_per_lot else lots_portfolio
        lots = max(1, min(lots_portfolio, lots_margin))
        if col_rs is not None:
            rs_month = round(float(mdf[col_rs].sum()) * lots)
        else:
            rs_month = round(pts_sum * LOT_SIZE * lots)
        wr_month = ((mdf[col_pts] > 0).mean() * 100) if len(mdf) else 0.0
        rows.append({
            "month": m, "lots": lots, "trades": len(mdf),
            "wr": round(wr_month, 1), "pts": round(pts_sum, 2),
            "rs": rs_month, "equity": round(equity + rs_month, 2),
        })
        equity += rs_month
    return rows, df


def print_ramp_table(rows):
    if not rows:
        print("\nNo trades executed.")
        return
    print("\n" + "=" * 104)
    print(f"{'MONTH':8s} | {'LOTS':>4s} | {'TRADES':>6s} | {'WR':>5s} | "
          f"{'NET PTS':>10s} | {'PROFIT (Rs)':>13s} | {'EQUITY (Rs)':>14s}")
    print("=" * 104)
    for r in rows:
        print(f"{r['month']:8s} | {r['lots']:4d} | {r['trades']:6d} | {r['wr']:4.1f}% | "
              f"{r['pts']:+10.2f} | {r['rs']:+13,d} | {r['equity']:14,.2f}")
    print("=" * 104)


def print_yearly_ramp(rows, start_capital):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["year"] = df["month"].str[:4]
    print(f"\n{'YEAR':6s} | {'TRADES':>7s} | {'NET PTS':>10s} | {'PROFIT (Rs)':>13s} | "
          f"{'END EQUITY (Rs)':>16s}")
    print("-" * 66)
    for year, g in df.groupby("year"):
        print(f"{year:6s} | {g['trades'].sum():7d} | {g['pts'].sum():+10.2f} | "
              f"{g['rs'].sum():+13,d} | {g['equity'].iloc[-1]:16,.2f}")
    print("-" * 66)
    total_rs = df["rs"].sum()
    print(f"{'TOTAL':6s} | {df['trades'].sum():7d} | {df['pts'].sum():+10.2f} | "
          f"{total_rs:+13,d} | {df['equity'].iloc[-1]:16,.2f}")
    print(f"\nStarting capital: Rs {start_capital:,.2f}")
    print(f"Total profit:     Rs {total_rs:+,.2f}")
    print(f"Final equity:     Rs {df['equity'].iloc[-1]:,.2f}")
    print(f"ROI on start:     {total_rs / start_capital * 100:+.1f}%  (5-year cumulative)")


def main():
    ap = argparse.ArgumentParser(description="Monthly Lot-Ramp Backtest (Champion ATR F6)")
    ap.add_argument("--smoke", action="store_true", help="5-day sanity check (AGENTS mandate)")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--capital", type=float, default=20000.0,
                    help="Starting capital (default 20,000)")
    ap.add_argument("--increment", type=float, default=40000.0,
                    help="Equity increment per +1 lot (default 40,000)")
    args = ap.parse_args()

    spot_all = load_spot()
    files = option_files(args.start, args.end)
    days = sorted(set(files.keys()) & set(spot_all.keys()))
    if args.smoke:
        days = days[:5]

    label = f"SMOKE TEST — 5 DAYS" if args.smoke else f"FULL RUN — {len(days)} DAYS"
    print(f"=== {label} ({days[0]}..{days[-1]}) | CHAMPION ATR F6 | daily stop 30 pts | "
          f"ramp +1 lot / Rs {args.increment:,.0f} ===", flush=True)
    print(f"Start capital: Rs {args.capital:,.0f} | workers: {WORKERS}", flush=True)

    t0 = time.time()
    with Pool(processes=WORKERS, initializer=init_worker,
              initargs=(spot_all,)) as pool:
        trades = run_days(pool, days, files)
    elapsed = time.time() - t0

    st = summarize(trades)
    print(f"\n[OK] Engine ran {len(days)} days in {elapsed:.1f}s | trades {st['trades']} | "
          f"WR {st['wr']:.1f}% | 1-lot pts {st['pts']:+,.2f} | 1-lot Rs {st['rs']:+,d} | "
          f"PF {st['pf']:.2f}")
    if args.smoke:
        ok = 15 <= st["trades"] <= 40
        print(f"SMOKE CHECK: {st['trades']} trades (expect 15-40) "
              f"-> {'OK' if ok else 'SUSPICIOUS'}")
        sys.exit(0 if ok else 1)

    rows, _ = apply_monthly_ramp(trades, args.capital, args.increment, MARGIN_PER_LOT)
    print_ramp_table(rows)
    print_yearly_ramp(rows, args.capital)


if __name__ == "__main__":
    main()
