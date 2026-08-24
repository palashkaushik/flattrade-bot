"""Test: F6 champion WITHOUT divergence filter.

Only change vs the ATR F6 champion:
  - Exit when S3 (slow stochastic 40,4 %D) touches the upper limit 80 -> close.
  - Stop loss = fixed 12 points in option premium (no TP).

Engine classes are reused from grid_optimize_f6_atr so the god-node file is
untouched. S3 is captured per (symbol, minute) via an independent 1m stochastic
pushed with the same candles the tracker consumes.

Usage:
  python test_f6_champion_s3exit.py --smoke      # 5-day sanity
  python test_f6_champion_s3exit.py               # full 2020-2024
"""

import argparse
import time
from multiprocessing import Pool

import numpy as np
import grid_optimize_f6_atr as G
from grid_optimize_f6_atr import (
    TF_SPECS, LOT_SIZE, CE_OFFSET, PE_OFFSET,
    SESSION_START, SESSION_END, DAY_LAST, DAILY_LOSS_RS,
    cached_day, init_worker_local, load_spot, option_files, SYM_RE,
    latest_spot, summarize, print_yearly_breakdown,
    IncrementalATR, ParamStoch, TFTracker, FlagNoDivScanner, MTFTracker,
)
from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.indicators.stochastic import IncrementalStochastic

S3_EXIT_LEVEL = 80.0      # S3 %D upper limit
FIXED_SL_PTS = 12.0       # fixed option-premium stop loss

# F6 champion params, divergence filter disabled.
CHAMPION = {
    "s1_k": 9, "s1_d": 3, "s4_k": 60,
    "atr_period": 14, "atr_sl_mult": 2.0, "atr_tp_mult": 4.0,
    "f6_s4_thresh": 79.5, "f6_s1_thresh": 20.5, "consec_loss": 6,
    "use_divergence": False,
}

WORKERS = 8


def process_day(args):
    day, fpath, fprev, p = args
    spot = G.GLOBAL_SPOT.get(day)
    if spot is None or not fpath:
        return []
    gc = cached_day(fpath)
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
        dp = cached_day(fprev)
        if dp:
            gp = filtered(dp)

    trk = {}
    s3track = {}
    for sym, g in gp.items():
        trk[sym] = MTFTracker(p)
        s3track.setdefault(sym, IncrementalStochastic(40, 4))
        for i in range(len(g["min"])):
            c = Candle(open=g["open"][i], high=g["high"][i],
                       low=g["low"][i], close=g["close"][i], minute=g["min"][i])
            trk[sym].push_1m(c)
            s3track[sym].push(g["high"][i], g["low"][i], g["close"][i])

    pmtrig = {}
    s3_by_min = {}
    slices = {}
    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = MTFTracker(p)
        t = trk[sym]
        slices[sym] = g
        mm2 = SYM_RE.match(sym)
        if not mm2:
            continue
        sv, side = int(mm2.group(2)), mm2.group(3)
        s3track.setdefault(sym, IncrementalStochastic(40, 4))
        for i in range(len(g["min"])):
            m = g["min"][i]
            c = Candle(open=g["open"][i], high=g["high"][i],
                       low=g["low"][i], close=g["close"][i], minute=m)
            for (tf, is_rev, stype, px, atr_val) in t.push_1m(c):
                pmtrig.setdefault(m, []).append(
                    (side, sv, sym, px, is_rev, tf, TF_SPECS[tf][2], TF_SPECS[tf][3], atr_val))
            s3v = s3track[sym].push(g["high"][i], g["low"][i], g["close"][i])
            s3_by_min.setdefault(sym, {})[m] = s3v

    daily_loss_pts = DAILY_LOSS_RS / LOT_SIZE
    consec_loss = p["consec_loss"]

    def bslice(sl, m):
        idx = int(np.searchsorted(sl["min"], m))
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
                if dpnl * LOT_SIZE + (c - pos["entry"]) * LOT_SIZE <= DAILY_LOSS_RS:
                    pts = round(c - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                                   "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                                   "exit": c, "pts": pts, "rs": round(pts * LOT_SIZE),
                                   "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                                   "reason": "SHUTDOWN_LOSS", "duration_min": pos["duration_min"], "tf": pos["tf"]})
                    dpnl += pts
                    pos = None
                    shut = True
                    continue
                ex, rsn = None, ""
                if l <= pos["sl"]:
                    ex, rsn = pos["sl"], "SL"
                if ex is None:
                    t1 = trk.get(pos["symbol"])
                    if t1:
                        t1m = t1.trackers["1m"]
                        t1m.div.update(c, t1m.prev_s1, low_price=l, high_price=h)
                        if t1m.div.has_bearish_peak_divergence():
                            ex, rsn = c, "BEARISH_PEAK_REVERSAL"
                if ex is None:
                    s3v = s3_by_min.get(pos["symbol"], {}).get(minute)
                    if s3v is not None and s3v >= S3_EXIT_LEVEL:
                        ex, rsn = c, "S3_80"
                if ex is not None:
                    pts = round(ex - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                                   "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                                   "exit": ex, "pts": pts, "rs": round(pts * LOT_SIZE), "reason": rsn,
                                   "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                                   "duration_min": pos["duration_min"], "tf": pos["tf"]})
                    dpnl += pts
                    closs = closs + 1 if pts <= 0 else 0
                    if closs >= consec_loss or dpnl <= daily_loss_pts:
                        shut = True
                    pos = None
        if minute >= SESSION_END and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                           "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                           "exit": pos["last_px"], "pts": pts, "rs": round(pts * LOT_SIZE),
                           "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                           "reason": "EOD", "duration_min": pos["duration_min"], "tf": pos["tf"]})
            dpnl += pts
            pos = None
            break
        if pos is not None or shut or minute >= SESSION_END:
            continue

        for (sig_side, sig_stk, sig_sym, c_px, is_rev, tf, sl_pts, tp_pts, atr_val) in pmtrig.get(minute, []):
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
                    sl_use = FIXED_SL_PTS
                    pos = {"side": as2, "symbol": asym, "slice": asl, "entry": ep,
                           "sl": ep - sl_use, "tgt": None, "sl_pts": round(sl_use, 2), "tp_pts": None,
                           "entry_min": minute, "last_px": ep, "duration_min": 0, "tf": tf}
                    break
    return trades


def run_days(pool, params, days, files, spot_all):
    tasks = [(day, str(files[day]), str(files[days[i - 1]]) if i > 0 else "", params)
             for i, day in enumerate(days)]
    all_trades = []
    for res in pool.map(process_day, tasks):
        all_trades.extend(res)
    return all_trades


def stats_for(trades, start_year=None, end_year=None):
    if start_year is not None or end_year is not None:
        trades = [t for t in trades
                  if (start_year is None or int(t["date"][:4]) >= start_year)
                  and (end_year is None or int(t["date"][:4]) <= end_year)]
    return summarize(trades), trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()

    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    print(f"CONFIG: F6 champion | divergence filter OFF | exit S3(40,4) %D >= {S3_EXIT_LEVEL} "
          f"-> close | SL = {FIXED_SL_PTS} option pts (no TP)", flush=True)
    print(f"CHAMPION params: {CHAMPION}", flush=True)

    if args.smoke:
        days5 = days[:5]
        print(f"=== SMOKE TEST — {len(days5)} DAYS ({days5[0]}..{days5[-1]}) ===", flush=True)
        with Pool(processes=WORKERS, initializer=init_worker_local, initargs=(spot_all,)) as pool:
            trades = run_days(pool, CHAMPION, days5, files, spot_all)
        st, _ = stats_for(trades)
        print(f"Trades: {st['trades']} | WR: {st['wr']:.1f}% | Net Rs: {st['rs']:+,d} | PF: {st['pf']:.2f}")
        print("SMOKE TEST OK" if 15 <= st["trades"] <= 40 else "SMOKE TEST SUSPICIOUS")
        return

    with Pool(processes=WORKERS, initializer=init_worker_local, initargs=(spot_all,)) as pool:
        t0 = time.time()
        trades = run_days(pool, CHAMPION, days, files, spot_all)
    st, _ = stats_for(trades)
    print(f"\n=== FULL RUN ({days[0]}..{days[-1]}, {len(days)} days) ===", flush=True)
    print(f"Trades: {st['trades']:,d} | WR: {st['wr']:.1f}% | Net Rs: {st['rs']:+,d} | PF: {st['pf']:.2f} "
          f"| Max DD Rs: {st.get('max_dd', 0):,.0f} | time {time.time()-t0:.0f}s", flush=True)
    print_yearly_breakdown(trades)


if __name__ == "__main__":
    main()
