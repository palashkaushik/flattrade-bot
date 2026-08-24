"""Fib-0.29 Stoch Wave strategy — 1-minute option backtest (2020-2026).

User spec (2026-08-19):

  - Timeframe: 1-minute option chart.
  - Primary trigger: Stochastic (9,1,3) — wave analysis on the %D line.
  - HTF filter (variant): Stochastic (60,1,10) %D line.
  - Custom tool: inverted Fibonacci anchored High->Low on the wave, 0.29 level.

  Wave identification:
    - Most recent COMPLETED bullish swing on Stoch(9,1,3) %D: a confirmed
      swing low (start of the up move) -> confirmed swing high (peak).
    - Valid only if amplitude (high - low) >= MIN_AMP (default 20 on the
      0-100 scale — user selected "minimum amplitude required").
    - Fib anchors: 1.0 at the swing high (top), 0.0 at the swing low (bottom)
      — the tool is inverted, so the 0.29 level sits near the top:
          level = high - 0.29 * (high - low)

  Entry trigger (both touches on %D):
    - Touch 1: %D pulls back and dips to/below the 0.29 level.
    - Touch 2: %D crosses back UP through the 0.29 level -> enter at that
      bar's close.

  Directional filter (VARIANT — one of):
    - 5m authority: the 5m bias chart (HA + UT Bot + LinReg white line) —
      green + close>WL -> CE only, red + close<WL -> PE only, else no entry.
    - 60stoc authority: option Stoch(60,1,10) %D — below 70 -> CE only,
      above 70 -> PE only, == 70 -> no entry.

  Risk:
    - 2nd ITM strike (CE atm-100 / PE atm+100). One position at a time.
    - SL 10 / TP 15 pts for CE (1:1.5), SL 10 / TP 10 pts for PE (1:1),
      on option premium; SL priority on same-bar touches.
    - No new trades after 15:00; EOD exit at the 15:00 bar close.
    - 4-consecutive-loss daily shutdown.
    - No re-anchoring mid-trade: the fib is frozen while a position is open;
      while flat the reference wave is always the most recent completed
      bullish swing (re-anchored automatically when a new wave completes).

Usage:
  python fib029_stoch_backtest.py --smoke --variant 5m
  python fib029_stoch_backtest.py --full --variant 60stoc
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import (
    option_files, SYM_RE, latest_spot, summarize,
    LOT_SIZE,
)
from flattrade_bot.strategies.pocket_money import OptionTracker  # live TV-verified stochastic engine
from artifacts.f6_hybrid.pocket_money_backtest import (
    build_index_filter,
    filter_allows,
    SESSION_START, SESSION_END, DAY_LAST,
    CONSEC_LOSS_LIMIT, TRACK_WINDOW_LOW, TRACK_WINDOW_HIGH, FEES_PER_TRADE,
)

FIB_LEVEL = 0.29
MIN_AMP = 20.0
CE_SL, CE_TP = 10.0, 15.0
PE_SL, PE_TP = 10.0, 10.0
D60_CE_BELOW, D60_PE_ABOVE = 70.0, 70.0
WORKERS = 8


class FibWaveTracker:
    """Bullish-wave + inverted-fib 0.29 tracker on a Stoch %D series.

    Turns are confirmed with one bar of confirmation (peak at bar t-1 when
    D[t-2] < D[t-1] and D[t] < D[t-1]; trough mirror-image). A bullish wave =
    last confirmed swing low -> next confirmed swing high with amplitude
    >= MIN_AMP. The 0.29 level is set when the wave completes; while flat the
    newest completed valid wave replaces the level. After the level is set,
    touch-1 is D <= level; the trigger is the first cross back UP through the
    level (D[t-1] <= level and D[t] > level) — checked on the bars AFTER the
    swing high (the high itself is always above the level).
    """

    def __init__(self, fib=FIB_LEVEL, min_amp=MIN_AMP):
        self.fib = fib
        self.min_amp = min_amp
        self.ds = []            # %D values (index = bar within the day)
        self.level = None       # active 0.29 level (None until a wave completes)
        self.pulled_below = False

    def push(self, d) -> bool:
        """Feed one bar's %D value. Returns True = trigger fired this bar."""
        self.ds.append(d)
        n = len(self.ds)
        if n < 3:
            return False
        d0, d1, d2 = self.ds[n - 3], self.ds[n - 2], self.ds[n - 1]
        if d0 < d1 and d2 < d1:
            self._swing_high(n - 2, d1)
        elif d0 > d1 and d2 > d1:
            self._swing_low(n - 2, d1)
        if self.level is None:
            return False
        if d <= self.level:
            self.pulled_below = True
        if self.pulled_below and self.ds[n - 2] <= self.level and d > self.level:
            return True
        return False

    def _swing_low(self, idx, value):
        self.last_low_idx = idx
        self.last_low_d = value

    def _swing_high(self, idx, value):
        high, low = value, getattr(self, "last_low_d", None)
        if low is None:
            return
        amp = high - low
        if amp >= self.min_amp:
            self.level = high - self.fib * amp
            self.pulled_below = False


def process_day(args):
    day, fpath, fprev, variant = args
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
    target_strikes = set(range(atm0 + TRACK_WINDOW_LOW, atm0 + TRACK_WINDOW_HIGH + 1, 50))

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
    waves = {}
    for sym, g in gp.items():
        t = OptionTracker()
        for i in range(len(g["min"])):
            t.push(g["high"][i], g["low"][i], g["close"][i])
        trk[sym] = t
        waves[sym] = FibWaveTracker()

    ifilter = build_index_filter(spot, day=day) if variant == "5m" else None
    trig = {}
    slices = {}
    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = OptionTracker()
            waves[sym] = FibWaveTracker()
        t = trk[sym]
        slices[sym] = g
        mm2 = SYM_RE.match(sym)
        if not mm2:
            continue
        sv, side = int(mm2.group(2)), mm2.group(3)
        w = waves[sym]
        for i in range(len(g["min"])):
            m = g["min"][i]
            s1, s2, s3, s4, prev = t.push(g["high"][i], g["low"][i], g["close"][i])
            if s1 is None:
                continue
            if w.push(s1):
                trig.setdefault(m, []).append((side, sv, sym, s4))

    def bslice(sl, m):
        idx = np_search(sl["min"], m)
        if idx is not None:
            return sl["open"][idx], sl["high"][idx], sl["low"][idx], sl["close"][idx]
        return None

    def ainfo(side, m):
        spx = latest_spot(spot, m)
        if spx is None:
            return None
        atm = int(round(spx / 50) * 50)
        stk = atm + (-100 if side == "CE" else 100)
        sym = f"{prefix}{stk}{side}"
        sl = slices.get(sym)
        return (sym, sl, stk) if sl is not None else None

    trades = []
    pos = None
    closs = 0
    shut = False
    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c)
                pos["duration_min"] += 1
                ex, rsn = None, ""
                if h >= pos["tgt"] and l <= pos["sl"]:
                    ex, rsn = pos["sl"], "SL"
                elif h >= pos["tgt"]:
                    ex, rsn = pos["tgt"], "TP"
                elif l <= pos["sl"]:
                    ex, rsn = pos["sl"], "SL"
                if ex is not None:
                    pts = round(ex - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                                   "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                                   "exit": ex, "pts": pts, "rs": round(pts * LOT_SIZE),
                                   "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                                   "reason": rsn, "duration_min": pos["duration_min"],
                                   "signal": pos["signal"]})
                    closs = closs + 1 if pts <= 0 else 0
                    if closs >= CONSEC_LOSS_LIMIT:
                        shut = True
                    pos = None
        if minute >= SESSION_END and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                           "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                           "exit": pos["last_px"], "pts": pts, "rs": round(pts * LOT_SIZE),
                           "sl_pts": pos["sl_pts"], "tp_pts": pos["tp_pts"],
                           "reason": "EOD", "duration_min": pos["duration_min"],
                           "signal": pos["signal"]})
            pos = None
            break
        if pos is not None or shut or minute >= SESSION_END:
            continue

        for (sig_side, sig_stk, sig_sym, s4) in trig.get(minute, []):
            if variant == "5m":
                allowed = filter_allows(ifilter, minute)
            else:
                if s4 is None:
                    continue
                allowed = "CE" if s4 < D60_CE_BELOW else ("PE" if s4 > D60_PE_ABOVE else None)
            if allowed is None or sig_side != allowed:
                continue
            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                bar = bslice(ai[1], minute)
                if bar:
                    ep = float(bar[3])
                    sl_use, tp_use = (CE_SL, CE_TP) if sig_side == "CE" else (PE_SL, PE_TP)
                    sl, tgt = ep - sl_use, ep + tp_use
                    pos = {"side": sig_side, "symbol": ai[0], "slice": ai[1], "entry": ep,
                           "sl": sl, "tgt": tgt, "sl_pts": sl_use, "tp_pts": tp_use,
                           "entry_min": minute, "last_px": ep, "duration_min": 0,
                           "signal": "fib029"}
                    break
    return trades


def np_search(arr, m):
    import numpy as np
    idx = np.searchsorted(arr, m)
    if idx < len(arr) and arr[idx] == m:
        return idx
    return None


def yearly_breakdown(trades):
    years = {}
    for t in trades:
        years.setdefault(t["date"][:4], []).append(t)
    return years


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-05-05")
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--variant", choices=["5m", "60stoc"], default="5m")
    ap.add_argument("--min-amp", type=float, default=MIN_AMP,
                    help="minimum wave amplitude on the 0-100 %D scale (default 20)")
    ap.add_argument("--fib", type=float, default=FIB_LEVEL, help="inverted fib level (default 0.29)")
    args = ap.parse_args()

    t0 = time.time()
    spot_all = grid.load_spot() if hasattr(grid, "load_spot") else None
    from artifacts.f6_hybrid.pocket_money_backtest import load_spot_ohlc
    spot_all = load_spot_ohlc()
    if args.smoke and args.start == "2020-01-01":
        start, end = "2023-06-12", "2023-06-16"
    else:
        start, end = args.start, args.end
    files = option_files(start, end)
    days = sorted(files)
    if args.days:
        days = days[: args.days]
    print(f"days: {len(days)}  files: {len(files)}  ({(time.time() - t0):.1f}s load)")

    pool = Pool(WORKERS, initializer=grid.init_worker_local, initargs=(spot_all,))
    tasks = [(day, str(files[day]), str(files[days[i - 1]]) if i > 0 else "", args.variant)
             for i, day in enumerate(days)]
    trades = []
    for res in pool.map(process_day, tasks):
        trades.extend(res)
    pool.close()
    pool.join()

    st = summarize(trades)
    print(f"\n=== FIB-0.29 ({args.variant} authority) ===  {start} -> {end}")
    print(f"trades: {st['trades']}  WR: {st['wr']:.1f}%  net: {st['pts']:+,.2f} pts  "
          f"rs: {st['rs']:+,d}  PF: {st['pf']:.2f}")
    fees = FEES_PER_TRADE * st["trades"]
    print(f"after fees (Rs {FEES_PER_TRADE}/trade): {st['rs'] - fees:+,d}  (fees {fees:,d})")
    if st["trades"]:
        wins = [t for t in trades if t["pts"] > 0]
        losses = [t for t in trades if t["pts"] <= 0]
        aw = sum(t["pts"] for t in wins) / len(wins) if wins else 0.0
        al = sum(t["pts"] for t in losses) / len(losses) if losses else 0.0
        print(f"avgWin {aw:+.2f}  avgLoss {al:+.2f}  "
              f"avgDuration {sum(t['duration_min'] for t in trades)/len(trades):.1f}m")
        by_reason = {}
        for t in trades:
            by_reason[t["reason"]] = by_reason.get(t["reason"], 0) + 1
        print("exits:", ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items())))
        by_side = {}
        for t in trades:
            by_side[t["side"]] = by_side.get(t["side"], 0) + 1
        print("sides:", ", ".join(f"{k}={v}" for k, v in sorted(by_side.items())))
        for y, ts in sorted(yearly_breakdown(trades).items()):
            s = summarize(ts)
            print(f"  {y}: trades {s['trades']:5d}  WR {s['wr']:5.1f}%  "
                  f"net {s['pts']:+10.1f} pts  rs {s['rs']:+12,d}  PF {s['pf']:5.2f}")
    print(f"elapsed {(time.time() - t0):.1f}s")

    out = Path(__file__).parent / f"fib029_stoch_backtest_{args.variant}.json"
    if out.exists():
        print(f"SKIP write: {out} exists")
        return
    payload = {
        "strategy": f"Fib-0.29 Stoch Wave ({args.variant} authority)",
        "start": start, "end": end, "n_days": len(days),
        "params": {
            "trigger": "Stoch(9,1,3) %D wave + inverted fib 0.29 cross-up",
            "wave": f"swing low->high, min amplitude {args.min_amp} on 0-100 scale",
            "fib_level": args.fib,
            "filter": ("5m bias chart HA+UT green>WL -> CE / red<WL -> PE"
                       if args.variant == "5m" else
                       "option Stoch(60,1,10) %D <70 -> CE / >70 -> PE"),
            "entry": "close of the %D-crosses-0.29 bar (2nd touch)",
            "sl_tp": "CE 10/15, PE 10/10 premium pts (SL priority)",
            "strike": "CE atm-100 / PE atm+100 (2nd ITM)",
            "consec_loss": CONSEC_LOSS_LIMIT,
        },
        "stats": st,
        "fees_per_trade": FEES_PER_TRADE,
        "net_rs_after_fees": st["rs"] - fees,
        "yearly": {y: summarize(ts) for y, ts in sorted(yearly_breakdown(trades).items())},
        "trades": trades,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()