"""Smart Fib + VSA — 1-minute option backtest (2020-2026).

User spec (2026-08-19/20):

  - Pattern: copy the smart fib INDEX logic — UT Bot (key=1.0, ATR=10) on
    regular Nifty 50 index 1m candles.
      RGR (red -> 5+ green -> red)  = bullish  -> CE, fib high_to_low
      GRG (green -> 5+ red -> green) = bearish -> PE, fib low_to_high
  - Fib anchors: min low / max high across all pattern candles.
  - Entry trigger (2026-08-20 user directive): NO volume gate. The indicator
    only colours the index candles (UT Bot) -> pattern completion + index bar
    CLOSE within the 0.5-1.0 fib zone -> enter at that bar's option close
    (2nd ITM strike).
  - SL 1.155 / TP -0.55 fib levels checked on INDEX price (SL priority on
    same-bar touches), exit fill at option close.
  - PM session 09:20-15:00, one position at a time, no re-anchor mid-trade,
    4-consecutive-loss shutdown, EOD exit at the 15:00 bar close.

Usage:
  python smart_fib_vsa_backtest.py --smoke
  python smart_fib_vsa_backtest.py --full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_5y_optimized import (
    option_files, SYM_RE, latest_spot, summarize, LOT_SIZE,
)
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.causal_live_parity_research import IncrementalATR
from artifacts.f6_hybrid.pocket_money_backtest import (
    SESSION_START, SESSION_END, DAY_LAST,
    CONSEC_LOSS_LIMIT, TRACK_WINDOW_LOW, TRACK_WINDOW_HIGH, FEES_PER_TRADE,
)

UT_KEY = 1.0
UT_ATR_PERIOD = 10
ZONE_LO, ZONE_HI = 0.5, 1.0
SL_LEVEL = 1.155
TP_LEVEL = -0.55
VSA_SHORT, VSA_MED, VSA_LONG = 4, 20, 100
CE_OFFSET, PE_OFFSET = -100, 100
WORKERS = 8
GLOBAL_SPOT = {}
GLOBAL_CACHE = {}


def init_worker(spot):
    global GLOBAL_SPOT, GLOBAL_CACHE
    GLOBAL_SPOT = spot
    GLOBAL_CACHE = {}


def cached_option_v(path_str):
    """Per-day option cache WITH volume (grid.cached_day drops volume)."""
    c = GLOBAL_CACHE.get(path_str)
    if c is not None:
        return c
    try:
        df = pd.read_csv(path_str, usecols=["symbol", "time", "open", "high", "low", "close", "volume"],
                         engine="c")
        if df.empty:
            GLOBAL_CACHE[path_str] = None
            return None
        df["min"] = np.array([int(str(t).split(":")[0]) * 60 + int(str(t).split(":")[1])
                              for t in df["time"]])
        df = df.drop_duplicates(subset=["symbol", "min"], keep="last")
        df = df.sort_values(["symbol", "min"], kind="stable")
        data = {}
        for sym, g in df.groupby("symbol"):
            data[sym] = {
                "min": g["min"].to_numpy(),
                "open": g["open"].to_numpy(dtype=float),
                "high": g["high"].to_numpy(dtype=float),
                "low": g["low"].to_numpy(dtype=float),
                "close": g["close"].to_numpy(dtype=float),
                "volume": g["volume"].to_numpy(dtype=float),
            }
        GLOBAL_CACHE[path_str] = data
        return data
    except Exception:
        GLOBAL_CACHE[path_str] = None
        return None


class UTBotState:
    """Causal translation of the supplied UT Bot Pine v4 logic."""

    def __init__(self):
        self.atr = IncrementalATR(UT_ATR_PERIOD)
        self.trailing_stop = 0.0
        self.previous_source = None
        self.position = 0

    def update(self, candle: Candle, source_close: float | None = None) -> str:
        source_price = candle.close if source_close is None else source_close
        atr = self.atr.update(candle.high, candle.low, candle.close)
        previous_source = self.previous_source
        previous_stop = self.trailing_stop
        self.previous_source = source_price
        if atr is None or previous_source is None:
            return "blue"

        loss = UT_KEY * atr
        if source_price > previous_stop and previous_source > previous_stop:
            self.trailing_stop = max(previous_stop, source_price - loss)
        elif source_price < previous_stop and previous_source < previous_stop:
            self.trailing_stop = min(previous_stop, source_price + loss)
        elif source_price > previous_stop:
            self.trailing_stop = source_price - loss
        else:
            self.trailing_stop = source_price + loss

        if previous_source < previous_stop and source_price > previous_stop:
            self.position = 1
        elif previous_source > previous_stop and source_price < previous_stop:
            self.position = -1
        return "green" if self.position == 1 else "red" if self.position == -1 else "blue"


class FibPattern:
    """UT-color sequence detector (smart fib index specs) with fib range."""

    def __init__(self, direction="bullish", first="red", middle="green", final="red", orientation="high_to_low"):
        self.direction = direction
        self.first_color = first
        self.middle_color = middle
        self.final_color = final
        self.orientation = orientation
        self.previous_color = None
        self.previous_candle = None
        self.phase = "idle"
        self.range_high = None
        self.range_low = None
        self.middle_count = 0

    def reset_session(self):
        self.previous_color = None
        self.previous_candle = None
        self.phase = "idle"
        self.range_high = None
        self.range_low = None
        self.middle_count = 0

    def update(self, candle: Candle, color: str):
        completed_setup = None
        if self.phase == "green":
            if color == self.middle_color:
                self.middle_count += 1
                self.range_high = max(self.range_high, candle.high)
                self.range_low = min(self.range_low, candle.low)
            elif color == self.final_color:
                if self.middle_count >= 5:
                    self.range_high = max(self.range_high, candle.high)
                    self.range_low = min(self.range_low, candle.low)
                    completed_setup = (
                        self.direction,
                        self.range_high,
                        self.range_low,
                        self.orientation,
                    )
                self.phase = "idle"
                self.range_high = None
                self.range_low = None
                self.middle_count = 0
            else:
                self.phase = "idle"
                self.range_high = None
                self.range_low = None
                self.middle_count = 0
        elif (
            color == self.middle_color
            and self.previous_color == self.first_color
            and self.previous_candle is not None
        ):
            self.phase = "green"
            self.middle_count = 1
            self.range_high = max(self.previous_candle.high, candle.high)
            self.range_low = min(self.previous_candle.low, candle.low)

        self.previous_color = color
        self.previous_candle = candle
        if completed_setup is not None:
            self.setup = completed_setup
        return completed_setup


class PineVSAState:
    """Vincent Kott VSA_MS Pine translation (spec reference impl).

    Non-white (red/purple/blue) = volume == max over 4/20/100 lookbacks.
    Warm with the previous day's volume rows so long lookbacks are live at
    session open, exactly like the indicator on a chart with full history.
    """

    def __init__(self, short_lb=VSA_SHORT, med_lb=VSA_MED, long_lb=VSA_LONG):
        self.short_lb = short_lb
        self.med_lb = med_lb
        self.long_lb = long_lb
        self.history: list[float] = []

    def warmup(self, volumes):
        for v in volumes:
            self.history.append(float(v))

    def update(self, delta_vol: float) -> str:
        self.history.append(delta_vol)
        n = len(self.history)
        if delta_vol <= 0 or n < 2:
            return "white"

        h_short = max(self.history[max(0, n - self.short_lb): n])
        h_med = max(self.history[max(0, n - self.med_lb): n])
        h_long = max(self.history[max(0, n - self.long_lb): n])

        if delta_vol == h_long and n >= 20:
            return "blue"
        elif delta_vol == h_med and n >= 5:
            return "purple"
        elif delta_vol == h_short and n >= 2:
            return "red"
        else:
            return "white"


def fib_price(high, low, level, orientation="high_to_low"):
    return (
        high - level * (high - low)
        if orientation == "high_to_low"
        else low + level * (high - low)
    )


def np_search(arr, m):
    idx = np.searchsorted(arr, m)
    if idx < len(arr) and arr[idx] == m:
        return idx
    return None


def process_day(args):
    day, fpath, fprev = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not fpath:
        return []
    gu = cached_option_v(fpath)
    if not gu:
        return []
    mm = SYM_RE.match(next(iter(gu)))
    if not mm:
        return []
    prefix = mm.group(1)
    sp0 = latest_spot(spot, 555) or latest_spot(spot, 560)
    if sp0 is None:
        return []
    atm0 = int(round(sp0 / 50) * 50)
    target_strikes = set(range(atm0 + TRACK_WINDOW_LOW, atm0 + TRACK_WINDOW_HIGH + 1, 50))
    gu = {sym: g for sym, g in gu.items()
          if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}

    gp = {}
    if fprev:
        dp = cached_option_v(fprev)
        if dp:
            gp = {sym: g for sym, g in dp.items()
                  if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}

    vsa = {}
    for sym, g in gu.items():
        st = PineVSAState()
        if sym in gp:
            st.warmup(gp[sym]["volume"])
        vsa[sym] = st

    smin = spot["min"]
    soff = int(np.searchsorted(smin, SESSION_START))
    smin_day = smin[soff:]
    prev_rows = []
    prev_days = sorted(k for k in GLOBAL_SPOT if k < day)
    if prev_days:
        ps = GLOBAL_SPOT[prev_days[-1]]
        pidx = np.searchsorted(ps["min"], SESSION_START)
        prev_rows = [
            (float(ps["open"][i]), float(ps["high"][i]), float(ps["low"][i]), float(ps["close"][i]), int(ps["min"][i]))
            for i in range(pidx, len(ps["min"]))
        ]

    ut = UTBotState()
    for o, h, l, c, m in prev_rows:
        ut.update(Candle(o, h, l, c, minute=m))

    patterns = [
        FibPattern("bullish", "red", "green", "red", "high_to_low"),
        FibPattern("bearish", "green", "red", "green", "low_to_high"),
    ]
    setups = []

    def sym_at(side, m):
        spx = latest_spot(spot, m)
        if spx is None:
            return None
        atm = int(round(spx / 50) * 50)
        stk = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        return f"{prefix}{stk}{side}"

    trades = []
    pos = None
    closs = 0
    shut = False
    for i in range(len(smin_day)):
        m = int(smin_day[i])
        if m > DAY_LAST:
            break
        o, h, l, c = (float(spot["open"][soff + i]), float(spot["high"][soff + i]),
                      float(spot["low"][soff + i]), float(spot["close"][soff + i]))
        candle = Candle(o, h, l, c, minute=m)
        color = ut.update(candle)
        for pattern in patterns:
            completed = pattern.update(candle, color)
            if completed is not None:
                setups.append(list(completed))

        if pos is not None:
            pos["duration_min"] += 1
            sl_lvl = fib_price(pos["fib_high"], pos["fib_low"], SL_LEVEL, pos["orientation"])
            tp_lvl = fib_price(pos["fib_high"], pos["fib_low"], TP_LEVEL, pos["orientation"])
            if pos["side"] == "CE":
                hit_stop = l <= sl_lvl
                hit_target = h >= tp_lvl
            else:
                hit_stop = h >= sl_lvl
                hit_target = l <= tp_lvl
            reason = "SL" if hit_stop else "TP" if hit_target else None
            if reason is None and m >= SESSION_END:
                reason = "EOD"
            if reason:
                bar = None
                g = gu.get(pos["symbol"])
                if g is not None:
                    idx = np_search(g["min"], m)
                    if idx is not None:
                        bar = (g["open"][idx], g["high"][idx], g["low"][idx], g["close"][idx])
                exit_px = float(bar[3]) if bar else pos["last_px"]
                pts = round(exit_px - pos["entry"], 2)
                trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": m,
                               "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                               "exit": exit_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                               "sl_lvl": round(sl_lvl, 1), "tp_lvl": round(tp_lvl, 1),
                               "reason": reason, "duration_min": pos["duration_min"],
                               "signal": "smart_fib_vsa"})
                closs = closs + 1 if pts <= 0 else 0
                if closs >= CONSEC_LOSS_LIMIT:
                    shut = True
                pos = None
        if m >= SESSION_END and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": m,
                           "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                           "exit": pos["last_px"], "pts": pts, "rs": round(pts * LOT_SIZE),
                           "sl_lvl": 0.0, "tp_lvl": 0.0,
                           "reason": "EOD", "duration_min": pos["duration_min"],
                           "signal": "smart_fib_vsa"})
            pos = None
            break
        if pos is not None or shut or m >= SESSION_END:
            continue

        for setup in setups:
            direction, fh, fl, orient = setup
            side = "CE" if direction == "bullish" else "PE"
            z_lo = fib_price(fh, fl, ZONE_LO, orient)
            z_hi = fib_price(fh, fl, ZONE_HI, orient)
            if not (min(z_lo, z_hi) <= c <= max(z_lo, z_hi)):
                continue
            sym = sym_at(side, m)
            if sym is None or sym not in vsa:
                continue
            g = gu.get(sym)
            if g is None:
                continue
            idx = np_search(g["min"], m)
            if idx is None:
                continue
            ep = float(g["close"][idx])
            pos = {"side": side, "symbol": sym, "entry": ep, "entry_min": m,
                   "last_px": ep, "duration_min": 0,
                   "fib_high": fh, "fib_low": fl, "orientation": orient}
            setups.remove(setup)
            break
    return trades


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
    args = ap.parse_args()

    t0 = time.time()
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

    pool = Pool(WORKERS, initializer=init_worker, initargs=(spot_all,))
    tasks = [(day, str(files[day]), str(files[days[i - 1]]) if i > 0 else "")
             for i, day in enumerate(days)]
    trades = []
    for res in pool.map(process_day, tasks):
        trades.extend(res)
    pool.close()
    pool.join()

    st = summarize(trades)
    print(f"\n=== SMART FIB + VSA ===  {start} -> {end}")
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

    out = Path(__file__).parent / "smart_fib_vsa_backtest.json"
    if out.exists():
        print(f"SKIP write: {out} exists")
        return
    payload = {
        "strategy": "Smart Fib + VSA (smart fib index pattern)",
        "start": start, "end": end, "n_days": len(days),
        "params": {
            "pattern": "RGR (bullish->CE, high_to_low) / GRG (bearish->PE, low_to_high), middle>=5, on index 1m UT Bot key=1.0 ATR=10 regular candles",
            "entry": "index bar close in fib 0.5-1.0 zone, pattern-only (NO volume gate per user 2026-08-20) -> enter at option close",
            "sl_tp": "fib 1.155 / -0.55 levels on index price, SL priority",
            "strike": "CE atm-100 / PE atm+100 (2nd ITM)",
            "session": "09:20-15:00, one position, 4-consec-loss shutdown, EOD close",
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