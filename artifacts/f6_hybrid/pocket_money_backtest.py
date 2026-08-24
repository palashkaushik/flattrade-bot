"""PocketMoney — 10-second scalping strategy, 1-minute proxy backtest (2020-2026).

The historical archive's finest resolution is 1 minute, so the 10s strategy is
run on 1-minute option bars. Everything else follows the user spec exactly:

  - 4 stochastics on EVERY option chart: S1(9,3) S2(14,3) S3(40,4) S4(60,10)
  - FLAG trigger: S1 %D touches 20.5 from the neutral zone (S1 crosses from
    >20.5 down to <=20.5) while S4 %D >= 79.5 — NO divergence required.
  - SUPER trigger: S1 %D crosses back ABOVE 20 while a bullish trough
    divergence is confirmed on the contract's own chart — the current
    confirmed trough (fully formed when S1 crossed above 20) has a LOWER low
    than the previous confirmed trough while its S1 OR S2 is HIGHER. Entry on
    the close of the S1-crosses-20 bar.
  - Trade the 2nd ITM strike: CE at ATM-100, PE at ATM+100 (index used ONLY
    for strike selection). One position at a time.
  - No new trades after 15:00; EOD exit at the 15:00 bar close.
  - SL = entry - 7, TP = entry + 7 premium points (SL priority), both sides
    (long CE and long PE — premium up is favorable for either).
  - One position at a time: entry only when flat (pos is None); at most one
    position opened per minute.
  - INDEX filter (5m Heikin-Ashi + UT Bot key=1.0 period=10 + 11-bar linreg of
    HA close, computed on 1-minute index bars aggregated to 5m):
        UT green AND HA close > linreg  -> CE side only
        UT red   AND HA close < linreg  -> PE side only
  - Option divergence engine (F6-style): feeds S1+S2 so a SUPER trigger on a
    contract requires the confirmed bullish trough divergence on that
    contract's own 1m chart (current confirmed trough lower low + higher S1
    or S2, confirmed when S1 crosses back above 20) — same DivergenceEngine
    the F6 strategy uses, fed on 1m bars. FLAG entries skip the divergence
    gate entirely.
  - 4-consecutive-loss daily shutdown (no daily -2000 cap).
  - No pin bars, no ATR exits, no reversal mode.

Usage:
  python pocket_money_backtest.py --smoke      # 5-day sanity check
  python pocket_money_backtest.py --full       # 2020-01-01 .. 2026-05-05
"""

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

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import (
    option_files, SYM_RE, to_minutes, latest_spot, summarize,
    print_yearly_breakdown, SPOT_PATH, LOT_SIZE,
)
from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from flattrade_bot.indicators.divergence import DivergenceEngine
from artifacts.f6_hybrid.causal_live_parity_research import IncrementalATR

SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
FILTER_SESSION_START = 555  # 09:15 — first 5m bar of the session (TradingView clock-aligned)
SL_POINTS, TP_POINTS = 7.0, 7.0
F6_S4_THRESH, F6_S1_THRESH = 79.5, 20.5
SUPER_CROSS_THRESH = 20.0
CONSEC_LOSS_LIMIT = 4
TRACK_WINDOW_LOW, TRACK_WINDOW_HIGH = -250, 300
FEES_PER_TRADE = 40
WORKERS = 8


class HeikinAshiState:
    """Heikin-Ashi conversion — TradingView chart-type HA, continuous across days.

    First bar:  ha_open = (open + close) / 2
    Later bars: ha_open = (prev_ha_open + prev_ha_close) / 2
                ha_close = (open + high + low + close) / 4
                ha_high  = max(high, ha_open, ha_close)
                ha_low   = min(low, ha_open, ha_close)
    """

    def __init__(self):
        self.open = None
        self.close = None

    def update(self, o, h, l, c):
        ha_close = (o + h + l + c) / 4.0
        ha_open = (o + c) / 2.0 if self.open is None else (self.open + self.close) / 2.0
        ha_high = max(h, ha_open, ha_close)
        ha_low = min(l, ha_open, ha_close)
        self.open = ha_open
        self.close = ha_close
        return {"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close}

    def peek(self, o, h, l, c):
        """HA values as-if the given (forming) bar were appended — no mutation."""
        ha_close = (o + h + l + c) / 4.0
        ha_open = (o + c) / 2.0 if self.open is None else (self.open + self.close) / 2.0
        ha_high = max(h, ha_open, ha_close)
        ha_low = min(l, ha_open, ha_close)
        return {"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close}


class PocketHTFFilter:
    """Index 5m filter matching TradingView exactly:
      - The chart is a Heikin-Ashi chart (chart type 8), so each 5m bar is
        HA-converted BEFORE UT Bot / LinReg (matches the original Marni filter).
      - UT Bot: key=1, period=10, src = HA close
        barcolor green when src > trailing_stop, red when src < trailing_stop
      - LinReg Candles: linreg(close, 11, 0) per OHLC component,
        signal = sma(bclose, 11) — the white line
    """

    def __init__(self, period=5, linreg_len=11, ut_key=1.0, ut_period=10):
        self.period = period
        self.linreg_len = linreg_len
        self.buf = []
        self._bucket = None
        self.ha = HeikinAshiState()
        self.raw_closes_1m = []  # HA closes, for linreg candle aggregation
        self.raw_opens_1m = []
        self.raw_highs_1m = []
        self.raw_lows_1m = []
        self.ut = _UTBotStandard(key=ut_key, period=ut_period)
        self.ut_color = "none"
        self.linreg_signal = None   # sma(bclose, 11) — the white line
        self.linreg_close = None    # linreg(close, 11, 0) — for bar coloring
        self.linreg_open = None
        self.bclose_history = []    # for sma(bclose, 11)
        self.bar_close = None       # HA 5m close

    def _linreg(self, values, length=11):
        """linreg(value, length, offset=0) — return the last value of the
        linear regression line fitted to `values` (pivoted at offset 0)."""
        if len(values) < length:
            return None
        y = values[-length:]
        n = length
        x_mean = (n - 1) / 2.0
        y_mean = sum(y) / float(n)
        denom = sum((xi - x_mean) ** 2 for xi in range(n))
        if denom == 0.0:
            return y_mean
        slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(range(n), y)) / denom
        return y_mean - slope * x_mean + slope * (n - 1)

    def update_1m(self, c):
        # Skip pre-session rows (broker emits a flat 09:14 placeholder row;
        # TradingView's first 5m bar starts at 09:15 = minute 555).
        if c.minute < FILTER_SESSION_START:
            return
        # Align bars to clock boundaries (09:15, 09:20, ...) like TradingView;
        # row-counting from the 09:14 placeholder would shift every bar 1m.
        bucket = c.minute - (c.minute % self.period)
        if self.buf and self._bucket is not None and bucket != self._bucket:
            self._commit_bar()
        self._bucket = bucket
        self.buf.append(c)

    def _commit_bar(self):
        # Aggregate 1m into 5m bar
        agg_open = self.buf[0].open
        agg_high = max(b.high for b in self.buf)
        agg_low = min(b.low for b in self.buf)
        agg_close = self.buf[-1].close
        self.buf.clear()

        # HA-convert the 5m bar first (the chart is Heikin-Ashi), then run
        # UT Bot + LinReg on HA values — matches TradingView exactly.
        ha = self.ha.update(agg_open, agg_high, agg_low, agg_close)
        self.bar_close = ha["close"]
        self.ut_color = self.ut.update_close(ha["close"], ha["high"], ha["low"])

        # LinReg Candles: linreg each HA OHLC component individually
        self.raw_opens_1m.append(ha["open"])
        self.raw_highs_1m.append(ha["high"])
        self.raw_lows_1m.append(ha["low"])
        self.raw_closes_1m.append(ha["close"])

        linreg_open = self._linreg(self.raw_opens_1m, self.linreg_len)
        linreg_high = self._linreg(self.raw_highs_1m, self.linreg_len)
        linreg_low = self._linreg(self.raw_lows_1m, self.linreg_len)
        linreg_close = self._linreg(self.raw_closes_1m, self.linreg_len)

        if linreg_open is None:
            return

        # Signal line: sma(linreg_close, 11) — the white line
        self.linreg_close = linreg_close
        self.linreg_open = linreg_open
        self.bclose_history.append(linreg_close)
        if len(self.bclose_history) >= self.linreg_len:
            self.linreg_signal = sum(self.bclose_history[-self.linreg_len:]) / self.linreg_len

    def start_day(self):
        """Flush the 5m buffer at a day boundary.

        TradingView starts a fresh 5m bar at 09:15 each session; without the
        flush the 376th 1m row of the previous day (376 % 5 == 1) leaks into
        today's first bar and skews it by one stale minute. The previous
        day's last bar (15:15..15:19...15:29) is committed first so the
        HA/UT/linreg state carries over exactly like the TradingView chart.
        """
        if self.buf:
            self._commit_bar()
        self.buf.clear()
        self._bucket = None

    def peek_snapshot(self):
        """Forming 5m bar state as TradingView draws it live: UT color and
        close-vs-white-line computed from the bar currently forming (last 1m
        close as the live price). None when not yet computable."""
        if not self.buf:
            return None
        f_open = self.buf[0].open
        f_high = max(b.high for b in self.buf)
        f_low = min(b.low for b in self.buf)
        f_close = self.buf[-1].close
        f_ha = self.ha.peek(f_open, f_high, f_low, f_close)

        ut = self.ut.peek_close(f_ha["close"], f_ha["high"], f_ha["low"])
        if ut == "none":
            return None

        linreg_close = None
        if len(self.raw_closes_1m) >= self.linreg_len - 1:
            linreg_close = self._linreg(self.raw_closes_1m + [f_ha["close"]], self.linreg_len)
        white_line = None
        if linreg_close is not None and len(self.bclose_history) >= self.linreg_len - 1:
            white_line = (sum(self.bclose_history[-(self.linreg_len - 1):]) + linreg_close) / self.linreg_len
        return {
            "linreg_plot": white_line,
            "ut_color": ut,
            "ha_close": f_ha["close"],
        }

    def snapshot(self):
        return {
            "linreg_plot": self.linreg_signal,
            "ut_color": self.ut_color,
            "ha_close": self.bar_close,
        }


class _UTBotStandard:
    """UT Bot matching TradingView Pine Script exactly.

    Pine source (h=false, src=close on an HA chart):
        xATR = atr(10)
        nLoss = 1 * xATR
        src = close  (Heikin-Ashi close on the HA chart)
        trail := iff(src > trail[1] and src[1] > trail[1], max(trail[1], src - nLoss),
                 iff(src < trail[1] and src[1] < trail[1], min(trail[1], src + nLoss),
                 iff(src > trail[1], src - nLoss, src + nLoss)))
        barcolor = src > trail ? green : red
    """

    def __init__(self, key=1.0, period=10):
        self.key = key
        self.atr = IncrementalATR(period)
        self.trailing_stop = 0.0
        self.prev_src = None
        self.prev_trail = None

    def update_close(self, src_close, src_high, src_low):
        """Update with HA close, high, low of the 5m bar.
        ATR is computed on the 5m bar's HA OHLC (matching TradingView)."""
        atr = self.atr.update(src_high, src_low, src_close)
        prev_trail = self.prev_trail if self.prev_trail is not None else 0.0
        if atr is None or atr == 0.0 or self.prev_src is None:
            self.prev_src = src_close
            self.prev_trail = prev_trail
            return "none"

        n_loss = self.key * atr

        # Pine trailing stop logic (exact port)
        if src_close > prev_trail and self.prev_src > prev_trail:
            self.trailing_stop = max(prev_trail, src_close - n_loss)
        elif src_close < prev_trail and self.prev_src < prev_trail:
            self.trailing_stop = min(prev_trail, src_close + n_loss)
        elif src_close > prev_trail:
            self.trailing_stop = src_close - n_loss
        else:
            self.trailing_stop = src_close + n_loss

        self.prev_src = src_close
        self.prev_trail = self.trailing_stop
        return "green" if src_close > prev_trail else "red"

    def peek_close(self, src_close, src_high, src_low):
        """Color the UT Bot would show on the FORMING bar right now.

        Non-mutating: ATR is peeked as-if the forming bar were appended and
        the Pine trail formula runs against the last completed bar's state.
        """
        atr = self.atr.peek(src_high, src_low, src_close)
        prev_trail = self.prev_trail if self.prev_trail is not None else 0.0
        if atr is None or atr == 0.0 or self.prev_src is None:
            return "none"

        n_loss = self.key * atr

        if src_close > prev_trail and self.prev_src > prev_trail:
            trail = max(prev_trail, src_close - n_loss)
        elif src_close < prev_trail and self.prev_src < prev_trail:
            trail = min(prev_trail, src_close + n_loss)
        elif src_close > prev_trail:
            trail = src_close - n_loss
        else:
            trail = src_close + n_loss
        return "green" if src_close > trail else "red"


class OptionTracker:
    """4-stochastic tracker for one option contract (1m bars)."""

    def __init__(self):
        self.s1 = IncrementalStochastic(9, 3)
        self.s2 = IncrementalStochastic(14, 3)
        self.s3 = IncrementalStochastic(40, 4)
        self.s4 = IncrementalStochastic(60, 10)
        self.prev_s1 = None

    def push(self, h, l, c):
        s1 = self.s1.push(h, l, c)
        s2 = self.s2.push(h, l, c)
        s3 = self.s3.push(h, l, c)
        s4 = self.s4.push(h, l, c)
        prev = self.prev_s1
        self.prev_s1 = s1
        return s1, s2, s3, s4, prev


def load_spot_ohlc():
    """Index 1-minute OHLC per day (SPOT_PATH).
    The merged CSV has IST timestamps for ammu data (09:15) but UTC for
    the user's Desktop 2026 data (03:45).  Detect UTC days (first bar < 09:00)
    and shift them +5h30m to IST so min values are consistent.
    """
    df = pd.read_csv(SPOT_PATH, parse_dates=["date"], engine="c")
    df = df.sort_values("date").reset_index(drop=True)
    # Normalize timezone: if tz-aware, convert to IST then strip
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    # Detect UTC days and shift to IST
    df["hour"] = df["date"].dt.hour
    for day, idx in df.groupby(df["date"].dt.strftime("%Y-%m-%d")).groups.items():
        first_hour = df.loc[idx[0], "hour"]
        if first_hour < 9:  # UTC timestamp — shift +5h30m
            df.loc[idx, "date"] = df.loc[idx, "date"] + pd.Timedelta(hours=5, minutes=30)
    df.drop(columns=["hour"], inplace=True)
    df["day"] = df["date"].dt.strftime("%Y-%m-%d")
    df["min"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    out = {}
    for day, g in df.groupby("day"):
        out[day] = {
            "min": g["min"].to_numpy(),
            "open": g["open"].to_numpy(dtype=float),
            "high": g["high"].to_numpy(dtype=float),
            "low": g["low"].to_numpy(dtype=float),
            "close": g["close"].to_numpy(dtype=float),
        }
    return out


def build_index_filter(spot, day=None, warm_days=12):
    """5m UT Bot + LinReg filter snapshots keyed by minute.

    Warms up the filter with `warm_days` prior trading days (TradingView has
    full history, so ATR/trail/linreg are live from the session open).
    Only the target day's snapshots are returned.

    Live forming parity: during each 5m bar, per-minute snapshots describe
    the FORMING bar (UT color + close-vs-white-line from the latest 1m close,
    TradingView-style), not the stale completed bar. This makes the first
    minutes of the day (before the first 5m completion) reflect the candle
    on screen, exactly like the live bot.
    """
    htf = PocketHTFFilter(period=5, linreg_len=11, ut_key=1.0, ut_period=10)

    def feed(s, record):
        mins, op, hi, lo, cl = s["min"], s["open"], s["high"], s["low"], s["close"]
        htf.start_day()
        for i in range(len(mins)):
            htf.update_1m(Candle(open=op[i], high=hi[i], low=lo[i], close=cl[i], minute=int(mins[i])))
            if record and htf.ut_color != "none":
                if len(htf.buf) > 0:
                    snap = htf.peek_snapshot()
                    if snap is None:
                        continue
                    yield int(mins[i]), snap
                else:
                    yield int(mins[i]), htf.snapshot()

    if day is not None:
        prior = sorted(k for k in grid.GLOBAL_SPOT if k < day)[-warm_days:]
        for pd_ in prior:
            for _ in feed(grid.GLOBAL_SPOT[pd_], False):
                pass
    return {m: s for m, s in feed(spot, True)}


def filter_allows(snapshots, minute):
    """Filter state of the last completed 5m bar at/before `minute`."""
    state = None
    for m, s in snapshots.items():
        if m <= minute:
            state = s
        else:
            break
    if state is None or state["linreg_plot"] is None:
        return None
    bull = state["ha_close"] > state["linreg_plot"] and state["ut_color"] == "green"
    bear = state["ha_close"] < state["linreg_plot"] and state["ut_color"] == "red"
    if bull and not bear:
        return "CE"
    if bear and not bull:
        return "PE"
    return None


def process_day(args):
    day, fpath, fprev = args
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
    divs = {}
    for sym, g in gp.items():
        t = OptionTracker()
        for i in range(len(g["min"])):
            t.push(g["high"][i], g["low"][i], g["close"][i])
        trk[sym] = t
        divs[sym] = DivergenceEngine()

    ifilter = build_index_filter(spot, day=day)
    pmtrig = {}
    slices = {}
    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = OptionTracker()
            divs[sym] = DivergenceEngine()
        t = trk[sym]
        slices[sym] = g
        mm2 = SYM_RE.match(sym)
        if not mm2:
            continue
        sv, side = int(mm2.group(2)), mm2.group(3)
        for i in range(len(g["min"])):
            m = g["min"][i]
            s1, s2, s3, s4, prev = t.push(g["high"][i], g["low"][i], g["close"][i])
            divs[sym].update(g["close"][i], s1, s2, low_price=g["low"][i], high_price=g["high"][i])
            if s1 is None or prev is None:
                continue
            # FLAG: plain setup, NO divergence requirement.
            flag = prev > F6_S1_THRESH and s1 <= F6_S1_THRESH and s4 is not None and s4 >= F6_S4_THRESH
            # SUPER: S1 crosses back above 20 (current trough fully formed) AND
            # bullish trough divergence confirmed at that bar (lower low + higher
            # S1 or S2 vs the previous confirmed trough).
            super_ = (
                prev <= SUPER_CROSS_THRESH and s1 > SUPER_CROSS_THRESH
                and divs[sym].divergence_confirmed_at_last_update() is not None
            )
            if flag or super_:
                pmtrig.setdefault(m, []).append((side, sv, sym, "flag" if flag else "super"))

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
                                   "reason": rsn, "duration_min": pos["duration_min"], "signal": pos["signal"]})
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
                           "reason": "EOD", "duration_min": pos["duration_min"], "signal": pos["signal"]})
            pos = None
            break
        if pos is not None or shut or minute >= SESSION_END:
            continue

        allowed = filter_allows(ifilter, minute)
        if allowed is None:
            continue
        for (sig_side, sig_stk, sig_sym, signal) in pmtrig.get(minute, []):
            if sig_side != allowed:
                continue
            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                bar = bslice(ai[1], minute)
                if bar:
                    ep = float(bar[3])
                    sl_use, tp_use = SL_POINTS, TP_POINTS
                    sl, tgt = ep - sl_use, ep + tp_use
                    pos = {"side": sig_side, "symbol": ai[0], "slice": ai[1], "entry": ep,
                           "sl": sl, "tgt": tgt, "sl_pts": sl_use, "tp_pts": tp_use,
                           "entry_min": minute, "last_px": ep, "duration_min": 0, "signal": signal}
                    break
    return trades


def run_days(pool, days, files):
    tasks = [(day, str(files[day]), str(files[days[i - 1]]) if i > 0 else "")
             for i, day in enumerate(days)]
    all_trades = []
    for res in pool.map(process_day, tasks):
        all_trades.extend(res)
    return all_trades


def yearly_breakdown(trades):
    years = {}
    for t in trades:
        y = t["date"][:4]
        years.setdefault(y, []).append(t)
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
    trades = run_days(pool, days, files)
    pool.close()
    pool.join()

    st = summarize(trades)
    print(f"\n=== POCKETMONEY (1m proxy) ===  {start} -> {end}")
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
        by_sig = {}
        for t in trades:
            by_sig[t["signal"]] = by_sig.get(t["signal"], 0) + 1
        print("signals:", ", ".join(f"{k}={v}" for k, v in sorted(by_sig.items())))
        for y, ts in sorted(yearly_breakdown(trades).items()):
            s = summarize(ts)
            print(f"  {y}: trades {s['trades']:5d}  WR {s['wr']:5.1f}%  "
                  f"net {s['pts']:+10.1f} pts  rs {s['rs']:+12,d}  PF {s['pf']:5.2f}")
    print(f"elapsed {(time.time() - t0):.1f}s")

    out = Path(__file__).parent / "pocket_money_backtest.json"
    if out.exists():
        print(f"SKIP write: {out} exists")
        return
    payload = {
        "strategy": "PocketMoney (1m proxy of 10s scalper)",
        "start": start, "end": end, "n_days": len(days),
        "params": {
            "s1": [9, 3], "s2": [14, 3], "s3": [40, 4], "s4": [60, 10],
            "flag": f"S1<=20.5 from neutral & S4>=79.5", "super": "S1 crosses >20 + trough divergence",
            "sl_pts": SL_POINTS, "tp_pts": TP_POINTS,
            "entry_side": "CE atm-100 / PE atm+100",
            "filter": "5m HA+UT green>linreg -> CE, red<linreg -> PE (index) + option bullish trough divergence (1m)",
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
