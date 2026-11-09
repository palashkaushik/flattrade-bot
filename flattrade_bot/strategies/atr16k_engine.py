"""ATR-16K engine — SR-level engulfing/harami with an ATR stop (live).

Faithful port of the validated reference implementation (ob_pat.py / sy2.py).
Every function below is a line-for-line translation of the reference math:

  swings / refine / ema_tf / st_map / atr_series / pat_any
  wave search (skip budget 4, 5>3>1 & 4>2>0 with a 1-pt buffer, first valid)
  a = first L after wave-5, b = first H after a, c = first L after b
  entry window = [b_conf + 1 , c_conf + 1]        (c extended one bar)
  trigger = engulfing OR harami (either direction) with an SR level inside
            the bar and the bar opening at/above that level
  entry = trigger close, SL = level - 1.5*ATR(14,1m), TP = level + 3*bar-range
  contract = 2nd ITM at the WAVE-START minute (CE atm-100 / PE atm+100)
  one signal per contract per session (reference: first candidate, then break)

It emits the same signal dict the live executor already consumes
(symbol/token/side/strike/entry/sl_points/tp_points/...), so order placement,
risk and the broker integration stay untouched.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---- strategy constants (identical to the validated build) ----
THR = 5.0            # close-zigzag threshold
BUF = 1.0            # wave validity buffer
WAVE_MODE = os.getenv("ATR16K_WAVE_MODE", "full").strip().lower()
# Trigger: "pattern" = engulf/harami/twin; "ob" = order-block touch +
# close above the nearest SR level above the bar (deployed 2026-11-09).
TRIGGER_MODE = os.getenv("ATR16K_TRIGGER", "ob").strip().lower()
SKIP = 4             # pivot skip budget
TP_MULT = 5.0        # TP = level + 5.0 * trigger range
ATR_LEN = 14
ATR_MULT = 1.0       # SL = level - 1.0 * ATR(14, 1m)
# Hard cap on the per-trade stop distance (points of premium). 0 = disabled.
MAX_SL_POINTS = float(os.getenv("ATR16K_MAX_SL", "20"))
# Minimum stop distance: when the trigger bar closes through the level, the
# level-based stop can land at or even ABOVE the entry (a 0.14-pt stop, or a
# fake win). Re-anchor it to at least this far below the entry.
MIN_RISK_POINTS = float(os.getenv("ATR16K_MIN_RISK", "3"))
C_EXT = 1            # c-window extended +1 bar past the c confirmation
SL_GRACE = 1.0       # reference stop fires at low <= SL - 1
# Overlapping waves: allowed (each wave is its own a-b-c). Overlapping TRADES
# are still impossible — one position at a time discards a signal that arrives
# while a trade is open.
ALLOW_OVERLAP = os.getenv("ATR16K_ALLOW_OVERLAP", "0") == "1"
# Carry-across: pivots/waves include the previous session's bars, so a wave may
# start on the prior day and trigger today (7-session test: +Rs 2,025 vs same-day).
CARRY = os.getenv("ATR16K_CARRY", "1") == "1"
# Twin-candle fallback (engulf-first): consecutive opposite-colour candles whose
# low OR high match within this tolerance, level touched anywhere in the bar.
TWIN_TOL = float(os.getenv("ATR16K_TWIN_TOL", "0.75"))
# Level test mode:
#   open      = the TRIGGER bar's open >= level (validated default)
#   close     = the trigger bar's close >= level (discretionary variant;
#               8-day test 2026-10-21..28: +37.20 pts / -Rs 1,518)
#   prev_open = the PREVIOUS bar's open >= level (price was above the level
#               before the trigger bar tested it)
LEVEL_MODE = os.getenv("ATR16K_LEVEL_MODE", "open").strip().lower()

SESSION_START_MIN = 555      # 09:15
ENTRY_END_MIN = 900          # 15:00 (backtest entry cutoff)
DAY_END_MIN = 930            # 15:30
# TEMPORARY DIAGNOSTIC (one session): log every bar that falls inside an open
# a/b/c window — OHLC, pattern, twin, levels inside, open>=level, ATM/strike gate.
BAR_TRACE = os.getenv("ATR16K_BAR_TRACE", "1") == "1"

# Daily cap - the validated build had none; configurable, default off.
LOSS_CAP = float(os.getenv("ATR16K_LOSS_CAP", "40"))
PROFIT_CAP = float(os.getenv("ATR16K_PROFIT_CAP", "inf"))


# ------------------------------------------------------------------ reference math
def swings(bd: List[dict], thr: float = THR) -> List[Tuple[int, str, int]]:
    """Close-based zigzag -> [(extreme_index, 'H'|'L', confirmation_index)]."""
    piv = [(0, "L", 0)]
    d, ext = 1, 1
    for i in range(2, len(bd)):
        c = bd[i]["c"]
        if d == 1:
            if c >= bd[ext]["c"]:
                ext = i
            elif bd[ext]["c"] - c >= thr:
                piv.append((ext, "H", i)); d, ext = -1, i
        else:
            if c <= bd[ext]["c"]:
                ext = i
            elif c - bd[ext]["c"] >= thr:
                piv.append((ext, "L", i)); d, ext = 1, i
    piv.append((ext, "H" if d == 1 else "L", len(bd) - 1))
    return piv


def refine(bd: List[dict], piv: List[Tuple[int, str, int]]) -> List[Tuple[Any, str, float, int, int]]:
    """(time, type, price, price_index, confirmation_index) with the leg extreme."""
    out = []
    for k, (i, typ, ci) in enumerate(piv):
        if k == 0:
            out.append((bd[i]["t"], typ, bd[i]["l"] if typ == "L" else bd[i]["h"], i, ci)); continue
        lo = piv[k - 1][0] + 1
        hi = piv[k + 1][0] if k + 1 < len(piv) else len(bd) - 1
        if hi < lo:
            hi = lo
        if typ == "H":
            j = max(range(lo, hi + 1), key=lambda x: bd[x]["h"])
            out.append((bd[j]["t"], "H", bd[j]["h"], j, ci))
        else:
            j = min(range(lo, hi + 1), key=lambda x: bd[x]["l"])
            out.append((bd[j]["t"], "L", bd[j]["l"], j, ci))
    return out


def ema_tf(closes: List[float], minutes: List[int], tf: int, length: int) -> List[float]:
    """Chart-style EMA: a bucket updates with the live close, previous bucket
    seeds each new bucket (identical to the reference)."""
    a = 2.0 / (length + 1.0)
    out: List[float] = []
    prev = None
    cur = None
    e = None
    for c, m in zip(closes, minutes):
        b = 555 + tf * ((m - 555) // tf)
        if b != cur:
            prev = e
            cur = b
        e = c if prev is None else prev + a * (c - prev)
        out.append(e)
    return out


def st_map(H, L, C, M, tf: int = 3, length: int = 10, mult: float = 3.0):
    gb: List[List[float]] = []
    gi: List[int] = []
    cur = None
    for h, l, c, m in zip(H, L, C, M):
        k = 555 + tf * ((m - 555) // tf)
        if k != cur:
            gb.append([h, l, c]); cur = k
        else:
            gb[-1][0] = max(gb[-1][0], h)
            gb[-1][1] = min(gb[-1][1], l)
            gb[-1][2] = c
        gi.append(len(gb) - 1)
    atr = up = dn = trend = pc = None
    trs: List[float] = []
    line: List[float] = []
    for h, l, c in gb:
        trv = (h - l) if pc is None else max(h - l, abs(h - pc), abs(l - pc))
        trs.append(trv)
        atr = sum(trs) / len(trs) if len(trs) <= length else (atr * (length - 1) + trv) / length
        hl2 = (h + l) / 2.0
        un, dnn = hl2 - mult * atr, hl2 + mult * atr
        if up is None:
            up, dn, trend = un, dnn, (1 if c > hl2 else -1)
        else:
            pu, pd = up, dn
            up = max(un, pu) if pc > pu else un
            dn = min(dnn, pd) if pc < pd else dnn
            if trend == -1 and c > pd:
                trend = 1
            elif trend == 1 and c < pu:
                trend = -1
        pc = c
        line.append(up if trend == 1 else dn)
    return [line[gi[i] - 1] if gi[i] > 0 else None for i in range(len(C))]


def atr_series(H, L, C, n: int = ATR_LEN) -> List[float]:
    trs: List[float] = []
    out: List[float] = []
    pc = None
    for i in range(len(C)):
        tr = (H[i] - L[i]) if pc is None else max(H[i] - L[i], abs(H[i] - pc), abs(L[i] - pc))
        trs.append(tr)
        out.append(sum(trs[-n:]) / min(len(trs), n))
        pc = C[i]
    return out


def pat_any(bd: List[dict], i: int) -> Optional[str]:
    """Direction-agnostic engulfing / harami on bodies. Returns the kind or None."""
    if i < 1:
        return None
    o, c = bd[i]["o"], bd[i]["c"]
    po, pc = bd[i - 1]["o"], bd[i - 1]["c"]
    cl, ch = min(o, c), max(o, c)
    pl, ph = min(po, pc), max(po, pc)
    if cl <= pl and ch >= ph:
        return "engulf"
    if cl >= pl and ch <= ph and (ch - cl) < (ph - pl):
        return "harami"
    return None


def twin_candles(bd: List[dict], i: int, tol: float = TWIN_TOL) -> bool:
    if i < 1:
        return False
    o, c = bd[i]["o"], bd[i]["c"]
    po, pc = bd[i - 1]["o"], bd[i - 1]["c"]
    red, green = c < o, c > o
    pred, pgreen = pc < po, pc > po
    if not ((red and pgreen) or (green and pred)):
        return False
    return (abs(bd[i]["l"] - bd[i - 1]["l"]) <= tol
            or abs(bd[i]["h"] - bd[i - 1]["h"]) <= tol)


class _Val:
    __slots__ = ("value",)

    def __init__(self, value=None):
        self.value = value


class _NullTF:
    last_s1 = None
    last_s3 = None
    last_s4 = None
    prev_s1 = None


# ------------------------------------------------------------------ per-contract state
@dataclass
class ATR16KContractState:
    symbol: str
    token: str
    side: str                 # "CE" | "PE"
    strike: int

    # 1m tick aggregation (same semantics as the other live engines)
    current_min: int = -1
    cur_open: Optional[float] = None
    cur_high: Optional[float] = None
    cur_low: Optional[float] = None
    cur_close: Optional[float] = None
    _bar_start_vol: Optional[float] = None

    bars: List[Any] = field(default_factory=list)          # today
    seeded: List[Any] = field(default_factory=list)        # prior day(s), warmup only
    day_levels: Dict[str, float] = field(default_factory=dict)
    session_date: Optional[str] = None
    seed_complete: bool = False

    fired_waves: set = field(default_factory=set)   # w0 keys already signalled today
    _seeding: bool = False
    _ctx: List[dict] = field(default_factory=list)
    _levels: Dict[str, float] = field(default_factory=dict)
    ok_ctx: List[dict] = field(default_factory=list)
    atm_fn: Optional[Callable[[int], Optional[int]]] = None

    # compatibility attributes for the dashboard (read-only in the UI)
    virgin: Dict[str, float] = field(default_factory=dict)
    pdvwap: Optional[float] = None
    bar_sink: Optional[Callable[[Any], None]] = None
    anchor_bar: int = -10**9
    flag_armed: bool = False
    super_armed: bool = False
    flag_arm_bar: int = -1
    super_arm_bar: int = -1
    display_levels: Dict[str, float] = field(default_factory=dict)
    ema20: _Val = field(default_factory=_Val)
    ema200: _Val = field(default_factory=_Val)
    vwap: _Val = field(default_factory=_Val)
    latest_atr: float = 0.0
    tf_trackers: Dict[int, Any] = field(default_factory=lambda: {i: _NullTF() for i in (1, 2, 3, 5)})
    tf: Dict[int, Any] = field(default_factory=lambda: {i: _NullTF() for i in (1, 2, 3, 5)})

    # ---- lifecycle ----
    def reset_session(self):
        self.bars = []
        self.seeded = []
        self.current_min = -1
        self.cur_open = self.cur_high = self.cur_low = self.cur_close = None
        self._bar_start_vol = None
        self.fired_waves = set()
        self._seeding = False
        self._ctx = []
        self._levels = {}
        self.seed_complete = False

    def set_day_levels(self, prev_high: float, prev_low: float, prev_close: float,
                       virgin: Optional[Dict[str, float]] = None,
                       pdvwap: Optional[float] = None):
        ph, pl, pc = float(prev_high), float(prev_low), float(prev_close)
        rng = ph - pl
        P = (ph + pl + pc) / 3.0
        BC = (ph + pl) / 2.0
        TC = 2.0 * P - BC
        r2 = lambda x: round(float(x), 2)
        self.day_levels = {
            "PDH": ph, "PDL": pl,
            "CPR": r2(P), "BC": r2(BC), "TC": r2(TC),
            "fibR3": r2(P + rng), "fibS3": r2(P - rng),
            "camR3": r2(pc + rng * 1.1 / 4.0), "camL3": r2(pc - rng * 1.1 / 4.0),
        }
        self.display_levels = {
            "PDH": ph, "PDL": pl, "CPR_BC": r2(BC), "CPR_Pivot": r2(P), "CPR_TC": r2(TC),
        }
        self.virgin = dict(virgin) if virgin else {}
        self.pdvwap = pdvwap

    def seed_bars(self, prior_bars: List[Any], today_bars: Optional[List[Any]] = None,
                  session_date: Optional[str] = None):
        self.seeded = list(prior_bars or [])
        if session_date is not None:
            self.session_date = session_date
        self._seeding = True
        for b in (today_bars or []):
            self.bars.append(b)
        self._seeding = False
        self.seed_complete = True

    def merge_today_bars(self, bars: List[Any]):
        """Re-anchor today's bars with exchange bars (by minute).

        A parked/restored contract misses the minutes it was unsubscribed, so
        its session VWAP/EMA would start mid-session (observed: P24700 VWAP
        356.66 vs the true 371.65 on 2026-11-05). Exchange bars replace the
        same-minute tick bars and fill the gaps; live-only minutes are kept.
        """
        by_min = {b.minute: b for b in self.bars}
        for b in bars:
            by_min[b.minute] = b
        self.bars = [by_min[m] for m in sorted(by_min)]

    # ---- live tick path ----
    def push_tick(self, ltp: float, dt: datetime,
                  cum_volume: Optional[float] = None) -> Optional[Dict[str, Any]]:
        minute = dt.hour * 60 + dt.minute
        if self.current_min == -1:
            self.current_min = minute
            self.cur_open = self.cur_high = self.cur_low = self.cur_close = ltp
            if cum_volume is not None:
                self._bar_start_vol = cum_volume
            return None
        if minute == self.current_min:
            self.cur_high = max(self.cur_high, ltp)
            self.cur_low = min(self.cur_low, ltp)
            self.cur_close = ltp
            return None
        if cum_volume is not None and self._bar_start_vol is not None:
            bar_vol = max(cum_volume - self._bar_start_vol, 1.0)
        else:
            bar_vol = 100.0

        class _B:
            __slots__ = ("minute", "open", "high", "low", "close", "timestamp", "volume")

        bar = _B()
        bar.minute = self.current_min
        bar.open, bar.high, bar.low, bar.close = self.cur_open, self.cur_high, self.cur_low, self.cur_close
        bar.timestamp = dt
        bar.volume = bar_vol

        if self.bar_sink is not None:
            try:
                self.bar_sink(bar)
            except Exception:
                pass

        self.current_min = minute
        self.cur_open = self.cur_high = self.cur_low = self.cur_close = ltp
        self._bar_start_vol = cum_volume
        return self.on_bar(bar)

    # ---- signal evaluation ----
    def on_bar(self, bar) -> Optional[Dict[str, Any]]:
        self.bars.append(bar)
        if self._seeding or len(self.bars) < 3:
            return None
        if not self.day_levels:
            self.set_day_levels(bar.high, bar.low, bar.close)
        self._compute()
        j = len(self._bd) - 1
        ei = len(self.bars) - 1
        b = self.bars[ei]
        if b.minute > ENTRY_END_MIN:
            return None
        if TRIGGER_MODE == "ob":
            # order-block trigger: bar touches the nearest single-candle OB
            # formed before it, and closes above the nearest level above it
            touched = None
            for (_q, _lo, _hi) in getattr(self, "_obs", []):
                if _q >= j:
                    continue
                if b.low <= _hi and b.high >= _lo:
                    _mid = (_lo + _hi) / 2.0
                    _d = abs(b.close - _mid)
                    if touched is None or _d < touched[0]:
                        touched = (_d, _lo, _hi)
            if touched is None:
                return None
            kind, _any_touch = "ob", False
        else:
            kind = pat_any(self._bd, j)
            _any_touch = False
            if not kind:
                # engulf-first fallback: twin candles, level touched anywhere in the bar
                if twin_candles(self._bd, j):
                    kind, _any_touch = "twin", True
                else:
                    return None

        if BAR_TRACE and self.ok_ctx:
            try:
                self._trace_bar(b, j, kind, _any_touch)
            except Exception:      # diagnostics must never affect trading
                pass

        for c in self.ok_ctx:
            if c["fired"] or not (c["b_conf"] + 1 <= j <= c["c_conf"]):
                continue
            atm = c["atm"]
            if atm is None:
                continue
            if self.side == "CE" and self.strike != atm - 100:
                continue
            if self.side == "PE" and self.strike != atm + 100:
                continue
            if TRIGGER_MODE == "ob":
                # nearest SR level ABOVE the bar; the close must reclaim it
                above = [(nm_, v_) for nm_, v_ in self._levels.items() if v_ > b.low]
                if not above:
                    continue
                nm, lvl = min(above, key=lambda z: z[1])
                if b.close < lvl:
                    continue
            else:
                cd = []
                for nm, v in self._levels.items():
                    if not (b.low <= v <= b.high):
                        continue
                    if _any_touch:
                        ok = True
                    elif LEVEL_MODE == "prev_open":
                        ok = self._bd[j - 1]["o"] >= v
                    elif LEVEL_MODE == "open_or_prev":
                        ok = (b.open >= v) or (self._bd[j - 1]["o"] >= v)
                    elif LEVEL_MODE == "open_and_prev":
                        ok = (b.open >= v) and (self._bd[j - 1]["o"] >= v)
                    elif LEVEL_MODE == "close":
                        ok = b.close >= v
                    elif LEVEL_MODE == "none":
                        ok = True          # level merely inside the bar
                    else:
                        ok = b.open >= v
                    if ok:
                        cd.append((nm, v))
                if not cd:
                    continue
                nm, lvl = max(cd, key=lambda z: z[1])
            entry = float(b.close)
            atr = float(self._atr[ei])
            sl = lvl - ATR_MULT * atr
            if MAX_SL_POINTS > 0 and entry - sl > MAX_SL_POINTS:
                sl = entry - MAX_SL_POINTS
            if MIN_RISK_POINTS > 0 and entry - sl < MIN_RISK_POINTS:
                sl = entry - MIN_RISK_POINTS
            tp = lvl + TP_MULT * (b.high - b.low)
            self.fired_waves.add((c["w_day"], c["w_min"]))
            return {
                "side": self.side,
                "symbol": self.symbol,
                "token": self.token,
                "strike": self.strike,
                "trigger": kind,
                "tf": "1m",
                "level": nm,
                "level_px": round(float(lvl), 2),
                "w_min": c["w_min"],
                "atm_w": atm,
                "entry": round(entry, 2),
                "dist": round(entry - sl, 2),
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "sl_points": round(entry - sl, 2),
                "tp_points": round(tp - entry, 2),
                "bar": b,
                "timestamp": getattr(b, "timestamp", None),
            }
        return None

    # ---- diagnostics (temporary) ----
    def _trace_bar(self, b, j: int, kind: Optional[str], any_touch: bool):
        """One line per bar that sits inside an open a/b/c window."""
        hm = f"{b.minute // 60:02d}:{b.minute % 60:02d}"
        twin = twin_candles(self._bd, j)
        for c in self.ok_ctx:
            if not (c["b_conf"] + 1 <= j <= c["c_conf"]):
                continue
            inside = []
            for nm, v in self._levels.items():
                if b.low <= v <= b.high:
                    inside.append(f"{nm}={v:.2f}(open>={'Y' if b.open >= v else 'N'})")
            atm = c["atm"]
            strike_ok = bool(atm is not None and (
                (self.side == "CE" and self.strike == atm - 100) or
                (self.side == "PE" and self.strike == atm + 100)))
            wm = f"{c['w_min'] // 60:02d}:{c['w_min'] % 60:02d}"
            logger.info(
                "BARTRACE %s %s O%.2f H%.2f L%.2f C%.2f pat=%s twin=%s any=%s | "
                "ctx %s %s b%d..c%d atm=%s strike_ok=%s | levels: %s",
                self.symbol, hm, b.open, b.high, b.low, b.close, kind or "-",
                int(bool(twin)), int(bool(any_touch)), c["w_day"], wm,
                c["b_conf"], c["c_conf"], atm, int(strike_ok),
                ", ".join(inside) if inside else "none")

    # ---- internals ----
    def _compute(self):
        n = len(self.bars)
        k = len(self.seeded)
        HC = [x.close for x in self.seeded] + [x.close for x in self.bars]
        HH = [x.high for x in self.seeded] + [x.high for x in self.bars]
        HL = [x.low for x in self.seeded] + [x.low for x in self.bars]
        HM = [x.minute for x in self.seeded] + [x.minute for x in self.bars]

        e20_5 = ema_tf(HC, HM, 5, 20)[k:]
        e20_3 = ema_tf(HC, HM, 3, 20)[k:]
        e200_3 = ema_tf(HC, HM, 3, 200)[k:]
        e200_5 = ema_tf(HC, HM, 5, 200)[k:]
        stl = st_map(HH, HL, HC, HM)[k:]

        H = [x.high for x in self.bars]
        L = [x.low for x in self.bars]
        C = [x.close for x in self.bars]
        self._atr = atr_series(H, L, C)

        pv_sum = 0.0
        vol_sum = 0.0
        vw: List[float] = []
        for i in range(n):
            v = getattr(self.bars[i], "volume", 0) or 0
            tp_ = (H[i] + L[i] + C[i]) / 3.0
            pv_sum += tp_ * v
            vol_sum += v
            vw.append(pv_sum / vol_sum if vol_sum > 0 else tp_)

        lv = dict(self.day_levels)
        lv["EMA20"] = e20_5[-1]
        lv["EMA20_3"] = e20_3[-1]
        lv["EMA200"] = e200_3[-1]
        lv["EMA200_5"] = e200_5[-1]
        lv["VWAP"] = vw[-1]
        if stl[-1] is not None:
            lv["ST"] = stl[-1]
        # Previous-day closing VWAP and virgin CPR/BC/TC (computed by the runner;
        # previously display-only, now part of the signal level set).
        if self.pdvwap:
            try:
                lv["PDVWAP"] = float(self.pdvwap)
            except (TypeError, ValueError):
                pass
        for _nm, _v in (self.virgin or {}).items():
            try:
                lv[_nm] = float(_v)
            except (TypeError, ValueError):
                pass
        self._levels = lv
        self.ema20 = _Val(e20_5[-1])
        self.ema200 = _Val(e200_3[-1])
        self.vwap = _Val(vw[-1])
        self.latest_atr = float(self._atr[-1])

        # ---- pivot/wave series: previous session + today (carry-across) ----
        prior_bars: List[Any] = []
        if CARRY and self.seeded:
            by_day: Dict[Any, List[Any]] = {}
            for x in self.seeded:
                _d = getattr(x, "timestamp", None)
                _d = _d.date() if _d is not None else None
                by_day.setdefault(_d, []).append(x)
            if by_day:
                last_day = max(by_day, key=lambda d: (d is None, d))
                prior_bars = [x for x in by_day[last_day][-400:] if x.minute <= 930]
        comb: List[dict] = []
        for x in prior_bars:
            _d = getattr(x, "timestamp", None)
            comb.append({"t": x.minute, "o": x.open, "h": x.high, "l": x.low,
                         "c": x.close, "min": x.minute,
                         "day": _d.date() if _d is not None else None, "src": "P"})
        for x in self.bars:
            _d = getattr(x, "timestamp", None)
            comb.append({"t": x.minute, "o": x.open, "h": x.high, "l": x.low,
                         "c": x.close, "min": x.minute,
                         "day": _d.date() if _d is not None else None, "src": "T"})
        self._bd = comb
        self._t0 = len(prior_bars)
        pv_ = refine(comb, swings(comb, THR))
        # ---- single-candle order blocks: last opposite candle before each
        # impulse pivot leg; zone = that candle's H/L (mirrors the backtest).
        _obs = []
        for _i in range(len(pv_) - 1):
            _a, _b2 = pv_[_i], pv_[_i + 1]
            if _a[1] == "L" and _b2[1] == "H":
                for _q in range(_a[3], max(-1, _a[3] - 6), -1):
                    if 0 <= _q < len(comb) and comb[_q]["c"] < comb[_q]["o"]:
                        _obs.append((_q, comb[_q]["l"], comb[_q]["h"])); break
            elif _a[1] == "H" and _b2[1] == "L":
                for _q in range(_a[3], max(-1, _a[3] - 6), -1):
                    if 0 <= _q < len(comb) and comb[_q]["c"] > comb[_q]["o"]:
                        _obs.append((_q, comb[_q]["l"], comb[_q]["h"])); break
        self._obs = _obs
        NP = len(pv_)
        waves = []
        for i0 in range(NP - 5):
            if pv_[i0][1] != "L":
                continue
            found = []

            def dfs(pos, need_t, chosen, budget):
                if len(chosen) == 6:
                    w = [pv_[q] for q in chosen]
                    a0, a1, a2, a3, a4, a5 = (p[2] for p in w)
                    if ((a4 > a2 - BUF and a2 > a0 - BUF) if WAVE_MODE == "correction"
                            else (a5 > a3 - BUF and a3 > a1 - BUF)):
                        found.append(list(chosen))
                    return
                for q in range(pos, min(NP, pos + 1 + budget)):
                    if pv_[q][1] == need_t:
                        chosen.append(q)
                        dfs(q + 1, "H" if need_t == "L" else "L", chosen, budget - (q - pos))
                        chosen.pop()

            dfs(i0 + 1, "H", [i0], SKIP)
            if found:
                waves.append((i0, found[0], [pv_[q] for q in found[0]]))

        ctx: List[dict] = []
        j = len(comb) - 1
        for w0, idxs, w in waves:
            last = idxs[-1]
            ai = next((q for q in range(last + 1, NP) if pv_[q][1] == "L"), None)
            if ai is None:
                continue
            bi = next((q for q in range(ai + 1, NP) if pv_[q][1] == "H"), None)
            if bi is None:
                continue
            b_conf = pv_[bi][4]
            ci = next((q for q in range(bi + 1, NP) if pv_[q][1] == "L"), None)
            c_conf = min((pv_[ci][4] if ci is not None else j) + C_EXT, j)
            if c_conf < self._t0:
                continue          # window closed before today — cannot trade today
            w_bar = comb[pv_[idxs[0]][3]]
            w_day, w_min = w_bar["day"], w_bar["min"]
            atm = self.atm_fn(w_day, w_min) if self.atm_fn else None
            ctx.append({"w0": w0, "end": last, "b_conf": b_conf, "c_conf": c_conf,
                        "w_min": w_min, "w_day": w_day, "atm": atm,
                        "fired": (w_day, w_min) in self.fired_waves})

        # non-overlap priority over the windows that can still trade today
        if ALLOW_OVERLAP:
            kept = ctx
        else:
            kept, last_end = [], -1
            for c in ctx:
                if c["w0"] > last_end:
                    kept.append(c); last_end = c["end"]
        self.ok_ctx = kept
        self._ctx = [c for c in kept if c["c_conf"] >= j]

    # ---- dashboard compatibility ----
    @property
    def baridx(self) -> int:
        return len(self.bars) - 1

    def _levels_now(self) -> List[Tuple[str, float]]:
        return [(nm, v) for nm, v in self._levels.items()]


# ------------------------------------------------------------------ coordinator
class ATR16KEngine:
    """Drop-in coordinator with the same surface the live runner already uses."""

    def __init__(self):
        self.contracts: Dict[str, ATR16KContractState] = {}
        self.spot_price: Optional[float] = None
        self.active_trade: Optional[Dict[str, Any]] = None
        self._atm_by_min: Dict[int, int] = {}
        self.index_hist: Dict[Any, Dict[int, float]] = {}
        self._last_minute: int = -1
        self._last_exit_min: int = -1
        self._day_key: Optional[str] = None
        self._day_cum_pts: float = 0.0
        self._day_halted: bool = False

    # ---- API parity ----
    def register_contract(self, key: str, symbol: str, token: str, side: str,
                          strike: int) -> ATR16KContractState:
        if key not in self.contracts:
            self.contracts[key] = ATR16KContractState(
                symbol=symbol, token=token, side=side, strike=int(strike))
        return self.contracts[key]

    def set_spot_price(self, spot: float):
        self.spot_price = float(spot)

    def _atm_at(self, day, minute: int) -> Optional[int]:
        d = self.index_hist.get(day)
        if d:
            ks = [k for k in d if k <= minute]
            if ks:
                return int(round(d[max(ks)] / 50.0) * 50)
        ks = [m for m in self._atm_by_min if m <= minute]
        return self._atm_by_min[max(ks)] if ks else None

    def set_index_history(self, hist: Dict[Any, Dict[int, float]]):
        """{date: {minute: spot}} for the prior days — day-aware ATM at the
        wave start (carry-across waves start on the previous session)."""
        self.index_hist = hist

    def desired_pairs(self, spot: float) -> List[Tuple[str, int]]:
        """Contract ladder the strategy may need: current 2nd ITM + watches +
        the exact 2nd ITM of every live wave context."""
        atm = int(round(float(spot) / 50.0) * 50)
        # Discovery ladder: every strike that can be 2nd ITM while the index
        # moves through the day (CE atm-off, PE atm+off), 12 contracts.
        want = {("CE", atm - off) for off in (50, 100, 150, 200, 250, 300)}
        want |= {("PE", atm + off) for off in (50, 100, 150, 200, 250, 300)}
        for cs in self.contracts.values():
            for c in getattr(cs, "_ctx", []):
                a = c.get("atm")
                if a and not c.get("fired"):
                    want.add(("CE", a - 100))
                    want.add(("PE", a + 100))
        return sorted(want)

    def on_trade_opened(self, trade: Dict[str, Any]):
        self.active_trade = dict(trade)

    def on_trade_closed(self):
        self._last_exit_min = self._last_minute
        self.active_trade = None

    def record_closed_points(self, pts: float, day_key: str):
        if day_key != self._day_key:
            self._day_key = day_key
            self._day_cum_pts = 0.0
            self._day_halted = False
        self._day_cum_pts += pts
        if self._day_cum_pts <= -LOSS_CAP or self._day_cum_pts >= PROFIT_CAP:
            self._day_halted = True

    def entries_halted(self, day_key: str) -> bool:
        return self._day_halted if day_key == self._day_key else False

    # ---- tick path ----
    def push_tick(self, key: str, ltp: float, dt: datetime,
                  cum_volume: Optional[float] = None) -> Optional[Dict[str, Any]]:
        cs = self.contracts.get(key)
        if not cs:
            return None
        minute = dt.hour * 60 + dt.minute
        if self.spot_price:
            self._atm_by_min[minute] = int(round(self.spot_price / 50.0) * 50)
        self._last_minute = minute

        cs.atm_fn = self._atm_at
        sig = cs.push_tick(ltp, dt, cum_volume=cum_volume)
        if self.active_trade:
            return None
        if not sig:
            return None
        if not (SESSION_START_MIN <= minute <= ENTRY_END_MIN):
            return None
        if self.entries_halted(f"{dt:%Y-%m-%d}"):
            return None
        if minute <= self._last_exit_min:
            return None
        return sig






