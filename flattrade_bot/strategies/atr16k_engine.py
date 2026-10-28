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

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---- strategy constants (identical to the validated build) ----
THR = 5.0            # close-zigzag threshold
BUF = 1.0            # wave validity buffer
SKIP = 4             # pivot skip budget
TP_MULT = 3.0        # TP = level + 3 * trigger range
ATR_LEN = 14
ATR_MULT = 1.5       # SL = level - 1.5 * ATR(14, 1m)
C_EXT = 1            # c-window extended +1 bar past the c confirmation
SL_GRACE = 1.0       # reference stop fires at low <= SL - 1

SESSION_START_MIN = 555      # 09:15
ENTRY_END_MIN = 915          # 15:15 (backtest entry cutoff)
DAY_END_MIN = 930            # 15:30

# Daily cap — the validated build had none; configurable, default off.
LOSS_CAP = float(os.getenv("ATR16K_LOSS_CAP", "inf"))
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
        j = len(self.bars) - 1
        b = self.bars[j]
        if b.minute > ENTRY_END_MIN:
            return None
        kind = pat_any(self._bd, j)
        if not kind:
            return None

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
            cd = [(nm, v) for nm, v in self._levels.items()
                  if b.low <= v <= b.high and b.open >= v]
            if not cd:
                continue
            nm, lvl = max(cd, key=lambda z: z[1])
            entry = float(b.close)
            atr = float(self._atr[j])
            sl = lvl - ATR_MULT * atr
            tp = lvl + TP_MULT * (b.high - b.low)
            self.fired_waves.add(c["w0"])
            return {
                "side": self.side,
                "symbol": self.symbol,
                "token": self.token,
                "strike": self.strike,
                "trigger": kind,
                "tf": "1m",
                "level": nm,
                "level_px": round(float(lvl), 2),
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
        self._levels = lv
        self.ema20 = _Val(e20_5[-1])
        self.ema200 = _Val(e200_3[-1])
        self.vwap = _Val(vw[-1])
        self.latest_atr = float(self._atr[-1])

        bd = [{"t": b.minute, "o": b.open, "h": b.high, "l": b.low, "c": b.close,
               "min": b.minute} for b in self.bars]
        self._bd = bd
        pv_ = refine(bd, swings(bd, THR))
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
                    if a5 > a3 - BUF and a3 > a1 - BUF and a4 > a2 - BUF and a2 > a0 - BUF:
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

        nonlap = []
        last_end = -1
        for w0, idxs, w in waves:
            if w0 > last_end:
                nonlap.append((w0, idxs, w)); last_end = idxs[-1]

        ctx: List[dict] = []
        j = n - 1
        for w0, idxs, w in nonlap:
            last = idxs[-1]
            ai = next((q for q in range(last + 1, NP) if pv_[q][1] == "L"), None)
            if ai is None:
                continue
            bi = next((q for q in range(ai + 1, NP) if pv_[q][1] == "H"), None)
            if bi is None:
                continue
            b_conf = pv_[bi][4]
            ci = next((q for q in range(bi + 1, NP) if pv_[q][1] == "L"), None)
            c_conf = min((pv_[ci][4] if ci is not None else n - 1) + C_EXT, n - 1)
            w_min = bd[pv_[idxs[0]][3]]["min"]
            atm = self.atm_fn(w_min) if self.atm_fn else None
            ctx.append({"w0": w0, "b_conf": b_conf, "c_conf": c_conf,
                        "w_min": w_min, "atm": atm,
                        "fired": w0 in self.fired_waves})
        self.ok_ctx = ctx
        self._ctx = [c for c in ctx if c["c_conf"] >= j]

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

    def _atm_at(self, minute: int) -> Optional[int]:
        ks = [m for m in self._atm_by_min if m <= minute]
        return self._atm_by_min[max(ks)] if ks else None

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
