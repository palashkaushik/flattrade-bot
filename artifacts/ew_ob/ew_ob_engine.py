"""Elliott OB (EW-OB) engine — signal detection + entry/exit logic.

Pure logic, no data loading (see ew_ob_runner.py). Operates on a sequential
stream of 1m index bars. UT Bot state carries across sessions; anchor windows
are reset at each session boundary so a correction cannot mix trading days.

Components:
  WaveDetector        — lenient 5-wave impulse count + ABC correction (1m only)
  OrderBlockTracker   — single-candle OB detection per TF + persistent registry
  EWOBEngine          — wires detector + OBs + entry/exit + option settlement
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Callable, Optional

# ── constants ────────────────────────────────────────────────────────────────
SESSION_START = 555  # 09:15 IST
SESSION_END = 930    # 15:30 IST
LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
ATR_PERIOD = 10
DEFAULT_TOL = 0.5
WAVE_TOL = 3.0
RISK_MODE_OB_W5 = "ob_w5"
RISK_MODE_OB_SAME_TF = "ob_same_tf"
RISK_MODE_ATR = "atr"
RISK_MODE_OPTION_FIXED = "option_fixed"
RISK_MODE_FIB = "fib"
DEFAULT_SL_MULT = 3.0
DEFAULT_TP_PTS = 60.0
POST_B_OB_DELAY = 12  # Aug 20 arm at 14:00, first C-wave OB at 14:12
SETUP_MAX_AGE_MINUTES = 60
UT_KEY = 1.0         # UT Bot ATR multiplier
UT_PERIOD = 10       # UT Bot ATR period

# fee model (mirrors backtest_walkforward_fees.py defaults; zero brokerage)
# One index point round trip: 0.5 on entry + 0.5 on exit.
SLIPPAGE_PTS = 0.5
STT_PCT = 0.0625
EXCHANGE_PCT = 0.035
SEBI_PCT = 0.0001
STAMP_PCT = 0.003
GST_PCT = 18.0

# candle colors
RED = 0
GREEN = 1


def fib_price(high: float, low: float, level: float, orientation: str) -> float:
    span = high - low
    if orientation == "high_to_low":
        return high - level * span
    return low + level * span


def candle_color(open_, close):
    return GREEN if close >= open_ else RED


# wave color patterns, positions 0..6 = W1..W5, A, B
WAVE_PATTERNS = (
    (RED, GREEN, RED),    # W1
    (GREEN, RED, GREEN),  # W2
    (RED, GREEN, RED),    # W3
    (GREEN, RED, GREEN),  # W4
    (RED, GREEN, RED),    # W5
    (GREEN, RED, GREEN),  # A
    (RED, GREEN, RED),    # B
)


class IncrementalATR:
    """Incremental Wilder ATR (period 10 for UT Bot)."""

    def __init__(self, period: int = UT_PERIOD):
        self.period = period
        self.prev_close: Optional[float] = None
        self.atr: Optional[float] = None
        self.tr_hist: list[float] = []

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self.prev_close is not None:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
            self.tr_hist.append(tr)
            if self.atr is None and len(self.tr_hist) == self.period:
                self.atr = sum(self.tr_hist) / self.period
            elif self.atr is not None:
                self.atr = (self.atr * (self.period - 1) + tr) / self.period
        self.prev_close = close
        return self.atr


class UTBot:
    """UT Bot (key=1.0, period=10) on regular 1m candles — exact Pine port.

    Pine source (h=false, src=close):
        xATR = atr(10)
        nLoss = 1 * xATR
        src = close
        trail := iff(src > trail[1] and src[1] > trail[1], max(trail[1], src - nLoss),
                 iff(src < trail[1] and src[1] < trail[1], min(trail[1], src + nLoss),
                 iff(src > trail[1], src - nLoss, src + nLoss)))
        barcolor = src > trail ? green : red
    """

    def __init__(self, key: float = UT_KEY, period: int = UT_PERIOD):
        self.key = key
        self.atr = IncrementalATR(period)
        self.trailing_stop = 0.0
        self.prev_src: Optional[float] = None
        self.prev_trail: Optional[float] = None

    def update_close(self, close: float, high: float, low: float) -> str:
        """Color of the completed bar: "green" | "red" | "none" (pre-warmup)."""
        atr = self.atr.update(high, low, close)
        prev_trail = self.prev_trail if self.prev_trail is not None else 0.0
        if atr is None or atr == 0.0 or self.prev_src is None:
            self.prev_src = close
            self.prev_trail = prev_trail
            return "none"
        n_loss = self.key * atr
        if close > prev_trail and self.prev_src > prev_trail:
            self.trailing_stop = max(prev_trail, close - n_loss)
        elif close < prev_trail and self.prev_src < prev_trail:
            self.trailing_stop = min(prev_trail, close + n_loss)
        elif close > prev_trail:
            self.trailing_stop = close - n_loss
        else:
            self.trailing_stop = close + n_loss
        self.prev_src = close
        self.prev_trail = self.trailing_stop
        return "green" if close > prev_trail else "red"


@dataclass
class Bar:
    gi: int
    day: str
    minute: int
    open: float
    high: float
    low: float
    close: float

    @property
    def color(self):
        return candle_color(self.open, self.close)


@dataclass
class Wave:
    start_gi: int
    end_gi: int
    peak: float
    trough: float
    colors: tuple


@dataclass
class Impulse:
    direction: str          # "bull" | "bear"
    start_gi: int           # origin anchor
    end_gi: int             # W5 anchor
    w1: Wave
    w2: Wave
    w3: Wave
    w4: Wave
    w5: Wave
    timeframe: int = 1
    origin_minute: Optional[int] = None
    w5_second_last_open: Optional[float] = None


@dataclass
class OrderBlock:
    tf: int
    lo: float
    hi: float
    formed_gi: int
    used: bool = False


@dataclass
class CandidateOB:
    ob: OrderBlock
    side: str               # "bull" (price above, pulls down) | "bear"
    untouched_top: float    # bull: highest still-untouched level inside zone
    untouched_bot: float    # bear: lowest still-untouched level inside zone
    dead: bool = False
    eligible_from_gi: int = -1
    entry_mode: str = "pullback"  # pullback | breakout


@dataclass
class QueuedSetup:
    impulse: Impulse
    armed_gi: int
    armed_minute: int


# anchor kinds (match RED/GREEN candle colors: red run -> L, green run -> H)
ANCHOR_L = RED
ANCHOR_H = GREEN


@dataclass
class Anchor:
    gi: int
    minute: int
    price: float
    kind: int           # ANCHOR_L (lowest low of a red run) | ANCHOR_H (highest high of a green run)


class WaveDetector:
    """UT Bot color-run anchor machine — verified exact vs TradingView.

    Anchors are the extremes of UT Bot color runs: a red run contributes its
    lowest low (L), a green run its highest high (H). An anchor completes
    when the run ends (the color flips). Because runs alternate red/green,
    the anchor stream alternates L/H by construction.

    Window: A0(origin) A1(W1) A2(W2) A3(W3) A4(W4) A5(W5) A6(A) A7(B).
    Condition 1 (bull, L-starting window):
        A5.price > A1.price and A5.price > A3.price
        and A0.price < A2.price and A0.price < A4.price
    Bear mirrors it for H-starting windows. On pass at A6 the impulse is
    locked (start_gi = A0.gi, end_gi = A5.gi); the detector arms when A7 (B)
    completes, with armed_gi = B's anchor bar. After arming, the next window
    starts at the anchor after B (the C wave is skipped). Windows may span
    days; the UT Bot and anchor state carry across sessions.
    """

    def __init__(self, timeframe: int = 1, wave_tol: float = WAVE_TOL,
                 carry_anchor_state: bool = False):
        self.timeframe = timeframe
        self.wave_tol = wave_tol
        self.carry_anchor_state = carry_anchor_state
        self.ut = UTBot()
        self.anchors: list[Anchor] = []
        self.win_start = 0
        self.run_color: Optional[str] = None
        self.run_extreme: Optional[Anchor] = None
        self.run_stall = 0
        self.run_anchor_emitted = False
        self.waves: list[Wave] = []
        self.origin: Optional[Anchor] = None
        self.impulse: Optional[Impulse] = None
        self.armed = False
        self.armed_gi: Optional[int] = None
        self.armed_impulse: Optional[Impulse] = None
        self.impulse_start_gi: Optional[int] = None
        self._last_day: Optional[str] = None
        self._pending: Optional[Impulse] = None
        self._last_impulse: Optional[Impulse] = None
        self._last_impulse_day: Optional[str] = None
        self._last_arm_day: Optional[str] = None
        self.history: list[Bar] = []

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _wave_from(a: Anchor) -> Wave:
        return Wave(
            start_gi=a.gi,
            end_gi=a.gi,
            peak=a.price,
            trough=a.price,
            colors=(a.kind,),
        )

    def _check_condition(self, win: list[Anchor]) -> Optional[str]:
        a0, a1, a2, a3, a4, a5, a6 = win
        tol = self.wave_tol
        if a0.kind == ANCHOR_L:                 # bull: 0<2<4 and 1<3<5
            strict_imp = int(a0.price < a2.price) + int(a2.price < a4.price)
            strict_cor = int(a1.price < a3.price) + int(a3.price < a5.price)
            if (strict_imp >= 1 and strict_cor >= 1
                    and a0.price < a2.price + tol and a2.price < a4.price + tol
                    and a1.price < a3.price + tol and a3.price < a5.price + tol
                    and a6.price + tol >= a0.price):
                previous = self._last_impulse
                if (previous is not None and self._last_impulse_day == self._last_day
                        and previous.direction == "bull"
                        and a0.price >= previous.w5.peak):
                    return None
                return "bull"
        else:                                   # bear: 4<2<0 and 5<3<1
            strict_imp = int(a4.price < a2.price) + int(a2.price < a0.price)
            strict_cor = int(a5.price < a3.price) + int(a3.price < a1.price)
            if (strict_imp >= 1 and strict_cor >= 1
                    and a4.price < a2.price - tol and a2.price < a0.price - tol
                    and a5.price < a3.price - tol and a3.price < a1.price - tol
                    and a6.price <= a0.price + tol):
                previous = self._last_impulse
                if (previous is not None and self._last_impulse_day == self._last_day
                        and previous.direction == "bull"
                        and a0.price <= previous.w5.peak):
                    return None
                return "bear"
        return None

    def _advance(self):
        """Drive the machine from newly completed anchors (loops while the
        window can move)."""
        while True:
            n = len(self.anchors)
            if self._pending is not None:
                b_idx = self.win_start + 7      # B = anchor right after A (A6)
                if n <= b_idx:
                    return
                b = self.anchors[b_idx]
                self.armed = True
                self.armed_gi = b.gi
                self.armed_impulse = self._pending
                self._last_arm_day = self._last_day
                self._pending = None
                self.win_start = b_idx + 1      # C skipped; next window after B
                continue
            if n - self.win_start < 7:
                return
            win = self.anchors[self.win_start:self.win_start + 7]
            direction = self._check_condition(win)
            if direction is None:
                self.win_start += 1
                continue
            a0, a5 = win[0], win[5]
            self.origin = a0
            self.impulse_start_gi = a0.gi
            self.waves = [self._wave_from(a) for a in win[1:6]]
            self.impulse = Impulse(
                direction=direction,
                start_gi=a0.gi,
                end_gi=a5.gi,
                w1=self.waves[0], w2=self.waves[1], w3=self.waves[2],
                w4=self.waves[3], w5=self.waves[4],
                timeframe=self.timeframe,
                origin_minute=a0.minute,
                w5_second_last_open=self._w5_second_last_open(a5.gi),
            )
            self._last_impulse = self.impulse
            self._last_impulse_day = self._last_day
            self._pending = self.impulse

    def _w5_second_last_open(self, anchor_gi: int) -> Optional[float]:
        """Open of the candle immediately before the W5 anchor candle."""
        anchor_index = None
        for index, bar in enumerate(self.history):
            if bar.gi == anchor_gi:
                anchor_index = index
        if anchor_index is None or anchor_index == 0:
            return None
        return self.history[anchor_index - 1].open

    # -- main entry ---------------------------------------------------------
    def feed(self, bar: Bar):
        self.history.append(bar)
        col = self.ut.update_close(bar.close, bar.high, bar.low)
        if col == "none":
            return
        previous_day = self._last_day
        if bar.day != previous_day:
            self._last_day = bar.day
            completed_window = (
                len(self.anchors) <= self.win_start
                and self._pending is None
            ) or self._last_arm_day == previous_day
            if not self.carry_anchor_state or completed_window:
                # UT history remains continuous, but completed anchor windows
                # cannot leak into the next session. An incomplete carried
                # window is retained for the cross-session 3m setup.
                self.anchors = []
                self.win_start = 0
                self.run_color = None
                self.run_extreme = None
                self.run_stall = 0
                self.run_anchor_emitted = False
                self._pending = None
                self._last_impulse = None
                self._last_impulse_day = None
                self._last_arm_day = None
        if col == self.run_color and self.run_extreme is not None:
            if self.run_anchor_emitted:
                return
            extended = (
                col == "red" and bar.low < self.run_extreme.price
            ) or (
                col == "green" and bar.high > self.run_extreme.price
            )
            if extended:
                self.run_extreme = Anchor(
                    bar.gi, bar.minute,
                    bar.low if col == "red" else bar.high,
                    ANCHOR_L if col == "red" else ANCHOR_H,
                )
                self.run_stall = 0
            else:
                self.run_stall += 1
                # B is the only anchor that must be confirmed before the
                # color run flips: the pending impulse already identifies the
                # run as the next correction leg. This lets the 14:01 B anchor
                # arm before the long red run ends at 14:27.
                if self._pending is not None and self.run_stall >= 1:
                    self.anchors.append(self.run_extreme)
                    self.run_anchor_emitted = True
                    self._advance()
                    self.run_stall = 0
            return
        # the previous run just ended -> its extreme becomes a completed anchor
        if self.run_extreme is not None:
            if not self.run_anchor_emitted:
                self.anchors.append(self.run_extreme)
                self._advance()
        self.run_color = col
        self.run_stall = 0
        self.run_anchor_emitted = False
        self.run_extreme = Anchor(
            bar.gi, bar.minute,
            bar.low if col == "red" else bar.high,
            ANCHOR_L if col == "red" else ANCHOR_H,
        )

    def consume_arm(self) -> Optional[Impulse]:
        """Return the armed impulse and clear the armed flag (one trade per arm)."""
        if not self.armed:
            return None
        imp = self.armed_impulse
        self.armed = False
        self.armed_impulse = None
        return imp


class OrderBlockTracker:
    """Single-candle OB detection on a per-TF bar stream (patterns A & B)."""

    def __init__(self):
        self.registry: list[OrderBlock] = []

    def feed_tf_bars(self, tf: int, high, low, gi):
        """Feed a TF's per-day bars (arrays aligned by index)."""
        n = len(high)
        seen = set()
        for i in range(1, n - 1):
            hits = []
            # pattern A: X breaks prior low, X+1 breaks X high
            if low[i] < low[i - 1] and high[i + 1] > high[i]:
                hits.append((low[i], high[i], gi[i + 1]))
            # pattern B: X breaks prior high, X+1 breaks X low
            if high[i] > high[i - 1] and low[i + 1] < low[i]:
                hits.append((low[i], high[i], gi[i + 1]))
            for lo, hi, fgi in hits:
                key = (tf, lo, hi, fgi)
                if key in seen:
                    continue
                seen.add(key)
                self.registry.append(OrderBlock(tf=tf, lo=lo, hi=hi, formed_gi=fgi))

    def snapshot(self, impulse: Impulse, bars: Optional[list[Bar]] = None,
                 gis: Optional[list[int]] = None, tol: float = 0.0) -> list[CandidateOB]:
        """Candidate OBs = zones formed within the impulse window (any TF).

        When bars/gis are supplied, bars formed after each OB are replayed so
        zones already mitigated between formation and arming are marked dead or
        consumed instead of being treated as fresh at arming time. `tol` relaxes
        the death boundary (a candle may pierce the zone edge by tol and stay
        alive), matching the engine's live candidate rules.
        """
        side = impulse.direction
        cands = []
        impulse_tf = getattr(impulse, "timeframe", 1)
        for ob in self.registry:
            if (not ob.used
                    and impulse.start_gi <= ob.formed_gi
                    <= impulse.end_gi + max(1, impulse_tf)):
                cands.append(CandidateOB(
                    ob=ob,
                    side=side,
                    untouched_top=ob.hi,
                    untouched_bot=ob.lo,
                ))
        if bars is not None and gis is not None:
            for c in cands:
                start = bisect_right(gis, c.ob.formed_gi)
                for b in bars[start:]:
                    ob = c.ob
                    if c.side == "bull":
                        if b.low < ob.lo - tol:
                            c.dead = True
                            break
                        elif b.low < c.untouched_top:
                            c.untouched_top = b.low
                    else:
                        if b.high > ob.hi + tol:
                            c.dead = True
                            break
                        elif b.high > c.untouched_bot:
                            c.untouched_bot = b.high
        return cands

    def select_for_impulse(self, impulse: Impulse, bars: Optional[list[Bar]] = None,
                           gis: Optional[list[int]] = None,
                           through_gi: Optional[int] = None,
                           tol: float = DEFAULT_TOL,
                           candidate_tf: Optional[int] = None) -> list[CandidateOB]:
        """Select the structural OB rather than every nearby candle range.

        The drawings identify the bullish entry block around W2. For a bearish
        setup, the second-attempt block is the 1m block around W4; the first
        attempt is added later from the C-wave. Keeping this selection narrow
        prevents a nearer but unrelated OB from stealing the entry.
        """
        candidates = self.snapshot(impulse)
        if candidate_tf is not None:
            candidates = [c for c in candidates if c.ob.tf == candidate_tf]
        if not candidates:
            return []

        if impulse.direction == "bull":
            target_gi = impulse.w2.start_gi + 1
            target_price = impulse.w2.trough
            ordered = sorted(
                candidates,
                key=lambda c: (
                    abs(c.ob.formed_gi - target_gi),
                    abs((c.ob.lo + c.ob.hi) / 2.0 - target_price),
                    c.ob.tf,
                ),
            )
            return self._first_live_candidate(ordered, bars, gis, through_gi, tol)

        if impulse.timeframe == 3:
            same_timeframe = [c for c in candidates if c.ob.tf == 3]
            pool = same_timeframe or candidates
        else:
            pool = candidates
        if impulse.timeframe == 3:
            target_gi = impulse.w5.start_gi + impulse.timeframe
            target_price = impulse.w5.trough
        else:
            target_gi = impulse.w4.start_gi + 1
            target_price = impulse.w4.peak
        ordered = sorted(
            pool,
            key=lambda c: (
                abs(c.ob.formed_gi - target_gi),
                abs((c.ob.lo + c.ob.hi) / 2.0 - target_price),
                c.ob.tf,
            ),
        )
        return self._first_live_candidate(ordered, bars, gis, through_gi, tol)

    @staticmethod
    def _first_live_candidate(candidates, bars, gis, through_gi, tol):
        for candidate in candidates:
            if bars is not None and gis is not None:
                start = bisect_right(gis, candidate.ob.formed_gi)
                stop = len(bars)
                if through_gi is not None:
                    stop = bisect_right(gis, through_gi)
                for bar in bars[start:stop]:
                    if candidate.side == "bull":
                        if bar.low < candidate.ob.lo - tol:
                            candidate.dead = True
                            break
                        if bar.low < candidate.untouched_top:
                            candidate.untouched_top = bar.low
                    else:
                        if bar.high > candidate.ob.hi + tol:
                            candidate.dead = True
                            break
                        if bar.high > candidate.untouched_bot:
                            candidate.untouched_bot = bar.high
            if not candidate.dead:
                return [candidate]
        return []


def trade_cost(entry_px: float, exit_px: float, brokerage_per_order: float = 0.0) -> float:
    """Rupees deducted for one option trade (buy + sell legs), NIFTY lot = 65."""
    prem_buy = entry_px * LOT_SIZE
    prem_sell = exit_px * LOT_SIZE
    stt = STT_PCT / 100.0 * prem_sell
    exch = EXCHANGE_PCT / 100.0 * (prem_buy + prem_sell)
    sebi = SEBI_PCT / 100.0 * (prem_buy + prem_sell)
    stamp = STAMP_PCT / 100.0 * prem_buy
    brokerage = brokerage_per_order * 2
    gst = GST_PCT / 100.0 * (brokerage + exch + sebi)
    return round(stt + exch + sebi + stamp + gst + brokerage, 2)


@dataclass
class Position:
    day: str
    side: str                       # "CE" | "PE"
    direction: str                  # "bull" | "bear"
    timeframe: int
    wave_zero_minute: Optional[int]
    strike: int
    entry_min: int
    entry_close: float
    sl: float
    tp: float
    atr_entry: float
    entry_prem: float
    consumed: CandidateOB
    same_tf_ob_tf: Optional[int]
    same_tf_ob_lo: Optional[float]
    same_tf_ob_hi: Optional[float]


class EWOBEngine:
    """Sequential engine over the concatenated 1m index stream.

    resolve_option(day, side, minute, spot_px, strike) -> option close | None
    (injected by runner). The strike is fixed after entry.
    """

    def __init__(self, tol: float = DEFAULT_TOL, sl_mult: float = DEFAULT_SL_MULT,
                 tp_pts: float = DEFAULT_TP_PTS, atr_period: int = ATR_PERIOD,
                 risk_mode: str = RISK_MODE_OB_W5,
                 tp_atr_mult: Optional[float] = None,
                 option_sl_pts: float = 12.0,
                 option_tp_pts: float = 36.0):
        if risk_mode not in (RISK_MODE_OB_W5, RISK_MODE_OB_SAME_TF, RISK_MODE_ATR, RISK_MODE_OPTION_FIXED, RISK_MODE_FIB):
            raise ValueError(f"unsupported risk mode: {risk_mode}")
        self.tol = tol
        self.sl_mult = sl_mult
        self.tp_pts = tp_pts
        self.atr_period = atr_period
        self.risk_mode = risk_mode
        self.tp_atr_mult = sl_mult if tp_atr_mult is None else tp_atr_mult
        self.option_sl_pts = option_sl_pts
        self.option_tp_pts = option_tp_pts
        self.opt_map: Optional[dict] = None
        self.wave_detectors = {
            tf: WaveDetector(tf, WAVE_TOL, carry_anchor_state=True)
            for tf in (1, 2, 3, 5)
        }
        self.wave = self.wave_detectors[1]
        self.obs = OrderBlockTracker()
        self.candidates: list[CandidateOB] = []
        self.armed = False
        self.positions: list[Position] = []
        self.trades: list[dict] = []
        self.resolve_option: Callable[[str, str, int, float, Optional[int]], Optional[float]] = None
        self._trs: list[float] = []
        self._prev_close: Optional[float] = None
        self._last_close: Optional[float] = None
        self._bars: list[Bar] = []
        self._gis: list[int] = []
        self._active_impulse: Optional[Impulse] = None
        self._fallback_candidate: Optional[CandidateOB] = None
        self._dynamic_candidate: Optional[CandidateOB] = None
        self._same_tf_stop_ob: Optional[OrderBlock] = None
        self._dynamic_added = False
        self._arm_gi: Optional[int] = None
        self._arm_minute: Optional[int] = None
        self._setup_queue: list[QueuedSetup] = []

    @property
    def pos(self) -> Optional[Position]:
        """Most recently opened position (None when flat)."""
        return self.positions[-1] if self.positions else None

    # -- helpers ------------------------------------------------------------
    def _update_tr(self, bar: Bar):
        h, l, c = bar.high, bar.low, bar.close
        tr = h - l
        if self._prev_close is not None:
            tr = max(tr, abs(h - self._prev_close), abs(l - self._prev_close))
        self._prev_close = c
        self._trs.append(tr)
        if len(self._trs) > self.atr_period:
            self._trs.pop(0)

    def _atr10(self) -> Optional[float]:
        if len(self._trs) < self.atr_period:
            return None
        return sum(self._trs[-self.atr_period:]) / self.atr_period

    def _update_candidates(self, bar: Bar):
        for c in self.candidates:
            if c.dead:
                continue
            ob = c.ob
            if c.entry_mode == "breakout":
                continue
            if c.side == "bull":
                if bar.low < ob.lo - self.tol:
                    c.dead = True                # traded through the whole zone
                elif bar.low < c.untouched_top:
                    c.untouched_top = bar.low    # consume the touched upper part
            else:
                if bar.high > ob.hi + self.tol:
                    c.dead = True
                elif bar.high > c.untouched_bot:
                    c.untouched_bot = bar.high   # consume the touched lower part

    def _add_cwave_order_block(self, bar: Bar):
        """Add the single first-attempt C-wave block for a bearish setup.

        The first C-wave block is the raw 1m candle at the configured delay
        after B. It must be revisited on a later candle. If that block is
        pierced, the preselected W4 block is the only fallback.
        """
        if (self._active_impulse is None
                or self._active_impulse.direction != "bear"
                or self._active_impulse.timeframe != 1):
            return
        if self._dynamic_added or self._arm_gi is None:
            return
        if self._arm_minute is None or bar.minute < self._arm_minute + POST_B_OB_DELAY:
            return
        ob = OrderBlock(tf=1, lo=bar.low, hi=bar.high, formed_gi=bar.gi)
        self._dynamic_candidate = CandidateOB(
            ob=ob,
            side="bear",
            untouched_top=ob.hi,
            untouched_bot=ob.lo,
            eligible_from_gi=bar.gi + 1,
        )
        self.candidates.insert(0, self._dynamic_candidate)
        self._dynamic_added = True

    def _activate_setup(self, setup: QueuedSetup):
        """Make one queued timeframe setup the active entry search."""
        self._active_impulse = setup.impulse
        self._arm_gi = setup.armed_gi
        self._arm_minute = setup.armed_minute
        # The 1m pass replays mitigation before B to reject stale micro-OBs.
        # Higher-timeframe drawings define their OB as the structural block at
        # the wave endpoint; those blocks remain eligible until the live entry
        # pass, including the cross-session 3m setup.
        replay = setup.impulse.timeframe == 1
        selected = self.obs.select_for_impulse(
            setup.impulse,
            bars=self._bars if replay else None,
            gis=self._gis if replay else None,
            through_gi=setup.armed_gi if replay else None,
            tol=self.tol,
        )
        same_tf = self.obs.select_for_impulse(
            setup.impulse,
            bars=self._bars if replay else None,
            gis=self._gis if replay else None,
            through_gi=setup.armed_gi if replay else None,
            tol=self.tol,
            candidate_tf=setup.impulse.timeframe,
        )
        self._same_tf_stop_ob = same_tf[0].ob if same_tf else None
        self._fallback_candidate = selected[0] if selected else None
        if self._fallback_candidate is not None:
            self._fallback_candidate.eligible_from_gi = setup.armed_gi + 1
            if (setup.impulse.timeframe == 3
                    and setup.impulse.direction == "bear"):
                self._fallback_candidate.entry_mode = "breakout"
        self._dynamic_candidate = None
        self._dynamic_added = False
        self.candidates = [self._fallback_candidate] if self._fallback_candidate else []
        self.armed = bool(self.candidates)
        if not self.armed:
            self._active_impulse = None
            self._arm_gi = None
            self._arm_minute = None

    def _discard_active_setup(self):
        self._active_impulse = None
        self._fallback_candidate = None
        self._dynamic_candidate = None
        self._dynamic_added = False
        self._arm_gi = None
        self._arm_minute = None
        self._same_tf_stop_ob = None
        self.candidates = []
        self.armed = False

    def _queue_wave_arms(self, bar: Bar, wave_bars):
        """Collect all detectors that armed on this 1m bar in TF order."""
        arms_before = len(self._setup_queue)
        for tf, tf_bar in sorted(wave_bars, key=lambda item: item[0]):
            detector = self.wave_detectors[tf]
            detector.feed(tf_bar)
            impulse = detector.consume_arm()
            if impulse is None or detector.armed_gi is None:
                continue
            self._setup_queue.append(QueuedSetup(
                impulse=impulse,
                armed_gi=detector.armed_gi,
                armed_minute=bar.minute,
            ))

        if (len(self._setup_queue) > arms_before
                and self._active_impulse is not None
                and self.armed and not self.positions):
            self._setup_queue.append(QueuedSetup(
                impulse=self._active_impulse,
                armed_gi=self._arm_gi,
                armed_minute=self._arm_minute,
            ))
            self._discard_active_setup()

        if (self._active_impulse is not None and self.armed and not self.positions
                and self._arm_minute is not None and self._setup_queue):
            age = bar.minute - self._arm_minute
            limit = SETUP_MAX_AGE_MINUTES * self._active_impulse.timeframe
            if age > limit and any(
                    s.impulse.timeframe >= self._active_impulse.timeframe
                    for s in self._setup_queue
            ):
                self._discard_active_setup()

        self._activate_queued_setup_if_flat()

    def _activate_queued_setup_if_flat(self):
        if self.positions or self.armed:
            return
        while self._setup_queue and self._active_impulse is None:
            index = max(
                range(len(self._setup_queue)),
                key=lambda i: (
                    self._setup_queue[i].armed_gi,
                    self._setup_queue[i].impulse.timeframe,
                ),
            )
            self._activate_setup(self._setup_queue.pop(index))

    # -- per-bar processing -------------------------------------------------
    def feed(self, bar: Bar, wave_bars=None):
        self._update_tr(bar)
        self._last_close = bar.close
        self._bars.append(bar)
        self._gis.append(bar.gi)
        if len(self._bars) > 3000:
            del self._bars[:-3000]
            del self._gis[:-3000]

        # 1) advance all wave detectors; arms are queued by timeframe
        if wave_bars is None:
            wave_bars = [(1, bar)]
        self._queue_wave_arms(bar, wave_bars)

        self._add_cwave_order_block(bar)

        # 2) monitor candidates for consumption/death
        self._update_candidates(bar)
        if (self.armed and not self.positions and self.candidates
                and all(candidate.dead for candidate in self.candidates)):
            self._discard_active_setup()

        # 3) manage open positions
        self._check_exits(bar)
        self._activate_queued_setup_if_flat()

        # 4) look for an entry (only after the arming bar itself)
        if (self.armed and not self.positions and bar.minute < SESSION_END
                and self._arm_gi is not None and bar.gi > self._arm_gi):
            self._check_entry(bar)

    def _check_entry(self, bar: Bar):
        for c in self.candidates:
            if c.dead:
                continue
            if bar.gi < c.eligible_from_gi:
                continue
            ob = c.ob
            if c.entry_mode == "breakout":
                qualifies = bar.low <= ob.lo + self.tol
            elif c.side == "bull":
                qualifies = (bar.low <= c.untouched_top + self.tol
                             and bar.low >= ob.lo - self.tol)
            else:
                qualifies = (bar.high >= c.untouched_bot - self.tol
                             and bar.high <= ob.hi + self.tol)
            if not qualifies:
                continue
            atr = self._atr10()
            if atr is None:
                continue
            side = "CE" if c.side == "bull" else "PE"
            atm = int(round(bar.close / 50) * 50)
            strike = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
            prem = self.resolve_option(bar.day, side, bar.minute, bar.close, strike)
            if prem is None:
                continue
            if self.risk_mode == RISK_MODE_OPTION_FIXED:
                sl = prem - self.option_sl_pts
                tp = prem + self.option_tp_pts
            elif self.risk_mode == RISK_MODE_FIB:
                imp = self._active_impulse
                if imp is not None:
                    # origin bar
                    origin_bar = None
                    for b in reversed(self._bars):
                        if b.gi == imp.start_gi:
                            origin_bar = b
                            break
                    if c.side == "bull":
                        fib_high = imp.w5.peak
                        fib_low = origin_bar.low if origin_bar is not None else min(imp.w1.trough, imp.w5.trough)
                        orientation = "high_to_low"
                        tp = fib_price(fib_high, fib_low, 0.0, orientation)
                        sl = fib_price(fib_high, fib_low, 1.25, orientation)
                    else:
                        fib_high = origin_bar.high if origin_bar is not None else imp.w1.peak
                        fib_low = imp.w5.trough
                        orientation = "low_to_high"
                        tp = fib_price(fib_high, fib_low, 0.0, orientation)
                        sl = fib_price(fib_high, fib_low, 1.25, orientation)
                else:
                    tp = bar.close + (self.tp_pts if c.side == "bull" else -self.tp_pts)
                    sl = bar.close - (self.sl_mult * atr if c.side == "bull" else -self.sl_mult * atr)
            elif c.side == "bull":
                if self.risk_mode == RISK_MODE_ATR:
                    sl = bar.close - self.sl_mult * atr
                    tp = bar.close + self.tp_atr_mult * atr
                else:
                    stop_ob = (self._same_tf_stop_ob
                               if self.risk_mode == RISK_MODE_OB_SAME_TF
                               and self._same_tf_stop_ob is not None else c.ob)
                    sl = stop_ob.lo
                    tp = (self._active_impulse.w5_second_last_open
                          if self._active_impulse is not None
                          and self._active_impulse.w5_second_last_open is not None
                          else bar.close + self.tp_pts)
            else:
                if self.risk_mode == RISK_MODE_ATR:
                    sl = bar.close + self.sl_mult * atr
                    tp = bar.close - self.tp_atr_mult * atr
                else:
                    stop_ob = (self._same_tf_stop_ob
                               if self.risk_mode == RISK_MODE_OB_SAME_TF
                               and self._same_tf_stop_ob is not None else c.ob)
                    sl = stop_ob.hi
                    tp = (self._active_impulse.w5_second_last_open
                          if self._active_impulse is not None
                          and self._active_impulse.w5_second_last_open is not None
                          else bar.close - self.tp_pts)
            self.positions.append(Position(
                day=bar.day, side=side, direction=c.side, strike=strike,
                timeframe=(self._active_impulse.timeframe
                           if self._active_impulse is not None else 1),
                wave_zero_minute=(self._active_impulse.origin_minute
                                  if self._active_impulse is not None else None),
                entry_min=bar.minute, entry_close=bar.close, sl=sl, tp=tp,
                atr_entry=atr, entry_prem=prem, consumed=c,
                same_tf_ob_tf=(self._same_tf_stop_ob.tf
                               if self._same_tf_stop_ob is not None else None),
                same_tf_ob_lo=(self._same_tf_stop_ob.lo
                               if self._same_tf_stop_ob is not None else None),
                same_tf_ob_hi=(self._same_tf_stop_ob.hi
                               if self._same_tf_stop_ob is not None else None),
            ))
            # consume the touched portion; one trade per arm
            c.ob.used = True
            if c.side == "bull":
                c.untouched_top = min(c.untouched_top, bar.low)
            else:
                c.untouched_bot = max(c.untouched_bot, bar.high)
            self.armed = False
            self._active_impulse = None
            self._fallback_candidate = None
            self._dynamic_candidate = None
            self._dynamic_added = False
            self._arm_gi = None
            self._arm_minute = None
            self._same_tf_stop_ob = None
            break

    def _option_bar(self, day: str, side: str, minute: int, strike: int):
        if self.opt_map is None:
            return None
        path = self.opt_map.get(day)
        if path is None:
            return None
        try:
            import opt_futures_quad as source
            rec = source.cached_option(str(path))
            if rec is None:
                return None
            df, groups, prefix = rec
            if prefix is None:
                return None
            symbol = f"{prefix}{strike}{side}"
            sl = source.make_slice(df, groups, symbol)
            if sl is None:
                return None
            bar = source.bar_at(sl, minute)
            if bar is None:
                return None
            return bar  # (open, high, low, close)
        except Exception:
            return None

    def _check_exits(self, bar: Bar):
        for p in list(self.positions):
            if self.risk_mode == RISK_MODE_OPTION_FIXED:
                opt_bar = self._option_bar(bar.day, p.side, bar.minute, p.strike)
                if opt_bar is None:
                    continue
                opt_high, opt_low = float(opt_bar[1]), float(opt_bar[2])
                opt_close = float(opt_bar[3])
                hit = None
                if opt_low <= p.sl:
                    hit = "SL"
                elif opt_high >= p.tp:
                    hit = "TP"
                if hit is None:
                    continue
                prem = opt_close
            else:
                if p.direction == "bull":
                    hit = "SL" if bar.low <= p.sl else ("TP" if bar.high >= p.tp else None)
                else:
                    hit = "SL" if bar.high >= p.sl else ("TP" if bar.low <= p.tp else None)
                if hit is None:
                    continue
                prem = self.resolve_option(bar.day, p.side, bar.minute, bar.close, p.strike)
            if prem is None:
                continue
            if self.risk_mode == RISK_MODE_OPTION_FIXED:
                pts = prem - p.entry_prem
            else:
                pts = (bar.close - p.entry_close) if p.direction == "bull" else (p.entry_close - bar.close)
            prem_diff = (prem - p.entry_prem)  # option premium diff (both directions)
            fee = trade_cost(p.entry_prem, prem)
            self.trades.append({
                "date": p.day,
                "entry_min": p.entry_min,
                "exit_min": bar.minute,
                "side": p.side,
                "strike": p.strike,
                "timeframe": p.timeframe,
                "wave_zero_minute": p.wave_zero_minute,
                "direction": p.direction,
                "entry": round(p.entry_close, 2),
                "exit": round(bar.close, 2),
                "sl": round(p.sl, 2),
                "tp": round(p.tp, 2),
                "atr_entry": round(p.atr_entry, 6),
                "entry_prem": round(p.entry_prem, 6),
                "entry_ob_tf": p.consumed.ob.tf,
                "entry_ob_lo": round(p.consumed.ob.lo, 6),
                "entry_ob_hi": round(p.consumed.ob.hi, 6),
                "same_tf_ob_tf": p.same_tf_ob_tf,
                "same_tf_ob_lo": p.same_tf_ob_lo,
                "same_tf_ob_hi": p.same_tf_ob_hi,
                "exit_reason": hit,
                "pts": round(pts, 2),
                "pts_net": round(pts - 2 * SLIPPAGE_PTS, 2),
                "fee": fee,
                "rs_net": round(prem_diff * LOT_SIZE - fee, 2),
            })
            self.positions.remove(p)

    def close_day(self):
        """Force-flat any open positions at the last close (end of session)."""
        if self.positions:
            last_close = self._last_close if self._last_close is not None else self.positions[-1].entry_close
            for p in list(self.positions):
                prem = self.resolve_option(p.day, p.side, SESSION_END, last_close, p.strike)
                if prem is None:
                    # degenerate: no exit bar -> mark flat with entry premium (0 P&L)
                    prem = p.entry_prem
                if self.risk_mode == RISK_MODE_OPTION_FIXED:
                    # EOD on option-fixed uses premium at session end
                    opt_bar = self._option_bar(p.day, p.side, SESSION_END, p.strike)
                    if opt_bar is not None:
                        last_close = float(opt_bar[3])
                        prem = self.resolve_option(p.day, p.side, SESSION_END, last_close, p.strike) or prem
                    pts = prem - p.entry_prem
                else:
                    pts = (last_close - p.entry_close) if p.direction == "bull" else (p.entry_close - last_close)
                prem_diff = (prem - p.entry_prem)
                fee = trade_cost(p.entry_prem, prem)
                self.trades.append({
                    "date": p.day,
                    "entry_min": p.entry_min,
                    "exit_min": SESSION_END,
                    "side": p.side,
                    "strike": p.strike,
                    "timeframe": p.timeframe,
                    "wave_zero_minute": p.wave_zero_minute,
                    "direction": p.direction,
                    "entry": round(p.entry_close, 2),
                    "exit": round(last_close, 2),
                    "sl": round(p.sl, 2),
                    "tp": round(p.tp, 2),
                    "atr_entry": round(p.atr_entry, 6),
                    "entry_prem": round(p.entry_prem, 6),
                    "entry_ob_tf": p.consumed.ob.tf,
                    "entry_ob_lo": round(p.consumed.ob.lo, 6),
                    "entry_ob_hi": round(p.consumed.ob.hi, 6),
                    "same_tf_ob_tf": p.same_tf_ob_tf,
                    "same_tf_ob_lo": p.same_tf_ob_lo,
                    "same_tf_ob_hi": p.same_tf_ob_hi,
                    "exit_reason": "EOD",
                    "pts": round(pts, 2),
                    "pts_net": round(pts - 2 * SLIPPAGE_PTS, 2),
                    "fee": fee,
                    "rs_net": round(prem_diff * LOT_SIZE - fee, 2),
                })
            self.positions.clear()
        self._setup_queue.clear()
        self._active_impulse = None
        self.candidates = []
        self.armed = False
