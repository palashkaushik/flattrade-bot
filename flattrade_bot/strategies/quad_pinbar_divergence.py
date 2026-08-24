"""Optuna Optimized ATR F6 Strategy Engine (Option A — Champion Strategy).

Strategy Specification:
  - Stochastics: S1=(12,3), S2=(14,3), S3=(40,4), S4=(50,10)
  - Timeframes: Concurrent 4-TF scanning (1m, 2m, 3m, 5m)
  - Triggers:
      1. F6 Flag No-Divergence (IMMEDIATE): First bar where S4 >= 79.5 AND S1 <= 25.0 (no div/pinbar required).
      2. Super/Flag PinBar Setup: S4 >= 79.5 & S1 <= 25.0 (Flag) or ALL S1..S4 <= 20.5 (Super)
         AND Bullish Trough Divergence confirmed, triggered by BullishPinBar vicinity breakout.
  - Embedded S4 Reversal:
      - If S4 <= 20.0 for > 25 consecutive bars, trade direction is reversed (buy PE instead of CE, CE instead of PE).
  - Dynamic ATR Exits:
      - ATR(10) lookback: SL = ATR * 3.0 points, TP = ATR * 6.0 points.
      - Fallback SL/TP if ATR <= 0.5: 1m=(6.0, 30.0), 2m=(10.0, 15.0), 3m=(8.0, 25.0), 5m=(10.0, 35.0).
  - Additional Exits:
      - 1m Bearish Peak Divergence Exit (divergence reversal while holding).
      - Daily Shutdown: Max Loss = Rs 2,000.
      - Consecutive Loss Limit = 8.
      - EOD Exit: 15:00.
"""

from collections import deque
from typing import Dict, Any, Optional, Tuple, List
from flattrade_bot.config import settings
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.divergence import DivergenceEngine

TF_SPECS = {
    "1m": (1, 10, 6.0, 30.0),
    "2m": (2, 5, 10.0, 15.0),
    "3m": (3, 4, 8.0, 25.0),
    "5m": (5, 3, 10.0, 35.0),
}


class IncrementalATR:
    """Calculates Wilder's Incremental Average True Range (ATR)."""

    def __init__(self, period: int = 10):
        self.period = period
        self._buf = deque(maxlen=period)
        self.atr: Optional[float] = None
        self.prev_close: Optional[float] = None
        self._n = 0

    def update(self, h: float, l: float, c: float) -> Optional[float]:
        tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close)) if self.prev_close is not None else h - l
        self._buf.append(tr)
        self._n += 1
        self.prev_close = c
        if self._n < self.period:
            self.atr = None
        elif self._n == self.period:
            self.atr = sum(self._buf) / self.period
        else:
            self.atr = (self.atr * (self.period - 1) + tr) / self.period
        return self.atr


class ParamStoch:
    """4-Stochastic instance with configurable periods (Option A defaults: 12,3 / 14,3 / 40,4 / 50,10)."""

    def __init__(self, s1_k: int = 12, s1_d: int = 3, s4_k: int = 50):
        self.s1 = IncrementalStochastic(s1_k, s1_d)
        self.s2 = IncrementalStochastic(14, 3)
        self.s3 = IncrementalStochastic(40, 4)
        self.s4 = IncrementalStochastic(s4_k, 10)
        self.latest_s1: Optional[float] = None
        self.latest_s2: Optional[float] = None
        self.latest_s3: Optional[float] = None
        self.latest_s4: Optional[float] = None

    def push(self, h: float, l: float, c: float) -> Dict[str, Optional[float]]:
        self.latest_s1 = self.s1.push(h, l, c)
        self.latest_s2 = self.s2.push(h, l, c)
        self.latest_s3 = self.s3.push(h, l, c)
        self.latest_s4 = self.s4.push(h, l, c)
        return {
            "s1d": self.latest_s1,
            "s2d": self.latest_s2,
            "s3d": self.latest_s3,
            "s4d": self.latest_s4,
        }


class TFTracker:
    """Tracks PinBar breakout + Divergence setup on a single timeframe."""

    def __init__(self, max_lookback: int, s1_k: int = 12, s1_d: int = 3, s4_k: int = 50,
                 f6_s4_thresh: float = 79.5, f6_s1_thresh: float = 20.5, atr_period: int = 10):
        self.max_lookback = max_lookback
        self.stoch = ParamStoch(s1_k, s1_d, s4_k)
        self.div = DivergenceEngine()
        self.hist: List[Candle] = []
        self.setup_active = False
        self.stype = ""
        self.prev_s1: Optional[float] = None
        self.s4_embedded_count = 0
        self.flag_ready = False
        self.super_ready = False
        self.has_bull_divergence = False
        self._armed_bullish_divergence = None
        self.atr = IncrementalATR(atr_period)
        self.f6_s4_thresh = f6_s4_thresh
        self.f6_s1_thresh = f6_s1_thresh

    def reset_session_state(self) -> None:
        """Drop pending setups without discarding indicator warm-up values."""
        self.hist.clear()
        self.setup_active = False
        self.stype = ""
        self.flag_ready = False
        self.super_ready = False
        self.has_bull_divergence = False
        self._armed_bullish_divergence = None
        self.s4_embedded_count = 0

    def push(self, c: Candle) -> Tuple[bool, bool, str, float, Optional[float]]:
        self.hist.append(c)
        if len(self.hist) > 40:
            self.hist.pop(0)

        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
        atr_val = self.atr.update(c.high, c.low, c.close)
        self.prev_s1 = s1

        if s4 is not None:
            self.s4_embedded_count = self.s4_embedded_count + 1 if s4 <= 20.0 else 0
        is_reverse_mode = self.s4_embedded_count >= 25

        self.div.update(c.close, s1, low_price=c.low, high_price=c.high)
        has_bull_div = self.div.has_bullish_trough_divergence()
        bullish_divergence_id = self.div.bullish_divergence_id()

        is_flag = s4 is not None and s1 is not None and s4 >= self.f6_s4_thresh and s1 <= self.f6_s1_thresh
        is_super = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
        self.flag_ready = is_flag
        self.super_ready = is_super
        self.has_bull_divergence = has_bull_div

        if (
            (is_flag or is_super)
            and has_bull_div
            and bullish_divergence_id != self._armed_bullish_divergence
        ):
            self.setup_active = True
            self.stype = "super" if is_super else "flag"
            self._armed_bullish_divergence = bullish_divergence_id

        is_reverse = is_reverse_mode and self.stype == "super"
        triggered = False

        if self.setup_active and len(self.hist) >= 2:
            if BullishPinBarDetector.check_vicinity_breakout(self.hist, self.max_lookback):
                triggered = True
                self.setup_active = False

        return triggered, is_reverse, self.stype, c.close, atr_val


class FlagNoDivScanner:
    """F6 Scanner: Fires immediately on the first bar where S4 >= 79.5 and S1 <= 25.0."""

    def __init__(self, s1_k: int = 12, s1_d: int = 3, s4_k: int = 50,
                 f6_s4_thresh: float = 79.5, f6_s1_thresh: float = 20.5):
        self.s1 = IncrementalStochastic(s1_k, s1_d)
        self.s4 = IncrementalStochastic(s4_k, 10)
        self.f6_s4_thresh = f6_s4_thresh
        self.f6_s1_thresh = f6_s1_thresh
        self._fired = False

    def reset_session_state(self) -> None:
        """Allow a new session to emit its first qualifying F6 bar."""
        self._fired = False

    def push(self, h: float, l: float, c: float) -> bool:
        s1v = self.s1.push(h, l, c)
        s4v = self.s4.push(h, l, c)
        if s1v is None or s4v is None:
            return False
        flag = s4v >= self.f6_s4_thresh and s1v <= self.f6_s1_thresh
        if flag and not self._fired:
            self._fired = True
            return True
        if not flag:
            self._fired = False
        return False


class MTFTracker:
    """Concurrent Multi-Timeframe Tracker across 1m, 2m, 3m, and 5m timeframes."""

    def __init__(self, s1_k: int = 12, s1_d: int = 3, s4_k: int = 50,
                 f6_s4_thresh: float = 79.5, f6_s1_thresh: float = 20.5, atr_period: int = 10):
        self.trackers = {tf: TFTracker(spec[1], s1_k, s1_d, s4_k, f6_s4_thresh, f6_s1_thresh, atr_period)
                         for tf, spec in TF_SPECS.items()}
        self.f6scans = {tf: FlagNoDivScanner(s1_k, s1_d, s4_k, f6_s4_thresh, f6_s1_thresh)
                        for tf in TF_SPECS}
        self.bufs: Dict[str, List[Candle]] = {tf: [] for tf in TF_SPECS}
        self._last_minute: Optional[int] = None
        self.reverse_regime_active = False

    def _reset_timeframe_buffers_if_session_rolled(self, minute: int) -> None:
        if self._last_minute is not None and minute > 0 and minute < self._last_minute:
            self.bufs = {tf: [] for tf in TF_SPECS}
            for tf in TF_SPECS:
                self.trackers[tf].reset_session_state()
                self.f6scans[tf].reset_session_state()
            self.reverse_regime_active = False
        self._last_minute = minute

    def push_1m(self, c1m: Candle) -> List[Tuple[str, bool, str, float, Optional[float]]]:
        out = []
        m = c1m.minute  # absolute minute of day (e.g. 558 = 09:18)
        self._reset_timeframe_buffers_if_session_rolled(m)
        for tf, spec in TF_SPECS.items():
            period = spec[0]
            self.bufs[tf].append(c1m)

            # Clock-aligned boundary: fire when absolute minute-of-day is divisible by period.
            # e.g. 3m candles close at 09:18 (558), 09:21 (561), 09:24 (564) — all ÷3 == 0.
            # If minute=0 (no timestamp parsed), fall back to count-based: fire every 'period' bars.
            if m > 0:
                is_boundary = (m % period == 0) and len(self.bufs[tf]) >= 1
            else:
                is_boundary = (len(self.bufs[tf]) >= period)

            if is_boundary:
                buf = self.bufs[tf]
                self.bufs[tf] = []
                ctf = Candle(open=buf[0].open, high=max(x.high for x in buf),
                             low=min(x.low for x in buf), close=buf[-1].close,
                             minute=buf[-1].minute)
                trig, is_rev, stype, px, atr_val = self.trackers[tf].push(ctf)
                if trig:
                    out.append((tf, is_rev, stype, px, atr_val))
                if self.f6scans[tf].push(ctf.high, ctf.low, ctf.close):
                    out.append((tf, False, "flag_nodiv", ctf.close, atr_val))
        self.reverse_regime_active = any(
            tracker.s4_embedded_count >= 25
            for tracker in self.trackers.values()
        )
        return [
            (
                tf,
                is_reverse or (self.reverse_regime_active and signal_type == "super"),
                signal_type,
                entry,
                atr_value,
            )
            for tf, is_reverse, signal_type, entry, atr_value in out
        ]


    def get_stoch_summary(self) -> Dict[str, Dict[str, Any]]:
        """Returns stochastic and ATR metrics per timeframe."""
        res = {}
        for tf, trk in self.trackers.items():
            s1 = trk.stoch.latest_s1
            s2 = trk.stoch.latest_s2
            s3 = trk.stoch.latest_s3
            s4 = trk.stoch.latest_s4
            atr_val = trk.atr.atr
            is_f6_ready = (s4 is not None and s1 is not None and s4 >= trk.f6_s4_thresh and s1 <= trk.f6_s1_thresh)
            res[tf] = {
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "s4": s4,
                "atr": atr_val,
                "f6_ready": is_f6_ready,
                "super_ready": trk.super_ready,
                "super_setup_active": trk.setup_active and trk.stype == "super",
                "setup_type": trk.stype,
                "s4_embedded": trk.s4_embedded_count >= 25,
            }
        return res


class QuadPinbarDivergenceStrategy:
    """Option A — Optuna Optimized ATR F6 Strategy Coordinator."""

    def __init__(
        self,
        s1_spec: Tuple[int, int] = (12, 3),
        s4_k: int = 50,
        f6_s4_thresh: float = 79.5,
        f6_s1_thresh: float = 20.5,
        atr_period: int = 10,
        atr_sl_mult: float = 3.0,
        atr_tp_mult: float = 6.0,
        sl_points: Optional[float] = None,
        tp_points: Optional[float] = None,
    ):
        self.s1_k, self.s1_d = s1_spec
        self.s4_k = s4_k
        self.f6_s4_thresh = f6_s4_thresh
        self.f6_s1_thresh = f6_s1_thresh
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.sl_points = sl_points or settings.SL_POINTS
        self.tp_points = tp_points or settings.TP_POINTS

        self.ce_tracker = MTFTracker(self.s1_k, self.s1_d, self.s4_k, self.f6_s4_thresh, self.f6_s1_thresh, self.atr_period)
        self.pe_tracker = MTFTracker(self.s1_k, self.s1_d, self.s4_k, self.f6_s4_thresh, self.f6_s1_thresh, self.atr_period)

    @staticmethod
    def get_itm2_strikes(spot_price: float) -> Tuple[int, int]:
        """Calculates 2nd ITM strikes: CE = ATM - 100, PE = ATM + 100."""
        atm = int(round(spot_price / 50.0) * 50)
        return atm - 100, atm + 100

    def get_stoch_summary(self, side: str = "CE") -> Dict[str, Dict[str, Any]]:
        tracker = self.ce_tracker if side == "CE" else self.pe_tracker
        return tracker.get_stoch_summary()

    def reset_side(self, side: str) -> None:
        """Rebuilds a side's MTF tracker from scratch (used on strike rotation).

        The old tracker's stochastic/ATR windows hold candles from the previous
        contract; mixing a new contract's prices into them corrupts S1..S4 and
        ATR for up to ~60 bars. Recreating the tracker gives a clean slate.
        """
        tracker = MTFTracker(self.s1_k, self.s1_d, self.s4_k,
                             self.f6_s4_thresh, self.f6_s1_thresh, self.atr_period)
        if side == "CE":
            self.ce_tracker = tracker
        else:
            self.pe_tracker = tracker

    def push_spot_candle(self, candle_1m: Candle, side: str) -> List[Tuple[str, bool, str, float, float, float]]:
        """Pushes a 1m candle for CE or PE chart.

        Returns list of triggers: (tf, is_reverse, stype, entry_price, sl_points, tp_points)
        """
        tracker = self.ce_tracker if side == "CE" else self.pe_tracker
        triggers = tracker.push_1m(candle_1m)
        out = []
        for tf, is_rev, stype, px, atr_val in triggers:
            sl_fallback, tp_fallback = TF_SPECS[tf][2], TF_SPECS[tf][3]
            if atr_val is not None and atr_val > 0.5:
                sl_pts = round(atr_val * self.atr_sl_mult, 2)
                tp_pts = round(atr_val * self.atr_tp_mult, 2)
            else:
                sl_pts = sl_fallback
                tp_pts = tp_fallback
            out.append((tf, is_rev, stype, px, sl_pts, tp_pts))
        return out
