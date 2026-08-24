"""Pocket Money Strategy — live 10-second scalping engine.

Ports the PocketMoney backtest (artifacts/f6_hybrid/pocket_money_backtest.py)
to live trading:

  - Official timeframe: 10 seconds (bars built from ~1s quote polls)
  - 4 stochastics on EVERY option chart: S1(9,3) S2(14,3) S3(40,4) S4(60,10)
  - FLAG trigger: S1 %D crosses from > 20.5 down to <= 20.5 while S4 %D >= 79.5
    (NO divergence required — flags are taken on the plain setup)
  - SUPER trigger: S1 %D crosses back ABOVE 20 while a bullish trough
    divergence is confirmed on the contract's own chart — the current
    confirmed trough (fully formed when S1 crossed above 20) has a LOWER low
    than the previous confirmed trough while its S1 OR S2 is HIGHER. Enter on
    the close of the S1-crosses-20 bar.
  - 2nd ITM strike only (CE at ATM-100, PE at ATM+100), rollover watch ATM±50
  - Index 5m filter (UT Bot key=1.0 period=10 on Heikin-Ashi close + linreg(11)
    per OHLC component, white line = sma(linreg_close, 11)); the chart this
    strategy is traded on is a Heikin-Ashi chart, so candles are HA-converted
    before UT Bot / LinReg:
        UT green AND close > white line  -> CE only
        UT red   AND close < white line  -> PE only
  - Option-chart divergence engine (copied from the F6 strategy): feeds S1+S2
    so a SUPER entry requires the confirmed trough divergence on that
    contract's own chart, computed on 1-minute bars so live matches the 1m
    backtest proxy. FLAG entries skip the divergence gate entirely.
  - Entry at trigger bar close; SL = entry - 7 / TP = entry + 7 (SL priority)
  - One position at a time; no new trades at/after 15:00; EOD exit 15:00
  - 4-consecutive-loss block (enforced by RiskManager)

Warmup (any start time):
  - Option stochastics: prior day's 1m rows (if the current series traded then)
    PLUS today's 1m rows up to the current minute. By mid/late session today's
    rows alone fully cover S4 (70 bars); prior-day rows cover the first ~70
    minutes of a fresh morning start. Bars are only used to warm indicators —
    triggers fire exclusively on new 10s bars (no replay on restart).
  - Index filter: 12 prior days of 1m spot rows + today's partial day.

This module is pure/synchronous — the async tick loop in main.py drives it.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from flattrade_bot.indicators.divergence import DivergenceEngine

logger = logging.getLogger("flattrade_bot.pocket_money")

# ── Strategy constants (mirror the backtest) ─────────────────────────────
S1_SPEC: Tuple[int, int] = (9, 3)
S2_SPEC: Tuple[int, int] = (14, 3)
S3_SPEC: Tuple[int, int] = (40, 4)
S4_SPEC: Tuple[int, int] = (60, 10)
S1_THRESH: float = 20.5      # FLAG touch level (from neutral zone)
S4_THRESH: float = 79.5      # FLAG trend condition
SUPER_CROSS_THRESH: float = 20.0   # SUPER: S1 crosses back above 20 to confirm the trough
SL_POINTS: float = 7.0       # entry - 7
TP_POINTS: float = 7.0       # entry + 7
SESSION_START_MIN: int = 560      # 09:20
SESSION_END_MIN: int = 900        # 15:00 (no new entries at/after; EOD exit)
FILTER_PERIOD_MIN: int = 5        # 5m index filter
FILTER_LINREG_LEN: int = 11
FILTER_UT_KEY: float = 1.0
FILTER_UT_PERIOD: int = 10
FILTER_SESSION_START: int = 555   # 09:15 — first 5m bar of the session
FILTER_WARM_DAYS: int = 12
BAR_SECONDS: int = 10             # official strategy timeframe
POLL_SECONDS: float = 1.0
FORMING_BAND: float = 25.0        # setup-forming proximity band (terminal only)

STRIKE_STEP: int = 50
ATM_OFFSET_SPEC: int = 100        # 2nd ITM: CE atm-100 / PE atm+100
ROLLOVER_WATCH_OFFSET: int = 50   # ATM±50 pair tracked warm for rollover


class BarBuilder:
    """Aggregates price ticks into fixed-interval bars (wall-clock aligned)."""

    def __init__(self, seconds: int = BAR_SECONDS):
        self.seconds = seconds
        self.reset()

    def reset(self) -> None:
        self.open: Optional[float] = None
        self.high: float = 0.0
        self.low: float = 0.0
        self.close: Optional[float] = None
        self.bucket_ts: Optional[datetime] = None

    def push(self, price: float, ts: datetime) -> Optional[Dict[str, Any]]:
        """Returns the completed bar when `ts` crosses into a new bucket."""
        bucket = ts.replace(second=ts.second // self.seconds * self.seconds, microsecond=0)
        if self.bucket_ts is None:
            self.bucket_ts = bucket
        completed = None
        if bucket != self.bucket_ts:
            if self.close is not None:
                completed = {
                    "open": self.open,
                    "high": self.high,
                    "low": self.low,
                    "close": self.close,
                    "ts": self.bucket_ts,
                }
            self.open = None
            self.high = 0.0
            self.low = 0.0
            self.close = None
            self.bucket_ts = bucket
        if self.open is None:
            self.open = price
        self.high = max(self.high, price) if self.high else price
        self.low = min(self.low, price) if self.low else price
        self.close = price
        return completed

    def forming(self) -> Optional[Dict[str, Any]]:
        """The in-progress bar (TV shows its live values intrabar)."""
        if self.open is None:
            return None
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "ts": self.bucket_ts,
        }


class IncrementalStochastic:
    """Incremental %D stochastic (matches flattrade_bot.indicators.stochastic)."""

    def __init__(self, k_period: int, d_period: int):
        self.k_period = k_period
        self.d_period = d_period
        self.highs: deque = deque(maxlen=k_period)
        self.lows: deque = deque(maxlen=k_period)
        self.raw_k: deque = deque(maxlen=d_period)

    def push(self, high: float, low: float, close: float) -> Optional[float]:
        self.highs.append(high)
        self.lows.append(low)
        if len(self.highs) < self.k_period:
            return None
        hh = max(self.highs)
        ll = min(self.lows)
        raw_k = 50.0 if hh == ll else ((close - ll) / (hh - ll)) * 100.0
        self.raw_k.append(raw_k)
        if len(self.raw_k) < self.d_period:
            return None
        return sum(self.raw_k) / len(self.raw_k)

    def peek(self, high: float, low: float, close: float) -> Optional[float]:
        """%D as if the forming bar were appended — no state change (TV intrabar style)."""
        highs = (list(self.highs) + [high])[-self.k_period:]
        lows = (list(self.lows) + [low])[-self.k_period:]
        if len(highs) < self.k_period:
            return None
        hh = max(highs)
        ll = min(lows)
        raw_k = 50.0 if hh == ll else ((close - ll) / (hh - ll)) * 100.0
        raw = (list(self.raw_k) + [raw_k])[-self.d_period:]
        if len(raw) < self.d_period:
            return None
        return sum(raw) / len(raw)


class OptionTracker:
    """4-stochastic tracker for one option contract."""

    def __init__(self):
        self.s1 = IncrementalStochastic(*S1_SPEC)
        self.s2 = IncrementalStochastic(*S2_SPEC)
        self.s3 = IncrementalStochastic(*S3_SPEC)
        self.s4 = IncrementalStochastic(*S4_SPEC)
        self.prev_s1: Optional[float] = None
        self.v1: Optional[float] = None
        self.v2: Optional[float] = None
        self.v3: Optional[float] = None
        self.v4: Optional[float] = None

    def push(self, high: float, low: float, close: float) -> Tuple[Optional[float], ...]:
        s1 = self.s1.push(high, low, close)
        s2 = self.s2.push(high, low, close)
        s3 = self.s3.push(high, low, close)
        s4 = self.s4.push(high, low, close)
        prev = self.prev_s1
        self.prev_s1 = s1
        if s1 is not None:
            self.v1, self.v2, self.v3, self.v4 = s1, s2, s3, s4
        return s1, s2, s3, s4, prev

    def peek(self, high: float, low: float, close: float) -> Tuple[Optional[float], ...]:
        """Forming-bar (intrabar) values — the tracker's window plus the live bar."""
        return (self.s1.peek(high, low, close), self.s2.peek(high, low, close),
                self.s3.peek(high, low, close), self.s4.peek(high, low, close))


class IncrementalATR:
    """Incremental Wilder ATR (period = 14 default; 10 for UT Bot)."""

    def __init__(self, period: int = 10):
        self.period = period
        self.prev_close: Optional[float] = None
        self.atr: Optional[float] = None
        self.tr_hist: deque = deque(maxlen=period)

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

    def peek(self, high: float, low: float, close: float) -> Optional[float]:
        """ATR as-if the given (forming) bar were appended — no mutation."""
        if self.prev_close is None:
            return None
        tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        if self.atr is not None:
            return (self.atr * (self.period - 1) + tr) / self.period
        if len(self.tr_hist) == self.period - 1:
            return (sum(self.tr_hist) + tr) / self.period
        return None


class HeikinAshiState:
    """Heikin-Ashi conversion — TradingView chart-type HA, continuous across days.

    First bar:  ha_open = (open + close) / 2
    Later bars: ha_open = (prev_ha_open + prev_ha_close) / 2
                ha_close = (open + high + low + close) / 4
                ha_high  = max(high, ha_open, ha_close)
                ha_low   = min(low, ha_open, ha_close)
    """

    def __init__(self):
        self.open: Optional[float] = None
        self.close: Optional[float] = None

    def update(self, o: float, h: float, l: float, c: float) -> Dict[str, float]:
        ha_close = (o + h + l + c) / 4.0
        ha_open = (o + c) / 2.0 if self.open is None else (self.open + self.close) / 2.0
        ha_high = max(h, ha_open, ha_close)
        ha_low = min(l, ha_open, ha_close)
        self.open = ha_open
        self.close = ha_close
        return {"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close}

    def peek(self, o: float, h: float, l: float, c: float) -> Dict[str, float]:
        """HA values as-if the given (forming) bar were appended — no mutation."""
        ha_close = (o + h + l + c) / 4.0
        ha_open = (o + c) / 2.0 if self.open is None else (self.open + self.close) / 2.0
        ha_high = max(h, ha_open, ha_close)
        ha_low = min(l, ha_open, ha_close)
        return {"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close}


class UTBot:
    """UT Bot (key=1.0, period=10) on Heikin-Ashi close — exact Pine port.

    The user's TradingView chart is an HA chart (chart type 8), so the UT Bot
    Alerts indicator receives HA candles there; the bot must do the same.
    """

    def __init__(self, key: float = FILTER_UT_KEY, period: int = FILTER_UT_PERIOD):
        self.key = key
        self.atr = IncrementalATR(period)
        self.trailing_stop = 0.0
        self.prev_src: Optional[float] = None
        self.prev_trail: Optional[float] = None

    def update_close(self, close: float, high: float, low: float) -> str:
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

    def peek_close(self, close: float, high: float, low: float) -> str:
        """Color the UT Bot would show on the FORMING bar right now.

        Non-mutating: ATR is peeked as-if the forming bar were appended, and
        the Pine trail formula runs against the last completed bar's state.
        """
        atr = self.atr.peek(high, low, close)
        prev_trail = self.prev_trail if self.prev_trail is not None else 0.0
        if atr is None or atr == 0.0 or self.prev_src is None:
            return "none"
        n_loss = self.key * atr
        if close > prev_trail and self.prev_src > prev_trail:
            trail = max(prev_trail, close - n_loss)
        elif close < prev_trail and self.prev_src < prev_trail:
            trail = min(prev_trail, close + n_loss)
        elif close > prev_trail:
            trail = close - n_loss
        else:
            trail = close + n_loss
        return "green" if close > trail else "red"


class IndexFilter:
    """Index 5m filter: UT Bot (Heikin-Ashi close) + LinReg candles + white line.

    Feed completed 1m bars (open/high/low/close). Bars are aligned to clock
    boundaries (09:15, 09:20, ...) exactly like TradingView: the 09:14
    pre-session placeholder row is skipped, and each 5m bar is converted to
    a Heikin-Ashi candle first (the chart the strategy was built on is an HA
    chart), then updates the UT color and the linreg white line. Exactly
    mirrors the backtest filter.

    Day scoping: snapshots are tagged with the trading day and only today's
    snapshots are ever used, so a restart at 10:30 cannot resurrect the
    previous day's filter state. Before today's first 5m bar completes, the
    FORMING bar's live attributes drive the side (TradingView-style: UT color
    and close-vs-white-line are computed on the bar currently forming).
    """

    def __init__(self):
        self.period = FILTER_PERIOD_MIN
        self.linreg_len = FILTER_LINREG_LEN
        self.buf: List[Dict[str, float]] = []
        self._bucket: Optional[int] = None
        self.ha = HeikinAshiState()
        self.ut = UTBot()
        self.ut_color: str = "none"
        self.raw_closes: List[float] = []
        self.raw_opens: List[float] = []
        self.raw_highs: List[float] = []
        self.raw_lows: List[float] = []
        self.bclose_history: List[float] = []
        self.bar_close: Optional[float] = None
        self.linreg_close: Optional[float] = None
        self.white_line: Optional[float] = None
        self.last_completed_minute: Optional[int] = None
        self.snapshots: Dict[int, Dict[str, Any]] = {}  # completion minute -> state
        self.today: Optional[str] = None                # "DD-MM-YYYY"
        self._cur_day: Optional[str] = None
        self.forming: Optional[Dict[str, Any]] = None   # live forming-bar state

    def set_today(self, day: str) -> None:
        self.today = day

    @staticmethod
    def _linreg(values: List[float], length: int) -> Optional[float]:
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

    def update_1m(self, bar: Dict[str, float], minute: int, day: Optional[str] = None) -> None:
        if day is not None:
            if self._cur_day is not None and day != self._cur_day:
                # Commit the previous day's last (15:15..15:29) bar before the
                # buffer is discarded — TradingView includes it, so the next
                # day's HA/UT/linreg state must start from it.
                if self.buf:
                    self._commit_bar()
                self.buf.clear()
                self._bucket: Optional[int] = None
            self._cur_day = day
        # Skip pre-session rows (the broker emits a flat 09:14 placeholder row).
        # TradingView's first 5m bar of the session starts at 09:15 (555).
        if minute < FILTER_SESSION_START:
            return
        # Align 5m bars to clock boundaries (09:15, 09:20, ...) exactly like
        # TradingView. The 09:14 placeholder would otherwise shift every bar
        # by one minute and skew the HA/UT trail vs the chart.
        bucket = minute - (minute % self.period)
        if self.buf and self._bucket is not None and bucket != self._bucket:
            self._commit_bar()
        self._bucket = bucket
        self.buf.append(bar)

    def _commit_bar(self) -> None:
        agg = {
            "open": self.buf[0]["open"],
            "high": max(b["high"] for b in self.buf),
            "low": min(b["low"] for b in self.buf),
            "close": self.buf[-1]["close"],
        }
        self.buf.clear()
        ha = self.ha.update(agg["open"], agg["high"], agg["low"], agg["close"])
        self.bar_close = ha["close"]
        self.last_completed_minute = self._bucket + self.period - 1
        self.ut_color = self.ut.update_close(ha["close"], ha["high"], ha["low"])

        self.raw_opens.append(ha["open"])
        self.raw_highs.append(ha["high"])
        self.raw_lows.append(ha["low"])
        self.raw_closes.append(ha["close"])

        lo = self._linreg(self.raw_opens, self.linreg_len)
        lh = self._linreg(self.raw_highs, self.linreg_len)
        ll = self._linreg(self.raw_lows, self.linreg_len)
        lc = self._linreg(self.raw_closes, self.linreg_len)
        if lo is None or lh is None or ll is None or lc is None:
            return
        self.linreg_close = lc
        self.bclose_history.append(lc)
        if len(self.bclose_history) >= self.linreg_len:
            self.white_line = sum(self.bclose_history[-self.linreg_len:]) / self.linreg_len

        self.snapshots[self.last_completed_minute] = {
            "ut_color": self.ut_color,
            "bar_close": self.bar_close,
            "white_line": self.white_line,
            "day": self._cur_day or self.today,
        }
        if len(self.snapshots) > 400:
            for old_min in list(self.snapshots)[:200]:
                self.snapshots.pop(old_min, None)

    def update_forming(self, price: float, minute: int) -> None:
        """Live attributes of the FORMING 5m bar (TradingView-style, per tick).

        ATR is peeked as-if the forming bar were appended; the UT color and
        close-vs-white-line are computed with the live price so the side is
        known at the exact moment of the trade.
        """
        if not self.buf:
            self.forming = None
            return
        f_open = self.buf[0]["open"]
        f_high = max(b["high"] for b in self.buf + [{"high": price, "low": price, "close": price, "open": f_open}])
        f_low = min(b["low"] for b in self.buf + [{"high": price, "low": price, "close": price, "open": f_open}])
        f_close = price
        f_ha = self.ha.peek(f_open, f_high, f_low, f_close)

        ut = self.ut.peek_close(f_ha["close"], f_ha["high"], f_ha["low"])
        lc = self._linreg(self.raw_closes + [f_ha["close"]], self.linreg_len) if len(self.raw_closes) >= self.linreg_len - 1 else None
        wl = None
        if lc is not None and len(self.bclose_history) >= self.linreg_len - 1:
            wl = (sum(self.bclose_history[-(self.linreg_len - 1):]) + lc) / self.linreg_len
        self.forming = {
            "ut_color": ut,
            "bar_close": f_ha["close"],
            "white_line": wl,
            "minute": minute,
        }

    def allowed_side(self, minute: int) -> Optional[str]:
        """Side allowed by the LIVE forming 5m bar, else today's last completed bar.

        Forming state takes precedence (the candle on screen right now);
        only TODAY's completed snapshots are ever considered (no prior-day
        carryover), so before today's first completion the answer is None
        unless the forming bar is already computable.
        """
        if self.forming is not None:
            f = self.forming
            if f["white_line"] is not None and f["ut_color"] not in (None, "none"):
                bull = f["bar_close"] > f["white_line"] and f["ut_color"] == "green"
                bear = f["bar_close"] < f["white_line"] and f["ut_color"] == "red"
                if bull and not bear:
                    return "CE"
                if bear and not bull:
                    return "PE"
                return None
        state: Optional[Dict[str, Any]] = None
        best_min: Optional[int] = None
        for m, s in self.snapshots.items():
            if s.get("day") is not None and self.today is not None and s["day"] != self.today:
                continue
            if m <= minute and (best_min is None or m > best_min):
                best_min = m
                state = s
        if state is None:
            return None
        wl = state.get("white_line")
        bc = state.get("bar_close")
        uc = state.get("ut_color")
        if wl is None or bc is None or uc in (None, "none"):
            return None
        bull = bc > wl and uc == "green"
        bear = bc < wl and uc == "red"
        if bull and not bear:
            return "CE"
        if bear and not bull:
            return "PE"
        return None


class ContractState:
    """Live state for one tracked option contract."""

    def __init__(self, side: str, strike: int, tsym: str, token: str):
        self.side = side
        self.strike = strike
        self.tsym = tsym
        self.token = token
        self.key = f"{side}:{strike}"
        self.bars = BarBuilder(BAR_SECONDS)
        self.tracker = OptionTracker()
        self.s1 = self.s2 = self.s3 = self.s4 = None
        self.prev_s1: Optional[float] = None
        self.last_bar: Optional[Dict[str, Any]] = None
        self.last_tick_price: Optional[float] = None
        self.last_signal: Optional[Dict[str, Any]] = None
        self._last_signal_epoch: Optional[datetime] = None  # one signal per 10s bar epoch
        # Divergence engine (F6-style): runs on the SAME stochastic series the
        # trigger uses (the tracker's S1/S2), so the crossing that fires the
        # SUPER trigger confirms the same trough the divergence compares. This
        # matches the backtest proxy (which feeds div the tracker's 1m S1/S2).
        self.div = DivergenceEngine()

    def seed_1m_rows(self, rows: List[Dict[str, Any]]) -> int:
        """Warms stochastics from 1m rows, synthesizing six 10s sub-bars per row
        (open, high, low, then close x3) so the tracker window spans the SAME
        ~10 minutes as the live 10s chart. Window extremes (HH/LL) then match
        TradingView's 10s window, and live values converge within ~1 minute
        instead of ~10 (no signals during seed).

        Also feeds the divergence engine with the tracker's S1/S2 sampled at
        each 1m row (after the six synthesized 10s sub-bars), so a mid-session
        restart reproduces the divergence state the live path would have built
        on the 10s tracker series by that point in the day."""
        n = 0
        for row in rows:
            try:
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
            except (KeyError, TypeError, ValueError):
                continue
            if h < l:
                h, l = l, h
            c = max(l, min(h, c))
            o = max(l, min(h, o))
            for (sh, sl, sc) in ((o, o, o), (h, h, h), (l, l, l), (c, c, c), (c, c, c), (c, c, c)):
                self.tracker.push(sh, sl, sc)
            self.div.update(c, self.tracker.v1, s2_val=self.tracker.v2, low_price=l, high_price=h)
            n += 1
        self.s1 = self.tracker.v1
        self.s2 = self.tracker.v2
        self.s3 = self.tracker.v3
        self.s4 = self.tracker.v4
        self.prev_s1 = self.tracker.prev_s1
        return n

    def push_tick(self, price: float, ts: datetime) -> Optional[Dict[str, Any]]:
        """Pushes a quote tick; evaluates the contract on every 10s bar close.

        Evaluation happens at the bar commit (the moment the S1 turn-up candle
        CLOSES), so the SUPER divergence gate is synchronous with the trigger:
        the div engine has just been fed the same committed S1/S2 the trigger
        reads. This matches the backtest proxy, which evaluates on committed
        1m bars. No intra-bar (forming-tick) evaluation, so the per-tick low
        wobble cannot create spurious divergence troughs. The forming bar is
        still peeked after each commit to keep the setup monitor values fresh."""
        self.last_tick_price = price
        if price <= 0:
            return None
        bar = self.bars.push(price, ts)
        if bar is not None:
            self.last_bar = bar
            s1, s2, s3, s4, prev = self.tracker.push(bar["high"], bar["low"], bar["close"])
            self.div.update(bar["close"], s1, s2, low_price=bar["low"], high_price=bar["high"])
            sig = self._evaluate(s1, s2, s3, s4, prev, bar["close"], bar["ts"])
            fb = self.bars.forming()
            if fb is not None:
                f1, f2, f3, f4 = self.tracker.peek(fb["high"], fb["low"], fb["close"])
                if f1 is not None:
                    self.s1, self.s2, self.s3, self.s4 = f1, f2, f3, f4
            return sig
        return None

    def push_1m_bar(self, high: float, low: float, close: float,
                    ts: datetime) -> Optional[Dict[str, Any]]:
        """Replay path: evaluates one 1m bar directly (backtest proxy, no BarBuilder).
        Also feeds the divergence engine the tracker's S1/S2 (same series as
        the trigger), so replay respects the SUPER divergence like the backtest."""
        s1, s2, s3, s4, prev = self.tracker.push(high, low, close)
        self.div.update(close, s1, s2, low_price=low, high_price=high)
        return self._evaluate(s1, s2, s3, s4, prev, close, ts)

    def _evaluate(self, s1, s2, s3, s4, prev, close, ts) -> Optional[Dict[str, Any]]:
        self.s1, self.s2, self.s3, self.s4, self.prev_s1 = s1, s2, s3, s4, prev
        if s1 is None or prev is None:
            return None
        # FLAG: plain setup, NO divergence requirement.
        flag = prev > S1_THRESH and s1 <= S1_THRESH and s4 is not None and s4 >= S4_THRESH
        # SUPER: S1 crosses back above 20 (the current trough becomes fully
        # formed) AND the bullish trough divergence is confirmed at that bar.
        # The divergence engine compares the current confirmed trough against
        # the previous one: lower low + higher S1 or S2.
        super_ = (
            prev <= SUPER_CROSS_THRESH and s1 > SUPER_CROSS_THRESH
            and self.div.divergence_confirmed_at_last_update() is not None
        )
        if flag or super_:
            if ts == self._last_signal_epoch:
                return None
            self._last_signal_epoch = ts
            self.last_signal = {
                "key": self.key,
                "side": self.side,
                "strike": self.strike,
                "signal": "flag" if flag else "super",
                "price": close,
                "ts": ts,
            }
            return self.last_signal
        return None

    def setup_status(self) -> Dict[str, Any]:
        """Terminal status: values + forming/ready state for the setup monitor."""
        values = (self.s1, self.s2, self.s3, self.s4)
        if any(v is None for v in values):
            return {"state": "WARMING", "s1": None, "s2": None, "s3": None, "s4": None}
        s1, s2, s3, s4 = values
        if self.last_signal is not None:
            state = "READY"
        elif s4 >= S4_THRESH and s1 <= FORMING_BAND:
            state = "FLAG FORMING"
        elif s1 <= SUPER_CROSS_THRESH:
            state = "SUPER FORMING"
        else:
            state = "NEUTRAL"
        return {"state": state, "s1": s1, "s2": s2, "s3": s3, "s4": s4}


class PocketMoneyEngine:
    """Drives the 10s scalping engine: ticks in, signals out. Synchronous."""

    def __init__(
        self,
        poll_seconds: float = POLL_SECONDS,
        bar_seconds: int = BAR_SECONDS,
        sl_points: float = SL_POINTS,
        tp_points: float = TP_POINTS,
    ):
        self.poll_seconds = poll_seconds
        self.bar_seconds = bar_seconds
        self.sl_points = sl_points
        self.tp_points = tp_points
        self.contracts: Dict[str, ContractState] = {}
        self.spec_keys: set = set()          # 2nd ITM keys eligible for entry
        self.spot_bars = BarBuilder(60)
        self.filter = IndexFilter()
        self.latest_spot_price: Optional[float] = None
        self.current_atm: Optional[int] = None
        self.position_open = False
        self.today: Optional[str] = None     # "DD-MM-YYYY" per Flattrade timestamps
        self.event_count = 0
        self.last_event: Optional[Dict[str, Any]] = None
        self._spot_seeded = False

    # ── Setup & warmup ───────────────────────────────────────────────────

    @staticmethod
    def _minute_of(row: Dict[str, Any]) -> int:
        try:
            dt = datetime.strptime(str(row.get("time", "")), "%d-%m-%Y %H:%M:%S")
            return dt.hour * 60 + dt.minute
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _day_of(row: Dict[str, Any]) -> Optional[str]:
        try:
            return str(row.get("time", "")).split(" ")[0] or None
        except (TypeError, ValueError, AttributeError):
            return None

    def set_today(self, day: str) -> None:
        self.today = day
        self.filter.set_today(day)

    def atm_for(self, spot: float) -> int:
        return int(round(spot / STRIKE_STEP) * STRIKE_STEP)

    def spec_keys_for(self, spot: float) -> set:
        atm = self.atm_for(spot)
        return {f"CE:{atm - ATM_OFFSET_SPEC}", f"PE:{atm + ATM_OFFSET_SPEC}"}

    def desired_keys(self, spot: float) -> set:
        atm = self.atm_for(spot)
        keys = {f"CE:{atm - ATM_OFFSET_SPEC}", f"PE:{atm + ATM_OFFSET_SPEC}"}
        keys.add(f"CE:{atm - ROLLOVER_WATCH_OFFSET}")
        keys.add(f"PE:{atm + ROLLOVER_WATCH_OFFSET}")
        return keys

    def seed_spot_1m(self, rows: List[Dict[str, Any]], today: Optional[str] = None) -> int:
        """Feeds historical 1m spot rows (warm days + today) into the filter.

        Each row carries its trading day; a day boundary flushes the 5m
        buffer so the first forming bar of TODAY starts fresh. Flattrade's
        spot TPSeries pre-fills the CURRENT day's remaining minutes with
        flat placeholder rows (volume 0, O=H=L=C) — feeding those would
        advance the 5m chain ahead of the clock and double-fill buckets
        once real ticks arrive, so today's rows at/after the current minute
        are dropped. Prior days are fed in full. A mid-session restart
        therefore reconstructs today's chain exactly and the live tick path
        continues seamlessly from the current minute.
        """
        if today is not None:
            self.today = today
            self.filter.set_today(today)
        now = datetime.now()
        cur_minute = now.hour * 60 + now.minute
        n = 0
        last_day: Optional[str] = None
        for row in rows:
            try:
                bar = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            day = self._day_of(row)
            if (
                day is not None and self.today is not None and day == self.today
                and self._minute_of(row) >= cur_minute
            ):
                continue
            self.filter.update_1m(bar, self._minute_of(row), day=day)
            if day is not None:
                last_day = day
            n += 1
        self._spot_seeded = True
        logger.info("PocketMoney: seeded filter with %d spot 1m rows (last day=%s, today=%s)",
                    n, last_day, self.today)
        return n

    def add_contract(self, side: str, strike: int, tsym: str, token: str,
                     seed_rows: Optional[List[Dict[str, Any]]] = None) -> ContractState:
        key = f"{side}:{strike}"
        if key in self.contracts:
            cs = self.contracts[key]
        else:
            cs = ContractState(side, strike, tsym, token)
            self.contracts[key] = cs
            logger.info("PocketMoney: tracking %s (%s)", key, tsym)
        if seed_rows:
            n = cs.seed_1m_rows(seed_rows)
            if n:
                logger.info("PocketMoney: seeded %s stochastics with %d 1m rows (%d 10s sub-bars)",
                            key, n // 6, n)
        return cs

    def update_spec_keys(self, spot: float) -> set:
        """Refreshes the spec set from spot; returns newly desired keys."""
        self.latest_spot_price = spot
        self.current_atm = self.atm_for(spot)
        self.spec_keys = self.spec_keys_for(spot)
        return self.desired_keys(spot)

    # ── Live tick path ───────────────────────────────────────────────────

    def push_spot_tick(self, price: float, ts: datetime) -> None:
        self.latest_spot_price = price
        minute = ts.hour * 60 + ts.minute
        bar = self.spot_bars.push(price, ts)
        if bar is not None:
            self.filter.update_1m(
                {"open": bar["open"], "high": bar["high"], "low": bar["low"], "close": bar["close"]},
                minute,
                day=self.today,
            )
        self.filter.update_forming(price, minute)

    def push_option_tick(self, key: str, price: float, ts: datetime) -> Optional[Dict[str, Any]]:
        """Pushes one option quote tick; returns a trigger when a 10s bar closes."""
        cs = self.contracts.get(key)
        if cs is None:
            return None
        signal = cs.push_tick(price, ts)
        return self._gate_signal(key, signal, ts)

    def push_option_1m(self, key: str, high: float, low: float, close: float,
                       ts: datetime) -> Optional[Dict[str, Any]]:
        """Replay path: pushes one 1m bar (backtest proxy); identical gating."""
        cs = self.contracts.get(key)
        if cs is None:
            return None
        signal = cs.push_1m_bar(high, low, close, ts)
        return self._gate_signal(key, signal, ts)

    def _gate_signal(self, key: str, signal: Optional[Dict[str, Any]],
                     ts: datetime) -> Optional[Dict[str, Any]]:
        if signal is None:
            return None
        if key not in self.spec_keys:
            logger.debug("PocketMoney: %s trigger on non-spec strike ignored", key)
            return None
        if self.position_open:
            return None
        minute = ts.hour * 60 + ts.minute
        if minute >= SESSION_END_MIN:
            return None
        allowed = self.filter.allowed_side(minute)
        if allowed is None or signal["side"] != allowed:
            logger.info(
                "PocketMoney: %s %s blocked by filter (allowed=%s)",
                signal["signal"], key, allowed,
            )
            return None
        signal["allowed_side"] = allowed
        signal["sl"] = round(signal["price"] - self.sl_points, 2)
        signal["tp"] = round(signal["price"] + self.tp_points, 2)
        self.event_count += 1
        self.last_event = signal
        return signal

    def on_position_opened(self) -> None:
        self.position_open = True

    def on_position_closed(self) -> None:
        self.position_open = False

    # ── Terminal / summary ───────────────────────────────────────────────

    def setup_monitor(self) -> List[Dict[str, Any]]:
        out = []
        for key in sorted(self.contracts):
            cs = self.contracts[key]
            status = cs.setup_status()
            out.append({
                "key": key,
                "side": cs.side,
                "strike": cs.strike,
                "tsym": cs.tsym,
                "spec": key in self.spec_keys,
                **status,
            })
        return out

    def get_summary(self) -> Dict[str, Any]:
        state = self.filter
        if state.forming is not None:
            ut_color = state.forming.get("ut_color") or state.ut_color
            white_line = state.forming.get("white_line") or state.white_line
            bar_close = state.forming.get("bar_close") or state.bar_close
            filter_minute = state.forming.get("minute") or state.last_completed_minute
        else:
            ut_color, white_line, bar_close = state.ut_color, state.white_line, state.bar_close
            filter_minute = state.last_completed_minute
        return {
            "ut_color": ut_color,
            "white_line": white_line,
            "bar_close": bar_close,
            "filter_minute": filter_minute,
            "forming": state.forming is not None,
            "allowed_side": state.allowed_side(datetime.now().hour * 60 + datetime.now().minute),
            "atm": self.current_atm,
            "spec_keys": sorted(self.spec_keys),
            "event_count": self.event_count,
        }