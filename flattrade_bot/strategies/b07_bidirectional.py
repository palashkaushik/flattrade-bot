"""B07: Best-TF 3-Minute Bidirectional CE+PE Strategy Engine.

Strategy Specification:
  - Timeframe: 3-Minute (3m) Spot / Futures Chart
  - Fast Stochastic (S1): Lookback = 30 bars, Smooth = 1 (%K raw)
  - Macro Trend Stochastic (S4): Lookback = 70 bars, Smooth = 1 (%K raw)
  - Volatility Measure: ATR(25) on 3m bars
  - CE Trigger: S4 >= 70.0 AND S1 <= 40.0 (Buy ATM/ITM Call Option on Bullish Dip)
  - PE Trigger: S4 <= 30.0 AND S1 >= 60.0 (Buy ATM/ITM Put Option on Bearish Bounce)
  - Option SL Distance: 4.4 * ATR(25) * 0.50 points
  - Option TP Distance: 10.0 * ATR(25) * 0.50 points (R:R = 2.27 : 1)
  - Max Active Trades: 1 at a time (Position Locked)
  - Daily Loss Circuit Breaker: 4 to 9 Nifty points (~Rs 260 - Rs 585)
"""

from collections import deque
from typing import Dict, Any, Optional, Tuple, List
from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.indicators.stochastic import IncrementalStochastic


class IncrementalATR:
    """Calculates Wilder's Incremental Average True Range (ATR)."""

    def __init__(self, period: int = 25):
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


class B07BidirectionalStrategy:
    """B07 3-Minute Nifty Spot Bidirectional (CE+PE) Strategy Coordinator."""

    def __init__(
        self,
        timeframe: int = 3,
        s1_k: int = 30,
        s4_k: int = 70,
        s4_ob: float = 70.0,
        s1_os: float = 40.0,
        atr_period: int = 25,
        sl_mult: float = 4.4,
        tp_mult: float = 10.0,
        max_trade_loss_rs: float = 3000.0,
        lot_size: int = 65,
    ):
        self.timeframe = timeframe
        self.s1_k = s1_k
        self.s4_k = s4_k
        self.s4_ob = s4_ob
        self.s1_os = s1_os
        self.pe_s4_os = 100.0 - s4_ob  # 30.0
        self.pe_s1_ob = 100.0 - s1_os  # 60.0
        self.atr_period = atr_period
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.max_trade_loss_rs = max_trade_loss_rs
        self.lot_size = lot_size

        # Indicators computed on 3m bars
        self.stoch_s1 = IncrementalStochastic(k_period=s1_k, d_period=1)
        self.stoch_s4 = IncrementalStochastic(k_period=s4_k, d_period=1)
        self.atr = IncrementalATR(period=atr_period)

        # 1m -> 3m Candle Aggregator Buffer
        self._buf_1m: List[Candle] = []
        self._last_minute: Optional[int] = None
        self._last_3m_candle: Optional[Candle] = None

        # Cached state
        self.latest_s1: Optional[float] = None
        self.latest_s4: Optional[float] = None
        self.latest_atr: Optional[float] = None
        self.last_signal_emitted: Optional[str] = None

    def reset_session_state(self) -> None:
        """Resets session buffer without discarding indicator warm-up values."""
        self._buf_1m.clear()
        self._last_minute = None
        self.last_signal_emitted = None

    @staticmethod
    def get_atm_strike(spot_price: float, strike_step: int = 50) -> int:
        """Calculates exact ATM Strike: round(spot / 50) * 50."""
        return int(round(spot_price / float(strike_step)) * strike_step)

    def push_1m_candle(self, c1m: Candle) -> List[Tuple[str, str, str, float, float, float, float]]:
        """Pushes a 1m Nifty Spot candle, aggregates into 3m bar on clock boundaries,

        and evaluates B07 entry triggers.

        Returns list of triggers:
            (timeframe_label, side, signal_type, spot_entry, option_sl_points, option_tp_points, spot_atr)
        """
        triggers = []
        m = c1m.minute  # Minute of day (e.g. 570 = 09:30)

        # Detect session rollover
        if self._last_minute is not None and m > 0 and m < self._last_minute:
            self.reset_session_state()
        self._last_minute = m

        self._buf_1m.append(c1m)

        # Check clock boundary: 3m closes at 09:18 (558), 09:21 (561), 09:24 (564), 09:30 (570)
        # All satisfy: minute % 3 == 0
        if m > 0:
            is_boundary = (m % self.timeframe == 0) and len(self._buf_1m) >= 1
        else:
            is_boundary = len(self._buf_1m) >= self.timeframe

        if is_boundary:
            buf = self._buf_1m
            self._buf_1m = []

            # Construct aggregated 3-minute candle
            c3m = Candle(
                open=buf[0].open,
                high=max(x.high for x in buf),
                low=min(x.low for x in buf),
                close=buf[-1].close,
                volume=sum(getattr(x, "volume", 0.0) for x in buf),
                minute=buf[-1].minute,
            )
            self._last_3m_candle = c3m

            # Update indicators on 3m bar
            self.latest_s1 = self.stoch_s1.push(c3m.high, c3m.low, c3m.close)
            self.latest_s4 = self.stoch_s4.push(c3m.high, c3m.low, c3m.close)
            self.latest_atr = self.atr.update(c3m.high, c3m.low, c3m.close)

            # Check if indicators are warmed up
            if self.latest_s1 is not None and self.latest_s4 is not None and self.latest_atr is not None:
                atr_val = self.latest_atr
                s1 = self.latest_s1
                s4 = self.latest_s4

                # Calculate option risk distance (delta ~ 0.50)
                opt_sl_pts = round(atr_val * self.sl_mult * 0.50, 2)
                opt_tp_pts = round(atr_val * self.tp_mult * 0.50, 2)
                sl_risk_rs = opt_sl_pts * self.lot_size

                # Check max trade SL risk cap
                if sl_risk_rs <= self.max_trade_loss_rs:
                    # 🟢 CE Signal: Uptrend Dip (S4 >= 70 & S1 <= 40)
                    if s4 >= self.s4_ob and s1 <= self.s1_os:
                        triggers.append(("3m", "CE", "B07_DIP_BUY_CE", c3m.close, opt_sl_pts, opt_tp_pts, atr_val))
                        self.last_signal_emitted = "CE"

                    # 🔴 PE Signal: Downtrend Bounce (S4 <= 30 & S1 >= 60)
                    elif s4 <= self.pe_s4_os and s1 >= self.pe_s1_ob:
                        triggers.append(("3m", "PE", "B07_BOUNCE_BUY_PE", c3m.close, opt_sl_pts, opt_tp_pts, atr_val))
                        self.last_signal_emitted = "PE"

        return triggers

    def get_summary(self) -> Dict[str, Any]:
        """Provides dashboard summary of B07 3m indicators and ready states."""
        s1 = self.latest_s1
        s4 = self.latest_s4
        atr_val = self.latest_atr

        ce_ready = s4 is not None and s1 is not None and (s4 >= self.s4_ob and s1 <= self.s1_os)
        pe_ready = s4 is not None and s1 is not None and (s4 <= self.pe_s4_os and s1 >= self.pe_s1_ob)

        return {
            "timeframe": f"{self.timeframe}m",
            "s1": s1,
            "s4": s4,
            "atr": atr_val,
            "s1_spec": f"({self.s1_k},1)",
            "s4_spec": f"({self.s4_k},1)",
            "atr_spec": f"ATR({self.atr_period})",
            "ce_ready": ce_ready,
            "pe_ready": pe_ready,
            "s4_ob": self.s4_ob,
            "s1_os": self.s1_os,
            "pe_s4_os": self.pe_s4_os,
            "pe_s1_ob": self.pe_s1_ob,
            "last_signal": self.last_signal_emitted,
        }
