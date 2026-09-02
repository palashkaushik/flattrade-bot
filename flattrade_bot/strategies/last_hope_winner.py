"""Last Hope Winner Strategy Engine — Net-Points (Max-Profit) Champion.

Strategy: 🏆 Last Hope GPU Winner (7-Year Net ₹2,108,703 | 63.89% Win Rate | Max DD ₹9,303 | Calmar 226.68)
Specifications from LAST_HOPE_WINNER.md:
  - Execution Timeframe: 1-minute option OHLC bars (09:15–15:00 IST)
  - Multi-TF Option Stochastics: 1m, 2m, 3m, 5m bars evaluated concurrently
      S1: %K=12, %D=3 (Fast)
      S3: %K=40, %D=4 (Slow)
      S4: %K=50, %D=10 (Macro trend)
  - 10-Bar Arming Window: S1 <= 25.0 arms the setup for up to 10 bars
  - Triggers (on any TF 1m/2m/3m/5m):
      FLAG (M6): S4 >= 79.5 and S1 < 79.5
      SUPER: S3 < 25 and S4 < 25 and S1 < 25 and S1 is rising (S1 > prev S1)
  - S/R Bounce Gate (touch_buffer = 0.0):
      Candle Low <= Level + 0.0 AND Candle Close >= Level - 0.5 on option S/R suite:
      (CPR BC/Pivot/TC, Camarilla H3/L3, PDH, PDL, EMA20, EMA200, VWAP)
  - Auxiliary Gates: Bias OFF, Elder OFF, RSI OFF, Reversal OFF, ST-Zone OFF (All-Day 09:15–15:00)
  - Risk Geometry:
      Distance: dist = min(ATR(10) * 1.5, 15.0 pts)
      Initial SL = Entry - dist
      Initial TP = Entry + dist
      Breakeven Stop (BE): When High/LTP >= Entry + BE_TRIGGER_RATIO (0.50) * dist, SL permanently hardens to Entry + 1.0 pt
      SL priority over TP if both hit on the same bar/tick
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("flattrade_bot.last_hope_winner")

# Strategy Constants — SEEDED CHAMPION (§41 max-net: arm15/atr10/x1.5/tb0.0/be0.5)
S1_K, S1_D = 12, 3
S3_K, S3_D = 40, 4
S4_K, S4_D = 50, 10
ARM_S1 = 25.0
ARM_WINDOW = 15  # §41: 15 bars (max-net; arm10 was the cold-start champion)
M6_S4 = 79.5
M6_S1 = 79.5
SUPER_THRESH = 25.0
ATR_PERIOD = 10
ATR_MULT = 1.5
TP_PTS_CAP = 15.0
BE_TRIGGER_RATIO = 0.50  # §41: 50% of SL distance (was 0.70 in cold-start champion)
BE_BUFFER_PTS = 1.0      # Hardened SL = Entry + 1.0 pt
TOUCH_BUFFER = 0.0       # Strict touch/pierce (no gap tolerance)

SESSION_START_MIN = 555  # 09:15 IST
SESSION_END_MIN = 900    # 15:00 IST


class IncrementalStoch:
    """Incremental Stochastic %D calculator matching gpu_stoch."""

    def __init__(self, k_period: int, d_period: int):
        self.k_period = k_period
        self.d_period = d_period
        self.highs = deque(maxlen=k_period)
        self.lows = deque(maxlen=k_period)
        self.raw_k_history = deque(maxlen=d_period)

    def update(self, high: float, low: float, close: float) -> float:
        self.highs.append(high)
        self.lows.append(low)

        hh = max(self.highs)
        ll = min(self.lows)
        denom = max(hh - ll, 1e-6)
        raw_k = ((close - ll) / denom) * 100.0
        self.raw_k_history.append(raw_k)

        return sum(self.raw_k_history) / len(self.raw_k_history)


class IncrementalATR:
    """EMA-smoothed True Range calculator (alpha = 2 / (period + 1))."""

    def __init__(self, period: int = ATR_PERIOD):
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.prev_close: Optional[float] = None
        self.value: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> float:
        if self.prev_close is None:
            tr = high - low
            self.value = tr
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
            self.value = (self.value * (1.0 - self.alpha)) + (tr * self.alpha)
        self.prev_close = close
        return self.value


class IncrementalEMA:
    """Incremental Exponential Moving Average."""

    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.value: Optional[float] = None

    def update(self, close: float) -> float:
        if self.value is None:
            self.value = close
        else:
            self.value = (close * self.alpha) + (self.value * (1.0 - self.alpha))
        return self.value


class IncrementalVWAP:
    """Volume Weighted Average Price (intraday session, resets daily)."""

    def __init__(self):
        self.cum_pv = 0.0
        self.cum_vol = 0.0
        self.value: Optional[float] = None

    def reset(self):
        self.cum_pv = 0.0
        self.cum_vol = 0.0
        self.value = None

    def update(self, high: float, low: float, close: float, volume: float = 100.0) -> float:
        hlc3 = (high + low + close) / 3.0
        vol = max(volume, 1.0)
        self.cum_pv += hlc3 * vol
        self.cum_vol += vol
        self.value = self.cum_pv / max(self.cum_vol, 1.0)
        return self.value


@dataclass(frozen=True)
class Bar1m:
    minute: int
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime
    volume: float = 100.0


@dataclass
class TFTracker:
    """Tracks aggregated TF bars (2m, 3m, 5m) from 1m completed bars."""
    tf: int
    s1: IncrementalStoch = field(default_factory=lambda: IncrementalStoch(S1_K, S1_D))
    s3: IncrementalStoch = field(default_factory=lambda: IncrementalStoch(S3_K, S3_D))
    s4: IncrementalStoch = field(default_factory=lambda: IncrementalStoch(S4_K, S4_D))
    cur_bars: List[Bar1m] = field(default_factory=list)
    last_s1: Optional[float] = None
    prev_s1: Optional[float] = None
    last_s3: Optional[float] = None
    last_s4: Optional[float] = None
    last_lo: Optional[float] = None
    last_cl: Optional[float] = None

    def push_1m_bar(self, bar: Bar1m) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
        """Pushes 1m bar; returns (s1, s3, s4, is_rising) when the TF bucket completes.

        CONGRUENCE: buckets are CLOCK-ALIGNED to the session start (minute 555 = 09:15),
        exactly like the backtest's reshape(D, T_tf, tf) which chunks from the day's
        first bar. A TF bar closes only on the tick whose 1m bar ends a boundary
        ((bar.minute + 1 - SESSION_START_MIN) % tf == 0)."""
        self.cur_bars.append(bar)
        boundary = ((bar.minute + 1 - SESSION_START_MIN) % self.tf) == 0
        if self.cur_bars and boundary:
            tf_hi = max(b.high for b in self.cur_bars)
            tf_lo = min(b.low for b in self.cur_bars)
            tf_cl = self.cur_bars[-1].close
            self.last_lo = tf_lo
            self.last_cl = tf_cl
            self.cur_bars = []

            s1_val = self.s1.update(tf_hi, tf_lo, tf_cl)
            s3_val = self.s3.update(tf_hi, tf_lo, tf_cl)
            s4_val = self.s4.update(tf_hi, tf_lo, tf_cl)

            if s1_val is not None:
                self.prev_s1 = self.last_s1
                self.last_s1 = s1_val
            if s3_val is not None:
                self.last_s3 = s3_val
            if s4_val is not None:
                self.last_s4 = s4_val
        elif len(self.cur_bars) > self.tf:
            # Safety: never accumulate more than tf bars (gap protection)
            self.cur_bars = self.cur_bars[-self.tf:]

        is_rising = (self.last_s1 is not None and self.prev_s1 is not None and self.last_s1 > self.prev_s1)
        return self.last_s1, self.last_s3, self.last_s4, is_rising


@dataclass
class OptionContractState:
    """State machine and indicators for one option contract (CE or PE)."""
    symbol: str
    token: str
    side: str  # "CE" or "PE"
    strike: int

    # Live tick aggregator into 1m bars
    current_min: int = -1
    cur_open: Optional[float] = None
    cur_high: Optional[float] = None
    cur_low: Optional[float] = None
    cur_close: Optional[float] = None
    cur_ticks: int = 0

    # 1m Bar History & Indicators
    bars: List[Bar1m] = field(default_factory=list)
    atr: IncrementalATR = field(default_factory=lambda: IncrementalATR(ATR_PERIOD))
    ema20: IncrementalEMA = field(default_factory=lambda: IncrementalEMA(20))
    ema200: IncrementalEMA = field(default_factory=lambda: IncrementalEMA(200))
    vwap: IncrementalVWAP = field(default_factory=IncrementalVWAP)

    # Multi-TF Trackers (1m, 2m, 3m, 5m)
    tf_trackers: Dict[int, TFTracker] = field(default_factory=lambda: {
        1: TFTracker(1),
        2: TFTracker(2),
        3: TFTracker(3),
        5: TFTracker(5),
    })

    # S/R Suite (Day levels + dynamic indicators)
    sr_levels: Dict[str, float] = field(default_factory=dict)

    # 10-Bar Arming State Machine
    flag_armed: bool = False
    flag_arm_bar: int = -999
    super_armed: bool = False
    super_arm_bar: int = -999

    latest_ltp: Optional[float] = None
    latest_atr: float = 6.0

    def reset_session(self):
        """Clears intraday indicator state. Call BEFORE seeding: the §41/§42
        champion runs SEEDED — warmup replays the prior day's final 300 bars
        through ATR/EMA/VWAP/TF-trackers, then today's completed bars, so the
        live state equals the validated seeded-sweep state at any minute.
        (Arming is intentionally reset — a new day needs fresh S1<=25 arming.)"""
        self.bars = []
        self.current_min = -1
        self.cur_open = self.cur_high = self.cur_low = self.cur_close = None
        self.cur_ticks = 0
        self.atr = IncrementalATR(ATR_PERIOD)
        self.ema20 = IncrementalEMA(20)
        self.ema200 = IncrementalEMA(200)
        self.vwap = IncrementalVWAP()
        self.tf_trackers = {
            1: TFTracker(1),
            2: TFTracker(2),
            3: TFTracker(3),
            5: TFTracker(5),
        }
        self.flag_armed = False
        self.super_armed = False
        self.flag_arm_bar = -999
        self.super_arm_bar = -999
        self.latest_atr = 6.0
        logger.info(f"Session reset for {self.symbol}: indicators cold-started (backtest parity)")

    def set_day_sr_levels(self, prev_high: float, prev_low: float, prev_close: float):
        """Builds CPR, Camarilla, and PDH/PDL from yesterday's option OHLC."""
        pivot = (prev_high + prev_low + prev_close) / 3.0
        bc = (prev_high + prev_low) / 2.0
        tc = 2.0 * pivot - bc
        rng = prev_high - prev_low
        self.sr_levels = {
            "CPR_BC": bc,
            "CPR_Pivot": pivot,
            "CPR_TC": tc,
            "Cam_H3": prev_close + rng * 1.1 / 4.0,
            "Cam_L3": prev_close - rng * 1.1 / 4.0,
            "PDH": prev_high,
            "PDL": prev_low,
        }

    def seed_1m_bars(self, prior_bars: List[Bar1m], today_bars: Optional[List[Bar1m]] = None):
        """Seeds prior completed bars for indicator warmup, and today's completed bars with intraday VWAP."""
        if today_bars is None:
            today_bars = []
        for bar in prior_bars:
            self.bars.append(bar)
            self.atr.update(bar.high, bar.low, bar.close)
            self.ema20.update(bar.close)
            self.ema200.update(bar.close)
            for tf, tracker in self.tf_trackers.items():
                tracker.push_1m_bar(bar)

        self.vwap.reset()

        for bar in today_bars:
            self.bars.append(bar)
            atr_val = self.atr.update(bar.high, bar.low, bar.close)
            self.ema20.update(bar.close)
            self.ema200.update(bar.close)
            self.vwap.update(bar.high, bar.low, bar.close, volume=bar.volume)
            self.latest_atr = atr_val

            for tf, tracker in self.tf_trackers.items():
                tracker.push_1m_bar(bar)

    def push_tick(self, ltp: float, dt: datetime) -> Optional[Dict[str, Any]]:
        """Processes a live tick quote. Returns a Trigger Signal if a 1m bar closes and triggers."""
        self.latest_ltp = ltp
        minute = dt.hour * 60 + dt.minute

        if self.current_min == -1:
            self.current_min = minute
            self.cur_open = ltp
            self.cur_high = ltp
            self.cur_low = ltp
            self.cur_close = ltp
            self.cur_ticks = 1
            return None

        if minute == self.current_min:
            self.cur_high = max(self.cur_high, ltp)
            self.cur_low = min(self.cur_low, ltp)
            self.cur_close = ltp
            self.cur_ticks += 1
            return None

        # ── 1-MINUTE BAR COMPLETED! ──
        completed_bar = Bar1m(
            minute=self.current_min,
            open=self.cur_open,
            high=self.cur_high,
            low=self.cur_low,
            close=self.cur_close,
            timestamp=dt,
        )
        self.bars.append(completed_bar)

        # Start new forming 1m bar
        self.current_min = minute
        self.cur_open = ltp
        self.cur_high = ltp
        self.cur_low = ltp
        self.cur_close = ltp
        self.cur_ticks = 1

        return self._on_1m_bar_close(completed_bar)

    def _on_1m_bar_close(self, bar: Bar1m) -> Optional[Dict[str, Any]]:
        """Evaluates completed 1m bar across indicators, arming, and S/R bounce triggers."""
        if not self.bars or self.bars[-1] is not bar:
            self.bars.append(bar)
        bar_idx = len(self.bars) - 1

        # 1. Update 1m Indicators
        atr_val = self.atr.update(bar.high, bar.low, bar.close)
        ema20_val = self.ema20.update(bar.close)
        ema200_val = self.ema200.update(bar.close)
        vwap_val = self.vwap.update(bar.high, bar.low, bar.close)
        self.latest_atr = atr_val

        # Dynamic S/R levels updated with latest live moving averages
        active_sr = dict(self.sr_levels)
        active_sr["EMA20"] = ema20_val
        active_sr["EMA200"] = ema200_val
        active_sr["VWAP"] = vwap_val

        # 2. Multi-TF Stochastics Push (1m, 2m, 3m, 5m)
        m6_tfs: List[str] = []
        super_tfs: List[str] = []
        s1_1m = None

        for tf, tracker in self.tf_trackers.items():
            s1, s3, s4, is_rising = tracker.push_1m_bar(bar)
            if tf == 1:
                s1_1m = s1

            if s1 is not None and s4 is not None:
                # FLAG (M6): S4 >= 79.5 and S1 < 79.5
                if s4 >= M6_S4 and s1 < M6_S1:
                    m6_tfs.append(f"{tf}m")

            if s1 is not None and s3 is not None and s4 is not None:
                # SUPER: S3 < 25 and S4 < 25 and S1 < 25 and S1 is rising
                if s3 < SUPER_THRESH and s4 < SUPER_THRESH and s1 < SUPER_THRESH and is_rising:
                    super_tfs.append(f"{tf}m")

        m6_trigger = bool(m6_tfs)
        super_trigger = bool(super_tfs)

        # 3. Arming State Machine: S1 <= 25 arms setup for up to 10 bars
        if s1_1m is not None and s1_1m <= ARM_S1:
            self.flag_armed = True
            self.flag_arm_bar = bar_idx
            self.super_armed = True
            self.super_arm_bar = bar_idx

        # Expire arming if older than 10 bars
        if self.flag_armed and (bar_idx - self.flag_arm_bar > ARM_WINDOW):
            self.flag_armed = False
        if self.super_armed and (bar_idx - self.super_arm_bar > ARM_WINDOW):
            self.super_armed = False

        # 4. Gated Signals Check (Flag or Super within armed window)
        flag_signal = self.flag_armed and m6_trigger and (bar_idx - self.flag_arm_bar <= ARM_WINDOW)
        super_signal = self.super_armed and super_trigger and (bar_idx - self.super_arm_bar <= ARM_WINDOW)

        if not (flag_signal or super_signal):
            return None

        # 5. Mandatory S/R Bounce Gate (touch_buffer = 0.0)
        # Condition: bar.low <= lvl + TOUCH_BUFFER AND bar.close >= lvl - 0.5
        bounced_level: Optional[str] = None
        for lvl_name, lvl_price in active_sr.items():
            if bar.low <= (lvl_price + TOUCH_BUFFER) and bar.close >= (lvl_price - 0.5):
                bounced_level = lvl_name
                break

        # Check combined lower timeframe low/close bounce if 1m didn't touch
        if not bounced_level:
            for tf, tracker in self.tf_trackers.items():
                if tf > 1 and tracker.last_lo is not None and tracker.last_cl is not None:
                    for lvl_name, lvl_price in active_sr.items():
                        if tracker.last_lo <= (lvl_price + TOUCH_BUFFER) and tracker.last_cl >= (lvl_price - 0.5):
                            bounced_level = f"{lvl_name}_{tf}m"
                            break
                    if bounced_level:
                        break

        if not bounced_level:
            return None  # No S/R bounce -> reject signal

        # 6. Build Trigger Payload with Risk Geometry
        # SL/TP distance = min(ATR(10) * 1.5, 15.0)
        dist = round(min(max(atr_val * ATR_MULT, 2.0), TP_PTS_CAP), 2)
        entry_price = bar.close
        sl_price = round(entry_price - dist, 2)
        tp_price = round(entry_price + dist, 2)
        be_trigger_px = round(entry_price + (BE_TRIGGER_RATIO * dist), 2)
        be_hardened_sl = round(entry_price + BE_BUFFER_PTS, 2)

        trigger_name = "FLAG" if flag_signal else "SUPER"
        trigger_tf_str = ",".join(m6_tfs if flag_signal else super_tfs) or "1m"
        logger.info(
            f"🔥 {self.side} {trigger_name} [{trigger_tf_str}] TRIGGERED @ {entry_price:.2f} on {self.symbol} | "
            f"SR={bounced_level} | dist={dist:.2f} SL={sl_price:.2f} TP={tp_price:.2f} BE_trig={be_trigger_px:.2f}"
        )

        # Disarm setup after firing
        self.flag_armed = False
        self.super_armed = False

        return {
            "side": self.side,
            "symbol": self.symbol,
            "token": self.token,
            "strike": self.strike,
            "trigger": trigger_name,
            "tf": trigger_tf_str,
            "level": bounced_level,
            "entry": entry_price,
            "dist": dist,
            "sl": sl_price,
            "tp": tp_price,
            "be_trigger_px": be_trigger_px,
            "be_hardened_sl": be_hardened_sl,
            "bar": bar,
            "timestamp": bar.timestamp,
        }


class LastHopeWinnerEngine:
    """Institutional Last Hope GPU Winner Strategy Coordinator."""

    def __init__(self):
        self.contracts: Dict[str, OptionContractState] = {}  # key -> OptionContractState
        self.spot_price: Optional[float] = None
        self.active_trade: Optional[Dict[str, Any]] = None

    def register_contract(self, key: str, symbol: str, token: str, side: str, strike: int) -> OptionContractState:
        if key not in self.contracts:
            self.contracts[key] = OptionContractState(symbol=symbol, token=token, side=side, strike=strike)
        return self.contracts[key]

    def set_spot_price(self, spot: float):
        self.spot_price = spot

    def desired_strikes(self, spot: float) -> Dict[str, int]:
        """Returns 2nd ITM strikes (CE = ATM - 100, PE = ATM + 100) + rollover watch pairs."""
        atm = int(round(spot / 50.0) * 50)
        return {
            "CE_SPEC": atm - 100,
            "PE_SPEC": atm + 100,
            "CE_WATCH_PLUS50": (atm + 50) - 100,
            "PE_WATCH_PLUS50": (atm + 50) + 100,
            "CE_WATCH_MINUS50": (atm - 50) - 100,
            "PE_WATCH_MINUS50": (atm - 50) + 100,
        }

    def push_tick(self, key: str, ltp: float, dt: datetime) -> Optional[Dict[str, Any]]:
        """Pushes tick to contract state. If trigger fires and we are flat, returns setup."""
        contract = self.contracts.get(key)
        if not contract:
            return None

        # Check Active Trade Breakeven & SL/TP
        if self.active_trade and self.active_trade.get("symbol") == contract.symbol:
            pos = self.active_trade
            if not pos.get("be_done") and ltp >= pos["be_trigger_px"]:
                pos["be_done"] = True
                pos["sl"] = pos["be_hardened_sl"]
                logger.info(f"🔒 BREAKEVEN STOP TRIGGERED on {pos['symbol']}: SL moved to {pos['sl']:.2f}")

        sig = contract.push_tick(ltp, dt)
        # CONGRUENCE (gpu_sim line 339): arming only while flat & not blocked. An
        # armed setup must never persist across a position open/close cycle.
        if self.active_trade:
            contract.flag_armed = False
            contract.super_armed = False
        if sig and not self.active_trade:
            minute = dt.hour * 60 + dt.minute
            if SESSION_START_MIN <= minute <= SESSION_END_MIN:
                return sig

        return None

    def on_trade_opened(self, trade: Dict[str, Any]):
        """Registers the live trade. Re-bases BE geometry on the ACTUAL fill price
        so the BE trigger locks to entry+1.0 of what we truly paid (not the signal
        price, which can differ by the entry slippage)."""
        entry = float(trade.get("entry", 0.0))
        target = float(trade.get("tp", trade.get("target", 0.0)))
        dist = float(trade.get("dist", target - entry if target > entry > 0 else 0.0))
        if entry > 0 and dist > 0:
            trade = dict(trade)
            trade["be_trigger_px"] = round(entry + BE_TRIGGER_RATIO * dist, 2)
            trade["be_hardened_sl"] = round(entry + BE_BUFFER_PTS, 2)
            trade["sl"] = round(entry - dist, 2)
            trade["tp"] = round(entry + dist, 2)
        self.active_trade = trade

    def on_trade_closed(self):
        self.active_trade = None
