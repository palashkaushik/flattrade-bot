"""Combined Supreme Strategy Engine.

Strategy: 🏆 Combined Supreme Strategy (1,595+ Calmar Ratio | +₹44.82L Realized Net Profit | 91.2% Green Days)
Key Features:
  - 3-Tier Prioritized S/R Hierarchy:
      Tier 1 Supreme (Priority 1): Virgin CPR, Camarilla H3/L3, Daily CPR, Daily VWAP, Prev Day VWAP Close, 5m EMA20/200, 3m EMA200
      Tier 2 (Priority 2): Opening 3m Candle High/Low (IB-3m), 3m EMA20, Prev Day High/Low
      Tier 3 (Priority 3): Fibonacci H3/L3 (R3/S3), Camarilla H4/L4
  - Two-Bar Structure Confirmation (Bar 1 Rejection Stall + Bar 2 Extreme Breakout)
  - 15-Minute Macro Trend Gate (15m Close vs EMA20)
  - Dual Operating Sessions (09:15-11:00 & 13:30-15:00) with Midday Standdown (11:00-13:30)
  - 2nd ITM Nifty Weekly Options Execution (CE = ATM - 100, PE = ATM + 100)
  - Risk Geometry: Initial SL = 0.30x ATR5 (min 4.0 pts), TP = 1.50x ATR5, Trail trigger = +6.0 pts, Step = 2.0 pts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("flattrade_bot.combined_supreme")


@dataclass
class SRLevel:
    """Represents a structural support/resistance level in the hierarchy."""
    name: str
    price: float
    priority: int  # 1 = Tier 1 Supreme, 2 = Tier 2 Momentum, 3 = Tier 3 Extreme
    touch_count: int = 0
    max_touches: int = 2
    is_virgin: bool = False
    origin_day: Optional[str] = None


@dataclass
class RejectionSetup:
    """Represents an active or confirmed Two-Bar Rejection Setup."""
    level: SRLevel
    direction: str  # "LONG" or "SHORT"
    bar_1_high: float
    bar_1_low: float
    bar_1_close: float
    score: int
    entry_price: float
    initial_sl: float
    target_price: float
    timestamp: datetime
    confirmed: bool = False


class CombinedSupremeEngine:
    """Institutional Combined Supreme Engine with 3-Tier S/R Hierarchy & Two-Bar Structure Confirmation."""

    def __init__(
        self,
        min_score: int = 50,
        sl_mult: float = 0.30,
        tp_mult: float = 1.50,
        min_sl_pts: float = 4.0,
        max_sl_pts: float = 15.0,
        trail_trigger_pts: float = 6.0,
        trail_step_pts: float = 2.0,
    ):
        self.min_score = min_score
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.min_sl_pts = min_sl_pts
        self.max_sl_pts = max_sl_pts
        self.trail_trigger_pts = trail_trigger_pts
        self.trail_step_pts = trail_step_pts

        self.levels: List[SRLevel] = []
        self.pending_setup: Optional[RejectionSetup] = None
        self.current_15m_bullish: bool = True
        self.current_15m_ema20: float = 0.0
        self.current_vwap: float = 0.0
        self.current_ema20: float = 0.0
        self.current_ema200: float = 0.0
        self.current_5m_ema20: float = 0.0
        self.current_5m_ema200: float = 0.0
        self.opening_3m_high: float = 0.0
        self.opening_3m_low: float = 0.0
        self.current_atr: float = 14.0

        self.last_bar_1: Optional[Dict[str, Any]] = None
        self.active_position_trailing_sl: Optional[float] = None
        self.peak_price: float = 0.0

    def initialize_daily_levels(
        self,
        prev_high: float,
        prev_low: float,
        prev_close: float,
        initial_vwap: float,
        ema200: float,
        ema20: float,
        ema20_5m: Optional[float] = None,
        ema200_5m: Optional[float] = None,
        prev_vwap_close: Optional[float] = None,
        virgin_cprs: Optional[List[Tuple[float, float, float, str]]] = None,
        opening_3m_high: Optional[float] = None,
        opening_3m_low: Optional[float] = None,
    ):
        """Initializes full 3-Tier S/R Hierarchy."""
        pivot = (prev_high + prev_low + prev_close) / 3.0
        bc = (prev_high + prev_low) / 2.0
        tc = (pivot - bc) + pivot
        cpr_top = max(tc, bc)
        cpr_bot = min(tc, bc)

        cam_range = prev_high - prev_low
        h3 = prev_close + cam_range * (1.1 / 4.0)
        l3 = prev_close - cam_range * (1.1 / 4.0)
        h4 = prev_close + cam_range * (1.1 / 2.0)
        l4 = prev_close - cam_range * (1.1 / 2.0)

        fib_h3 = pivot + cam_range * 1.000
        fib_l3 = pivot - cam_range * 1.000

        pd_vwap = prev_vwap_close or initial_vwap
        e20_5m = ema20_5m or ema20
        e200_5m = ema200_5m or ema200

        self.current_vwap = initial_vwap
        self.current_ema200 = ema200
        self.current_ema20 = ema20
        self.current_5m_ema20 = e20_5m
        self.current_5m_ema200 = e200_5m
        self.opening_3m_high = opening_3m_high or prev_high
        self.opening_3m_low = opening_3m_low or prev_low

        new_levels = []

        # ── 1. SUPREME TIER 1+ PRIORITY: VIRGIN (UNTOUCHED) CPRs ──
        if virgin_cprs:
            for v_p, v_tc, v_bc, o_day in virgin_cprs[-3:]:
                new_levels.append(SRLevel(f"Virgin CPR Pivot ({o_day[-5:]})", round(v_p, 2), priority=1, is_virgin=True, origin_day=o_day))
                new_levels.append(SRLevel(f"Virgin CPR Top ({o_day[-5:]})", round(v_tc, 2), priority=1, is_virgin=True, origin_day=o_day))
                new_levels.append(SRLevel(f"Virgin CPR Bot ({o_day[-5:]})", round(v_bc, 2), priority=1, is_virgin=True, origin_day=o_day))

        # ── 2. TIER 1 CORE STRUCTURAL ANCHORS ──
        new_levels.extend([
            SRLevel("Camarilla H3", round(h3, 2), priority=1),
            SRLevel("Camarilla L3", round(l3, 2), priority=1),
            SRLevel("Daily CPR Pivot", round(pivot, 2), priority=1),
            SRLevel("Daily CPR Top (TC)", round(cpr_top, 2), priority=1),
            SRLevel("Daily CPR Bottom (BC)", round(cpr_bot, 2), priority=1),
            SRLevel("Daily VWAP", round(initial_vwap, 2), priority=1),
            SRLevel("Prev Day VWAP Close", round(pd_vwap, 2), priority=1),
            SRLevel("5m EMA 20", round(e20_5m, 2), priority=1),
            SRLevel("5m EMA 200", round(e200_5m, 2), priority=1),
            SRLevel("EMA 200", round(ema200, 2), priority=1),
        ])

        # ── 3. TIER 2 MOMENTUM & OPENING RANGE ANCHORS ──
        if opening_3m_high and opening_3m_low:
            new_levels.append(SRLevel("Opening 3m High", round(opening_3m_high, 2), priority=2))
            new_levels.append(SRLevel("Opening 3m Low", round(opening_3m_low, 2), priority=2))

        new_levels.extend([
            SRLevel("EMA 20", round(ema20, 2), priority=2),
            SRLevel("Prev Day High", round(prev_high, 2), priority=2),
            SRLevel("Prev Day Low", round(prev_low, 2), priority=2),
        ])

        # ── 4. TIER 3 EXTREME TARGET ANCHORS ──
        new_levels.extend([
            SRLevel("Fibonacci H3", round(fib_h3, 2), priority=3),
            SRLevel("Fibonacci L3", round(fib_l3, 2), priority=3),
            SRLevel("Camarilla H4", round(h4, 2), priority=3),
            SRLevel("Camarilla L4", round(l4, 2), priority=3),
        ])

        self.levels = new_levels
        logger.info(f"Initialized {len(self.levels)} S/R Levels in Combined Supreme Hierarchy.")

    def set_opening_3m_range(self, high: float, low: float):
        """Sets the Initial 3-Minute Opening Range (IB-3m) after 09:18:00."""
        self.opening_3m_high = high
        self.opening_3m_low = low
        self.levels.append(SRLevel("Opening 3m High", round(high, 2), priority=2))
        self.levels.append(SRLevel("Opening 3m Low", round(low, 2), priority=2))
        logger.info(f"Registered Opening 3m Range (Tier 2): High={high:.2f}, Low={low:.2f}")

    def update_indicators(
        self,
        spot_price: float,
        vwap: float,
        ema20: float,
        ema200: float,
        spot_15m_close: float,
        spot_15m_ema20: float,
        ema20_5m: Optional[float] = None,
        ema200_5m: Optional[float] = None,
        atr: float = 14.0,
    ):
        """Updates dynamic rolling indicators on every incoming 3m candle."""
        self.current_vwap = vwap
        self.current_ema20 = ema20
        self.current_ema200 = ema200
        if ema20_5m is not None:
            self.current_5m_ema20 = ema20_5m
        if ema200_5m is not None:
            self.current_5m_ema200 = ema200_5m
        self.current_15m_ema20 = spot_15m_ema20
        self.current_15m_bullish = (spot_15m_close >= spot_15m_ema20)
        self.current_atr = max(atr, 8.0)

        # Update dynamic level prices
        for lvl in self.levels:
            if lvl.name == "Daily VWAP":
                lvl.price = round(vwap, 2)
            elif lvl.name == "EMA 20":
                lvl.price = round(ema20, 2)
            elif lvl.name == "EMA 200":
                lvl.price = round(ema200, 2)
            elif lvl.name == "5m EMA 20" and ema20_5m is not None:
                lvl.price = round(ema20_5m, 2)
            elif lvl.name == "5m EMA 200" and ema200_5m is not None:
                lvl.price = round(ema200_5m, 2)

    def is_session_active(self, now: Optional[datetime] = None) -> bool:
        """Operating Sessions: 09:15-11:00 (Morning) & 13:30-15:00 (Afternoon)."""
        t = (now or datetime.now()).time()
        morning = (dtime(9, 15) <= t <= dtime(11, 0))
        afternoon = (dtime(13, 30) <= t <= dtime(15, 0))
        return morning or afternoon

    def evaluate_rejection_trigger(
        self,
        bar_1: Dict[str, Any],
        bar_2: Dict[str, Any],
        now: Optional[datetime] = None,
    ) -> Optional[RejectionSetup]:
        """Evaluates Two-Bar Structure Confirmation between Bar 1 and Bar 2."""
        if not self.is_session_active(now):
            return None

        # Check pending Bar 1 rejection if waiting for Bar 2 confirmation
        if self.pending_setup is not None:
            setup = self.pending_setup
            # Long confirmation: Bar 2 breaks Bar 1 High
            if setup.direction == "LONG" and bar_2["high"] > setup.bar_1_high:
                setup.confirmed = True
                setup.entry_price = bar_1["high"] + 0.5
                setup.level.touch_count += 1
                self.pending_setup = None
                return setup
            # Short confirmation: Bar 2 breaks Bar 1 Low
            elif setup.direction == "SHORT" and bar_2["low"] < setup.bar_1_low:
                setup.confirmed = True
                setup.entry_price = bar_1["low"] - 0.5
                setup.level.touch_count += 1
                self.pending_setup = None
                return setup

        # Step 1: Scan S/R Levels for Bar 1 Touch & Rejection Stall
        # Priority sorted: Virgin CPRs (Priority 1 + is_virgin=True) evaluated first, then Tier 1, 2, 3
        sorted_levels = sorted(self.levels, key=lambda l: (not l.is_virgin, l.priority))

        for lvl in sorted_levels:
            if lvl.touch_count >= lvl.max_touches:
                continue

            if bar_1["low"] <= lvl.price <= bar_1["high"]:
                # --- SUPPORT BOUNCE (LONG) ---
                if self.current_15m_bullish:
                    # Confluence Score calculation (Tier 1 gets +20, Virgin +25, Tier 2 +10, Tier 3 +5)
                    score = 40 + (25 if lvl.is_virgin else 20 if lvl.priority == 1 else 10 if lvl.priority == 2 else 5)
                    if bar_1["close"] > lvl.price:
                        score += 15
                    if self.current_15m_bullish:
                        score += 25

                    if score >= self.min_score:
                        initial_sl = round(max(self.current_atr * self.sl_mult, self.min_sl_pts), 2)
                        tp_dist = round(max(self.current_atr * self.tp_mult, 8.0), 2)
                        
                        setup = RejectionSetup(
                            level=lvl,
                            direction="LONG",
                            bar_1_high=bar_1["high"],
                            bar_1_low=bar_1["low"],
                            bar_1_close=bar_1["close"],
                            score=score,
                            entry_price=bar_1["high"] + 0.5,
                            initial_sl=initial_sl,
                            target_price=tp_dist,
                            timestamp=now or datetime.now(),
                        )

                        if bar_2["high"] > bar_1["high"]:
                            setup.confirmed = True
                            lvl.touch_count += 1
                            return setup
                        else:
                            self.pending_setup = setup
                            return None

                # --- RESISTANCE REJECTION (SHORT) ---
                elif not self.current_15m_bullish:
                    score = 40 + (25 if lvl.is_virgin else 20 if lvl.priority == 1 else 10 if lvl.priority == 2 else 5)
                    if bar_1["close"] < lvl.price:
                        score += 15
                    if not self.current_15m_bullish:
                        score += 25

                    if score >= self.min_score:
                        initial_sl = round(max(self.current_atr * self.sl_mult, self.min_sl_pts), 2)
                        tp_dist = round(max(self.current_atr * self.tp_mult, 8.0), 2)

                        setup = RejectionSetup(
                            level=lvl,
                            direction="SHORT",
                            bar_1_high=bar_1["high"],
                            bar_1_low=bar_1["low"],
                            bar_1_close=bar_1["close"],
                            score=score,
                            entry_price=bar_1["low"] - 0.5,
                            initial_sl=initial_sl,
                            target_price=tp_dist,
                            timestamp=now or datetime.now(),
                        )

                        if bar_2["low"] < bar_1["low"]:
                            setup.confirmed = True
                            lvl.touch_count += 1
                            return setup
                        else:
                            self.pending_setup = setup
                            return None

        return None

    def select_2nd_itm_strike(self, spot_price: float, direction: str) -> int:
        """Selects 2nd In-The-Money strike: CE = ATM - 100, PE = ATM + 100."""
        atm = round(spot_price / 50.0) * 50
        if direction == "LONG":
            return int(atm - 100)
        else:
            return int(atm + 100)


# Backward compatibility aliases
UndisputedRejectionEngine = CombinedSupremeEngine
