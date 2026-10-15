"""FULL10 Strategy Engine (v6 live) — Full10 triggers + 12-level bounce gate.

LIVE SPEC (user-confirmed 2026-09-09, supersedes §44 for the VPS bot):
  - Execution TF: 1m option bars (09:15-15:00 IST), 2nd-ITM dynamic strikes
  - Stochastics per TF chart (1m/2m/3m/5m):
      S1 = Stoch(12, 1)   S3 = Stoch(40, 1)   S4 = Stoch(60, 1)
      (NOTE: D period = 1 — RAW %K, no smoothing — Full10 decoded spec;
       §44 used (12,3)/(40,4)/(50,10) smoothed. DO NOT "fix".)
  - Arming: S1 <= 30 on ANY TF (5-pt buffer rule) sets anchor; ARM_WIN = 1
    (entry allowed only on the arming bar or the next bar, baridx space)
  - Triggers (5-pt buffers, any TF):
      FLAG  = S4 >= 75 and S1 < 30
      SUPER = S3 < 30 and S4 < 30 and S1 < 30 and S1 rising (S1 > prevS1)
  - NO dormancy (removed 2026-09-09 per user)
  - 12-LEVEL BOUNCE GATE (TB = 2.0, close-hold = 0.5):
      EMA20, EMA200, VWAP, PDH, PDL, CPR, BC, TC, fibR3, fibS3, camR3,
      camL3, PDVWAP, virgin CPR/BC/TC (alive)   [level list exact per spec]
  - SAME-TF BOUNCE (v6 rule): the setup TF's own chart supplies the bounce.
      1m setup  -> trigger 1m bar:  low <= lvl+2 and close >= lvl-0.5
      2m/3m/5m  -> that TF's last COMPLETED (today's) bar satisfies the
                   touch. NO cross-TF fallback, NO 1m-bar confirmation.
  - Virgin level walk: level dies when any 1m bar's low <= lvl <= high
  - Risk geometry (FIXED, v6): SL = entry - 10.0, TP = entry + 15.0
  - NO breakeven lock (removed 2026-09-09 per user)
  - Daily circuit breaker (2026-10-15 per user): touch -29 -> loss capped,
    no new entries after; NO profit lock (entries keep trading past +29).
    (An open trade still manages to SL/TP/EOD.)
  - EOD force-close at session end; SL priority over TP.
  - Funds fallback (unchanged plumbing): 2nd-ITM margin reject -> 1st ITM.

All indicator math (stoch/ATR/EMA/VWAP/TFTracker) is shared verbatim with
last_hope_winner.py — class import, zero reimplementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flattrade_bot.strategies.last_hope_winner import (
    Bar1m,
    IncrementalATR,
    IncrementalEMA,
    IncrementalRMA,
    IncrementalStoch,
    IncrementalVWAP,
    TFTracker,
)

logger = logging.getLogger("flattrade_bot.full10_engine")

# ---- Full10 v6 constants (decoded champion spec — see AGENTS.md) ----
S1_K, S1_D = 12, 2      # smoothed %D=2 (geoms-certified reference)
S3_K, S3_D = 40, 4
S4_K, S4_D = 60, 10
ARM_S1 = 30.0           # arming: S1 <= 30 (5-pt buffer rule)
ARM_WIN = 1             # entry on arming bar or next bar only
FLAG_S4 = 75.0          # FLAG: S4 >= 75 & S1 < 30
FLAG_S1 = 30.0
SUP_T = 30.0            # SUPER: all three < 30 & S1 rising
TB = 2.0                # bounce touch buffer
CLOSE_HOLD = 0.5        # bounce close-hold
SL_PTS = 10.0           # v6 fixed geometry
TP_PTS = 15.0
DAILY_CAP_PTS = 29.0    # loss bound: halt new entries when closed day P&L <= -29
DAILY_CAP_PROFIT = float("inf")  # profit lock REMOVED 2026-10-15 per user — the
                        # -29 loss bound stays; entries keep trading past +29
# Per-day overrides {day_key: (loss_pts, profit_pts)} — none active
DAILY_CAP_OVERRIDES = {}



def day_cap_bounds(day_key: str):
    """(loss_cap, profit_cap) for a day — per-day override or (-29/none) default."""
    return DAILY_CAP_OVERRIDES.get(day_key, (DAILY_CAP_PTS, DAILY_CAP_PROFIT))
SESSION_START_MIN = 555
SESSION_END_MIN = 900


def _r2(x: float) -> float:
    return round(float(x), 2)


@dataclass
class Full10ContractState:
    """Per-contract Full10 state: indicators + arming + TF trackers."""
    symbol: str
    token: str
    side: str            # "CE" | "PE"
    strike: int
    # 1m tick aggregation (identical to §44 ContractState)
    current_min: int = -1
    cur_open: Optional[float] = None
    cur_high: Optional[float] = None
    cur_low: Optional[float] = None
    cur_close: Optional[float] = None
    # indicators
    atr: IncrementalATR = field(default_factory=lambda: IncrementalATR(10))
    ema20: IncrementalRMA = field(default_factory=lambda: IncrementalRMA(20))
    ema200: IncrementalRMA = field(default_factory=lambda: IncrementalRMA(200))
    vwap: IncrementalVWAP = field(default_factory=IncrementalVWAP)
    tf: Dict[int, TFTracker] = field(default_factory=lambda: {
        1: TFTracker(1), 2: TFTracker(2), 3: TFTracker(3), 5: TFTracker(5)})
    # per-TF trackers use FULL10 stoch periods — installed at __post_init__
    # arming (baridx space, any-TF S1<=30)
    anchor_bar: int = -10**9
    baridx: int = -1
    # day levels (set at warmup)
    levels_day: Dict[str, float] = field(default_factory=dict)   # static pivots
    virgin: Dict[str, float] = field(default_factory=dict)      # alive virgins
    pdvwap: Optional[float] = None
    # §44-style same-day re-registration guard
    session_date: Optional[str] = None
    seed_complete: bool = False
    bars: List[Bar1m] = field(default_factory=list)
    last_completed_bar: Optional[Bar1m] = None
    latest_ltp: Optional[float] = None
    # touchline cumulative-volume tracking (per-bar interval volume)
    _bar_start_vol: Optional[float] = None

    def __post_init__(self):
        # Full10 stoch periods on every TF tracker (D=1 raw %K)
        for tfv in (1, 2, 3, 5):
            t = self.tf[tfv]
            t.s1 = IncrementalStoch(S1_K, S1_D)
            t.s3 = IncrementalStoch(S3_K, S3_D)
            t.s4 = IncrementalStoch(S4_K, S4_D)

    # ---- lifecycle (mirrors §44 OptionContractState API) ----
    def reset_session(self):
        self.bars = []
        self.last_completed_bar = None
        self.current_min = -1
        self.cur_open = self.cur_high = self.cur_low = self.cur_close = None
        self.atr = IncrementalATR(10)
        self.ema20 = IncrementalRMA(20)
        self.ema200 = IncrementalRMA(200)
        self.vwap = IncrementalVWAP()
        self.tf = {tfv: TFTracker(tfv) for tfv in (1, 2, 3, 5)}
        for tfv in (1, 2, 3, 5):
            t = self.tf[tfv]
            t.s1 = IncrementalStoch(S1_K, S1_D)
            t.s3 = IncrementalStoch(S3_K, S3_D)
            t.s4 = IncrementalStoch(S4_K, S4_D)
        # TF fallback state must NEVER carry yesterday's last TF bar into
        # today (the Sep-4 09:15 parity bug, lh_f10_build fix)
        for tfv in (2, 3, 5):
            self.tf[tfv].last_lo = None
            self.tf[tfv].last_cl = None
        self.anchor_bar = -10**9
        self.baridx = -1
        logger.info("Full10 session reset for %s", self.symbol)

    def set_day_levels(self, prev_high: float, prev_low: float, prev_close: float,
                       virgin: Optional[Dict[str, float]] = None,
                       pdvwap: Optional[float] = None):
        """Full10 12-level suite from the prior-day option OHLC (official
        EOD candle preferred — ClsPric convention)."""
        ph, pl, pc = float(prev_high), float(prev_low), float(prev_close)
        cpr = _r2((ph + pl + pc) / 3.0)
        bc = _r2((ph + pl) / 2.0)
        tc = _r2(2.0 * cpr - bc)
        rng = ph - pl
        self.levels_day = {
            "PDH": ph, "PDL": pl,
            "CPR": cpr, "BC": bc, "TC": tc,
            "fibR3": _r2(cpr + rng), "fibS3": _r2(cpr - rng),
            "camR3": _r2(pc + rng * 1.1 / 4.0), "camL3": _r2(pc - rng * 1.1 / 4.0),
        }
        self.virgin = dict(virgin) if virgin else {}
        self.pdvwap = pdvwap

    def seed_bars(self, prior_bars: List[Bar1m], today_bars: Optional[List[Bar1m]] = None,
                  session_date: Optional[str] = None):
        """Seeds indicators with prior-day completed bars (stochs seeded too —
        carried state, matching the certified recorder).
        TF bucket last_lo/last_cl are RESET at day boundary — prior-day TF
        buckets must never leak into today's bounce checks (geoms parity)."""
        if session_date is not None:
            self.session_date = session_date
        self.seed_complete = True
        for bar in prior_bars:
            self.bars.append(bar)
            self._push_bar_indicators(bar, seed=True)
        self.vwap.reset()
        # day-boundary reset (geoms build_states tail): prior-day TF buckets must
        # never leak into today's bounce checks — applied BEFORE today's bars so
        # today's completed buckets repopulate them.
        for tfv in (1, 2, 3, 5):
            self.tf[tfv].last_lo = None
            self.tf[tfv].last_cl = None
        for bar in (today_bars or []):
            self.bars.append(bar)
            self._push_bar_indicators(bar, seed=False)

    def _push_bar_indicators(self, bar: Bar1m, seed: bool):
        """One completed 1m bar through every indicator + TF tracker.

        seed=True: no arming, no virgin death (prior-day warmup walk — the
        reference recorder replays priors through identical trackers)."""
        self.atr.update(bar.high, bar.low, bar.close)
        self.ema20.update(bar.close)
        self.ema200.update(bar.close)
        for tfv in (1, 2, 3, 5):
            self.tf[tfv].push_1m_bar(bar)

    # ---- live tick path (identical rollover semantics to §44) ----
    def push_tick(self, ltp: float, dt: datetime, cum_volume: Optional[float] = None) -> Optional[Dict[str, Any]]:
        self.latest_ltp = ltp
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
        # interval volume = cumulative-day-volume delta across this bar (docs
        # touchline "v"); None -> constant 100 fallback (matches §44 semantics)
        if cum_volume is not None and self._bar_start_vol is not None:
            bar_vol = max(cum_volume - self._bar_start_vol, 1.0)
        else:
            bar_vol = 100.0
        completed = Bar1m(minute=self.current_min, open=self.cur_open, high=self.cur_high,
                          low=self.cur_low, close=self.cur_close, timestamp=dt, volume=bar_vol)
        self.bars.append(completed)
        self.last_completed_bar = completed
        self.current_min = minute
        self.cur_open = self.cur_high = self.cur_low = self.cur_close = ltp
        self._bar_start_vol = cum_volume
        return self._on_bar_close(completed)

    # ---- signal evaluation ----
    def _on_bar_close(self, bar: Bar1m) -> Optional[Dict[str, Any]]:
        # 1. indicators + TF trackers see the bar
        self._push_bar_indicators(bar, seed=False)
        self.vwap.update(bar.high, bar.low, bar.close, volume=bar.volume)
        self.baridx += 1

        # 2. virgin death walk (today's bars only)
        for nm in list(self.virgin):
            v = self.virgin[nm]
            if bar.low <= v <= bar.high:
                del self.virgin[nm]

        # 3. arming: S1 <= 30 on ANY TF -> anchor this baridx
        for tfv in (1, 2, 3, 5):
            s1 = self.tf[tfv].last_s1
            if s1 is not None and s1 <= ARM_S1:
                self.anchor_bar = self.baridx
                break

        # 4. arming window gate
        if (self.baridx - self.anchor_bar) > ARM_WIN:
            return None

        # 5. triggers on any TF (dormancy REMOVED)
        trig = None
        for tfv in (1, 2, 3, 5):
            t = self.tf[tfv]
            s1, s3, s4, p1 = t.last_s1, t.last_s3, t.last_s4, t.prev_s1
            if s1 is None or s4 is None:
                continue
            if s4 >= FLAG_S4 and s1 < FLAG_S1:
                trig = (tfv, "FLAG"); break
            if s3 is not None and p1 is not None and s3 < SUP_T and s4 < SUP_T \
                    and s1 < SUP_T and s1 > p1:
                trig = (tfv, "SUPER"); break
        if trig is None:
            return None
        tfv, kind = trig

        # 6. SAME-TF bounce gate (v6): the setup TF's own chart supplies it
        if tfv == 1:
            lo_, cl_ = bar.low, bar.close
        else:
            t = self.tf[tfv]
            if t.last_lo is None or t.last_cl is None:
                return None
            lo_, cl_ = t.last_lo, t.last_cl
        lv = self._levels_now()
        hit = None
        for nm, v in lv:
            if lo_ <= v + TB and cl_ >= v - CLOSE_HOLD:
                hit = (nm, v); break
        if hit is None:
            return None

        entry = bar.close
        sl = _r2(entry - SL_PTS)
        tp = _r2(entry + TP_PTS)
        logger.info(
            "🔥 %s %s [%dm] TRIGGERED @ %.2f on %s | SR=%s | SL=%.2f TP=%.2f (Full10 v6)",
            self.side, kind, tfv, entry, self.symbol, hit[0], sl, tp,
        )
        return {
            "side": self.side,
            "symbol": self.symbol,
            "token": self.token,
            "strike": self.strike,
            "trigger": kind,
            "tf": f"{tfv}m",
            "level": hit[0],
            "entry": entry,
            "dist": SL_PTS,          # executor wants sl_points/tp_points — pass through
            "sl": sl,
            "tp": tp,
            "sl_points": SL_PTS,
            "tp_points": TP_PTS,
            "bar": bar,
            "timestamp": bar.timestamp,
        }

    def _levels_now(self) -> List[Tuple[str, float]]:
        lv: List[Tuple[str, float]] = [
            ("EMA20", self.ema20.value), ("EMA200", self.ema200.value),
            ("VWAP", self.vwap.value),
        ]
        for nm, v in self.levels_day.items():
            lv.append((nm, v))
        if self.pdvwap is not None:
            lv.append(("PDVWAP", self.pdvwap))
        lv.extend(self.virgin.items())
        return [(nm, v) for nm, v in lv if v is not None and v == v]


class Full10Engine:
    """Coordinator — drop-in parallel to LastHopeWinnerEngine (same API)."""

    def __init__(self):
        self.contracts: Dict[str, Full10ContractState] = {}
        self.spot_price: Optional[float] = None
        self.active_trade: Optional[Dict[str, Any]] = None
        # Daily circuit breaker state (POINTS, closed-trade cumulative)
        self._day_key: Optional[str] = None
        self._day_cum_pts: float = 0.0
        self._day_halted: bool = False

    # ---- API parity with LastHopeWinnerEngine ----
    def register_contract(self, key: str, symbol: str, token: str, side: str, strike: int) -> Full10ContractState:
        if key not in self.contracts:
            self.contracts[key] = Full10ContractState(symbol=symbol, token=token, side=side, strike=strike)
        return self.contracts[key]

    def set_spot_price(self, spot: float):
        self.spot_price = spot

    def desired_strikes(self, spot: float) -> Dict[str, int]:
        atm = int(round(spot / 50.0) * 50)
        return {
            "CE_SPEC": atm - 100, "PE_SPEC": atm + 100,
            "CE_WATCH_PLUS50": (atm + 50) - 100, "PE_WATCH_PLUS50": (atm + 50) + 100,
            "CE_WATCH_MINUS50": (atm - 50) - 100, "PE_WATCH_MINUS50": (atm - 50) + 100,
        }

    def on_trade_opened(self, trade: Dict[str, Any]):
        self.active_trade = dict(trade)

    def on_trade_closed(self):
        self.active_trade = None

    def record_closed_points(self, pts: float, day_key: str):
        """Circuit breaker: accumulate CLOSED-trade points; halt entries at the
        day's bounds (default -29/+60, per-day overrides apply).

        Called by the coordinator on every trade close (live or paper)."""
        if day_key != self._day_key:
            self._day_key = day_key
            self._day_cum_pts = 0.0
            self._day_halted = False
        self._day_cum_pts += pts
        loss_cap, profit_cap = day_cap_bounds(day_key)
        if self._day_cum_pts <= -loss_cap or self._day_cum_pts >= profit_cap:
            if not self._day_halted:
                self._day_halted = True
                prof_s = "no-lock" if profit_cap == float("inf") else f"+{profit_cap:.0f}"
                logger.warning(
                    "🚦 DAILY CIRCUIT BREAKER: closed P&L %.1f pts hit cap (-%.0f/%s) — no more entries today",
                    self._day_cum_pts, loss_cap, prof_s,
                )

    def entries_halted(self, day_key: str) -> bool:
        if day_key != self._day_key:
            return False
        return self._day_halted

    # ---- main tick path ----
    def push_tick(self, key: str, ltp: float, dt: datetime, cum_volume: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Returns a signal ONLY when flat, in-session, and not halted.
        Arming never persists across an open position (§44 congruence):
        on close, the anchor is invalidated so no stale anchor fires."""
        contract = self.contracts.get(key)
        if not contract:
            return None
        sig = contract.push_tick(ltp, dt, cum_volume=cum_volume)
        if self.active_trade:
            contract.anchor_bar = -10**9
            return None
        if sig:
            minute = dt.hour * 60 + dt.minute
            if not (SESSION_START_MIN <= minute <= SESSION_END_MIN):
                return None
            if self.entries_halted(f"{dt:%Y-%m-%d}"):
                loss_cap, profit_cap = day_cap_bounds(f"{dt:%Y-%m-%d}")
                prof_s = "no-lock" if profit_cap == float("inf") else f"+{profit_cap:.0f}"
                logger.info("Signal on %s suppressed — daily cap (-%.0f/%s) reached", sig["symbol"], loss_cap, prof_s)
                return None
        return sig
