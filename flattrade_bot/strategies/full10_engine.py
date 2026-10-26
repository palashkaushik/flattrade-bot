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
      2m/3m/5m  -> the FORMING bucket supplies it (LIVE BOUNCE, 2026-10-16 per
                   user): touch = bucket low-so-far <= lvl+TB and hold =
                   CURRENT price >= lvl-CLOSE_HOLD — no higher-TF close
                   required (waiting for the close loses the chase).
                   NO cross-TF fallback.
  - Virgin level walk: level dies when any 1m bar's low <= lvl <= high
  - Risk geometry (FIXED, v6): SL = entry - 10.0, TP = entry + 15.0
  - NO breakeven lock (removed 2026-09-09 per user)
  - Daily circuit breaker (2026-10-15 per user): touch -29 -> loss capped,
    no new entries after; NO profit lock (entries keep trading past +29).
    (An open trade still manages to SL/TP/EOD.)
  - FUTURES SIGNAL SOURCE (2026-10-16 per user): signals are generated from
    the NIFTY front-month FUTURES 3m chart — FUT:CN (standard rules -> CE)
    and FUT:MR (a negated-price mirror twin -> PE; negation turns dips into
    tops, so the same code yields the mirrored rules). Both states gate on
    the FORMING 3m ELDER candle: allow GREEN/BLUE, block RED (on the mirror
    twin, G/B == the real chart's R/B). Option-chart signals no longer enter
    (fallback only while the futures source is warming).
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
ARM_S1 = 50.0           # arming: S1 <= 50 (per user 2026-10-16; was 30.0)
ARM_WIN = 1             # entry on arming bar or next bar only
FLAG_S4 = 75.0          # FLAG: S4 >= 75 & S1 < 50 (S1 limit 30->50 per user 2026-10-16)
FLAG_S1 = 50.0
ST_ATR = 10.0          # SuperTrend ATR period (3m)
ST_MULT = 3.0          # SuperTrend multiplier
SUP_T = 30.0            # SUPER: all three < 30 & S1 rising
TB = 2.0                # bounce touch buffer
CLOSE_HOLD = 0.5        # bounce close-hold
TOUCH_CAP = 2.0         # pin/eng: wick must stay within +/-2 pts of the level
SL_PTS = 10.0           # v6 fixed geometry
TP_PTS = 15.0
DAILY_CAP_PTS = 29.0    # loss bound: halt new entries when closed day P&L <= -29
DAILY_CAP_PROFIT = float("inf")  # profit lock REMOVED 2026-10-15 per user — the
                        # -29 loss bound stays; entries keep trading past +29
# Per-day overrides {day_key: (loss_pts, profit_pts)} — 2026-10-16 CAP-FREE
# per user (trade the whole session; the -29 bound resumes on the next day).
DAILY_CAP_OVERRIDES = {
    "2026-10-16": (float("inf"), float("inf")),
}

# ---- 3m-Elder gate (futures states only; partial-candle evaluation) ----
# allow iff the FORMING 3m candle color is GREEN or BLUE; RED blocks.
# On the mirrored PE twin this single rule = "real RED or BLUE allows PE".
ELDER3M_A13 = 2.0 / 14.0    # EMA13
ELDER3M_A12 = 2.0 / 13.0    # MACD fast EMA
ELDER3M_A26 = 2.0 / 27.0    # MACD slow EMA
ELDER3M_A9 = 2.0 / 10.0     # MACD signal EMA
ELDER3M_MIN_BUCKETS = 31    # warmup: color allowed only at bucket index >= 30
ELDER3M_FIRST_MIN = 555     # 09:15
ELDER3M_LAST_MIN = 929      # 15:29 (last 1m bar start; buckets are 09:15-aligned)


def _bkt3(minute: int) -> int:
    """3m bucket start for a 1m minute (09:15-aligned: 555, 558, 561, ...)."""
    return ELDER3M_FIRST_MIN + 3 * ((minute - ELDER3M_FIRST_MIN) // 3)



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
    # TFs whose triggers/arming may emit signals (futures states: (3,) — 3m only)
    signal_tfs: Tuple[int, ...] = (1, 2, 3, 5)
    # Elder gate (futures signal states only): allow iff forming 3m color in {G,B}
    elder_gate: bool = False
    # 3m-chart levels (futures states): STANDARD EMA (alpha=2/(N+1) — TV ta.ema)
    # on 3m closes, matching the chart's "EMA (20/200, close, 0)" lines; the
    # DISPLAYED level steps the forming bucket with the latest price.
    use_3m_levels: bool = False
    ema20_3m: IncrementalEMA = field(default_factory=lambda: IncrementalEMA(20))
    ema200_3m: IncrementalEMA = field(default_factory=lambda: IncrementalEMA(200))
    # 5m-chart EMA levels (2026-10-26 per user): the SR levels EMA20/EMA200 are
    # the FIVE-minute chart's EMAs; signals stay on the 3m timeframe.
    ema20_5m: IncrementalEMA = field(default_factory=lambda: IncrementalEMA(20))
    ema200_5m: IncrementalEMA = field(default_factory=lambda: IncrementalEMA(200))
    _b5_start: int = -1
    _b5_last_close: Optional[float] = None
    # 3m Elder Impulse state (partial-candle gate)
    _b3_start: int = -1
    _b3_last_close: Optional[float] = None
    _b3_count: int = 0
    # 3m SuperTrend state (direction filter)
    _st_atr: Optional[float] = None
    _st_up: Optional[float] = None
    _st_dn: Optional[float] = None
    _st_trend: Optional[int] = None
    _st_prev_close: Optional[float] = None
    _b3_hi: Optional[float] = None
    _b3_lo: Optional[float] = None
    _e13_3m: Optional[float] = None
    _e12_3m: Optional[float] = None
    _e26_3m: Optional[float] = None
    _sig9_3m: Optional[float] = None
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
    # U1S union pattern watcher state (3m pin/engulfing, closed candle)
    _pb_idx: Optional[int] = None
    _pb_o: Optional[float] = None
    _pb_h: Optional[float] = None
    _pb_l: Optional[float] = None
    _pb_c: Optional[float] = None
    _pb_prev: Optional[Tuple[float, float, float, float]] = None
    _prev_hit: bool = False

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
        # 3m Elder state resets with the session (re-seeded via seed_bars)
        self._b3_start = -1
        self._b3_last_close = None
        self._b3_count = 0
        self._st_atr = None
        self._st_up = None
        self._st_dn = None
        self._st_trend = None
        self._st_prev_close = None
        self._b3_hi = None
        self._b3_lo = None
        self._e13_3m = self._e12_3m = self._e26_3m = self._sig9_3m = None
        self.ema20_3m = IncrementalEMA(20)
        self.ema200_3m = IncrementalEMA(200)
        self.ema20_5m = IncrementalEMA(20)
        self.ema200_5m = IncrementalEMA(200)
        self._b5_start = -1
        self._b5_last_close = None
        self._pb_idx = None
        self._pb_o = self._pb_h = self._pb_l = self._pb_c = None
        self._pb_prev = None
        self._prev_hit = False
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
            # VWAP must reflect the WHOLE session even on a mid-day restart
            # (previously seeded today-bars skipped VWAP -> wrong session VWAP)
            self.vwap.update(bar.high, bar.low, bar.close, volume=bar.volume)

    def _push_bar_indicators(self, bar: Bar1m, seed: bool):
        """One completed 1m bar through every indicator + TF tracker.

        seed=True: no arming, no virgin death (prior-day warmup walk — the
        reference recorder replays priors through identical trackers)."""
        self.atr.update(bar.high, bar.low, bar.close)
        self.ema20.update(bar.close)
        self.ema200.update(bar.close)
        for tfv in (1, 2, 3, 5):
            self.tf[tfv].push_1m_bar(bar)
        self._feed_3m_bar(bar)
        self._feed_5m_bar(bar)

    # ---- 3m Elder Impulse (partial-candle gate; futures states) ----
    def _feed_3m_bar(self, bar: Bar1m):
        """One completed 1m bar through the 3m resampler (day-boundary safe).

        The 15:30:00 exchange auction print (minute 930) belongs to the LAST
        3m candle of the day ([15:27]) — chart convention (verified against
        the futures chart: the day's final candle close = the official close).
        """
        m = bar.minute
        if not (ELDER3M_FIRST_MIN <= m <= ELDER3M_LAST_MIN + 1):
            return
        b = _bkt3(m if m <= ELDER3M_LAST_MIN else ELDER3M_LAST_MIN)
        if self._b3_start == -1:
            self._b3_start = b
            self._b3_hi = bar.high
            self._b3_lo = bar.low
        elif b != self._b3_start:
            # Bucket advance — within-day rollover (b > start) OR session
            # boundary (b < start: 09:15-aligned starts reset each day, so a
            # decrease is ALWAYS a new day; the post-15:29 phantom start 930
            # must not wedge the feed). Close out the pending bucket with its
            # last available close (matches the certified replay).
            if self._b3_last_close is not None:
                self._push_3m_close(self._b3_last_close)
                self._st_push(self._b3_hi, self._b3_lo, self._b3_last_close)
            self._b3_start = b
            self._b3_last_close = None
            self._b3_hi = bar.high
            self._b3_lo = bar.low
        if self._b3_hi is None or bar.high > self._b3_hi:
            self._b3_hi = bar.high
        if self._b3_lo is None or bar.low < self._b3_lo:
            self._b3_lo = bar.low
        self._b3_last_close = bar.close
        if m == self._b3_start + 2:
            self._push_3m_close(bar.close)
            self._st_push(self._b3_hi, self._b3_lo, bar.close)
            self._b3_start += 3
            self._b3_last_close = None
            self._b3_hi = None
            self._b3_lo = None

    def _feed_5m_bar(self, bar: Bar1m):
        """One completed 1m bar through the 5m resampler (for the 5m EMA levels).

        Signals remain 3m; only the EMA20/EMA200 SR levels use this series.
        """
        m = bar.minute
        if not (ELDER3M_FIRST_MIN <= m <= ELDER3M_LAST_MIN + 1):
            return
        b = 555 + 5 * ((m - 555) // 5)
        if self._b5_start == -1:
            self._b5_start = b
        elif b != self._b5_start:
            if self._b5_last_close is not None:
                self.ema20_5m.update(self._b5_last_close)
                self.ema200_5m.update(self._b5_last_close)
            self._b5_start = b
            self._b5_last_close = None
        self._b5_last_close = bar.close
        if m == self._b5_start + 4:
            self.ema20_5m.update(bar.close)
            self.ema200_5m.update(bar.close)
            self._b5_start += 5
            self._b5_last_close = None

    def _st_push(self, h: Optional[float], l: Optional[float], c: float):
        """One completed 3m bucket through the SuperTrend (Pine ta.supertrend,
        ATR(10) Wilder, mult 3.0). trend +1 = price above ST, -1 = below."""
        if h is None or l is None:
            return
        hl2 = (h + l) / 2.0
        if self._st_atr is None:
            self._st_atr = h - l
            self._st_up = hl2 - ST_MULT * self._st_atr
            self._st_dn = hl2 + ST_MULT * self._st_atr
            self._st_trend = 1 if c > hl2 else -1
            self._st_prev_close = c
            return
        tr = max(h - l, abs(h - self._st_prev_close), abs(l - self._st_prev_close))
        a = 1.0 / ST_ATR
        self._st_atr = a * tr + (1 - a) * self._st_atr
        up = hl2 - ST_MULT * self._st_atr
        dn = hl2 + ST_MULT * self._st_atr
        prev_up, prev_dn = self._st_up, self._st_dn
        if self._st_prev_close > prev_up:
            up = max(up, prev_up)
        if self._st_prev_close < prev_dn:
            dn = min(dn, prev_dn)
        prev_trend = self._st_trend
        if prev_trend == -1 and c > prev_dn:
            trend = 1
        elif prev_trend == 1 and c < prev_up:
            trend = -1
        else:
            trend = prev_trend
        self._st_up, self._st_dn, self._st_trend = up, dn, trend
        self._st_prev_close = c

    def _st_partial(self, px: float, h: float, l: float):
        """SuperTrend evaluated on the FORMING 3m bucket (2026-10-26 per user).

        Uses the last COMPLETED bucket's state and steps it with the forming
        candle's high/low-so-far and the current price — exactly what the chart
        shows intrabar. Does not mutate state (the bucket close commits it).
        Returns (trend, line); (None, None) during warmup.
        """
        if self._st_atr is None or self._st_up is None or self._st_dn is None:
            return None, None
        hl2 = (h + l) / 2.0
        tr = max(h - l, abs(h - self._st_prev_close), abs(l - self._st_prev_close))
        a = 1.0 / ST_ATR
        atr = a * tr + (1 - a) * self._st_atr
        up_new = hl2 - ST_MULT * atr
        dn_new = hl2 + ST_MULT * atr
        prev_up, prev_dn = self._st_up, self._st_dn
        up = max(up_new, prev_up) if self._st_prev_close > prev_up else up_new
        dn = min(dn_new, prev_dn) if self._st_prev_close < prev_dn else dn_new
        trend = self._st_trend
        if trend == -1 and px > prev_dn:
            trend = 1
        elif trend == 1 and px < prev_up:
            trend = -1
        return trend, (up if trend == 1 else dn)

    def _push_3m_close(self, c: float):
        """EMA13 / MACD(12,26,9) + chart RMA20/200 update on one completed 3m
        close (first-value seed)."""
        self.ema20_3m.update(c)
        self.ema200_3m.update(c)
        if self._e13_3m is None:
            self._e13_3m = self._e12_3m = self._e26_3m = c
            self._sig9_3m = 0.0   # first macd-line = e12-e26 = 0 (replay parity)
        else:
            self._e13_3m = ELDER3M_A13 * c + (1 - ELDER3M_A13) * self._e13_3m
            self._e12_3m = ELDER3M_A12 * c + (1 - ELDER3M_A12) * self._e12_3m
            self._e26_3m = ELDER3M_A26 * c + (1 - ELDER3M_A26) * self._e26_3m
            ml = self._e12_3m - self._e26_3m
            self._sig9_3m = ELDER3M_A9 * ml + (1 - ELDER3M_A9) * self._sig9_3m
        self._b3_count += 1

    def _elder3m_partial_color(self, px: float) -> Optional[str]:
        """Color of the FORMING 3m candle at price px — EXACT conformance with
        the Pine reference (Elder Impulse System, CC0): GREEN = ema[0]>ema[1]
        AND hist[0]>hist[1]; RED = ema[0]<ema[1] AND hist[0]<hist[1]; else
        BLUE. Prior state = last completed bucket; one partial step with px.
        On a negated (mirror) series the color comes out flipped (G<->R, B stays B)."""
        if self._b3_count < ELDER3M_MIN_BUCKETS or self._e13_3m is None:
            return None
        e13p = self._e13_3m
        e12p = self._e12_3m
        e26p = self._e26_3m
        s9p = self._sig9_3m
        hp = (e12p - e26p) - s9p
        e13c = ELDER3M_A13 * px + (1 - ELDER3M_A13) * e13p
        e12c = ELDER3M_A12 * px + (1 - ELDER3M_A12) * e12p
        e26c = ELDER3M_A26 * px + (1 - ELDER3M_A26) * e26p
        mlc = e12c - e26c
        s9c = ELDER3M_A9 * mlc + (1 - ELDER3M_A9) * s9p
        hc = mlc - s9c
        if (e13c > e13p) and (hc > hp):
            return "G"
        if (e13c < e13p) and (hc < hp):
            return "R"
        return "B"

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

        # === UNION TRIGGER (2026-10-16, U1S live): the stoch flow (arming →
        #     FLAG/SUPER → same-TF LIVE bounce → SR level) OR the strict
        #     closed-candle pin/engulfing rejecting an SR level. Both run on
        #     every bar with independent state; when both fire on the same
        #     bar the stoch signal wins (the U1S backtest tie-break).
        s_hit = None
        s_tfv = None
        s_kind = None

        # 3. arming: S1 <= 30 on ANY signal-TF -> anchor this baridx
        for tfv in self.signal_tfs:
            s1 = self.tf[tfv].last_s1
            if s1 is not None and s1 <= ARM_S1:
                self.anchor_bar = self.baridx
                break

        # 4. arming window gate (stoch path only)
        if (self.baridx - self.anchor_bar) <= ARM_WIN:

            # 5. triggers on any signal-TF (dormancy REMOVED)
            trig = None
            for tfv in self.signal_tfs:
                t = self.tf[tfv]
                s1, s3, s4, p1 = t.last_s1, t.last_s3, t.last_s4, t.prev_s1
                if s1 is None or s4 is None:
                    continue
                if s4 >= FLAG_S4 and s1 < FLAG_S1:
                    trig = (tfv, "FLAG"); break
                if s3 is not None and p1 is not None and s3 < SUP_T and s4 < SUP_T \
                        and s1 < SUP_T and s1 > p1:
                    trig = (tfv, "SUPER"); break

            if trig is not None:
                tfv, _k = trig

                # 6. SAME-TF bounce gate (v6): the setup TF's own chart
                #    supplies it. LIVE BOUNCE (2026-10-16 per user): for >1m
                #    setups the bounce is checked against the FORMING bucket —
                #    the touch comes from the bucket's low-so-far and the hold
                #    from the CURRENT price — so a valid bounce does NOT need
                #    the higher-TF candle to close first (closing first loses
                #    the chase). At a bucket boundary the just-completed bar
                #    is used, exactly as before.
                if tfv == 1:
                    lo_, cl_ = bar.low, bar.close
                else:
                    t = self.tf[tfv]
                    if t.cur_bars:
                        lo_ = min(b.low for b in t.cur_bars)
                    else:
                        lo_ = t.last_lo
                    cl_ = bar.close
                if lo_ is not None:
                    lv = self._levels_now()
                    for nm, v in lv:
                        if lo_ <= v + TB and cl_ >= v - CLOSE_HOLD:
                            s_hit = (nm, v)
                            s_tfv = tfv
                            s_kind = _k
                            break

        # 7a. 3m PIN/ENG PATH (strict, closed candle — verbatim from the
        #     backtested U1S pattern engine; independent edge state).
        bidx = (bar.minute - 555) // 3
        if getattr(self, "_pb_idx", None) != bidx:
            if getattr(self, "_pb_idx", None) is not None:
                self._pb_prev = (self._pb_o, self._pb_h, self._pb_l,
                                 getattr(self, "_pb_c", bar.open))
            self._pb_idx = bidx
            self._pb_o = bar.open
            self._pb_h = bar.high
            self._pb_l = bar.low
        else:
            self._pb_h = max(self._pb_h, bar.high)
            self._pb_l = min(self._pb_l, bar.low)
        self._pb_c = bar.close
        # PARTIAL-CANDLE PIN (2026-10-26 per user): pin/engulfing is evaluated
        # on the FORMING 3m candle at every 1m close — no waiting for the
        # bucket close. The shape must still reject an SR level.
        p_hit = None
        c = bar.close
        o_ = self._pb_o
        hi_ = self._pb_h
        lo2 = self._pb_l
        rng = hi_ - lo2
        if rng > 0:
            body = abs(c - o_)
            upw = hi_ - max(o_, c)
            dnw = min(o_, c) - lo2
            # LENIENT PIN (2026-10-26 per user): wick >= 0.5x body (was 2.0x);
            # still requires the wick to be >= the opposite wick and to
            # reject an SR level.
            is_pin = (dnw >= 0.5 * body and dnw >= upw)
            is_eng = False
            prev = getattr(self, "_pb_prev", None)
            if prev is not None:
                po, ph, pl, pc_ = prev
                if c > o_ and c >= max(po, pc_) and o_ <= min(po, pc_):
                    is_eng = True
            if is_pin or is_eng:
                lv2 = self._levels_now()
                for nm, v in lv2:
                    if v is None:
                        continue
                    # PRECISE TOUCH (2026-10-26 per user): the wick must sit
                    # within +/-2 pts of the level — it must reach it, but may
                    # not pierce more than 2 pts beyond (this blocks the 12:15
                    # case where the PE wick ran 28 pts through the EMA20).
                    if (v - TOUCH_CAP) <= lo2 <= (v + TB) and c >= v - CLOSE_HOLD:
                        p_hit = (nm, v)
                        break
                if p_hit is None:
                    self._prev_hit = False
                elif getattr(self, "_prev_hit", False):
                    self._prev_hit = True
                    p_hit = None
                else:
                    self._prev_hit = True
                    self._pb_kind = "PIN" if is_pin else "ENG"

        # Selection: stoch flow first (U1S replay tie-break), else pattern.
        if s_hit is not None:
            kind, tfv, hit = s_kind, s_tfv, s_hit
        elif p_hit is not None:
            kind, tfv, hit = getattr(self, "_pb_kind", "PIN"), 3, p_hit
        else:
            return None

        # 7. 3m-ELDER GATE (futures signal states only): allow iff the FORMING
        #    3m Elder candle is GREEN or BLUE; RED (and warmup) block.
        #    On the mirrored PE twin this exact rule = "real RED or BLUE".
        if self.elder_gate:
            _col = self._elder3m_partial_color(bar.close)
            if _col not in ("G", "B"):
                logger.info(
                    "3m-Elder filter: %s %s [%dm] blocked @ %.2f (color=%s) on %s",
                    self.side, kind, tfv, bar.close, _col or "warmup", self.symbol,
                )
                return None

        # SUPERTREND DIRECTION FILTER (2026-10-26 per user: PARTIAL candle):
        # CE entries only while price is ABOVE the 3m SuperTrend (trend == +1);
        # PE only while BELOW. Evaluated on the FORMING 3m bucket (last closed
        # state stepped with the bucket's high/low-so-far + current price), so
        # intrabar flips are seen immediately. The mirrored PE twin negates the
        # series, so its SuperTrend flips too and the same test yields
        # "real price below ST".
        _st_h = self._pb_h if self._pb_h is not None else bar.high
        _st_l = self._pb_l if self._pb_l is not None else bar.low
        _st_trend_p, _st_line = self._st_partial(bar.close, _st_h, _st_l)
        if _st_trend_p is None:
            logger.info(
                "SuperTrend filter: %s %s [%dm] blocked @ %.2f (warmup) on %s",
                self.side, kind, tfv, bar.close, self.symbol)
            return None
        if _st_trend_p != 1:
            logger.info(
                "SuperTrend filter: %s %s [%dm] blocked @ %.2f (below ST %.2f) on %s",
                self.side, kind, tfv, bar.close, _st_line or 0.0, self.symbol)
            return None

        # FUTURES-CANDLE SL (2026-10-26 per user): long -> 0.5 pts below the
        # entry 3m candle's low; the mirrored short twin evaluates the same
        # rule in mirror space = 0.5 pts above the real candle's high.
        # Snapshot at signal time (the forming bucket, or the close-handled
        # bucket at a boundary).
        _t3 = self.tf[3]
        if _t3.cur_bars:
            _lo3 = min(b.low for b in _t3.cur_bars)
        else:
            _lo3 = _t3.last_lo
        if _lo3 is None:
            _lo3 = bar.low
        sl_fut = _r2(_lo3 - 0.5)

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
            "sl_fut": sl_fut,
            "sl_points": SL_PTS,
            "tp_points": TP_PTS,
            "bar": bar,
            "timestamp": bar.timestamp,
        }

    def _levels_now(self) -> List[Tuple[str, float]]:
        # 2026-10-26 per user: EMA20/EMA200 SR levels are the FIVE-minute
        # chart's EMAs (signals stay 3m). Chart-live semantics: the forming 5m
        # candle's close-so-far steps the EMA (exactly what the chart displays).
        if self.use_3m_levels and self.ema20_5m.value is not None and self.ema200_5m.value is not None:
            px = self.latest_ltp
            if px is None and self.bars:
                px = self.bars[-1].close
            if px is not None:
                a20 = self.ema20_5m.alpha
                a200 = self.ema200_5m.alpha
                e20 = a20 * px + (1 - a20) * self.ema20_5m.value
                e200 = a200 * px + (1 - a200) * self.ema200_5m.value
            else:
                e20, e200 = self.ema20_5m.value, self.ema200_5m.value
        else:
            e20, e200 = self.ema20.value, self.ema200.value
        lv: List[Tuple[str, float]] = [
            ("EMA20", e20), ("EMA200", e200),
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
