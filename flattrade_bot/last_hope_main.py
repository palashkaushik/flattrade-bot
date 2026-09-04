"""Last Hope Winner Strategy Bot — Main Entry Point & Dashboard.

Strategy: 🏆 Last Hope GPU Winner (7-Year Net ₹2,108,703 | 63.89% Win Rate | Max DD ₹9,303 | Calmar 226.68)
Specifications from LAST_HOPE_WINNER.md:
  - 1-minute option OHLC bars (09:15–15:00 IST)
  - 2nd ITM strikes (CE = ATM - 100, PE = ATM + 100)
  - Multi-TF Option Stochastics (1m, 2m, 3m, 5m): S1(12,3), S3(40,4), S4(50,10)
  - 10-bar Arming window (S1 <= 25.0)
  - Triggers: Flag (M6: S4 >= 79.5 & S1 < 79.5) / Super (S3,S4,S1 < 25 & S1 rising)
  - S/R Bounce Gate (touch_buffer = 0.0): Candle low <= S/R level and close >= level - 0.5
  - Risk Geometry: dist = min(ATR(10) * 1.5, 15.0 pts), SL = Entry - dist, TP = Entry + dist
  - Breakeven Stop (BE): At +50% of distance, SL hardens to Entry + 1.0 pt
  - SL priority over TP
"""

import asyncio
import logging
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattrade_bot.broker.client import FlattradeClient
from flattrade_bot.broker.history import FlattradeHistoryFetcher, is_session_expired_response
from flattrade_bot.config import settings
from flattrade_bot.control import touch_runtime_record
from flattrade_bot.execution import TradeExecutor
from flattrade_bot.risk.manager import RiskManager
from flattrade_bot.strategies.last_hope_winner import (
    ARM_WINDOW,
    ATR_MULT,
    Bar1m,
    LastHopeWinnerEngine,
    M6_S1,
    M6_S4,
    OptionContractState,
    SESSION_END_MIN,
    SESSION_START_MIN,
    SUPER_THRESH,
    TP_PTS_CAP,
)
from flattrade_bot.utils.discord import DiscordNotifier

# Rich is used OFF-SCREEN ONLY: render tables to a string (no Live thread —
# that caused the SSH blank screens), then redraw via zero-flicker ANSI.
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Live's console: real stdout, auto-detected terminal (tmux PTY on the VPS).
# Live(screen=True) uses the alternate buffer and its own 1 fps refresher —
# only the event loop calls live.update() with a freshly built renderable.
_live_console = Console()

STRATEGY_LABEL = "Last Hope Winner Strategy"
IST = timezone(timedelta(hours=5, minutes=30))

log_dir = ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "last_hope_bot.log"

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[file_handler])
logger = logging.getLogger("flattrade_bot.last_hope_main")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def ist_now() -> datetime:
    return datetime.now(IST)


def minute_of(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


_EXP_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def expiry_info(symbol: str, now: Optional[datetime] = None) -> str:
    """Parses 'NIFTY08SEP26C24000' -> '08SEP26' + days-to-expiry tag.

    Returns '' when the symbol carries no parseable expiry token.
    """
    m = re.search(r"(\d{2})([A-Z]{3})(\d{2})", str(symbol or ""))
    if not m:
        return ""
    try:
        d = date(2000 + int(m.group(3)), _EXP_MONTHS[m.group(2)], int(m.group(1)))
    except (KeyError, ValueError):
        return ""
    tok = f"{m.group(1)}{m.group(2)}{m.group(3)}"
    ref = (now or ist_now())
    days = (d - ref.date()).days
    if days < 0:
        return f"{tok} (EXP)"
    if days == 0:
        return f"{tok} (EXPIRY TODAY)"
    return f"{tok} ({days}d)"


class LastHopeTradingEngine:
    """Live trading engine for the Last Hope GPU Winner strategy."""

    def __init__(self, live_orders: bool = False):
        self.live_orders = live_orders
        self.engine = LastHopeWinnerEngine()
        self.discord = DiscordNotifier(strategy=STRATEGY_LABEL)
        self.client = FlattradeClient()
        self.history = FlattradeHistoryFetcher()
        from flattrade_bot.broker.ws_feed import FlattradeWebSocketFeed
        self.ws_feed = FlattradeWebSocketFeed()
        self.risk = RiskManager(
            max_daily_loss_points=math.inf,
            quantity=settings.LOT_SIZE,
            consecutive_loss_limit=4,
            session_start_min=SESSION_START_MIN,
            session_end_min=SESSION_END_MIN,
        )
        self.executor = (
            TradeExecutor(self.client, self.risk, self.discord, quantity=settings.LOT_SIZE, live_orders=True)
            if live_orders
            else None
        )

        self.spot_price: Optional[float] = None
        self.active_position_key: Optional[str] = None
        self.paper_position: Optional[Dict[str, Any]] = None  # SIM mode only
        self.trades_today: List[Dict[str, Any]] = []
        self._wins_today = 0
        self._broker_status = "INITIALIZING..."
        self._last_ltp: Dict[str, float] = {}
        self._last_contract_check = 0.0
        self._last_token_renew = 0.0
        self._eod_done = False
        self._current_day: Optional[date] = None
        self._contract_task: Optional[asyncio.Task] = None
        self._exit_task: Optional[asyncio.Task] = None
        self._entering: bool = False
        self._dash_lines_drawn: int = 0
        self._alt_screen_on: bool = False

    async def initialize(self):
        logger.info("Initializing %s (live_orders=%s)...", STRATEGY_LABEL, self.live_orders)

        token = os.getenv("FLATTRADE_TOKEN", "")
        if token:
            self.client.set_token(token)
            self.history.set_token(token)
            q = await self.client.get_quotes(exchange="NSE", token="26000")
            if q.get("stat") != "Ok" or is_session_expired_response(q):
                logger.warning("Token expired. Triggering automated zero-touch login...")
                token = ""

        if not token and settings.FLATTRADE_USER_ID and settings.FLATTRADE_API_KEY:
            if settings.FLATTRADE_TOTP_KEY:
                try:
                    from flattrade_bot.broker.auto_login import automated_flattrade_login
                    token = automated_flattrade_login(
                        user_id=settings.FLATTRADE_USER_ID,
                        password=settings.FLATTRADE_PASSWORD,
                        totp_key=settings.FLATTRADE_TOTP_KEY,
                        api_key=settings.FLATTRADE_API_KEY,
                        api_secret=settings.FLATTRADE_API_SECRET,
                        headless=True,
                    )
                except Exception as e:
                    logger.warning(f"Automated broker login attempt: {e}")

        if token:
            self.client.set_token(token)
            self.history.set_token(token)
            self.ws_feed.set_token(token)   # WS feed auths with the same token
            self._broker_status = "[bold green]LIVE AUTHENTICATED[/bold green]"
            logger.info("Flattrade live broker session authenticated.")
        else:
            self._broker_status = "[yellow]SIMULATION MODE[/yellow]"
            logger.warning("Running in simulation mode (no live broker token).")

        # Fetch initial spot quote
        if self.history.auth_token:
            try:
                q = await self.client.get_quotes(exchange="NSE", token="26000")
                if q.get("stat") == "Ok" and "lp" in q:
                    self.spot_price = float(q["lp"])
                    self.engine.set_spot_price(self.spot_price)
            except Exception as e:
                logger.warning(f"Initial spot quote failed: {e}")

            await self.ensure_contracts(force=True)

        # Start Discord retry loop (retries failed notifications every 30s)
        self.discord.start_retry_loop()

        asyncio.create_task(
            self.discord._post_embed({
                "title": "FLATTRADE LAST HOPE WINNER BOT ONLINE",
                "color": 0x2ECC71,
                "fields": [
                    {"name": "Strategy", "value": "Last Hope GPU Winner (FLAG/SUPER 1m OHLC)", "inline": True},
                    {"name": "Session", "value": "09:15 - 15:00 IST", "inline": True},
                    {"name": "Risk Geometry", "value": "ATR(10)x1.5 Breakeven at +40% move (§44 dynamic-strike champion)", "inline": True},
                    {"name": "Mode", "value": "LIVE ORDERS" if self.live_orders else "PAPER SIM", "inline": True},
                    {"name": "Nifty Spot", "value": f"Rs {self.spot_price:,.2f}" if self.spot_price else "--", "inline": True},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "Flattrade Last Hope Bot"},
            })
        )

    async def _warmup_contract(self, contract_state: OptionContractState, token: str, symbol: str, today_str: str, now: datetime):
        """Asynchronously warms up historical candles and S/R levels without blocking quote polling."""
        try:
            seed_rows = await self.history.fetch_historical_candles(
                token=token, exchange="NFO", interval="1", days_back=5
            )
            if not seed_rows:
                return
            by_day: Dict[str, List[Dict[str, Any]]] = {}
            for r in seed_rows:
                t_str = str(r.get("time", ""))
                d_part = t_str.split(" ")[0] if " " in t_str else t_str
                by_day.setdefault(d_part, []).append(r)

            def parse_d(d_str: str) -> datetime:
                try:
                    return datetime.strptime(d_str, "%d-%m-%Y")
                except Exception:
                    return datetime.min

            # Filter valid trading sessions (with >= 50 candles) and sort strictly chronologically
            valid_prev_days = sorted(
                [d for d in by_day.keys() if d != today_str and len(by_day[d]) >= 50],
                key=parse_d,
            )

            if valid_prev_days:
                last_trading_day = valid_prev_days[-1]

                # ── TRUE EOD OHLC via /EODChartData (official daily candle) ─────
                # 1m candles on illiquid strikes stop at the last trade, so the
                # last-bar close understates the official session close (what
                # TradingView/Upstox use for CPR/PDH/PDL). The daily candle is
                # authoritative. Falls back to the 1m-derived session H/L/C.
                yh = yl = yc = None
                try:
                    daily_rows = await self.history.fetch_daily_candles(
                        tradingsymbol=symbol, exchange="NFO", days_back=10
                    )
                    # daily_rows is chronological; find the last trading day present
                    for r in reversed(daily_rows):
                        d_part = str(r.get("time", "")).split(" ")[0]
                        if d_part == last_trading_day:
                            dh = float(r.get("high", 0.0))
                            dl = float(r.get("low", 0.0))
                            dc = float(r.get("close", 0.0))
                            if dh >= dl > 0 and dc > 0:
                                yh, yl, yc = dh, dl, dc
                            break
                    if yh is not None:
                        logger.info(f"EODChartData hit: {symbol} {last_trading_day} H={yh:.2f} L={yl:.2f} C={yc:.2f}")
                except Exception as e:
                    logger.debug(f"Daily candle fetch failed for {symbol}: {e}")

                # Fallback: 1m-derived session H/L/C (09:15-15:30 filter)
                if yh is None:
                    parsed_y_rows = []
                    for x in by_day[last_trading_day]:
                        try:
                            x_dt = datetime.strptime(str(x["time"]), "%d-%m-%Y %H:%M:%S")
                            x_m = x_dt.hour * 60 + x_dt.minute
                            if 555 <= x_m <= 930:
                                parsed_y_rows.append((x_dt, x))
                        except Exception:
                            pass

                    parsed_y_rows.sort(key=lambda item: item[0])
                    if parsed_y_rows:
                        yesterday_rows = [item[1] for item in parsed_y_rows]
                        yh = max(float(x.get("high", x.get("inth", 0.0))) for x in yesterday_rows)
                        yl = min(float(x.get("low", x.get("intl", 0.0))) for x in yesterday_rows if float(x.get("low", x.get("intl", 0.0))) > 0)
                        yc = float(yesterday_rows[-1].get("close", yesterday_rows[-1].get("intc", 0.0)))
                        logger.info(f"EODChartData miss — 1m fallback for {symbol}")

                if yh is not None:
                    contract_state.set_day_sr_levels(yh, yl, yc)
                    logger.info(f"Initialized Day S/R for {symbol} from {last_trading_day}: H={yh:.2f} L={yl:.2f} C={yc:.2f}")

            # SEEDED CONGRUENCE (§41/§42 champion): the validated strategy runs
            # indicators SEEDED from the prior day's final 300 1m bars (60 x 5m
            # bars >= S4(50,10) full warmup) with clock-aligned TF boundaries.
            # seed_1m_bars replays bars through ATR/EMA/VWAP/TF-trackers, so
            # seeding prior-day + today's completed bars reproduces the sweep's
            # exact state at this minute.
            prior_bars: List[Bar1m] = []
            if valid_prev_days:
                seed_day = valid_prev_days[-1]
                seed_rows = by_day.get(seed_day, [])
                for r in seed_rows:
                    try:
                        dt = datetime.strptime(str(r["time"]), "%d-%m-%Y %H:%M:%S")
                    except (TypeError, ValueError):
                        continue
                    m = dt.hour * 60 + dt.minute
                    if 555 <= m <= 930:
                        o = float(r.get("open", r.get("into", 0.0)))
                        h = float(r.get("high", r.get("inth", 0.0)))
                        l = float(r.get("low", r.get("intl", 0.0)))
                        c = float(r.get("close", r.get("intc", 0.0)))
                        v = float(r.get("volume", r.get("intv", r.get("v", 100.0))))
                        if h >= l > 0:
                            prior_bars.append(Bar1m(minute=m, open=o, high=h, low=l, close=c, timestamp=dt, volume=v))
                # Keep only the LAST 300 session bars of the prior day (the sweep's seed)
                prior_bars = prior_bars[-300:]

            # Today's completed candles (backtest-identical state reconstruction)
            today_bars: List[Bar1m] = []
            cur_m = minute_of(now)
            if today_str in by_day:
                for r in by_day[today_str]:
                    try:
                        dt = datetime.strptime(str(r["time"]), "%d-%m-%Y %H:%M:%S")
                    except (TypeError, ValueError):
                        continue
                    m = dt.hour * 60 + dt.minute
                    if m < cur_m:
                        o = float(r.get("open", r.get("into", 0.0)))
                        h = float(r.get("high", r.get("inth", 0.0)))
                        l = float(r.get("low", r.get("intl", 0.0)))
                        c = float(r.get("close", r.get("intc", 0.0)))
                        v = float(r.get("volume", r.get("intv", r.get("v", 100.0))))
                        if h >= l > 0:
                            today_bars.append(Bar1m(minute=m, open=o, high=h, low=l, close=c, timestamp=dt, volume=v))

            if prior_bars or today_bars:
                contract_state.seed_1m_bars(prior_bars, today_bars, session_date=today_str)
                logger.info(f"✅ Fully seeded {len(prior_bars)} prior + {len(today_bars)} today 1m bars for {symbol}")
        except Exception as e:
            logger.warning(f"Async warmup failed for {symbol}: {e}")

    async def ensure_contracts(self, force: bool = False):
        """Resolves 2nd ITM contracts + rollover watch pairs; HTTP work runs in background.

        The tick loop calls this every second — scrip token search must NEVER run
        inline (up to 6 sequential 10s HTTP calls would freeze quote polling).
        """
        if not self.history.auth_token or self.spot_price is None:
            return
        if not force and (time.time() - self._last_contract_check) < 3.0:
            return
        self._last_contract_check = time.time()

        desired = self.engine.desired_strikes(self.spot_price)
        pairs = [
            ("CE", desired["CE_SPEC"]),
            ("PE", desired["PE_SPEC"]),
            ("CE", desired["CE_WATCH_PLUS50"]),
            ("PE", desired["PE_WATCH_PLUS50"]),
            ("CE", desired["CE_WATCH_MINUS50"]),
            ("PE", desired["PE_WATCH_MINUS50"]),
        ]
        desired_keys = {f"{side}:{strike}" for side, strike in pairs}

        now = ist_now()
        today_str = now.strftime("%d-%m-%Y")

        # Prune stale contracts not in desired_keys or active position (pure CPU, safe inline)
        # PARK, DON'T DESTROY: when spot hovers a ±50 boundary the key flips
        # out and back within seconds. Deleting the object forced a full cold
        # re-warm on every flip (42 "Session reset" logs on Sep-4 09:xx).
        # Parked same-day state is restored on re-registration with its
        # seeded indicators intact (backtest parity preserved).
        if not hasattr(self, "_parked_contracts"):
            self._parked_contracts: Dict[str, Any] = {}
        for k in list(self.engine.contracts.keys()):
            if k not in desired_keys and k != self.active_position_key:
                cs = self.engine.contracts[k]
                if cs is not None:
                    if cs.token:
                        try:
                            self.ws_feed.unsubscribe("NFO", cs.token)
                        except Exception:
                            pass
                    if getattr(cs, "seed_complete", False) and getattr(cs, "session_date", None) == today_str:
                        self._parked_contracts[k] = cs
                del self.engine.contracts[k]
                self._last_ltp.pop(k, None)

        # HTTP resolution (token search + warmup) in background — never blocks the tick loop
        if self._contract_task is None or self._contract_task.done():
            self._contract_task = asyncio.create_task(
                self._resolve_and_warm_contracts(pairs, desired_keys, today_str, now)
            )

    async def _resolve_and_warm_contracts(self, pairs, desired_keys, today_str: str, now: datetime):
        """Background task: searches scrip tokens for missing contracts and launches warmup."""
        for side, strike in pairs:
            key = f"{side}:{strike}"
            if key in self.engine.contracts:
                continue
            try:
                scrip = await self.history.search_option_token(f"NIFTY {strike} {side}")
                if not scrip or not scrip.get("token"):
                    continue
                # Spot may have moved mid-resolution; register only if still desired
                if key not in desired_keys:
                    continue

                # PARKED-STATE RESTORE: if this key was parked earlier today
                # (±50 boundary flip), resurrect the SAME object — seeded
                # indicators, arming and session_date intact. No cold reset.
                parked = getattr(self, "_parked_contracts", {}).pop(key, None)
                if parked is not None and parked.session_date == today_str and parked.seed_complete:
                    self.engine.contracts[key] = parked
                    logger.info("♻️ Restored parked contract %s (seeded state intact)", key)
                    continue

                contract_state = self.engine.register_contract(
                    key=key,
                    symbol=scrip["tsym"],
                    token=scrip["token"],
                    side=side,
                    strike=strike,
                )
                # Per-day cold start (backtest parity): fresh indicators, then
                # warmup replays today's completed bars only. SAME-DAY
                # RE-REGISTRATION GUARD: when spot hovers on a ±50 boundary
                # the contract key flips in/out of the desired set every few
                # seconds — resetting here wiped the seeded indicators in a
                # loop ("Session reset" spam). Only cold-start ONCE per day
                # per contract: if this contract already holds today's
                # seeded state, keep it (indicator continuity = backtest
                # parity for dynamic-strike rollovers).
                already_warm_today = (
                    contract_state.session_date == today_str
                    and contract_state.seed_complete
                )
                if not already_warm_today:
                    contract_state.reset_session()

                # Launch async warmup in background without blocking main loop
                asyncio.create_task(
                    self._warmup_contract(contract_state, scrip["token"], scrip["tsym"], today_str, now)
                )
            except Exception as e:
                logger.warning(f"Contract resolution failed for {key}: {e}")

    def _has_position(self) -> bool:
        if self.paper_position is not None:
            return True
        return bool(self.executor and self.executor.position)

    def _is_funds_rejection(self, res: Dict[str, Any]) -> bool:
        """True when the broker rejected the order for margin/funds reasons."""
        blob = " ".join(str(res.get(k, "")) for k in ("reason", "emsg")).lower()
        return any(tok in blob for tok in (
            "margin", "fund", "insufficient", "limit", "available", "exposure",
            "rms", "blocked due to",
        ))

    async def _resolve_first_itm(self, sig: Dict[str, Any], now: datetime) -> Optional[Dict[str, Any]]:
        """Resolves the 1st ITM contract for a funds-rejected 2nd-ITM signal.

        CE: ATM-50, PE: ATM+50 (one strike closer to money = cheaper).
        Returns {'symbol','token','ltp'} or None. Used ONLY for that single
        fallback buy — the next trade defaults back to 2nd ITM.
        """
        try:
            side = sig["side"]
            if self.spot_price:
                atm = int(round(self.spot_price / 50.0) * 50)
            else:
                return None
            strike = atm - 50 if side == "CE" else atm + 50
            scrip = await self.history.search_option_token(f"NIFTY {strike} {side}")
            if not scrip or not scrip.get("token"):
                return None
            q = await self.client.get_quotes(exchange="NFO", token=scrip["token"])
            if not isinstance(q, dict) or q.get("stat") != "Ok" or "lp" not in q:
                return None
            ltp = float(q["lp"])
            if ltp <= 0:
                return None
            return {"symbol": scrip["tsym"], "token": scrip["token"], "ltp": ltp}
        except Exception as e:
            logger.warning("1st-ITM fallback resolution failed: %s", e)
            return None

    async def _try_enter(self, sig: Dict[str, Any]):
        # Re-entrancy guard: while one entry is confirming at the broker, ignore
        # all other signals (this is what caused duplicate live positions).
        if self._entering:
            logger.info("Entry skipped: another entry is being confirmed")
            return
        if self._has_position():
            return

        now = ist_now()
        cur_min = minute_of(now)
        can, reason = self.risk.can_open_trade(cur_min, 0)
        if not can:
            logger.info("Entry blocked by risk: %s", reason)
            return

        display = sig["symbol"]
        token = sig["token"]
        entry = sig["entry"]
        dist = sig["dist"]
        sl = sig["sl"]
        tp = sig["tp"]

        if self.live_orders and self.executor is not None:
            self._entering = True
            try:
                res = await self.executor.open_trade(
                    side=sig["side"],
                    order_symbol=display,
                    display_symbol=display,
                    token=token,
                    timeframe="1m",
                    signal=f"{sig['trigger']}_{sig['level']}",
                    entry_price=entry,
                    sl_points=dist,
                    tp_points=dist,
                    current_min=cur_min,
                    opened_at=now,
                )

                # FUNDS FALLBACK (per-buy only; default stays 2nd ITM):
                # if the 2nd ITM buy was rejected for margin/funds, retry the
                # SAME signal on the cheaper 1st ITM (CE: ATM-50, PE: ATM+50).
                if not res.get("accepted") and self._is_funds_rejection(res):
                    fb = await self._resolve_first_itm(sig, now)
                    if fb is not None:
                        logger.warning(
                            "Funds rejection on %s — retrying on 1st ITM %s @ %.2f",
                            display, fb["symbol"], fb["ltp"],
                        )
                        res = await self.executor.open_trade(
                            side=sig["side"],
                            order_symbol=fb["symbol"],
                            display_symbol=fb["symbol"],
                            token=fb["token"],
                            timeframe="1m",
                            signal=f"{sig['trigger']}_{sig['level']}_1ITM",
                            entry_price=fb["ltp"],
                            sl_points=dist,
                            tp_points=dist,
                            current_min=cur_min,
                            opened_at=now,
                        )
            finally:
                self._entering = False
            if not res.get("accepted"):
                logger.warning("Live entry rejected: %s", res.get("reason"))
                return
            fill = float(res["position"]["entry"])
            sl = float(res["position"]["sl"])
            tp = float(res["position"]["target"])
            # Update active trade in engine
            self.engine.on_trade_opened({
                **sig,
                "entry": fill,
                "sl": sl,
                "tp": tp,
                "be_done": False,
            })
        else:
            self.paper_position = {
                "side": sig["side"],
                "symbol": display,
                "order_symbol": display,
                "token": token,
                "quantity": settings.LOT_SIZE,
                "entry": entry,
                "sl": sl,
                "target": tp,
                "dist": dist,
                "be_trigger_px": sig["be_trigger_px"],
                "be_hardened_sl": sig["be_hardened_sl"],
                "be_done": False,
                "opened_at": now,
                "signal": f"{sig['trigger']}_{sig['level']}",
            }
            self.engine.on_trade_opened(self.paper_position)
            fill = entry
            sl = float(sl)
            tp = float(tp)

        # POSITION KEY = the contract ACTUALLY held, not the signal's original
        # strike. The 2026-09-04 10:10 incident: funds-fallback bought the
        # 1st-ITM (P24000) but the key stayed PE:24050 -> exit watchdog read
        # P24050's LTP (~134) for a P24000 position (~110), "TARGET" fired on
        # the wrong instrument and the exit sell priced 19 pts above market —
        # unfillable, retried every second as a stacked naked short.
        held_symbol = str(res.get("position", {}).get("order_symbol", display)) if (self.live_orders and self.executor is not None) else display
        held_strike = self._strike_from_symbol(held_symbol, sig["strike"], sig["side"])
        self.active_position_key = f"{sig['side']}:{held_strike}"
        logger.info(f"ENTRY {display} @ {fill:.2f} SL={sl:.2f} TP={tp:.2f} BE_trig={sig['be_trigger_px']:.2f}")

        # Fire-and-forget Discord alert — must never block the tick loop (10s webhook timeout)
        asyncio.create_task(
            self.discord.send_trade_alert(
                strategy=STRATEGY_LABEL,
                direction="LONG",
                symbol=display,
                entry_price=fill,
                sl_price=sl,
                tp_price=tp,
                notes=f"{sig['trigger']} trigger | SR={sig['level']} | dist={dist:.2f} | BE at {sig['be_trigger_px']:.2f}",
                lot_size=settings.LOT_SIZE,
                mode="LIVE" if self.live_orders else "PAPER",
            )
        )

    def _strike_from_symbol(self, symbol: str, default: int, side: str) -> int:
        """Extracts the numeric strike from an option symbol like
        NIFTY08SEP26P24000; falls back to the provided default."""
        import re
        m = re.search(r"([CP])(\d{5})\s*$", str(symbol))
        if m:
            try:
                return int(m.group(2))
            except ValueError:
                pass
        return default

    async def _execute_close(self, ltp: float, now: datetime, reason: str):
        """Background: performs the broker close + fill confirmation (may take seconds).
        The tick loop keeps running while this executes."""
        try:
            res = await self.executor.close_position(ltp, now, reason)
            if res.get("accepted"):
                await self._record_close(res.get("trade", {}))
                logger.info(f"✅ Position Closed on Flattrade: {res.get('trade', {}).get('reason')}")
            else:
                logger.warning("Background close not accepted: %s — will retrigger next tick", res.get("reason"))
        except Exception as e:
            logger.error("Background close failed: %s — will retrigger next tick", e, exc_info=True)

    async def _record_close(self, trade: Dict[str, Any]):
        self.trades_today.append(trade)
        # Persist across restarts (append-only JSONL): the 2026-09-03 restart
        # showed 0 trades / Rs +0.00 on the dashboard while the broker book
        # held a realized -715 — restarts must never erase the day's record.
        try:
            trade = dict(trade)
            trade["_recorded_at"] = ist_now().isoformat()
            os.makedirs("logs", exist_ok=True)
            with open(f"logs/trades_{ist_now().strftime('%Y-%m-%d')}.jsonl", "a") as f:
                f.write(json.dumps(trade, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Trade persistence failed: {e}")
        if float(trade.get("pts", 0.0)) > 0:
            self._wins_today += 1
        self.engine.on_trade_closed()
        self.active_position_key = None
        self.paper_position = None

    def _load_today_trades(self):
        """Reloads this session's already-recorded trades from the JSONL."""
        try:
            path = f"logs/trades_{ist_now().strftime('%Y-%m-%d')}.jsonl"
            if not os.path.exists(path):
                return
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.trades_today.append(json.loads(line))
                    except ValueError:
                        continue
            self._wins_today = sum(1 for t in self.trades_today if float(t.get("pts", 0.0)) > 0)
            if self.trades_today:
                logger.info(f"📂 Restored {len(self.trades_today)} earlier trades from {path}")
        except Exception as e:
            logger.warning(f"Trade restore failed: {e}")

    async def _manage_exit(self):
        """Exit watchdog — runs EVERY tick, must never silently skip.

        Never returns early on a missing active_position_key: if the broker
        executor holds a position, we exit it using whatever LTP we can find.
        """
        if not self._has_position():
            return

        now = ist_now()

        # Live Broker Position Exit
        if self.live_orders and self.executor is not None and self.executor.position:
            pos = self.executor.position
            # Resolve LTP: prefer the position's own key; fall back to scanning
            # by token so an orphaned/desynced position can never dodge the exit check.
            ltp = self._last_ltp.get(self.active_position_key)
            if ltp is None:
                key_by_token = next(
                    (k for k, cs in self.engine.contracts.items()
                     if cs and str(cs.token) == str(pos.get("token", ""))),
                    None,
                )
                ltp = self._last_ltp.get(key_by_token) if key_by_token else None
            if ltp is None:
                ltp = float(pos.get("ltp", 0.0) or 0.0)
            if ltp <= 0:
                logger.warning("Exit check skipped: no LTP available for %s", pos.get("order_symbol"))
                return

            # Sync SL if Breakeven triggered (engine is authoritative for BE)
            if self.engine.active_trade:
                if self.engine.active_trade.get("be_done") and not pos.get("be_notified"):
                    pos["be_notified"] = True
                    pos["sl"] = self.engine.active_trade["sl"]
                    asyncio.create_task(
                        self.discord.notify_breakeven_locked(
                            symbol=pos["symbol"],
                            entry=float(pos["entry"]),
                            new_sl=float(self.engine.active_trade["sl"]),
                            ltp=ltp,
                        )
                    )

            # SL/TP/EOD detection (pure CPU — fast). Exit EXECUTION (broker order +
            # up to 12x0.3s order-book polls) runs as a guarded background task so
            # the tick loop NEVER blocks on fill confirmation (the 10s freeze bug).
            res = await self.executor.check_exit(ltp, now, dry_run=True)
            if res.get("exit_reason"):
                reason = res["exit_reason"]
                if self._exit_task is None or self._exit_task.done():
                    logger.info("EXIT triggered (%s) — dispatching broker close in background", reason)
                    self._exit_task = asyncio.create_task(
                        self._execute_close(ltp, now, reason)
                    )
                else:
                    logger.warning("EXIT signal (%s) while close already in flight — skipped", reason)
            return

        # Paper Position Exit
        pos = self.paper_position
        if pos is None:
            return
        ltp = self._last_ltp.get(self.active_position_key)
        if ltp is None:
            return

        # Breakeven Stop Check
        if not pos.get("be_done") and ltp >= pos["be_trigger_px"]:
            pos["be_done"] = True
            pos["sl"] = pos["be_hardened_sl"]
            logger.info(f"Paper Breakeven Triggered on {pos['symbol']}: SL moved to {pos['sl']:.2f}")
            asyncio.create_task(
                self.discord.notify_breakeven_locked(
                    symbol=pos["symbol"],
                    entry=float(pos["entry"]),
                    new_sl=float(pos["sl"]),
                    ltp=ltp,
                )
            )

        reason = None
        if ltp <= pos["sl"]:       # SL priority over TP (backtest rule)
            reason = "STOP_LOSS"
        elif ltp >= pos["target"]:
            reason = "TARGET"
        elif minute_of(now) >= SESSION_END_MIN:
            reason = "EOD"

        if not reason:
            return

        pts = round(ltp - pos["entry"], 2)
        pnl_rs = round(pts * pos["quantity"], 2)
        duration_min = max(0, int((now - pos["opened_at"]).total_seconds() // 60))
        self.risk.record_trade_result(pnl_rs)
        trade_info = {
            **pos,
            "exit": ltp,
            "pts": pts,
            "rs": pnl_rs,
            "duration_min": duration_min,
            "reason": reason,
        }
        await self._record_close(trade_info)
        asyncio.create_task(self.discord.notify_trade_close(trade_info))

    async def recover_open_positions(self):
        """On startup, squares off orphaned broker positions from a previous crash."""
        if not self.live_orders or not self.client.auth_token:
            return
        try:
            pos_res = await self.client.get_positions()
            positions = []
            if isinstance(pos_res, list):
                positions = pos_res
            elif isinstance(pos_res, dict) and pos_res.get("stat") == "Ok":
                positions = pos_res.get("positions", pos_res) if "positions" in pos_res else []
            for p in positions:
                if not isinstance(p, dict):
                    continue
                tsym = p.get("tsym", "")
                try:
                    netqty = int(float(p.get("netqty", 0) or 0))
                except (TypeError, ValueError):
                    continue
                if "NIFTY" in tsym and netqty != 0:
                    side = "SELL" if netqty > 0 else "BUY"
                    qty = abs(netqty)
                    ltp = float(p.get("lp", p.get("urmtom", 100.0)))
                    logger.warning(f"🔄 RECOVERING orphaned position {tsym} qty={netqty}")
                    res = await self.client.place_market_order(
                        symbol=tsym, side=side, quantity=qty,
                        ltp=max(ltp, 1.0), product="MIS", slippage_buffer=5.0,
                        force_mkt=True,  # rescue path: execution certainty over price
                    )
                    if res.get("stat") == "Ok":
                        logger.info(f"✅ Orphaned position {tsym} squared off successfully.")
                    else:
                        logger.error(f"❌ Failed to square off {tsym}: {res.get('emsg')}")
        except Exception as e:
            logger.error(f"Position recovery check failed: {e}")

    async def _force_eod_square_off(self):
        """Safety net at 15:15 IST: flatten anything still open."""
        if self.live_orders and self.executor is not None and self.executor.position:
            pos = self.executor.position
            try:
                res = await self.client.place_market_order(
                    symbol=pos["order_symbol"], side="SELL",
                    quantity=pos["quantity"], ltp=max(self._last_ltp.get(self.active_position_key, pos["entry"]), 1.0),
                    product="MIS", slippage_buffer=5.0,
                    force_mkt=True,  # EOD flatten: market order, no price-band rejections
                )
                logger.info(f"⏰ EOD square-off order for {pos['order_symbol']}: {res.get('stat')}")
            except Exception as e:
                logger.error(f"EOD square-off error: {e}")
            await self._record_close({**pos, "exit": self._last_ltp.get(self.active_position_key, pos["entry"]),
                                      "pts": 0.0, "rs": 0.0, "duration_min": 0, "reason": "EOD_FORCE"})
        elif self.paper_position is not None:
            pos = self.paper_position
            ltp = self._last_ltp.get(self.active_position_key, pos["entry"])
            pts = round(ltp - pos["entry"], 2)
            pnl_rs = round(pts * pos["quantity"], 2)
            self.risk.record_trade_result(pnl_rs)
            await self._record_close({
                **pos, "exit": ltp, "pts": pts, "rs": pnl_rs,
                "duration_min": 0, "reason": "EOD_FORCE",
            })
        await self.recover_open_positions()

    async def _renew_token_if_expired(self, response: Any) -> bool:
        if not is_session_expired_response(response):
            return False
        now_ts = time.time()
        if now_ts - getattr(self, "_last_token_renew", 0.0) < 60:
            return True
        logger.warning("Session expired. Renewing token via REST API...")
        try:
            from flattrade_bot.broker.auto_login import automated_flattrade_login_rest
            loop = asyncio.get_running_loop()
            new_token = await loop.run_in_executor(
                None,
                lambda: automated_flattrade_login_rest(
                    user_id=settings.FLATTRADE_USER_ID,
                    password=settings.FLATTRADE_PASSWORD,
                    totp_key=settings.FLATTRADE_TOTP_KEY,
                    api_key=settings.FLATTRADE_API_KEY,
                    api_secret=settings.FLATTRADE_API_SECRET,
                ),
            )
            if new_token:
                self.client.set_token(new_token)
                self.history.set_token(new_token)
                self.ws_feed.set_token(new_token)  # WS reconnects with the new token
                self._last_token_renew = time.time()
                logger.info("Token renewed successfully.")
                return True
            else:
                logger.error("Token renewal returned empty — will retry in 60s")
        except Exception as e:
            logger.error(f"Token renewal failed: {e} — will retry in 60s")
        return True

    def render_dashboard(self):
        """Builds the full dashboard as one Rich renderable (banner + tables +
        active-trade cockpit). Consumed by Live(screen=True, refresh 1/s) in
        run() — Rich diffs segments and repaints ONLY changed cells, and the
        alternate buffer redraws in place (the proven Aug-25 flicker fix)."""
        now = ist_now()
        time_str = now.strftime("%H:%M:%S")
        sess_active = SESSION_START_MIN <= minute_of(now) < SESSION_END_MIN
        mode = "LIVE BROKER" if self.live_orders else "PAPER SIM"
        net_rs = sum(float(t.get("rs", 0.0)) for t in self.trades_today)
        net_pts = sum(float(t.get("pts", 0.0)) for t in self.trades_today)
        atm = int(round(self.spot_price / 50.0) * 50) if self.spot_price else 0
        total = len(self.trades_today)
        wr = (self._wins_today / total * 100.0) if total > 0 else 0.0
        sess = "ACTIVE" if sess_active else "CLOSED"
        pnl_style = "bold green" if net_rs >= 0 else "bold red"

        header = Text.from_markup(
            f" [bold bright_yellow]LAST HOPE GPU WINNER[/bold bright_yellow] "
            f"| [bold white]2nd ITM (dynamic) · Multi-TF Stoch · S/R Bounce · ATRx1.5 + BE@40%[/bold white] "
            f"| [cyan]{time_str} IST[/cyan]"
        )
        banner = Panel(header, box=box.ROUNDED, style="bright_blue", padding=(0, 1))

        sys_table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
        for col, style in (("MODE", "bold white"), ("SPOT (ATM)", "bold yellow"),
                           ("2nd ITM (CE/PE)", "bold cyan"), ("EXPIRY", "bold magenta"),
                           ("SESSION", "bold green"),
                           ("TRADES (W/L)", "bold white"), ("NET P&L", "bold white")):
            sys_table.add_column(col, style=style, justify="center", no_wrap=True)
        # Weekly expiry shared by all tracked contracts (from the first symbol)
        first_cs = next(iter(self.engine.contracts.values()), None)
        exp_str = expiry_info(first_cs.symbol, now) if first_cs else "--"
        sys_table.add_row(
            f"[bold {'red' if self.live_orders else 'blue'}]{mode}[/bold {'red' if self.live_orders else 'blue'}]",
            f"Rs {self.spot_price:,.1f} ({atm})" if self.spot_price else "--",
            f"Rs {atm-100} CE / Rs {atm+100} PE",
            exp_str or "--",
            "[bold green]ACTIVE[/bold green]" if sess_active else "[yellow]CLOSED[/yellow]",
            f"{total} ({self._wins_today}W/{total - self._wins_today}L | {wr:.0f}%)",
            f"[{pnl_style}]Rs {net_rs:+,.2f} ({net_pts:+.1f}p)[/{pnl_style}]",
        )

        mon_table = Table(
            title="[bold cyan]STRATEGY SETUP RADAR & ARMING MATRIX (S1 12,3 · S3 40,4 · S4 50,10 · SR Suite)[/bold cyan]",
            box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
        for col, style, j in (("STRIKE", "bold white", "left"), ("ARMING", "bold yellow", "center"),
                              ("1m S1/S3/S4", "bold white", "center"), ("2m/3m/5m S4", "bold white", "center"),
                              ("ACTIVE TF SETUPS", "bold magenta", "left"),
                              ("S/R LEVEL (PROX)", "bold cyan", "left"), ("LTP", "bold yellow", "right")):
            mon_table.add_column(col, style=style, justify=j, no_wrap=True)

        for key, cs in self.engine.contracts.items():
            s1 = cs.tf_trackers[1].last_s1
            s3 = cs.tf_trackers[1].last_s3
            s4 = cs.tf_trackers[1].last_s4
            bar_count = len(cs.bars)
            ltp = self._last_ltp.get(key, 0.0)
            f0 = lambda v: f"{v:.0f}" if isinstance(v, (int, float)) and v else "--"

            arm = "[dim]FLAT[/dim]"
            if cs.flag_armed or cs.super_armed:
                arm_age = max(0, bar_count - max(cs.flag_arm_bar, cs.super_arm_bar))
                arm_rem = max(0, ARM_WINDOW - arm_age)
                arm = f"[bold green]ARMED ({arm_rem}b)[/bold green]"

            tf_signals = []
            for tf, trk in cs.tf_trackers.items():
                t_s1, t_s3, t_s4 = trk.last_s1, trk.last_s3, trk.last_s4
                if t_s4 is not None and t_s1 is not None:
                    if t_s4 >= M6_S4 and t_s1 < M6_S1:
                        tf_signals.append(f"[bold green]FLAG {tf}m[/bold green]")
                    elif t_s4 >= 72.0:
                        tf_signals.append(f"[yellow]Flag {tf}m[/yellow]")
                if t_s1 is not None and t_s3 is not None and t_s4 is not None:
                    if t_s1 < SUPER_THRESH and t_s3 < SUPER_THRESH and t_s4 < SUPER_THRESH:
                        is_rise = t_s1 > (trk.prev_s1 or 0)
                        tf_signals.append(f"[bold green]SUPER {tf}m{'+' if is_rise else ''}[/bold green]")
            setup = ", ".join(tf_signals) if tf_signals else "[dim]Scanning...[/dim]"

            prox = "--"
            # §43: proximity display vs the GATE level (EMA20) + display suite
            if ltp > 0 and cs.ema20.value:
                gate_lv = {"EMA20": cs.ema20.value}
                display_sr = dict(gate_lv)
                if cs.display_levels:
                    display_sr.update(cs.display_levels)
                if cs.vwap.value: display_sr["VWAP"] = cs.vwap.value
                closest = min(gate_lv.items(), key=lambda x: abs(ltp - x[1]))
                diff = ltp - closest[1]
                # nearest overall level (any family) for context
                nearest_all = min(display_sr.items(), key=lambda x: abs(ltp - x[1]))
                if abs(diff) <= 0.5:
                    prox = f"{closest[0]} [bold green](TOUCH {diff:+.1f})[/bold green]"
                elif nearest_all[0] != "EMA20" and abs(ltp - nearest_all[1]) <= 0.5:
                    prox = f"[dim]{nearest_all[0]} ({ltp - nearest_all[1]:+.1f}p) | EMA20 {diff:+.1f}p[/dim]"
                else:
                    prox = f"{closest[0]} ({diff:+.1f}p)"

            side_color = "bold green" if cs.side == "CE" else "bold red"
            exp_tok = expiry_info(cs.symbol, now)
            mon_table.add_row(
                f"[{side_color}]{cs.strike} {cs.side}[/{side_color}][dim] {exp_tok.split(' ')[0] if exp_tok else ''}[/dim]",
                arm,
                f"{f0(s1)}/{f0(s3)}/{f0(s4)}",
                f"{f0(cs.tf_trackers[2].last_s4)}/{f0(cs.tf_trackers[3].last_s4)}/{f0(cs.tf_trackers[5].last_s4)}",
                setup, prox,
                f"Rs {ltp:.2f}" if ltp > 0 else "--",
            )

        sr_table = Table(
            title="[bold cyan]S/R LEVELS (TradingView verify)[/bold cyan]",
            box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
        for col, style, j in (("STRIKE", "bold white", "left"), ("LTP", "bold yellow", "right"),
                              ("PDH/PDL", "bold white", "center"), ("CPR (BC/P/T)", "bold white", "center"),
                              ("EMA20/200", "bold white", "center"), ("VWAP", "bold white", "center"),
                              ("ATR DIST", "bold white", "center")):
            sr_table.add_column(col, style=style, justify=j, no_wrap=True)

        for key, cs in self.engine.contracts.items():
            ltp = self._last_ltp.get(key, 0.0)
            f2 = lambda v: f"{v:.0f}" if isinstance(v, (int, float)) and v else "--"
            atr = min(max(cs.latest_atr * ATR_MULT, 2.0), TP_PTS_CAP)
            side_color = "bold green" if cs.side == "CE" else "bold red"
            exp_tok = expiry_info(cs.symbol, now)
            sr_table.add_row(
                f"[{side_color}]{cs.strike} {cs.side}[/{side_color}][dim] {exp_tok.split(' ')[0] if exp_tok else ''}[/dim]",
                f"Rs {ltp:.0f}" if ltp > 0 else "--",
                f"{f2(cs.display_levels.get('PDH'))}/{f2(cs.display_levels.get('PDL'))}",
                f"{f2(cs.display_levels.get('CPR_BC'))}/{f2(cs.display_levels.get('CPR_Pivot'))}/{f2(cs.display_levels.get('CPR_TC'))}",
                f"{f2(cs.ema20.value)}/{f2(cs.ema200.value)}",
                f"{f2(cs.vwap.value)}",
                f"+/-{atr:.1f}",
            )

        pos_lines = []
        pos_data = None
        if self.live_orders and self.executor is not None and self.executor.position:
            p = self.executor.position
            be_trig = self.engine.active_trade.get("be_trigger_px", 0.0) if self.engine.active_trade else 0.0
            be_done = self.engine.active_trade.get("be_done", False) if self.engine.active_trade else False
            pos_data = {**p, "be_trigger_px": be_trig, "be_done": be_done}
        elif self.paper_position is not None:
            pos_data = self.paper_position
        if pos_data is not None:
            ltp = float(self._last_ltp.get(self.active_position_key, pos_data["entry"]))
            pts = ltp - float(pos_data["entry"])
            c = "green" if pts >= 0 else "red"
            be_status = "[bold green]LOCKED (+1.0pt BE)[/bold green]" if pos_data.get("be_done") \
                else f"[yellow]trig Rs {float(pos_data.get('be_trigger_px', 0)):.2f}[/yellow]"
            pos_table = Table(
                title=f"[bold green]ACTIVE TRADE — {pos_data['symbol']} | P&L [{c}]{pts:+.2f} pts[/{c}] [/bold green]",
                box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
            for col in ("ENTRY", "LTP", "SL", "TP", "BE STATUS", "SIGNAL"):
                pos_table.add_column(col, justify="center")
            pos_table.add_row(
                f"Rs {float(pos_data['entry']):.2f}", f"Rs {ltp:.2f}",
                f"Rs {float(pos_data['sl']):.2f}", f"Rs {float(pos_data.get('target', pos_data.get('tp', 0))):.2f}",
                be_status, str(pos_data.get("signal", "--")),
            )
            pos_lines = [pos_table]

        group = Group(banner, sys_table, mon_table, sr_table, *pos_lines)
        return group


    async def _on_day_rollover(self, now: datetime):
        """Resets all daily state and forces full re-warmup when the calendar date changes (IST)."""
        logger.warning("🌅 NEW TRADING DAY %s — resetting daily state and re-seeding S/R levels...", now.date())

        # Preserve open-position bookkeeping, reset everything else
        self.trades_today = []
        self._wins_today = 0
        self._eod_done = False
        self.risk.reset_day()

        # Parked contracts are same-day only — never carry across midnight
        if hasattr(self, "_parked_contracts"):
            self._parked_contracts.clear()

        # Drop ALL contracts so ensure_contracts() re-resolves + re-warms with fresh S/R
        # (active position contract is preserved via active_position_key)
        for k in list(self.engine.contracts.keys()):
            if k != self.active_position_key:
                del self.engine.contracts[k]
                self._last_ltp.pop(k, None)
        # Per-day cold start on any surviving contract (backtest parity)
        for cs in self.engine.contracts.values():
            cs.reset_session()

        # Reset token-renew cooldown so a stale overnight token refreshes immediately
        self._last_token_renew = 0.0

        # Force immediate contract resolution + warmup on next tick
        self._last_contract_check = 0.0
        await self.ensure_contracts(force=True)

        asyncio.create_task(
            self.discord._post_embed({
                "title": "🌅 NEW TRADING DAY — BOT RESET & RE-SEEDED",
                "color": 0x3498DB,
                "fields": [
                    {"name": "Date", "value": str(now.date()), "inline": True},
                    {"name": "Mode", "value": "LIVE ORDERS" if self.live_orders else "PAPER SIM", "inline": True},
                    {"name": "Risk", "value": "Daily counters reset (4-loss block, shutdown cleared)", "inline": False},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "Flattrade Last Hope Bot"},
            })
        )

    async def _main_loop_body(self) -> float:
        """Execute one iteration of the trading loop. Returns elapsed seconds."""
        started = time.time()
        now = ist_now()

        # 0. Day rollover — reset daily state + re-seed S/R for the new trading day
        today_id = now.date()
        if today_id != self._current_day:
            if self._current_day is not None:
                await self._on_day_rollover(now)
            self._current_day = today_id

        # 1. Dynamic Contract Rollover Watch (runs async warmup in background)
        await self.ensure_contracts()

        # 2. TICK ACQUISITION — WebSocket-first with surgical REST fallback.
        # The WS feed pushes every tick (no polling, no rate limit). Any
        # instrument whose WS tick is stale (>3s old, e.g. after a WS
        # reconnect) is refreshed via a single REST GetQuotes batch — only
        # the stale ones, so steady-state REST load is ZERO.
        poll_keys = sorted(self.engine.contracts.keys())
        if self.active_position_key and self.active_position_key not in poll_keys:
            poll_keys.append(self.active_position_key)

        STALE_SEC = 3.0
        spot_ltp = self.ws_feed.last_ltp("NSE", "26000")
        if self.ws_feed.connected and spot_ltp is not None and self.ws_feed.age_seconds("NSE", "26000") <= STALE_SEC:
            self.spot_price = spot_ltp
            self.engine.set_spot_price(self.spot_price)
            self._broker_status = "LIVE CONNECTED (WS)"
        elif self.ws_feed.connected and spot_ltp is not None:
            pass  # stale spot: refreshed in the REST batch below

        # Which option contracts need a REST refresh? (not yet WS-subscribed,
        # or WS ticks gone stale)
        ws_fresh_keys: set = set()
        rest_keys: List[str] = []
        for key in poll_keys:
            cs = self.engine.contracts.get(key)
            if not cs:
                continue
            # Subscribe unconditionally (idempotent, broker pushes on change).
            # The old code only subscribed when a tick was ALREADY fresh —
            # circular: never subscribed -> never fresh -> REST forever ->
            # throttled -> stuck dashboard prices.
            if self.ws_feed.connected:
                self.ws_feed.subscribe("NFO", cs.token)
            ltp_ws = self.ws_feed.last_ltp("NFO", cs.token)
            if (self.ws_feed.connected and ltp_ws is not None
                    and self.ws_feed.age_seconds("NFO", cs.token) <= STALE_SEC):
                ws_fresh_keys.add(key)
            else:
                rest_keys.append(key)

        # Also make sure the spot is subscribed for next time
        if self.ws_feed.connected:
            self.ws_feed.subscribe("NSE", "26000")

        # 1.5 — SIGNAL GATE: only the CURRENT 2nd-ITM pair (CE_SPEC/PE_SPEC)
        # may open trades. Watch pairs (+/-50 rollover) are warm-only — the
        # 11:20 Sep-2 incident fired 6 entries in 1 second across ALL
        # registered strikes. Backtest semantics: trade only the active pair.
        spec_keys: set = set()
        try:
            if self.spot_price:
                _d = self.engine.desired_strikes(self.spot_price)
                spec_keys = {f"CE:{_d['CE_SPEC']}", f"PE:{_d['PE_SPEC']}"}
        except Exception:
            spec_keys = set()

        # Process WS-fresh option ticks first (zero REST cost)
        for key in ws_fresh_keys:
            cs = self.engine.contracts.get(key)
            if not cs:
                continue
            opt_ltp = self.ws_feed.last_ltp("NFO", cs.token)
            if opt_ltp and opt_ltp > 0:
                self._last_ltp[key] = opt_ltp
                sig = self.engine.push_tick(key, opt_ltp, now)
                if sig and not self._has_position() and key in spec_keys:
                    await self._try_enter(sig)

        # REST fallback: ONLY stale/unsubscribed instruments (+ spot if stale)
        need_spot = not (self.ws_feed.connected and spot_ltp is not None
                         and self.ws_feed.age_seconds("NSE", "26000") <= STALE_SEC)
        tasks: List[Any] = []
        task_labels: List[str] = []
        if need_spot:
            tasks.append(self.client.get_quotes(exchange="NSE", token="26000"))
            task_labels.append("spot")
        for key in rest_keys:
            cs = self.engine.contracts.get(key)
            if cs:
                tasks.append(self.client.get_quotes(exchange="NFO", token=cs.token))
                task_labels.append(key)
        if tasks:
            try:
                results: List[Any] = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                results = [None] * len(tasks)
            for label, result in zip(task_labels, results):
                if isinstance(result, Exception):
                    logger.warning("Quote poll exception for %s: %s", label, result)
                    continue
                if not isinstance(result, dict):
                    continue
                if is_session_expired_response(result):
                    await self._renew_token_if_expired(result)
                    continue
                if result.get("stat") != "Ok" or "lp" not in result:
                    continue
                try:
                    ltp_val = float(result["lp"])
                except (ValueError, TypeError):
                    continue
                if ltp_val <= 0:
                    continue
                if label == "spot":
                    self.spot_price = ltp_val
                    self.engine.set_spot_price(self.spot_price)
                    self._broker_status = "LIVE CONNECTED (REST)"
                else:
                    self._last_ltp[label] = ltp_val
                    sig = self.engine.push_tick(label, ltp_val, now)
                    if sig and not self._has_position() and label in spec_keys:
                        await self._try_enter(sig)

        # 3. Check Exits (SL, TP, Breakeven, EOD)
        await self._manage_exit()

        # 4. EOD Safety Square-Off at 15:15 IST
        cur_min = minute_of(now)
        if cur_min >= 915 and not self._eod_done:
            self._eod_done = True
            logger.warning("15:15 IST - triggering EOD safety square-off.")
            await self._force_eod_square_off()

        touch_runtime_record(
            extra={
                "strategy_name": STRATEGY_LABEL,
                "spot_price": self.spot_price or 0.0,
                "active_position": self.active_position_key if self._has_position() else None,
                "trades_count": len(self.trades_today),
                "wins": self._wins_today,
                "net_rs": round(sum(float(t.get("rs", 0.0)) for t in self.trades_today), 2),
                "broker_status": self._broker_status,
            },
            live_orders=self.live_orders,
        )
        return time.time() - started

    def _log_status_line(self):
        """One-line status summary for headless (systemd) mode."""
        now = ist_now()
        total = len(self.trades_today)
        wr = (self._wins_today / total * 100) if total else 0
        net = sum(float(t.get("rs", 0.0)) for t in self.trades_today)
        pos = "FLAT"
        if self._has_position():
            sym = self.active_position_key or "?"
            ltp = self._last_ltp.get(self.active_position_key, 0)
            pos = f"IN {sym} LTP={ltp:.2f}"
        logger.info(
            "[STATUS] %s | spot=%.1f | %s | trades=%d (%dW/%dL %.0f%%) | net=Rs %+.2f",
            now.strftime("%H:%M:%S"),
            self.spot_price or 0,
            pos,
            total,
            self._wins_today,
            total - self._wins_today,
            wr,
            net,
        )
        # Log S/R levels for TradingView verification (display suite + live gate level)
        for key, cs in self.engine.contracts.items():
            if not cs.display_levels:
                continue
            ltp = self._last_ltp.get(key, 0)
            sr_parts = []
            for name, price in cs.display_levels.items():
                sr_parts.append(f"{name}={price:.2f}")
            if cs.ema20.value:
                sr_parts.append(f"EMA20={cs.ema20.value:.2f}")
            if cs.ema200.value:
                sr_parts.append(f"EMA200={cs.ema200.value:.2f}")
            if cs.vwap.value:
                sr_parts.append(f"VWAP={cs.vwap.value:.2f}")
            atr_pts = min(max(cs.latest_atr * ATR_MULT, 2.0), TP_PTS_CAP)
            logger.info(
                "[SR] %s %s | LTP=%.2f | ATR=%.2f dist=%.1f | %s",
                cs.side,
                cs.strike,
                ltp,
                cs.latest_atr,
                atr_pts,
                " | ".join(sr_parts),
            )

    async def run(self):
        await self.initialize()
        self._load_today_trades()   # restore the day's record across restarts
        await self.recover_open_positions()

        has_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

        if has_tty:
            # THE PROVEN AUG-25 FLICKER FIX (Combined Supreme era): Rich Live
            # in alternate-screen mode at 1 fps. Live diffs segments and
            # repaints ONLY changed cells (minimal SSH bytes vs full-frame
            # blits), screen=True gives htop-style in-place redraw on a
            # dedicated buffer, and 1 fps keeps redraws far below SSH
            # round-trip time. The earlier blank-screen problem was the
            # tee-pipe/StreamHandler setup — logging is file-only now, and
            # the bot runs on a clean tmux PTY.
            from rich.live import Live
            with Live(
                self.render_dashboard(),
                console=_live_console,
                screen=True,
                refresh_per_second=1,
            ) as live:
                loop_count = 0
                while True:
                    try:
                        elapsed = await self._main_loop_body()
                        loop_count += 1
                        if loop_count % 2 == 0:
                            live.update(self.render_dashboard())
                        await asyncio.sleep(max(0.0, 1.0 - elapsed))
                    except Exception as e:
                        logger.error(f"Error in main loop: {e}", exc_info=True)
                        await asyncio.sleep(2.0)
        else:
            # Headless mode (systemd): log status every 10 seconds
            loop_count = 0
            while True:
                try:
                    elapsed = await self._main_loop_body()
                    loop_count += 1
                    if loop_count % 10 == 0:
                        self._log_status_line()
                    await asyncio.sleep(max(0.0, 1.0 - elapsed))
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    await asyncio.sleep(2.0)


class ProcessSingletonLock:
    """Guarantees only ONE instance of the bot can ever run at a time."""

    def __init__(self, lockfile_path: Path):
        self.lockfile_path = lockfile_path
        self.fp = None

    def acquire(self) -> bool:
        try:
            self.lockfile_path.parent.mkdir(parents=True, exist_ok=True)
            self.fp = open(self.lockfile_path, "a+")
            if sys.platform == "win32":
                import msvcrt
                try:
                    msvcrt.locking(self.fp.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    return False
            else:
                import fcntl
                try:
                    fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (OSError, IOError):
                    return False
            self.fp.seek(0)
            self.fp.truncate()
            self.fp.write(str(os.getpid()))
            self.fp.flush()
            return True
        except Exception:
            return False

    def release(self):
        if self.fp:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    self.fp.seek(0)
                    msvcrt.locking(self.fp.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
                self.fp.close()
            except Exception:
                pass
            self.fp = None


async def main():
    lock = ProcessSingletonLock(ROOT / "logs" / "trading_bot.lock")
    if not lock.acquire():
        logger.warning("⚠️ Another trading bot instance is already active and running! Exiting duplicate process.")
        print("\n[!] Another trading bot instance is already running. Please attach to the existing session via: screen -r bot\n")
        return

    engine = None
    try:
        live_mode = "--live" in sys.argv or "--live-orders" in sys.argv or settings.LIVE_TRADING
        engine = LastHopeTradingEngine(live_orders=live_mode)
        await engine.run()
    finally:
        # Restore terminal: leave alternate screen, show cursor (htop-style exit)
        if engine is not None and getattr(engine, "_alt_screen_on", False):
            try:
                sys.stdout.write("\033[?25h\033[?1049l")
                sys.stdout.flush()
            except Exception:
                pass
        lock.release()


if __name__ == "__main__":
    asyncio.run(main())
