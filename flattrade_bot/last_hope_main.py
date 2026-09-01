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
import math
import os
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


class LastHopeTradingEngine:
    """Live trading engine for the Last Hope GPU Winner strategy."""

    def __init__(self, live_orders: bool = False):
        self.live_orders = live_orders
        self.engine = LastHopeWinnerEngine()
        self.discord = DiscordNotifier(strategy=STRATEGY_LABEL)
        self.client = FlattradeClient()
        self.history = FlattradeHistoryFetcher()
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
                    {"name": "Risk Geometry", "value": "ATR(10)x1.5 Breakeven at +50% move (§42 champion)", "inline": True},
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
                contract_state.seed_1m_bars(prior_bars, today_bars)
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
        for k in list(self.engine.contracts.keys()):
            if k not in desired_keys and k != self.active_position_key:
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

                contract_state = self.engine.register_contract(
                    key=key,
                    symbol=scrip["tsym"],
                    token=scrip["token"],
                    side=side,
                    strike=strike,
                )
                # Per-day cold start (backtest parity): fresh indicators, then
                # warmup replays today's completed bars only.
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

        self.active_position_key = f"{sig['side']}:{sig['strike']}"
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
        if float(trade.get("pts", 0.0)) > 0:
            self._wins_today += 1
        self.engine.on_trade_closed()
        self.active_position_key = None
        self.paper_position = None

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
                self._last_token_renew = time.time()
                logger.info("Token renewed successfully.")
                return True
            else:
                logger.error("Token renewal returned empty — will retry in 60s")
        except Exception as e:
            logger.error(f"Token renewal failed: {e} — will retry in 60s")
        return True

    def print_dashboard(self):
        """Zero-flicker dashboard: redraw in place via ANSI cursor-up + line-erase.
        Never full-screen clears (that's the blink). No Rich, no threads."""
        L: List[str] = []
        p = L.append

        now = ist_now()
        time_str = now.strftime("%H:%M:%S")
        sess_active = SESSION_START_MIN <= minute_of(now) < SESSION_END_MIN
        mode = "LIVE" if self.live_orders else "PAPER"
        net_rs = sum(float(t.get("rs", 0.0)) for t in self.trades_today)
        net_pts = sum(float(t.get("pts", 0.0)) for t in self.trades_today)
        atm = int(round(self.spot_price / 50.0) * 50) if self.spot_price else 0
        total = len(self.trades_today)
        wr = (self._wins_today / total * 100.0) if total > 0 else 0.0
        sess = "ACTIVE" if sess_active else "CLOSED"

        p(f"{'='*90}")
        p(f"  LAST HOPE GPU WINNER | 2nd ITM | Multi-TF Stoch | S/R Bounce | ATRx1.5 + BE | {time_str} IST")
        p(f"{'='*90}")
        p(f"  MODE: {mode} | SPOT: {self.spot_price:.1f} ({atm}) | ITM: {atm-100} CE / {atm+100} PE | SESSION: {sess}")
        p(f"  TRADES: {total} ({self._wins_today}W/{total - self._wins_today}L {wr:.0f}%) | NET: Rs {net_rs:+,.2f} ({net_pts:+.1f}p)")
        p(f"{'-'*90}")
        p(f"  {'STRIKE':<12} {'ARM':<10} {'STOCH 1m':<12} {'STOCH multi':<14} {'SETUP':<20} {'PROX S/R':<16} {'LTP':>8}")
        p(f"  {'-'*86}")

        for key, cs in self.engine.contracts.items():
            s1 = cs.tf_trackers[1].last_s1
            s3 = cs.tf_trackers[1].last_s3
            s4 = cs.tf_trackers[1].last_s4
            s4_2 = cs.tf_trackers[2].last_s4
            s4_3 = cs.tf_trackers[3].last_s4
            s4_5 = cs.tf_trackers[5].last_s4
            bar_count = len(cs.bars)
            ltp = self._last_ltp.get(key, 0.0)
            f0 = lambda v: f"{v:.0f}" if isinstance(v, (int, float)) and v else "--"

            arm = "FLAT"
            if cs.flag_armed or cs.super_armed:
                arm_age = max(0, bar_count - max(cs.flag_arm_bar, cs.super_arm_bar))
                arm_rem = max(0, ARM_WINDOW - arm_age)
                arm = f"ARMED({arm_rem}b)"

            tf_signals = []
            for tf, trk in cs.tf_trackers.items():
                t_s1, t_s3, t_s4 = trk.last_s1, trk.last_s3, trk.last_s4
                if t_s4 is not None and t_s1 is not None:
                    if t_s4 >= M6_S4 and t_s1 < M6_S1:
                        tf_signals.append(f"FLAG {tf}m")
                    elif t_s4 >= 72.0:
                        tf_signals.append(f"Flag {tf}m")
                if t_s1 is not None and t_s3 is not None and t_s4 is not None:
                    if t_s1 < SUPER_THRESH and t_s3 < SUPER_THRESH and t_s4 < SUPER_THRESH:
                        is_rise = t_s1 > (trk.prev_s1 or 0)
                        tf_signals.append(f"SUPER {tf}m" + ("+" if is_rise else ""))
            setup = ", ".join(tf_signals) if tf_signals else "..."

            prox = "--"
            if cs.sr_levels and ltp > 0:
                active_sr = dict(cs.sr_levels)
                if cs.ema20.value: active_sr["EMA20"] = cs.ema20.value
                if cs.ema200.value: active_sr["EMA200"] = cs.ema200.value
                if cs.vwap.value: active_sr["VWAP"] = cs.vwap.value
                closest = min(active_sr.items(), key=lambda x: abs(ltp - x[1]))
                diff = ltp - closest[1]
                prox = f"{closest[0]} ({diff:+.1f}p)"

            stoch1 = f"{f0(s1)}/{f0(s3)}/{f0(s4)}"
            stochm = f"{f0(s4_2)}/{f0(s4_3)}/{f0(s4_5)}"

            p(f"  {cs.strike} {cs.side:<4} {arm:<10} {stoch1:<12} {stochm:<14} {setup:<20} {prox:<16} {ltp:>8.2f}")

        p(f"  {'-'*86}")
        p(f"  S/R LEVELS (TradingView verify)")
        p(f"  {'STRIKE':<12} {'LTP':>8}  {'PDH/PDL':<14} {'CPR(B/P/T)':<18} {'EMA20/200 VWAP':<20} {'ATR':>6}")
        p(f"  {'-'*86}")

        for key, cs in self.engine.contracts.items():
            ltp = self._last_ltp.get(key, 0.0)
            f2 = lambda v: f"{v:.0f}" if isinstance(v, (int, float)) and v else "--"
            pdh = f2(cs.sr_levels.get("PDH"))
            pdl = f2(cs.sr_levels.get("PDL"))
            cpr = f"{f2(cs.sr_levels.get('CPR_BC'))}/{f2(cs.sr_levels.get('CPR_Pivot'))}/{f2(cs.sr_levels.get('CPR_TC'))}"
            ema = f"{f2(cs.ema20.value)}/{f2(cs.ema200.value)} {f2(cs.vwap.value)}"
            atr = min(max(cs.latest_atr * ATR_MULT, 2.0), TP_PTS_CAP)
            p(f"  {cs.strike} {cs.side:<4} {ltp:>8.0f}  {pdh}/{pdl:<12} {cpr:<18} {ema:<20} {atr:>5.1f}")

        p(f"{'='*90}")

        # Zero-flicker in-place redraw (SO 34828142 / Rich #2726):
        # cursor-up N lines, then erase+write each line. Full-screen clear blinks.
        if self._dash_lines_drawn == len(L):
            buf = [f"\033[{len(L)}A"]
            buf.extend("\033[2K" + line for line in L)
            sys.stdout.write("\n".join(buf) + "\n")
        elif self._dash_lines_drawn == 0:
            sys.stdout.write("\n".join(L) + "\n")
        else:
            # Frame height changed (contract count changed): clear region safely.
            sys.stdout.write(f"\033[{self._dash_lines_drawn}A\033[J")
            sys.stdout.write("\n".join(L) + "\n")
        sys.stdout.flush()
        self._dash_lines_drawn = len(L)

    async def _on_day_rollover(self, now: datetime):
        """Resets all daily state and forces full re-warmup when the calendar date changes (IST)."""
        logger.warning("🌅 NEW TRADING DAY %s — resetting daily state and re-seeding S/R levels...", now.date())

        # Preserve open-position bookkeeping, reset everything else
        self.trades_today = []
        self._wins_today = 0
        self._eod_done = False
        self.risk.reset_day()

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

        # 2. Poll Spot and all Option Quotes concurrently in a single parallel burst
        poll_keys = sorted(self.engine.contracts.keys())
        if self.active_position_key and self.active_position_key not in poll_keys:
            poll_keys.append(self.active_position_key)

        tasks = [self.client.get_quotes(exchange="NSE", token="26000")]
        for key in poll_keys:
            cs = self.engine.contracts.get(key)
            if cs:
                tasks.append(self.client.get_quotes(exchange="NFO", token=cs.token))
            else:
                tasks.append(asyncio.sleep(0))

        # Timeout-capped gather: one slow endpoint must never stall the tick loop.
        # Each request gets a private 2.0s deadline; laggards are cancelled and
        # their slot left as None (skipped downstream).
        results: List[Any] = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=2.0,
        )

        # Log any exceptions from the gather (timeouts, connection errors, etc.)
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                label = "spot" if idx == 0 else f"option[{poll_keys[idx-1] if idx-1 < len(poll_keys) else '?'}]"
                logger.warning("Quote poll exception for %s: %s", label, result)

        # Process Spot Quote
        spot_res = results[0] if results else None
        if isinstance(spot_res, dict):
            if await self._renew_token_if_expired(spot_res):
                pass  # Token renewed; next iteration will use it
            elif spot_res.get("stat") == "Ok" and "lp" in spot_res:
                try:
                    self.spot_price = float(spot_res["lp"])
                    self.engine.set_spot_price(self.spot_price)
                    self._broker_status = "LIVE CONNECTED"
                except (ValueError, TypeError):
                    pass

        # Process Option Quotes
        for idx, key in enumerate(poll_keys, start=1):
            if idx >= len(results):
                break
            oq = results[idx]
            if not isinstance(oq, dict):
                continue
            if is_session_expired_response(oq):
                await self._renew_token_if_expired(oq)
                continue
            if oq.get("stat") != "Ok" or "lp" not in oq:
                if oq.get("stat") != "Ok":
                    logger.debug("Quote %s stat=%s emsg=%s", key, oq.get("stat"), oq.get("emsg", ""))
                continue
            try:
                opt_ltp = float(oq["lp"])
            except (ValueError, TypeError):
                continue
            if opt_ltp <= 0:
                continue
            self._last_ltp[key] = opt_ltp

            # Push tick to strategy engine
            sig = self.engine.push_tick(key, opt_ltp, now)
            if sig and not self._has_position():
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
        # Log S/R levels for TradingView verification
        for key, cs in self.engine.contracts.items():
            if not cs.sr_levels:
                continue
            ltp = self._last_ltp.get(key, 0)
            sr_parts = []
            for name, price in cs.sr_levels.items():
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
        await self.recover_open_positions()

        has_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

        loop_count = 0
        while True:
            try:
                elapsed = await self._main_loop_body()
                loop_count += 1
                if has_tty:
                    if loop_count % 2 == 0:
                        self.print_dashboard()
                else:
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

    try:
        live_mode = "--live" in sys.argv or "--live-orders" in sys.argv or settings.LIVE_TRADING
        engine = LastHopeTradingEngine(live_orders=live_mode)
        await engine.run()
    finally:
        lock.release()


if __name__ == "__main__":
    asyncio.run(main())
