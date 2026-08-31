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
  - Breakeven Stop (BE): At +70% of distance, SL hardens to Entry + 1.0 pt
  - SL priority over TP
"""

import asyncio
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
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

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console(force_terminal=True, legacy_windows=False)

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
                    {"name": "Risk Geometry", "value": "ATR(10)x1.5 Breakeven at +70% move", "inline": True},
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
                # Filter to regular market trading hours (09:15 to 15:29 IST, minute 555 to 929)
                parsed_y_rows = []
                for x in by_day[last_trading_day]:
                    try:
                        x_dt = datetime.strptime(str(x["time"]), "%d-%m-%Y %H:%M:%S")
                        x_m = x_dt.hour * 60 + x_dt.minute
                        if 555 <= x_m <= 929:
                            parsed_y_rows.append((x_dt, x))
                    except Exception:
                        pass

                parsed_y_rows.sort(key=lambda item: item[0])
                if parsed_y_rows:
                    yesterday_rows = [item[1] for item in parsed_y_rows]
                    yh = max(float(x.get("high", x.get("inth", 0.0))) for x in yesterday_rows)
                    yl = min(float(x.get("low", x.get("intl", 0.0))) for x in yesterday_rows if float(x.get("low", x.get("intl", 0.0))) > 0)
                    yc = float(yesterday_rows[-1].get("close", yesterday_rows[-1].get("intc", 0.0)))
                    contract_state.set_day_sr_levels(yh, yl, yc)
                    logger.info(f"Initialized Day S/R for {symbol} from {last_trading_day}: H={yh:.2f} L={yl:.2f} C={yc:.2f}")

            # Collect completed 1m bars from previous days for macro stochastics / ATR warmup
            prior_bars: List[Bar1m] = []
            for d in valid_prev_days[-2:]:
                for r in by_day[d]:
                    try:
                        dt = datetime.strptime(str(r["time"]), "%d-%m-%Y %H:%M:%S")
                    except (TypeError, ValueError):
                        continue
                    m = dt.hour * 60 + dt.minute
                    o = float(r.get("open", r.get("into", 0.0)))
                    h = float(r.get("high", r.get("inth", 0.0)))
                    l = float(r.get("low", r.get("intl", 0.0)))
                    c = float(r.get("close", r.get("intc", 0.0)))
                    v = float(r.get("volume", r.get("intv", r.get("v", 100.0))))
                    if h >= l > 0:
                        prior_bars.append(Bar1m(minute=m, open=o, high=h, low=l, close=c, timestamp=dt, volume=v))

            # Today's completed candles (intraday session VWAP & live indicators)
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
        """Resolves 2nd ITM contracts + rollover watch pairs and warms up indicators & daily S/R levels."""
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

        # Prune stale contracts not in desired_keys or active position
        for k in list(self.engine.contracts.keys()):
            if k not in desired_keys and k != self.active_position_key:
                del self.engine.contracts[k]
                self._last_ltp.pop(k, None)

        for side, strike in pairs:
            key = f"{side}:{strike}"
            if key in self.engine.contracts:
                continue
            try:
                scrip = await self.history.search_option_token(f"NIFTY {strike} {side}")
                if not scrip or not scrip.get("token"):
                    continue

                contract_state = self.engine.register_contract(
                    key=key,
                    symbol=scrip["tsym"],
                    token=scrip["token"],
                    side=side,
                    strike=strike,
                )

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

        self.active_position_key = f"{sig['side']}:{sig['strike']}"
        logger.info(f"ENTRY {display} @ {fill:.2f} SL={sl:.2f} TP={tp:.2f} BE_trig={sig['be_trigger_px']:.2f}")

        await self.discord.send_trade_alert(
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

    async def _record_close(self, trade: Dict[str, Any]):
        self.trades_today.append(trade)
        if float(trade.get("pts", 0.0)) > 0:
            self._wins_today += 1
        self.engine.on_trade_closed()
        self.active_position_key = None
        self.paper_position = None

    async def _manage_exit(self):
        if not self._has_position() or self.active_position_key is None:
            return
        ltp = self._last_ltp.get(self.active_position_key)
        if ltp is None:
            return
        now = ist_now()

        # Handle Live Broker Position Exit
        if self.live_orders and self.executor is not None:
            # Sync SL if Breakeven triggered
            if self.engine.active_trade and self.executor.position:
                if self.engine.active_trade.get("be_done") and not self.executor.position.get("be_notified"):
                    self.executor.position["be_notified"] = True
                    self.executor.position["sl"] = self.engine.active_trade["sl"]
                    await self.discord.notify_breakeven_locked(
                        symbol=self.executor.position["symbol"],
                        entry=float(self.executor.position["entry"]),
                        new_sl=float(self.engine.active_trade["sl"]),
                        ltp=ltp,
                    )

            res = await self.executor.check_exit(ltp, now)
            if res.get("accepted"):
                await self._record_close(res.get("trade", {}))
                logger.info(f"✅ Position Closed on Flattrade: {res.get('trade', {}).get('reason')}")
            return

        # Handle Paper Position Exit
        pos = self.paper_position
        if pos is None:
            return

        # Breakeven Stop Check
        if not pos.get("be_done") and ltp >= pos["be_trigger_px"]:
            pos["be_done"] = True
            pos["sl"] = pos["be_hardened_sl"]
            logger.info(f"Paper Breakeven Triggered on {pos['symbol']}: SL moved to {pos['sl']:.2f}")
            await self.discord.notify_breakeven_locked(
                symbol=pos["symbol"],
                entry=float(pos["entry"]),
                new_sl=float(pos["sl"]),
                ltp=ltp,
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
        await self.discord.notify_trade_close(trade_info)

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
        if now_ts - getattr(self, "_last_token_renew", 0.0) < 300:
            return True
        self._last_token_renew = now_ts
        logger.warning("Session expired. Renewing token via REST API...")
        try:
            from flattrade_bot.broker.auto_login import automated_flattrade_login_rest
            new_token = automated_flattrade_login_rest(
                user_id=settings.FLATTRADE_USER_ID,
                password=settings.FLATTRADE_PASSWORD,
                totp_key=settings.FLATTRADE_TOTP_KEY,
                api_key=settings.FLATTRADE_API_KEY,
                api_secret=settings.FLATTRADE_API_SECRET,
            )
            if new_token:
                self.client.set_token(new_token)
                self.history.set_token(new_token)
                return True
        except Exception as e:
            logger.error(f"Token renewal failed: {e}")
        return True

    def render_dashboard(self) -> Group:
        now = ist_now()
        time_str = now.strftime("%H:%M:%S")
        header = Text.from_markup(
            f" [bold bright_yellow]🏆 LAST HOPE GPU WINNER[/bold bright_yellow] "
            f"| [bold white]2nd ITM · Multi-TF Stoch · S/R Bounce · ATR×1.5 + BE[/bold white] "
            f"| [cyan]{time_str} IST[/cyan]"
        )
        banner = Panel(Align.center(header), box=box.ROUNDED, style="bright_blue", padding=(0, 1))

        sess_active = SESSION_START_MIN <= minute_of(now) < SESSION_END_MIN
        mode_str = "[bold white on red] LIVE BROKER [/bold white on red]" if self.live_orders else "[bold white on blue] PAPER SIM [/bold white on blue]"
        net_rs = sum(float(t.get("rs", 0.0)) for t in self.trades_today)
        net_pts = sum(float(t.get("pts", 0.0)) for t in self.trades_today)
        pnl_color = "bold green" if net_rs >= 0 else "bold red"
        atm = int(round(self.spot_price / 50.0) * 50) if self.spot_price else "--"
        total_trades = len(self.trades_today)
        wr = (self._wins_today / total_trades * 100.0) if total_trades > 0 else 0.0

        # ── 1. Compact System Telemetry Bar ─────
        sys_table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
        for col, style in (
            ("MODE", "bold white"),
            ("SPOT (ATM)", "bold yellow"),
            ("2nd ITM (CE / PE)", "bold cyan"),
            ("SESSION", "bold green"),
            ("TRADES (W/L)", "bold white"),
            ("NET P&L (₹)", "bold white"),
        ):
            sys_table.add_column(col, style=style, justify="center", no_wrap=True)

        spot_atm_str = f"₹{self.spot_price:,.1f} ({atm})" if self.spot_price else "--"
        itm_pair_str = f"₹{atm - 100} CE / ₹{atm + 100} PE" if isinstance(atm, int) else "--"

        sys_table.add_row(
            mode_str,
            spot_atm_str,
            itm_pair_str,
            "[bold green]ACTIVE[/bold green]" if sess_active else "[yellow]CLOSED[/yellow]",
            f"{total_trades} ({self._wins_today}W/{total_trades - self._wins_today}L | {wr:.0f}%)",
            f"[{pnl_color}]₹{net_rs:+,.2f} ({net_pts:+.1f}p)[/{pnl_color}]",
        )

        # ── 2. Strategy Setup Forming & Arming Radar ─────────────────────────
        mon_table = Table(
            title="[bold cyan]📡 STRATEGY SETUP RADAR & ARMING MATRIX (1m/2m/3m/5m Option Stochastics · S/R Suite)[/bold cyan]",
            box=box.SIMPLE_HEAD, expand=True, padding=(0, 1),
        )
        for col, style, j in (
            ("STRIKE", "bold white", "left"),
            ("ARMING", "bold yellow", "center"),
            ("1m S1/S3/S4", "bold white", "center"),
            ("2m/3m/5m S4", "bold white", "center"),
            ("ACTIVE TF SETUPS", "bold magenta", "left"),
            ("S/R LEVEL (PROXIMITY)", "bold cyan", "left"),
            ("SL / TP", "bold white", "center"),
            ("LTP", "bold yellow", "right"),
        ):
            mon_table.add_column(col, style=style, justify=j, no_wrap=True)

        for key, cs in self.engine.contracts.items():
            s1_val = cs.tf_trackers[1].last_s1
            s3_val = cs.tf_trackers[1].last_s3
            s4_val = cs.tf_trackers[1].last_s4
            s4_2m = cs.tf_trackers[2].last_s4
            s4_3m = cs.tf_trackers[3].last_s4
            s4_5m = cs.tf_trackers[5].last_s4
            bar_count = len(cs.bars)

            contract_clean = f"{cs.strike} {cs.side}"

            # Arming state string with countdown
            if cs.flag_armed or cs.super_armed:
                arm_age = max(0, bar_count - max(cs.flag_arm_bar, cs.super_arm_bar))
                arm_rem = max(0, ARM_WINDOW - arm_age)
                armed_str = f"[bold green]ARMED ({arm_rem}b)[/bold green]"
            else:
                armed_str = "[dim]FLAT[/dim]"

            fmt = lambda v: f"{v:.0f}" if isinstance(v, (int, float)) and v else "--"

            # Multi-TF Setup Forming Check across 1m, 2m, 3m, 5m
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
                        if is_rise:
                            tf_signals.append(f"[bold green]SUPER {tf}m↑[/bold green]")
                        else:
                            tf_signals.append(f"[yellow]Super {tf}m[/yellow]")

            setup_str = ", ".join(tf_signals) if tf_signals else "[dim]Scanning...[/dim]"

            # Nearest S/R Level Proximity
            ltp = self._last_ltp.get(key, 0.0)
            nearest_sr_str = "--"
            if cs.sr_levels and ltp > 0:
                active_sr = dict(cs.sr_levels)
                if cs.ema20.value: active_sr["EMA20"] = cs.ema20.value
                if cs.ema200.value: active_sr["EMA200"] = cs.ema200.value
                if cs.vwap.value: active_sr["VWAP"] = cs.vwap.value

                closest_lvl = min(active_sr.items(), key=lambda item: abs(ltp - item[1]))
                diff = ltp - closest_lvl[1]
                if abs(diff) <= 0.5:
                    nearest_sr_str = f"{closest_lvl[0]} [bold green](TOUCH {diff:+.1f})[/bold green]"
                else:
                    nearest_sr_str = f"{closest_lvl[0]} ({diff:+.1f}p)"

            dist_pts = min(max(cs.latest_atr * ATR_MULT, 2.0), TP_PTS_CAP)
            side_color = "bold green" if cs.side == "CE" else "bold red"
            stoch_1m = f"{fmt(s1_val)}/{fmt(s3_val)}/{fmt(s4_val)}"
            stoch_macro = f"{fmt(s4_2m)}/{fmt(s4_3m)}/{fmt(s4_5m)}"

            mon_table.add_row(
                f"[{side_color}]{contract_clean}[/{side_color}]",
                armed_str,
                stoch_1m,
                stoch_macro,
                setup_str,
                nearest_sr_str,
                f"±{dist_pts:.1f}",
                f"₹{ltp:.2f}" if ltp > 0 else "--",
            )

        # ── 3. Option S/R Suite & Technical Indicators Table ─────────────────
        sr_table = Table(
            title="[bold cyan]📐 OPTION S/R SUITE & TECHNICAL INDICATORS (CPR · Camarilla · EMA20/200 · VWAP · ATR)[/bold cyan]",
            box=box.SIMPLE_HEAD, expand=True, padding=(0, 1),
        )
        for col, style, j in (
            ("STRIKE", "bold white", "left"),
            ("LTP", "bold yellow", "right"),
            ("PDH / PDL", "bold white", "center"),
            ("CPR (BC / PIVOT / TC)", "bold cyan", "center"),
            ("EMA20 / EMA200", "bold white", "center"),
            ("VWAP", "bold magenta", "center"),
            ("ATR (DIST)", "bold green", "right"),
        ):
            sr_table.add_column(col, style=style, justify=j, no_wrap=True)

        for key, cs in self.engine.contracts.items():
            ltp = self._last_ltp.get(key, 0.0)
            f2 = lambda v: f"{v:.1f}" if isinstance(v, (int, float)) and v else "--"
            pdh = f2(cs.sr_levels.get("PDH"))
            pdl = f2(cs.sr_levels.get("PDL"))
            cpr = f"{f2(cs.sr_levels.get('CPR_BC'))} / {f2(cs.sr_levels.get('CPR_Pivot'))} / {f2(cs.sr_levels.get('CPR_TC'))}"
            ema = f"{f2(cs.ema20.value)} / {f2(cs.ema200.value)}"
            vwap = f2(cs.vwap.value)
            atr = min(max(cs.latest_atr * ATR_MULT, 2.0), TP_PTS_CAP)
            side_color = "bold green" if cs.side == "CE" else "bold red"
            contract_clean = f"{cs.strike} {cs.side}"

            sr_table.add_row(
                f"[{side_color}]{contract_clean}[/{side_color}]",
                f"₹{ltp:.2f}" if ltp > 0 else "--",
                f"{pdh} / {pdl}",
                cpr,
                ema,
                vwap,
                f"±{atr:.1f} pts",
            )

        # ── 4. Active Trade Cockpit (if position is open) ─────────────────────
        tables = [banner, sys_table, mon_table, sr_table]
        pos = self.paper_position or (self.executor.position if self.executor else None)
        if pos:
            ltp = self._last_ltp.get(self.active_position_key, pos["entry"])
            pts = round(ltp - pos["entry"], 2)
            rs = round(pts * pos["quantity"], 2)
            c = "bold green" if rs >= 0 else "bold red"
            be_status = "[bold green]🔒 LOCKED (+1.0 pt)[/bold green]" if pos.get("be_done") else f"Target: ₹{pos.get('be_trigger_px', 0):.2f}"

            pos_table = Table(
                title=f"[bold green]🎯 ACTIVE TRADE COCKPIT — {pos.get('symbol', '--')}[/bold green]",
                box=box.ROUNDED, expand=True, padding=(0, 1),
            )
            for col in ("SIGNAL", "ENTRY", "LTP", "STOP LOSS", "BREAKEVEN (70%)", "TARGET", "PTS", "P&L (₹)"):
                pos_table.add_column(col, style="bold white", justify="center", no_wrap=True)

            pos_table.add_row(
                pos.get("signal", "--"),
                f"₹{pos['entry']:.2f}",
                f"₹{ltp:.2f}",
                f"[bold red]₹{pos['sl']:.2f}[/bold red]",
                be_status,
                f"[bold green]₹{pos.get('target', 0):.2f}[/bold green]",
                f"[{c}]{pts:+.2f}[/{c}]",
                f"[{c}]₹{rs:+,.2f}[/{c}]",
            )
            tables.append(pos_table)

        return Group(*tables)

    async def _main_loop_body(self) -> float:
        """Execute one iteration of the trading loop. Returns elapsed seconds."""
        started = time.time()
        now = ist_now()

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

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process Spot Quote
        spot_res = results[0] if results else None
        if isinstance(spot_res, dict):
            if await self._renew_token_if_expired(spot_res):
                pass
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

        touch_runtime_record()
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

        if has_tty:
            with Live(
                self.render_dashboard(),
                console=console,
                refresh_per_second=1,
                screen=False,
                vertical_overflow="visible",
            ) as live:
                while True:
                    try:
                        elapsed = await self._main_loop_body()
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

    try:
        live_mode = "--live" in sys.argv or "--live-orders" in sys.argv or settings.LIVE_TRADING
        engine = LastHopeTradingEngine(live_orders=live_mode)
        await engine.run()
    finally:
        lock.release()


if __name__ == "__main__":
    asyncio.run(main())
