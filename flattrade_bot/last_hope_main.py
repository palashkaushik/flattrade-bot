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

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
console = Console(legacy_windows=False)

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

        asyncio.create_task(
            self.discord._post_embed({
                "title": "🟢 FLATTRADE LAST HOPE WINNER BOT ONLINE",
                "color": 0x2ECC71,
                "fields": [
                    {"name": "Strategy", "value": "Last Hope GPU Winner (FLAG/SUPER · 1m OHLC)", "inline": True},
                    {"name": "Session", "value": "09:15 - 15:00 IST", "inline": True},
                    {"name": "Risk Geometry", "value": "ATR(10)×1.5 · Breakeven at +70% move", "inline": True},
                    {"name": "Mode", "value": "🔴 LIVE ORDERS" if self.live_orders else "🟣 PAPER / SIM", "inline": True},
                    {"name": "Nifty Spot", "value": f"₹{self.spot_price:,.2f}" if self.spot_price else "--", "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "Flattrade Last Hope Bot"},
            })
        )

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

        now = ist_now()
        today_str = now.strftime("%d-%m-%Y")

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

                # Fetch historical 1m bars for warm-up and Daily CPR / Camarilla initialization
                try:
                    seed_rows = await self.history.fetch_historical_candles(
                        token=scrip["token"], exchange="NFO", interval="1", days_back=3
                    )
                    if seed_rows:
                        # Group rows by day
                        by_day: Dict[str, List[Dict[str, Any]]] = {}
                        for r in seed_rows:
                            t_str = str(r.get("time", ""))
                            d_part = t_str.split(" ")[0] if " " in t_str else t_str
                            by_day.setdefault(d_part, []).append(r)

                        # Find previous day to build CPR / Camarilla
                        prev_days = [d for d in sorted(by_day.keys()) if d != today_str]
                        if prev_days:
                            yesterday_rows = by_day[prev_days[-1]]
                            yh = max(float(x["into"]) for x in yesterday_rows if "into" in x)
                            yl = min(float(x["intl"]) for x in yesterday_rows if "intl" in x)
                            yc = float(yesterday_rows[-1]["intc"])
                            contract_state.set_day_sr_levels(yh, yl, yc)
                            logger.info(f"Initialized Day S/R for {scrip['tsym']}: H={yh:.2f} L={yl:.2f} C={yc:.2f}")

                        # Warm up 1m bars from today's past completed bars
                        if today_str in by_day:
                            cur_m = minute_of(now)
                            for r in by_day[today_str]:
                                try:
                                    dt = datetime.strptime(str(r["time"]), "%d-%m-%Y %H:%M:%S")
                                except (TypeError, ValueError):
                                    continue
                                if (dt.hour * 60 + dt.minute) < cur_m:
                                    contract_state.push_tick(float(r["into"]), dt)
                                    contract_state.push_tick(float(r["inth"]), dt)
                                    contract_state.push_tick(float(r["intl"]), dt)
                                    contract_state.push_tick(float(r["intc"]), dt)
                except Exception as e:
                    logger.warning(f"Warmup fetch failed for {key}: {e}")

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
        logger.info(f"🚨 {sig['trigger']} ENTRY {display} @ {fill:.2f} SL={sl:.2f} TP={tp:.2f} BE_trig={sig['be_trigger_px']:.2f}")

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
                    asyncio.create_task(
                        self.discord.notify_breakeven_locked(
                            symbol=self.executor.position["symbol"],
                            entry=float(self.executor.position["entry"]),
                            new_sl=float(self.engine.active_trade["sl"]),
                            ltp=ltp,
                        )
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
            logger.info(f"🔒 Paper Breakeven Triggered on {pos['symbol']}: SL moved to {pos['sl']:.2f}")
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
            f" [bold bright_yellow]🏆 LAST HOPE GPU WINNER STRATEGY[/bold bright_yellow] "
            f"| [bold white]2nd ITM · Multi-TF Stoch (1m/2m/3m/5m) · S/R Bounce · ATR×1.5 + BE[/bold white] "
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

        # ── 1. System Telemetry Bar ──────────────────────────────────────────
        sys_table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
        for col, style in (("BROKER SESSION", "bold green"), ("EXEC MODE", "bold white"), ("SESSION STATUS", "bold white"),
                           ("NIFTY 50 SPOT", "bold yellow"), ("ATM STRIKE", "bold white"), ("2nd ITM CE", "bold green"),
                           ("2nd ITM PE", "bold red"), ("TRADES (W/L)", "bold white"), ("WIN RATE", "bold white"),
                           ("NET PTS", "bold white"), ("NET P&L (₹)", "bold white")):
            sys_table.add_column(col, style=style, justify="center")

        sys_table.add_row(
            self._broker_status,
            mode_str,
            "[bold green]ACTIVE (09:15-15:00)[/bold green]" if sess_active else "[yellow]MARKET CLOSED[/yellow]",
            f"₹{self.spot_price:,.2f}" if self.spot_price else "--",
            str(atm),
            f"₹{atm - 100}" if isinstance(atm, int) else "--",
            f"₹{atm + 100}" if isinstance(atm, int) else "--",
            f"{total_trades} ({self._wins_today}W / {total_trades - self._wins_today}L)",
            f"[bold cyan]{wr:.1f}%[/bold cyan]",
            f"[{pnl_color}]{net_pts:+.2f} pts[/{pnl_color}]",
            f"[{pnl_color}]₹{net_rs:+,.2f}[/{pnl_color}]",
        )

        # ── 2. Strategy Setup Forming & Arming Radar ─────────────────────────
        mon_table = Table(
            title="[bold cyan]📡 STRATEGY SETUP RADAR & ARMING MATRIX (1m/2m/3m/5m Option Stochastics · S/R Suite)[/bold cyan]",
            box=box.SIMPLE_HEAD, expand=True, padding=(0, 1),
        )
        for col, style in (("CONTRACT", "bold white"), ("SIDE", "bold white"), ("ARMING STATUS", "bold yellow"),
                           ("S1 %D (12,3)", "bold white"), ("S3 %D (40,4)", "bold white"), ("S4 %D (50,10)", "bold white"),
                           ("FLAG / SUPER SETUP", "bold magenta"), ("NEAREST S/R LEVEL", "bold cyan"),
                           ("S/R DIST", "bold white"), ("ATR(10) / SL DIST", "bold white"), ("LTP", "bold yellow")):
            mon_table.add_column(col, style=style, justify="center")

        for key, cs in self.engine.contracts.items():
            s1_val = cs.tf_trackers[1].last_s1
            s3_val = cs.tf_trackers[1].last_s3
            s4_val = cs.tf_trackers[1].last_s4
            bar_count = len(cs.bars)

            # Arming state string with countdown
            if cs.flag_armed or cs.super_armed:
                arm_age = max(0, bar_count - max(cs.flag_arm_bar, cs.super_arm_bar))
                arm_rem = max(0, ARM_WINDOW - arm_age)
                armed_str = f"[bold green]ARMED ({arm_rem}/10 bars left)[/bold green]"
            else:
                armed_str = "[dim]FLAT (S1 > 25)[/dim]"

            fmt = lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "--"

            # Setup Forming Readiness Check
            setup_forming = []
            if s4_val is not None and s1_val is not None:
                if s4_val >= M6_S4 and s1_val < M6_S1:
                    setup_forming.append("[bold green]FLAG (M6) READY[/bold green]")
                elif s4_val >= 70.0:
                    setup_forming.append(f"[yellow]Flag Forming (S4={s4_val:.0f})[/yellow]")

            if s1_val is not None and s3_val is not None and s4_val is not None:
                if s1_val < SUPER_THRESH and s3_val < SUPER_THRESH and s4_val < SUPER_THRESH:
                    rising = cs.tf_trackers[1].last_s1 > (cs.tf_trackers[1].prev_s1 or 0)
                    if rising:
                        setup_forming.append("[bold green]SUPER READY (S1↑)[/bold green]")
                    else:
                        setup_forming.append("[yellow]Super (Waiting S1↑)[/yellow]")

            setup_str = " | ".join(setup_forming) if setup_forming else "[dim]Scanning...[/dim]"

            # Nearest S/R Level Proximity
            ltp = self._last_ltp.get(key, 0.0)
            nearest_sr_name = "--"
            nearest_sr_dist = "--"
            if cs.sr_levels and ltp > 0:
                active_sr = dict(cs.sr_levels)
                if cs.ema20.value: active_sr["EMA20"] = cs.ema20.value
                if cs.ema200.value: active_sr["EMA200"] = cs.ema200.value
                if cs.vwap.value: active_sr["VWAP"] = cs.vwap.value

                # Find closest level
                closest_lvl = min(active_sr.items(), key=lambda item: abs(ltp - item[1]))
                nearest_sr_name = f"{closest_lvl[0]} ({closest_lvl[1]:.1f})"
                diff = ltp - closest_lvl[1]
                if abs(diff) <= 0.5:
                    nearest_sr_dist = f"[bold green]TOUCHED ({diff:+.1f})[/bold green]"
                else:
                    nearest_sr_dist = f"{diff:+.1f} pts"

            dist_pts = min(max(cs.latest_atr * ATR_MULT, 2.0), TP_PTS_CAP)
            side_color = "bold green" if cs.side == "CE" else "bold red"

            mon_table.add_row(
                cs.symbol,
                f"[{side_color}]{cs.side} {cs.strike}[/{side_color}]",
                armed_str,
                fmt(s1_val), fmt(s3_val), fmt(s4_val),
                setup_str,
                nearest_sr_name,
                nearest_sr_dist,
                f"±{dist_pts:.1f} pts",
                f"₹{ltp:.2f}" if ltp > 0 else "--",
            )

        tables: List[Any] = [banner, sys_table, mon_table]

        # ── 3. Active Trade Live Cockpit (Ongoing Position Telemetry) ────────
        pos_data = None
        if self.live_orders and self.executor is not None and self.executor.position:
            p = self.executor.position
            be_trig = self.engine.active_trade.get("be_trigger_px", 0.0) if self.engine.active_trade else 0.0
            be_done = self.engine.active_trade.get("be_done", False) if self.engine.active_trade else False
            pos_data = {**p, "be_trigger_px": be_trig, "be_done": be_done}
        elif self.paper_position is not None:
            pos_data = self.paper_position

        if pos_data is not None:
            sym = pos_data["symbol"]
            side = pos_data.get("side", "BUY")
            entry = float(pos_data["entry"])
            sl = float(pos_data["sl"])
            tgt = float(pos_data["target"])
            be_trig = float(pos_data.get("be_trigger_px", entry + 0.70 * (tgt - entry)))
            be_done = bool(pos_data.get("be_done", False))
            ltp = float(self._last_ltp.get(self.active_position_key, entry))

            pts = ltp - entry
            pnl_rs = pts * pos_data.get("quantity", settings.LOT_SIZE) - 45.0  # Net after ₹45 statutory fee
            color = "bold green" if pts >= 0 else "bold red"

            # Breakeven Progress Meter
            total_be_move = max(be_trig - entry, 0.1)
            current_move = max(0.0, ltp - entry)
            be_pct = min(100.0, max(0.0, (current_move / total_be_move) * 100.0))
            bar_len = 10
            filled = int((be_pct / 100.0) * bar_len)
            meter = f"[{'█' * filled}{'░' * (bar_len - filled)}] {be_pct:.0f}%"

            if be_done:
                be_status = "[bold green]🔒 LOCKED (+1.0 pt BE Hardened SL)[/bold green]"
            elif ltp >= be_trig:
                be_status = "[bold green]🔒 TRIGGERING BREAKEVEN[/bold green]"
            else:
                be_status = f"[yellow]⏳ {meter} (Trig: ₹{be_trig:.2f})[/yellow]"

            # Target & SL Distances
            dist_to_tp = max(0.0, tgt - ltp)
            dist_to_sl = max(0.0, ltp - sl)

            opened_at = pos_data.get("opened_at", now)
            duration_sec = int((now - opened_at).total_seconds()) if isinstance(opened_at, datetime) else 0
            dur_str = f"{duration_sec // 60:02d}m {duration_sec % 60:02d}s"

            pos_table = Table(
                title=f"[bold green]🎯 ACTIVE TRADE COCKPIT — {sym} [LONG {side}] | Duration: {dur_str}[/bold green]",
                box=box.ROUNDED, expand=True, padding=(0, 1),
            )
            for col in ("ENTRY PRICE", "CURRENT LTP", "LIVE P&L (PTS)", "NET P&L (₹)", "BREAKEVEN (BE) STATUS",
                        "CURRENT SL (DIST)", "TARGET (DIST)", "SIGNAL TRIGGER"):
                pos_table.add_column(col, justify="center")

            pos_table.add_row(
                f"₹{entry:.2f}",
                f"[bold yellow]₹{ltp:.2f}[/bold yellow]",
                f"[{color}]{pts:+.2f} pts[/{color}]",
                f"[{color}]₹{pnl_rs:+,.2f}[/{color}]",
                be_status,
                f"₹{sl:.2f} (-{dist_to_sl:.1f} pts)",
                f"₹{tgt:.2f} (+{dist_to_tp:.1f} pts)",
                pos_data.get("signal", "FLAG / SUPER"),
            )
            tables.append(pos_table)

        # ── 4. Completed Trades Performance Ledger ───────────────────────────
        if self.trades_today:
            tr_table = Table(
                title=f"[bold green]📜 TODAY'S COMPLETED TRADES ({len(self.trades_today)} Trades | {self._wins_today} Wins | WR: {wr:.1f}% | Net: ₹{net_rs:+,.2f})[/bold green]",
                box=box.SIMPLE_HEAD, expand=True, padding=(0, 1),
            )
            for col in ("#", "TIME", "SIGNAL / SR", "CONTRACT", "ENTRY", "EXIT", "OUTCOME", "PTS", "NET P&L (₹)"):
                tr_table.add_column(col, justify="center")
            for i, t in enumerate(self.trades_today, start=1):
                pts = float(t.get("pts", 0.0))
                color = "bold green" if pts >= 0 else "bold red"
                opened = t.get("opened_at")
                t_str = opened.strftime("%H:%M:%S") if isinstance(opened, datetime) else "--"
                reason = t.get("reason", "TARGET" if pts > 0 else "STOP_LOSS")
                tr_table.add_row(
                    str(i), t_str, t.get("signal", "--"), t.get("symbol", "--"),
                    f"₹{t.get('entry', 0.0):.2f}", f"₹{t.get('exit', 0.0):.2f}",
                    f"[{color}]{reason}[/{color}]",
                    f"[{color}]{pts:+.2f}[/{color}]",
                    f"[{color}]₹{float(t.get('rs', 0.0)):+,.2f}[/{color}]",
                )
            tables.append(tr_table)

        return Group(*tables)

    async def run(self):
        await self.initialize()
        await self.recover_open_positions()

        with Live(self.render_dashboard(), console=console, refresh_per_second=1, screen=True) as live:
            while True:
                try:
                    started = time.time()
                    now = ist_now()

                    # 1. Fetch live Nifty spot price
                    if self.client.auth_token:
                        quote = await self.client.get_quotes(exchange="NSE", token="26000")
                        if await self._renew_token_if_expired(quote):
                            pass
                        elif quote.get("stat") == "Ok" and "lp" in quote:
                            try:
                                self.spot_price = float(quote["lp"])
                                self.engine.set_spot_price(self.spot_price)
                                self._broker_status = "[bold green]LIVE CONNECTED[/bold green]"
                            except (ValueError, TypeError):
                                pass

                    # 2. Dynamic Contract Rollover Watch
                    await self.ensure_contracts()

                    # 3. Poll quotes for registered option contracts
                    poll_keys = set(self.engine.contracts.keys())
                    if self.active_position_key:
                        poll_keys.add(self.active_position_key)

                    for key in sorted(poll_keys):
                        cs = self.engine.contracts.get(key)
                        if cs is None:
                            continue
                        oq = await self.client.get_quotes(exchange="NFO", token=cs.token)
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

                    # 4. Check Exits (SL, TP, Breakeven, EOD)
                    await self._manage_exit()

                    # 5. EOD Safety Square-Off at 15:15 IST
                    cur_min = minute_of(now)
                    if cur_min >= 915 and not self._eod_done:
                        self._eod_done = True
                        logger.warning("15:15 IST — triggering EOD safety square-off.")
                        await self._force_eod_square_off()

                    touch_runtime_record()
                    elapsed = time.time() - started
                    await asyncio.sleep(max(0.0, 1.0 - elapsed))

                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    await asyncio.sleep(2.0)


async def main():
    live_mode = "--live" in sys.argv
    engine = LastHopeTradingEngine(live_orders=live_mode)
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
