"""Pocket Money Strategy Bot — Main Entry Point & Compact Dashboard.

Strategy: 💰 Pocket Money — Nifty 50 options intraday scalper on true 10-second bars.
Triggers: FLAG (S1%D <= 20.5 while S4%D >= 79.5) / SUPER (S1 crosses > 20 + bullish trough divergence).
Strikes:  2nd ITM only (CE = ATM - 100, PE = ATM + 100); rollover watch pair ATM±50 kept warm.
Filter:   Index 5m Heikin-Ashi UT Bot + LinReg white-line side gate (CE-only / PE-only).
Exits:    SL = entry - 7 premium pts, TP = entry + 7 (SL priority), EOD flat at 15:00.
Session:  Entries 09:20–14:59 IST; 4 consecutive losses block the day.
Docs:     POCKET_MONEY_STRATEGY.md (verified congruent with artifacts/f6_hybrid/pocket_money_backtest.py).
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
from flattrade_bot.strategies.pocket_money import (
    BAR_SECONDS,
    POLL_SECONDS,
    SESSION_END_MIN,
    SESSION_START_MIN,
    SL_POINTS,
    TP_POINTS,
    PocketMoneyEngine,
)
from flattrade_bot.utils.discord import DiscordNotifier

STRATEGY_LABEL = "Pocket Money Strategy"
IST = timezone(timedelta(hours=5, minutes=30))

log_dir = ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "pocket_money_bot.log"

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[file_handler])
logger = logging.getLogger("flattrade_bot.pocket_money_main")
console = Console(legacy_windows=False)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def ist_now() -> datetime:
    return datetime.now(IST)


def minute_of(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


class PocketMoneyTradingEngine:
    """Live trading engine for the Pocket Money 10s strategy."""

    PM_CONSECUTIVE_LOSS_LIMIT = 4  # POCKET_MONEY_STRATEGY.md §4 (no daily ₹ cap)

    def __init__(self, live_orders: bool = False):
        self.live_orders = live_orders
        self.engine = PocketMoneyEngine(
            poll_seconds=POLL_SECONDS,
            bar_seconds=BAR_SECONDS,
            sl_points=SL_POINTS,
            tp_points=TP_POINTS,
        )
        self.discord = DiscordNotifier(strategy=STRATEGY_LABEL)
        self.client = FlattradeClient()
        self.history = FlattradeHistoryFetcher()
        self.risk = RiskManager(
            max_daily_loss_points=math.inf,  # backtest parity: no daily ₹ cap
            quantity=settings.LOT_SIZE,
            consecutive_loss_limit=self.PM_CONSECUTIVE_LOSS_LIMIT,
        )
        self.executor = (
            TradeExecutor(self.client, self.risk, self.discord, quantity=settings.LOT_SIZE, live_orders=True)
            if live_orders
            else None
        )

        self.spot_price: Optional[float] = None
        self.pos_key: Optional[str] = None
        self.paper_position: Optional[Dict[str, Any]] = None  # SIM mode only
        self.trades_today: List[Dict[str, Any]] = []
        self._wins_today = 0
        self._broker_status = "INITIALIZING..."
        self._last_ltp: Dict[str, float] = {}
        self._last_contract_check = 0.0
        self._last_token_renew = 0.0
        self._eod_done = False
        self._summary_sent = False

    # ── Auth & warmup ────────────────────────────────────────────────────

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

        now = ist_now()
        today = now.strftime("%d-%m-%Y")
        self.engine.set_today(today)

        if self.history.auth_token:
            try:
                rows = await self.history.fetch_historical_candles(
                    token="26000", exchange="NSE", interval="1",
                    days_back=12 + 2,  # 12 warm days + weekend buffer
                )
                if rows:
                    n = self.engine.seed_spot_1m(rows, today=today, now=now.replace(tzinfo=None))
                    logger.info("Index filter seeded with %d spot 1m rows.", n)
            except Exception as e:
                logger.warning(f"Spot filter warmup failed: {e}")

            try:
                q = await self.client.get_quotes(exchange="NSE", token="26000")
                if q.get("stat") == "Ok" and "lp" in q:
                    self.spot_price = float(q["lp"])
            except Exception as e:
                logger.warning(f"Initial spot quote failed: {e}")

            await self.ensure_contracts(force=True)

        asyncio.create_task(
            self.discord._post_embed({
                "title": "🟢 FLATTRADE POCKET MONEY BOT ONLINE",
                "color": 0x2ECC71,
                "fields": [
                    {"name": "Strategy", "value": "Pocket Money (FLAG/SUPER, 10s bars)", "inline": True},
                    {"name": "Session", "value": "09:20 - 15:00 IST", "inline": True},
                    {"name": "SL / TP", "value": "±7.0 premium pts (SL priority)", "inline": True},
                    {"name": "Mode", "value": "🔴 LIVE ORDERS" if self.live_orders else "🟣 PAPER / SIM", "inline": True},
                    {"name": "Nifty Spot", "value": f"₹{self.spot_price:,.2f}" if self.spot_price else "--", "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "Flattrade Pocket Money Bot"},
            })
        )

    @staticmethod
    def _drop_future_rows(rows: List[Dict[str, Any]], now: datetime) -> List[Dict[str, Any]]:
        """Drops today's placeholder rows at/after the current IST minute."""
        today = now.strftime("%d-%m-%Y")
        cur = minute_of(now)
        out = []
        for row in rows:
            try:
                dt = datetime.strptime(str(row.get("time", "")), "%d-%m-%Y %H:%M:%S")
            except (TypeError, ValueError):
                out.append(row)
                continue
            if dt.strftime("%d-%m-%Y") == today and (dt.hour * 60 + dt.minute) >= cur:
                continue
            out.append(row)
        return out

    async def ensure_contracts(self, force: bool = False):
        """Tracks the spec pair + rollover watch pair for the current ATM band."""
        if not self.history.auth_token or self.spot_price is None:
            return
        if not force and (time.time() - self._last_contract_check) < 3.0:
            return
        self._last_contract_check = time.time()

        desired = self.engine.update_spec_keys(self.spot_price)
        for key in sorted(desired):
            if key in self.engine.contracts:
                continue
            side, strike_txt = key.split(":")
            strike = int(strike_txt)
            try:
                scrip = await self.history.search_option_token(f"NIFTY {strike} {side}")
                if not scrip or not scrip.get("token"):
                    logger.warning("No scrip found for %s", key)
                    continue
                seed_rows: List[Dict[str, Any]] = []
                try:
                    seed_rows = await self.history.fetch_historical_candles(
                        token=scrip["token"], exchange="NFO", interval="1", days_back=2
                    )
                    seed_rows = self._drop_future_rows(seed_rows, ist_now())
                except Exception as e:
                    logger.warning(f"Warmup fetch failed for {key}: {e}")
                self.engine.add_contract(side, strike, scrip["tsym"], scrip["token"], seed_rows=seed_rows)
            except Exception as e:
                logger.warning(f"Contract resolution failed for {key}: {e}")

    # ── Position handling ────────────────────────────────────────────────

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

        cs = self.engine.contracts.get(sig["key"])
        if cs is None:
            return
        display = f"NIFTY {sig['strike']} {sig['side']}"
        trigger = str(sig.get("signal", "")).upper()

        if self.live_orders and self.executor is not None:
            res = await self.executor.open_trade(
                side="LONG",
                order_symbol=cs.tsym,
                display_symbol=display,
                token=cs.token,
                timeframe=f"{BAR_SECONDS}s",
                signal=trigger,
                entry_price=sig["price"],
                sl_points=self.engine.sl_points,
                tp_points=self.engine.tp_points,
                current_min=cur_min,
                opened_at=now,
            )
            if not res.get("accepted"):
                logger.warning("Entry rejected: %s", res.get("reason"))
                return
            fill = float(res["position"]["entry"])
            sl = float(res["position"]["sl"])
            tp = float(res["position"]["target"])
        else:
            fill = sig["price"]
            sl = sig["sl"]
            tp = sig["tp"]
            self.paper_position = {
                "symbol": display,
                "order_symbol": cs.tsym,
                "token": cs.token,
                "quantity": settings.LOT_SIZE,
                "signal": trigger,
                "entry": fill,
                "sl": sl,
                "target": tp,
                "opened_at": now,
            }

        self.engine.on_position_opened()
        self.pos_key = sig["key"]
        logger.info("🚨 %s ENTRY %s @ %.2f SL=%.2f TP=%.2f (%s)", trigger, display, fill, sl, tp,
                    "LIVE" if self.live_orders else "PAPER")
        asyncio.create_task(
            self.discord.send_trade_alert(
                strategy=STRATEGY_LABEL,
                direction="LONG",
                symbol=display,
                entry_price=fill,
                sl_price=sl,
                tp_price=tp,
                notes=f"{trigger} trigger | 10s bar close | filter={sig.get('allowed_side')}",
                lot_size=settings.LOT_SIZE,
                mode="LIVE" if self.live_orders else "PAPER",
            )
        )

    async def _record_close(self, trade: Dict[str, Any]):
        self.trades_today.append(trade)
        if float(trade.get("pts", 0.0)) > 0:
            self._wins_today += 1
        self.engine.on_position_closed()
        self.pos_key = None
        self.paper_position = None

    async def _manage_exit(self):
        if not self._has_position() or self.pos_key is None:
            return
        ltp = self._last_ltp.get(self.pos_key)
        if ltp is None:
            return
        now = ist_now()

        if self.live_orders and self.executor is not None:
            res = await self.executor.check_exit(ltp, now)
            if res.get("accepted"):
                await self._record_close(res.get("trade", {}))
                logger.info("Position closed: %s", res.get("trade", {}).get("reason"))
            return

        pos = self.paper_position
        if pos is None:
            return
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
        await self._record_close({
            **pos,
            "exit": ltp,
            "pts": pts,
            "rs": pnl_rs,
            "duration_min": duration_min,
            "reason": reason,
        })
        asyncio.create_task(
            self.discord.notify_trade_close({
                "symbol": pos["symbol"],
                "reason": reason,
                "pts": pts,
                "rs": pnl_rs,
                "entry": pos["entry"],
                "exit": ltp,
                "duration_min": duration_min,
            })
        )

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
                    logger.warning("Recovering orphaned position %s qty=%d", tsym, netqty)
                    res = await self.client.place_market_order(
                        symbol=tsym, side=side, quantity=qty,
                        ltp=max(ltp, 1.0), product="MIS", slippage_buffer=3.0,
                    )
                    if res.get("stat") == "Ok":
                        logger.info("Orphaned position %s squared off.", tsym)
                    else:
                        logger.error("Failed to square off %s: %s", tsym, res.get("emsg"))
        except Exception as e:
            logger.error(f"Position recovery check failed: {e}")

    async def _force_eod_square_off(self):
        """Safety net at 15:15 IST: flatten anything still open."""
        if self.live_orders and self.executor is not None and self.executor.position:
            pos = self.executor.position
            try:
                res = await self.client.place_market_order(
                    symbol=pos["order_symbol"], side="SELL",
                    quantity=pos["quantity"], ltp=max(self._last_ltp.get(self.pos_key, pos["entry"]), 1.0),
                    product="MIS", slippage_buffer=3.0,
                )
                logger.info("EOD square-off order for %s: %s", pos["order_symbol"], res.get("stat"))
            except Exception as e:
                logger.error(f"EOD square-off error: {e}")
            await self._record_close({**pos, "exit": self._last_ltp.get(self.pos_key, pos["entry"]),
                                      "pts": 0.0, "rs": 0.0, "duration_min": 0, "reason": "EOD_FORCE"})
        elif self.paper_position is not None:
            pos = self.paper_position
            ltp = self._last_ltp.get(self.pos_key, pos["entry"])
            pts = round(ltp - pos["entry"], 2)
            pnl_rs = round(pts * pos["quantity"], 2)
            self.risk.record_trade_result(pnl_rs)
            await self._record_close({
                **pos, "exit": ltp, "pts": pts, "rs": pnl_rs,
                "duration_min": 0, "reason": "EOD_FORCE",
            })
        await self.recover_open_positions()

    async def _send_eod_summary(self):
        net_rs = sum(float(t.get("rs", 0.0)) for t in self.trades_today)
        total = len(self.trades_today)
        wr = (100.0 * self._wins_today / total) if total else 0.0
        asyncio.create_task(
            self.discord._post_embed({
                "title": "🏁 POCKET MONEY — EOD SUMMARY",
                "color": 0x2ECC71 if net_rs >= 0 else 0xE74C3C,
                "fields": [
                    {"name": "Trades", "value": str(total), "inline": True},
                    {"name": "Win Rate", "value": f"{wr:.1f}%", "inline": True},
                    {"name": "Net P&L", "value": f"₹{net_rs:+,.2f}", "inline": True},
                    {"name": "Consec. Losses", "value": str(self.risk.state.consecutive_losses), "inline": True},
                ],
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "Flattrade Pocket Money Bot"},
            })
        )

    # ── Token renewal ────────────────────────────────────────────────────

    async def _renew_token_if_expired(self, response: Any) -> bool:
        if not is_session_expired_response(response):
            return False
        now_ts = time.time()
        if now_ts - getattr(self, "_last_token_renew", 0.0) < 300:
            return True
        self._last_token_renew = now_ts
        logger.warning("Session expired in live loop. Renewing token via lightweight REST API...")
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

    # ── Dashboard ────────────────────────────────────────────────────────

    def render_dashboard(self) -> Group:
        now = ist_now()
        time_str = now.strftime("%H:%M:%S")
        header = Text.from_markup(
            f" [bold bright_yellow]💰 POCKET MONEY STRATEGY[/bold bright_yellow] "
            f"| [bold white]FLAG/SUPER · 10s bars · 2nd ITM · ±7 pts[/bold white] "
            f"| [cyan]{time_str}[/cyan]"
        )
        banner = Panel(Align.center(header), box=box.ROUNDED, style="bright_blue", padding=(0, 1))

        summary = self.engine.get_summary()
        allowed = summary.get("allowed_side") or "--"
        ut = (summary.get("ut_color") or "--").upper()
        ut_str = f"[bold green]{ut}[/bold green]" if ut == "GREEN" else (
            f"[bold red]{ut}[/bold red]" if ut == "RED" else ut)
        sess_active = SESSION_START_MIN <= minute_of(now) < SESSION_END_MIN
        mode_str = "[bold white on red] LIVE [/bold white on red]" if self.live_orders else "[bold white on blue] PAPER [/bold white on blue]"
        net_rs = sum(float(t.get("rs", 0.0)) for t in self.trades_today)
        pnl_color = "bold green" if net_rs >= 0 else "bold red"

        sys_table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
        for col, style in (("BROKER", "bold green"), ("MODE", "bold white"), ("SESSION", "bold white"),
                           ("NIFTY SPOT", "bold yellow"), ("FILTER", "bold white"), ("UT BOT", "bold white"),
                           ("ATM", "bold white"), ("TRADES", "bold white"), ("DAY P&L", "bold white")):
            sys_table.add_column(col, style=style, justify="center")
        sys_table.add_row(
            self._broker_status,
            mode_str,
            "[bold green]ACTIVE[/bold green]" if sess_active else "[yellow]CLOSED[/yellow]",
            f"₹{self.spot_price:,.2f}" if self.spot_price else "--",
            f"[bold green]{allowed}[/bold green]" if allowed in ("CE", "PE") else "[dim]NO ENTRY[/dim]",
            ut_str,
            str(summary.get("atm") or "--"),
            f"{len(self.trades_today)} ({self._wins_today}W/{len(self.trades_today) - self._wins_today}L)",
            f"[{pnl_color}]₹{net_rs:+,.2f}[/{pnl_color}]",
        )

        mon_table = Table(
            title="[bold cyan]OPTION CHART SETUP MONITOR (S1 9,3 · S2 14,3 · S3 40,4 · S4 60,10)[/bold cyan]",
            box=box.SIMPLE_HEAD, expand=True, padding=(0, 1),
        )
        for col, style in (("CONTRACT", "bold white"), ("SPEC", "bold white"), ("STATE", "bold yellow"),
                           ("S1 %D", "bold white"), ("S2 %D", "bold white"), ("S3 %D", "bold white"),
                           ("S4 %D", "bold white"), ("LTP", "bold yellow")):
            mon_table.add_column(col, style=style, justify="center")

        for m in self.engine.setup_monitor():
            state = m.get("state", "?")
            state_str = {
                "READY": "[bold green]READY[/bold green]",
                "FLAG FORMING": "[bold yellow]FLAG FORMING[/bold yellow]",
                "SUPER FORMING": "[bold yellow]SUPER FORMING[/bold yellow]",
                "WARMING": "[dim]WARMING[/dim]",
            }.get(state, f"[dim]{state}[/dim]")
            fmt = lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "--"
            ltp = self._last_ltp.get(m["key"])
            mon_table.add_row(
                f"{m['tsym']}",
                "[bold green]SPEC[/bold green]" if m.get("spec") else "[dim]watch[/dim]",
                state_str,
                fmt(m.get("s1")), fmt(m.get("s2")), fmt(m.get("s3")), fmt(m.get("s4")),
                f"{ltp:.2f}" if ltp else "--",
            )

        tables: List[Any] = [banner, sys_table, mon_table]

        pos_row = None
        if self.live_orders and self.executor is not None and self.executor.position:
            p = self.executor.position
            pos_row = (p["symbol"], p["entry"], p["sl"], p["target"], self._last_ltp.get(self.pos_key))
        elif self.paper_position is not None:
            p = self.paper_position
            pos_row = (p["symbol"], p["entry"], p["sl"], p["target"], self._last_ltp.get(self.pos_key))
        if pos_row is not None:
            sym, entry, sl, tgt, ltp = pos_row
            pts = (ltp - entry) if ltp is not None else 0.0
            color = "bold green" if pts >= 0 else "bold red"
            pos_table = Table(title="[bold green]ACTIVE POSITION[/bold green]", box=box.SIMPLE_HEAD,
                              expand=True, padding=(0, 1))
            for col in ("SYMBOL", "ENTRY", "SL", "TARGET", "LTP", "P&L PTS"):
                pos_table.add_column(col, justify="center")
            pos_table.add_row(sym, f"₹{entry:.2f}", f"₹{sl:.2f}", f"₹{tgt:.2f}",
                              f"₹{ltp:.2f}" if ltp else "--", f"[{color}]{pts:+.2f}[/{color}]")
            tables.append(pos_table)

        if self.trades_today:
            tr_table = Table(
                title=f"[bold green]COMPLETED TRADES ({len(self.trades_today)} | {self._wins_today} WINS)[/bold green]",
                box=box.SIMPLE_HEAD, expand=True, padding=(0, 1),
            )
            for col in ("#", "SIGNAL", "CONTRACT", "ENTRY", "EXIT", "PTS", "NET ₹"):
                tr_table.add_column(col, justify="center")
            for i, t in enumerate(self.trades_today, start=1):
                pts = float(t.get("pts", 0.0))
                color = "bold green" if pts >= 0 else "bold red"
                tr_table.add_row(
                    str(i), t.get("signal", "--"), t.get("symbol", "--"),
                    f"₹{t.get('entry', 0.0):.2f}", f"₹{t.get('exit', 0.0):.2f}",
                    f"[{color}]{pts:+.2f}[/{color}]",
                    f"[{color}]₹{float(t.get('rs', 0.0)):+,.2f}[/{color}]",
                )
            tables.append(tr_table)

        return Group(*tables)

    # ── Main loop ────────────────────────────────────────────────────────

    async def run(self):
        await self.initialize()
        await self.recover_open_positions()

        with Live(self.render_dashboard(), console=console, refresh_per_second=1, screen=True) as live:
            while True:
                try:
                    started = time.time()
                    now = ist_now()

                    if self.client.auth_token:
                        quote = await self.client.get_quotes(exchange="NSE", token="26000")
                        if await self._renew_token_if_expired(quote):
                            pass
                        elif quote.get("stat") == "Ok" and "lp" in quote:
                            try:
                                self.spot_price = float(quote["lp"])
                                self._broker_status = "[bold green]LIVE CONNECTED[/bold green]"
                                self.engine.push_spot_tick(self.spot_price, now)
                            except (ValueError, TypeError):
                                pass

                    await self.ensure_contracts()

                    poll_keys: Set[str] = set(sorted(self.engine.desired_keys(self.spot_price or 0.0)))
                    if self.pos_key:
                        poll_keys.add(self.pos_key)

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
                        sig = self.engine.push_option_tick(key, opt_ltp, now)
                        if sig and not self._has_position():
                            await self._try_enter(sig)

                    await self._manage_exit()

                    cur_min = minute_of(now)
                    if cur_min >= 915 and not self._eod_done:  # 15:15 safety net
                        self._eod_done = True
                        logger.warning("15:15 IST — triggering EOD safety square-off.")
                        await self._force_eod_square_off()
                    if cur_min >= 905 and not self._summary_sent and self.trades_today:
                        self._summary_sent = True
                        await self._send_eod_summary()

                    touch_runtime_record(
                        path=settings.BOT_RUNTIME_FILE,
                        pid=os.getpid(),
                        extra={
                            "strategy_name": STRATEGY_LABEL,
                            "symbol": "NIFTY",
                            "timeframe": f"{BAR_SECONDS}s",
                            "spot_price": self.spot_price,
                            "filter_side": self.engine.filter.allowed_side(minute_of(now)),
                            "active_position": self._has_position(),
                            "session_active": SESSION_START_MIN <= cur_min < SESSION_END_MIN,
                            "trades_count": len(self.trades_today),
                        },
                    )

                    live.update(self.render_dashboard())
                    elapsed = time.time() - started
                    await asyncio.sleep(max(0.05, POLL_SECONDS - elapsed))

                except asyncio.CancelledError:
                    logger.info("Pocket Money engine stopped cleanly.")
                    break
                except Exception as e:
                    logger.error(f"Error in engine loop: {e}", exc_info=True)
                    await asyncio.sleep(2.0)


async def main():
    live_mode = "--live" in sys.argv
    engine = PocketMoneyTradingEngine(live_orders=live_mode)
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
