"""Guarded live trade execution and position lifecycle management."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from flattrade_bot.broker.client import FlattradeClient
from flattrade_bot.broker.history import SessionExpiredError
from flattrade_bot.config import settings
from flattrade_bot.risk.manager import RiskManager
from flattrade_bot.utils.discord import DiscordNotifier


class TradeExecutor:
    """Coordinates risk checks, broker orders, exits, and local position state."""

    def __init__(
        self,
        client: FlattradeClient,
        risk: RiskManager,
        notifier: DiscordNotifier,
        quantity: int = settings.LOT_SIZE,
        live_orders: bool = False,
    ):
        self.client = client
        self.risk = risk
        self.notifier = notifier
        self.quantity = quantity
        self.live_orders = live_orders
        if hasattr(self.risk, "set_quantity"):
            self.risk.set_quantity(self.quantity)
        self.position: Optional[Dict[str, Any]] = None
        self._last_exit_attempt_at: Optional[datetime] = None
        self.last_reconcile_conflict = False

    @staticmethod
    def _accepted(response: Dict[str, Any]) -> bool:
        return str(response.get("stat", "")).lower() == "ok"

    @staticmethod
    def _fill_price(response: Dict[str, Any], fallback: float) -> float:
        for key in ("avgprc", "avg_price", "price", "prc"):
            value = response.get(key)
            if value not in (None, ""):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return fallback

    async def _confirm_fill(
        self, order_id: str, fallback: float, max_attempts: int = 10, is_exit: bool = False
    ) -> Dict[str, Any]:
        """Confirms broker fill status; never treats an unconfirmed accepted request as a fill."""
        for attempt in range(max_attempts):
            try:
                order_book = await self.client.get_order_book()
                records = order_book if isinstance(order_book, list) else order_book.get("orders", [])
                record = next(
                    (item for item in records if str(item.get("norenordno", "")).strip() == str(order_id).strip()),
                    None,
                )
                if record:
                    status = str(record.get("status", "")).strip().upper()
                    rejection = str(record.get("rejreason", "")).strip()
                    if status in {"REJECT", "REJECTED", "CANCEL", "CANCELLED"}:
                        return {"filled": False, "reason": rejection or status, "record": record}

                    try:
                        fillshares = float(record.get("fillshares", 0) or 0)
                    except (TypeError, ValueError):
                        fillshares = 0.0
                    if status in {"COMPLETE", "FILLED", "TRADED"} or fillshares > 0:
                        return {
                            "filled": True,
                            "price": self._fill_price(record, fallback),
                            "record": record,
                        }
                    if rejection:
                        return {"filled": False, "reason": rejection, "record": record}
            except Exception as e:
                pass

            if attempt < max_attempts - 1:
                await asyncio.sleep(0.3)

        # For exits, check position book directly: if broker shows 0 qty, trade did fill!
        if is_exit:
            is_open = await self.reconcile_broker_position()
            if is_open is False:
                return {"filled": True, "price": fallback, "record": None}

        cancel = getattr(self.client, "cancel_order", None)
        if cancel and not is_exit:
            await cancel(order_id)
        return {"filled": False, "reason": "Order fill not confirmed within timeout", "record": None}

    async def reconcile_broker_position(self) -> Optional[bool]:
        """Returns whether the broker still has the locally tracked position open."""
        self.last_reconcile_conflict = False
        if not self.position:
            return None

        get_positions = getattr(self.client, "get_positions", None)
        if get_positions is None:
            return None
        try:
            payload = await get_positions()
        except SessionExpiredError:
            raise
        except Exception:
            return None

        if isinstance(payload, dict) and payload.get("stat") != "Ok":
            return None
        records = payload if isinstance(payload, list) else payload.get("positions", [])
        tracked_token = str(self.position.get("token", ""))
        tracked_symbol = str(self.position.get("order_symbol", ""))

        for record in records:
            if (
                str(record.get("token", "")) != tracked_token
                and str(record.get("tsym", "")) != tracked_symbol
            ):
                continue
            try:
                net_quantity = float(record.get("netqty", 0) or 0)
            except (TypeError, ValueError):
                return None
            if net_quantity <= 0:
                return False
            self.position["quantity"] = int(net_quantity)
            try:
                broker_ltp = float(record.get("lp", 0) or 0)
            except (TypeError, ValueError):
                broker_ltp = 0.0
            if broker_ltp > 0:
                self.position["ltp"] = broker_ltp
            return True

        for record in records:
            try:
                if float(record.get("netqty", 0) or 0) > 0:
                    self.last_reconcile_conflict = True
                    return None
            except (TypeError, ValueError):
                continue
        return False

    async def open_trade(
        self,
        *,
        side: str,
        order_symbol: str,
        display_symbol: str,
        token: str,
        timeframe: str,
        signal: str,
        entry_price: float,
        sl_points: float,
        tp_points: float,
        current_min: int,
        opened_at: datetime,
        reverse: bool = False,
        sl_level: Optional[float] = None,
        tp_level: Optional[float] = None,
        price_rise: bool = True,
        monitor_token: str = "",
        monitor_exchange: str = "",
        be_trigger_px: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Attempts one long option entry after risk and live-mode checks.

        Fib-level mode (B17): when ``sl_level``/``tp_level`` are absolute
        prices, exits are evaluated on those levels with ``price_rise``
        direction on the ``monitor_token`` quote instead of entry-relative
        SL/TP points.
        """
        if self.position:
            return {"accepted": False, "reason": "Position already open"}
        if not self.live_orders:
            return {"accepted": False, "reason": "Live orders disabled"}
        if not order_symbol or not token or entry_price <= 0 or self.quantity <= 0:
            return {"accepted": False, "reason": "Invalid order details"}

        can_trade, reason = self.risk.can_open_trade(current_min, 0)
        if not can_trade:
            return {"accepted": False, "reason": reason}

        response = await self.client.place_market_order(
            symbol=order_symbol,
            side="BUY",
            quantity=self.quantity,
            ltp=entry_price,
            product="MIS",
            slippage_buffer=3.0,  # 3.0 pt aggressive buffer guarantees instant fill across the spread
        )
        if not self._accepted(response):
            return {
                "accepted": False,
                "reason": response.get("emsg", "Broker rejected entry order"),
                "response": response,
            }

        order_id = response.get("norenordno")
        if not order_id:
            return {"accepted": False, "reason": "Broker returned no order ID", "response": response}
        fill = await self._confirm_fill(order_id, entry_price)
        if not fill["filled"]:
            return {
                "accepted": False,
                "reason": fill["reason"],
                "response": response,
                "order": fill.get("record"),
            }

        fill_price = fill["price"]
        self.position = {
            "side": side,
            "symbol": display_symbol,
            "order_symbol": order_symbol,
            "token": token,
            "quantity": self.quantity,
            "timeframe": timeframe,
            "signal": signal,
            "reverse": reverse,
            "entry": fill_price,
            "ltp": fill_price,
            "sl": round(fill_price - sl_points, 2),
            "target": round(fill_price + tp_points, 2),
            "sl_points": sl_points,
            "tp_points": tp_points,
            "sl_level": sl_level,
            "tp_level": tp_level,
            "price_rise": price_rise,
            "monitor_token": monitor_token,
            "monitor_exchange": monitor_exchange,
            "order_id": order_id,
            "opened_at": opened_at,
        }
        # Fire-and-forget: never block the tick loop on a 10s webhook timeout
        asyncio.create_task(
            self.notifier.notify_trade_open({
                **self.position,
                "tgt": self.position["target"],
                "lot_size": self.quantity,
                "reason": signal,
            })
        )
        return {"accepted": True, "response": response, "position": self.position}

    async def check_exit(
        self, ltp: float, now: datetime, order_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Closes the open long position on SL, target, or the 15:00 cutoff.

        B17 fib-level positions exit on absolute ``sl_level``/``tp_level``
        with ``price_rise`` direction; fallback positions use the
        entry-relative ``sl``/``target``.

        ``ltp`` is the monitored quote used for the exit decision. For
        index-monitor positions the monitored quote is the index, so the
        closing order must be priced at the option's own LTP — pass it as
        ``order_price``; otherwise ``ltp`` is used for both.
        """
        if not self.position:
            return {"accepted": False, "reason": "No open position"}

        self.position["ltp"] = ltp
        now_min = now.hour * 60 + now.minute
        pending = self.position.get("pending_exit")
        if pending:
            if now_min > int(pending["touch_minute"]):
                result = await self.close_position(
                    ltp, now, pending["reason"], order_price=order_price
                )
                if result.get("accepted") and self.position is not None:
                    self.position.pop("pending_exit", None)
                return result
            return {"accepted": False, "reason": "Exit pending bar close"}

        exit_reason = None
        sl_level = self.position.get("sl_level")
        tp_level = self.position.get("tp_level")
        if sl_level is not None and tp_level is not None:
            if self.position.get("price_rise", True):
                if ltp <= float(sl_level):
                    exit_reason = "STOP_LOSS"
                elif ltp >= float(tp_level):
                    exit_reason = "TARGET"
            else:
                if ltp >= float(sl_level):
                    exit_reason = "STOP_LOSS"
                elif ltp <= float(tp_level):
                    exit_reason = "TARGET"
            if exit_reason:
                if now_min >= 15 * 60:
                    return await self.close_position(ltp, now, "EOD", order_price=order_price)
                self.position["pending_exit"] = {
                    "reason": exit_reason,
                    "touch_minute": now_min,
                }
                return {
                    "accepted": False,
                    "reason": f"Level touched ({exit_reason}); staging bar-close exit",
                }
        else:
            if ltp <= self.position["sl"]:
                exit_reason = "STOP_LOSS"
            elif ltp >= self.position["target"]:
                exit_reason = "TARGET"

        if not exit_reason and now_min >= 15 * 60:
            exit_reason = "EOD"

        if not exit_reason:
            return {"accepted": False, "reason": "Position remains open"}

        return await self.close_position(ltp, now, exit_reason, order_price=order_price)

    async def close_position(
        self, ltp: float, now: datetime, reason: str, order_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Closes and confirms the current position for an explicit shutdown reason."""
        if not self.position:
            return {"accepted": False, "reason": "No open position"}
        if self._last_exit_attempt_at is not None:
            if now - self._last_exit_attempt_at < timedelta(seconds=10):
                return {"accepted": False, "reason": "Exit retry backoff active"}
        self._last_exit_attempt_at = now

        response = await self.client.place_market_order(
            symbol=self.position["order_symbol"],
            side="SELL",
            quantity=self.position["quantity"],
            ltp=order_price if order_price is not None else ltp,
            product="MIS",
            slippage_buffer=5.0,  # Generous buffer guarantees instant fill across the bid-ask spread
        )
        if not self._accepted(response):
            self._last_exit_attempt_at = None  # Allow immediate retry on error
            return {
                "accepted": False,
                "reason": response.get("emsg", "Broker rejected exit order"),
                "response": response,
            }

        order_id = response.get("norenordno")
        if not order_id:
            self._last_exit_attempt_at = None
            return {"accepted": False, "reason": "Broker returned no exit order ID", "response": response}

        fill = await self._confirm_fill(
            order_id, order_price if order_price is not None else ltp, max_attempts=12, is_exit=True
        )
        if not fill["filled"]:
            self._last_exit_attempt_at = None  # Allow immediate retry
            return {
                "accepted": False,
                "reason": fill["reason"],
                "response": response,
                "order": fill.get("record"),
            }

        exit_price = fill["price"]
        points = round(exit_price - self.position["entry"], 2)
        pnl_rs = round(points * self.position["quantity"], 2)
        opened_at = self.position["opened_at"]
        duration_min = max(0, int((now - opened_at).total_seconds() // 60))
        close_info = {
            **self.position,
            "exit": exit_price,
            "pts": points,
            "rs": pnl_rs,
            "duration_min": duration_min,
            "reason": reason,
        }
        self.risk.record_trade_result(pnl_rs)
        # Fire-and-forget: never block the tick loop on a 10s webhook timeout
        asyncio.create_task(self.notifier.notify_trade_close(close_info))
        self.position = None
        self._last_exit_attempt_at = None
        return {"accepted": True, "response": response, "trade": close_info}
