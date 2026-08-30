"""Discord Webhook Notification Notifier for Last Hope GPU Winner Strategy Alerts."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from flattrade_bot.config import settings

logger = logging.getLogger(__name__)

STRATEGY_NAME = "Last Hope GPU Winner Strategy"


def _footer_text(strategy: str) -> str:
    return "Flattrade Last Hope Winner Bot"


class DiscordNotifier:
    """Sends rich, instant Discord Webhook embeds for entries, exits, and breakeven locks."""

    def __init__(self, webhook_url: Optional[str] = None, strategy: Optional[str] = None):
        self.webhook_url = webhook_url or settings.DISCORD_WEBHOOK_URL
        self.enabled = bool(self.webhook_url)
        self.strategy = strategy or STRATEGY_NAME

    async def _post_embed(self, embed: Dict[str, Any]):
        if not self.enabled:
            return
        try:
            import httpx
            payload = {"embeds": [embed]}
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(self.webhook_url, json=payload)
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    async def notify_trade_open(self, trade_info: Dict[str, Any], strategy: Optional[str] = None):
        """Sends Discord notification instantly when an order is executed."""
        strategy = strategy or self.strategy
        symbol = trade_info.get("symbol", "NIFTY")
        direction = trade_info.get("side", "BUY")
        level = trade_info.get("level", "S/R Bounce")
        mode = trade_info.get("mode", "LIVE" if settings.LIVE_TRADING else "PAPER")
        mode_str = "🔴 LIVE BROKER" if mode == "LIVE" else "🟣 PAPER SIMULATION"
        be_trig = trade_info.get("be_trigger_px", 0.0)

        fields = [
            {"name": "Strategy", "value": f"🏆 {strategy}", "inline": True},
            {"name": "Option Strike", "value": f"**{symbol}**", "inline": True},
            {"name": "Execution Mode", "value": f"`{mode_str}`", "inline": True},
            {"name": "Trigger Setup", "value": f"`{trade_info.get('signal', level)}`", "inline": True},
            {"name": "Fill Price", "value": f"**₹{trade_info.get('entry', 0.0):.2f}**", "inline": True},
            {"name": "Stop Loss (SL)", "value": f"₹{trade_info.get('sl', 0.0):.2f}", "inline": True},
            {"name": "Target (TP)", "value": f"₹{trade_info.get('tgt', trade_info.get('tp', 0.0)):.2f}", "inline": True},
            {"name": "Breakeven Trigger", "value": f"₹{be_trig:.2f} (+70% move)", "inline": True} if be_trig else {"name": "Lot Size", "value": f"{trade_info.get('lot_size', settings.LOT_SIZE)} qty", "inline": True},
            {"name": "Lot Size", "value": f"{trade_info.get('lot_size', settings.LOT_SIZE)} qty", "inline": True},
        ]

        embed = {
            "title": f"🚀 NEW ENTRY: {symbol} [{direction}]",
            "color": 0x2ECC71,  # Vibrant Green
            "fields": fields,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": _footer_text(strategy)},
        }
        await self._post_embed(embed)

    async def notify_breakeven_locked(self, symbol: str, entry: float, new_sl: float, ltp: float, strategy: Optional[str] = None):
        """Sends instant Discord alert when the Breakeven Stop hardens to Entry + 1.0 pt."""
        strategy = strategy or self.strategy
        embed = {
            "title": f"🔒 BREAKEVEN STOP LOCKED: {symbol}",
            "color": 0x3498DB,  # Vibrant Blue
            "fields": [
                {"name": "Strategy", "value": strategy, "inline": True},
                {"name": "Option Symbol", "value": f"**{symbol}**", "inline": True},
                {"name": "Current LTP", "value": f"**₹{ltp:.2f}**", "inline": True},
                {"name": "Entry Price", "value": f"₹{entry:.2f}", "inline": True},
                {"name": "New Hardened SL", "value": f"**₹{new_sl:.2f} (+1.0 pt locked)**", "inline": True},
                {"name": "Risk Status", "value": "🛡️ **RISK-FREE TRADE (Zero Loss Guaranteed)**", "inline": False},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": _footer_text(strategy)},
        }
        await self._post_embed(embed)

    async def notify_trade_close(self, trade_info: Dict[str, Any], strategy: Optional[str] = None):
        """Sends Discord notification instantly when a trade is closed."""
        strategy = strategy or self.strategy
        pts = float(trade_info.get("pts", 0.0))
        pnl_rs = float(trade_info.get("rs", 0.0))
        is_win = pts > 0
        color = 0x2ECC71 if is_win else 0xE74C3C  # Green vs Red
        reason = trade_info.get("reason", "TARGET" if is_win else "STOP_LOSS")

        outcome_badge = "🟢 WIN (TARGET)" if reason == "TARGET" else ("🛡️ BREAKEVEN" if pts > 0 and reason == "STOP_LOSS" else "🔴 LOSS (STOP LOSS)")

        embed = {
            "title": f"{outcome_badge}: {trade_info.get('symbol', 'NIFTY')}",
            "color": color,
            "fields": [
                {"name": "Strategy", "value": strategy, "inline": True},
                {"name": "Exit Reason", "value": f"**{reason}**", "inline": True},
                {"name": "Net Points", "value": f"**{pts:+.2f} pts**", "inline": True},
                {"name": "Realized Net P&L", "value": f"**₹{pnl_rs:+,.2f}**", "inline": True},
                {"name": "Entry Price", "value": f"₹{trade_info.get('entry', 0.0):.2f}", "inline": True},
                {"name": "Exit Fill Price", "value": f"₹{trade_info.get('exit', 0.0):.2f}", "inline": True},
                {"name": "Duration", "value": f"{trade_info.get('duration_min', 0)} mins", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": _footer_text(strategy)},
        }
        await self._post_embed(embed)

    async def send_trade_alert(
        self,
        strategy: Optional[str] = None,
        direction: str = "LONG",
        symbol: str = "NIFTY",
        entry_price: float = 0.0,
        sl_price: float = 0.0,
        tp_price: float = 0.0,
        notes: str = "",
        lot_size: Optional[int] = None,
        mode: Optional[str] = None,
    ):
        """Helper to send a formatted trade alert."""
        await self.notify_trade_open({
            "symbol": symbol,
            "side": "BUY" if direction == "LONG" else "BUY (PE)",
            "level": notes or "S/R Level",
            "signal": notes or "FLAG / SUPER",
            "entry": entry_price,
            "sl": sl_price,
            "tgt": tp_price,
            "lot_size": lot_size or settings.LOT_SIZE,
            "mode": mode or "--",
        }, strategy=strategy)