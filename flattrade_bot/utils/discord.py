"""Discord Webhook Notification Notifier for Undisputed Rejection Champion & Pocket Money Alerts."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from flattrade_bot.config import settings

logger = logging.getLogger(__name__)

LEGACY_STRATEGY = "Undisputed Rejection Champion"


def _footer_text(strategy: str) -> str:
    if strategy == LEGACY_STRATEGY:
        return "Flattrade Undisputed Rejection Bot"
    return f"Flattrade {strategy.replace(' Strategy', '')} Bot"


class DiscordNotifier:
    """Sends rich Discord Webhook embeds for trading events, level touches, and trailing stops."""

    def __init__(self, webhook_url: Optional[str] = None, strategy: Optional[str] = None):
        self.webhook_url = webhook_url or settings.DISCORD_WEBHOOK_URL
        self.enabled = bool(self.webhook_url)
        self.strategy = strategy or LEGACY_STRATEGY

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

    async def notify_setup_formed(self, setup_info: Dict[str, Any]):
        """Alerts when Bar 1 tests an S/R Level and arms the Two-Bar Confirmation."""
        level_name = setup_info.get("level", "S/R Level")
        score = setup_info.get("score", 50)
        direction = setup_info.get("direction", "LONG")
        
        embed = {
            "title": f"🎯 SETUP FORMED: {level_name} ({direction})",
            "color": 0xF39C12,  # Amber / Warning
            "fields": [
                {"name": "Strategy", "value": "Undisputed Rejection Champion", "inline": True},
                {"name": "Level Tested", "value": level_name, "inline": True},
                {"name": "Direction", "value": "🟢 LONG (CE Bounce)" if direction == "LONG" else "🔴 SHORT (PE Rejection)", "inline": True},
                {"name": "Confluence Score", "value": f"**{score} / 100 pts**", "inline": True},
                {"name": "Bar 1 Extreme", "value": f"High: ₹{setup_info.get('high', 0):.2f} | Low: ₹{setup_info.get('low', 0):.2f}", "inline": True},
                {"name": "Status", "value": "⏳ Waiting for Bar 2 Confirmation Break", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Flattrade Undisputed Rejection Bot"}
        }
        await self._post_embed(embed)

    async def notify_trade_open(self, trade_info: Dict[str, Any], strategy: Optional[str] = None):
        """Sends Discord notification when an order is executed."""
        strategy = strategy or self.strategy
        symbol = trade_info.get("symbol", "NIFTY")
        direction = trade_info.get("side", "BUY")
        level = trade_info.get("level", "S/R Anchor")
        score = trade_info.get("score", 50)
        legacy = strategy == LEGACY_STRATEGY

        embed = {
            "title": f"🚀 TWO-BAR CONFIRMATION TRIGGER: {symbol}" if legacy else f"🚀 {strategy.upper()} ENTRY: {symbol}",
            "color": 0x2ECC71,  # Green
            "fields": [
                {"name": "Strategy", "value": f"🏆 {strategy}" if legacy else strategy, "inline": True},
                {"name": "Option Strike", "value": symbol, "inline": True},
                {"name": "Direction", "value": f"**{direction}**", "inline": True},
                {"name": "S/R Level" if legacy else "Trigger / Notes", "value": level, "inline": True},
                {"name": "Confluence Score", "value": f"**{score} pts**", "inline": True} if legacy else {"name": "Mode", "value": trade_info.get("mode", "--"), "inline": True},
                {"name": "Fill Price", "value": f"₹{trade_info.get('entry', 0.0):.2f}", "inline": True},
                {"name": "Initial Stop Loss", "value": f"₹{trade_info.get('sl', 0.0):.2f}", "inline": True},
                {"name": "Take Profit Target", "value": f"₹{trade_info.get('tgt', 0.0):.2f}", "inline": True},
                {"name": "Lot Size", "value": f"{trade_info.get('lot_size', 65)} qty", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": _footer_text(strategy)},
        }
        await self._post_embed(embed)

    async def notify_trailing_sl_updated(self, trail_info: Dict[str, Any], strategy: Optional[str] = None):
        """Alerts when the trailing stop moves up/down to lock in profit."""
        strategy = strategy or self.strategy

        embed = {
            "title": f"🛡️ TRAILING STOP LOCKED: {trail_info.get('symbol', 'NIFTY')}",
            "color": 0x3498DB,  # Blue
            "fields": [
                {"name": "Strategy", "value": strategy, "inline": True},
                {"name": "Current Gain", "value": f"**{trail_info.get('gain_pts', 0.0):+.2f} pts**", "inline": True},
                {"name": "New Protected SL", "value": f"**₹{trail_info.get('new_sl', 0.0):.2f}**", "inline": True},
                {"name": "Trailing Step", "value": "2.0 pts trailing behind peak", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": _footer_text(strategy)},
        }
        await self._post_embed(embed)

    async def notify_trade_close(self, trade_info: Dict[str, Any], strategy: Optional[str] = None):
        """Sends Discord notification when a trade is closed."""
        strategy = strategy or self.strategy
        pts = trade_info.get("pts", 0.0)
        pnl_rs = trade_info.get("rs", 0.0)
        is_win = pts > 0
        color = 0x2ECC71 if is_win else 0xE74C3C

        embed = {
            "title": f"{'🟢 WIN' if is_win else '🔴 LOSS'} TRADE CLOSED: {trade_info.get('symbol', 'NIFTY')}",
            "color": color,
            "fields": [
                {"name": "Strategy", "value": strategy, "inline": True},
                {"name": "Exit Reason", "value": f"**{trade_info.get('reason', 'EXIT')}**", "inline": True},
                {"name": "Net Points", "value": f"**{pts:+.2f} pts**", "inline": True},
                {"name": "Realized P&L", "value": f"**₹{pnl_rs:+,.2f}**", "inline": True},
                {"name": "Entry Price", "value": f"₹{trade_info.get('entry', 0.0):.2f}", "inline": True},
                {"name": "Exit Price", "value": f"₹{trade_info.get('exit', 0.0):.2f}", "inline": True},
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
            "score": 50,
            "entry": entry_price,
            "sl": sl_price,
            "tgt": tp_price,
            "lot_size": lot_size or settings.LOT_SIZE,
            "mode": mode or "--",
        }, strategy=strategy)