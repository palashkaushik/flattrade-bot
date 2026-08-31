"""Discord Webhook Notification Notifier for Last Hope GPU Winner Strategy Alerts.

Reliable delivery with automatic retry queue:
- Primary POST with 10s timeout
- On failure: queued to in-memory + disk (logs/discord_retry_queue.json)
- Background retry loop retries every 30s (3 attempts per embed, exponential backoff)
- Survives bot restarts: queue persisted to disk
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flattrade_bot.config import settings

logger = logging.getLogger(__name__)

STRATEGY_NAME = "Last Hope GPU Winner Strategy"
RETRY_QUEUE_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "discord_retry_queue.json"
MAX_RETRIES = 3
RETRY_INTERVAL_SEC = 30
POST_TIMEOUT = 10.0


def _footer_text(strategy: str) -> str:
    return "Flattrade Last Hope Winner Bot"


class DiscordNotifier:
    """Sends rich, instant Discord Webhook embeds for entries, exits, and breakeven locks.

    Features automatic retry: failed embeds are queued to memory + disk and retried
    every 30 seconds by a background coroutine started via ``start_retry_loop()``.
    """

    def __init__(self, webhook_url: Optional[str] = None, strategy: Optional[str] = None):
        self.webhook_url = webhook_url or settings.DISCORD_WEBHOOK_URL
        self.enabled = bool(self.webhook_url)
        self.strategy = strategy or STRATEGY_NAME
        self._retry_queue: List[Dict[str, Any]] = []
        self._retry_task: Optional[asyncio.Task] = None
        self._load_queue_from_disk()
        if not self.enabled:
            logger.warning("Discord webhook URL not configured — notifications disabled")

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_queue_from_disk(self):
        """Load pending notifications from disk (survives restarts)."""
        try:
            if RETRY_QUEUE_PATH.exists():
                data = json.loads(RETRY_QUEUE_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._retry_queue = [item for item in data if isinstance(item, dict) and "embed" in item]
                    if self._retry_queue:
                        logger.info(f"Loaded {len(self._retry_queue)} pending Discord notifications from disk")
        except Exception as e:
            logger.warning(f"Could not load Discord retry queue: {e}")
            self._retry_queue = []

    def _save_queue_to_disk(self):
        """Persist retry queue to disk (crash-safe)."""
        try:
            RETRY_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
            RETRY_QUEUE_PATH.write_text(
                json.dumps(self._retry_queue, default=str), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Could not persist Discord retry queue: {e}")

    # ── Core Send ─────────────────────────────────────────────────────────

    async def _post_embed(self, embed: Dict[str, Any], _attempt: int = 0) -> bool:
        """POST an embed to the webhook. Returns True on success."""
        if not self.enabled:
            return False
        try:
            import httpx
            payload = {"embeds": [embed]}
            async with httpx.AsyncClient(timeout=POST_TIMEOUT) as client:
                response = await client.post(self.webhook_url, json=payload)
                if response.status_code == 429:
                    # Rate limited — extract retry_after and wait
                    try:
                        body = response.json()
                        retry_after = body.get("retry_after", 5.0)
                    except Exception:
                        retry_after = 5.0
                    logger.warning(f"Discord rate limited — retrying after {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._post_embed(embed, _attempt=_attempt + 1)
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Discord POST failed (attempt {_attempt + 1}): {e}")
            return False

    async def _send_with_retry(self, embed: Dict[str, Any], label: str = ""):
        """Send embed, queue to retry on failure."""
        success = await self._post_embed(embed)
        if not success:
            entry = {
                "embed": embed,
                "label": label,
                "queued_at": time.time(),
                "attempts": 0,
            }
            self._retry_queue.append(entry)
            self._save_queue_to_disk()
            logger.warning(f"Queued Discord notification for retry: {label}")

    # ── Background Retry Loop ─────────────────────────────────────────────

    async def _retry_loop(self):
        """Background loop: retries queued notifications every 30s."""
        while True:
            await asyncio.sleep(RETRY_INTERVAL_SEC)
            if not self._retry_queue:
                continue
            pending = list(self._retry_queue)
            self._retry_queue.clear()
            for entry in pending:
                embed = entry.get("embed", {})
                attempts = entry.get("attempts", 0) + 1
                if attempts > MAX_RETRIES:
                    logger.error(f"Discord notification failed after {MAX_RETRIES} attempts — dropping: {entry.get('label', '?')}")
                    continue
                entry["attempts"] = attempts
                # Exponential backoff: 5s, 10s, 20s
                backoff = min(5 * (2 ** (attempts - 1)), 20)
                await asyncio.sleep(backoff)
                success = await self._post_embed(embed, _attempt=attempts)
                if not success:
                    self._retry_queue.append(entry)
            if self._retry_queue:
                self._save_queue_to_disk()
            else:
                # Clear disk file if queue is empty
                try:
                    if RETRY_QUEUE_PATH.exists():
                        RETRY_QUEUE_PATH.write_text("[]", encoding="utf-8")
                except Exception:
                    pass

    def start_retry_loop(self):
        """Start the background retry coroutine (call once from the main async loop)."""
        if self._retry_task is None or self._retry_task.done():
            try:
                self._retry_task = asyncio.create_task(self._retry_loop())
                logger.info("Discord retry loop started")
            except RuntimeError:
                pass  # No running event loop yet — will start on first send

    def stop_retry_loop(self):
        """Cancel the retry loop gracefully."""
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
            self._retry_task = None

    # ── Notification Methods ──────────────────────────────────────────────

    async def notify_trade_open(self, trade_info: Dict[str, Any], strategy: Optional[str] = None):
        """Sends Discord notification instantly when an order is executed."""
        strategy = strategy or self.strategy
        symbol = trade_info.get("symbol", "NIFTY")
        direction = trade_info.get("side", "BUY")
        level = trade_info.get("level", "S/R Bounce")
        mode = trade_info.get("mode", "LIVE" if settings.LIVE_TRADING else "PAPER")
        mode_str = "LIVE BROKER" if mode == "LIVE" else "PAPER SIMULATION"
        be_trig = trade_info.get("be_trigger_px", 0.0)

        fields = [
            {"name": "Strategy", "value": f"{strategy}", "inline": True},
            {"name": "Option Strike", "value": f"**{symbol}**", "inline": True},
            {"name": "Execution Mode", "value": f"`{mode_str}`", "inline": True},
            {"name": "Trigger Setup", "value": f"`{trade_info.get('signal', level)}`", "inline": True},
            {"name": "Fill Price", "value": f"**Rs {trade_info.get('entry', 0.0):.2f}**", "inline": True},
            {"name": "Stop Loss (SL)", "value": f"Rs {trade_info.get('sl', 0.0):.2f}", "inline": True},
            {"name": "Target (TP)", "value": f"Rs {trade_info.get('tgt', trade_info.get('tp', 0.0)):.2f}", "inline": True},
            {"name": "Lot Size", "value": f"{trade_info.get('lot_size', settings.LOT_SIZE)} qty", "inline": True},
        ]
        if be_trig:
            fields.append({"name": "Breakeven Trigger", "value": f"Rs {be_trig:.2f} (+70% move)", "inline": True})

        embed = {
            "title": f"NEW ENTRY: {symbol} [{direction}]",
            "color": 0x2ECC71,
            "fields": fields,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": _footer_text(strategy)},
        }
        await self._send_with_retry(embed, label=f"ENTRY {symbol}")

    async def notify_breakeven_locked(self, symbol: str, entry: float, new_sl: float, ltp: float, strategy: Optional[str] = None):
        """Sends instant Discord alert when the Breakeven Stop hardens to Entry + 1.0 pt."""
        strategy = strategy or self.strategy
        embed = {
            "title": f"BREAKEVEN STOP LOCKED: {symbol}",
            "color": 0x3498DB,
            "fields": [
                {"name": "Strategy", "value": strategy, "inline": True},
                {"name": "Option Symbol", "value": f"**{symbol}**", "inline": True},
                {"name": "Current LTP", "value": f"**Rs {ltp:.2f}**", "inline": True},
                {"name": "Entry Price", "value": f"Rs {entry:.2f}", "inline": True},
                {"name": "New Hardened SL", "value": f"**Rs {new_sl:.2f} (+1.0 pt locked)**", "inline": True},
                {"name": "Risk Status", "value": "RISK-FREE TRADE (Zero Loss Guaranteed)", "inline": False},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": _footer_text(strategy)},
        }
        await self._send_with_retry(embed, label=f"BE LOCK {symbol}")

    async def notify_trade_close(self, trade_info: Dict[str, Any], strategy: Optional[str] = None):
        """Sends Discord notification instantly when a trade is closed."""
        strategy = strategy or self.strategy
        pts = float(trade_info.get("pts", 0.0))
        pnl_rs = float(trade_info.get("rs", 0.0))
        is_win = pts > 0
        color = 0x2ECC71 if is_win else 0xE74C3C
        reason = trade_info.get("reason", "TARGET" if is_win else "STOP_LOSS")

        if reason == "TARGET":
            outcome_badge = "WIN (TARGET)"
        elif pts > 0 and reason == "STOP_LOSS":
            outcome_badge = "BREAKEVEN"
        else:
            outcome_badge = "LOSS (STOP LOSS)"

        embed = {
            "title": f"{outcome_badge}: {trade_info.get('symbol', 'NIFTY')}",
            "color": color,
            "fields": [
                {"name": "Strategy", "value": strategy, "inline": True},
                {"name": "Exit Reason", "value": f"**{reason}**", "inline": True},
                {"name": "Net Points", "value": f"**{pts:+.2f} pts**", "inline": True},
                {"name": "Realized Net P&L", "value": f"**Rs {pnl_rs:+,.2f}**", "inline": True},
                {"name": "Entry Price", "value": f"Rs {trade_info.get('entry', 0.0):.2f}", "inline": True},
                {"name": "Exit Fill Price", "value": f"Rs {trade_info.get('exit', 0.0):.2f}", "inline": True},
                {"name": "Duration", "value": f"{trade_info.get('duration_min', 0)} mins", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": _footer_text(strategy)},
        }
        await self._send_with_retry(embed, label=f"CLOSE {trade_info.get('symbol', '?')}")

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
