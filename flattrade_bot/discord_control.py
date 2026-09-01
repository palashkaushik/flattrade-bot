"""Discord Gateway control plane for the Last Hope trading process.

On the VPS the bot lifecycle is owned by systemd (flattrade-bot.service):
Discord commands start/stop/restart via systemctl, and status is read from
the bot.runtime.json heartbeat the live bot writes every tick.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from flattrade_bot.config import settings
from flattrade_bot.control import (
    is_control_authorized,
    parse_id_list,
    read_runtime_record,
)

logger = logging.getLogger("flattrade_bot.discord_control")

SYSTEMD_UNIT = "flattrade-bot.service"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYSTEMD_AVAILABLE = Path("/etc/systemd/system").is_dir() and Path("/usr/bin/systemctl").exists()


def _systemctl(*args: str) -> tuple[int, str]:
    """Runs a systemctl command (may prompt for sudo rights on the VPS)."""
    proc = subprocess.run(
        ["sudo", "-n", "systemctl", *args, SYSTEMD_UNIT],
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _is_systemd_active() -> bool:
    if not SYSTEMD_AVAILABLE:
        return False
    code, _ = _systemctl("is-active")
    return code == 0


def parse_hhmm(value: str) -> int:
    """Converts a 24-hour HH:MM value into minutes since midnight."""
    try:
        hour_text, minute_text = value.strip().split(":", 1)
        hour, minute = int(hour_text), int(minute_text)
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"Invalid time value: {value!r}") from None
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"Invalid time value: {value!r}")
    return hour * 60 + minute


def scheduled_action(
    now: datetime,
    start_min: int,
    stop_min: int,
    started_on: Optional[date],
    stopped_on: Optional[date],
    start_grace_minutes: int = 5,
) -> Optional[str]:
    """Returns a schedule action for the current minute."""
    if now.weekday() >= 5:
        return None
    current_min = now.hour * 60 + now.minute
    if (
        started_on != now.date()
        and start_min <= current_min < min(stop_min, start_min + max(1, start_grace_minutes))
    ):
        return "start"
    if stopped_on != now.date() and current_min >= stop_min:
        return "stop"
    return None


def _format_status(status) -> str:
    rec = read_runtime_record(settings.BOT_RUNTIME_FILE) or {}
    extra = rec.get("extra", {})
    started_at = rec.get("started_at")
    heartbeat_at = rec.get("heartbeat_at")

    # Heartbeat freshness check (bot writes every ~1s tick)
    responsive = True
    if heartbeat_at:
        try:
            hb = datetime.fromisoformat(str(heartbeat_at))
            responsive = (datetime.now() - hb).total_seconds() <= 15.0
        except (ValueError, TypeError):
            responsive = False

    pid = rec.get("pid")
    if pid and responsive:
        started = started_at.split("T")[0] + " " + started_at.split("T")[1][:8] if isinstance(started_at, str) and "T" in started_at else "unknown"
        orders = "🟢 REAL ORDERS ACTIVE" if rec.get("live_orders") else "🟣 SIMULATION / PAPER MODE"
        spot_str = f"Rs {extra.get('spot_price', 0.0):.2f}" if extra.get("spot_price") else "Live Feed Active"
        atm_str = str(int(round(extra.get("spot_price", 0.0) / 50.0) * 50)) if extra.get("spot_price") else "--"
        pos_str = f"🟢 Open Trade ({extra.get('active_position')})" if extra.get("active_position") else "⚪ Flat (Zero Exposure)"
        trades_cnt = extra.get("trades_count", len(extra.get("trades", [])))
        strategy_name = extra.get("strategy_name", "Last Hope GPU Winner Strategy")
        systemd_str = "systemd ✅" if _is_systemd_active() else "systemd ⚠️ INACTIVE"

        return (
            f"🏆 **{strategy_name} Bot is RUNNING**\n"
            f"• **Execution:** `{orders}` (PID `{pid}`)\n"
            f"• **Supervision:** `{systemd_str}` | **Started:** `{started}`\n"
            f"• **Spot Price:** `{spot_str}` | **ATM:** `{atm_str}`\n"
            f"• **Position:** `{pos_str}` | **Trades Today:** `{trades_cnt}`\n"
            f"• **Session:** 09:15-15:00 IST | Multi-TF Stoch (1m/2m/3m/5m) + S/R Bounce\n"
            f"• **Risk Geometry:** ATR(10)×1.5 SL/TP | Breakeven Stop @ +70% target move\n"
            f"• **Execution:** 5.0 pt aggressive limit buffer | Broker PositionBook verified"
        )
    if pid and not responsive:
        return f"⚠️ **Bot is RUNNING but UNRESPONSIVE**, PID `{pid}`. Last heartbeat is stale (>15s). Check: `journalctl -u flattrade-bot -n 50`"
    return "⏹️ **Bot is STOPPED.** (Use `/trading start` or wait for the 09:05 IST systemd auto-start)"


def _validate_configuration() -> None:
    missing = []
    if not settings.DISCORD_CONTROL_ENABLED:
        missing.append("DISCORD_CONTROL_ENABLED=true")
    for name, value in (
        ("DISCORD_BOT_TOKEN", settings.DISCORD_BOT_TOKEN),
        ("DISCORD_APPLICATION_ID", settings.DISCORD_APPLICATION_ID),
        ("DISCORD_GUILD_ID", settings.DISCORD_GUILD_ID),
        ("DISCORD_CONTROL_CHANNEL_ID", settings.DISCORD_CONTROL_CHANNEL_ID),
        ("DISCORD_ALLOWED_USER_IDS", settings.DISCORD_ALLOWED_USER_IDS),
    ):
        if not value:
            missing.append(name)
    if missing:
        raise RuntimeError("Discord control is not configured: " + ", ".join(missing))

def create_client():
    """Builds the discord.py client lazily. Bot lifecycle is delegated to systemd."""
    try:
        import discord
        from discord import app_commands
    except ImportError as exc:
        raise RuntimeError("Install discord.py before starting Discord control.") from exc

    intents = discord.Intents.none()
    intents.guilds = True
    client = discord.Client(intents=intents, application_id=int(settings.DISCORD_APPLICATION_ID))
    tree = app_commands.CommandTree(client)
    trading = app_commands.Group(name="trading", description="Control the Last Hope GPU Winner Trading Bot")
    guild = discord.Object(id=int(settings.DISCORD_GUILD_ID))

    allowed_users = parse_id_list(settings.DISCORD_ALLOWED_USER_IDS)

    async def reply(interaction, content: str) -> None:
        if interaction.response.is_done():
            await interaction.followup.send(content, ephemeral=True)
        else:
            await interaction.response.send_message(content, ephemeral=True)

    def authorized(interaction) -> bool:
        return is_control_authorized(
            str(interaction.user.id),
            str(interaction.guild_id) if interaction.guild_id is not None else None,
            str(interaction.channel_id) if interaction.channel_id is not None else None,
            allowed_users,
            settings.DISCORD_GUILD_ID,
            settings.DISCORD_CONTROL_CHANNEL_ID,
        )

    async def run_start(interaction) -> None:
        if not SYSTEMD_AVAILABLE:
            await reply(interaction, "⚠️ systemd control not available on this host (local dev machine).")
            return
        if _is_systemd_active():
            await reply(interaction, "⚠️ Last Hope Winner Bot is already running.")
            return
        await interaction.response.defer(ephemeral=True)
        code, out = await asyncio.to_thread(_systemctl, "start")
        if code == 0:
            await interaction.followup.send("🚀 **Live Last Hope Winner Bot started** (systemd + tmux dashboard). Attach: `tmux attach -t bot`")
        else:
            await interaction.followup.send(f"⚠️ Bot start failed: `{out}`")

    async def run_stop(interaction, hard: bool = False) -> None:
        if not SYSTEMD_AVAILABLE:
            await reply(interaction, "⚠️ systemd control not available on this host (local dev machine).")
            return
        await interaction.response.defer(ephemeral=True)
        code, out = await asyncio.to_thread(_systemctl, "stop")
        if code == 0:
            action = "hard-stopped" if hard else "stopped"
            await interaction.followup.send(f"🛑 **Bot {action}** (systemd unit stopped; tmux session closed).")
        else:
            await interaction.followup.send(f"⚠️ Bot stop failed: `{out}`")

    @trading.command(name="start", description="Start the live Last Hope Winner Bot (systemd + tmux dashboard)")
    async def trading_start(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        await run_start(interaction)

    @trading.command(name="stop", description="Stop the live bot via systemd")
    async def trading_stop(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        await run_stop(interaction)

    @trading.command(name="close", description="Close any open trade and stop the bot")
    async def trading_close(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        # NOTE: /trading stop now performs a clean systemctl stop. The bot's
        # EOD square-off + graceful SIGTERM handling flatten open trades first.
        await run_stop(interaction)

    @trading.command(name="status", description="Show live Last Hope Winner Bot status and risk metrics")
    async def trading_status(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        await reply(interaction, _format_status(None))

    @trading.command(name="levels", description="View Last Hope Winner strategy spec (strikes, stochastics, S/R bounce, BE stop)")
    async def trading_levels(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        msg = (
            "🏆 **Last Hope GPU Winner Strategy Spec (1m bars | Backtest-Verified)**\n"
            "• **Triggers:** FLAG (S4%D >= 79.5 & S1 < 79.5) | SUPER (S3, S4, S1 < 25 & S1 rising)\n"
            "• **Stochastics:** S1(12,3) S3(40,4) S4(50,10) evaluated across 1m, 2m, 3m, 5m bars\n"
            "• **Arming:** S1 <= 25.0 arms the setup for up to 10 bars\n"
            "• **S/R Bounce Gate:** Strict touch_buffer=0.0 (Low <= Level & Close >= Level - 0.5) on CPR/Camarilla/EMA/VWAP\n"
            "• **Strikes:** 2nd ITM only — CE = ATM - 100 / PE = ATM + 100\n"
            "• **Exits:** SL & TP = min(ATR(10) * 1.5, 15 pts) | Breakeven stop locked to Entry + 1.0 pt at +70% target move\n"
            "• **Session:** 09:15–15:00 IST | 1 position at a time"
        )
        await reply(interaction, msg)

    @trading.command(name="logs", description="View the latest 15 lines from the live trading bot log")
    async def trading_logs(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        log_file = Path("logs/last_hope_bot.log")
        if not log_file.exists():
            log_file = Path("logs/pocket_money_bot.log")
        if not log_file.exists():
            log_file = Path("logs/bot.log")
        if not log_file.exists():
            await reply(interaction, "⚠️ No active log file found yet.")
            return
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            tail_lines = lines[-15:] if len(lines) >= 15 else lines
            log_text = "\n".join(tail_lines)
            if len(log_text) > 1900:
                log_text = log_text[-1900:]
            await reply(interaction, f"📜 **Latest Bot Logs:**\n```text\n{log_text}\n```")
        except Exception as exc:
            await reply(interaction, f"⚠️ Error reading logs: {exc}")

    @trading.command(name="risk", description="View active risk limits, daily loss guard, and lot sizing")
    async def trading_risk(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        msg = (
            f"🛡️ **Last Hope Winner Risk Management Protocol**\n"
            f"• **Stop Loss / Target:** `min(ATR(10)×1.5, 15.0 pts)` symmetric (SL priority on same-bar touch)\n"
            f"• **Position Sizing:** `1 lot = {settings.LOT_SIZE} qty`, MIS, long options only\n"
            f"• **Consecutive Loss Cutoff:** `4 losses` blocks trading for the rest of the day\n"
            f"• **Breakeven:** SL hardens to Entry + 1.0 pt at +70% of target move\n"
            f"• **Daily Rs Cap:** None (backtest parity) — EOD flat at 15:15 IST hard rule\n"
            f"• **Entries:** 09:15–15:00 IST | One position at a time | No averaging\n"
            f"• **Order Type:** Aggressive limit (+5.0 pt buffer) for guaranteed fills"
        )
        await reply(interaction, msg)

    @trading.command(name="restart", description="Cleanly restart the live trading bot via systemd")
    async def trading_restart(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        if not SYSTEMD_AVAILABLE:
            await reply(interaction, "⚠️ systemd control not available on this host (local dev machine).")
            return
        await interaction.response.defer(ephemeral=True)
        code, out = await asyncio.to_thread(_systemctl, "restart")
        if code == 0:
            await interaction.followup.send("🔄 **Bot successfully restarted in live mode** (fresh day-state + S/R re-seed).")
        else:
            await interaction.followup.send(f"⚠️ Restart failed: `{out}`")

    @trading.command(name="help", description="Show all available Discord control commands")
    async def trading_help(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        msg = (
            "🎮 **Available Discord Trading Commands:**\n\n"
            "• `/trading status` — Live engine status, Nifty spot, position & heartbeat\n"
            "• `/trading logs` — Latest 15 lines of live bot execution logs\n"
            "• `/trading levels` — Last Hope Winner spec (triggers, strikes, S/R bounce, BE)\n"
            "• `/trading risk` — Active risk parameters (ATR×1.5 SL/TP, 4-loss block)\n"
            "• `/trading start` — Start live bot via systemd (tmux dashboard: `tmux attach -t bot`)\n"
            "• `/trading stop` — Stop the bot via systemd\n"
            "• `/trading restart` — Cleanly restart the trading engine\n"
            "• `/trading help` — Display this command reference guide"
        )
        await reply(interaction, msg)

    client.tree = tree

    @client.event
    async def on_ready():
        logger.info("Discord control connected as %s (systemd bridge mode)", client.user)

    async def setup_hook():
        tree.add_command(trading, guild=guild)
        await tree.sync(guild=guild)
        logger.info("Discord trading commands synced to guild %s", settings.DISCORD_GUILD_ID)

    client.setup_hook = setup_hook
    return client


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(Path("logs") / "discord_control.log", encoding="utf-8"), logging.StreamHandler()],
    )
    try:
        _validate_configuration()
        client = create_client()
        client.run(settings.DISCORD_BOT_TOKEN)
    except Exception as exc:
        logger.error("Discord control stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
