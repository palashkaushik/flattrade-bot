"""Discord Gateway control plane for the Pocket Money trading process."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from flattrade_bot.config import settings
from flattrade_bot.control import (
    TradingProcessManager,
    is_control_authorized,
    parse_id_list,
)

logger = logging.getLogger("flattrade_bot.discord_control")


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
    from flattrade_bot.control import read_runtime_record
    rec = read_runtime_record(settings.BOT_RUNTIME_FILE) or {}
    extra = rec.get("extra", {})
    if status.running:
        started = status.started_at.strftime("%Y-%m-%d %H:%M:%S") if status.started_at else "unknown"
        suffix = " (stop requested)" if status.stop_requested else ""
        origin = "externally attached" if status.external else "managed"
        orders = "🟢 REAL ORDERS ACTIVE" if status.live_orders else "🟣 SIMULATION / PAPER MODE"
        if not status.responsive:
            return f"⚠️ **Unresponsive bot**, PID `{status.pid}`, {origin}, last heartbeat is stale."
        
        spot_str = f"Rs {extra.get('spot_price', 0.0):.2f}" if extra.get("spot_price") else "Live Feed Active"
        atm_str = str(int(round(extra.get("spot_price", 0.0) / 50.0) * 50)) if extra.get("spot_price") else "--"
        pos_str = f"🟢 Open Trade ({extra.get('active_position')})" if extra.get("active_position") else "⚪ Flat (Zero Exposure)"
        trades_cnt = extra.get("trades_count", len(extra.get("trades", [])))
        strategy_name = extra.get("strategy_name", "Last Hope GPU Winner Strategy")

        return (
            f"🏆 **{strategy_name} Bot is RUNNING**\n"
            f"• **Execution:** `{orders}` (PID `{status.pid}`)\n"
            f"• **Started:** `{started}`{suffix}\n"
            f"• **Spot Price:** `{spot_str}` | **ATM:** `{atm_str}`\n"
            f"• **Position:** `{pos_str}` | **Trades Today:** `{trades_cnt}`\n"
            f"• **Session:** 09:15-15:00 IST | Multi-TF Stoch (1m/2m/3m/5m) + S/R Bounce\n"
            f"• **Risk Geometry:** ATR(10)×1.5 SL/TP | Breakeven Stop @ +70% target move\n"
            f"• **Execution:** 5.0 pt aggressive limit buffer | Broker PositionBook verified"
        )
    if status.returncode is None:
        return "⏹️ **Bot is STOPPED.** (Use `/trading start` or `/trading start-visible` to launch)"
    return f"⏹️ **Bot is STOPPED.** (Last exit code: `{status.returncode}`)"


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


def create_client(manager: TradingProcessManager):
    """Builds the discord.py client lazily."""
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
    start_min = parse_hhmm(settings.BOT_START_TIME)
    stop_min = parse_hhmm(settings.BOT_STOP_TIME)
    timezone = ZoneInfo(settings.TRADING_TIMEZONE)
    schedule_state = {"started_on": None, "stopped_on": None}
    schedule_task = None

    async def reply(interaction, content: str) -> None:
        if interaction.response.is_done():
            await interaction.followup.send(content, ephemeral=True)
        else:
            await interaction.response.send_message(content, ephemeral=True)

    async def acknowledge(interaction) -> bool:
        if interaction.response.is_done():
            return True
        try:
            await interaction.response.defer(ephemeral=True)
            return True
        except discord.NotFound:
            logger.warning("Discord interaction expired before acknowledgement")
            return False
        except discord.HTTPException:
            logger.exception("Discord interaction acknowledgement failed")
            return False

    def authorized(interaction) -> bool:
        return is_control_authorized(
            str(interaction.user.id),
            str(interaction.guild_id) if interaction.guild_id is not None else None,
            str(interaction.channel_id) if interaction.channel_id is not None else None,
            allowed_users,
            settings.DISCORD_GUILD_ID,
            settings.DISCORD_CONTROL_CHANNEL_ID,
        )

    async def run_start(interaction, visible_console: Optional[bool] = None) -> None:
        if not await acknowledge(interaction):
            return
        try:
            if visible_console is None:
                started = await asyncio.to_thread(manager.start, True)
            else:
                started = await asyncio.to_thread(manager.start, True, visible_console=visible_console)
        except Exception as exc:
            logger.exception("Discord start command failed")
            await interaction.followup.send(f"Bot start failed: {exc}", ephemeral=True)
            return
        started_message = (
            "🚀 **Live Last Hope Winner Bot started in visible terminal window.**"
            if visible_console
            else "🚀 **Live Last Hope Winner Bot started in background.**"
        )
        await interaction.followup.send(
            started_message if started else "⚠️ Last Hope Winner Bot is already running.",
            ephemeral=True,
        )

    async def run_stop(interaction) -> None:
        if not await acknowledge(interaction):
            return
        requested = await asyncio.to_thread(manager.request_stop)
        if not requested:
            await interaction.followup.send("⚠️ Bot is already stopped.", ephemeral=True)
            return
        await interaction.followup.send(
            "🛑 **Shutdown requested.** The bot will gracefully exit after any active trade is closed. Use `/trading status` to verify.",
            ephemeral=True,
        )

    @trading.command(name="start", description="Start the live Last Hope Winner Bot in background")
    async def trading_start(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        await run_start(interaction)

    @trading.command(name="start-visible", description="Start the live bot with visible interactive terminal dashboard")
    async def trading_start_visible(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        await run_start(interaction, visible_console=True)

    @trading.command(name="stop", description="Gracefully stop the Last Hope Winner Bot")
    async def trading_stop(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        await run_stop(interaction)

    @trading.command(name="close", description="Close active trade and stop the bot")
    async def trading_close(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        await run_stop(interaction)

    @trading.command(name="status", description="Show live Last Hope Winner Bot status and risk metrics")
    async def trading_status(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        await reply(interaction, _format_status(manager.status()))

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
            f"🛡️ **Pocket Money Risk Management Protocol**\n"
            f"• **Stop Loss / Target:** `±7.0 premium pts` (SL priority on same-bar touch)\n"
            f"• **Position Sizing:** `1 lot = {settings.LOT_SIZE} qty`, MIS, long options only\n"
            f"• **Consecutive Loss Cutoff:** `4 losses` blocks trading for the rest of the day\n"
            f"• **Daily Rs Cap:** None (backtest parity) — EOD flat at 15:00 IST hard rule\n"
            f"• **Entries:** 09:20–14:59 IST | One position at a time | No averaging\n"
            f"• **Order Type:** Marketable Aggressive Market Order"
        )
        await reply(interaction, msg)

    @trading.command(name="restart", description="Cleanly restart the live trading bot")
    async def trading_restart(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        if not await acknowledge(interaction):
            return
        await asyncio.to_thread(manager.request_stop)
        await asyncio.to_thread(manager.wait_for_exit, 10.0)
        started = await asyncio.to_thread(manager.start, True)
        if started:
            await interaction.followup.send("🔄 **Bot successfully restarted in live mode.**", ephemeral=True)
        else:
            await interaction.followup.send("⚠️ Restart attempt completed. Use `/trading status` to verify.", ephemeral=True)

    @trading.command(name="help", description="Show all available Discord control commands")
    async def trading_help(interaction: discord.Interaction):
        if not authorized(interaction):
            await reply(interaction, "🚫 Not authorized for this control channel.")
            return
        msg = (
            "🎮 **Available Discord Trading Commands:**\n\n"
            "• `/trading status` — View live engine status, Nifty spot, filter side & active position\n"
            "• `/trading logs` — Read the latest 15 lines of live bot execution logs\n"
            "• `/trading levels` — View the Pocket Money spec (triggers, strikes, filter)\n"
            "• `/trading risk` — View active risk parameters (±7 pts SL/TP, 4-loss block)\n"
            "• `/trading start` — Launch live bot in background\n"
            "• `/trading start-visible` — Launch live bot with interactive terminal GUI\n"
            "• `/trading stop` — Gracefully stop the bot after any active trade closes\n"
            "• `/trading restart` — Cleanly restart the trading engine\n"
            "• `/trading help` — Display this command reference guide"
        )
        await reply(interaction, msg)

    client.tree = tree

    async def schedule_loop() -> None:
        while not client.is_closed():
            try:
                now = datetime.now(timezone)
                action = scheduled_action(
                    now,
                    start_min,
                    stop_min,
                    schedule_state["started_on"],
                    schedule_state["stopped_on"],
                    settings.BOT_START_GRACE_MINUTES,
                )
                if action == "start":
                    if await asyncio.to_thread(manager.start, True):
                        schedule_state["started_on"] = now.date()
                        logger.info("Scheduled live bot start completed.")
                elif action == "stop":
                    requested = await asyncio.to_thread(manager.request_stop)
                    exited = True
                    if requested:
                        exited = await asyncio.to_thread(manager.wait_for_exit, 45.0)
                        if not exited:
                            logger.error("Scheduled stop timed out; no force-kill was performed.")
                    if exited:
                        schedule_state["stopped_on"] = now.date()
                        logger.info("Scheduled bot stop completed.")
            except Exception:
                logger.exception("Discord control scheduler iteration failed")
            await asyncio.sleep(5)

    @client.event
    async def on_ready():
        nonlocal schedule_task
        if schedule_task is None:
            schedule_task = asyncio.create_task(schedule_loop())
        logger.info("Discord control connected as %s", client.user)

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
        manager = TradingProcessManager(
            project_root=Path(__file__).resolve().parent.parent,
            python_executable=sys.executable,
            stop_file=settings.BOT_STOP_FILE,
            pid_file=settings.BOT_RUNTIME_FILE,
            visible_console=settings.BOT_VISIBLE_CONSOLE,
            visible_task_name=settings.BOT_VISIBLE_TASK_NAME,
        )
        client = create_client(manager)
        client.run(settings.DISCORD_BOT_TOKEN)
    except Exception as exc:
        logger.error("Discord control stopped: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
