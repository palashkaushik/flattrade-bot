"""Sync Discord Slash Commands for Undisputed Rejection Champion Bot."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattrade_bot.config import settings
from flattrade_bot.control import TradingProcessManager
from flattrade_bot.discord_control import create_client

async def sync_commands():
    print("=" * 100)
    print("SYNCING DISCORD SLASH COMMANDS FOR UNDISPUTED REJECTION BOT")
    print("=" * 100)
    
    manager = TradingProcessManager(
        project_root=ROOT,
        python_executable=sys.executable,
        stop_file=settings.BOT_STOP_FILE,
        pid_file=settings.BOT_RUNTIME_FILE,
        visible_console=settings.BOT_VISIBLE_CONSOLE,
        visible_task_name=settings.BOT_VISIBLE_TASK_NAME,
    )
    
    client = create_client(manager)
    
    @client.event
    async def on_ready():
        print(f"Logged in as {client.user} (ID: {client.user.id})")
        guild = client.get_guild(int(settings.DISCORD_GUILD_ID))
        if guild:
            print(f"Connected to Guild: {guild.name} ({guild.id})")
        
        print("Synchronizing updated slash commands with Discord Gateway...")
        guild_obj = client.get_guild(int(settings.DISCORD_GUILD_ID)) or client.guilds[0]
        synced = await client.tree.sync(guild=guild_obj)
        print(f"Successfully synced {len(synced)} slash commands!")
        for cmd in synced:
            print(f"  - /{cmd.name}: {cmd.description}")
            if hasattr(cmd, "options"):
                for opt in cmd.options:
                    print(f"    * {opt.name}: {opt.description}")
        
        print("\nDiscord Slash Command Cache has been updated successfully!")
        await client.close()

    try:
        await client.start(settings.DISCORD_BOT_TOKEN)
    except Exception as e:
        print(f"Discord Sync Exception: {e}")

if __name__ == "__main__":
    asyncio.run(sync_commands())
