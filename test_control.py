import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from flattrade_bot.control import (
    TradingProcessManager,
    is_control_authorized,
    parse_id_list,
)
from flattrade_bot.discord_control import scheduled_action
from flattrade_bot.discord_control import _format_status
from flattrade_bot.discord_control import create_client
from flattrade_bot.config import settings
from flattrade_bot.control import ProcessStatus
from flattrade_bot.control import _pid_is_alive
from flattrade_bot.risk.manager import RiskManager


ROOT = Path(__file__).resolve().parent


def test_parse_id_list_ignores_empty_values_and_duplicates():
    assert parse_id_list(" 1, 2,1,, 3 ") == frozenset({"1", "2", "3"})


def test_control_authorization_requires_all_configured_scope_values():
    allowed = {"user-1"}

    assert is_control_authorized("user-1", "guild-1", "channel-1", allowed, "guild-1", "channel-1")
    assert not is_control_authorized("user-2", "guild-1", "channel-1", allowed, "guild-1", "channel-1")
    assert not is_control_authorized("user-1", "guild-2", "channel-1", allowed, "guild-1", "channel-1")
    assert not is_control_authorized("user-1", "guild-1", "channel-2", allowed, "guild-1", "channel-1")


def test_risk_manager_marks_the_session_complete_at_end_minute():
    risk = RiskManager(session_end_min=900)

    assert risk.is_session_complete(899) is False
    assert risk.is_session_complete(900) is True


def test_schedule_starts_only_inside_the_short_start_grace_window():
    assert scheduled_action(datetime(2026, 8, 10, 9, 15), 555, 900, None, None) == "start"
    assert scheduled_action(datetime(2026, 8, 10, 9, 19), 555, 900, None, None) == "start"
    assert scheduled_action(datetime(2026, 8, 10, 9, 20), 555, 900, None, None) is None
    assert scheduled_action(datetime(2026, 8, 8, 9, 15), 555, 900, None, None) is None


def test_default_schedule_starts_at_905_and_stops_at_1500():
    config_source = (ROOT / "flattrade_bot" / "config.py").read_text(encoding="utf-8")
    example_source = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert 'os.getenv("BOT_START_TIME", "09:15")' in config_source
    assert "BOT_START_TIME=09:15" in example_source
    assert "BOT_STOP_TIME=15:00" in example_source
    assert scheduled_action(datetime(2026, 8, 10, 9, 15), 555, 900, None, None) == "start"
    assert scheduled_action(datetime(2026, 8, 10, 9, 19), 555, 900, None, None) == "start"
    assert scheduled_action(datetime(2026, 8, 10, 9, 20), 555, 900, None, None) is None


def test_schedule_stops_once_at_the_session_end():
    day = datetime(2026, 8, 10).date()

    assert scheduled_action(datetime(2026, 8, 10, 15, 0), 555, 900, day, None) == "stop"
    assert scheduled_action(datetime(2026, 8, 10, 15, 1), 555, 900, day, day) is None


class FakeProcess:
    pid = 4242

    def __init__(self):
        self.returncode = None
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, signal):
        self.signals.append(signal)


def test_process_manager_starts_live_child_and_writes_stop_request(tmp_path):
    process = FakeProcess()
    calls = []
    stop_file = Path(tmp_path) / "stop.requested"

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        return process

    manager = TradingProcessManager(
        project_root=tmp_path,
        python_executable="python.exe",
        stop_file=stop_file,
        popen_factory=fake_popen,
    )

    started = manager.start(live_orders=True)
    assert started is True
    assert "--auto-login" in calls[0][0]
    assert "--live-orders" in calls[0][0]
    assert manager.status().pid == 4242

    assert manager.request_stop() is True
    assert stop_file.exists()


def test_pid_probe_handles_low_level_windows_kill_failure(monkeypatch):
    monkeypatch.setattr("flattrade_bot.control.os.name", "posix")

    def broken_kill(_pid, _signal):
        raise SystemError("simulated Windows kill failure")

    monkeypatch.setattr("flattrade_bot.control.os.kill", broken_kill)

    assert _pid_is_alive(4242) is False


def test_process_manager_status_attaches_to_child_after_manager_restart(tmp_path):
    process = FakeProcess()
    stop_file = Path(tmp_path) / "stop.requested"
    pid_file = Path(tmp_path) / "managed_bot.pid"

    def fake_popen(_args, **_kwargs):
        return process

    first_manager = TradingProcessManager(
        project_root=tmp_path,
        python_executable="python.exe",
        stop_file=stop_file,
        popen_factory=fake_popen,
        pid_file=pid_file,
    )
    assert first_manager.start(live_orders=True) is True

    second_manager = TradingProcessManager(
        project_root=tmp_path,
        python_executable="python.exe",
        stop_file=stop_file,
        popen_factory=fake_popen,
        pid_file=pid_file,
        pid_probe=lambda pid: pid == process.pid,
    )

    status = second_manager.status()

    assert status.running is True
    assert status.pid == process.pid


def test_process_manager_discovers_manually_started_bot_and_blocks_duplicate(tmp_path):
    process = FakeProcess()
    manager = TradingProcessManager(
        project_root=tmp_path,
        python_executable="python.exe",
        stop_file=Path(tmp_path) / "stop.requested",
        pid_file=Path(tmp_path) / "managed_bot.pid",
        pid_probe=lambda pid: pid == process.pid,
        process_discovery=lambda: {
            "pid": process.pid,
            "started_at": datetime(2026, 8, 11, 9, 15),
            "heartbeat_at": datetime.now(),
            "live_orders": True,
        },
    )

    status = manager.status()

    assert status.running is True
    assert status.external is True
    assert status.live_orders is True
    assert manager.start(live_orders=True) is False


def test_process_manager_marks_stale_heartbeat_unresponsive(tmp_path):
    pid_file = Path(tmp_path) / "managed_bot.pid"
    pid_file.write_text(json.dumps({
        "pid": 4242,
        "started_at": "2026-08-11T09:15:00",
        "heartbeat_at": "2020-01-01T00:00:00",
        "live_orders": True,
    }), encoding="ascii")
    manager = TradingProcessManager(
        project_root=tmp_path,
        stop_file=Path(tmp_path) / "stop.requested",
        pid_file=pid_file,
        pid_probe=lambda pid: pid == 4242,
        heartbeat_timeout=15,
    )

    status = manager.status()

    assert status.running is True
    assert status.responsive is False


def test_status_format_identifies_external_process_and_order_mode():
    message = _format_status(ProcessStatus(
        running=True,
        pid=4242,
        returncode=None,
        started_at=datetime(2026, 8, 11, 9, 15),
        stop_requested=False,
        external=True,
        live_orders=True,
        responsive=True,
    ))

    assert "externally attached" in message
    assert "live orders enabled" in message


def test_process_manager_can_launch_a_visible_verification_console(tmp_path):
    process = FakeProcess()
    calls = []

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        return process

    manager = TradingProcessManager(
        project_root=tmp_path,
        python_executable="python.exe",
        stop_file=Path(tmp_path) / "stop.requested",
        pid_file=Path(tmp_path) / "managed_bot.pid",
        popen_factory=fake_popen,
        visible_console=True,
    )

    assert manager.start(live_orders=True) is True
    kwargs = calls[0][1]
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs
    if os.name == "nt":
        assert kwargs["creationflags"] & subprocess.CREATE_NEW_CONSOLE


def test_process_manager_can_override_console_visibility_for_one_start(tmp_path):
    process = FakeProcess()
    calls = []

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        return process

    manager = TradingProcessManager(
        project_root=tmp_path,
        python_executable="python.exe",
        stop_file=Path(tmp_path) / "stop.requested",
        pid_file=Path(tmp_path) / "managed_bot.pid",
        popen_factory=fake_popen,
    )

    assert manager.start(live_orders=True, visible_console=True) is True
    kwargs = calls[0][1]
    assert "stdout" not in kwargs
    assert "stderr" not in kwargs
    if os.name == "nt":
        assert kwargs["creationflags"] & subprocess.CREATE_NEW_CONSOLE


def test_visible_start_uses_the_interactive_task_when_controller_is_in_session_zero(tmp_path):
    calls = []

    def run_visible_task(args):
        calls.append(args)
        return 4242

    manager = TradingProcessManager(
        project_root=tmp_path,
        python_executable="python.exe",
        stop_file=Path(tmp_path) / "stop.requested",
        pid_file=Path(tmp_path) / "managed_bot.pid",
        session_id_provider=lambda: 0,
        visible_task_runner=run_visible_task,
        pid_probe=lambda pid: pid == 4242,
    )

    assert manager.start(live_orders=True, visible_console=True) is True
    assert manager.status().pid == 4242
    assert "--live-orders" in calls[0]


def test_discord_registers_start_visible_and_requests_visible_live_start(monkeypatch):
    class FakeManager:
        def __init__(self):
            self.start_calls = []

        def start(self, live_orders=True, visible_console=None):
            self.start_calls.append((live_orders, visible_console))
            return True

    class FakeResponse:
        def __init__(self):
            self.deferred = False

        def is_done(self):
            return self.deferred

        async def defer(self, ephemeral=False):
            self.deferred = True

    class FakeFollowup:
        def __init__(self):
            self.messages = []

        async def send(self, content, ephemeral=False):
            self.messages.append((content, ephemeral))

    class FakeInteraction:
        user = type("User", (), {"id": 4})()
        guild_id = 2
        channel_id = 3

        def __init__(self):
            self.response = FakeResponse()
            self.followup = FakeFollowup()

    monkeypatch.setattr(settings, "DISCORD_APPLICATION_ID", "1")
    monkeypatch.setattr(settings, "DISCORD_GUILD_ID", "2")
    monkeypatch.setattr(settings, "DISCORD_CONTROL_CHANNEL_ID", "3")
    monkeypatch.setattr(settings, "DISCORD_ALLOWED_USER_IDS", "4")
    manager = FakeManager()
    client = create_client(manager)

    async def setup():
        async def no_network_sync(**_kwargs):
            return []

        client.tree.sync = no_network_sync
        await client.setup_hook()

    import asyncio
    asyncio.run(setup())

    import discord
    trading = client.tree.get_command("trading", guild=discord.Object(id=2))
    start_visible = trading.get_command("start-visible")
    assert start_visible is not None
    interaction = FakeInteraction()

    async def invoke():
        await start_visible.callback(interaction)

    asyncio.run(invoke())

    assert manager.start_calls == [(True, True)]
    assert "visible" in interaction.followup.messages[0][0].lower()


def test_discord_stop_acknowledges_immediately_and_registers_close_alias(monkeypatch):
    class FakeManager:
        def request_stop(self):
            return True

        def wait_for_exit(self, _timeout):
            raise AssertionError("Discord command must not wait for process exit")

    class FakeResponse:
        def __init__(self):
            self.deferred = False

        def is_done(self):
            return self.deferred

        async def defer(self, ephemeral=False):
            self.deferred = True

    class FakeFollowup:
        def __init__(self):
            self.messages = []

        async def send(self, content, ephemeral=False):
            self.messages.append((content, ephemeral))

    class FakeInteraction:
        user = type("User", (), {"id": 4})()
        guild_id = 2
        channel_id = 3

        def __init__(self):
            self.response = FakeResponse()
            self.followup = FakeFollowup()

    monkeypatch.setattr(settings, "DISCORD_APPLICATION_ID", "1")
    monkeypatch.setattr(settings, "DISCORD_GUILD_ID", "2")
    monkeypatch.setattr(settings, "DISCORD_CONTROL_CHANNEL_ID", "3")
    monkeypatch.setattr(settings, "DISCORD_ALLOWED_USER_IDS", "4")

    client = create_client(FakeManager())

    async def setup():
        client.tree.sync = lambda **_kwargs: None
        original_sync = client.tree.sync

        async def no_network_sync(**_kwargs):
            return []

        client.tree.sync = no_network_sync
        await client.setup_hook()
        client.tree.sync = original_sync

    import asyncio
    asyncio.run(setup())

    import discord
    trading = client.tree.get_command("trading", guild=discord.Object(id=2))
    stop = trading.get_command("stop")
    close = trading.get_command("close")
    interaction = FakeInteraction()

    async def invoke():
        await stop.callback(interaction)

    asyncio.run(invoke())

    assert close is not None
    assert interaction.response.deferred is True
    assert interaction.followup.messages
    assert "Close requested" in interaction.followup.messages[0][0]
