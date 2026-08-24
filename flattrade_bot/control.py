"""Secure local process control for the Discord trading supervisor."""

from __future__ import annotations

import os
import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, FrozenSet, Optional, Sequence


def _pid_is_alive(pid: int) -> bool:
    """Check a persisted child PID without requiring a process handle."""
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is unreliable on Windows and can raise a low-level
        # SystemError/WinError 87 that poisons the caller's next C-extension call.
        try:
            import psutil

            return bool(psutil.pid_exists(pid))
        except ImportError:
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, SystemError):
        return False
    return True


def _current_windows_session_id() -> Optional[int]:
    """Returns the current Windows desktop session, when available."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        session_id = ctypes.c_ulong()
        if ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session_id)):
            return int(session_id.value)
    except (AttributeError, OSError, TypeError):
        pass
    return None


def read_runtime_record(path: Path) -> Optional[dict[str, Any]]:
    try:
        record = json.loads(Path(path).read_text(encoding="ascii"))
        return record if isinstance(record, dict) else None
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def write_runtime_record(
    path: Path,
    pid: int,
    started_at: Optional[datetime],
    live_orders: Optional[bool] = None,
    external: Optional[bool] = None,
    heartbeat_at: Optional[datetime] = None,
) -> None:
    path = Path(path)
    previous = read_runtime_record(path) or {}
    record = {
        **previous,
        "pid": pid,
        "started_at": started_at.isoformat() if started_at else None,
        "heartbeat_at": heartbeat_at.isoformat() if heartbeat_at else previous.get("heartbeat_at"),
    }
    if live_orders is not None:
        record["live_orders"] = bool(live_orders)
    if external is not None:
        record["external"] = bool(external)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="ascii")


def touch_runtime_record(
    path: Path = None,
    pid: int = None,
    extra: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> bool:
    from flattrade_bot.config import settings
    target_path = Path(path) if path is not None else settings.BOT_RUNTIME_FILE
    target_pid = pid if pid is not None else os.getpid()
    record = read_runtime_record(target_path)
    if record is None or record.get("pid") != target_pid:
        # If record is missing or belongs to current pid, re-create / initialize
        record = {"pid": target_pid, "started_at": datetime.now().isoformat()}
    record["heartbeat_at"] = datetime.now().isoformat()
    if extra:
        record["extra"] = extra
    if kwargs:
        record.update(kwargs)
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(record), encoding="ascii")
        return True
    except (OSError, ValueError):
        return False


def clear_runtime_record(path: Path, pid: int) -> None:
    path = Path(path)
    record = read_runtime_record(path)
    if record is None or record.get("pid") == pid:
        path.unlink(missing_ok=True)


def parse_id_list(value: str) -> FrozenSet[str]:
    """Parses a comma-separated Discord ID list without accepting blanks."""
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def is_control_authorized(
    user_id: str,
    guild_id: Optional[str],
    channel_id: Optional[str],
    allowed_user_ids: FrozenSet[str],
    expected_guild_id: str,
    expected_channel_id: str,
) -> bool:
    """Requires an explicit user, guild, and channel match."""
    return bool(allowed_user_ids) and user_id in allowed_user_ids and guild_id == expected_guild_id and channel_id == expected_channel_id


@dataclass(frozen=True)
class ProcessStatus:
    running: bool
    pid: Optional[int]
    returncode: Optional[int]
    started_at: Optional[datetime]
    stop_requested: bool
    external: bool = False
    live_orders: Optional[bool] = None
    responsive: bool = True


class TradingProcessManager:
    """Starts the live bot and requests a graceful stop through a sentinel file."""

    def __init__(
        self,
        project_root: Path,
        python_executable: Optional[str] = None,
        stop_file: Optional[Path] = None,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        pid_file: Optional[Path] = None,
        pid_probe: Callable[[int], bool] = _pid_is_alive,
        process_discovery: Optional[Callable[[], Optional[dict[str, Any]]]] = None,
        heartbeat_timeout: float = 15.0,
        visible_console: bool = False,
        session_id_provider: Callable[[], Optional[int]] = _current_windows_session_id,
        visible_task_runner: Optional[Callable[[Sequence[str]], Optional[int]]] = None,
        visible_task_name: str = "\\Flattrade Bot Visible",
    ) -> None:
        self.project_root = Path(project_root)
        self.python_executable = python_executable or sys.executable
        self.stop_file = Path(stop_file or self.project_root / "logs" / "stop.requested")
        if not self.stop_file.is_absolute():
            self.stop_file = self.project_root / self.stop_file
        self.pid_file = Path(pid_file or self.project_root / "logs" / "bot.runtime.json")
        if not self.pid_file.is_absolute():
            self.pid_file = self.project_root / self.pid_file
        self._popen_factory = popen_factory
        self._pid_probe = pid_probe
        self._process_discovery = process_discovery or self._discover_external_bot
        self._heartbeat_timeout = heartbeat_timeout
        self.visible_console = visible_console
        self._session_id_provider = session_id_provider
        self._visible_task_runner = visible_task_runner
        self._visible_task_name = visible_task_name
        self._process: Optional[subprocess.Popen] = None
        self._attached_pid: Optional[int] = None
        self._started_at: Optional[datetime] = None
        self._external = False
        self._live_orders: Optional[bool] = None
        self._responsive = True
        self._last_returncode: Optional[int] = None
        self._output_handle = None
        self._lock = threading.Lock()

    def _write_pid_record(self, live_orders: Optional[bool] = None) -> None:
        write_runtime_record(
            self.pid_file,
            self._process.pid,
            self._started_at,
            live_orders=live_orders,
            external=False,
            heartbeat_at=datetime.now(),
        )

    def _read_pid_record(self) -> Optional[dict[str, Any]]:
        return read_runtime_record(self.pid_file)

    def _clear_pid_record(self, pid: Optional[int] = None) -> None:
        record = self._read_pid_record()
        if record is None or pid is None or record.get("pid") == pid:
            self.pid_file.unlink(missing_ok=True)

    def _launch_visible_task(self, args: Sequence[str]) -> Optional[int]:
        """Runs the pre-registered interactive task and waits for its bot PID."""
        if self._visible_task_runner is not None:
            return self._visible_task_runner(args)

        result = subprocess.run(
            ["schtasks", "/Run", "/TN", self._visible_task_name],
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"Interactive visible task could not start ({result.returncode}): {detail}"
            )

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            record = self._read_pid_record()
            if record is not None:
                try:
                    pid = int(record["pid"])
                except (KeyError, TypeError, ValueError):
                    pid = None
                if pid is not None and self._pid_probe(pid):
                    return pid
            time.sleep(0.2)
        raise RuntimeError("Interactive visible task started but no bot runtime record appeared")

    def _discover_external_bot(self) -> Optional[dict[str, Any]]:
        try:
            import psutil
        except ImportError:
            return None

        project_root = self.project_root.resolve()
        for process in psutil.process_iter(["pid", "cmdline", "create_time", "cwd"]):
            try:
                if process.pid == os.getpid():
                    continue
                cmdline = process.info.get("cmdline") or []
                if "flattrade_bot.main" not in cmdline:
                    continue
                cwd = process.info.get("cwd")
                if cwd and Path(cwd).resolve() != project_root:
                    continue
                created = process.info.get("create_time")
                return {
                    "pid": process.pid,
                    "started_at": datetime.fromtimestamp(created) if created else None,
                    "heartbeat_at": None,
                    "live_orders": "--live-orders" in cmdline,
                    "external": True,
                }
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError, ValueError):
                continue
        return None

    def _attach_persisted_process(self) -> None:
        if self._process is not None or self._attached_pid is not None:
            return
        record = self._read_pid_record() or self._process_discovery()
        if record is None:
            return
        pid = int(record["pid"])
        started_at = record.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        if not self._pid_probe(pid):
            self._clear_pid_record(pid)
            return
        self._attached_pid = pid
        self._started_at = started_at
        self._external = bool(record.get("external", True))
        self._live_orders = record.get("live_orders")
        heartbeat_at = record.get("heartbeat_at")
        if isinstance(heartbeat_at, str):
            heartbeat_at = datetime.fromisoformat(heartbeat_at)
        self._responsive = (
            heartbeat_at is None
            or (datetime.now() - heartbeat_at).total_seconds() <= self._heartbeat_timeout
        )
        if record.get("external"):
            write_runtime_record(
                self.pid_file,
                pid,
                started_at,
                live_orders=self._live_orders,
                external=True,
                heartbeat_at=heartbeat_at,
            )
        self._last_returncode = None

    def _refresh_runtime_health(self) -> None:
        pid = self._process.pid if self._process is not None else self._attached_pid
        if pid is None:
            return
        record = self._read_pid_record()
        if record is None or int(record.get("pid", -1)) != pid:
            return
        self._live_orders = record.get("live_orders", self._live_orders)
        heartbeat_at = record.get("heartbeat_at")
        if isinstance(heartbeat_at, str):
            heartbeat_at = datetime.fromisoformat(heartbeat_at)
        self._responsive = (
            heartbeat_at is None
            or (datetime.now() - heartbeat_at).total_seconds() <= self._heartbeat_timeout
        )

    def _refresh(self) -> None:
        if self._process is None:
            if self._attached_pid is not None and not self._pid_probe(self._attached_pid):
                pid = self._attached_pid
                self._attached_pid = None
                self._started_at = None
                self._external = False
                self._live_orders = None
                self._responsive = True
                self._clear_pid_record(pid)
            return
        process = self._process
        returncode = self._process.poll()
        if returncode is None:
            return
        self._last_returncode = returncode
        self._process = None
        self._clear_pid_record(process.pid)
        self._started_at = None
        self._external = False
        self._live_orders = None
        self._responsive = True
        if self._output_handle is not None:
            self._output_handle.close()
            self._output_handle = None

    def start(self, live_orders: bool = True, visible_console: Optional[bool] = None) -> bool:
        """Starts one live bot instance, returning False when one is already running."""
        with self._lock:
            self._refresh()
            self._attach_persisted_process()
            if self._process is not None or self._attached_pid is not None:
                return False

            self.stop_file.unlink(missing_ok=True)
            log_dir = self.project_root / "logs"
            log_dir.mkdir(exist_ok=True)
            output_handle = None
            use_visible_console = self.visible_console if visible_console is None else visible_console
            if not use_visible_console:
                output_handle = (log_dir / "managed_bot.log").open(
                    "a", encoding="utf-8", buffering=1
                )
            args: Sequence[str] = [
                self.python_executable,
                "-m",
                "flattrade_bot.undisputed_main",
            ]
            if live_orders:
                args = [*args, "--live"]

            if use_visible_console and self._session_id_provider() == 0:
                pid = self._launch_visible_task(args)
                if pid is None:
                    raise RuntimeError("Interactive visible bot task returned no process ID")
                self._process = None
                self._attached_pid = pid
                self._started_at = datetime.now()
                self._last_returncode = None
                self._external = True
                self._live_orders = live_orders
                self._responsive = True
                return True

            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")
            kwargs = {
                "cwd": str(self.project_root),
                "env": env,
            }
            if output_handle is not None:
                kwargs["stdout"] = output_handle
                kwargs["stderr"] = subprocess.STDOUT
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if use_visible_console:
                creation_flags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            if creation_flags:
                kwargs["creationflags"] = creation_flags

            try:
                self._process = self._popen_factory(args, **kwargs)
            except Exception:
                if output_handle is not None:
                    output_handle.close()
                raise
            self._output_handle = output_handle
            self._started_at = datetime.now()
            self._last_returncode = None
            self._external = False
            self._live_orders = live_orders
            self._responsive = True
            self._write_pid_record(live_orders=live_orders)
            return True

    def request_stop(self) -> bool:
        """Requests a graceful stop and nudges the child to process it promptly."""
        with self._lock:
            self._refresh()
            self._attach_persisted_process()
            if self._process is None and self._attached_pid is None:
                return False

            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.write_text("stop\n", encoding="ascii")
            try:
                if os.name == "nt":
                    if self._process is not None:
                        try:
                            self._process.send_signal(signal.CTRL_BREAK_EVENT)
                        except (AttributeError, OSError, SystemError, ValueError):
                            pass
                    # Attached Windows tasks receive the stop request through
                    # the sentinel file; os.kill is not safe for them here.
                else:
                    if self._process is not None:
                        self._process.send_signal(signal.SIGTERM)
                    else:
                        os.kill(self._attached_pid, signal.SIGTERM)
            except (AttributeError, OSError, ValueError):
                # The stop file remains authoritative when console signals are unavailable.
                pass
            return True

    def wait_for_exit(self, timeout: float = 45.0) -> bool:
        with self._lock:
            process = self._process
            attached_pid = self._attached_pid
        if process is None:
            if attached_pid is None:
                return True
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                with self._lock:
                    self._refresh()
                    if self._attached_pid is None:
                        return True
                time.sleep(0.1)
            return False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        with self._lock:
            self._refresh()
        return True

    def status(self) -> ProcessStatus:
        with self._lock:
            self._refresh()
            self._attach_persisted_process()
            self._refresh_runtime_health()
            process = self._process
            pid = process.pid if process is not None else self._attached_pid
            return ProcessStatus(
                running=pid is not None,
                pid=pid,
                returncode=self._last_returncode,
                started_at=self._started_at,
                stop_requested=self.stop_file.exists(),
                external=self._external,
                live_orders=self._live_orders,
                responsive=self._responsive,
            )
