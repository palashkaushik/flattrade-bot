"""Process lifecycle helpers for terminal-bound bot execution."""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import threading
from ctypes import wintypes
from typing import Callable, Dict, Optional


logger = logging.getLogger("flattrade_bot.lifecycle")

_SYNCHRONIZE = 0x00100000
_WAIT_TIMEOUT = 0x00000102
_ERROR_INVALID_PARAMETER = 87


def _process_is_alive(pid: int) -> bool:
    """Checks whether a process still exists without depending on psutil."""
    if pid <= 0:
        return False

    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
        if not handle:
            # Access denied does not prove that the process exited. An invalid
            # PID does, so keep watching protected parent processes.
            return ctypes.get_last_error() != _ERROR_INVALID_PARAMETER
        try:
            return kernel32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


class TerminalLifecycle:
    """Stops a bot when its launching terminal or console closes.

    Windows does not reliably turn a console-close event into Python's
    ``KeyboardInterrupt``. The native console handler covers that event, and
    the parent watcher covers launchers that detach the child from the shell.
    """

    def __init__(
        self,
        on_shutdown: Callable[[], None],
        parent_pid: Optional[int] = None,
        poll_interval: float = 0.5,
    ) -> None:
        self._on_shutdown = on_shutdown
        self._parent_pid = os.getppid() if parent_pid is None else parent_pid
        self._poll_interval = poll_interval
        self._stop_watcher = threading.Event()
        self._shutdown_requested = threading.Event()
        self._watcher: Optional[threading.Thread] = None
        self._previous_handlers: Dict[int, object] = {}
        self._console_handler = None

    def start(self) -> None:
        """Installs shutdown handlers and starts the parent liveness watcher."""
        self._install_signal_handlers()
        if self._parent_pid > 1 and self._parent_pid != os.getpid():
            self._watcher = threading.Thread(
                target=self._watch_parent,
                name="terminal-lifecycle-watchdog",
                daemon=True,
            )
            self._watcher.start()

    def request_shutdown(self, reason: str) -> None:
        """Requests shutdown once; callbacks must only perform lightweight work."""
        if self._shutdown_requested.is_set():
            return
        self._shutdown_requested.set()
        logger.info("Shutdown requested: %s", reason)
        self._on_shutdown()

    def stop(self) -> None:
        """Stops the watcher and restores signal handlers."""
        self._stop_watcher.set()
        if self._watcher and self._watcher is not threading.current_thread():
            self._watcher.join(timeout=max(1.0, self._poll_interval * 2))
        self._watcher = None

        for sig, previous in self._previous_handlers.items():
            try:
                signal.signal(sig, previous)
            except (OSError, ValueError, RuntimeError):
                pass
        self._previous_handlers.clear()

        if self._console_handler is not None and os.name == "nt":
            try:
                ctypes.WinDLL("kernel32").SetConsoleCtrlHandler(self._console_handler, False)
            except (OSError, AttributeError):
                pass
            self._console_handler = None

    def _watch_parent(self) -> None:
        while not self._stop_watcher.wait(self._poll_interval):
            if not _process_is_alive(self._parent_pid):
                self.request_shutdown("launching terminal exited")
                return

    def _signal_handler(self, signum: int, _frame: object) -> None:
        self.request_shutdown(f"signal {signum}")

    def _install_signal_handlers(self) -> None:
        for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, signal_name, None)
            if sig is None:
                continue
            try:
                self._previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._signal_handler)
            except (OSError, ValueError, RuntimeError):
                logger.debug("Could not install %s handler", signal_name, exc_info=True)

        if os.name != "nt":
            return

        handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        @handler_type
        def console_handler(event_type: int) -> int:
            if event_type in (0, 1, 2, 5, 6):
                self.request_shutdown("Windows console closed")
                return 1
            return 0

        self._console_handler = console_handler
        kernel32 = ctypes.WinDLL("kernel32")
        if not kernel32.SetConsoleCtrlHandler(self._console_handler, True):
            logger.debug("Could not install Windows console-close handler", exc_info=True)
