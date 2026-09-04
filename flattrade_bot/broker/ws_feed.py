"""Flattrade WebSocket tick feed — push-based market data (replaces REST polling).

Protocol (flattrade documentation.md §6, Pi API v2.0 — verified live Sep 4 2026):
  - URL:         wss://piconnect.flattrade.in/PiConnectWSAPI/
  - CONNECT (mandatory, immediately on open — WITHOUT it the server accepts
    the socket, answers heartbeats, and SILENTLY IGNORES every subscription:
    the Sep-4 stuck-prices bug):
      {"t":"a","uid":UID,"actid":UID,"source":"API","accesstoken":TOKEN}
    Expect {"t":"ak","s":"Ok"} (case varies: "Ok"/"OK") before subscribing.
  - SUBSCRIBE touchline:  {"t":"t","k":"NSE|26000#NFO|42631"}  (#-batched)
  - UNSUBSCRIBE:          {"t":"u","k":"..."}
  - HEARTBEAT: run_forever(ping_interval=3, ping_payload='{"t":"h"}') —
    WS-layer PING frames every 3s with the heartbeat JSON as payload, EXACTLY
    as the official NorenRestApiPy client does (NorenApi.py:119). A separate
    TEXT-message heartbeat thread is INSUFFICIENT: the 2026-09-04 11:05
    incident — WS authed at 10:42, ticks flowed at 10:44, then the server
    silently dropped the connection ~20 min in (no close frame, no error).
    Text sends into the half-dead TCP buffer still "succeed", so nothing
    detected the failure and the feed froze with zero errors. WS-layer PINGs
    make recv() fail within seconds of a dead peer -> on_close -> reconnect.
  - Ticks: "tk" (subscribe ack w/ full quote), "tf" (touchline feed updates)

Design:
  - Dedicated daemon thread runs websocket-client's run_forever with the
    official 3s PING keepalive + auto-reconnect (exponential backoff, capped).
  - On connect ack: re-subscribes ALL active tokens + NIFTY spot (26000@NSE).
  - Ticks land in a thread-safe dict; the async engine reads them every tick
    loop - no locks held during processing (dict swap semantics).
  - Message watchdog: if no WS frame of ANY kind for 90s, force-close the
    socket (server half-dead) -> reconnect loop takes over.
"""
import json
import logging
import threading
import time
from typing import Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

try:
    import websocket  # websocket-client package
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

from flattrade_bot.config import settings

WS_URL = "wss://piconnect.flattrade.in/PiConnectWSAPI/"
PING_INTERVAL_SEC = 3.0          # official Noren client: ping_interval=3
NO_MSG_WATCHDOG_SEC = 90.0       # no frame at all -> force reconnect
SPOT_SUBSCRIBE = ("NSE", "26000")  # Nifty 50 index


class FlattradeWebSocketFeed:
    """Push-based tick feed with connect handshake, official 3s PING
    keepalive, auto-reconnect and REST-fallback hooks."""

    def __init__(self):
        self._token: Optional[str] = None
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._subscribed: Set[str] = set()          # "EXCH|token" keys
        self._last_tick_at: Dict[str, float] = {}   # "EXCH|token" -> monotonic ts
        self._latest: Dict[str, float] = {}         # "EXCH|token" -> ltp
        self._lock = threading.Lock()
        self._connected_at: Optional[float] = None
        self._last_msg_at: Optional[float] = None   # any-frame liveness
        self._authed = threading.Event()            # connect ack received
        self._on_tick: Optional[Callable[[str, str, float], None]] = None

    # ------------------------------------------------------------------ state
    @property
    def connected(self) -> bool:
        # "connected" = AUTHED (handshake acked). The old `monotonic() -
        # connected_at < 60` clause made connected() permanently False after
        # 60s and silently disabled WS + REST fallback logic downstream.
        return (self._authed.is_set() and self._ws is not None)

    def last_ltp(self, exchange: str, token: str) -> Optional[float]:
        key = f"{exchange}|{token}"
        with self._lock:
            return self._latest.get(key)

    def age_seconds(self, exchange: str, token: str) -> float:
        """Seconds since the last tick for this instrument (inf if none)."""
        key = f"{exchange}|{token}"
        with self._lock:
            ts = self._last_tick_at.get(key)
        return float("inf") if ts is None else time.monotonic() - ts

    # ------------------------------------------------------------- lifecycle
    def set_token(self, token: str):
        """(Re)authenticates. If the token changed, reconnects the socket."""
        prev = self._token
        self._token = token
        if token and (prev != token):
            self.start()

    def start(self):
        if not _HAS_WS:
            logger.warning("websocket-client not installed — WS feed unavailable, REST polling continues")
            return
        if self._thread and self._thread.is_alive():
            return
        if not self._token:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, daemon=True, name="flattrade-ws")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------ subscribe
    def subscribe(self, exchange: str, token: str):
        key = f"{exchange}|{token}"
        with self._lock:
            if key in self._subscribed:
                return
            self._subscribed.add(key)
        # Send immediately (harmless pre-auth: dropped by server) — the
        # connect-ack flush (_resubscribe_all) guarantees delivery after the
        # t:a handshake. Duplicate subscribes just re-ack.
        self._send({"t": "t", "k": key})

    def unsubscribe(self, exchange: str, token: str):
        key = f"{exchange}|{token}"
        with self._lock:
            self._subscribed.discard(key)
        self._send({"t": "u", "k": key})

    def _resubscribe_all(self):
        with self._lock:
            keys = list(self._subscribed)
        if not keys:
            return
        # Docs: one message, #-batched keys — fewer round-trips, atomic re-sub.
        self._send({"t": "t", "k": "#".join(keys)})
        logger.info("WS: re-subscribed %d instruments (batched)", len(keys))

    # ------------------------------------------------------------- internals
    def _send(self, payload: dict):
        ws = self._ws
        if ws is None:
            return
        try:
            ws.send(json.dumps(payload))
        except Exception as e:
            logger.debug("WS send failed: %s", e)

    def _run_forever(self):
        backoff = 1.0
        while not self._stop.is_set():
            token = self._token
            if not token:
                time.sleep(1.0)
                continue
            try:
                self._authed.clear()
                self._last_msg_at = None
                self._ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # OFFICIAL NOREN KEEPALIVE (NorenApi.py:119): WS-layer PING
                # every 3s with the heartbeat JSON as payload. This is BOTH
                # keepalive and dead-peer DETECTION — recv() fails within
                # seconds of a silently-dropped connection (the 11:05
                # incident: text-only heartbeat never detected it).
                self._ws.run_forever(
                    ping_interval=PING_INTERVAL_SEC,
                    ping_payload='{"t":"h"}',
                )
            except Exception as e:
                logger.warning("WS run_forever exception: %s", e)
            self._connected_at = None
            self._authed.clear()
            if self._stop.is_set():
                break
            # reconnect with capped exponential backoff
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _on_open(self, ws):
        self._connected_at = time.monotonic()
        self._last_msg_at = time.monotonic()
        # Watchdog: if the server goes half-dead (no frames at all, not even
        # PONG/heartbeat acks), force-close so run_forever exits and the
        # reconnect loop takes over. Runs as a daemon thread per connection.
        threading.Thread(target=self._msg_watchdog, args=(ws,), daemon=True,
                         name="flattrade-ws-watchdog").start()
        # DOCS §6.1: connection request MUST be the first message, with uid +
        # actid + source "API" + accesstoken. Without the ack, subscriptions
        # are silently dropped (verified live — zero ticks without it).
        self._send({
            "t": "a",
            "uid": settings.FLATTRADE_USER_ID,
            "actid": settings.FLATTRADE_USER_ID,
            "source": "API",
            "accesstoken": self._token,
        })
        logger.info("WS socket open — connect handshake sent (t:a), awaiting ack")

    def _msg_watchdog(self, ws):
        """Kills the connection if NO frame arrives for NO_MSG_WATCHDOG_SEC."""
        while (not self._stop.is_set()
               and self._ws is ws
               and self._last_msg_at is not None):
            time.sleep(5.0)
            last = self._last_msg_at
            if (self._ws is ws and last is not None
                    and time.monotonic() - last > NO_MSG_WATCHDOG_SEC):
                logger.warning("WS watchdog: no frames for %ds — forcing reconnect",
                               NO_MSG_WATCHDOG_SEC)
                try:
                    ws.keep_running = False
                    ws.close()
                except Exception:
                    pass
                return

    def _on_message(self, ws, message: str):
        self._last_msg_at = time.monotonic()
        try:
            res = json.loads(message)
        except (ValueError, TypeError):
            return
        t = res.get("t")

        if t == "ak":
            ok = str(res.get("s", "")).lower() == "ok"
            if ok:
                self._authed.set()
                logger.info("WS authed (ak: Ok) — subscribing %d instruments + spot",
                            len(self._subscribed))
                self._resubscribe_all()
                self._send({"t": "t", "k": f"{SPOT_SUBSCRIBE[0]}|{SPOT_SUBSCRIBE[1]}"})
            else:
                logger.error("WS connect REJECTED (ak s=%s) — token invalid/expired; "
                             "feed will retry on next token set", res.get("s"))
            return
        if t == "hk":
            return  # heartbeat ack

        if t not in ("tk", "tf", "dk", "df"):
            return
        exchange = str(res.get("e", ""))
        token = str(res.get("tk", ""))
        lp = res.get("lp")
        if not exchange or not token or lp in (None, ""):
            # tf updates may carry only OI/volume; treat as keep-alive
            with self._lock:
                self._last_tick_at[f"{exchange}|{token}"] = time.monotonic()
            return
        try:
            ltp = float(lp)
        except (TypeError, ValueError):
            return
        if ltp <= 0:
            return
        key = f"{exchange}|{token}"
        with self._lock:
            self._latest[key] = ltp
            self._last_tick_at[key] = time.monotonic()
        if self._on_tick is not None:
            try:
                self._on_tick(exchange, token, ltp)
            except Exception:
                pass

    def _on_error(self, ws, error):
        logger.warning("WS error: %s", error)

    def _on_close(self, ws, code, msg):
        self._connected_at = None
        self._authed.clear()
        logger.info("WS closed (code=%s) — reconnecting", code)
