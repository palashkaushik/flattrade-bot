"""Flattrade WebSocket tick feed — replaces 1s REST quote polling.

Why: the REST poller fires ~7 GetQuotes/second (~420/min), which runs into
Flattrade's retail API throttling under load -> the stuck-then-alive price
pattern on the dashboard. The WebSocket subscribes each instrument ONCE and
the broker PUSHES every tick — no polling, no rate limit, ~50ms latency.

Protocol (extracted from the official NorenRestApiPy SDK + Flattrade's
api_helper):
  - URL:      wss://piconnect.flattrade.in/PiConnectWSAPI/{access_token}
  - Heartbeat: {"t":"h"} every 3s (broker expects it; SDK uses ping_interval=3)
  - Subscribe: {"t":"t","k":"NFO|<token>"}   (touchline feed)
  - Ticks:     {"t":"tk","lp":"123.45","e":"NFO","tk":"<token>", ...}
               (also "tf" = touchline-formatted variant; both handled)

Design:
  - Dedicated daemon thread runs websocket-client's run_forever with auto
    reconnect (exponential backoff, capped).
  - On connect: re-subscribes ALL active tokens + NIFTY spot (26000@NSE).
  - Ticks land in a thread-safe dict; the async engine reads them every tick
    loop — no locks held during processing (dict swap semantics).
  - Staleness watchdog: if no tick from an instrument for N seconds, the main
    loop automatically falls back to REST GetQuotes for that instrument
    (belt-and-braces; also covers WS outages transparently).
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

WS_URL_TEMPLATE = "wss://piconnect.flattrade.in/PiConnectWSAPI/{token}"
HEARTBEAT_SEC = 3.0
SPOT_SUBSCRIBE = ("NSE", "26000")  # Nifty 50 index


class FlattradeWebSocketFeed:
    """Push-based tick feed with auto-reconnect and REST-fallback hooks."""

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
        self._on_tick: Optional[Callable[[str, str, float], None]] = None

    # ------------------------------------------------------------------ state
    @property
    def connected(self) -> bool:
        return (self._connected_at is not None
                and self._ws is not None
                and time.monotonic() - self._connected_at < 30)

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
        self._send({"t": "t", "k": key})

    def unsubscribe(self, exchange: str, token: str):
        key = f"{exchange}|{token}"
        with self._lock:
            self._subscribed.discard(key)
        self._send({"t": "u", "k": key})

    def _resubscribe_all(self):
        with self._lock:
            keys = list(self._subscribed)
        for k in keys:
            self._send({"t": "t", "k": k})
        if keys:
            logger.info("WS: re-subscribed %d instruments", len(keys))

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
            url = WS_URL_TEMPLATE.format(token=token)
            try:
                self._ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # ping_interval matches the SDK's heartbeat cadence
                self._ws.run_forever(ping_interval=HEARTBEAT_SEC, ping_payload='{"t":"h"}')
            except Exception as e:
                logger.warning("WS run_forever exception: %s", e)
            self._connected_at = None
            if self._stop.is_set():
                break
            # reconnect with capped exponential backoff
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _on_open(self, ws):
        self._connected_at = time.monotonic()
        logger.info("WS connected — subscribing %d instruments + spot", len(self._subscribed))
        self._resubscribe_all()
        self._send({"t": "t", "k": f"{SPOT_SUBSCRIBE[0]}|{SPOT_SUBSCRIBE[1]}"})

    def _on_message(self, ws, message: str):
        try:
            res = json.loads(message)
        except (ValueError, TypeError):
            return
        t = res.get("t")
        if t not in ("tk", "tf", "dk", "df"):
            return
        exchange = str(res.get("e", ""))
        token = str(res.get("tk", res.get("ts", "")))
        lp = res.get("lp")
        if not exchange or not token or lp in (None, ""):
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
        logger.info("WS closed (code=%s) — reconnecting", code)
