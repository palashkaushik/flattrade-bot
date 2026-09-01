"""Flattrade Historical Data Fetcher via TPSeries REST API."""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import httpx

from flattrade_bot.config import settings
from flattrade_bot.broker.network import force_ipv4

logger = logging.getLogger(__name__)


class SessionExpiredError(RuntimeError):
    """Raised when Flattrade rejects a request with an expired session token."""


def is_session_expired_response(data: Any) -> bool:
    """Identifies the broker's expired/invalid session responses."""
    if not isinstance(data, dict):
        return False
    message = str(data.get("emsg", "")).lower()
    return "session expired" in message or "invalid session" in message


class FlattradeHistoryFetcher:
    """Fetches historical OHLC candle data from Flattrade API (TPSeries endpoint)."""

    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token
        self.base_url = settings.FLATTRADE_API_URL
        self._client: Optional[httpx.AsyncClient] = None

    def set_token(self, token: str):
        self.auth_token = token

    def _get_client(self) -> httpx.AsyncClient:
        """Returns the persistent connection-pooled client (created once)."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=2.0),
                limits=httpx.Limits(max_keepalive_connections=30, max_connections=50, keepalive_expiry=30.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post(self, url: str, body: str) -> Any:
        """POSTs with connection reuse; keeps IPv4 pinning per request."""
        with force_ipv4():
            res = await self._get_client().post(url, data=body)
            return res.json()

    async def fetch_historical_candles(
        self,
        token: str,
        exchange: str = "NFO",
        interval: str = "1",  # 1-minute candles
        days_back: int = 5,
    ) -> List[Dict[str, Any]]:
        """Fetches historical time-series candles for a specific instrument token.

        Parameters:
          token: Instrument token (e.g. "26000" for Nifty 50 Spot, or Option Token).
          exchange: "NSE" or "NFO".
          interval: Candle timeframe in minutes ("1", "3", "5", "15").
          days_back: Number of past days to fetch.

        Returns:
          List of candle dicts with keys: 'time', 'open', 'high', 'low', 'close', 'volume'.
        """
        if not self.auth_token:
            logger.warning("Auth token missing. Cannot fetch historical data from Flattrade.")
            return []

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        st_unix = str(int(start_time.timestamp()))
        et_unix = str(int(end_time.timestamp()))

        payload = {
            "uid": settings.FLATTRADE_USER_ID,
            "exch": exchange,
            "token": token,
            "st": st_unix,
            "et": et_unix,
            "intrv": interval,
        }

        url = f"{self.base_url}TPSeries"
        body = f"jData={json.dumps(payload)}&jKey={self.auth_token}"

        try:
            data = await self._post(url, body)

            if is_session_expired_response(data):
                raise SessionExpiredError(data.get("emsg", "Session expired"))
            
            if isinstance(data, list):
                candles = []
                for row in data:
                    candles.append({
                        "time": row.get("time"),
                        "open": float(row.get("into", 0.0)),
                        "high": float(row.get("inth", 0.0)),
                        "low": float(row.get("intl", 0.0)),
                        "close": float(row.get("intc", 0.0)),
                        "volume": float(row.get("intv", row.get("v", row.get("vol", 0.0)))),
                    })
                # Flattrade TPSeries returns candles in reverse chronological order (newest first).
                # Reverse list so candles are in strict chronological order (oldest first, newest last).
                candles.reverse()
                logger.info(f"✅ Downloaded {len(candles)} historical candles for token {token} from Flattrade.")
                return candles
            else:
                logger.error(f"❌ Flattrade TPSeries error: {data.get('emsg') if isinstance(data, dict) else data}")
                return []
        except SessionExpiredError:
            raise
        except Exception as e:
            logger.error(f"Flattrade historical fetch error: {e}")
            return []

    async def fetch_daily_candles(
        self,
        tradingsymbol: str,
        exchange: str = "NFO",
        days_back: int = 10,
    ) -> List[Dict[str, Any]]:
        """Fetches TRUE daily EOD candles via Flattrade's /EODChartData endpoint.

        This is the official daily-candle source (same data TradingView charts).
        Takes the trading symbol (not token): sym format = "NFO:NIFTY01SEP26C23950".

        Returns chronological list of dicts: {'time': '31-08-2026', 'open','high','low','close','volume'}
        — one candle per trading day with the OFFICIAL session close.
        """
        if not self.auth_token:
            logger.warning("Auth token missing. Cannot fetch daily candles from Flattrade.")
            return []

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        payload = {
            "uid": settings.FLATTRADE_USER_ID,
            "sym": f"{exchange}:{tradingsymbol}",
            "from": str(int(start_time.timestamp())),
            "to": str(int(end_time.timestamp())),
        }
        url = f"{self.base_url}EODChartData"
        body = f"jData={json.dumps(payload)}&jKey={self.auth_token}"

        try:
            data = await self._post(url, body)
            if is_session_expired_response(data):
                raise SessionExpiredError(data.get("emsg", "Session expired"))
            if isinstance(data, list):
                candles = []
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    # API returns "DD-MON-YYYY" (e.g. "31-AUG-2026"); normalize to DD-MM-YYYY
                    raw_t = str(row.get("time", ""))
                    try:
                        if "-" in raw_t and raw_t.count("-") == 2 and not raw_t[2].isdigit():
                            dt_obj = datetime.strptime(raw_t, "%d-%b-%Y")
                            norm_t = dt_obj.strftime("%d-%m-%Y")
                        else:
                            norm_t = raw_t
                    except ValueError:
                        norm_t = raw_t
                    candles.append({
                        "time": norm_t,
                        "open": float(row.get("into", 0.0)),
                        "high": float(row.get("inth", 0.0)),
                        "low": float(row.get("intl", 0.0)),
                        "close": float(row.get("intc", 0.0)),
                        "volume": float(row.get("intv", 0.0)),
                    })
                # API returns newest first; normalize to chronological order
                candles.reverse()
                logger.info(f"✅ Downloaded {len(candles)} daily EOD candles for {tradingsymbol}.")
                return candles
            logger.error(f"❌ Flattrade EODChartData error: {data.get('emsg') if isinstance(data, dict) else data}")
            return []
        except SessionExpiredError:
            raise
        except Exception as e:
            logger.error(f"Flattrade daily candle fetch error: {e}")
            return []

    async def fetch_live_quote(self, token: str = "26000", exchange: str = "NSE") -> Optional[Dict[str, Any]]:
        """Fetches live real-time quote (LTP, Open, High, Low, Close) for an instrument token."""
        if not self.auth_token:
            return None

        payload = {
            "uid": settings.FLATTRADE_USER_ID,
            "exch": exchange,
            "token": token
        }
        url = f"{self.base_url}GetQuotes"
        body = f"jData={json.dumps(payload)}&jKey={self.auth_token}"

        try:
            data = await self._post(url, body)
            if is_session_expired_response(data):
                raise SessionExpiredError(data.get("emsg", "Session expired"))
            if data.get("stat") == "Ok":
                return {
                    "lp": float(data.get("lp", 0.0)),
                    "open": float(data.get("o", 0.0)),
                    "high": float(data.get("h", 0.0)),
                    "low": float(data.get("l", 0.0)),
                    "close": float(data.get("c", 0.0)),
                    "volume": float(data.get("v", 0.0)),
                }
            return None
        except SessionExpiredError:
            raise
        except Exception as e:
            logger.error(f"Error fetching quote for token {token}: {e}")
            return None

    async def search_futures_token(self, symbol_text: str = "NIFTY") -> Optional[Dict[str, str]]:
        """Searches Flattrade NFO security master for an index futures contract token."""
        if not self.auth_token:
            return None
        payload = {
            "uid": settings.FLATTRADE_USER_ID,
            "stext": symbol_text,
            "exch": "NFO",
        }
        url = f"{self.base_url}SearchScrip"
        body = f"jData={json.dumps(payload)}&jKey={self.auth_token}"
        try:
            data = await self._post(url, body)
            if is_session_expired_response(data):
                raise SessionExpiredError(data.get("emsg", "Session expired"))
            if data.get("stat") == "Ok" and data.get("values"):
                for item in data["values"]:
                    tsym = item.get("tsym", "")
                    dname = item.get("dname", "").strip()
                    # The front-month index futures contract has tsym == symbol_text
                    # and a dname that mentions FUT (not CE/PE option legs).
                    if tsym == symbol_text and "FUT" in dname.upper() and "CE" not in tsym and "PE" not in tsym:
                        return {
                            "token": item["token"],
                            "tsym": item["tsym"],
                            "dname": item["dname"].strip(),
                        }
                # Prefer the actual NIFTY futures contract over the broker's
                # FPI instrument, whose historical series is sparse and not
                # the NIFTY index-futures price series.
                futures = [
                    item
                    for item in data["values"]
                    if "FUT" in item.get("dname", "").upper()
                    and "CE" not in item.get("dname", "").upper()
                    and "PE" not in item.get("dname", "").upper()
                ]
                preferred = [
                    item
                    for item in futures
                    if item.get("tsym", "").upper().startswith("NIFTY")
                    and len(item.get("tsym", "")) > 5
                    and item.get("tsym", "")[5].isdigit()
                    and "FPI" not in item.get("tsym", "").upper()
                    and "FPI" not in item.get("dname", "").upper()
                ]
                # Fallback: first remaining value whose dname mentions FUT.
                for item in preferred + futures:
                    dname = item.get("dname", "").strip()
                    if "FUT" in dname.upper() and "CE" not in dname.upper() and "PE" not in dname.upper():
                        return {
                            "token": item["token"],
                            "tsym": item["tsym"],
                            "dname": dname,
                        }
            return None
        except SessionExpiredError:
            raise
        except Exception as e:
            logger.error(f"Error searching futures token for '{symbol_text}': {e}")
            return None

    async def search_option_token(self, symbol_text: str) -> Optional[Dict[str, str]]:
        """Searches Flattrade NFO security master for option contract token and symbol name."""
        if not self.auth_token:
            return None

        payload = {
            "uid": settings.FLATTRADE_USER_ID,
            "stext": symbol_text,
            "exch": "NFO"
        }
        url = f"{self.base_url}SearchScrip"
        body = f"jData={json.dumps(payload)}&jKey={self.auth_token}"

        try:
            data = await self._post(url, body)
            if is_session_expired_response(data):
                raise SessionExpiredError(data.get("emsg", "Session expired"))
            if data.get("stat") == "Ok" and data.get("values"):
                item = data["values"][0]
                return {
                    "token": item["token"],
                    "tsym": item["tsym"],
                    "dname": item["dname"].strip()
                }
            return None
        except SessionExpiredError:
            raise
        except Exception as e:
            logger.error(f"Error searching option token for '{symbol_text}': {e}")
            return None
