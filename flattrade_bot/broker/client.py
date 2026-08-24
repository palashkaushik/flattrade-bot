"""Flattrade REST API Client for Market Order Placement and Position Management."""

import logging
from typing import Dict, Any, Optional

import httpx

from flattrade_bot.config import settings
from flattrade_bot.broker.network import force_ipv4

logger = logging.getLogger(__name__)


class FlattradeClient:
    """Handles Flattrade REST API order execution and position tracking."""

    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token
        self.base_url = settings.FLATTRADE_API_URL
        self._client: Optional[httpx.AsyncClient] = None

    def set_token(self, token: str):
        self.auth_token = token

    def _get_client(self) -> httpx.AsyncClient:
        """Returns the persistent connection-pooled client (created once)."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=5.0)
        return self._client

    async def _post(self, url: str, body: str) -> Dict[str, Any]:
        """POSTs with connection reuse; keeps IPv4 pinning per request."""
        with force_ipv4():
            res = await self._get_client().post(url, data=body)
            return res.json()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def place_market_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: int,
        ltp: float,
        product: str = "MIS",  # Intraday
        slippage_buffer: float = 1.0,
    ) -> Dict[str, Any]:
        """Places a market buy/sell order via Flattrade API.

        Per API v2 rules, uses an Aggressive Limit Order (LTP + buffer for Buy, LTP - buffer for Sell)
        to guarantee immediate fill at market price without API v2 rejection.
        """
        limit_price = round(ltp + slippage_buffer if side.upper() == "BUY" else ltp - slippage_buffer, 2)
        buy_or_sell = "B" if side.upper() == "BUY" else "S"

        payload = {
            "uid": settings.FLATTRADE_USER_ID,
            "actid": settings.FLATTRADE_USER_ID,
            "trantype": buy_or_sell,
            "prd": product,
            "exch": "NFO",
            "tsym": symbol,
            "qty": str(quantity),
            "prctyp": "LMT",  # Aggressive Limit Order
            "prc": str(limit_price),
            "ret": "DAY",
            "remarks": "QuadRotation_Bot"
        }

        logger.info(f"🚀 Submitting Flattrade Order: {side} {quantity} {symbol} @ Limit ₹{limit_price:.2f} (LTP ₹{ltp:.2f})")

        if not self.auth_token:
            logger.warning("No auth token present. Order executed in simulation mode.")
            return {"stat": "Ok", "norenordno": "SIM_ORD_12345", "price": ltp}

        # Map product to NorenApi product code ("I" for MIS, "M" for NRML)
        prd_code = "I" if product.upper() in ("MIS", "I") else "M"
        payload["prd"] = prd_code

        url = f"{self.base_url}PlaceOrder"
        body = f"jData={json_dumps(payload)}&jKey={self.auth_token}"

        try:
            data = await self._post(url, body)
            if data.get("stat") == "Ok":
                logger.info(f"✅ Flattrade Order Placed Successfully. Order ID: {data.get('norenordno')}")
            else:
                logger.error(f"❌ Flattrade Order Failed: {data.get('emsg')}")
            return data
        except Exception as e:
            logger.error(f"Flattrade order error: {e}")
            return {"stat": "Not_Ok", "emsg": str(e)}

    async def get_positions(self) -> Dict[str, Any]:
        """Fetches current open positions from Flattrade."""
        if not self.auth_token:
            return {"stat": "Ok", "positions": []}

        url = f"{self.base_url}PositionBook"
        body = f"jData={{\"uid\":\"{settings.FLATTRADE_USER_ID}\",\"actid\":\"{settings.FLATTRADE_USER_ID}\"}}&jKey={self.auth_token}"

        try:
            return await self._post(url, body)
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {"stat": "Not_Ok", "emsg": str(e)}

    async def get_order_book(self) -> Dict[str, Any]:
        """Fetches broker order status for live-fire verification."""
        if not self.auth_token:
            return {"stat": "Ok", "orders": []}

        url = f"{self.base_url}OrderBook"
        body = f"jData={{\"uid\":\"{settings.FLATTRADE_USER_ID}\"}}&jKey={self.auth_token}"

        try:
            return await self._post(url, body)
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return {"stat": "Not_Ok", "emsg": str(e)}

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancels a pending order during fill confirmation."""
        if not self.auth_token:
            return {"stat": "Ok", "result": "SIM_CANCEL"}

        url = f"{self.base_url}CancelOrder"
        body = (
            f"jData={{\"uid\":\"{settings.FLATTRADE_USER_ID}\","
            f"\"norenordno\":\"{order_id}\"}}&jKey={self.auth_token}"
        )

        try:
            return await self._post(url, body)
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {"stat": "Not_Ok", "emsg": str(e)}

    async def get_trade_book(self) -> Dict[str, Any]:
        """Fetches filled trades for confirming whether an order executed."""
        if not self.auth_token:
            return {"stat": "Ok", "trades": []}

        url = f"{self.base_url}TradeBook"
        body = (
            f"jData={{\"uid\":\"{settings.FLATTRADE_USER_ID}\","
            f"\"actid\":\"{settings.FLATTRADE_USER_ID}\"}}&jKey={self.auth_token}"
        )

        try:
            return await self._post(url, body)
        except Exception as e:
            logger.error(f"Error fetching trade book: {e}")
            return {"stat": "Not_Ok", "emsg": str(e)}


def json_dumps(payload: Dict[str, Any]) -> str:
    """JSON-serializes payloads (single import point keeps encoding consistent)."""
    import json
    return json.dumps(payload)