"""Flattrade API Authentication Module with TOTP Auto-login."""

import hashlib
import logging
from typing import Optional, Dict, Any

from flattrade_bot.config import settings

logger = logging.getLogger(__name__)


class FlattradeAuth:
    """Manages session creation and token generation for Flattrade API."""

    def __init__(
        self,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        totp_key: Optional[str] = None,
    ):
        self.user_id = user_id or settings.FLATTRADE_USER_ID
        self.api_key = api_key or settings.FLATTRADE_API_KEY
        self.api_secret = api_secret or settings.FLATTRADE_API_SECRET
        self.totp_key = totp_key or settings.FLATTRADE_TOTP_KEY
        self.token: Optional[str] = None

    def generate_totp(self) -> str:
        """Generates 6-digit TOTP code using pyotp."""
        if not self.totp_key:
            raise ValueError("TOTP Key missing from configuration")
        import pyotp
        totp = pyotp.TOTP(self.totp_key)
        return totp.now()

    def generate_password_hash(self, password: str) -> str:
        """Generates SHA-256 hash of password as required by Flattrade NorenApi."""
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    async def get_token_from_request_code(self, request_code: str) -> Optional[str]:
        """Exchanges request_code from OAuth browser login for session token via authapi.flattrade.in/trade/apitoken."""
        import hashlib
        import socket
        import httpx

        # Force IPv4 socket resolution to match Flattrade Wall registered IPv4
        from flattrade_bot.broker.network import _ensure_ipv4_patch
        _ensure_ipv4_patch()

        url = "https://authapi.flattrade.in/trade/apitoken"

        # SHA256(api_key + request_code + api_secret)
        hash_input = f"{self.api_key}{request_code}{self.api_secret}"
        secret_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        payload = {
            "api_key": self.api_key,
            "request_code": request_code,
            "api_secret": secret_hash
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(url, json=payload)
                data = res.json()
                if data.get("stat") == "Ok" or data.get("status") == "Ok":
                    self.token = data.get("token")
                    logger.info(f"✅ Flattrade Token Generated Successfully: {self.token[:8]}...")
                    return self.token
                else:
                    logger.error(f"❌ Flattrade token generation failed: {data.get('emsg')}")
                    return None
        except Exception as e:
            logger.error(f"Error requesting token: {e}")
            return None

    async def login(self) -> Optional[str]:
        """Authenticates with Flattrade REST API and retrieves session token."""
        if not (self.user_id and self.api_key):
            logger.warning("Flattrade credentials missing. Running in simulation mode.")
            return None

        import json
        import httpx
        totp_code = self.generate_totp()

        # Flattrade NorenApi login payload format
        pwd_hash = hashlib.sha256(self.api_secret.encode("utf-8")).hexdigest()
        appkey_hash = hashlib.sha256(f"{self.user_id}{self.api_key}".encode("utf-8")).hexdigest()

        payload = {
            "apkversion": "js:1.0.0",
            "uid": self.user_id,
            "pwd": pwd_hash,
            "factor2": totp_code,
            "vc": self.user_id + "_U",
            "appkey": appkey_hash,
            "imei": "134.209.155.20",
            "source": "API"
        }

        url = f"{settings.FLATTRADE_API_URL}QuickAuth"
        body = f"jData={json.dumps(payload)}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(url, data=body)
                data = res.json()
                if data.get("stat") == "Ok":
                    self.token = data.get("susertoken")
                    logger.info(f"✅ Successfully authenticated with Flattrade. Token: {self.token[:8]}...")
                    return self.token
                else:
                    logger.warning(f"⚠️ Flattrade QuickAuth response: {data.get('emsg')}")
                    return None
        except Exception as e:
            logger.error(f"Flattrade auth request error: {e}")
            return None
