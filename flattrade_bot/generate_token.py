"""Flattrade Daily Token Generator & Broker Health Checker.

Run this script to:
1. Generate / refresh your daily Flattrade 24-hour Session Token.
2. Automatically save the token into your .env file.
3. Test live connectivity and fetch live Nifty 50 Spot quote.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx
import pyotp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattrade_bot.broker.client import FlattradeClient
from flattrade_bot.config import settings


async def get_token_from_request_code(api_key: str, api_secret: str, request_code: str) -> str | None:
    url = "https://authapi.flattrade.in/trade/apitoken"
    hash_input = f"{api_key}{request_code}{api_secret}"
    secret_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    payload = {
        "api_key": api_key,
        "request_code": request_code,
        "api_secret": secret_hash,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.post(url, json=payload)
        data = res.json()
        if data.get("stat") == "Ok" or data.get("status") == "Ok":
            return data.get("token")
        else:
            print(f"❌ Error from Flattrade API: {data.get('emsg', data)}")
            return None


def save_token_to_env(token: str):
    env_file = ROOT / ".env"
    lines = []
    found = False
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.startswith("FLATTRADE_TOKEN="):
                lines[i] = f"FLATTRADE_TOKEN={token}"
                found = True
                break
    if not found:
        lines.append(f"FLATTRADE_TOKEN={token}")
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ Token successfully saved to {env_file}")


async def test_quote(token: str):
    client = FlattradeClient(token)
    print("\n🔍 Querying live Nifty 50 Spot quote from Flattrade...")
    quote = await client.get_quotes(exchange="NSE", token="26000")
    if quote.get("stat") == "Ok" and "lp" in quote:
        print(f"  [PASS] Live Nifty Spot Price: Rs {quote.get('lp')}")
        print(f"  [PASS] High: Rs {quote.get('h')} | Low: Rs {quote.get('l')} | Close: Rs {quote.get('c')}")
        print("\n🎉 BROKER CONNECTIVITY 100% OPERATIONAL!")
    else:
        print(f"  [WARN] Quote response: {quote}")


async def main():
    print("=" * 70)
    print(" 🔑 FLATTRADE DAILY BROKER TOKEN GENERATOR")
    print("=" * 70)

    user_id = settings.FLATTRADE_USER_ID
    api_key = settings.FLATTRADE_API_KEY
    api_secret = settings.FLATTRADE_API_SECRET
    totp_key = settings.FLATTRADE_TOTP_KEY

    print(f"• User ID:    {user_id or '[MISSING]'}")
    print(f"• API Key:    {api_key[:8] + '...' if api_key else '[MISSING]'}")
    print(f"• API Secret: {'[CONFIGURED]' if api_secret else '[MISSING]'}")

    if not (user_id and api_key and api_secret):
        print("\n❌ Error: FLATTRADE_USER_ID, FLATTRADE_API_KEY, or FLATTRADE_API_SECRET missing in .env")
        return

    # Check if a token is passed via command line
    if len(sys.argv) > 1:
        req_code = sys.argv[1].strip()
        print(f"\nExchanging Request Code '{req_code}' for Token...")
        token = await get_token_from_request_code(api_key, api_secret, req_code)
        if token:
            print(f"✅ Token Acquired: {token[:12]}...")
            save_token_to_env(token)
            await test_quote(token)
        return

    # Option: Generate TOTP code to help login
    if totp_key:
        try:
            totp = pyotp.TOTP(totp_key)
            print(f"\n🔐 Current Live TOTP Code: {totp.now()}")
        except Exception:
            pass

    auth_url = f"https://auth.flattrade.in/?app_key={api_key}"
    print(f"\n👉 Login URL:")
    print(f"   {auth_url}\n")
    print("1. Open the URL above in any browser.")
    print("2. Log in with your User ID, Password, and TOTP.")
    print("3. After login, copy the 'request_code' from the redirected URL (e.g. ?request_code=XXXXXX).")
    print("4. Paste the request_code below and press Enter:\n")

    try:
        req_code = input("Enter Request Code: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    if not req_code:
        print("No request code entered. Exiting.")
        return

    token = await get_token_from_request_code(api_key, api_secret, req_code)
    if token:
        print(f"\n✅ Token Acquired: {token[:12]}...")
        save_token_to_env(token)
        await test_quote(token)


if __name__ == "__main__":
    asyncio.run(main())
