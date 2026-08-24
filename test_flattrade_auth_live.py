import asyncio
import logging
import sys
from flattrade_bot.broker.auth import FlattradeAuth
from flattrade_bot.broker.client import FlattradeClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def test_auth():
    auth = FlattradeAuth()
    print("Testing TOTP generation...")
    try:
        totp = auth.generate_totp()
        print(f"Generated TOTP Code: {totp}")
    except Exception as e:
        print(f"TOTP Generation Error: {e}")
        return

    print("Attempting Flattrade QuickAuth login...")
    token = await auth.login()
    if token:
        print(f"SUCCESS! Flattrade Token obtained: {token}")
        client = FlattradeClient(auth_token=token)
        print("Fetching position book...")
        positions = await client.get_positions()
        print(f"Positions Response: {positions}")
    else:
        print("FAILED to obtain Flattrade Token.")

if __name__ == "__main__":
    asyncio.run(test_auth())
