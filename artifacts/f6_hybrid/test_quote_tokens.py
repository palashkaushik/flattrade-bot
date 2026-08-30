import asyncio
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import httpx
from flattrade_bot.config import settings
from flattrade_bot.broker.network import _ensure_ipv4_patch

_ensure_ipv4_patch()

async def test_q():
    url = f"{settings.FLATTRADE_API_URL}GetQuotes"
    token = os.getenv("FLATTRADE_TOKEN", "")
    print(f"Testing with Token: {token[:12]}...")

    tokens_to_test = [
        ("NSE", "26000"),
        ("NSE", "Nifty 50"),
        ("NSE", "99992000"),
        ("NFO", "26000"),
    ]

    for exch, tk in tokens_to_test:
        body = f'jData={{"uid":"{settings.FLATTRADE_USER_ID}","exch":"{exch}","token":"{tk}"}}&jKey={token}'
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.post(url, data=body, headers=headers)
                print(f"[{exch}:{tk}] Response: {r.text}")
            except Exception as e:
                print(f"[{exch}:{tk}] Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_q())
