"""Auth via headless Selenium (OAuth) then fetch Nifty spot 1m for Aug 12-14, 2026."""
import asyncio, json, sys, os
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.getcwd())
from flattrade_bot.config import settings
from flattrade_bot.broker.auto_login import automated_flattrade_login
from flattrade_bot.broker.network import force_ipv4
import httpx

IST = ZoneInfo("Asia/Kolkata")
TOKEN = "26000"; EXCH = "NSE"; INTERVAL = "1"
st = datetime(2026, 8, 12, 9, 10, 0, tzinfo=IST)
et = datetime(2026, 8, 14, 15, 35, 0, tzinfo=IST)

async def main():
    token = automated_flattrade_login(
        user_id=settings.FLATTRADE_USER_ID,
        password=settings.FLATTRADE_PASSWORD,
        totp_key=settings.FLATTRADE_TOTP_KEY,
        api_key=settings.FLATTRADE_API_KEY,
        api_secret=settings.FLATTRADE_API_SECRET,
        headless=True,
    )
    if not token:
        print("AUTH_FAILED")
        return
    print("AUTH_OK", len(token))
    payload = {"uid": settings.FLATTRADE_USER_ID, "exch": EXCH, "token": TOKEN,
               "st": str(int(st.timestamp())), "et": str(int(et.timestamp())), "intrv": INTERVAL}
    url = f"{settings.FLATTRADE_API_URL}TPSeries"
    body = f"jData={json.dumps(payload)}&jKey={token}"
    with force_ipv4():
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(url, data=body)
            data = res.json()
    if isinstance(data, list):
        candles = []
        for row in data:
            candles.append({"time": row.get("time"), "open": float(row.get("into",0)),
                            "high": float(row.get("inth",0)), "low": float(row.get("intl",0)),
                            "close": float(row.get("intc",0)), "v": float(row.get("v",0))})
        candles.reverse()
        print("CANDLES", len(candles))
        if candles:
            print("FIRST", candles[0]); print("LAST", candles[-1])
        json.dump(candles, open("artifacts/f6_hybrid/nifty_spot_aug12_14_2026.json","w"), indent=1)
        print("SAVED")
    else:
        print("ERR", str(data)[:500])

asyncio.run(main())
