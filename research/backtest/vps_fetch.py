"""Run ON THE VPS: mint a token, then fetch the last available sessions."""
import os
import sys

sys.path.insert(0, "/home/ubuntu/FLATTRADE_BOT")
sys.path.insert(0, "/home/ubuntu/FT")
from flattrade_bot.config import settings  # noqa: E402
from flattrade_bot.broker.auto_login import automated_flattrade_login_rest  # noqa: E402

tok = automated_flattrade_login_rest(
    user_id=settings.FLATTRADE_USER_ID,
    password=settings.FLATTRADE_PASSWORD,
    totp_key=settings.FLATTRADE_TOTP_KEY,
    api_key=settings.FLATTRADE_API_KEY,
    api_secret=settings.FLATTRADE_API_SECRET,
)
print("token", "OK" if tok else "FAILED")
if not tok:
    raise SystemExit(1)
os.environ["FT_UID"] = settings.FLATTRADE_USER_ID
os.environ["FT_TOKEN"] = tok

import fetch_day  # noqa: E402

os.makedirs("/home/ubuntu/FT_DATA", exist_ok=True)
for day, tag in (("2026-10-30", "oct30"), ("2026-11-02", "nov02"), ("2026-11-03", "nov03"), ("2026-11-04", "nov04"), ("2026-11-05", "nov05"), ("2026-11-06", "nov06"), ("2026-11-09", "nov09")):
    fetch_day.main(day, "NIFTY09NOV26",
                   f"/home/ubuntu/FT_DATA/{tag}_options.csv",
                   f"/home/ubuntu/FT_DATA/{tag}_index.csv")
print("done")


