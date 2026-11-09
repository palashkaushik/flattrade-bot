"""Fetch the 8 Nov 2026 Muhurat session (evening window) for the 09NOV26 ladder.

Runs ON THE VPS (registered IP). Window 17:30-19:30 IST, ladder +/-600 around the
session's opening spot. Output uses the CSV convention (strike-first symbols).
"""
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, "/home/ubuntu/FLATTRADE_BOT")
sys.path.insert(0, "/home/ubuntu/FT")
from flattrade_bot.config import settings  # noqa: E402
from flattrade_bot.broker.auto_login import automated_flattrade_login_rest  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
DAY = "2026-11-08"
EXPIRY = "NIFTY09NOV26"
OUT_OPT = "/home/ubuntu/FT_DATA/nov08_options.csv"
OUT_IDX = "/home/ubuntu/FT_DATA/nov08_index.csv"

tok = automated_flattrade_login_rest(
    user_id=settings.FLATTRADE_USER_ID, password=settings.FLATTRADE_PASSWORD,
    totp_key=settings.FLATTRADE_TOTP_KEY, api_key=settings.FLATTRADE_API_KEY,
    api_secret=settings.FLATTRADE_API_SECRET)
print("token", "OK" if tok else "FAILED")
if not tok:
    raise SystemExit(1)
os.environ["FT_UID"] = settings.FLATTRADE_USER_ID
os.environ["FT_TOKEN"] = tok
import fetch_day as F  # noqa: E402

d = datetime.strptime(DAY, "%Y-%m-%d")
st = int(datetime(d.year, d.month, d.day, 13, 30, tzinfo=IST).timestamp())
et = int(datetime(d.year, d.month, d.day, 16, 0, tzinfo=IST).timestamp())

ix = F.series("26000", "NSE", st, et)
with open(OUT_IDX, "w", encoding="utf-8") as fh:
    fh.write("date,open,high,low,close\n")
    for x in ix:
        fh.write(f"{x['time']},{x['into']},{x['inth']},{x['intl']},{x['intc']}\n")
print("index bars", len(ix))
if not ix:
    raise SystemExit("no index bars — check the Muhurat window")

spot = float(ix[0]["into"])
atm = int(round(spot / 50.0) * 50)
print("muhurat open", spot, "atm", atm)

rows = []
for strike in range(atm - 600, atm + 601, 50):
    for side in ("CE", "PE"):
        r = F.post("SearchScrip", {"uid": F.UID, "exch": "NFO",
                                   "stext": f"NIFTY {strike} {side}"})
        tok_c = None
        want = f"{EXPIRY}{'C' if side == 'CE' else 'P'}{strike}"
        for v in ((r or {}).get("values") or []):
            if v.get("tsym") == want:
                tok_c = v["token"]; break
        if not tok_c:
            continue
        n = 0
        for x in F.series(tok_c, "NFO", st, et):
            rows.append((x["time"], f"{EXPIRY}{strike}{side}",
                         x["into"], x["inth"], x["intl"], x["intc"], x.get("intv", 0)))
            n += 1
        if n:
            print(want, n)

with open(OUT_OPT, "w", encoding="utf-8") as fh:
    fh.write("date,time,symbol,open,high,low,close,volume\n")
    for t, sym, o, h, l, c, v in rows:
        dt = datetime.strptime(t, "%d-%m-%Y %H:%M:%S")
        fh.write(f"{dt:%Y-%m-%d},{dt:%H:%M:%S},{sym},{o},{h},{l},{c},{v}\n")
print("wrote", OUT_OPT, len(rows), "rows")

