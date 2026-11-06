"""Fetch one trading day (nearest-expiry option ladder + NSE index) from Flattrade.

    set FT_UID / FT_TOKEN   (from the VPS .env)
    python fetch_day.py 2026-11-09 NIFTY09NOV26 data/nov09_options.csv data/nov09_index.csv
"""
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen

IST = timezone(timedelta(hours=5, minutes=30))
BASE = "https://piconnect.flattrade.in/PiConnectAPI/"
UID = os.environ.get("FT_UID", "")
TOK = os.environ.get("FT_TOKEN", "")


def post(ep, p):
    body = "jData=" + json.dumps(p) + "&jKey=" + TOK
    for _ in range(4):
        try:
            req = Request(BASE + ep, data=body.encode(),
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(2)
    return None


def series(token, exchange, st, et, intrv="1"):
    r = post("TPSeries", {"uid": UID, "exch": exchange, "token": token,
                          "st": str(st), "et": str(et), "intrv": intrv})
    return [x for x in (r if isinstance(r, list) else [])
            if isinstance(x, dict) and x.get("stat") == "Ok"]


def main(day, expiry, out_opts, out_idx):
    d = datetime.strptime(day, "%Y-%m-%d")
    st = int(datetime(d.year, d.month, d.day, 9, 0, tzinfo=IST).timestamp())
    et = int(datetime(d.year, d.month, d.day, 15, 45, tzinfo=IST).timestamp())

    ix = series("26000", "NSE", st, et)
    with open(out_idx, "w", encoding="utf-8") as fh:
        fh.write("date,open,high,low,close\n")
        for x in ix:
            fh.write(f"{x['time']},{x['into']},{x['inth']},{x['intl']},{x['intc']}\n")
    print("index", len(ix))

    # spot at 09:15 -> ladder strikes
    spot = None
    for x in ix:
        if x["time"].endswith("09:15:00"):
            spot = float(x["into"]); break
    atm = int(round(spot / 50.0) * 50) if spot else None
    print("spot", spot, "atm", atm)

    strikes = [(atm + k, side) for k in range(-600, 601, 50) for side in ("CE", "PE")]
    rows, seen = [], set()
    for strike, side in strikes:
        r = post("SearchScrip", {"uid": UID, "exch": "NFO", "stext": f"NIFTY {strike} {side}"})
        tok = tsym = None
        for v in ((r or {}).get("values") or []):
            if v.get("tsym") == f"{expiry}{'C' if side == 'CE' else 'P'}{strike}":
                tok, tsym = v["token"], v["tsym"]; break
        if not tok or tsym in seen:
            continue
        seen.add(tsym)
        sym_out = f"{expiry}{strike}{side}"
        n = 0
        for x in series(tok, "NFO", st, et):
            rows.append((x["time"], sym_out,
                         x["into"], x["inth"], x["intl"], x["intc"], x.get("intv", 0)))
            n += 1
        print(tsym, n)
        time.sleep(0.05)

    with open(out_opts, "w", encoding="utf-8") as fh:
        fh.write("date,time,symbol,open,high,low,close,volume\n")
        for t, sym, o, h, l, c, v in rows:
            dt = datetime.strptime(t, "%d-%m-%Y %H:%M:%S")
            fh.write(f"{dt:%Y-%m-%d},{dt:%H:%M:%S},{sym},{o},{h},{l},{c},{v}\n")
    print("wrote", out_opts, len(rows), "rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


