"""Explicit B17 Smart Fib live-fire verification.

Resolves the latest B17 combined event from live Flattrade data (spot +
tracked option rows), computes fib SL/TP levels, and submits ONE
confirmed one-lot order. Market closed -> order path is exercised without
execution. Requires the exact confirmation phrase before submitting.
"""

import argparse
import asyncio
from datetime import date, datetime
import os

from flattrade_bot.broker.auto_login import automated_flattrade_login
from flattrade_bot.broker.client import FlattradeClient
from flattrade_bot.broker.history import FlattradeHistoryFetcher
from flattrade_bot.config import settings
from flattrade_bot.strategies.smart_fib_combined import (
    LiveSmartFibCombinedStrategy,
    _row_minute,
)


CONFIRMATION = "I_UNDERSTAND_ONE_LOT_LIVE_ORDER_B17"


def _safe_response(data):
    if not isinstance(data, dict):
        return data
    keys = ("stat", "emsg", "norenordno", "result", "status", "rejreason", "remarks")
    return {key: data[key] for key in keys if key in data}


async def main():
    parser = argparse.ArgumentParser(description="Fire one explicitly confirmed B17 Smart Fib live test order")
    parser.add_argument("--confirm-live-order", required=True)
    parser.add_argument("--quantity", type=int, default=settings.LOT_SIZE)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Skip the order; print what would fire")
    args = parser.parse_args()

    if args.confirm_live_order != CONFIRMATION:
        raise SystemExit(f"Refusing live order. Pass --confirm-live-order {CONFIRMATION}")
    if args.quantity <= 0 or args.quantity % settings.LOT_SIZE != 0:
        raise SystemExit(f"Quantity must be a positive multiple of lot size {settings.LOT_SIZE}")

    token = os.getenv("FLATTRADE_TOKEN", "")
    if not token:
        token = automated_flattrade_login(
            user_id=settings.FLATTRADE_USER_ID,
            password=settings.FLATTRADE_PASSWORD,
            totp_key=settings.FLATTRADE_TOTP_KEY,
            api_key=settings.FLATTRADE_API_KEY,
            api_secret=settings.FLATTRADE_API_SECRET,
            headless=not args.no_headless,
        )
    if not token:
        raise SystemExit("Live login failed; no order submitted")

    client = FlattradeClient(token)
    history = FlattradeHistoryFetcher(token)

    strat = LiveSmartFibCombinedStrategy(timeframes=settings.B17_TIMEFRAMES)
    strat.set_today(date.today())

    spot_candles = await history.fetch_historical_candles("26000", "NSE", "1", 3)
    if not spot_candles:
        raise SystemExit("Could not read Nifty spot; no order submitted")
    strat.add_spot_rows(spot_candles)
    spot = spot_candles[-1]["close"]

    now = datetime.now()
    current_min = now.hour * 60 + now.minute
    candidates = strat.candidate_strikes(spot, current_min)
    half = len(candidates) // 2
    tracked = {}
    for idx, strike in enumerate(candidates):
        side = "CE" if idx < half else "PE"
        key = f"{side}:{int(strike)}"
        info = await history.search_option_token(f"NIFTY {int(strike)} {side}")
        if not info:
            continue
        tracked[key] = info
        rows = await history.fetch_historical_candles(info["token"], "NFO", "1", 2)
        strat.add_contract_rows(side, int(strike), info["tsym"], info["token"], rows)
    if not tracked:
        raise SystemExit("Could not resolve option contracts; no order submitted")

    closed_minutes = [
        _row_minute(row)
        for row in spot_candles
        if str(row.get("time", "")).split(" ")[0] == date.today().strftime("%d-%m-%Y")
    ]
    if not closed_minutes:
        raise SystemExit("No closed minutes for today in spot feed; no order submitted")
    events = strat.evaluate(max(closed_minutes))
    if not events:
        raise SystemExit("No B17 events from live data; no order submitted")

    event = events[-1]
    side = str(event["side"])
    strike = int(event["strike"])
    key = f"{side}:{strike}"
    info = tracked.get(key)
    if not info:
        raise SystemExit(f"No tracked contract for {key}; no order submitted")

    sl_level, tp_level, price_rise, monitor = strat.exit_levels(event)
    entry = float(event.get("option_entry", 0.0))
    if entry <= 0:
        raise SystemExit("B17 event has no option entry price; no order submitted")

    monitor_token = "26000" if monitor == "index" else info["token"]
    monitor_exchange = "NSE" if monitor == "index" else "NFO"

    print(
        f"B17 EVENT: {side} {info['tsym']} @min {int(event['minute'])} entry ₹{entry:.2f} | "
        f"fib SL ₹{sl_level:.2f} TP ₹{tp_level:.2f} (rise={price_rise}, monitor={monitor} {monitor_token}) | "
        f"spot={spot:.2f} time={now.isoformat(timespec='seconds')}"
    )
    if args.dry_run:
        print("DRY_RUN=1 — no order submitted")
        return

    response = await client.place_market_order(
        symbol=info["tsym"],
        side="BUY",
        quantity=args.quantity,
        ltp=entry,
        product="MIS",
        slippage_buffer=1.0,
    )
    print(f"PLACE_ORDER_RESPONSE={_safe_response(response)}")

    order_book = await client.get_order_book()
    print(f"ORDER_BOOK_RESPONSE={_safe_response(order_book)}")
    trade_book = await client.get_trade_book()
    print(f"TRADE_BOOK_RESPONSE={_safe_response(trade_book)}")


if __name__ == "__main__":
    asyncio.run(main())