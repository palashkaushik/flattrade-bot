"""Explicit one-lot Flattrade order-fire verification.

This is intentionally separate from the test suite and requires an exact
confirmation phrase before submitting a live order.
"""

import argparse
import asyncio
from datetime import datetime
import os

from flattrade_bot.broker.auto_login import automated_flattrade_login
from flattrade_bot.broker.client import FlattradeClient
from flattrade_bot.broker.history import FlattradeHistoryFetcher
from flattrade_bot.config import settings
from flattrade_bot.strategies.quad_pinbar_divergence import QuadPinbarDivergenceStrategy


CONFIRMATION = "I_UNDERSTAND_ONE_LOT_LIVE_ORDER"


def _safe_response(data):
    if not isinstance(data, dict):
        return data
    keys = ("stat", "emsg", "norenordno", "result", "status", "rejreason", "remarks")
    return {key: data[key] for key in keys if key in data}


async def main():
    parser = argparse.ArgumentParser(description="Fire one explicitly confirmed live Flattrade test order")
    parser.add_argument("--confirm-live-order", required=True)
    parser.add_argument("--contract-side", choices=("CE", "PE"), default="CE")
    parser.add_argument("--order-side", choices=("BUY", "SELL"), default="BUY")
    parser.add_argument("--quantity", type=int, default=settings.LOT_SIZE)
    parser.add_argument("--no-headless", action="store_true")
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
    strategy = QuadPinbarDivergenceStrategy()
    spot_candles = await history.fetch_historical_candles("26000", "NSE", "1", 1)
    if not spot_candles:
        raise SystemExit("Could not read Nifty spot; no order submitted")

    spot = spot_candles[-1]["close"]
    ce_strike, pe_strike = strategy.get_itm2_strikes(spot)
    strike = ce_strike if args.contract_side == "CE" else pe_strike
    info = await history.search_option_token(f"NIFTY {strike} {args.contract_side}")
    if not info:
        raise SystemExit("Could not resolve current option contract; no order submitted")

    quote = await history.fetch_live_quote(info["token"], "NFO")
    if not quote or quote.get("lp", 0.0) <= 0:
        raise SystemExit("Could not read current option LTP; no order submitted")

    print(
        f"Submitting {args.order_side} {args.quantity} {info['tsym']} "
        f"({info['dname']}) at LTP {quote['lp']:.2f}; spot={spot:.2f}; "
        f"time={datetime.now().isoformat(timespec='seconds')}"
    )
    response = await client.place_market_order(
        symbol=info["tsym"],
        side=args.order_side,
        quantity=args.quantity,
        ltp=quote["lp"],
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
