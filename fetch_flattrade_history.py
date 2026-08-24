"""Script to Download Historical Intraday Data from Flattrade API."""

import asyncio
import argparse
import pandas as pd

from flattrade_bot.broker.auth import FlattradeAuth
from flattrade_bot.broker.history import FlattradeHistoryFetcher


async def main():
    parser = argparse.ArgumentParser(description="Download Flattrade Historical Data")
    parser.add_argument("--token", default="26000", help="Scrip token (e.g. 26000 for Nifty 50)")
    parser.add_argument("--exchange", default="NSE", help="Exchange (NSE or NFO)")
    parser.add_argument("--days", type=int, default=5, help="Days of history to fetch")
    parser.add_argument("--out", default="flattrade_history.csv", help="Output CSV path")
    args = parser.parse_args()

    auth = FlattradeAuth()
    print("Authenticating with Flattrade...", flush=True)
    token = await auth.login()

    if not token:
        print("❌ Authentication failed. Please set FLATTRADE_USER_ID, FLATTRADE_API_KEY, FLATTRADE_API_SECRET, and FLATTRADE_TOTP_KEY in environment or .env file.")
        return

    fetcher = FlattradeHistoryFetcher(auth_token=token)
    print(f"Fetching {args.days} days of historical 1m data for token {args.token} on {args.exchange}...", flush=True)
    candles = await fetcher.fetch_historical_candles(
        token=args.token,
        exchange=args.exchange,
        interval="1",
        days_back=args.days
    )

    if candles:
        df = pd.DataFrame(candles)
        df.to_csv(args.out, index=False)
        print(f"✅ Downloaded {len(df)} candles and saved to {args.out}")
    else:
        print("❌ No historical candles retrieved.")


if __name__ == "__main__":
    asyncio.run(main())
