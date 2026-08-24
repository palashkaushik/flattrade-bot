import asyncio
from unittest.mock import patch

from flattrade_bot.utils.discord import DiscordNotifier


class FakeResponse:
    def __init__(self):
        self.raise_called = False

    def raise_for_status(self):
        self.raise_called = True


class FakeAsyncClient:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, *_args, **_kwargs):
        return self.response


def test_discord_notifier_checks_webhook_response_status():
    response = FakeResponse()
    client = FakeAsyncClient(response)

    with patch("httpx.AsyncClient", return_value=client):
        asyncio.run(DiscordNotifier("https://discord.test/webhook")._post_embed({"title": "test"}))

    assert response.raise_called is True


def test_discord_trade_entry_includes_the_signal_timeframe():
    notifier = DiscordNotifier("https://discord.test/webhook")
    embeds = []

    async def capture(embed):
        embeds.append(embed)

    notifier._post_embed = capture

    asyncio.run(notifier.notify_trade_open({
        "symbol": "NIFTY 18AUG26 24450 PE",
        "side": "PE",
        "entry": 160.55,
        "sl": 136.22,
        "tgt": 209.21,
        "lot_size": 65,
        "reason": "super",
        "timeframe": "2m",
    }))

    timeframe = next(field for field in embeds[0]["fields"] if field["name"] == "Timeframe")
    assert timeframe["value"] == "2m"
