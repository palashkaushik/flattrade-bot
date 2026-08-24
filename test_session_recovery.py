import asyncio

import pytest

from flattrade_bot.main import TradingEngine
from flattrade_bot.broker.history import FlattradeHistoryFetcher, SessionExpiredError
from flattrade_bot.execution import TradeExecutor
from flattrade_bot.risk.manager import RiskManager


class FakeAuth:
    def __init__(self):
        self.calls = 0

    async def obtain_token(self):
        self.calls += 1
        return "refreshed-token"


class FakePositionsClient:
    def __init__(self):
        self.auth_token = "expired-token"
        self.calls = 0

    def set_token(self, token):
        self.auth_token = token

    async def get_positions(self):
        self.calls += 1
        if self.calls == 1:
            return {"stat": "Not_Ok", "emsg": "Session Expired : Invalid Session Key"}
        return {"stat": "Ok", "positions": []}


def test_engine_refreshes_both_broker_clients_after_session_expiry():
    async def scenario():
        engine = TradingEngine(live_orders=True)
        engine.auth = FakeAuth()
        engine.client.set_token("expired-token")
        engine.history.set_token("expired-token")
        engine._obtain_flattrade_session_token = engine.auth.obtain_token

        assert await engine._refresh_flattrade_session("test") is True
        assert engine.auth.calls == 1
        assert engine.client.auth_token == "refreshed-token"
        assert engine.history.auth_token == "refreshed-token"

    asyncio.run(scenario())


def test_daily_risk_sync_refreshes_and_retries_after_session_expiry():
    async def scenario():
        engine = TradingEngine(live_orders=True)
        engine.auth = FakeAuth()
        client = FakePositionsClient()
        engine.client = client
        engine._obtain_flattrade_session_token = engine.auth.obtain_token

        await engine._sync_daily_risk_from_broker()

        assert client.calls == 2
        assert engine.auth.calls == 1
        assert client.auth_token == "refreshed-token"

    asyncio.run(scenario())


def test_history_quote_raises_a_distinct_session_expired_error(monkeypatch):
    class FakeResponse:
        def json(self):
            return {"stat": "Not_Ok", "emsg": "Session Expired : Invalid Session Key"}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: FakeClient())
    fetcher = FlattradeHistoryFetcher("expired-token")

    with pytest.raises(SessionExpiredError):
        asyncio.run(fetcher.fetch_live_quote("45100", "NFO"))


def test_reconcile_marks_unmatched_broker_exposure_as_a_conflict():
    class ConflictingClient:
        async def get_positions(self):
            return {
                "stat": "Ok",
                "positions": [{
                    "token": "45109",
                    "tsym": "NIFTY18AUG26P24550",
                    "netqty": "65",
                    "lp": "157.90",
                }],
            }

    class Notifier:
        async def notify_trade_open(self, _trade):
            pass

        async def notify_trade_close(self, _trade):
            pass

    async def scenario():
        executor = TradeExecutor(
            ConflictingClient(),
            RiskManager(),
            Notifier(),
            quantity=65,
            live_orders=True,
        )
        executor.position = {
            "token": "45100",
            "order_symbol": "NIFTY18AUG26C24250",
        }

        result = await executor.reconcile_broker_position()

        assert result is None
        assert executor.last_reconcile_conflict is True

    asyncio.run(scenario())
