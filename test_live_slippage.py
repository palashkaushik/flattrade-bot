import asyncio
import inspect
import unittest
from datetime import datetime

from flattrade_bot.broker.client import FlattradeClient
from flattrade_bot.execution import TradeExecutor


class _FakeRisk:
    def can_open_trade(self, current_min, daily_pnl):
        return True, ""

    def record_trade_result(self, pnl_rs):
        pass


class _FakeNotifier:
    async def notify_trade_open(self, trade):
        pass

    async def notify_trade_close(self, trade):
        pass


class _FakeClient:
    def __init__(self):
        self.calls = []

    async def place_market_order(self, **kwargs):
        self.calls.append(kwargs)
        return {"stat": "Ok", "norenordno": f"order-{len(self.calls)}"}

    async def get_order_book(self):
        order_id = f"order-{len(self.calls)}"
        price = 100.0 if len(self.calls) == 1 else 101.0
        return [{
            "norenordno": order_id,
            "status": "COMPLETE",
            "fillshares": "65",
            "avgprc": str(price),
        }]


class LiveSlippageTests(unittest.TestCase):
    def test_broker_market_order_default_is_one_point(self):
        default = inspect.signature(
            FlattradeClient.place_market_order
        ).parameters["slippage_buffer"].default
        self.assertEqual(default, 1.0)

    def test_trade_executor_uses_one_point_for_entry_and_exit(self):
        client = _FakeClient()
        executor = TradeExecutor(
            client=client,
            risk=_FakeRisk(),
            notifier=_FakeNotifier(),
            quantity=65,
            live_orders=True,
        )
        opened_at = datetime(2026, 8, 11, 9, 20)

        opened = asyncio.run(executor.open_trade(
            side="CE",
            order_symbol="NIFTYCE",
            display_symbol="NIFTYCE",
            token="token",
            timeframe="1m",
            signal="TEST",
            entry_price=100.0,
            sl_points=5.0,
            tp_points=10.0,
            current_min=560,
            opened_at=opened_at,
        ))
        self.assertTrue(opened["accepted"])

        closed = asyncio.run(executor.close_position(
            101.0, datetime(2026, 8, 11, 9, 21), "TEST"
        ))
        self.assertTrue(closed["accepted"])
        self.assertEqual(
            [call["slippage_buffer"] for call in client.calls],
            [1.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
