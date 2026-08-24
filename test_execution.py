import json
from datetime import datetime

from flattrade_bot.execution import TradeExecutor
from flattrade_bot.config import settings
from flattrade_bot.main import TradingEngine
from flattrade_bot.risk.manager import RiskManager


class FakeClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [{"stat": "Ok", "norenordno": "ORD-1", "avgprc": "100.00"}])
        self.calls = []
        self.submitted = []

    async def place_market_order(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        self.submitted.append(response)
        return response

    async def get_order_book(self):
        response = self.submitted[-1]
        return [{
            "norenordno": response.get("norenordno"),
            "status": "COMPLETE",
            "rejreason": " ",
            "fillshares": "65",
            "avgprc": response.get("avgprc", response.get("price", "100.00")),
        }]


class FakeNotifier:
    def __init__(self):
        self.opened = []
        self.closed = []

    async def notify_trade_open(self, trade_info):
        self.opened.append(trade_info)

    async def notify_trade_close(self, trade_info):
        self.closed.append(trade_info)


class FakeHistory:
    async def fetch_live_quote(self, token, exchange):
        if token == "26000":
            return {"lp": 24450.0}
        return {"lp": 101.0}


class RejectingClient(FakeClient):
    async def get_order_book(self):
        response = self.submitted[-1]
        return [{
            "norenordno": response.get("norenordno"),
            "status": "REJECTED",
            "rejreason": "RED:Margin Shortfall",
            "fillshares": "0",
        }]


def test_trade_executor_opens_and_closes_a_long_position():
    async def scenario():
        client = FakeClient([
            {"stat": "Ok", "norenordno": "ENTRY-1", "avgprc": "100.00"},
            {"stat": "Ok", "norenordno": "EXIT-1", "avgprc": "112.00"},
        ])
        notifier = FakeNotifier()
        executor = TradeExecutor(client, RiskManager(), notifier, quantity=65, live_orders=True)

        opened = await executor.open_trade(
            side="CE",
            order_symbol="NIFTY11AUG26C24450",
            display_symbol="NIFTY 11AUG26 24450 CE",
            token="41009",
            timeframe="1m",
            signal="flag_nodiv",
            entry_price=100.0,
            sl_points=6.0,
            tp_points=12.0,
            current_min=600,
            opened_at=datetime(2026, 8, 7, 10, 0),
        )

        assert opened["accepted"] is True
        assert executor.position["sl"] == 94.0
        assert executor.position["target"] == 112.0

        closed = await executor.check_exit(112.0, datetime(2026, 8, 7, 10, 5))

        assert closed["accepted"] is True
        assert executor.position is None
        assert client.calls[0]["side"] == "BUY"
        assert client.calls[1]["side"] == "SELL"
        assert notifier.opened[0]["entry"] == 100.0
        assert notifier.closed[0]["rs"] == 780.0

    import asyncio
    asyncio.run(scenario())


def test_trade_executor_does_not_submit_when_risk_blocks_entry():
    async def scenario():
        client = FakeClient()
        notifier = FakeNotifier()
        risk = RiskManager()
        risk.state.is_shutdown = True
        executor = TradeExecutor(client, risk, notifier, quantity=65, live_orders=True)

        result = await executor.open_trade(
            side="PE",
            order_symbol="NIFTY11AUG26P24650",
            display_symbol="NIFTY 11AUG26 24650 PE",
            token="41019",
            timeframe="1m",
            signal="flag_nodiv",
            entry_price=100.0,
            sl_points=6.0,
            tp_points=12.0,
            current_min=600,
            opened_at=datetime(2026, 8, 7, 10, 0),
        )

        assert result["accepted"] is False
        assert result["reason"] == "Daily shutdown active (Max Loss hit)"
        assert client.calls == []

    import asyncio
    asyncio.run(scenario())


def test_trade_executor_does_not_create_position_after_orderbook_rejection():
    async def scenario():
        client = RejectingClient([
            {"stat": "Ok", "norenordno": "ENTRY-REJECTED", "avgprc": "100.00"},
        ])
        executor = TradeExecutor(client, RiskManager(), FakeNotifier(), quantity=65, live_orders=True)

        result = await executor.open_trade(
            side="CE",
            order_symbol="NIFTY11AUG26C24450",
            display_symbol="NIFTY 11AUG26 24450 CE",
            token="41009",
            timeframe="1m",
            signal="flag_nodiv",
            entry_price=100.0,
            sl_points=6.0,
            tp_points=12.0,
            current_min=600,
            opened_at=datetime(2026, 8, 7, 10, 0),
        )

        assert result["accepted"] is False
        assert "Margin Shortfall" in result["reason"]
        assert executor.position is None

    import asyncio
    asyncio.run(scenario())


def test_trade_executor_force_closes_position_and_notifies():
    async def scenario():
        client = FakeClient([
            {"stat": "Ok", "norenordno": "ENTRY-1", "avgprc": "100.00"},
            {"stat": "Ok", "norenordno": "STOP-1", "avgprc": "98.50"},
        ])
        notifier = FakeNotifier()
        executor = TradeExecutor(client, RiskManager(), notifier, quantity=65, live_orders=True)

        await executor.open_trade(
            side="CE",
            order_symbol="NIFTY11AUG26C24450",
            display_symbol="NIFTY 11AUG26 24450 CE",
            token="41009",
            timeframe="1m",
            signal="flag_nodiv",
            entry_price=100.0,
            sl_points=6.0,
            tp_points=12.0,
            current_min=600,
            opened_at=datetime(2026, 8, 7, 10, 0),
        )

        closed = await executor.close_position(98.5, datetime(2026, 8, 7, 10, 5), "MANUAL_STOP")

        assert closed["accepted"] is True
        assert closed["trade"]["reason"] == "MANUAL_STOP"
        assert executor.position is None
        assert notifier.closed[0]["reason"] == "MANUAL_STOP"
        assert client.calls[-1]["side"] == "SELL"

    import asyncio
    asyncio.run(scenario())


def test_live_signal_wiring_routes_current_contract_to_executor():
    async def scenario():
        client = FakeClient()
        notifier = FakeNotifier()
        engine = TradingEngine(live_orders=True)
        engine.client = client
        engine.history = FakeHistory()
        engine.risk = RiskManager(session_start_min=0, session_end_min=24 * 60)
        engine.executor = TradeExecutor(client, engine.risk, notifier, quantity=65, live_orders=True)
        engine.b17_tracked = {
            "CE:24450": {
                "token": "41009",
                "tsym": "NIFTY11AUG26C24450",
                "dname": "NIFTY 11AUG26 24450 CE",
            }
        }
        engine.latest_spot_price = 24550.0

        await engine._process_b17_events([
            {
                "side": "CE",
                "strike": 24450,
                "minute": 610,
                "trigger": "index",
                "symbol": "NIFTY 24450 CE",
                "option_entry": 100.0,
                "fib_high": 24600.0,
                "fib_low": 24450.0,
                "orientation": "high_to_low",
                "fib_source": "index",
            }
        ])

        assert engine.active_position["order_symbol"] == "NIFTY11AUG26C24450"
        assert engine.active_position["ltp"] == 100.0
        assert engine.active_position["signal"].startswith("SmartFib index@min610")
        assert engine.active_position["monitor_exchange"] == "NSE"
        assert client.calls[0]["symbol"] == "NIFTY11AUG26C24450"
        assert client.calls[0]["side"] == "BUY"

        await engine._monitor_active_position()

        assert engine.active_position["ltp"] == 24450.0

    import asyncio
    asyncio.run(scenario())


def test_engine_restores_broker_verified_position_with_exit_levels(tmp_path, monkeypatch):
    async def scenario():
        state_path = tmp_path / "bot.position.json"
        state_path.write_text(
            json.dumps({
                "side": "CE",
                "symbol": "NIFTY 11AUG26 24450 CE",
                "order_symbol": "NIFTY11AUG26C24450",
                "token": "41009",
                "quantity": 65,
                "timeframe": "1m",
                "signal": "super",
                "entry": 100.0,
                "ltp": 100.0,
                "sl": 94.0,
                "target": 112.0,
                "opened_at": "2026-08-07T10:00:00",
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(settings, "BOT_POSITION_FILE", state_path)

        client = FakeClient()
        client.auth_token = "live-token"

        async def get_positions():
            return [{
                "token": "41009",
                "tsym": "NIFTY11AUG26C24450",
                "dname": "NIFTY 11AUG26 24450 CE",
                "netqty": "65",
                "netavgprc": "100.00",
                "lp": "101.00",
            }]

        client.get_positions = get_positions
        engine = TradingEngine(live_orders=True)
        engine.client = client
        engine.executor = TradeExecutor(client, RiskManager(), FakeNotifier(), quantity=65, live_orders=True)

        await engine._restore_saved_position()

        assert engine.active_position["symbol"] == "NIFTY 11AUG26 24450 CE"
        assert engine.active_position["ltp"] == 101.0
        assert engine.active_position["sl"] == 94.0
        assert engine.active_position["target"] == 112.0
        assert engine.executor.position is engine.active_position

    import asyncio
    asyncio.run(scenario())


def test_engine_clears_local_position_after_broker_reports_manual_close(tmp_path, monkeypatch):
    async def scenario():
        state_path = tmp_path / "bot.position.json"
        state_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(settings, "BOT_POSITION_FILE", state_path)

        client = FakeClient()
        client.auth_token = "live-token"

        async def get_positions():
            return [{
                "token": "41009",
                "tsym": "NIFTY11AUG26C24450",
                "netqty": "0",
            }]

        client.get_positions = get_positions
        engine = TradingEngine(live_orders=True)
        engine.client = client
        engine.history = FakeHistory()
        engine.executor = TradeExecutor(client, RiskManager(), FakeNotifier(), quantity=65, live_orders=True)
        engine.executor.position = {
            "side": "CE",
            "symbol": "NIFTY 11AUG26 24450 CE",
            "order_symbol": "NIFTY11AUG26C24450",
            "token": "41009",
            "quantity": 65,
            "entry": 100.0,
            "ltp": 101.0,
            "sl": 94.0,
            "target": 112.0,
            "opened_at": datetime(2026, 8, 7, 10, 0),
        }
        engine.active_position = engine.executor.position

        await engine._monitor_active_position()

        assert engine.executor.position is None
        assert engine.active_position is None
        assert not state_path.exists()

    import asyncio
    asyncio.run(scenario())


def test_exit_failure_uses_backoff_instead_of_repeated_orders():
    async def scenario():
        client = RejectingClient([
            {"stat": "Ok", "norenordno": "EXIT-1", "avgprc": "101.00"},
        ])
        executor = TradeExecutor(client, RiskManager(), FakeNotifier(), quantity=65, live_orders=True)
        executor.position = {
            "side": "CE",
            "symbol": "NIFTY 11AUG26 24450 CE",
            "order_symbol": "NIFTY11AUG26C24450",
            "token": "41009",
            "quantity": 65,
            "entry": 100.0,
            "ltp": 101.0,
            "sl": 94.0,
            "target": 112.0,
            "opened_at": datetime(2026, 8, 7, 10, 0),
        }

        first = await executor.close_position(101.0, datetime(2026, 8, 7, 10, 0), "TARGET")
        second = await executor.close_position(101.0, datetime(2026, 8, 7, 10, 0, 2), "TARGET")

        assert first["accepted"] is False
        assert "backoff" in second["reason"].lower()
        assert len(client.calls) == 1

    import asyncio
    asyncio.run(scenario())


def test_engine_syncs_realized_broker_pnl_after_manual_close():
    async def scenario():
        client = FakeClient()
        client.auth_token = "live-token"

        async def get_positions():
            return [
                {
                    "tsym": "NIFTY18AUG26C24250",
                    "netqty": "0",
                    "daybuyqty": "65",
                    "daysellqty": "65",
                    "rpnl": "890.50",
                },
                {
                    "tsym": "NIFTY18AUG26P24500",
                    "netqty": "0",
                    "daybuyqty": "65",
                    "daysellqty": "65",
                    "rpnl": "1111.50",
                },
            ]

        client.get_positions = get_positions
        engine = TradingEngine(live_orders=True)
        engine.client = client

        await engine._sync_daily_risk_from_broker()

        assert engine.risk.state.daily_pnl_rs == 2002.0
        assert engine.risk.state.trades_today == 2

    import asyncio
    asyncio.run(scenario())


def test_index_monitor_exit_prices_sell_at_option_ltp_not_index_ltp():
    async def scenario():
        client = FakeClient([
            {"stat": "Ok", "norenordno": "EXIT-1", "avgprc": "101.00"},
        ])
        notifier = FakeNotifier()
        engine = TradingEngine(live_orders=True)
        engine.client = client
        engine.history = FakeHistory()
        engine.risk = RiskManager(session_start_min=0, session_end_min=24 * 60)
        engine.executor = TradeExecutor(client, engine.risk, notifier, quantity=65, live_orders=True)
        engine.executor.position = {
            "side": "PE",
            "symbol": "NIFTY 18AUG26 24250 PE",
            "order_symbol": "NIFTY18AUG26P24250",
            "token": "45101",
            "quantity": 65,
            "timeframe": "combined",
            "signal": "SmartFib index@min600",
            "entry": 43.25,
            "ltp": 43.25,
            "sl": 0.0,
            "target": 0.0,
            "sl_level": 26000.0,
            "tp_level": 25000.0,
            "price_rise": False,
            "monitor_token": "26000",
            "monitor_exchange": "NSE",
            "opened_at": datetime(2026, 8, 18, 10, 33),
        }
        engine.active_position = engine.executor.position

        await engine._monitor_active_position()

        assert engine.active_position is None
        sell = client.calls[-1]
        assert sell["side"] == "SELL"
        assert sell["symbol"] == "NIFTY18AUG26P24250"
        assert sell["ltp"] == 101.0, f"SELL must be priced at option LTP, got {sell['ltp']}"
        assert notifier.closed[0]["reason"] == "TARGET"

    import asyncio
    asyncio.run(scenario())


def test_trading_engine_can_request_terminal_shutdown():
    engine = TradingEngine(live_orders=False)

    assert engine.is_running is True
    engine.request_shutdown()

    assert engine.is_running is True
    assert engine.shutdown_requested is True
