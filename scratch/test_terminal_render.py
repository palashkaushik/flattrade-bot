"""Quick syntax & import test for build_dashboard in main.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Just test that main.py parses without syntax errors
import ast
with open("flattrade_bot/main.py", "r", encoding="utf-8") as f:
    source = f.read()

try:
    ast.parse(source)
    print("✅ main.py — AST parse OK (no syntax errors)")
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR in main.py: {e}")
    sys.exit(1)

# Now verify the build_dashboard method renders via a mock
from unittest.mock import MagicMock, patch
from rich.console import Console, Group

# Patch heavy dependencies so we can instantiate TradingEngine cheaply
with patch("flattrade_bot.main.FlattradeAuth"), \
     patch("flattrade_bot.main.FlattradeClient") as MockClient, \
     patch("flattrade_bot.main.FlattradeHistoryFetcher"), \
     patch("flattrade_bot.main.DiscordNotifier"), \
     patch("flattrade_bot.main.TradeExecutor"), \
     patch("flattrade_bot.main.RiskManager") as MockRisk:

    # Configure mocks
    mock_client_inst = MockClient.return_value
    mock_client_inst.auth_token = None  # paper mode
    mock_risk_inst = MockRisk.return_value
    mock_risk_inst.state = MagicMock()
    mock_risk_inst.state.daily_pnl_rs = -120.50
    mock_risk_inst.state.trades_today = 3
    mock_risk_inst.state.consecutive_losses = 1

    from flattrade_bot.main import TradingEngine
    engine = TradingEngine(live_orders=False)
    engine.latest_spot_price = 24562.50
    engine.ce_symbol = "NIFTY 24550 CE"
    engine.pe_symbol = "NIFTY 24550 PE"
    engine.ce_token = "54321"
    engine.pe_token = "54322"
    engine.latest_option_ltp = {"CE": 125.40, "PE": 118.20}
    engine._broker_status = "OK"
    engine._position_conflict = False
    engine.active_position = None
    engine.last_signal = None

    # Feed dummy candles to warm up B07
    from flattrade_bot.indicators.patterns import Candle
    for i in range(35):
        c = Candle(open=24500+i*5, high=24520+i*5, low=24490+i*5, close=24510+i*5, minute=570+i)
        engine.b07_strategy.push_1m_candle(c)

    # === Render FLAT state ===
    console = Console(width=140)
    dashboard = engine.build_dashboard()
    assert isinstance(dashboard, Group), f"Expected Group, got {type(dashboard)}"
    console.print("\n=== FLAT STATE ===")
    console.print(dashboard)

    # === Render IN-POSITION state ===
    engine.active_position = {
        "side": "CE",
        "symbol": "NIFTY 24550 CE",
        "entry": 125.00,
        "ltp": 146.50,
        "sl": 84.30,
        "target": 217.50,
        "quantity": 65,
        "timeframe": "3m",
        "signal": "B07_DIP_BUY_CE",
    }
    console.print("\n=== IN-POSITION STATE ===")
    console.print(engine.build_dashboard())

    print("\n✅ build_dashboard renders both FLAT and IN-POSITION states successfully!")
