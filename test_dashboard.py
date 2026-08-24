from io import StringIO

from rich.console import Console

from flattrade_bot.main import TradingEngine


def test_dashboard_shows_b17_system_engine_and_position_telemetry():
    engine = TradingEngine()
    engine.latest_spot_price = 24550.0
    engine.b17_tracked = {
        "CE:24450": {"token": "41009", "tsym": "NIFTY11AUG26C24450", "dname": "NIFTY 11AUG26 24450 CE"},
        "PE:24650": {"token": "41019", "tsym": "NIFTY11AUG26P24650", "dname": "NIFTY 11AUG26 24650 PE"},
    }
    engine.b17_last_event = {
        "side": "CE",
        "symbol": "NIFTY 24450 CE",
        "minute": 610,
        "option_entry": 100.0,
        "fib_high": 24600.0,
        "fib_low": 24450.0,
    }
    engine.active_position = {
        "side": "CE",
        "symbol": "NIFTY 24450 CE",
        "entry": 100.0,
        "ltp": 105.0,
        "sl": 94.0,
        "target": 118.0,
        "timeframe": "combined",
        "signal": "SmartFib index@610",
    }

    console = Console(file=StringIO(), record=True, width=120, force_terminal=False)
    console.print(engine.build_dashboard())
    output = console.export_text(styles=False)

    assert "B17" in output
    assert "SMART FIB" in output
    assert "NIFTY 50 Spot" in output
    assert "1m" in output
    assert "2m" in output
    assert "3m" in output
    assert "5m" in output
    assert "POSITION" in output
    assert "EVENT" in output
    assert "Tracked" in output
    assert "24450" in output
    assert "24650" in output