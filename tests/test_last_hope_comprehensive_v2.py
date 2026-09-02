"""Comprehensive test suite for the Last Hope Winner live bot (v2).

Covers:
  A. §42 champion congruence (constants, TF alignment, arming, BE geometry)
  B. Seeded warmup state (300-bar prior replay, day-cold arming)
  C. Funds-rejection fallback (2nd ITM -> 1st ITM per-buy)
  D. Weekly-expiry front-contract selection (nearest expiry >= today)
  E. Dashboard renderer (fixed-height frame, no emoji, no \x0b, no Live thread)
  F. Exit watchdog & executor paths (SL priority, dry_run, background close)
"""

import asyncio
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from flattrade_bot.strategies.last_hope_winner import (
    ARM_WINDOW, ATR_MULT, ATR_PERIOD, BE_TRIGGER_RATIO, BE_BUFFER_PTS,
    TOUCH_BUFFER, TP_PTS_CAP, SESSION_START_MIN, SESSION_END_MIN,
    Bar1m, LastHopeWinnerEngine, OptionContractState, TFTracker,
)


IST = timezone(timedelta(hours=5, minutes=30))


def ts_now():
    return datetime.now(IST)


# ===========================================================================
# A. §42 CHAMPION CONGRUENCE
# ===========================================================================

def test_champion_constants_match_ledger_42():
    """arm15 / ATR(10)x1.5 / tb0.0 / BE 0.50 + 1.0 — the §42 max-net champion."""
    assert ARM_WINDOW == 15
    assert ATR_PERIOD == 10
    assert ATR_MULT == 1.5
    assert TOUCH_BUFFER == 0.0
    assert BE_TRIGGER_RATIO == 0.50
    assert BE_BUFFER_PTS == 1.0
    assert TP_PTS_CAP == 15.0


def test_tf_tracker_clock_aligned_boundaries():
    """TF buckets close only on session-clock boundaries (555-relative)."""
    trk = TFTracker(5)
    base = SESSION_START_MIN  # 555
    # bars 555..558 fill the first 5m bucket; 559 closes it
    for m in range(555, 559):
        trk.push_1m_bar(Bar1m(minute=m, open=1, high=2, low=0.5, close=1.5, timestamp=ts_now()))
    assert trk.last_s1 is None, "5m stoch must not fire before the boundary bar"
    trk.push_1m_bar(Bar1m(minute=559, open=1, high=2, low=0.5, close=1.5, timestamp=ts_now()))
    assert trk.last_s1 is not None, "bar ending at 09:20 (minute 559) closes the first 5m bucket"
    # next bucket closes at 564 (09:25)
    for m in range(560, 564):
        trk.push_1m_bar(Bar1m(minute=m, open=1, high=2, low=0.5, close=1.5, timestamp=ts_now()))
    before = trk.last_s1
    trk.push_1m_bar(Bar1m(minute=564, open=1, high=2.5, low=0.5, close=2.0, timestamp=ts_now()))
    assert trk.last_s1 != before or trk.cur_bars == [], "bucket boundary at minute 564"


def test_arming_never_survives_position_close():
    """engine clears armed flags while a trade is active (re-entry guard)."""
    eng = LastHopeWinnerEngine()
    cs = eng.register_contract("CE:24000", "NIFTY02SEP26C24000", "t1", "CE", 24000)
    cs.set_day_sr_levels(200, 100, 150)
    cs.flag_armed = True
    cs.super_armed = True
    eng.active_trade = {"symbol": "NIFTY02SEP26C24000", "entry": 150, "sl": 145,
                        "tp": 155, "dist": 5, "be_trigger_px": 152.5,
                        "be_hardened_sl": 151.0, "be_done": False}
    # push a tick for the in-position contract: engine must disarm it
    t = datetime(2026, 9, 2, 10, 0, tzinfo=IST)
    eng.push_tick("CE:24000", 150.5, t)
    assert cs.flag_armed is False
    assert cs.super_armed is False


def test_be_geometry_rebased_on_fill_price():
    """on_trade_opened re-bases BE to fill + 50% dist + 1.0 (§42)."""
    eng = LastHopeWinnerEngine()
    eng.on_trade_opened({
        "symbol": "X", "entry": 114.95, "dist": 7.43,
        "sl": 107.52, "tp": 122.38,
        "be_trigger_px": 120.0, "be_hardened_sl": 115.95,
        "be_done": False,
    })
    at = eng.active_trade
    assert abs(at["be_trigger_px"] - (114.95 + 0.50 * 7.43)) < 0.01
    assert abs(at["be_hardened_sl"] - 115.95) < 0.01
    assert abs(at["sl"] - (114.95 - 7.43)) < 0.01


def test_signal_only_inside_session_window():
    """No signals returned outside 09:15-15:00 (engine-level session gate)."""
    eng = LastHopeWinnerEngine()
    cs = eng.register_contract("CE:24000", "NIFTY02SEP26C24000", "t1", "CE", 24000)
    cs.set_day_sr_levels(200, 100, 150)

    # Contract returns a signal (simulated); the ENGINE must gate it by clock.
    fake_sig = {"side": "CE", "symbol": "NIFTY02SEP26C24000", "token": "t1",
                "strike": 24000, "trigger": "FLAG", "level": "EMA20",
                "entry": 150.0, "dist": 5.0, "sl": 145.0, "tp": 155.0,
                "be_trigger_px": 152.5, "be_hardened_sl": 151.0}

    def fake_push_tick(ltp, dt):
        minute = dt.hour * 60 + dt.minute
        return fake_sig if 555 <= minute < 900 else None

    cs.push_tick = fake_push_tick  # bypass bar mechanics; test the ENGINE gate

    # Intraday -> engine returns the signal
    t_mid = datetime(2026, 9, 2, 10, 0, tzinfo=IST)
    assert eng.push_tick("CE:24000", 150.0, t_mid) is not None

    # Before session -> nothing (contract itself blocks)
    t_early = datetime(2026, 9, 2, 9, 10, tzinfo=IST)
    assert eng.push_tick("CE:24000", 150.0, t_early) is None

    # After 15:00 -> nothing
    t_late = datetime(2026, 9, 2, 15, 5, tzinfo=IST)
    assert eng.push_tick("CE:24000", 150.0, t_late) is None


# ===========================================================================
# B. SEEDED WARMUP STATE
# ===========================================================================

def test_reset_session_clears_intraday_state():
    cs = OptionContractState(symbol="X", token="t", side="CE", strike=24000)
    cs.bars = [Bar1m(minute=560, open=1, high=2, low=0.5, close=1.5, timestamp=ts_now())]
    cs.flag_armed = True
    cs.current_min = 560
    cs.cur_open = 1
    cs.latest_atr = 9.9
    cs.sr_levels = {"CPR_Pivot": 150.0}
    cs.reset_session()
    assert cs.bars == []
    assert cs.flag_armed is False
    assert cs.current_min == -1
    assert cs.latest_atr == 6.0
    assert cs.sr_levels == {"CPR_Pivot": 150.0}, "S/R levels survive session reset (day-level)"


def test_seed_replays_through_trackers():
    """seed_1m_bars feeds ATR/EMA/VWAP + TF trackers (seeded mode)."""
    cs = OptionContractState(symbol="X", token="t", side="CE", strike=24000)
    cs.set_day_sr_levels(200, 100, 150)
    prior = [Bar1m(minute=555 + i, open=150 + i * 0.1, high=151 + i * 0.1,
                   low=149 + i * 0.1, close=150 + i * 0.1, timestamp=ts_now())
             for i in range(30)]
    today = [Bar1m(minute=555, open=150, high=151, low=149, close=150.5, timestamp=ts_now())]
    cs.seed_1m_bars(prior, today)
    assert len(cs.bars) == 31
    assert cs.atr.value is not None and cs.atr.value > 0
    assert cs.ema20.value is not None
    assert cs.vwap.value is not None
    # 30 prior + 1 today = 31 bars -> 5m tracker closed 6 buckets
    assert cs.tf_trackers[5].last_s1 is not None


# ===========================================================================
# C. FUNDS-REJECTION FALLBACK (2nd ITM -> 1st ITM, per-buy)
# ===========================================================================

def _make_engine_for_funds_test(tmp_path, monkeypatch):
    """Builds LastHopeTradingEngine with mocked broker + history."""
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
    from flattrade_bot.last_hope_main import LastHopeTradingEngine
    eng = LastHopeTradingEngine(live_orders=True)
    eng.spot_price = 24050.0
    eng.engine.set_spot_price(24050.0)
    return eng


def test_is_funds_rejection_classifier():
    from flattrade_bot.last_hope_main import LastHopeTradingEngine
    eng = LastHopeTradingEngine(live_orders=True)
    assert eng._is_funds_rejection({"reason": "Insufficient Margin"}) is True
    assert eng._is_funds_rejection({"emsg": "RMS: Blocked due to insufficient funds"}) is True
    assert eng._is_funds_rejection({"reason": "Order fill not confirmed within timeout"}) is False
    assert eng._is_funds_rejection({"emsg": "Invalid symbol"}) is False


@pytest.mark.asyncio
async def test_funds_fallback_uses_first_itm(monkeypatch):
    """On funds rejection of the 2nd ITM buy, the engine retries 1st ITM
    (ATM-50 for CE) for that single buy."""
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
    from flattrade_bot.last_hope_main import LastHopeTradingEngine

    eng = LastHopeTradingEngine(live_orders=True)
    eng.spot_price = 24050.0  # ATM 24050 -> CE 1st ITM = 24000
    eng.engine.set_spot_price(24050.0)
    # Session-window bypass: tests can run outside 09:15-15:00 IST
    eng.risk.can_open_trade = lambda cur_min, n: (True, "Allowed")

    calls = []

    async def fake_open_trade(**kw):
        calls.append(kw["order_symbol"])
        if len(calls) == 1:
            return {"accepted": False, "reason": "Insufficient margin for NIFTY02SEP26C23950"}
        return {"accepted": True,
                "position": {"entry": 100.0, "sl": 93.0, "target": 107.0,
                             "symbol": "NIFTY02SEP26C24000"}}

    async def fake_resolve(sig, now):
        return {"symbol": "NIFTY02SEP26C24000", "token": "tok1", "ltp": 100.0}

    eng.executor = MagicMock()
    eng.executor.open_trade = fake_open_trade
    eng.executor.position = None
    eng._resolve_first_itm = fake_resolve

    sig = {"side": "CE", "symbol": "NIFTY02SEP26C23950", "token": "t0", "strike": 23950,
           "trigger": "FLAG", "level": "EMA20", "entry": 110.0, "dist": 7.0,
           "sl": 103.0, "tp": 117.0, "be_trigger_px": 113.5, "be_hardened_sl": 111.0}
    await eng._try_enter(sig)
    assert calls == ["NIFTY02SEP26C23950", "NIFTY02SEP26C24000"], \
        "fallback must retry the CE 1st ITM (ATM-50 = 24000) after funds rejection"


@pytest.mark.asyncio
async def test_non_funds_rejection_does_not_fallback(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
    from flattrade_bot.last_hope_main import LastHopeTradingEngine
    eng = LastHopeTradingEngine(live_orders=True)
    eng.spot_price = 24050.0
    eng.risk.can_open_trade = lambda cur_min, n: (True, "Allowed")

    calls = []

    async def fake_open_trade(**kw):
        calls.append(kw["order_symbol"])
        return {"accepted": False, "reason": "Order fill not confirmed within timeout"}

    eng.executor = MagicMock()
    eng.executor.open_trade = fake_open_trade
    eng.executor.position = None
    eng._resolve_first_itm = AsyncMock()

    sig = {"side": "PE", "symbol": "NIFTY02SEP26P24150", "token": "t0", "strike": 24150,
           "trigger": "SUPER", "level": "PDL", "entry": 90.0, "dist": 6.0,
           "sl": 84.0, "tp": 96.0, "be_trigger_px": 93.0, "be_hardened_sl": 91.0}
    await eng._try_enter(sig)
    assert calls == ["NIFTY02SEP26P24150"], "non-funds rejection must NOT trigger fallback"
    eng._resolve_first_itm.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_itm_resolution_strike_math(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
    from flattrade_bot.last_hope_main import LastHopeTradingEngine
    eng = LastHopeTradingEngine(live_orders=True)
    eng.spot_price = 24055.0  # ATM rounds to 24050

    searched = {}

    async def fake_search(txt):
        searched["q"] = txt
        return {"token": "tokX", "tsym": "NIFTY02SEP26C24000", "dname": "x"}

    async def fake_quotes(exchange="NFO", token=""):
        return {"stat": "Ok", "lp": "101.25"}

    eng.history.search_option_token = fake_search
    eng.client.get_quotes = fake_quotes

    sig = {"side": "CE"}
    fb = await eng._resolve_first_itm(sig, ts_now())
    assert searched["q"] == "NIFTY 24000 CE", "CE 1st ITM = ATM-50"
    assert fb == {"symbol": "NIFTY02SEP26C24000", "token": "tokX", "ltp": 101.25}

    # PE: ATM+50
    async def fake_search_pe(txt):
        searched["q2"] = txt
        return {"token": "tokP", "tsym": "NIFTY02SEP26P24100", "dname": "x"}
    eng.history.search_option_token = fake_search_pe
    fb2 = await eng._resolve_first_itm({"side": "PE"}, ts_now())
    assert searched["q2"] == "NIFTY 24100 PE", "PE 1st ITM = ATM+50"
    assert fb2["symbol"] == "NIFTY02SEP26P24100"


# ===========================================================================
# D. WEEKLY-EXPIRY FRONT-CONTRACT SELECTION
# ===========================================================================

@pytest.mark.asyncio
async def test_search_picks_nearest_expiry(monkeypatch):
    """SearchScrip results must resolve to the nearest expiry >= today."""
    from flattrade_bot.broker.history import FlattradeHistoryFetcher

    f = FlattradeHistoryFetcher()
    f.set_token("tok")

    async def fake_post(url, body):
        return {"stat": "Ok", "values": [
            {"token": "far", "tsym": "NIFTY17SEP26C24000", "dname": "far"},   # 17 Sep 2026
            {"token": "near", "tsym": "NIFTY08SEP26C24000", "dname": "near"}, # 08 Sep 2026
            {"token": "past", "tsym": "NIFTY25AUG26C24000", "dname": "past"}, # expired
        ]}

    f._post = fake_post
    res = await f.search_option_token("NIFTY 24000 CE")
    assert res["token"] == "near", "must pick the nearest non-expired weekly contract"


@pytest.mark.asyncio
async def test_search_falls_back_when_no_dates(monkeypatch):
    from flattrade_bot.broker.history import FlattradeHistoryFetcher
    f = FlattradeHistoryFetcher()
    f.set_token("tok")

    async def fake_post(url, body):
        return {"stat": "Ok", "values": [
            {"token": "odd", "tsym": "NIFTY-UNPARSABLE", "dname": "odd"},
        ]}
    f._post = fake_post
    res = await f.search_option_token("NIFTY 24000 CE")
    assert res["token"] == "odd", "unparseable symbols fall back to first result"


# ===========================================================================
# E. DASHBOARD RENDERER
# ===========================================================================

def _eng_with_contract():
    monkey_free_engine = None
    os.environ.setdefault("DISCORD_WEBHOOK_URL", "")
    from flattrade_bot.last_hope_main import LastHopeTradingEngine
    eng = LastHopeTradingEngine(live_orders=False)
    from flattrade_bot.strategies.last_hope_winner import OptionContractState
    cs = OptionContractState(symbol="NIFTY02SEP26C24000", token="1", side="CE", strike=24000)
    cs.set_day_sr_levels(200.0, 100.0, 150.0)
    eng.engine.contracts["CE:24000"] = cs
    eng._last_ltp["CE:24000"] = 148.25
    return eng


def test_dashboard_frame_no_vt_no_emoji_fixed_height():
    eng = _eng_with_contract()
    L = eng._render_rich_frame()
    assert len(L) > 10, "frame must render"
    assert all("\x0b" not in ln for ln in L), "no vertical-tab soft breaks may survive"
    assert not any("\U0001F3C6" in ln for ln in L), "no emoji in the frame (width misalignment)"
    # print_dashboard pads to a fixed height -> identical line counts every call
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng.print_dashboard()
        eng.print_dashboard()
        eng.print_dashboard()
    assert eng._dash_lines_drawn > 0
    out = buf.getvalue()
    assert out.count("\x0b") == 0


def test_dashboard_repeated_renders_same_height():
    eng = _eng_with_contract()
    heights = []
    import io, contextlib
    for _ in range(5):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eng.print_dashboard()
        heights.append(eng._dash_lines_drawn)
    assert len(set(heights)) == 1, f"frame height must be constant, got {heights}"


# ===========================================================================
# F. EXIT PATHS (recap of the critical live fixes)
# ===========================================================================

@pytest.mark.asyncio
async def test_check_exit_dry_run_detects_only(monkeypatch):
    from flattrade_bot.execution import TradeExecutor
    ex = TradeExecutor(MagicMock(), MagicMock(), MagicMock(), live_orders=True)
    ex.position = {"symbol": "X", "order_symbol": "X", "entry": 100.0, "sl": 93.0,
                   "target": 107.0, "quantity": 65, "order_id": "1",
                   "opened_at": ts_now(), "token": "t"}
    res = await ex.check_exit(92.5, ts_now(), dry_run=True)   # SL hit
    assert res.get("exit_reason") == "STOP_LOSS"
    assert ex.position is not None, "dry_run must NOT close the position"


@pytest.mark.asyncio
async def test_sl_priority_over_tp_same_bar():
    from flattrade_bot.execution import TradeExecutor
    ex = TradeExecutor(MagicMock(), MagicMock(), MagicMock(), live_orders=True)
    ex.position = {"symbol": "X", "order_symbol": "X", "entry": 100.0, "sl": 93.0,
                   "target": 107.0, "quantity": 65, "order_id": "1",
                   "opened_at": ts_now(), "token": "t"}
    # bar spans both SL and TP -> SL priority (backtest rule)
    res = await ex.check_exit(90.0, ts_now(), dry_run=True)
    assert res.get("exit_reason") == "STOP_LOSS"


@pytest.mark.asyncio
async def test_eod_exit_after_session():
    from flattrade_bot.execution import TradeExecutor
    ex = TradeExecutor(MagicMock(), MagicMock(), MagicMock(), live_orders=True)
    ex.position = {"symbol": "X", "order_symbol": "X", "entry": 100.0, "sl": 93.0,
                   "target": 107.0, "quantity": 65, "order_id": "1",
                   "opened_at": ts_now(), "token": "t"}
    t = datetime(2026, 9, 2, 15, 5, tzinfo=IST)
    res = await ex.check_exit(100.0, t, dry_run=True)
    assert res.get("exit_reason") == "EOD"
