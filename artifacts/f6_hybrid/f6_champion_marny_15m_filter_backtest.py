"""F6 Champion Strategy with Marny Core 15-Minute Option-Chart HTF Filter (2020-2026).

Strategy Architecture:
1. Core Signal: F6 Champion Multi-Timeframe Engine (1m, 2m, 3m, 5m) evaluated on Option Charts.
   - Fast Stochastic S1 (12, 3), Slow Stochastic S4 (50, 10).
   - ATR lookback 10, ATR SL multiplier = 3.0, ATR TP multiplier = 6.0.
   - F6 Flag No-Divergence Trigger: S4 >= 79.5 AND S1 <= 25.0.
   - Standard Bullish Pin Bar Divergence Triggers (Super, Flag, Reversal).
2. Marny Core 15-Minute HTF Filter on Option Chart:
   - 15m Heikin-Ashi on the Option Contract.
   - 11-period Linear Regression Candles Signal Curve (11-period SMA of 15m HA Close).
   - 15m UT Bot Alerts (Key=1.0, Period=10) on the Option Contract.
   - Marny Bullish Gate: 15m HA Close > 11-period LinReg Plot AND 15m UT Bot == "green".
3. Filter Gate:
   - CE Option Chart: F6 Champion BUY CE is ONLY executed if CE 15m Marny Filter is BULLISH.
   - PE Option Chart: F6 Champion BUY PE is ONLY executed if PE 15m Marny Filter is BULLISH.
4. Risk Management:
   - Option ATR SL (Entry - 3.0 * ATR) & Option ATR TP (Entry + 6.0 * ATR).
   - Trailing Stop active at +1.5 * ATR.
   - Consecutive Loss limit (8) & Daily Loss protection (-30 pts).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
import grid_optimize_f6_atr as f6_eng
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.marny_option_chart_backtest import Option5mHTFBias, IncrementalATR

# F6 Champion 12/50 params (ledger §19 champion: S1=(12,3), S4 k=50, ATR(10) SL x3 / TP x6,
# F6 S4>=79.5 and S1<=25, consecutive-loss stop=8, daily raw shutdown=-30 points).
# 2026-08-20 user override: TP multiplier 6.0 -> 5.0 (SL:TP ratio 3:5).
CHAMPION_12_50 = {
    "s1_k": 12, "s1_d": 3, "s4_k": 50,
    "atr_period": 10, "atr_sl_mult": 3.0, "atr_tp_mult": 5.0,
    "f6_s4_thresh": 79.5, "f6_s1_thresh": 25.0, "consec_loss": 8,
}


class Option15mHTFBias:
    """Causal 15-Minute HTF Bias computed directly on Option bars."""
    def __init__(self):
        self.ha = None
        self.ut = None
        self.ha_closes = deque(maxlen=11)
        self.buffer = []
        self.bullish = False
        self.ut_color = "blue"

    def update_1m(self, candle: Candle):
        self.buffer.append(candle)
        if candle.minute % 15 != 0 or not self.buffer:
            return
        buf = self.buffer
        self.buffer = []
        raw_15m = Candle(
            open=buf[0].open,
            high=max(c.high for c in buf),
            low=min(c.low for c in buf),
            close=buf[-1].close,
            minute=candle.minute,
        )
        if self.ha is None:
            from artifacts.f6_hybrid.marny_option_chart_backtest import HeikinAshiState
            from artifacts.f6_hybrid.marny_option_chart_backtest import UTBotState
            self.ha = HeikinAshiState()
            self.ut = UTBotState(key=1.0, period=10)
        ha_15m = self.ha.update(raw_15m)
        self.ha_closes.append(ha_15m.close)
        self.ut_color = self.ut.update(raw_15m)
        if len(self.ha_closes) >= 11:
            linreg_plot = sum(self.ha_closes) / len(self.ha_closes)
            self.bullish = ha_15m.close > linreg_plot and self.ut_color == "green"
        else:
            self.bullish = False

    def snapshot(self) -> bool:
        return self.bullish


class TFTrackerS1TurnUp:
    """F6 / Stochastic tracker using S1 Turn-Up as the entry trigger (no pinbar)."""
    def __init__(self, lb, p):
        self.lb = lb
        self.stoch = f6_eng.ParamStoch(p["s1_k"], p["s1_d"], p["s4_k"])
        self.div = f6_eng.DivergenceEngine()
        self.hist = []
        self.setup = False
        self.stype = ""
        self.prev_s1 = None
        self.s4_emb = 0
        self.atr = IncrementalATR(p["atr_period"])
        self.p_f6_s4 = p["f6_s4_thresh"]
        self.p_f6_s1 = p["f6_s1_thresh"]

    def push(self, c: Candle):
        self.hist.append(c)
        if len(self.hist) > 40:
            self.hist.pop(0)
        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
        atr_val = self.atr.update(c.high, c.low, c.close)

        if s4 is not None:
            self.s4_emb = self.s4_emb + 1 if s4 <= 20 else 0

        is_flag = s4 is not None and s1 is not None and s4 >= self.p_f6_s4 and s1 <= self.p_f6_s1
        is_super = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))

        if is_flag or is_super:
            self.setup = True
            self.stype = "super" if is_super else "flag"

        triggered = False
        if self.setup and s1 is not None and self.prev_s1 is not None:
            if s1 > self.prev_s1:
                triggered = True
                self.setup = False

        self.prev_s1 = s1
        is_rev = (self.s4_emb >= 25) and (self.stype == "super")
        return triggered, is_rev, self.stype, c.close, atr_val


class MTFTrackerS1TurnUp:
    def __init__(self, p, tfs=None):
        self.tfs = list(tfs) if tfs else list(f6_eng.TF_SPECS)
        self.trackers = {tf: TFTrackerS1TurnUp(f6_eng.TF_SPECS[tf][1], p) for tf in self.tfs}
        self.bufs = {tf: [] for tf in self.tfs}
        self._last_minute = None

    def push_1m(self, c1m: Candle):
        out = []
        for tf in self.tfs:
            spec = f6_eng.TF_SPECS[tf]
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf]) == spec[0]:
                buf = self.bufs[tf]
                self.bufs[tf] = []
                ctf = Candle(open=buf[0].open, high=max(x.high for x in buf),
                             low=min(x.low for x in buf), close=buf[-1].close,
                             minute=buf[-1].minute)
                trig, is_rev, stype, px, atr_val = self.trackers[tf].push(ctf)
                if trig:
                    out.append((tf, is_rev, stype, px, atr_val))
        return out


class MTFTrackerTFs(f6_eng.MTFTracker):
    """f6_eng.MTFTracker restricted to a subset of timeframes (15m bias stays fixed)."""

    def __init__(self, p, tfs=None):
        self.tfs = list(tfs) if tfs else list(f6_eng.TF_SPECS)
        self.trackers = {tf: f6_eng.TFTracker(f6_eng.TF_SPECS[tf][1], p) for tf in self.tfs}
        self.f6scans = {tf: f6_eng.FlagNoDivScanner(p["s1_k"], p["s1_d"], p["s4_k"],
                                                    p["f6_s4_thresh"], p["f6_s1_thresh"]) for tf in self.tfs}
        self.bufs = {tf: [] for tf in self.tfs}
        self._last_minute = None
        self.reverse_regime_active = False
        self.require_ema20 = any(
            tracker.require_ema20 for tracker in self.trackers.values()
        )

    def _reset_timeframe_buffers_if_session_rolled(self, minute):
        if self._last_minute is not None and minute > 0 and minute < self._last_minute:
            self.bufs = {tf: [] for tf in self.tfs}
            for tf in self.tfs:
                self.trackers[tf].reset_session_state()
                self.f6scans[tf].reset_session_state()
            self.reverse_regime_active = False
        self._last_minute = minute

    def push_1m(self, c1m: Candle):
        out = []
        self._reset_timeframe_buffers_if_session_rolled(c1m.minute)
        for tf in self.tfs:
            spec = f6_eng.TF_SPECS[tf]
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf]) == spec[0]:
                buf = self.bufs[tf]
                self.bufs[tf] = []
                ctf = Candle(open=buf[0].open, high=max(x.high for x in buf),
                             low=min(x.low for x in buf), close=buf[-1].close,
                             minute=buf[-1].minute)
                trig, is_rev, stype, px, atr_val = self.trackers[tf].push(ctf)
                if trig:
                    out.append((tf, is_rev, stype, px, atr_val))
                ema_ok = (
                    not self.require_ema20
                    or (
                        self.trackers[tf].ema20_value is not None
                        and ctf.close > self.trackers[tf].ema20_value
                    )
                )
                if ema_ok and self.f6scans[tf].push(ctf.high, ctf.low, ctf.close):
                    out.append((tf, False, "flag_nodiv", ctf.close, atr_val))
        self.reverse_regime_active = any(
            tracker.s4_emb >= 25
            for tracker in self.trackers.values()
        )
        return [
            (
                tf,
                is_reverse or (self.reverse_regime_active and signal_type == "super"),
                signal_type,
                entry,
                atr_value,
            )
            for tf, is_reverse, signal_type, entry, atr_value in out
        ]


LOT_SIZE = 65
SESSION_START = 560
SESSION_END = 900
DAY_LAST = 930
GLOBAL_SPOT = {}


def init_worker(spot):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot
    source.GLOBAL_CACHE = {}


def process_day_f6_marny_15m_filter(args):
    day, opt_path, prev_opt_path, p = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not opt_path:
        return {}

    include_fees = p.get("include_fees", True)
    f6_params = p.get("f6_params", CHAMPION_12_50)
    atr_sl_mult = f6_params.get("atr_sl_mult", 3.0)
    atr_tp_mult = f6_params.get("atr_tp_mult", 6.0)
    theta_offset = p.get("theta_offset", 0.0)
    time_stop_min = p.get("time_stop_min", None)
    trail_sl = p.get("trail_sl", True)
    daily_loss_pts = p.get("daily_loss_pts", -30.0)
    s1_turn_up = p.get("s1_turn_up", False)
    no_pinbar = p.get("no_pinbar", False)
    tfs = p.get("tfs")

    rec = source.cached_option(str(opt_path))
    if rec is None:
        return {}
    df, groups, prefix = rec
    if prefix is None:
        return {}

    spot_mins = spot["min"]
    all_events = []
    option_bars_by_key = {}

    for side in ("CE", "PE"):
        atm_strikes = set()
        for m in range(SESSION_START, SESSION_END + 1):
            idx = np.searchsorted(spot_mins, m, side="right") - 1
            if idx >= 0:
                spot_px = float(spot["close"][idx])
                atm = int(round(spot_px / 50) * 50)
                strike = atm - 100 if side == "CE" else atm + 100
                atm_strikes.add(strike)

        for strike in atm_strikes:
            sym = f"{prefix}{strike}{side}"
            sl = source.make_slice(df, groups, sym)
            if sl is None or len(sl["times"]) < 15:
                continue

            key = (side, strike)
            option_bars_by_key[key] = {}
            for j in range(len(sl["times"])):
                m_j = int(sl["times"][j])
                option_bars_by_key[key][m_j] = {
                    "minute": m_j,
                    "open": float(sl["open"][j]),
                    "high": float(sl["high"][j]),
                    "low": float(sl["low"][j]),
                    "close": float(sl["close"][j]),
                }

            # Instantiate Marny 15m Option HTF Filter & Tracker
            marny_15m = Option15mHTFBias()
            f6_tracker = MTFTrackerS1TurnUp(f6_params, tfs) if s1_turn_up else MTFTrackerTFs(f6_params, tfs)

            # Previous day warmup
            if prev_opt_path:
                prev_rec = source.cached_option(str(prev_opt_path))
                if prev_rec:
                    p_df, p_groups, p_prefix = prev_rec
                    p_sl = source.make_slice(p_df, p_groups, sym)
                    if p_sl is not None:
                        for pj in range(len(p_sl["times"])):
                            pc = Candle(float(p_sl["open"][pj]), float(p_sl["high"][pj]), float(p_sl["low"][pj]), float(p_sl["close"][pj]), minute=int(p_sl["times"][pj]))
                            marny_15m.update_1m(pc)
                            f6_tracker.push_1m(pc)

            # Current day processing
            for j in range(len(sl["times"])):
                m = int(sl["times"][j])
                c = Candle(float(sl["open"][j]), float(sl["high"][j]), float(sl["low"][j]), float(sl["close"][j]), minute=m)
                marny_15m.update_1m(c)
                is_marny_bullish = marny_15m.snapshot()

                f6_triggers = f6_tracker.push_1m(c)
                for trig in f6_triggers:
                    tf, is_rev, stype, px, atr_val = trig
                    if no_pinbar and not s1_turn_up and stype != "flag_nodiv":
                        continue
                    # Marny 15-minute Filter Gate on Option Chart
                    if is_marny_bullish:
                        all_events.append({
                            "minute": m,
                            "side": side,
                            "strike": strike,
                            "symbol": sym,
                            "key": key,
                            "option_entry": c.close,
                            "atr": atr_val if atr_val and atr_val > 0 else 5.0,
                            "stype": stype,
                            "tf": tf,
                        })

    # Deterministic ordering: first setup in time wins; same-minute ties broken
    # by TF priority (1m > 2m > 3m > 5m), then CE > PE, then lower strike.
    # This mirrors the live bot's decision order exactly (congruence requirement).
    TF_PRIORITY = {"1m": 0, "2m": 1, "3m": 2, "5m": 3}
    all_events.sort(key=lambda ev: (
        ev["minute"],
        TF_PRIORITY.get(ev["tf"], 99),
        0 if ev["side"] == "CE" else 1,
        ev["strike"],
    ))

    # Simulate trades
    events_by_min = defaultdict(list)
    for ev in all_events:
        events_by_min[ev["minute"]].append(ev)

    trades = []
    pos = None
    dpnl = 0.0
    closs = 0
    shut = False

    fixed_sl = p.get("fixed_sl", None)
    fixed_tp = p.get("fixed_tp", None)

    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None and minute > pos["entry_min"]:
            bar = option_bars_by_key[pos["key"]].get(minute)
            if bar is not None:
                high, low, close = bar["high"], bar["low"], bar["close"]
                sl_px = pos["sl"]
                tp_px = pos["tp"]

                # Trailing SL: update highest price and ratchet stop
                if trail_sl:
                    if high > pos["highest"]:
                        pos["highest"] = high
                        if fixed_sl is not None:
                            if pos["highest"] - pos["entry"] >= fixed_sl:
                                new_sl = pos["highest"] - fixed_sl
                                if new_sl > pos["sl"]:
                                    pos["sl"] = new_sl
                        else:
                            if pos["highest"] - pos["entry"] >= 1.5 * pos["atr"]:
                                new_sl = pos["highest"] - (atr_sl_mult * pos["atr"]) - theta_offset
                                if new_sl > pos["sl"]:
                                    pos["sl"] = new_sl
                    sl_px = pos["sl"]

                ex, rsn = None, ""
                if dpnl + (low - pos["entry"]) <= daily_loss_pts:
                    ex, rsn = pos["entry"] + (daily_loss_pts - dpnl), "DAILY_LOSS"
                    shut = True
                elif low <= sl_px and high >= tp_px:
                    ex, rsn = sl_px, "SL"
                elif high >= tp_px:
                    ex, rsn = tp_px, "TP"
                elif low <= sl_px:
                    ex, rsn = sl_px, "SL"
                elif time_stop_min is not None and minute - pos["entry_min"] >= time_stop_min:
                    ex, rsn = close, "TIME"
                elif minute >= SESSION_END:
                    ex, rsn = close, "EOD"

                if ex is not None:
                    slip = SLIPPAGE_PTS if include_fees else 0.0
                    brokerage = BROKERAGE_PER_ORDER if include_fees else 0.0
                    entry_fill = pos["entry"] + slip
                    exit_fill = ex - slip
                    pts = round(exit_fill - entry_fill, 2)
                    fee = trade_cost(entry_fill, exit_fill, brokerage) if include_fees else 0.0
                    net_rs = round(pts * LOT_SIZE - fee, 2)
                    trades.append({
                        "date": day,
                        "entry_min": pos["entry_min"],
                        "exit_min": minute,
                        "side": pos["side"],
                        "symbol": pos["symbol"],
                        "stype": pos["stype"],
                        "tf": pos["tf"],
                        "entry": entry_fill,
                        "exit": exit_fill,
                        "reason": rsn,
                        "points": pts,
                        "rs_net": net_rs,
                        "fee": fee,
                    })
                    dpnl += pts
                    closs = closs + 1 if net_rs <= 0 else 0
                    if closs >= f6_params.get("consec_loss", 8) or dpnl <= daily_loss_pts:
                        shut = True
                    pos = None

        if pos is not None or shut or minute >= SESSION_END:
            continue

        for ev in events_by_min.get(minute, []):
            if pos is not None:
                break
            atr_v = ev["atr"]
            sl_val = ev["option_entry"] - fixed_sl if fixed_sl is not None else ev["option_entry"] - (atr_sl_mult * atr_v) - theta_offset
            tp_val = ev["option_entry"] + fixed_tp if fixed_tp is not None else ev["option_entry"] + (atr_tp_mult * atr_v) - theta_offset
            pos = {
                **ev,
                "entry_min": minute,
                "entry": ev["option_entry"],
                "sl": sl_val,
                "tp": tp_val,
                "highest": ev["option_entry"],
            }

    return trades


def run_f6_marny_15m_filter_backtest(params, days_subset=None, workers=8,
                                     spot_all=None, opt_map=None):
    spot_all = spot_all if spot_all is not None else source.load_spot()
    opt_map = opt_map if opt_map is not None else source.option_day_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days if days_subset is None else days_subset
    previous = {day: max((c for c in opt_map if c < day), default="") for day in days}

    tasks = [
        (day, opt_map[day], opt_map.get(previous[day], ""), params)
        for day in days
    ]

    with Pool(processes=min(cpu_count(), workers), initializer=init_worker, initargs=(spot_all,)) as pool:
        all_day_trades = pool.map(process_day_f6_marny_15m_filter, tasks)

    all_trades = [t for day_trs in all_day_trades for t in day_trs]
    wins = [t for t in all_trades if t["rs_net"] > 0]
    losses = [t for t in all_trades if t["rs_net"] <= 0]
    loss_tot = abs(sum(t["rs_net"] for t in losses))
    win_tot = sum(t["rs_net"] for t in wins)
    net_rs = sum(t["rs_net"] for t in all_trades)
    net_pts = sum(t["points"] for t in all_trades)
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0.0
    pf = win_tot / loss_tot if loss_tot else 0.0
    fees = sum(t["fee"] for t in all_trades)

    equity, peak, max_dd = 0.0, 0.0, 0.0
    for t in sorted(all_trades, key=lambda x: (x["date"], x["entry_min"])):
        equity += t["rs_net"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": len(all_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr, 2),
        "net_rs": round(net_rs, 2),
        "net_points": round(net_pts, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown_rs": round(max_dd, 2),
        "fees_rs": round(fees, 2),
        "all_trades": all_trades,
    }


def main():
    parser = argparse.ArgumentParser(description="F6 Champion Strategy with Marny 15m Option Filter")
    parser.add_argument("--smoke", action="store_true", help="Run 5-day smoke test")
    parser.add_argument("--no-fees", action="store_true", help="Disable transaction fees")
    parser.add_argument("--no-trail", action="store_true", help="Disable trailing stop")
    parser.add_argument("--no-pinbar", action="store_true", help="Disable pinbar triggers (pure F6 Flag No-Div only)")
    parser.add_argument("--s1-turn-up", action="store_true", help="Trigger when S1 turns up (no pinbar)")
    parser.add_argument("--tf", action="append", choices=["1m", "2m", "3m", "5m"],
                        help="Restrict F6 engine to specific timeframes (repeatable; default = all)")
    parser.add_argument("--fixed-sl", type=float, default=None, help="Fixed Stop Loss in option points (e.g. 10.0)")
    parser.add_argument("--fixed-tp", type=float, default=None, help="Fixed Take Profit in option points (e.g. 15.0)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPU workers")
    args = parser.parse_args()

    include_fees = not args.no_fees
    trail_sl = not args.no_trail
    print("=" * 115)
    print("F6 CHAMPION 12/50 STRATEGY WITH MARNY CORE 15-MINUTE OPTION FILTER (2020-2026)")
    sl_tp_desc = f"Fixed SL={args.fixed_sl} pts, Fixed TP={args.fixed_tp} pts" if args.fixed_sl is not None else "ATR Dynamic SL/TP (x3/x6)"
    print(f"Risk Mechanics: {sl_tp_desc} | Trigger Mode: {'S1 Turn-Up' if args.s1_turn_up else 'No Pinbar (Flag No-Div)' if args.no_pinbar else 'Pinbar Breakout'} | Trailing SL: {trail_sl} | Include Fees: {include_fees} | Workers: {args.workers}")
    print("=" * 115)

    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    days = all_days[:5] if args.smoke else all_days

    params = {
        "f6_params": CHAMPION_12_50,
        "include_fees": include_fees,
        "trail_sl": trail_sl,
        "daily_loss_pts": -30.0,
        "no_pinbar": args.no_pinbar,
        "s1_turn_up": args.s1_turn_up,
        "fixed_sl": args.fixed_sl,
        "fixed_tp": args.fixed_tp,
        "tfs": args.tf,
    }

    t0 = time.time()
    res = run_f6_marny_15m_filter_backtest(params, days, args.workers)
    el = time.time() - t0

    print(f"\nExecution finished in {el:.2f}s ({len(days)/el:.1f} days/sec)")
    print("=" * 115)
    print(f"Overall Metrics: Trades={res['trades']} | Win Rate={res['win_rate']}% | Net Points={res['net_points']:+,.2f} | Net Realized Rs=Rs {res['net_rs']:+,.2f} | PF={res['profit_factor']} | MaxDD=Rs {res['max_drawdown_rs']:,.2f} | Fees=Rs {res['fees_rs']:,.2f}")

    # Year by year breakdown
    trades_by_year = defaultdict(list)
    for t in res["all_trades"]:
        yr = t["date"].split("-")[0]
        trades_by_year[yr].append(t)

    print("\n--- YEAR-BY-YEAR PERFORMANCE BREAKDOWN ---")
    print(f"{'Year':6s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'Profit Factor':13s} | {'Total Fees':12s}")
    print("-" * 115)
    for yr in sorted(trades_by_year.keys()):
        y_trs = trades_by_year[yr]
        wins = [t for t in y_trs if t["rs_net"] > 0]
        losses = [t for t in y_trs if t["rs_net"] <= 0]
        wr = len(wins) / len(y_trs) * 100 if y_trs else 0
        net_pts = sum(t["points"] for t in y_trs)
        net_rs = sum(t["rs_net"] for t in y_trs)
        loss_tot = abs(sum(t["rs_net"] for t in losses))
        win_tot = sum(t["rs_net"] for t in wins)
        pf = win_tot / loss_tot if loss_tot else 0.0
        fees = sum(t["fee"] for t in y_trs)
        print(f"{yr:6s} | {len(y_trs):7d} | {wr:7.1f}% | {net_pts:+10.2f} | Rs {net_rs:+13.2f} | {pf:13.2f} | Rs {fees:10.2f}")
    print("=" * 115)

    # Save summary
    out_file = ROOT / "artifacts" / "f6_hybrid" / "f6_champion_marny_15m_filter_results.json"
    with open(out_file, "w") as f:
        clean_res = {k: v for k, v in res.items() if k != "all_trades"}
        json.dump(clean_res, f, indent=2)
    print(f"\nSaved summary to: {out_file}")


if __name__ == "__main__":
    main()