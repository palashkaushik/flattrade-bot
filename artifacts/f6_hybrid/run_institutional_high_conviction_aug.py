"""Institutional High-Conviction Engine on August 18, 19, 20, 2026.

Integrates the 4 Institutional Pillars:
  Pillar 1: 09:30 AM Session Start Gate (No opening 15m gap-chop entries)
  Pillar 2: 15-Minute TradingView-Exact HTF Index Trend Filter (PocketHTFFilter)
  Pillar 3: 4-Tier Theta-Beating Exit Engine (15-Min Theta Cut, BE Lock at +1.25x ATR, Trail Peak - 0.5x ATR)
  Pillar 4: Max 2-3 High-Conviction Trades / Day & 15-Minute Cooldown

Simulated with tick-exact 1-minute option candles & predecessor day warmup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_5y_optimized import SYM_RE, latest_spot, option_files
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.pocket_money_backtest import (
    build_index_filter, filter_allows,
)
from artifacts.f6_hybrid.test_super_only_aug import extend_with_august, ParamStoch, IncrementalATR, bslice, to_hhmm, LOT_SIZE, FEE
from artifacts.f6_hybrid.compare_rules_1_and_2 import load_full_ohlc_spot
import grid_optimize_f6_atr as grid


class InstitutionalDualDetector:
    def __init__(self):
        self.stoch = ParamStoch()
        self.atr = IncrementalATR(14)
        self.prev_s1 = None
        self._fired = False

    def push(self, c: Candle):
        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
        atr_val = self.atr.update(c.high, c.low, c.close)

        is_super = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
        is_flag = s4 is not None and s1 is not None and s4 >= 79.5 and s1 <= 20.5
        s1_turn_up = self.prev_s1 is not None and s1 is not None and s1 > self.prev_s1

        cond = (is_super or is_flag) and s1_turn_up
        trig = False
        if cond and not self._fired:
            trig = True
            self._fired = True
        if not cond:
            self._fired = False

        self.prev_s1 = s1
        return trig, "SUPER" if is_super else "FLAG", c.close, atr_val, (s1, s2, s3, s4)


def simulate_institutional_day(
    day: str,
    opt_map: dict,
    all_cal: list,
    cal_idx: dict,
    spot_all: dict,
    start_minute: int = 570,    # 09:30 AM
    end_minute: int = 900,      # 03:00 PM
    max_trades_day: int = 3,    # Max 3 trades per day
    cooldown_min: int = 15,     # 15 min cooldown between trades
    sl_mult: float = 2.00,      # 2.00 x ATR protective cushion
    be_trig_mult: float = 1.25, # +1.25x ATR gain -> BE lock
    trail_trig_mult: float = 1.50,
    trail_dist_mult: float = 0.50,
    time_stop_min: int = 15,    # 15 min Theta cut
    tp_mult: float = 3.50,      # 3.50 x ATR target
    use_htf: bool = True,
):
    fpath = opt_map.get(day)
    fprev = opt_map.get(all_cal[cal_idx[day] - 1]) if cal_idx.get(day, 0) > 0 else ""
    gc = grid.cached_day(str(fpath))
    if not gc:
        return []

    spot = spot_all.get(day)
    sp0 = latest_spot(spot, 555) or latest_spot(spot, 560)
    if sp0 is None:
        return []
    atm0 = int(round(sp0 / 50) * 50)
    target_strikes = set(range(atm0 - 250, atm0 + 300, 50))

    prefix = "NIFTY"
    for s in gc.keys():
        if (m := SYM_RE.match(s)):
            prefix = m.group(1)
            break

    def filtered(data):
        return {sym: g for sym, g in data.items()
                if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}

    gu = filtered(gc)
    gp = filtered(grid.cached_day(str(fprev))) if fprev else {}

    trk = {}
    for sym, g in gp.items():
        trk[sym] = InstitutionalDualDetector()
        for i in range(len(g["min"])):
            c = Candle(open=g["open"][i], high=g["high"][i],
                       low=g["low"][i], close=g["close"][i], minute=g["min"][i])
            trk[sym].push(c)

    pmtrig = {}
    slices = {}

    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = InstitutionalDualDetector()
        t = trk[sym]
        slices[sym] = g
        mm2 = SYM_RE.match(sym)
        if not mm2:
            continue
        sv, side = int(mm2.group(2)), mm2.group(3)
        for i in range(len(g["min"])):
            m = g["min"][i]
            c = Candle(open=g["open"][i], high=g["high"][i],
                       low=g["low"][i], close=g["close"][i], minute=m)
            trig, stype, px, atr_val, stoch_tuple = t.push(c)
            if trig:
                pmtrig.setdefault(m, []).append((side, sv, sym, c.close, stype, atr_val, stoch_tuple))

    def ainfo(side, m):
        spx = latest_spot(spot, m)
        if spx is None:
            return None
        atm = int(round(spx / 50) * 50)
        stk = atm + (-100 if side == "CE" else 100)
        sym = f"{prefix}{stk}{side}"
        sl = slices.get(sym)
        return (sym, sl, stk) if sl is not None else None

    # Compute TradingView-Exact 15m HTF Trend Filter Snapshots
    htf_snaps = build_index_filter(spot, day=day, warm_days=12) if (use_htf and spot is not None) else {}

    trades = []
    pos = None
    trades_today = 0
    last_exit_minute = -999

    for minute in range(560, 931):
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c)
                pos["duration_min"] += 1
                if h > pos["peak_px"]:
                    pos["peak_px"] = float(h)

                gain = pos["peak_px"] - pos["entry"]
                eff_atr = pos["eff_atr"]

                # Tier 4: 15-Min Theta Time Stop
                if pos["duration_min"] >= time_stop_min and not pos["is_be_locked"]:
                    theta_sl = pos["entry"] - (0.50 * eff_atr)
                    if theta_sl > pos["sl"]:
                        pos["sl"] = round(theta_sl, 2)
                        pos["is_theta_decay_active"] = True

                # Tier 2: Breakeven Lock at +1.25x ATR Gain
                if gain >= be_trig_mult * eff_atr:
                    be_sl = pos["entry"] + 0.50
                    if be_sl > pos["sl"]:
                        pos["sl"] = round(be_sl, 2)
                        pos["is_be_locked"] = True

                # Tier 3: Asymmetric Trailing Stop at +1.50x ATR Gain
                if gain >= trail_trig_mult * eff_atr:
                    trail_sl = pos["peak_px"] - (trail_dist_mult * eff_atr)
                    if trail_sl > pos["sl"]:
                        pos["sl"] = round(trail_sl, 2)
                        pos["is_trailing"] = True

                ex, rsn = None, ""
                if l <= pos["sl"] and h >= pos["tp"]:
                    ex, rsn = pos["sl"], "BE_LOCK" if pos.get("is_be_locked") else ("TRAIL_SL" if pos.get("is_trailing") else ("THETA_STOP" if pos.get("is_theta_decay_active") else "SL"))
                elif h >= pos["tp"]:
                    ex, rsn = pos["tp"], "TP"
                elif l <= pos["sl"]:
                    ex, rsn = pos["sl"], "BE_LOCK" if pos.get("is_be_locked") else ("TRAIL_SL" if pos.get("is_trailing") else ("THETA_STOP" if pos.get("is_theta_decay_active") else "SL"))

                if ex is not None:
                    pts = round(ex - pos["entry"], 2)
                    rs_net = round(pts * LOT_SIZE - FEE, 2)
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": ex, "pts": pts, "rs_net": rs_net, "fee": FEE,
                        "reason": rsn, "duration_min": pos["duration_min"], "stype": pos["stype"],
                    })
                    last_exit_minute = minute
                    pos = None

        if minute >= end_minute and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            rs_net = round(pts * LOT_SIZE - FEE, 2)
            trades.append({
                "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                "exit": pos["last_px"], "pts": pts, "rs_net": rs_net, "fee": FEE,
                "reason": "EOD", "duration_min": pos["duration_min"], "stype": pos["stype"],
            })
            last_exit_minute = minute
            pos = None
            break

        # Check Gates: Active Position, Session Window, Max Daily Trades, Cooldown
        if pos is not None or minute < start_minute or minute >= end_minute or trades_today >= max_trades_day:
            continue
        if minute < last_exit_minute + cooldown_min:
            continue  # Cooldown active

        for (sig_side, sig_stk, sig_sym, c_px, stype, atr_val, stoch_tuple) in pmtrig.get(minute, []):
            # Apply 15m Higher Timeframe Trend Filter Gate
            if use_htf:
                allowed_side = filter_allows(htf_snaps, minute)
                if allowed_side != sig_side:
                    continue

            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                bar = bslice(ai[1], minute)
                if bar:
                    ep = float(bar[3])
                    atr_effective = atr_val if atr_val is not None and atr_val > 0 else 8.0
                    sl_d = max(5.0, min(30.0, sl_mult * atr_effective))
                    tp_d = max(8.0, min(60.0, tp_mult * atr_effective))

                    pos = {
                        "entry": ep,
                        "sl": round(ep - sl_d, 2),
                        "tp": round(ep + tp_d, 2),
                        "side": sig_side, "symbol": ai[0], "entry_min": minute,
                        "last_px": ep, "peak_px": ep, "slice": ai[1],
                        "duration_min": 0, "eff_atr": atr_effective,
                        "is_trailing": False, "is_be_locked": False, "is_theta_decay_active": False,
                        "stype": stype,
                    }
                    trades_today += 1
                    break

    return trades


def main():
    spot_all = load_full_ohlc_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    grid.GLOBAL_SPOT = spot_all
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}

    target_days = ["2026-08-18", "2026-08-19", "2026-08-20"]
    all_trs = []

    print("=" * 135)
    print("INSTITUTIONAL HIGH-CONVICTION STRATEGY: AUGUST 18, 19, 20, 2026 EXECUTION")
    print("Core Rules: 09:30 AM Start | 15m Trend Gate | 15-Min Theta Cut | BE Lock +1.25x ATR | Trail +1.50x/0.50x | Max 3 Tr/Day")
    print("=" * 135)

    for day in target_days:
        trs = simulate_institutional_day(day, opt_map, all_cal, cal_idx, spot_all)
        all_trs.extend(trs)
        day_w = [t for t in trs if t["rs_net"] > 0]
        day_l = [t for t in trs if t["rs_net"] <= 0]
        day_net = sum(t["rs_net"] for t in trs)
        day_pts = sum(t["pts"] for t in trs)
        day_wr = len(day_w) / len(trs) * 100 if trs else 0.0

        print(f"\n>>> DATE: {day} (Total Trades: {len(trs)} | Wins: {len(day_w)}, Losses: {len(day_l)} | Win Rate: {day_wr:.1f}% | Net PnL: Rs {day_net:+,.2f}):")
        if len(trs) == 0:
            print("  * 0 Trades Triggered (100% Protected from counter-trend / chop market conditions)")
        else:
            print(f"{'#':2s} | {'Time':11s} | {'Symbol':18s} | {'Side':4s} | {'Entry':7s} | {'Exit':7s} | {'Duration':8s} | {'Points':7s} | {'Net Rs':12s} | {'Exit Reason':12s}")
            print("-" * 115)
            for i, t in enumerate(trs, 1):
                time_str = f"{to_hhmm(t['entry_min'])}->{to_hhmm(t['exit_min'])}"
                print(f"{i:2d} | {time_str:11s} | {t['symbol']:18s} | {t['side']:4s} | {t['entry']:7.2f} | {t['exit']:7.2f} | {t['duration_min']:4d} min | {t['pts']:+6.2f} | Rs {t['rs_net']:+9.2f} | {t['reason']:12s}")

    wins = [t for t in all_trs if t["rs_net"] > 0]
    losses = [t for t in all_trs if t["rs_net"] <= 0]
    net_rs = sum(t["rs_net"] for t in all_trs)
    net_pts = sum(t["pts"] for t in all_trs)
    wr = len(wins) / len(all_trs) * 100 if all_trs else 0.0
    win_tot = sum(t["rs_net"] for t in wins)
    loss_tot = abs(sum(t["rs_net"] for t in losses))
    pf = win_tot / loss_tot if loss_tot > 0 else 99.0

    print("\n" + "=" * 115)
    print(f"3-DAY INSTITUTIONAL SUMMARY (August 18, 19, 20, 2026):")
    print(f"  * Total Trades Taken:           {len(all_trs)} trades (vs 26 trades in raw high-frequency)")
    print(f"  * Total Wins / Losses:          {len(wins)} Wins / {len(losses)} Losses")
    print(f"  * Win Rate:                     {wr:.2f}%")
    print(f"  * Profit Factor:                {pf:.3f}")
    print(f"  * Net Points Captured:          {net_pts:+,.2f} pts")
    print(f"  * 3-Day Net Realized Profit:    Rs {net_rs:+,.2f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
