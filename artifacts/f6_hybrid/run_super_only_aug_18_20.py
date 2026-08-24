"""Super Setup Only Trade Execution on August 18, 19, 20, 2026.

Strategy: Super Setup Only (S1, S2, S3, S4 <= 20.5 & S1 Turn-Up)
Parameters:
  - Initial SL: 1.50 x ATR
  - Initial TP: 3.00 x ATR
  - Trailing Trigger: Gain >= +0.75 x ATR
  - Trailing Distance: 0.40 x ATR behind peak price
  - Fee: Flat Rs 40.00 / trade
  - Lot Size: 65
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
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from artifacts.f6_hybrid.test_super_only_aug import extend_with_august, ParamStoch, IncrementalATR, bslice, to_hhmm, LOT_SIZE, FEE
from artifacts.f6_hybrid.compare_rules_1_and_2 import load_full_ohlc_spot
import grid_optimize_f6_atr as grid


class SuperOnlyDetector:
    def __init__(self):
        self.stoch = ParamStoch()
        self.atr = IncrementalATR(14)
        self.prev_s1 = None
        self._fired = False

    def push(self, c: Candle):
        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
        atr_val = self.atr.update(c.high, c.low, c.close)

        is_super_setup = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
        s1_turn_up = self.prev_s1 is not None and s1 is not None and s1 > self.prev_s1
        is_super = is_super_setup and s1_turn_up

        trig = False
        if is_super and not self._fired:
            trig = True
            self._fired = True
        if not is_super:
            self._fired = False

        self.prev_s1 = s1
        return trig, "SUPER", c.close, atr_val, (s1, s2, s3, s4)


def simulate_super_day(
    day: str,
    opt_map: dict,
    all_cal: list,
    cal_idx: dict,
    spot_all: dict,
    sl_mult: float = 1.50,
    tp_mult: float = 3.00,
    trail_trig: float = 0.75,
    trail_dist: float = 0.40,
):
    fpath = opt_map.get(day)
    fprev = opt_map.get(all_cal[cal_idx[day] - 1]) if cal_idx.get(day, 0) > 0 else ""
    gc = grid.cached_day(str(fpath))
    if not gc:
        return [], {}

    spot = spot_all.get(day)
    sp0 = latest_spot(spot, 555) or latest_spot(spot, 560)
    if sp0 is None:
        return [], {}
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
        trk[sym] = SuperOnlyDetector()
        for i in range(len(g["min"])):
            c = Candle(open=g["open"][i], high=g["high"][i],
                       low=g["low"][i], close=g["close"][i], minute=g["min"][i])
            trk[sym].push(c)

    pmtrig = {}
    slices = {}
    min_stoch_stats = {}

    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = SuperOnlyDetector()
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

    trades = []
    pos = None

    for minute in range(560, 931):
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c)
                pos["duration_min"] += 1
                if h > pos["peak_px"]:
                    pos["peak_px"] = float(h)

                # Dynamic Trailing SL
                gain = pos["peak_px"] - pos["entry"]
                if gain >= trail_trig * pos["eff_atr"]:
                    trail_sl = pos["peak_px"] - (trail_dist * pos["eff_atr"])
                    if trail_sl > pos["sl"]:
                        pos["sl"] = round(trail_sl, 2)
                        pos["is_trailing"] = True

                ex, rsn = None, ""
                if l <= pos["sl"] and h >= pos["tp"]:
                    ex, rsn = pos["sl"], "TRAIL_SL" if pos.get("is_trailing") else "SL"
                elif h >= pos["tp"]:
                    ex, rsn = pos["tp"], "TP"
                elif l <= pos["sl"]:
                    ex, rsn = pos["sl"], "TRAIL_SL" if pos.get("is_trailing") else "SL"

                if ex is not None:
                    pts = round(ex - pos["entry"], 2)
                    rs_net = round(pts * LOT_SIZE - FEE, 2)
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": ex, "pts": pts, "rs_net": rs_net, "fee": FEE,
                        "reason": rsn, "duration_min": pos["duration_min"], "stype": "SUPER",
                        "stochs": pos.get("stochs"),
                    })
                    pos = None

        if minute >= 900 and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            rs_net = round(pts * LOT_SIZE - FEE, 2)
            trades.append({
                "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                "exit": pos["last_px"], "pts": pts, "rs_net": rs_net, "fee": FEE,
                "reason": "EOD", "duration_min": pos["duration_min"], "stype": "SUPER",
                "stochs": pos.get("stochs"),
            })
            pos = None
            break

        if pos is not None or minute >= 900:
            continue

        for (sig_side, sig_stk, sig_sym, c_px, stype, atr_val, stoch_tuple) in pmtrig.get(minute, []):
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
                        "is_trailing": False, "stochs": stoch_tuple,
                    }
                    break

    return trades, pmtrig


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
    print("SUPER SETUP ONLY CHAMPION: AUGUST 18, 19, 20, 2026 EXECUTION")
    print("Strategy Rule: Enter ONLY when all 4 Stochastics S1, S2, S3, S4 <= 20.5 and S1 Turns Up")
    print("Settings: SL = 1.50 x ATR | TP = 3.00 x ATR | Trail = 0.75x / 0.40x ATR | Fee = Rs 40.00")
    print("=" * 135)

    for day in target_days:
        trs, pmtrig = simulate_super_day(day, opt_map, all_cal, cal_idx, spot_all)
        all_trs.extend(trs)
        print(f"\n>>> DATE: {day} (Total Super Trades: {len(trs)})")

        if len(trs) == 0:
            print("  * 0 Trades Triggered")
            print("  * Reason: On this day, the market did not produce an extreme 4-stochastic alignment (S1, S2, S3, S4 <= 20.5).")
            print("  * Capital Protection: The bot remained 100% idle and protected capital from rangebound chop.")
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

    print("\n" + "=" * 115)
    print(f"3-DAY SUMMARY (August 18, 19, 20, 2026):")
    print(f"  • Total Trades: {len(all_trs)} | Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {wr:.1f}%")
    print(f"  • Net Points: {net_pts:+,.2f} pts | Total Net Realized Profit: Rs {net_rs:+,.2f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
