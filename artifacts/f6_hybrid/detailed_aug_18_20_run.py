"""Detailed Execution & Signal Trace on August 18, 19, 20, 2026.

Audits:
  1. Super Setup Only (S1, S2, S3, S4 <= 20.5 & S1 Turn-Up)
  2. Super + Flag Setup (All Signals) with Trail 0.75x/0.40x
  3. Signal & Stochastic telemetry on Aug 18, 19, 20
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

from backtest_5y_optimized import SYM_RE, latest_spot, load_spot, option_files
from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.indicators.stochastic import IncrementalStochastic
import grid_optimize_f6_atr as grid
from artifacts.f6_hybrid.test_super_only_aug import extend_with_august, ParamStoch, IncrementalATR, bslice, to_hhmm, LOT_SIZE, FEE


class DualTracker:
    def __init__(self):
        self.stoch = ParamStoch()
        self.atr = IncrementalATR(14)
        self.prev_s1 = None
        self._super_fired = False
        self._flag_fired = False

    def push(self, c: Candle):
        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
        atr_val = self.atr.update(c.high, c.low, c.close)

        is_super_setup = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
        is_flag_setup = s4 is not None and s1 is not None and s4 >= 79.5 and s1 <= 20.5
        s1_turn_up = self.prev_s1 is not None and s1 is not None and s1 > self.prev_s1

        is_super = is_super_setup and s1_turn_up
        is_flag = is_flag_setup and s1_turn_up

        trig_super = False
        if is_super and not self._super_fired:
            trig_super = True
            self._super_fired = True
        if not is_super:
            self._super_fired = False

        trig_flag = False
        if is_flag and not self._flag_fired:
            trig_flag = True
            self._flag_fired = True
        if not is_flag:
            self._flag_fired = False

        self.prev_s1 = s1
        return {
            "trig_super": trig_super,
            "trig_flag": trig_flag,
            "s1": s1, "s2": s2, "s3": s3, "s4": s4,
            "atr": atr_val,
            "close": c.close,
        }


def sim_august_mode(day, mode="super_only", sl_mult=1.50, tp_mult=3.00, trail_trig=0.75, trail_dist=0.40):
    spot_all = load_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}

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

    def filtered(data):
        return {sym: g for sym, g in data.items()
                if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}

    gu = filtered(gc)
    gp = filtered(grid.cached_day(str(fprev))) if fprev else {}

    trk = {}
    for sym, g in gp.items():
        trk[sym] = DualTracker()
        for i in range(len(g["min"])):
            c = Candle(open=g["open"][i], high=g["high"][i],
                       low=g["low"][i], close=g["close"][i], minute=g["min"][i])
            trk[sym].push(c)

    pmtrig = {}
    slices = {}
    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = DualTracker()
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
            info = t.push(c)
            should_trig = info["trig_super"] if mode == "super_only" else (info["trig_super"] or info["trig_flag"])
            stype = "SUPER" if info["trig_super"] else "FLAG"
            if should_trig:
                pmtrig.setdefault(m, []).append((side, sv, sym, c.close, stype, info["atr"]))

    prefix = "NIFTY"
    for s in gc.keys():
        if (m := SYM_RE.match(s)):
            prefix = m.group(1)
            break

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
                if trail_trig is not None:
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
                        "reason": rsn, "duration_min": pos["duration_min"], "stype": pos["stype"],
                    })
                    pos = None

        if minute >= 900 and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            rs_net = round(pts * LOT_SIZE - FEE, 2)
            trades.append({
                "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                "exit": pos["last_px"], "pts": pts, "rs_net": rs_net, "fee": FEE,
                "reason": "EOD", "duration_min": pos["duration_min"], "stype": pos["stype"],
            })
            pos = None
            break

        if pos is not None or minute >= 900:
            continue

        for (sig_side, sig_stk, sig_sym, c_px, stype, atr_val) in pmtrig.get(minute, []):
            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                bar = bslice(ai[1], minute)
                if bar:
                    ep = float(bar[3])
                    atr_effective = atr_val if atr_val is not None and atr_val > 0 else 8.0
                    sl_dist = max(5.0, min(30.0, sl_mult * atr_effective))
                    tp_dist = max(8.0, min(60.0, tp_mult * atr_effective))

                    pos = {
                        "entry": ep,
                        "sl": round(ep - sl_dist, 2),
                        "tp": round(ep + tp_dist, 2),
                        "side": sig_side, "symbol": ai[0], "entry_min": minute,
                        "last_px": ep, "peak_px": ep, "slice": ai[1],
                        "duration_min": 0, "eff_atr": atr_effective, "is_trailing": False,
                        "stype": stype,
                    }
                    break

    return trades


def main():
    target_days = ["2026-08-18", "2026-08-19", "2026-08-20"]

    print("=" * 135)
    print("AUGUST 18, 19, 20, 2026 EXECUTION COMPARISON")
    print("=" * 135)

    # 1. Strategy A: Super Setup Only (Consistent Champion)
    print("\n" + "#" * 40 + " 1. SUPER SETUP ONLY (CONSISTENT CHAMPION) " + "#" * 40)
    all_super = []
    for d in target_days:
        trs = sim_august_mode(d, mode="super_only", sl_mult=1.50, tp_mult=3.00, trail_trig=0.75, trail_dist=0.40)
        all_super.extend(trs)
        print(f"\n>>> DATE: {d} (Trades: {len(trs)})")
        if not trs:
            print("  [IDLE] 0 Trades Triggered (Market in non-extreme zone; no chop taken)")
        else:
            print(f"{'#':2s} | {'Time':11s} | {'Symbol':18s} | {'Side':4s} | {'Entry':7s} | {'Exit':7s} | {'Duration':8s} | {'Pts':7s} | {'Net Rs':12s} | {'Reason':10s}")
            print("-" * 110)
            for i, t in enumerate(trs, 1):
                time_str = f"{to_hhmm(t['entry_min'])}->{to_hhmm(t['exit_min'])}"
                print(f"{i:2d} | {time_str:11s} | {t['symbol']:18s} | {t['side']:4s} | {t['entry']:7.2f} | {t['exit']:7.2f} | {t['duration_min']:4d} min | {t['pts']:+6.2f} | Rs {t['rs_net']:+9.2f} | {t['reason']:10s}")

    net_super_rs = sum(t["rs_net"] for t in all_super)
    net_super_pts = sum(t["pts"] for t in all_super)
    print(f"\n  [SUPER ONLY SUMMARY 18-20 AUG]: Total Trades: {len(all_super)} | Net PnL: Rs {net_super_rs:+,.2f} | Net Points: {net_super_pts:+,.2f}")

    # 2. Strategy B: Super + Flag Setup with Tight Trailing (Trail 0.75x / 0.40x)
    print("\n" + "#" * 40 + " 2. SUPER + FLAG SETUP (WITH TRAIL=0.75x/0.40x) " + "#" * 40)
    all_all = []
    for d in target_days:
        trs = sim_august_mode(d, mode="all_signals", sl_mult=1.50, tp_mult=3.00, trail_trig=0.75, trail_dist=0.40)
        all_all.extend(trs)
        print(f"\n>>> DATE: {d} (Trades: {len(trs)})")
        if not trs:
            print("  0 Trades Triggered")
        else:
            print(f"{'#':2s} | {'Time':11s} | {'Symbol':18s} | {'Side':4s} | {'Entry':7s} | {'Exit':7s} | {'Duration':8s} | {'Pts':7s} | {'Net Rs':12s} | {'Reason':10s}")
            print("-" * 110)
            for i, t in enumerate(trs, 1):
                time_str = f"{to_hhmm(t['entry_min'])}->{to_hhmm(t['exit_min'])}"
                print(f"{i:2d} | {time_str:11s} | {t['symbol']:18s} | {t['side']:4s} | {t['entry']:7.2f} | {t['exit']:7.2f} | {t['duration_min']:4d} min | {t['pts']:+6.2f} | Rs {t['rs_net']:+9.2f} | {t['reason']:10s}")

    wins = [t for t in all_all if t["rs_net"] > 0]
    losses = [t for t in all_all if t["rs_net"] <= 0]
    net_all_rs = sum(t["rs_net"] for t in all_all)
    net_all_pts = sum(t["pts"] for t in all_all)
    wr = len(wins) / len(all_all) * 100 if all_all else 0.0

    print(f"\n  [SUPER + FLAG SUMMARY 18-20 AUG]: Total Trades: {len(all_all)} | Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {wr:.1f}% | Net PnL: Rs {net_all_rs:+,.2f} | Net Points: {net_all_pts:+,.2f}")
    print("=" * 135)


if __name__ == "__main__":
    main()
