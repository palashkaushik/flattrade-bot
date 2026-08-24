"""Run Golden Sweet Spot Champion and Twin-Peak Multi-Tier on August 18, 19, 20, 2026."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_5y_optimized import SYM_RE, latest_spot, option_files
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.test_super_only_aug import (
    extend_with_august, ParamStoch, bslice, to_hhmm, LOT_SIZE, FEE
)
from artifacts.f6_hybrid.compare_rules_1_and_2 import load_full_ohlc_spot
import grid_optimize_f6_atr as grid


class MultiTierDetector:
    def __init__(self):
        self.stoch = ParamStoch()
        self.prev_s1 = None
        self._fired = False

    def push(self, c: Candle):
        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
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
        return trig, "SUPER" if is_super else "FLAG", c.close


def run_strategy(
    days: list[str],
    mode: str = "GOLDEN_WINDOW",  # "GOLDEN_WINDOW" or "TWIN_PEAK"
    initial_sl_pts: float = 3.0,
    t1_trig: float = 5.0,
    t1_lock: float = 4.0,
    t2_trig: float = 10.0,
    t2_lock: float = 9.0,
    trail_dist: float = 1.5,
    hard_tp: float = 25.0,
):
    spot_all = load_full_ohlc_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    grid.GLOBAL_SPOT = spot_all
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}

    all_trades = []

    for day in days:
        fpath = opt_map.get(day)
        fprev = opt_map.get(all_cal[cal_idx[day] - 1]) if cal_idx.get(day, 0) > 0 else ""
        gc = grid.cached_day(str(fpath))
        if not gc:
            continue

        spot = spot_all.get(day)
        sp0 = latest_spot(spot, 555) or latest_spot(spot, 560)
        if sp0 is None:
            continue
        atm0 = int(round(sp0 / 50) * 50)
        target_strikes = set(range(atm0 - 250, atm0 + 300, 50))

        prefix = "NIFTY"
        for s in gc.keys():
            if (m := SYM_RE.match(s)):
                prefix = m.group(1)
                break

        gu = {sym: g for sym, g in gc.items() if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}
        gp = {sym: g for sym, g in grid.cached_day(str(fprev)).items() if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes} if fprev else {}

        trk = {}
        for sym, g in gp.items():
            trk[sym] = MultiTierDetector()
            for i in range(len(g["min"])):
                c = Candle(open=g["open"][i], high=g["high"][i], low=g["low"][i], close=g["close"][i], minute=g["min"][i])
                trk[sym].push(c)

        pmtrig = {}
        slices = {}
        for sym, g in gu.items():
            if sym not in trk:
                trk[sym] = MultiTierDetector()
            t = trk[sym]
            slices[sym] = g
            mm2 = SYM_RE.match(sym)
            if not mm2:
                continue
            sv, side = int(mm2.group(2)), mm2.group(3)
            for i in range(len(g["min"])):
                m = g["min"][i]
                c = Candle(open=g["open"][i], high=g["high"][i], low=g["low"][i], close=g["close"][i], minute=m)
                trig, stype, px = t.push(c)
                if trig:
                    pmtrig.setdefault(m, []).append((side, sv, sym, c.close, stype))

        def ainfo(side, m):
            spx = latest_spot(spot, m)
            if spx is None:
                return None
            atm = int(round(spx / 50) * 50)
            stk = atm + (-100 if side == "CE" else 100)
            sym = f"{prefix}{stk}{side}"
            sl = slices.get(sym)
            return (sym, sl, stk) if sl is not None else None

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

                    gain = pos["peak_px"] - pos["entry"]

                    # Tier 1 Lock
                    if gain >= t1_trig:
                        lsl1 = pos["entry"] + t1_lock
                        if lsl1 > pos["sl"]:
                            pos["sl"] = round(lsl1, 2)
                            pos["is_t1"] = True

                    # Tier 2 Lock & Trail
                    if gain >= t2_trig:
                        lsl2 = pos["entry"] + t2_lock
                        if lsl2 > pos["sl"]:
                            pos["sl"] = round(lsl2, 2)
                            pos["is_t2"] = True
                        tsl = pos["peak_px"] - trail_dist
                        if tsl > pos["sl"]:
                            pos["sl"] = round(tsl, 2)

                    ex, rsn = None, ""
                    if l <= pos["sl"] and h >= pos["tp"]:
                        ex, rsn = pos["sl"], "TIER_LOCK" if (pos.get("is_t1") or pos.get("is_t2")) else "SL"
                    elif h >= pos["tp"]:
                        ex, rsn = pos["tp"], "TARGET_TP"
                    elif l <= pos["sl"]:
                        ex, rsn = pos["sl"], "TIER_LOCK" if (pos.get("is_t1") or pos.get("is_t2")) else "SL"

                    if ex is not None:
                        pts = round(ex - pos["entry"], 2)
                        rs_net = round(pts * LOT_SIZE - FEE, 2)
                        all_trades.append({
                            "mode": mode, "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                            "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                            "exit": ex, "pts": pts, "rs_net": rs_net, "peak_gain": round(pos["peak_px"] - pos["entry"], 2),
                            "reason": rsn, "duration_min": pos["duration_min"], "stype": pos["stype"],
                        })
                        pos = None

            if minute >= 914 and pos is not None:
                pts = round(pos["last_px"] - pos["entry"], 2)
                rs_net = round(pts * LOT_SIZE - FEE, 2)
                all_trades.append({
                    "mode": mode, "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                    "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                    "exit": pos["last_px"], "pts": pts, "rs_net": rs_net, "peak_gain": round(pos["peak_px"] - pos["entry"], 2),
                    "reason": "EOD_EXIT", "duration_min": pos["duration_min"], "stype": pos["stype"],
                })
                pos = None
                break

            if pos is not None:
                continue

            # SESSION WINDOW
            is_permitted = False
            if mode == "GOLDEN_WINDOW":
                if 810 <= minute < 915:  # 13:30 to 15:15
                    is_permitted = True
            elif mode == "TWIN_PEAK":
                if (570 <= minute < 615) or (840 <= minute < 915):  # 09:30-10:15 OR 14:00-15:15
                    is_permitted = True

            if not is_permitted:
                continue

            for (sig_side, sig_stk, sig_sym, c_px, stype) in pmtrig.get(minute, []):
                ai = ainfo(sig_side, minute)
                if ai and ai[2] == sig_stk:
                    bar = bslice(ai[1], minute)
                    if bar:
                        ep = float(bar[3])
                        pos = {
                            "entry": ep,
                            "sl": round(ep - initial_sl_pts, 2),
                            "tp": round(ep + hard_tp, 2),
                            "side": sig_side, "symbol": ai[0], "entry_min": minute,
                            "last_px": ep, "peak_px": ep, "slice": ai[1],
                            "duration_min": 0, "is_t1": False, "is_t2": False,
                            "stype": stype,
                        }
                        break

    return all_trades


def main():
    days = ["2026-08-18", "2026-08-19", "2026-08-20"]
    print("=" * 145, flush=True)
    print("AUGUST 18-20, 2026 EXACT AUDIT: GOLDEN SWEET SPOT vs TWIN-PEAK MULTI-TIER", flush=True)
    print("Geometry: SL = -3.00 pt | Tier 1: Lock +4.0 @ +5.0 pt | Tier 2: Lock +9.0 @ +10.0 pt | Trail = 1.5 pt | TP = +25.0 pt", flush=True)
    print("=" * 145, flush=True)

    experiments = [
        ("1. GOLDEN SWEET SPOT CHAMPION (13:30-15:15 IST)", "GOLDEN_WINDOW"),
        ("2. TWIN-PEAK MULTI-TIER (09:30-10:15 + 14:00-15:15 IST)", "TWIN_PEAK"),
    ]

    for label, mode_key in experiments:
        trades = run_strategy(days, mode=mode_key)
        wins = [t for t in trades if t["rs_net"] > 0]
        losses = [t for t in trades if t["rs_net"] <= 0]
        wr = len(wins) / len(trades) * 100 if trades else 0.0
        tot_pts = sum(t["pts"] for t in trades)
        tot_rs = sum(t["rs_net"] for t in trades)
        pf = sum(t["rs_net"] for t in wins) / abs(sum(t["rs_net"] for t in losses)) if losses and abs(sum(t["rs_net"] for t in losses)) > 0 else 99.0

        print(f"\n{'='*145}", flush=True)
        print(f"{label.upper()} -> Total Trades: {len(trades)} | Wins/Loss: {len(wins)}W/{len(losses)}L ({wr:.1f}%) | Net Points: {tot_pts:+.2f} pts | Net Realized Rs: {tot_rs:+,.2f} | PF: {pf:.3f}", flush=True)
        print(f"{'='*145}", flush=True)

        for day in days:
            day_trs = [t for t in trades if t["date"] == day]
            day_w = [t for t in day_trs if t["rs_net"] > 0]
            day_l = [t for t in day_trs if t["rs_net"] <= 0]
            day_rs = sum(t["rs_net"] for t in day_trs)
            day_pts = sum(t["pts"] for t in day_trs)
            day_wr = len(day_w) / len(day_trs) * 100 if day_trs else 0

            status_str = "GREEN" if day_rs > 0 else ("RED" if day_rs < 0 else "FLAT (0 TRADES)")
            print(f"\nDATE: {day} | {status_str} ({len(day_trs)} Trades | {len(day_w)} Wins, {len(day_l)} Losses | Net Points: {day_pts:+.2f} pts | Net PnL: Rs {day_rs:+,.2f})", flush=True)
            if not day_trs:
                print("  [Zero Trades Taken - Midday & Neutral Chop Successfully Avoided]", flush=True)
                continue

            print(f"{'#':2s} | {'Time (IST)':11s} | {'Setup':5s} | {'Symbol':17s} | {'Side':4s} | {'Entry':7s} | {'Peak':7s} | {'Exit':7s} | {'Dur':6s} | {'Points':8s} | {'Net Rs':11s} | {'Exit Reason':12s}", flush=True)
            print("-" * 125, flush=True)
            for i, t in enumerate(day_trs, 1):
                time_str = f"{to_hhmm(t['entry_min'])}->{to_hhmm(t['exit_min'])}"
                peak_val = f"+{t['peak_gain']:.2f}"
                print(f"{i:2d} | {time_str:11s} | {t['stype']:5s} | {t['symbol']:17s} | {t['side']:4s} | {t['entry']:7.2f} | {peak_val:7s} | {t['exit']:7.2f} | {t['duration_min']:3d}m | {t['pts']:+7.2f} | Rs {t['rs_net']:+8.2f} | {t['reason']:12s}", flush=True)


if __name__ == "__main__":
    main()
