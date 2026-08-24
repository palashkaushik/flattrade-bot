"""Audit Wide SL/TP Parameters on August 18, 19, 20, 2026."""

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


class WideDetector:
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


def run_wide_aug(days: list[str], sl: float, tp: float, l_trig: float, l_prof: float, trail: float, start_min: int = 570, end_min: int = 915):
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
            trk[sym] = WideDetector()
            for i in range(len(g["min"])):
                c = Candle(open=g["open"][i], high=g["high"][i], low=g["low"][i], close=g["close"][i], minute=g["min"][i])
                trk[sym].push(c)

        pmtrig = {}
        slices = {}
        for sym, g in gu.items():
            if sym not in trk:
                trk[sym] = WideDetector()
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

        for minute in range(start_min, end_min):
            if pos is not None:
                held = bslice(pos["slice"], minute)
                if held:
                    o, h, l, c = held
                    pos["last_px"] = float(c)
                    pos["duration_min"] += 1
                    if h > pos["peak_px"]:
                        pos["peak_px"] = float(h)

                    gain = pos["peak_px"] - pos["entry"]

                    # Profit lock
                    if gain >= l_trig:
                        lsl = pos["entry"] + l_prof
                        if lsl > pos["sl"]:
                            pos["sl"] = round(lsl, 2)
                            pos["is_locked"] = True

                    # Trail
                    if pos.get("is_locked"):
                        tsl = pos["peak_px"] - trail
                        if tsl > pos["sl"]:
                            pos["sl"] = round(tsl, 2)

                    ex, rsn = None, ""
                    if l <= pos["sl"] and h >= pos["tp"]:
                        ex, rsn = pos["tp"], "TARGET_TP"
                    elif h >= pos["tp"]:
                        ex, rsn = pos["tp"], "TARGET_TP"
                    elif l <= pos["sl"]:
                        ex, rsn = pos["sl"], "PROFIT_LOCK" if pos.get("is_locked") else "SL"

                    if ex is not None:
                        pts = round(ex - pos["entry"], 2)
                        rs_net = round(pts * LOT_SIZE - FEE, 2)
                        all_trades.append({
                            "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                            "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                            "exit": ex, "pts": pts, "rs_net": rs_net, "peak_gain": round(pos["peak_px"] - pos["entry"], 2),
                            "reason": rsn, "duration_min": pos["duration_min"], "stype": pos["stype"],
                        })
                        pos = None

            if minute >= end_min - 1 and pos is not None:
                pts = round(pos["last_px"] - pos["entry"], 2)
                rs_net = round(pts * LOT_SIZE - FEE, 2)
                all_trades.append({
                    "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                    "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                    "exit": pos["last_px"], "pts": pts, "rs_net": rs_net, "peak_gain": round(pos["peak_px"] - pos["entry"], 2),
                    "reason": "EOD_EXIT", "duration_min": pos["duration_min"], "stype": pos["stype"],
                })
                pos = None
                break

            if pos is not None:
                continue

            for (sig_side, sig_stk, sig_sym, c_px, stype) in pmtrig.get(minute, []):
                ai = ainfo(sig_side, minute)
                if ai and ai[2] == sig_stk:
                    bar = bslice(ai[1], minute)
                    if bar:
                        ep = float(bar[3])
                        pos = {
                            "entry": ep,
                            "sl": round(ep - sl, 2),
                            "tp": round(ep + tp, 2),
                            "side": sig_side, "symbol": ai[0], "entry_min": minute,
                            "last_px": ep, "peak_px": ep, "slice": ai[1],
                            "duration_min": 0, "is_locked": False,
                            "stype": stype,
                        }
                        break

    return all_trades


def main():
    days = ["2026-08-18", "2026-08-19", "2026-08-20"]
    print("=" * 145)
    print("AUGUST 18-20, 2026 AUDIT: WIDE STOP LOSS & TAKE PROFIT (SL >= 10.0 & TP >= 10.0)")
    print("=" * 145)

    configs = [
        ("Config A: SL = -10.0 pt | Lock +8.0 @ +10.0 pt | Trail = 3.0 pt | TP = +15.0 pt", 10.0, 15.0, 10.0, 8.0, 3.0),
        ("Config B: SL = -12.0 pt | Lock +10.0 @ +12.0 pt | Trail = 4.0 pt | TP = +20.0 pt", 12.0, 20.0, 12.0, 10.0, 4.0),
        ("Config C: SL = -15.0 pt | Lock +10.0 @ +15.0 pt | Trail = 4.0 pt | TP = +25.0 pt", 15.0, 25.0, 15.0, 10.0, 4.0),
        ("Config D: SL = -20.0 pt | Lock +8.0 @ +10.0 pt | Trail = 3.0 pt | TP = +15.0 pt", 20.0, 15.0, 10.0, 8.0, 3.0),
    ]

    for label, sl, tp, l_trig, l_prof, trail in configs:
        trades = run_wide_aug(days, sl, tp, l_trig, l_prof, trail)
        wins = [t for t in trades if t["rs_net"] > 0]
        losses = [t for t in trades if t["rs_net"] <= 0]
        wr = len(wins) / len(trades) * 100 if trades else 0.0
        tot_pts = sum(t["pts"] for t in trades)
        tot_rs = sum(t["rs_net"] for t in trades)
        pf = sum(t["rs_net"] for t in wins) / abs(sum(t["rs_net"] for t in losses)) if losses and abs(sum(t["rs_net"] for t in losses)) > 0 else 99.0

        print(f"\n{label}")
        print(f"Total Trades: {len(trades)} | Wins/Loss: {len(wins)}W / {len(losses)}L ({wr:.1f}%) | Net Points: {tot_pts:+.2f} pts | Net Realized Rs: Rs {tot_rs:+,.2f} | PF: {pf:.3f}")
        print("-" * 145)

        for day in days:
            d_trs = [t for t in trades if t["date"] == day]
            d_rs = sum(t["rs_net"] for t in d_trs)
            d_pts = sum(t["pts"] for t in d_trs)
            d_w = len([t for t in d_trs if t["rs_net"] > 0])
            status = "GREEN" if d_rs > 0 else ("RED" if d_rs < 0 else "FLAT")
            print(f"  {day}: {status:5s} | Trades: {len(d_trs)} ({d_w}W / {len(d_trs)-d_w}L) | Net Points: {d_pts:+6.2f} pts | Net Rs: Rs {d_rs:+8.2f}")


if __name__ == "__main__":
    main()
