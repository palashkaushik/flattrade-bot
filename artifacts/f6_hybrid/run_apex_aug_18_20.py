"""Execute APEX RUNNER on August 18, 19, 20, 2026 with detailed trade frequency analysis."""

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


class HighYieldDetector:
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


def run_aug_simulation(
    days: list[str],
    start_min: int = 570,  # 09:30 AM (570) or 14:00 (840)
    end_min: int = 900,    # 15:00 PM
    max_trades_day: int = 999,
    cooldown_min: int = 0,
    allow_concurrent: bool = False,
):
    spot_all = load_full_ohlc_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    grid.GLOBAL_SPOT = spot_all
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}

    initial_sl_pts = 6.0
    lock_trigger_pts = 12.0
    locked_profit_pts = 10.0
    trail_dist_pts = 4.0
    hard_tp_pts = 20.0

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

        def filtered(data):
            return {sym: g for sym, g in data.items()
                    if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}

        gu = filtered(gc)
        gp = filtered(grid.cached_day(str(fprev))) if fprev else {}

        trk = {}
        for sym, g in gp.items():
            trk[sym] = HighYieldDetector()
            for i in range(len(g["min"])):
                c = Candle(open=g["open"][i], high=g["high"][i], low=g["low"][i], close=g["close"][i], minute=g["min"][i])
                trk[sym].push(c)

        pmtrig = {}
        slices = {}
        for sym, g in gu.items():
            if sym not in trk:
                trk[sym] = HighYieldDetector()
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

        active_positions = []
        trades_today = 0
        last_exit_minute = -999

        for minute in range(560, 931):
            # Process open positions
            remaining_positions = []
            for pos in active_positions:
                held = bslice(pos["slice"], minute)
                if held:
                    o, h, l, c = held
                    pos["last_px"] = float(c)
                    pos["duration_min"] += 1
                    if h > pos["peak_px"]:
                        pos["peak_px"] = float(h)

                    gain = pos["peak_px"] - pos["entry"]

                    # Profit lock
                    if gain >= lock_trigger_pts:
                        locked_sl = pos["entry"] + locked_profit_pts
                        if locked_sl > pos["sl"]:
                            pos["sl"] = round(locked_sl, 2)
                            pos["is_locked"] = True

                    # Trail
                    if pos["is_locked"]:
                        trail_sl = pos["peak_px"] - trail_dist_pts
                        if trail_sl > pos["sl"]:
                            pos["sl"] = round(trail_sl, 2)
                            pos["is_trailing"] = True

                    ex, rsn = None, ""
                    if l <= pos["sl"] and h >= pos["tp"]:
                        ex, rsn = pos["sl"], "PROFIT_LOCK" if pos.get("is_locked") else "SL"
                    elif h >= pos["tp"]:
                        ex, rsn = pos["tp"], "BIG_TP"
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
                        last_exit_minute = minute
                        continue

                # Force EOD exit at 15:00
                if minute >= end_min:
                    pts = round(pos["last_px"] - pos["entry"], 2)
                    rs_net = round(pts * LOT_SIZE - FEE, 2)
                    all_trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": pos["last_px"], "pts": pts, "rs_net": rs_net, "peak_gain": round(pos["peak_px"] - pos["entry"], 2),
                        "reason": "EOD_EXIT", "duration_min": pos["duration_min"], "stype": pos["stype"],
                    })
                    last_exit_minute = minute
                    continue

                remaining_positions.append(pos)

            active_positions = remaining_positions

            # Check new entries
            if minute < start_min or minute >= end_min or trades_today >= max_trades_day:
                continue
            if not allow_concurrent and len(active_positions) > 0:
                continue
            if minute < last_exit_minute + cooldown_min:
                continue

            for (sig_side, sig_stk, sig_sym, c_px, stype) in pmtrig.get(minute, []):
                ai = ainfo(sig_side, minute)
                if ai and ai[2] == sig_stk:
                    bar = bslice(ai[1], minute)
                    if bar:
                        ep = float(bar[3])
                        new_pos = {
                            "entry": ep,
                            "sl": round(ep - initial_sl_pts, 2),
                            "tp": round(ep + hard_tp_pts, 2),
                            "side": sig_side, "symbol": ai[0], "entry_min": minute,
                            "last_px": ep, "peak_px": ep, "slice": ai[1],
                            "duration_min": 0, "is_locked": False, "is_trailing": False,
                            "stype": stype,
                        }
                        active_positions.append(new_pos)
                        trades_today += 1
                        if not allow_concurrent:
                            break

    return all_trades


def main():
    days = ["2026-08-18", "2026-08-19", "2026-08-20"]
    print("=" * 135)
    print("APEX RUNNER: LIVE AUGUST 18-20, 2026 EXECUTION VERIFICATION")
    print("Settings: SL = -6.00 pts | Lock +10.00 pts at +12.00 pt Gain | Trail = 4.00 pts | Hard TP = +20.00 pts")
    print("=" * 135)

    # 1. Compare Trade Frequency Caps for Aug 18-20
    caps = [1, 2, 3, 5, 8, 999]
    print("\n--- A. DAILY TRADE FREQUENCY CAP COMPARISON (AUG 18–20, 2026) ---")
    print(f"{'Execution Mode / Cap':30s} | {'Trades':7s} | {'Wins/Loss':10s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'Profit Factor':13s}")
    print("-" * 115)

    for cap in caps:
        lbl = f"Sequential: {cap} Trade{'s' if cap>1 else ''}/Day" if cap < 999 else "Sequential: Uncapped"
        trs = run_aug_simulation(days, start_min=570, max_trades_day=cap, cooldown_min=0, allow_concurrent=False)
        w = [t for t in trs if t["rs_net"] > 0]
        l = [t for t in trs if t["rs_net"] <= 0]
        wr = len(w) / len(trs) * 100 if trs else 0
        tot_pts = sum(t["pts"] for t in trs)
        tot_rs = sum(t["rs_net"] for t in trs)
        pf = sum(t["rs_net"] for t in w) / abs(sum(t["rs_net"] for t in l)) if l and abs(sum(t["rs_net"] for t in l)) > 0 else 99.0
        print(f"{lbl:30s} | {len(trs):7d} | {len(w):3d}W / {len(l):3d}L | {wr:7.1f}% | {tot_pts:+10.2f} | Rs {tot_rs:+14.2f} | {pf:10.3f}")

    # Master Benchmark (All signals concurrent)
    trs_master = run_aug_simulation(days, start_min=570, max_trades_day=999, allow_concurrent=True)
    w_m = [t for t in trs_master if t["rs_net"] > 0]
    l_m = [t for t in trs_master if t["rs_net"] <= 0]
    wr_m = len(w_m) / len(trs_master) * 100 if trs_master else 0
    tot_pts_m = sum(t["pts"] for t in trs_master)
    tot_rs_m = sum(t["rs_net"] for t in trs_master)
    pf_m = sum(t["rs_net"] for t in w_m) / abs(sum(t["rs_net"] for t in l_m)) if l_m and abs(sum(t["rs_net"] for t in l_m)) > 0 else 99.0
    print(f"{'Master Concurrent (All Signals)':30s} | {len(trs_master):7d} | {len(w_m):3d}W / {len(l_m):3d}L | {wr_m:7.1f}% | {tot_pts_m:+10.2f} | Rs {tot_rs_m:+14.2f} | {pf_m:10.3f}")

    # Afternoon Power Session (14:00 - 15:00)
    trs_pm = run_aug_simulation(days, start_min=840, max_trades_day=999, allow_concurrent=False)
    w_pm = [t for t in trs_pm if t["rs_net"] > 0]
    l_pm = [t for t in trs_pm if t["rs_net"] <= 0]
    wr_pm = len(w_pm) / len(trs_pm) * 100 if trs_pm else 0
    tot_pts_pm = sum(t["pts"] for t in trs_pm)
    tot_rs_pm = sum(t["rs_net"] for t in trs_pm)
    pf_pm = sum(t["rs_net"] for t in w_pm) / abs(sum(t["rs_net"] for t in l_pm)) if l_pm and abs(sum(t["rs_net"] for t in l_pm)) > 0 else 99.0
    print(f"{'Afternoon Only (14:00–15:00)':30s} | {len(trs_pm):7d} | {len(w_pm):3d}W / {len(l_pm):3d}L | {wr_pm:7.1f}% | {tot_pts_pm:+10.2f} | Rs {tot_rs_pm:+14.2f} | {pf_pm:10.3f}")

    # 2. Detailed Trade-by-Trade Ledger for Sequential Uncapped (Sweet Spot)
    trs_seq = run_aug_simulation(days, start_min=570, max_trades_day=999, allow_concurrent=False)
    print("\n" + "=" * 145)
    print("--- B. COMPLETE TRADE-BY-TRADE AUDIT LEDGER (AUG 18, 19, 20, 2026) ---")
    print("=" * 145)

    for day in days:
        day_trs = [t for t in trs_seq if t["date"] == day]
        day_w = [t for t in day_trs if t["rs_net"] > 0]
        day_l = [t for t in day_trs if t["rs_net"] <= 0]
        day_rs = sum(t["rs_net"] for t in day_trs)
        day_pts = sum(t["pts"] for t in day_trs)
        day_wr = len(day_w) / len(day_trs) * 100 if day_trs else 0

        print(f"\nDATE: {day} (Total Trades: {len(day_trs)} | {len(day_w)} Wins, {len(day_l)} Losses | Win Rate: {day_wr:.1f}% | Net Points: {day_pts:+.2f} pts | Net PnL: Rs {day_rs:+,.2f})")
        print(f"{'#':2s} | {'Time (IST)':11s} | {'Setup':5s} | {'Symbol':17s} | {'Side':4s} | {'Entry':7s} | {'Peak':7s} | {'Exit':7s} | {'Dur':6s} | {'Points':8s} | {'Net Rs':11s} | {'Exit Reason':12s}")
        print("-" * 125)
        for i, t in enumerate(day_trs, 1):
            time_str = f"{to_hhmm(t['entry_min'])}->{to_hhmm(t['exit_min'])}"
            peak_val = f"+{t['peak_gain']:.2f}"
            print(f"{i:2d} | {time_str:11s} | {t['stype']:5s} | {t['symbol']:17s} | {t['side']:4s} | {t['entry']:7.2f} | {peak_val:7s} | {t['exit']:7.2f} | {t['duration_min']:3d}m | {t['pts']:+7.2f} | Rs {t['rs_net']:+8.2f} | {t['reason']:12s}")


if __name__ == "__main__":
    main()
