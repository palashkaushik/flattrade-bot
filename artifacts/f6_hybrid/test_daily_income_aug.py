"""Test Daily Income Cash Machine with Confirmation on August 18, 19, 20, 2026."""

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


class DailyIncomeDetector:
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


def run_daily_income_aug(
    days: list[str],
    daily_target_rs: float = 800.0,
    daily_loss_guard_rs: float = 400.0,
    max_daily_trades: int = 3,
    initial_sl_pts: float = 3.0,
    hard_tp_pts: float = 8.0,
    lock_trigger_pts: float = 6.0,
    locked_profit_pts: float = 4.0,
    trail_dist_pts: float = 1.5,
    start_min: int = 840,  # 14:00 PM Afternoon Power
):
    spot_all = load_full_ohlc_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    grid.GLOBAL_SPOT = spot_all
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}

    all_trades = []
    daily_summary = []

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
            trk[sym] = DailyIncomeDetector()
            for i in range(len(g["min"])):
                c = Candle(open=g["open"][i], high=g["high"][i], low=g["low"][i], close=g["close"][i], minute=g["min"][i])
                trk[sym].push(c)

        pmtrig = {}
        slices = {}
        for sym, g in gu.items():
            if sym not in trk:
                trk[sym] = DailyIncomeDetector()
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
        day_rs = 0.0
        day_pts = 0.0
        day_trades = 0
        day_finished = False
        day_finish_reason = ""

        for minute in range(start_min, 931):
            if day_finished:
                break

            if pos is not None:
                held = bslice(pos["slice"], minute)
                if held:
                    o, h, l, c = held
                    pos["last_px"] = float(c)
                    pos["duration_min"] += 1
                    if h > pos["peak_px"]:
                        pos["peak_px"] = float(h)

                    gain = pos["peak_px"] - pos["entry"]

                    # Quick Profit Lock
                    if gain >= lock_trigger_pts:
                        locked_sl = pos["entry"] + locked_profit_pts
                        if locked_sl > pos["sl"]:
                            pos["sl"] = round(locked_sl, 2)
                            pos["is_locked"] = True

                    # Micro Trail
                    if pos.get("is_locked"):
                        trail_sl = pos["peak_px"] - trail_dist_pts
                        if trail_sl > pos["sl"]:
                            pos["sl"] = round(trail_sl, 2)

                    ex, rsn = None, ""
                    if l <= pos["sl"] and h >= pos["tp"]:
                        ex, rsn = pos["tp"], "TARGET_HIT"
                    elif h >= pos["tp"]:
                        ex, rsn = pos["tp"], "TARGET_HIT"
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
                        day_rs += rs_net
                        day_pts += pts
                        day_trades += 1
                        pos = None

                        # Check Daily Circuit Breakers
                        if day_rs >= daily_target_rs:
                            day_finished = True
                            day_finish_reason = f"TARGET_ACHIEVED (+Rs {day_rs:,.2f})"
                            break
                        if day_rs <= -daily_loss_guard_rs:
                            day_finished = True
                            day_finish_reason = f"LOSS_GUARD_STOP (-Rs {abs(day_rs):,.2f})"
                            break
                        if day_trades >= max_daily_trades:
                            day_finished = True
                            day_finish_reason = f"MAX_TRADES_REACHED ({day_trades} trades)"
                            break

            if minute >= 900 and pos is not None:
                pts = round(pos["last_px"] - pos["entry"], 2)
                rs_net = round(pts * LOT_SIZE - FEE, 2)
                all_trades.append({
                    "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                    "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                    "exit": pos["last_px"], "pts": pts, "rs_net": rs_net, "peak_gain": round(pos["peak_px"] - pos["entry"], 2),
                    "reason": "EOD_EXIT", "duration_min": pos["duration_min"], "stype": pos["stype"],
                })
                day_rs += rs_net
                day_pts += pts
                pos = None
                break

            if pos is not None or day_finished:
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
                            "tp": round(ep + hard_tp_pts, 2),
                            "side": sig_side, "symbol": ai[0], "entry_min": minute,
                            "last_px": ep, "peak_px": ep, "slice": ai[1],
                            "duration_min": 0, "is_locked": False,
                            "stype": stype,
                        }
                        break

        daily_summary.append({
            "date": day, "trades": day_trades, "pts": day_pts, "rs_net": day_rs,
            "status": "GREEN" if day_rs > 0 else ("RED" if day_rs < 0 else "FLAT"),
            "reason": day_finish_reason or "SESSION_END",
        })

    return all_trades, daily_summary


def main():
    days = ["2026-08-18", "2026-08-19", "2026-08-20"]
    print("=" * 145, flush=True)
    print("DAILY INCOME MACHINE: AUGUST 18-20, 2026 EXACT AUDIT", flush=True)
    print("Settings: Daily Target = Rs 800 | Loss Guard = Rs 400 | Max Trades = 2 | SL = -3.0 pt | TP = +8.0 pt | Lock +4 @ +6", flush=True)
    print("=" * 145, flush=True)

    trades, d_sum = run_daily_income_aug(days, daily_target_rs=800.0, daily_loss_guard_rs=400.0, max_daily_trades=2, initial_sl_pts=3.0, hard_tp_pts=8.0)

    print("\n--- DAILY PERFORMANCE SUMMARY ---", flush=True)
    for d in d_sum:
        color = "GREEN" if d["rs_net"] > 0 else ("RED" if d["rs_net"] < 0 else "FLAT")
        print(f"Date: {d['date']} | Status: {color:5s} | Trades: {d['trades']} | Net Points: {d['pts']:+6.2f} pts | Net PnL: Rs {d['rs_net']:+8.2f} | Reason: {d['reason']}", flush=True)

    tot_rs = sum(d["rs_net"] for d in d_sum)
    tot_pts = sum(d["pts"] for d in d_sum)
    green_cnt = sum(1 for d in d_sum if d["rs_net"] > 0)
    print(f"\n3-Day Total: Net Rs {tot_rs:+,.2f} | Net Points: {tot_pts:+.2f} pts | Green Days: {green_cnt}/{len(d_sum)} ({green_cnt/len(d_sum)*100:.1f}%)", flush=True)

    print("\n--- TRADE-BY-TRADE LEDGER ---", flush=True)
    print(f"{'#':2s} | {'Date':10s} | {'Time (IST)':11s} | {'Setup':5s} | {'Symbol':17s} | {'Side':4s} | {'Entry':7s} | {'Peak':7s} | {'Exit':7s} | {'Dur':6s} | {'Points':8s} | {'Net Rs':11s} | {'Exit Reason':12s}", flush=True)
    print("-" * 140, flush=True)
    for i, t in enumerate(trades, 1):
        time_str = f"{to_hhmm(t['entry_min'])}->{to_hhmm(t['exit_min'])}"
        peak_val = f"+{t['peak_gain']:.2f}"
        print(f"{i:2d} | {t['date']:10s} | {time_str:11s} | {t['stype']:5s} | {t['symbol']:17s} | {t['side']:4s} | {t['entry']:7.2f} | {peak_val:7s} | {t['exit']:7.2f} | {t['duration_min']:3d}m | {t['pts']:+7.2f} | Rs {t['rs_net']:+8.2f} | {t['reason']:12s}", flush=True)


if __name__ == "__main__":
    main()
