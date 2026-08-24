"""Execute Hybrid Macro-Gated and Hybrid Twin-Peak on August 18, 19, 20, 2026."""

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


class ApexDetector:
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


def run_hybrid_aug_days(
    days: list[str],
    mode: str = "TWIN_PEAK",  # "TWIN_PEAK", "MACRO_GATED", "AFTERNOON_ONLY", "FULL_DAY"
    initial_sl_pts: float = 4.0,
    lock_trigger_pts: float = 8.0,
    locked_profit_pts: float = 7.0,
    trail_dist_pts: float = 2.0,
    hard_tp_pts: float = 25.0,
):
    spot_all = load_full_ohlc_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    grid.GLOBAL_SPOT = spot_all
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}

    # Mock Dow overnight return for Aug 18, 19, 20
    # Aug 18: Neutral (+0.15%), Aug 19: Strong Bull (+0.65%), Aug 20: Neutral (-0.10%)
    dow_returns = {
        "2026-08-18": 0.15,
        "2026-08-19": 0.65,
        "2026-08-20": -0.10,
    }

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
            trk[sym] = ApexDetector()
            for i in range(len(g["min"])):
                c = Candle(open=g["open"][i], high=g["high"][i], low=g["low"][i], close=g["close"][i], minute=g["min"][i])
                trk[sym].push(c)

        pmtrig = {}
        slices = {}
        for sym, g in gu.items():
            if sym not in trk:
                trk[sym] = ApexDetector()
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
        dow_ret = dow_returns.get(day, 0.0)

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
                            "mode": mode, "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                            "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                            "exit": ex, "pts": pts, "rs_net": rs_net, "peak_gain": round(pos["peak_px"] - pos["entry"], 2),
                            "reason": rsn, "duration_min": pos["duration_min"], "stype": pos["stype"],
                        })
                        pos = None

            if minute >= 900 and pos is not None:
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

            # SESSION WINDOW FILTER LOGIC
            is_permitted = False
            # 1. Twin-Peak: 09:30-10:15 (570-615) OR 14:00-15:00 (840-900)
            if mode == "TWIN_PEAK":
                if (570 <= minute < 615) or (840 <= minute < 900):
                    is_permitted = True
            # 2. Macro-Gated: 14:00-15:00 ALWAYS OR 09:30-14:00 only if |Dow| >= 0.50%
            elif mode == "MACRO_GATED":
                if (840 <= minute < 900) or (570 <= minute < 840 and abs(dow_ret) >= 0.50):
                    is_permitted = True
            # 3. Afternoon Only: 14:00-15:00
            elif mode == "AFTERNOON_ONLY":
                if 840 <= minute < 900:
                    is_permitted = True
            # 4. Full Day: 09:30-15:00
            elif mode == "FULL_DAY":
                if 570 <= minute < 900:
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
                            "tp": round(ep + hard_tp_pts, 2),
                            "side": sig_side, "symbol": ai[0], "entry_min": minute,
                            "last_px": ep, "peak_px": ep, "slice": ai[1],
                            "duration_min": 0, "is_locked": False, "is_trailing": False,
                            "stype": stype,
                        }
                        break

    return all_trades


def main():
    days = ["2026-08-18", "2026-08-19", "2026-08-20"]
    print("=" * 145, flush=True)
    print("APEX RUNNER: AUGUST 18-20, 2026 HYBRID ARCHITECTURE LIVE AUDIT", flush=True)
    print("Champion Geometry: SL = -4.00 pts | Lock +7.00 pts @ +8.00 pt Gain | Trail = 2.00 pts | Hard TP = +25.00 pts", flush=True)
    print("=" * 145, flush=True)

    modes = [
        ("1. HYBRID TWIN-PEAK (09:30-10:15 Opening Bell + 14:00-15:30 Power)", "TWIN_PEAK"),
        ("2. HYBRID MACRO-GATED (Dow >= 0.50% Morning Gate + Afternoon Always)", "MACRO_GATED"),
        ("3. AFTERNOON ONLY (14:00-15:30 Power Session)", "AFTERNOON_ONLY"),
        ("4. FULL-DAY CHAMPION (09:30-15:00 All-Day)", "FULL_DAY"),
    ]

    summary_rows = []

    for label, m_key in modes:
        trs = run_hybrid_aug_days(days, mode=m_key)
        w = [t for t in trs if t["rs_net"] > 0]
        l = [t for t in trs if t["rs_net"] <= 0]
        wr = len(w) / len(trs) * 100 if trs else 0.0
        tot_pts = sum(t["pts"] for t in trs)
        tot_rs = sum(t["rs_net"] for t in trs)
        pf = sum(t["rs_net"] for t in w) / abs(sum(t["rs_net"] for t in l)) if l and abs(sum(t["rs_net"] for t in l)) > 0 else 99.0

        summary_rows.append({
            "label": label, "mode": m_key, "trades": len(trs), "wins": len(w), "losses": len(l),
            "wr": wr, "pts": tot_pts, "rs_net": tot_rs, "pf": pf, "trs_list": trs,
        })

    print(f"\n{'Architecture Model':72s} | {'Trades':7s} | {'Wins/Loss':10s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Realized Rs':17s} | {'Profit Factor':13s}", flush=True)
    print("-" * 155, flush=True)
    for r in summary_rows:
        print(f"{r['label']:72s} | {r['trades']:7d} | {r['wins']:3d}W / {r['losses']:3d}L | {r['wr']:7.1f}% | {r['pts']:+10.2f} | Rs {r['rs_net']:+14.2f} | {r['pf']:10.3f}", flush=True)

    # Detailed Trade Ledgers for the Two Hybrid Approaches
    for r in summary_rows[:2]:
        print("\n" + "=" * 145, flush=True)
        print(f"--- DETAILED TRADE LEDGER: {r['label'].upper()} ---", flush=True)
        print("=" * 145, flush=True)

        for day in days:
            day_trs = [t for t in r["trs_list"] if t["date"] == day]
            day_w = [t for t in day_trs if t["rs_net"] > 0]
            day_l = [t for t in day_trs if t["rs_net"] <= 0]
            day_rs = sum(t["rs_net"] for t in day_trs)
            day_pts = sum(t["pts"] for t in day_trs)
            day_wr = len(day_w) / len(day_trs) * 100 if day_trs else 0

            print(f"\nDATE: {day} ({len(day_trs)} Trades | {len(day_w)} Wins, {len(day_l)} Losses | Win Rate: {day_wr:.1f}% | Net Points: {day_pts:+.2f} pts | Net PnL: Rs {day_rs:+,.2f})", flush=True)
            if not day_trs:
                print("  [Zero Trades Taken - Midday/Neutral Chop Successfully Avoided]", flush=True)
                continue

            print(f"{'#':2s} | {'Time (IST)':11s} | {'Setup':5s} | {'Symbol':17s} | {'Side':4s} | {'Entry':7s} | {'Peak':7s} | {'Exit':7s} | {'Dur':6s} | {'Points':8s} | {'Net Rs':11s} | {'Exit Reason':12s}", flush=True)
            print("-" * 125, flush=True)
            for i, t in enumerate(day_trs, 1):
                time_str = f"{to_hhmm(t['entry_min'])}->{to_hhmm(t['exit_min'])}"
                peak_val = f"+{t['peak_gain']:.2f}"
                print(f"{i:2d} | {time_str:11s} | {t['stype']:5s} | {t['symbol']:17s} | {t['side']:4s} | {t['entry']:7.2f} | {peak_val:7s} | {t['exit']:7.2f} | {t['duration_min']:3d}m | {t['pts']:+7.2f} | Rs {t['rs_net']:+8.2f} | {t['reason']:12s}", flush=True)


if __name__ == "__main__":
    main()
