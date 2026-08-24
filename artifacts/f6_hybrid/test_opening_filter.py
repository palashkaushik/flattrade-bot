"""Test Start Time Filter (09:45 AM) + 2:1 R:R on August 18, 19, 20."""

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
from artifacts.f6_hybrid.test_super_only_aug import extend_with_august, bslice, to_hhmm, LOT_SIZE, FEE

from artifacts.f6_hybrid.test_true_consistent_engine import SuperTracker, PocketHTFFilter
import grid_optimize_f6_atr as grid

def run_session_filtered(
    day: str,
    opt_map: dict,
    all_cal: list,
    cal_idx: dict,
    spot_all: dict,
    start_min: int = 585, # 09:45 AM
    fixed_sl_pts: float = 8.0,
    fixed_tp_pts: float = 16.0,
    be_trigger_pts: float = 6.0,
    max_trades_day: int = 3,
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
        trk[sym] = SuperTracker()
        for i in range(len(g["min"])):
            c = Candle(open=g["open"][i], high=g["high"][i],
                       low=g["low"][i], close=g["close"][i], minute=g["min"][i])
            trk[sym].push(c)

    pmtrig = {}
    slices = {}
    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = SuperTracker()
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
            trig, stype, px, atr_val = t.push(c)
            if trig:
                pmtrig.setdefault(m, []).append((side, sv, sym, c.close, stype, atr_val))

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
    trades_today = 0
    htf = PocketHTFFilter()


    for minute in range(560, 931):
        idx_sp = int(np.searchsorted(spot["min"], minute))
        if idx_sp < len(spot["min"]) and spot["min"][idx_sp] == minute:
            htf.update_1m(minute, spot["open"][idx_sp], spot["high"][idx_sp], spot["low"][idx_sp], spot["close"][idx_sp])

        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c)
                pos["duration_min"] += 1
                if h > pos["peak_px"]:
                    pos["peak_px"] = float(h)

                gain = pos["peak_px"] - pos["entry"]
                if gain >= be_trigger_pts and pos["sl"] < pos["entry"] + 1.0:
                    pos["sl"] = pos["entry"] + 1.0
                    pos["is_be_locked"] = True

                ex, rsn = None, ""
                if l <= pos["sl"] and h >= pos["tp"]:
                    ex, rsn = pos["sl"], "BE_LOCK" if pos.get("is_be_locked") else "SL"
                elif h >= pos["tp"]:
                    ex, rsn = pos["tp"], "TP"
                elif l <= pos["sl"]:
                    ex, rsn = pos["sl"], "BE_LOCK" if pos.get("is_be_locked") else "SL"

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

        # ONLY ENTER AFTER OPENING NOISE (minute >= start_min e.g. 09:45 AM)
        if pos is not None or minute >= 900 or minute < start_min or trades_today >= max_trades_day:
            continue

        trend_15m = htf.get_trend()
        for (sig_side, sig_stk, sig_sym, c_px, stype, atr_val) in pmtrig.get(minute, []):
            if sig_side == "CE" and trend_15m == "BEARISH":
                continue
            if sig_side == "PE" and trend_15m == "BULLISH":
                continue

            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                bar = bslice(ai[1], minute)
                if bar:
                    ep = float(bar[3])
                    pos = {
                        "entry": ep,
                        "sl": round(ep - fixed_sl_pts, 2),
                        "tp": round(ep + fixed_tp_pts, 2),
                        "side": sig_side, "symbol": ai[0], "entry_min": minute,
                        "last_px": ep, "peak_px": ep, "slice": ai[1],
                        "duration_min": 0, "eff_atr": fixed_sl_pts, "is_be_locked": False,
                        "stype": stype,
                    }
                    trades_today += 1
                    break

    return trades


def main():
    spot_all = load_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}

    target_days = ["2026-08-18", "2026-08-19", "2026-08-20"]

    print("=" * 115)
    print("SESSION START TIME 09:45 AM + 15M TREND FILTER + 2:1 POSITIVE R:R")
    print("=" * 115)

    all_trs = []
    for d in target_days:
        trs = run_session_filtered(d, opt_map, all_cal, cal_idx, spot_all, start_min=585, fixed_sl_pts=8.0, fixed_tp_pts=16.0, be_trigger_pts=6.0, max_trades_day=2)
        all_trs.extend(trs)
        print(f"\n>>> DATE: {d} (Trades: {len(trs)})")
        for i, t in enumerate(trs, 1):
            time_str = f"{to_hhmm(t['entry_min'])}->{to_hhmm(t['exit_min'])}"
            print(f" {i:2d} | {time_str:11s} | {t['symbol']:18s} | {t['side']:4s} | {t['entry']:7.2f} | {t['exit']:7.2f} | {t['duration_min']:4d} min | {t['pts']:+6.2f} | Rs {t['rs_net']:+9.2f} | {t['reason']:10s}")

    wins = [t for t in all_trs if t["rs_net"] > 0]
    losses = [t for t in all_trs if t["rs_net"] <= 0]
    net_rs = sum(t["rs_net"] for t in all_trs)
    net_pts = sum(t["pts"] for t in all_trs)
    wr = len(wins) / len(all_trs) * 100 if all_trs else 0.0

    print("\n" + "-" * 110)
    print(f"3-DAY RESULT (09:45+ Session Filter): Total Trades: {len(all_trs)} | Wins: {len(wins)} | Losses: {len(losses)} | WR: {wr:.1f}% | Net Rs: Rs {net_rs:+,.2f} | Net Points: {net_pts:+,.2f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
