"""High-Yield Mega-Runner Execution on August 18, 19, 20, 2026.

Parameters:
  - Initial SL: -6.0 option points (Strict Risk Cap)
  - Lock Milestone: When Gain >= +12.0 pts, lock SL at +10.0 pts (Guaranteed Big Win)
  - Chandelier Trail: Trails 4.0 pts behind peak price once locked
  - Hard TP: +20.0 to +30.0 option points
  - Fee: Flat Rs 40.00 / trade | Lot Size: 65
"""

from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_5y_optimized import SYM_RE, latest_spot, option_files
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.test_super_only_aug import extend_with_august, ParamStoch, IncrementalATR, bslice, to_hhmm, LOT_SIZE, FEE
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

def simulate_high_yield_day(
    day: str, opt_map: dict, all_cal: list, cal_idx: dict, spot_all: dict,
    initial_sl_pts: float = 6.0,
    lock_trigger_pts: float = 12.0,
    locked_profit_pts: float = 10.0,
    trail_dist_pts: float = 4.0,
    hard_tp_pts: float = 20.0,
    start_minute: int = 570,  # 09:30 AM
    max_trades_day: int = 3,
    cooldown_min: int = 15,
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

                # High-Yield Lock at +12 pts -> Lock +10 pts
                if gain >= lock_trigger_pts:
                    locked_sl = pos["entry"] + locked_profit_pts
                    if locked_sl > pos["sl"]:
                        pos["sl"] = round(locked_sl, 2)
                        pos["is_locked"] = True

                # Chandelier Trail after locking
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
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": ex, "pts": pts, "rs_net": rs_net, "fee": FEE,
                        "reason": rsn, "duration_min": pos["duration_min"], "stype": pos["stype"],
                    })
                    last_exit_minute = minute
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
            last_exit_minute = minute
            pos = None
            break

        if pos is not None or minute < start_minute or minute >= 900 or trades_today >= max_trades_day:
            continue
        if minute < last_exit_minute + cooldown_min:
            continue

        for (sig_side, sig_stk, sig_sym, c_px, stype) in pmtrig.get(minute, []):
            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
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

    print("=" * 135)
    print("HIGH-YIELD MEGA-RUNNER STRATEGY: AUGUST 18, 19, 20, 2026 EXECUTION")
    print("Settings: Initial SL = -6.0 pts | Lock +10.0 pts at +12.0 pts gain | Trail = 4.0 pts | Target TP = +20.0 pts")
    print("=" * 135)

    all_trs = []
    for day in ["2026-08-18", "2026-08-19", "2026-08-20"]:
        trs = simulate_high_yield_day(day, opt_map, all_cal, cal_idx, spot_all)
        all_trs.extend(trs)
        day_w = [t for t in trs if t["rs_net"] > 0]
        day_l = [t for t in trs if t["rs_net"] <= 0]
        day_net = sum(t["rs_net"] for t in trs)
        day_pts = sum(t["pts"] for t in trs)
        day_wr = len(day_w) / len(trs) * 100 if trs else 0.0

        print(f"\n>>> DATE: {day} ({len(trs)} Trades | {len(day_w)} Wins, {len(day_l)} Losses | Win Rate: {day_wr:.1f}% | Net PnL: Rs {day_net:+,.2f}):")
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
    print(f"  * Total Trades Taken:           {len(all_trs)} trades")
    print(f"  * Total Wins / Losses:          {len(wins)} Wins / {len(losses)} Losses")
    print(f"  * Win Rate:                     {wr:.2f}%")
    print(f"  * Net Points Captured:          {net_pts:+,.2f} pts")
    print(f"  * 3-Day Net Realized Profit:    Rs {net_rs:+,.2f}")
    print("=" * 115)

if __name__ == "__main__":
    main()
