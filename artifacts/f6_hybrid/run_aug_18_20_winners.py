"""Trade-by-Trade Execution of Champion Strategies on August 18, 19, 20, 2026.

Simulates:
  1. Champion Low-Drawdown (Trail=0.75x/0.50x, Initial SL=1.50x, TP=3.00x)
  2. Champion Max-Profit (Trail=1.25x/0.50x, Initial SL=1.50x, TP=3.50x)
  3. Dynamic VIX Champion (Power gamma=1.0, Initial SL=3.00x, TP=6.00x)

Outputs complete trade log with entry time, exit time, symbol, points, Rs net, and exit reason.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
import grid_optimize_f6_atr as grid
from backtest_5y_optimized import SYM_RE, latest_spot, load_spot, option_files
from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.indicators.stochastic import IncrementalStochastic

LOT_SIZE = 65
FEE = 40.0
SLIPPAGE_PTS = 0.0
BASE_SESSION_START = 5   # 09:20
BASE_SESSION_END = 345   # 15:00

DESKTOP_OPTS = Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options\2026\8")
DESKTOP_VIX = Path(r"C:\Users\user\Desktop\nifty50 data\INDIA VIX_minute.csv")
AMMU_DATA = Path(r"C:\Websites\ammu\data")


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & EXTENSIONS
# ═══════════════════════════════════════════════════════════════════════════
def extend_with_august(opt_map: dict, spot_all: dict):
    opt_map = dict(opt_map)
    spot_all = dict(spot_all)
    if DESKTOP_OPTS.exists():
        for p in sorted(DESKTOP_OPTS.glob("nifty_options_*.csv")):
            parts = p.stem.split("_")
            day = f"{parts[4]}-{parts[3]}-{parts[2]}"
            opt_map[day] = str(p)
    if AMMU_DATA.exists():
        for d in sorted(AMMU_DATA.glob("2026-08-*")):
            day = d.name
            f = d / f"nifty50_index_1m_{day}.csv"
            if not f.exists():
                continue
            rows = []
            with open(f) as fh:
                header = fh.readline().strip().split(",")
                t_col = header.index("timestamp")
                for line in fh:
                    fields = line.strip().split(",")
                    if len(fields) <= t_col:
                        continue
                    ts = fields[t_col]
                    try:
                        o = float(fields[t_col + 1])
                        h = float(fields[t_col + 2])
                        l = float(fields[t_col + 3])
                        c = float(fields[t_col + 4])
                        dt = pd.to_datetime(ts)
                        rows.append((dt.hour * 60 + dt.minute, o, h, l, c))
                    except Exception:
                        continue
            if not rows:
                continue
            arr = np.array(rows)
            spot_all[day] = {
                "min": arr[:, 0].astype(int),
                "open": arr[:, 1],
                "high": arr[:, 2],
                "low": arr[:, 3],
                "close": arr[:, 4],
            }
    return opt_map, spot_all


class ParamStoch:
    def __init__(self):
        self.s1 = IncrementalStochastic(12, 3)
        self.s2 = IncrementalStochastic(14, 3)
        self.s3 = IncrementalStochastic(40, 4)
        self.s4 = IncrementalStochastic(50, 10)

    def push(self, h, l, c):
        return {
            "s1d": self.s1.push(h, l, c),
            "s2d": self.s2.push(h, l, c),
            "s3d": self.s3.push(h, l, c),
            "s4d": self.s4.push(h, l, c),
        }


class IncrementalATR:
    def __init__(self, period=14):
        self.period = period
        self._buf = []
        self.atr = None
        self.prev_close = None
        self._n = 0

    def update(self, h, l, c):
        tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close)) if self.prev_close else h - l
        self._buf.append(tr)
        self._n += 1
        self.prev_close = c
        if self._n < self.period:
            self.atr = None
        elif self._n == self.period:
            self.atr = sum(self._buf) / self.period
        else:
            self.atr = (self.atr * (self.period - 1) + tr) / self.period
        return self.atr


class SingleTFTracker:
    def __init__(self):
        self.stoch = ParamStoch()
        self.atr = IncrementalATR(14)
        self.prev_s1 = None
        self._fired = False

    def push(self, c: Candle):
        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
        atr_val = self.atr.update(c.high, c.low, c.close)

        is_flag = s4 is not None and s1 is not None and s4 >= 79.5 and s1 <= 20.5
        is_super_setup = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
        s1_turn_up = self.prev_s1 is not None and s1 is not None and s1 > self.prev_s1
        is_super = is_super_setup and s1_turn_up

        cond = is_flag or is_super
        triggered = False
        if cond and not self._fired:
            triggered = True
            self._fired = True
        if not cond:
            self._fired = False

        self.prev_s1 = s1
        stype = "super" if is_super else "flag"
        return triggered, stype, c.close, atr_val


def simulate_day_trades(
    day: str,
    fpath: str,
    fprev: str,
    spot: dict,
    sl_mult: float,
    tp_mult: float,
    trail_trig_atr: float | None,
    trail_dist_atr: float,
    vix_val: float = 15.0,
    gamma: float = 0.0,
):
    gc = grid.cached_day(str(fpath))
    if not gc:
        return []

    prefix = "NIFTY"
    for s in gc.keys():
        if (m := SYM_RE.match(s)):
            prefix = m.group(1)
            break

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
        trk[sym] = SingleTFTracker()
        for i in range(len(g["min"])):
            c = Candle(open=g["open"][i], high=g["high"][i],
                       low=g["low"][i], close=g["close"][i], minute=g["min"][i])
            trk[sym].push(c)

    pmtrig = {}
    slices = {}
    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = SingleTFTracker()
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
                pmtrig.setdefault(m, []).append((side, sv, sym, px, stype, atr_val))

    def bslice(sl, m):
        idx = int(np.searchsorted(sl["min"], m))
        if idx < len(sl["min"]) and sl["min"][idx] == m:
            return sl["open"][idx], sl["high"][idx], sl["low"][idx], sl["close"][idx]
        return None

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
                if trail_trig_atr is not None:
                    gain = pos["peak_px"] - pos["entry"]
                    if gain >= trail_trig_atr * pos["eff_atr"]:
                        trail_sl = pos["peak_px"] - (trail_dist_atr * pos["eff_atr"])
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
                        "reason": rsn, "duration_min": pos["duration_min"],
                    })
                    pos = None

        if minute >= 900 and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            rs_net = round(pts * LOT_SIZE - FEE, 2)
            trades.append({
                "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                "exit": pos["last_px"], "pts": pts, "rs_net": rs_net, "fee": FEE,
                "reason": "EOD", "duration_min": pos["duration_min"],
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
                    vix_scale = (vix_val / 15.0) ** gamma if gamma > 0 else 1.0
                    eff_atr = atr_effective * max(0.6, min(2.0, vix_scale))
                    sl_dist = max(5.0, min(30.0, sl_mult * eff_atr))
                    tp_dist = max(8.0, min(60.0, tp_mult * eff_atr))

                    pos = {
                        "entry": ep,
                        "sl": round(ep - sl_dist, 2),
                        "tp": round(ep + tp_dist, 2),
                        "side": sig_side, "symbol": ai[0], "entry_min": minute,
                        "last_px": ep, "peak_px": ep, "slice": ai[1],
                        "duration_min": 0, "eff_atr": eff_atr, "is_trailing": False,
                    }
                    break

    return trades


def to_hhmm(minute):
    h = minute // 60
    m = minute % 60
    return f"{h:02d}:{m:02d}"


def main():
    spot_all = load_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}

    target_days = ["2026-08-18", "2026-08-19", "2026-08-20"]

    strategies = [
        {
            "name": "CHAMPION LOW-DRAWDOWN (Trail=0.75x/0.50x, Initial SL=1.50x, TP=3.00x)",
            "sl_mult": 1.5, "tp_mult": 3.0, "trail_trig": 0.75, "trail_dist": 0.5, "gamma": 0.0,
        },
        {
            "name": "CHAMPION MAX-PROFIT (Trail=1.25x/0.50x, Initial SL=1.50x, TP=3.50x)",
            "sl_mult": 1.5, "tp_mult": 3.5, "trail_trig": 1.25, "trail_dist": 0.5, "gamma": 0.0,
        },
        {
            "name": "DYNAMIC VIX CHAMPION (Power gamma=1.0, Initial SL=3.00x, TP=6.00x)",
            "sl_mult": 3.0, "tp_mult": 6.0, "trail_trig": None, "trail_dist": 0.5, "gamma": 1.0,
        },
    ]

    for strat in strategies:
        print("=" * 135)
        print(f"STRATEGY: {strat['name']}")
        print("=" * 135)

        all_trs = []
        for day in target_days:
            fpath = opt_map.get(day)
            fprev = opt_map.get(all_cal[cal_idx[day] - 1]) if cal_idx.get(day, 0) > 0 else ""
            trs = simulate_day_trades(
                day, fpath, fprev, spot_all.get(day),
                strat["sl_mult"], strat["tp_mult"],
                strat["trail_trig"], strat["trail_dist"],
                vix_val=15.0, gamma=strat["gamma"]
            )
            all_trs.extend(trs)

            print(f"\n>>> DATE: {day} (Total Trades: {len(trs)})")
            print(f"{'#':2s} | {'Time':11s} | {'Symbol':18s} | {'Side':4s} | {'Entry':7s} | {'Exit':7s} | {'Duration':8s} | {'Pts':7s} | {'Net Rs':12s} | {'Reason':10s}")
            print("-" * 110)
            for i, t in enumerate(trs, 1):
                time_str = f"{to_hhmm(t['entry_min'])}->{to_hhmm(t['exit_min'])}"
                print(f"{i:2d} | {time_str:11s} | {t['symbol']:18s} | {t['side']:4s} | {t['entry']:7.2f} | {t['exit']:7.2f} | {t['duration_min']:4d} min | {t['pts']:+6.2f} | Rs {t['rs_net']:+9.2f} | {t['reason']:10s}")

        wins = [t for t in all_trs if t["rs_net"] > 0]
        losses = [t for t in all_trs if t["rs_net"] <= 0]
        net_pts = sum(t["pts"] for t in all_trs)
        net_rs = sum(t["rs_net"] for t in all_trs)
        wr = len(wins) / len(all_trs) * 100 if all_trs else 0.0
        win_tot = sum(t["rs_net"] for t in wins)
        loss_tot = abs(sum(t["rs_net"] for t in losses))
        pf = win_tot / loss_tot if loss_tot > 0 else (99.0 if win_tot > 0 else 0.0)

        print("\n" + "-" * 110)
        print(f"3-DAY COMBINED SUMMARY (18, 19, 20 August 2026):")
        print(f"  Total Trades: {len(all_trs)} | Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {wr:.1f}%")
        print(f"  Net Points Captured: {net_pts:+,.2f} pts | Total Net Realized Profit: Rs {net_rs:+,.2f} | Profit Factor: {pf:.3f}")
        print("=" * 135 + "\n")


if __name__ == "__main__":
    main()
