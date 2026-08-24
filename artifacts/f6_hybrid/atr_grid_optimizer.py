"""ATR SL/TP Factor Grid Optimization Runner (2020-2026).

Finds top 5 ranked ATR factor settings for:
  - Max Net Points / Profit
  - Least Drawdown
  - Best Calmar / Profit Factor

Parameters:
  - S1=(12, 1, 3), S2=(14, 1, 3), S3=(40, 1, 4), S4=(50, 1, 10)
  - Timeframe: 1m
  - Reversal: Embedded S4 >= 25 with Mercy on 1st reversal (and option for without)
  - Cost: Flat Rs 40 per trade, 0 slippage
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
import grid_optimize_f6_atr as grid
from backtest_5y_optimized import SYM_RE, latest_spot, load_spot, option_files
from flattrade_bot.indicators.patterns import BullishPinBarDetector, Candle
from flattrade_bot.indicators.stochastic import IncrementalStochastic

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_LOSS_RS = -2000.0
FEE_PER_TRADE_RS = 40.0

TF_SPECS = {
    "1m": (1, 10),
}

CHAMPION_BASE = {
    "s1_k": 12, "s1_d": 3, "s4_k": 50,
    "atr_period": 14,
    "f6_s4_thresh": 79.5, "f6_s1_thresh": 20.5,
    "consec_loss": 6,
    "mercy": True,
}

FOLDS = [
    {"is_start": "2020", "is_end": "2022", "oos_year": "2023"},
    {"is_start": "2021", "is_end": "2023", "oos_year": "2024"},
    {"is_start": "2022", "is_end": "2024", "oos_year": "2025"},
    {"is_start": "2023", "is_end": "2025", "oos_year": "2026"},
]

DESKTOP_OPTS = Path(r"C:\Users\user\Desktop\nifty50 data\nifty_options\2026\8")
AMMU_DATA = Path(r"C:\Websites\ammu\data")


# ----------------- Indicators -----------------
class ParamStoch:
    def __init__(self, s1_k, s1_d, s4_k):
        self.s1 = IncrementalStochastic(s1_k, s1_d)
        self.s2 = IncrementalStochastic(14, 3)
        self.s3 = IncrementalStochastic(40, 4)
        self.s4 = IncrementalStochastic(s4_k, 10)

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
        self._buf = deque(maxlen=period)
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


class CustomTFTracker:
    def __init__(self, lb, p, trigger_mode="s1_turn_up"):
        self.lb = lb
        self.trigger_mode = trigger_mode
        self.stoch = ParamStoch(p["s1_k"], p["s1_d"], p["s4_k"])
        self.hist = []
        self.setup = False
        self.stype = ""
        self.prev_s1 = None
        self.s4_emb = 0
        self.atr = IncrementalATR(p["atr_period"])
        self.p_f6_s4 = p["f6_s4_thresh"]
        self.p_f6_s1 = p["f6_s1_thresh"]
        self.mercy = p.get("mercy", True)
        self._fired = False
        self.rev_signal_count = 0

    def reset_session_state(self):
        self.hist.clear()
        self.setup = False
        self.stype = ""
        self._fired = False
        self.rev_signal_count = 0

    def push(self, c: Candle):
        self.hist.append(c)
        if len(self.hist) > 60:
            self.hist.pop(0)

        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
        atr_val = self.atr.update(c.high, c.low, c.close)

        if s4 is not None:
            self.s4_emb = self.s4_emb + 1 if s4 <= 20.0 else 0
            if self.s4_emb < 25:
                self.rev_signal_count = 0
        emb = self.s4_emb >= 25

        is_flag = s4 is not None and s1 is not None and s4 >= self.p_f6_s4 and s1 <= self.p_f6_s1
        is_super_setup = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
        s1_turn_up = self.prev_s1 is not None and s1 is not None and s1 > self.prev_s1

        triggered = False
        if self.trigger_mode == "s1_turn_up":
            is_super = is_super_setup and s1_turn_up
            cond = (is_flag or is_super)
            if cond:
                self.stype = "super" if is_super else "flag"
            if cond and not self._fired:
                triggered = True
                self._fired = True
            if not cond:
                self._fired = False
        else:  # "pin_bar" trigger mode
            if is_flag or is_super_setup:
                self.setup = True
                self.stype = "super" if is_super_setup else "flag"
            if self.setup and len(self.hist) >= 2:
                if BullishPinBarDetector.check_vicinity_breakout(self.hist, self.lb):
                    triggered = True
                    self.setup = False

        self.prev_s1 = s1
        is_rev = False
        if triggered and emb and self.stype == "super":
            if self.mercy:
                self.rev_signal_count += 1
                if self.rev_signal_count >= 2:
                    is_rev = True
                else:
                    is_rev = False
            else:
                is_rev = True

        return triggered, is_rev, self.stype, c.close, atr_val


class CustomMTFTracker:
    def __init__(self, p, trigger_mode="s1_turn_up"):
        self.trackers = {
            tf: CustomTFTracker(spec[1], p, trigger_mode=trigger_mode)
            for tf, spec in TF_SPECS.items()
        }
        self.bufs = {tf: [] for tf in TF_SPECS}
        self.reverse_regime_active = False
        self._last_minute = None

    def push_1m(self, c1m: Candle):
        minute = c1m.minute
        if self._last_minute is not None and minute > 0 and minute < self._last_minute:
            self.bufs = {tf: [] for tf in TF_SPECS}
            for tf in TF_SPECS:
                self.trackers[tf].reset_session_state()
            self.reverse_regime_active = False
        self._last_minute = minute

        out = []
        for tf, spec in TF_SPECS.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf]) == spec[0]:
                buf = self.bufs[tf]
                self.bufs[tf] = []
                ctf = Candle(
                    open=buf[0].open,
                    high=max(x.high for x in buf),
                    low=min(x.low for x in buf),
                    close=buf[-1].close,
                    minute=buf[-1].minute,
                )
                trig, is_rev, stype, px, atr_val = self.trackers[tf].push(ctf)
                if trig:
                    out.append((tf, is_rev, stype, px, atr_val))

        return [
            (
                tf,
                is_rev,
                stype,
                px,
                atr_val,
            )
            for (tf, is_rev, stype, px, atr_val) in out
        ]


# ----------------- August Extension -----------------
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
                        dt = datetime.fromisoformat(ts)
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



# ----------------- Worker State -----------------
_WORKER_SPOT = None

def init_worker(spot_data):
    global _WORKER_SPOT
    _WORKER_SPOT = spot_data


def process_day(args):
    (
        day,
        fpath,
        fprev,
        p,
        trigger_mode,
        include_fees,
        sl_mult,
        tp_mult,
    ) = args

    global _WORKER_SPOT
    spot = _WORKER_SPOT.get(day)
    if spot is None or not fpath:
        return []

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
    gp = {}
    if fprev:
        dp = grid.cached_day(str(fprev))
        if dp:
            gp = filtered(dp)

    trk = {}
    for sym, g in gp.items():
        trk[sym] = CustomMTFTracker(p, trigger_mode=trigger_mode)
        for i in range(len(g["min"])):
            c = Candle(open=g["open"][i], high=g["high"][i],
                       low=g["low"][i], close=g["close"][i], minute=g["min"][i])
            trk[sym].push_1m(c)

    pmtrig = {}
    slices = {}
    for sym, g in gu.items():
        if sym not in trk:
            trk[sym] = CustomMTFTracker(p, trigger_mode=trigger_mode)
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
            for (tf, is_rev, stype, px, atr_val) in t.push_1m(c):
                pmtrig.setdefault(m, []).append(
                    (side, sv, sym, px, is_rev, tf, atr_val))

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
        stk = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        sym = f"{prefix}{stk}{side}"
        sl = slices.get(sym)
        return (sym, sl, stk) if sl is not None else None

    trades = []
    pos = None
    dpnl = 0.0
    daily_loss_pts = DAILY_LOSS_RS / LOT_SIZE
    shut = False
    closs = 0

    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c)
                pos["duration_min"] += 1

                # Check shutdown loss
                if dpnl * LOT_SIZE + (c - pos["entry"]) * LOT_SIZE <= DAILY_LOSS_RS:
                    pts = round(c - pos["entry"], 2)
                    fee = FEE_PER_TRADE_RS if include_fees else 0.0
                    rs_net = round(pts * LOT_SIZE - fee, 2)
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": c, "pts": pts, "rs_net": rs_net, "fee": fee,
                        "reason": "SHUTDOWN_LOSS", "duration_min": pos["duration_min"], "tf": pos["tf"],
                    })
                    dpnl += pts
                    pos = None
                    shut = True
                    continue

                ex, rsn = None, ""
                # ATR-Based SL and TP Checks
                if l <= pos["sl"] and h >= pos["tp"]:
                    ex, rsn = pos["sl"], "SL"
                elif h >= pos["tp"]:
                    ex, rsn = pos["tp"], "TP"
                elif l <= pos["sl"]:
                    ex, rsn = pos["sl"], "SL"

                if ex is not None:
                    entry_fill = pos["entry"]
                    exit_fill = ex
                    pts = round(exit_fill - entry_fill, 2)
                    fee = FEE_PER_TRADE_RS if include_fees else 0.0
                    rs_net = round(pts * LOT_SIZE - fee, 2)
                    trades.append({
                        "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": entry_fill,
                        "exit": exit_fill, "pts": pts, "rs_net": rs_net, "fee": fee,
                        "reason": rsn, "duration_min": pos["duration_min"], "tf": pos["tf"],
                    })
                    dpnl += pts
                    closs = closs + 1 if rs_net <= 0 else 0
                    if closs >= p["consec_loss"] or dpnl <= daily_loss_pts:
                        shut = True
                    pos = None

        if minute >= SESSION_END and pos is not None:
            entry_fill = pos["entry"]
            exit_fill = pos["last_px"]
            pts = round(exit_fill - entry_fill, 2)
            fee = FEE_PER_TRADE_RS if include_fees else 0.0
            rs_net = round(pts * LOT_SIZE - fee, 2)
            trades.append({
                "date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                "side": pos["side"], "symbol": pos["symbol"], "entry": entry_fill,
                "exit": exit_fill, "pts": pts, "rs_net": rs_net, "fee": fee,
                "reason": "EOD", "duration_min": pos["duration_min"], "tf": pos["tf"],
            })
            dpnl += pts
            pos = None
            break

        if pos is not None or shut or minute >= SESSION_END:
            continue

        # Check triggers
        for (sig_side, sig_stk, sig_sym, c_px, is_rev, tf, atr_val) in pmtrig.get(minute, []):
            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                if is_rev:
                    as2 = "PE" if sig_side == "CE" else "CE"
                    ai2 = ainfo(as2, minute)
                    if ai2 is None:
                        continue
                    asym, asl, _ = ai2
                else:
                    as2 = sig_side
                    asym = sig_sym
                    asl = ai[1]

                bar = bslice(asl, minute)
                if bar:
                    ep = float(bar[3])
                    # ATR dynamic SL/TP calculation
                    atr_effective = atr_val if atr_val is not None and atr_val > 0 else 8.0
                    sl_dist = max(6.0, min(30.0, sl_mult * atr_effective))
                    tp_dist = max(8.0, min(60.0, tp_mult * atr_effective))

                    pos = {
                        "entry": ep,
                        "sl": round(ep - sl_dist, 2),
                        "tp": round(ep + tp_dist, 2),
                        "side": as2, "symbol": asym, "entry_min": minute,
                        "last_px": ep, "slice": asl, "duration_min": 0, "tf": tf,
                    }
                    break

    return trades


def summarize_trades(trades, day_count=None):
    wins = [t for t in trades if t["rs_net"] > 0]
    losses = [t for t in trades if t["rs_net"] <= 0]
    win_tot = sum(t["rs_net"] for t in wins)
    loss_tot = abs(sum(t["rs_net"] for t in losses))
    net_rs = sum(t["rs_net"] for t in trades)
    net_pts = sum(t["pts"] for t in trades)
    fees = sum(t["fee"] for t in trades)
    wr = len(wins) / len(trades) * 100 if trades else 0.0
    pf = win_tot / loss_tot if loss_tot else (float("inf") if win_tot > 0 else 0.0)

    equity, peak, max_dd = 0.0, 0.0, 0.0
    for t in sorted(trades, key=lambda x: (x["date"], x["entry_min"])):
        equity += t["rs_net"]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    calmar = net_rs / max_dd if max_dd > 0 else 0.0

    return {
        "days": day_count if day_count is not None else len(set(t["date"] for t in trades)),
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr, 2),
        "net_points": round(net_pts, 2),
        "net_rs": round(net_rs, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown_rs": round(max_dd, 2),
        "calmar_ratio": round(calmar, 3),
        "fees_rs": round(fees, 2),
    }


def run_backtest_pool(pool, days, opt_map, params, trigger_mode, include_fees, sl_mult, tp_mult, full_calendar=None):
    all_cal = full_calendar if full_calendar is not None else sorted(opt_map.keys())
    cal_index = {d: i for i, d in enumerate(all_cal)}
    tasks = [
        (
            day,
            opt_map[day],
            opt_map.get(all_cal[cal_index[day] - 1], "") if cal_index.get(day, 0) > 0 else "",
            params,
            trigger_mode,
            include_fees,
            sl_mult,
            tp_mult,
        )
        for day in days
    ]
    all_trades = []
    for day_trs in pool.imap(process_day, tasks):
        all_trades.extend(day_trs)
    return all_trades


# ----------------- Main Driver -----------------
def main():
    parser = argparse.ArgumentParser(description="ATR SL/TP Factor Grid Optimization Runner")
    parser.add_argument("--smoke", action="store_true", help="Run 5-day smoke test")
    parser.add_argument("--full", action="store_true", help="Run full 7-year grid search")
    parser.add_argument("--mode", choices=("s1_turn_up", "pin_bar", "both"), default="both")
    parser.add_argument("--mercy", choices=("with_mercy", "without_mercy", "both"), default="with_mercy")
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    args = parser.parse_args()

    spot_all = load_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    all_days = sorted(set(opt_map.keys()) & set(spot_all.keys()))

    modes = ["s1_turn_up", "pin_bar"] if args.mode == "both" else [args.mode]
    mercy_options = [True, False] if args.mercy == "both" else ([True] if args.mercy == "with_mercy" else [False])

    # ATR Multiplier Grid
    sl_mults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    tp_mults = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    if args.smoke:
        print("=" * 115)
        print(f"MANDATORY SMOKE TEST (5 Days: {all_days[0]} .. {all_days[4]})")
        print("=" * 115)
        smoke_days = all_days[:5]
        with Pool(processes=args.workers, initializer=init_worker, initargs=(spot_all,)) as pool:
            for mercy in mercy_options:
                m_label = "WITH_MERCY" if mercy else "WITHOUT_MERCY"
                cfg = dict(CHAMPION_BASE)
                cfg["mercy"] = mercy
                for m in modes:
                    for sl_m, tp_m in [(1.5, 2.5), (1.25, 3.0)]:
                        t0 = time.time()
                        trs = run_backtest_pool(pool, smoke_days, opt_map, cfg, m, True, sl_m, tp_m, full_calendar=all_days)
                        st = summarize_trades(trs, len(smoke_days))
                        print(f"[{m_label:13s}] Mode [{m:10s}] | SL={sl_m:.2f}x TP={tp_m:.2f}x | Trades: {st['trades']:2d} | WR: {st['win_rate']:5.1f}% | Net Rs: Rs {st['net_rs']:+8.2f} | PF: {st['profit_factor']:5.2f} | Time: {time.time()-t0:.2f}s")
                        status = "PASS" if 5 <= st["trades"] <= 50 and 15.0 <= st["win_rate"] <= 85.0 else "SUSPICIOUS"
                        print(f"Smoke Test Status: {status}")
        print("=" * 115)
        return

    # Full Grid Optimization
    print("=" * 115)
    print(f"7-YEAR (2020-2026) ATR SL/TP GRID OPTIMIZATION ({len(all_days)} Days, {len(sl_mults)*len(tp_mults)} Combinations/mode)")
    print("=" * 115)

    all_results = []
    with Pool(processes=args.workers, initializer=init_worker, initargs=(spot_all,)) as pool:
        for mercy in mercy_options:
            mercy_name = "with_mercy" if mercy else "without_mercy"
            cfg = dict(CHAMPION_BASE)
            cfg["mercy"] = mercy

            for m in modes:
                print(f"\n{'#'*35} OPTIMIZING: {m.upper()} ({mercy_name.upper()}) {'#'*35}")
                grid_results = []
                comb_idx = 0
                total_comb = len(sl_mults) * len(tp_mults)

                for sl_m in sl_mults:
                    for tp_m in tp_mults:
                        comb_idx += 1
                        t0 = time.time()
                        trs = run_backtest_pool(pool, all_days, opt_map, cfg, m, True, sl_m, tp_m, full_calendar=all_days)
                        st = summarize_trades(trs, len(all_days))
                        el = time.time() - t0

                        res_entry = {
                            "mode": m,
                            "mercy": mercy_name,
                            "sl_mult": sl_m,
                            "tp_mult": tp_m,
                            "trades": st["trades"],
                            "win_rate": st["win_rate"],
                            "net_points": st["net_points"],
                            "net_rs": st["net_rs"],
                            "profit_factor": st["profit_factor"],
                            "max_drawdown_rs": st["max_drawdown_rs"],
                            "calmar_ratio": st["calmar_ratio"],
                            "fees_rs": st["fees_rs"],
                        }
                        grid_results.append(res_entry)
                        all_results.append(res_entry)

                        print(f"[{comb_idx:2d}/{total_comb}] SL={sl_m:4.2f}x TP={tp_m:4.2f}x | Trades: {st['trades']:5d} | WR: {st['win_rate']:5.1f}% | Net Pts: {st['net_points']:+9.2f} | Net Rs: Rs {st['net_rs']:+11.2f} | PF: {st['profit_factor']:5.3f} | MaxDD: Rs {st['max_drawdown_rs']:9.2f} | Time: {el:.1f}s")

                # Print Top 5 for this mode
                print(f"\n>>> TOP 5 BY NET PROFIT [{m.upper()} - {mercy_name.upper()}]:")
                top_profit = sorted(grid_results, key=lambda x: x["net_rs"], reverse=True)[:5]
                print(f"{'Rank':4s} | {'SL Mult':7s} | {'TP Mult':7s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
                print("-" * 105)
                for r, item in enumerate(top_profit, 1):
                    print(f"{r:4d} | {item['sl_mult']:7.2f} | {item['tp_mult']:7.2f} | {item['trades']:7d} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['calmar_ratio']:7.3f}")

                print(f"\n>>> TOP 5 BY LEAST DRAWDOWN [{m.upper()} - {mercy_name.upper()}]:")
                top_least_dd = sorted(grid_results, key=lambda x: x["max_drawdown_rs"])[:5]
                print(f"{'Rank':4s} | {'SL Mult':7s} | {'TP Mult':7s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
                print("-" * 105)
                for r, item in enumerate(top_least_dd, 1):
                    print(f"{r:4d} | {item['sl_mult']:7.2f} | {item['tp_mult']:7.2f} | {item['trades']:7d} | {item['win_rate']:7.1f}% | {item['net_points']:+10.2f} | Rs {item['net_rs']:+12.2f} | {item['profit_factor']:6.3f} | Rs {item['max_drawdown_rs']:9.2f} | {item['calmar_ratio']:7.3f}")

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "atr_optimization_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\n[Saved Optimization JSON]: {out_file}")
    print("=" * 115)


if __name__ == "__main__":
    main()
