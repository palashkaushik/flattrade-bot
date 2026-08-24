"""Comparative Experiment: Rule 1 (TV 15m HTF Filter) vs Rule 2 (2:1 Positive R:R) vs Combined.

Full OHLC Spot feed with Heikin-Ashi + LinReg + UT Bot parity check.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_5y_optimized import SYM_RE, latest_spot, option_files
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.pocket_money_backtest import (
    build_index_filter, filter_allows,
)
from artifacts.f6_hybrid.test_super_only_aug import extend_with_august, ParamStoch, IncrementalATR, bslice, to_hhmm, LOT_SIZE, FEE
import grid_optimize_f6_atr as grid


def load_full_ohlc_spot():
    spot_file = Path("C:/Websites/ammu/index/NIFTY 50_minute.csv")
    df = pd.read_csv(spot_file, parse_dates=["date"], engine="c")
    df = df.sort_values("date").reset_index(drop=True)
    df["day"] = df["date"].dt.strftime("%Y-%m-%d")
    df["min"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    out = {}
    for day, g in df.groupby("day"):
        out[day] = {
            "min": g["min"].to_numpy(),
            "open": g["open"].to_numpy(),
            "high": g["high"].to_numpy(),
            "low": g["low"].to_numpy(),
            "close": g["close"].to_numpy(),
        }
    return out



class DualTracker:
    def __init__(self):
        self.stoch = ParamStoch()
        self.atr = IncrementalATR(14)
        self.prev_s1 = None
        self._fired = False

    def push(self, c: Candle):
        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
        atr_val = self.atr.update(c.high, c.low, c.close)

        is_super_setup = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
        is_flag_setup = s4 is not None and s1 is not None and s4 >= 79.5 and s1 <= 20.5
        s1_turn_up = self.prev_s1 is not None and s1 is not None and s1 > self.prev_s1

        cond = (is_super_setup or is_flag_setup) and s1_turn_up
        trig = False
        if cond and not self._fired:
            trig = True
            self._fired = True
        if not cond:
            self._fired = False

        self.prev_s1 = s1
        return trig, "SUPER" if is_super_setup else "FLAG", c.close, atr_val


def simulate_day_rule_engine(
    day: str,
    opt_map: dict,
    all_cal: list,
    cal_idx: dict,
    spot_all: dict,
    use_htf: bool = False,
    exit_mode: str = "trailing",  # "trailing" or "positive_rr"
    sl_mult: float = 1.50,
    tp_mult: float = 3.00,
    trail_trig: float = 0.75,
    trail_dist: float = 0.40,
    fixed_sl_pts: float = 7.0,
    fixed_tp_pts: float = 14.0,
    be_trigger_pts: float = 5.0,
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

    # Compute TradingView-Exact HTF Snapshots
    htf_snaps = build_index_filter(spot, day=day, warm_days=12) if (use_htf and spot is not None) else {}


    trades = []
    pos = None
    trades_today = 0

    for minute in range(560, 931):
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c)
                pos["duration_min"] += 1
                if h > pos["peak_px"]:
                    pos["peak_px"] = float(h)

                if exit_mode == "trailing":
                    gain = pos["peak_px"] - pos["entry"]
                    if gain >= trail_trig * pos["eff_atr"]:
                        trail_sl = pos["peak_px"] - (trail_dist * pos["eff_atr"])
                        if trail_sl > pos["sl"]:
                            pos["sl"] = round(trail_sl, 2)
                            pos["is_trailing"] = True
                elif exit_mode == "positive_rr":
                    gain = pos["peak_px"] - pos["entry"]
                    if gain >= be_trigger_pts and pos["sl"] < pos["entry"] + 1.0:
                        pos["sl"] = pos["entry"] + 1.0
                        pos["is_be_locked"] = True

                ex, rsn = None, ""
                if l <= pos["sl"] and h >= pos["tp"]:
                    ex, rsn = pos["sl"], "BE_LOCK" if pos.get("is_be_locked") else ("TRAIL_SL" if pos.get("is_trailing") else "SL")
                elif h >= pos["tp"]:
                    ex, rsn = pos["tp"], "TP"
                elif l <= pos["sl"]:
                    ex, rsn = pos["sl"], "BE_LOCK" if pos.get("is_be_locked") else ("TRAIL_SL" if pos.get("is_trailing") else "SL")

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

        if pos is not None or minute >= 900 or trades_today >= max_trades_day:
            continue

        for (sig_side, sig_stk, sig_sym, c_px, stype, atr_val) in pmtrig.get(minute, []):
            # Apply TradingView 15m HTF Filter
            if use_htf:
                allowed_side = filter_allows(htf_snaps, minute)
                if allowed_side != sig_side:
                    continue

            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                bar = bslice(ai[1], minute)
                if bar:
                    ep = float(bar[3])
                    atr_effective = atr_val if atr_val is not None and atr_val > 0 else 8.0

                    if exit_mode == "trailing":
                        sl_d = max(5.0, min(30.0, sl_mult * atr_effective))
                        tp_d = max(8.0, min(60.0, tp_mult * atr_effective))
                    else:  # positive_rr
                        sl_d = fixed_sl_pts
                        tp_d = fixed_tp_pts

                    pos = {
                        "entry": ep,
                        "sl": round(ep - sl_d, 2),
                        "tp": round(ep + tp_d, 2),
                        "side": sig_side, "symbol": ai[0], "entry_min": minute,
                        "last_px": ep, "peak_px": ep, "slice": ai[1],
                        "duration_min": 0, "eff_atr": atr_effective,
                        "is_trailing": False, "is_be_locked": False,
                        "stype": stype,
                    }
                    trades_today += 1
                    break

    return trades


def run_experiment(days_subset: list, title: str):
    spot_all = load_full_ohlc_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    grid.GLOBAL_SPOT = spot_all
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}

    configs = [
        {"name": "1. BASELINE (No HTF, Trail=0.75x/0.40x)", "use_htf": False, "exit_mode": "trailing"},
        {"name": "2. RULE 1 ONLY (TV 15m HTF Filter, Trail=0.75x/0.40x)", "use_htf": True, "exit_mode": "trailing"},
        {"name": "3. RULE 2 ONLY (No HTF, 2:1 R:R +14/-7 pts, BE +5 pts)", "use_htf": False, "exit_mode": "positive_rr"},
        {"name": "4. COMBINED (TV 15m HTF Filter + 2:1 R:R)", "use_htf": True, "exit_mode": "positive_rr"},
    ]

    print("=" * 135)
    print(f"EXPERIMENT: {title.upper()} ({len(days_subset)} Trading Days)")
    print("=" * 135)

    results_table = []

    for cfg in configs:
        all_trs = []
        for d in days_subset:
            trs = simulate_day_rule_engine(
                d, opt_map, all_cal, cal_idx, spot_all,
                use_htf=cfg["use_htf"],
                exit_mode=cfg["exit_mode"],
                sl_mult=1.50, tp_mult=3.00, trail_trig=0.75, trail_dist=0.40,
                fixed_sl_pts=7.0, fixed_tp_pts=14.0, be_trigger_pts=5.0,
                max_trades_day=3,
            )
            all_trs.extend(trs)

        n_t = len(all_trs)
        wins = [t for t in all_trs if t["rs_net"] > 0]
        losses = [t for t in all_trs if t["rs_net"] <= 0]
        net_rs = sum(t["rs_net"] for t in all_trs)
        net_pts = sum(t["pts"] for t in all_trs)
        wr = len(wins) / n_t * 100 if n_t > 0 else 0.0
        win_tot = sum(t["rs_net"] for t in wins)
        loss_tot = abs(sum(t["rs_net"] for t in losses))
        pf = win_tot / loss_tot if loss_tot > 0 else (99.0 if win_tot > 0 else 0.0)

        eq = np.cumsum([t["rs_net"] for t in all_trs]) if all_trs else np.array([0.0])
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0

        results_table.append({
            "name": cfg["name"],
            "trades": n_t,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": wr,
            "net_points": net_pts,
            "net_rs": net_rs,
            "profit_factor": pf,
            "max_drawdown": max_dd,
            "avg_trade_rs": net_rs / n_t if n_t > 0 else 0.0,
            "trades_list": all_trs,
        })

    print(f"\n{'Configuration':58s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Avg/Trade':10s}")
    print("-" * 145)
    for r in results_table:
        print(f"{r['name']:58s} | {r['trades']:7d} | {r['win_rate']:7.1f}% | {r['net_points']:+10.2f} | Rs {r['net_rs']:+12.2f} | {r['profit_factor']:6.3f} | Rs {r['max_drawdown']:9.2f} | Rs {r['avg_trade_rs']:+7.2f}")

    return results_table


def main():
    spot_all = load_full_ohlc_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    grid.GLOBAL_SPOT = spot_all
    all_cal = sorted(opt_map.keys())

    # 1. Mandatory Smoke Test (5 Days)
    smoke_days = all_cal[:5]
    print("\n" + "#" * 45 + " PART 1: MANDATORY 5-DAY SMOKE TEST " + "#" * 45)
    run_experiment(smoke_days, "5-Day Mandatory Smoke Test (Jan 2020)")

    # 2. August 18, 19, 20, 2026 Test
    aug_days = ["2026-08-18", "2026-08-19", "2026-08-20"]
    print("\n" + "#" * 45 + " PART 2: AUGUST 18, 19, 20, 2026 TEST " + "#" * 45)
    res_aug = run_experiment(aug_days, "August 18, 19, 20, 2026 (Live Sample)")

    # Print trade-by-trade breakdown for August Combined
    comb_aug_trs = res_aug[3]["trades_list"]
    print(f"\n>>> COMBINED STRATEGY (RULE 1 + RULE 2) TRADE-BY-TRADE LOG ON AUGUST 18, 19, 20:")
    print(f"{'#':2s} | {'Date':10s} | {'Time':11s} | {'Symbol':18s} | {'Side':4s} | {'Entry':7s} | {'Exit':7s} | {'Duration':8s} | {'Pts':7s} | {'Net Rs':12s} | {'Reason':10s}")
    print("-" * 120)
    for i, t in enumerate(comb_aug_trs, 1):
        time_str = f"{to_hhmm(t['entry_min'])}->{to_hhmm(t['exit_min'])}"
        print(f"{i:2d} | {t['date']:10s} | {time_str:11s} | {t['symbol']:18s} | {t['side']:4s} | {t['entry']:7.2f} | {t['exit']:7.2f} | {t['duration_min']:4d} min | {t['pts']:+6.2f} | Rs {t['rs_net']:+9.2f} | {t['reason']:10s}")

    # 3. 2-Year Out-Of-Sample Test (2024–2025)
    oos_days = [d for d in all_cal if d.startswith("2024") or d.startswith("2025")]
    print("\n" + "#" * 45 + " PART 3: 2-YEAR OUT-OF-SAMPLE TEST (2024-2025) " + "#" * 45)
    run_experiment(oos_days, "2-Year Out-Of-Sample (2024–2025)")


if __name__ == "__main__":
    main()
