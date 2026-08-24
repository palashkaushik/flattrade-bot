"""Aug 12-14 2026 Marni verification: index-bias + fib entry (A/B).

Runs entirely off the cached day snapshots (artifacts/flattrade_day_cache),
which already contain spot (Nifty 50) + option contracts for these dates.
Bias decider = Nifty 50 index 5m HA UT Bot + LinReg.
A/B compare two entry triggers:
  * idxbias       -> 0.786 retracement detected on the OPTION chart (theta-distorted)
  * idxbias_index -> 0.786 tap detected on the INDEX chart (theta-free), option
                     entered at the tap bar's real price (beta-consistent).
2nd ITM strike = ATM-100 (CE) / ATM+100 (PE) for Nifty 50-pt strikes.
"""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import artifacts.f6_hybrid.marni_fib_flattrade_cache as mc
from artifacts.flattrade_day_cache import load_day_cache

mc.GLOBAL_CACHE_DIR = Path("artifacts/flattrade_day_cache")

DATES = ["2026-08-12", "2026-08-13", "2026-08-14"]
TF = ("1m", "2m", "3m", "5m")
TP = [0.290]
SL = [1.155]


def resolve_atm(cache, minute_int):
    for r in cache["spot_rows"]:
        t = datetime.strptime(r["time"], "%d-%m-%Y %H:%M:%S")
        m = t.hour * 60 + t.minute
        if m == minute_int:
            return int(round(r["close"] / 50.0)) * 50
    return None


def strike_from_symbol(symbol):
    import re
    m = re.search(r"(\d+)$", symbol)
    return int(m.group(1)) if m else None


def fmt_min(minute_int):
    return f"{minute_int // 60:02d}:{minute_int % 60:02d}"


def run_mode(mode):
    all_trades = []
    for d in DATES:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        cache = load_day_cache(mc.GLOBAL_CACHE_DIR, dt)
        out = mc.run_day(d, TF, TP, SL, mode)
        for key, trades in out.items():
            for t in trades:
                atm = resolve_atm(cache, t["entry_min"])
                if atm is None:
                    continue
                strike = strike_from_symbol(t["symbol"])
                second_itm = (t["side"] == "CE" and strike == atm - 100) or \
                             (t["side"] == "PE" and strike == atm + 100)
                t["atm"] = atm
                t["strike"] = strike
                t["second_itm"] = second_itm
                t["mode"] = mode
                t["date"] = d
                all_trades.append(t)
    all_trades.sort(key=lambda t: t["entry_min"])
    return all_trades


def print_mode(mode, all_trades):
    trades = [t for t in all_trades if t["mode"] == mode]
    wins = sum(1 for t in trades if t["points"] > 0)
    print("=" * 100)
    label = "index-anchored (theta-free)" if mode == "idxbias_index" else "option-chart (theta-distorted)"
    print(f"MARNI {mode.upper()}  ({label})  Aug 12-14 2026")
    print("=" * 100)
    print(f"Total signals: {len(trades)} | Wins: {wins} | Losses: {len(trades)-wins} "
          f"| Win%: {100*wins/len(trades):.1f}" if trades else "No signals")
    print("-" * 100)
    print(f"{'EntryTime':<18}{'TF':<5}{'Side':<5}{'Strike':<8}{'2ndITM':<7}{'ATM':<7}{'Entry':<9}{'Exit':<9}{'R':<4}{'Pts':<8}{'Net':<9}Reason")
    print("-" * 100)
    for t in trades:
        tf = t.get("timeframe") or "-"
        side = t.get("side") or "-"
        strike = t.get("strike") or 0
        atm = t.get("atm") or 0
        print(f"{fmt_min(t['entry_min']):<18}{tf:<5}{side:<5}{strike:<8}{'YES' if t['second_itm'] else '-':<7}"
              f"{atm:<7}{t['entry']:<9.2f}{t['exit']:<9.2f}{t['reason']:<4}{t['points']:<8.2f}{t['rs_net']:<9.2f}{t['reason']}")
    print("-" * 100)
    second = [t for t in trades if t["second_itm"]]
    sw = sum(1 for t in second if t["points"] > 0)
    if second:
        print(f"2nd-ITM-only signals: {len(second)} | Wins: {sw} | Win%: {100*sw/len(second):.1f}")
    else:
        print("NOTE: no 2nd-ITM strikes present in cached band.")


def main():
    all_by_mode = {m: run_mode(m) for m in ("idxbias", "idxbias_index")}
    for m in ("idxbias", "idxbias_index"):
        print_mode(m, all_by_mode[m])
    a = all_by_mode["idxbias"]
    b = all_by_mode["idxbias_index"]
    wa, wb = sum(1 for t in a if t["points"] > 0), sum(1 for t in b if t["points"] > 0)
    na, nb = sum(t["rs_net"] for t in a), sum(t["rs_net"] for t in b)
    print("=" * 100)
    print("A/B COMPARISON (entry trigger only; identical bias/sl/tp/strike rules)")
    print("=" * 100)
    print(f"  idxbias       : {len(a):>3} sig | {wa:>2} win | {100*wa/len(a):.1f}% | net Rs {na:,.0f}" if a else "  idxbias       : no signals")
    print(f"  idxbias_index : {len(b):>3} sig | {wb:>2} win | {100*wb/len(b):.1f}% | net Rs {nb:,.0f}" if b else "  idxbias_index : no signals")
    print("-" * 100)


if __name__ == "__main__":
    main()
