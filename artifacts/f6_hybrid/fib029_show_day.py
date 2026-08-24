"""Run the Fib-0.29 Stoch Wave strategy on a single day from the ammu data dir.

Usage:
  python fib029_show_day.py --day 2026-08-19 [--variant 5m|60stoc] [--min-amp 20] [--fib 0.29]

Data layout (per day): data/<day>/nifty_options_1m_<day>.csv + nifty50_index_1m_<day>.csv
  index:  timestamp,open,high,low,close,volume[,extra]
  opts:   option_type,strike,timestamp,open,high,low,close,volume,open_interest
Timestamps are IST (+05:30).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.fib029_stoch_backtest import (
    FibWaveTracker, CE_SL, CE_TP, PE_SL, PE_TP, SESSION_START, SESSION_END,
    CONSEC_LOSS_LIMIT, LOT_SIZE, FEES_PER_TRADE,
)
from artifacts.f6_hybrid.pocket_money_backtest import build_index_filter, filter_allows
from flattrade_bot.strategies.pocket_money import OptionTracker


def load_day(day: str, data_dir: Path):
    opt = pd.read_csv(data_dir / f"nifty_options_1m_{day}.csv", engine="c")
    idx = pd.read_csv(data_dir / f"nifty50_index_1m_{day}.csv", engine="c",
                      names=["timestamp", "open", "high", "low", "close", "volume", "extra"],
                      skiprows=1)
    opt["ts"] = pd.to_datetime(opt["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    idx["ts"] = pd.to_datetime(idx["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    idx["min"] = idx["ts"].dt.hour * 60 + idx["ts"].dt.minute
    opt["min"] = opt["ts"].dt.hour * 60 + opt["ts"].dt.minute
    opt = opt.sort_values("ts").reset_index(drop=True)
    idx = idx.sort_values("ts").reset_index(drop=True)
    return idx, opt


def run_day(idx, opt, variant, fib, min_amp, verb=True):
    spot = {
        "min": idx["min"].to_numpy(),
        "open": idx["open"].to_numpy(dtype=float),
        "high": idx["high"].to_numpy(dtype=float),
        "low": idx["low"].to_numpy(dtype=float),
        "close": idx["close"].to_numpy(dtype=float),
    }
    import numpy as np

    def latest_spot(minute):
        ii = np.searchsorted(spot["min"], minute, side="right") - 1
        return None if ii < 0 else float(spot["close"][ii])

    def bslice(sl, minute):
        idx2 = np.searchsorted(sl["min"], minute)
        if idx2 < len(sl["min"]) and sl["min"][idx2] == minute:
            return float(sl["open"][idx2]), float(sl["high"][idx2]), float(sl["low"][idx2]), float(sl["close"][idx2])
        return None

    ifilter = build_index_filter(spot, day="") if variant == "5m" else None

    trk, waves = {}, {}
    for (typ, strike), g in opt.groupby(["option_type", "strike"]):
        key = f"{typ}{int(strike)}"
        trk[key] = OptionTracker()
        waves[key] = FibWaveTracker(fib=fib, min_amp=min_amp)

    slices = {}
    for (typ, strike), g in opt.groupby(["option_type", "strike"]):
        key = f"{typ}{int(strike)}"
        slices[key] = {
            "min": g["min"].to_numpy(),
            "open": g["open"].to_numpy(), "high": g["high"].to_numpy(),
            "low": g["low"].to_numpy(), "close": g["close"].to_numpy(),
        }

    trig = {}
    for key, t in trk.items():
        sl = slices[key]
        side = "CE" if key.startswith("CE") else "PE"
        w = waves[key]
        for i in range(len(sl["min"])):
            s1, s2, s3, s4, prev = t.push(sl["high"][i], sl["low"][i], sl["close"][i])
            if s1 is None:
                continue
            if w.push(s1):
                trig.setdefault(int(sl["min"][i]), []).append((side, key, s4))

    trades, pos, closs, shut = [], None, 0, False

    def close_pos(minute, ex, rsn):
        nonlocal pos, closs, shut
        pts = round(ex - pos["entry"], 2)
        trades.append({"entry_min": pos["entry_min"], "exit_min": minute, "side": pos["side"],
                       "symbol": pos["symbol"], "entry": pos["entry"], "exit": ex, "pts": pts,
                       "rs": round(pts * LOT_SIZE), "reason": rsn, "signal": "fib029",
                       "duration_min": minute - pos["entry_min"], "level": pos["level"]})
        closs = closs + 1 if pts <= 0 else 0
        if closs >= CONSEC_LOSS_LIMIT:
            shut = True
        pos = None

    for minute in range(SESSION_START, 930):
        if pos is not None:
            held = bslice(slices[pos["symbol"]], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = c
                if h >= pos["tgt"] and l <= pos["sl"]:
                    close_pos(minute, pos["sl"], "SL")
                elif h >= pos["tgt"]:
                    close_pos(minute, pos["tgt"], "TP")
                elif l <= pos["sl"]:
                    close_pos(minute, pos["sl"], "SL")
        if minute >= SESSION_END and pos is not None:
            close_pos(minute, pos["last_px"], "EOD")
            break
        if pos is not None or shut or minute >= SESSION_END:
            continue
        for sig_side, key, s4 in trig.get(minute, []):
            if variant == "5m":
                allowed = filter_allows(ifilter, minute)
            else:
                if s4 is None:
                    continue
                allowed = "CE" if s4 < 70.0 else ("PE" if s4 > 70.0 else None)
            if allowed is None or sig_side != allowed:
                continue
            spx = latest_spot(minute)
            if spx is None:
                continue
            atm = int(round(spx / 50) * 50)
            stk = atm + (-100 if sig_side == "CE" else 100)
            sym = f"{sig_side}{stk}"
            if sym not in slices:
                continue
            bar = bslice(slices[sym], minute)
            if bar is None:
                continue
            ep = bar[3]
            sl_use, tp_use = (CE_SL, CE_TP) if sig_side == "CE" else (PE_SL, PE_TP)
            pos = {"side": sig_side, "symbol": sym, "entry": ep, "sl": ep - sl_use,
                   "tgt": ep + tp_use, "entry_min": minute, "last_px": ep, "level": waves[sym].level}

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--data-dir", default=r"C:\Websites\ammu\data\2026-08-19")
    ap.add_argument("--variant", choices=["5m", "60stoc"], default="60stoc")
    ap.add_argument("--min-amp", type=float, default=20.0)
    ap.add_argument("--fib", type=float, default=0.29)
    args = ap.parse_args()

    idx, opt = load_day(args.day, Path(args.data_dir))
    if idx.empty or opt.empty:
        print(f"NO DATA for {args.day}")
        return

    trades = run_day(idx, opt, args.variant, args.fib, args.min_amp)
    print(f"\n=== FIB-0.29 ({args.variant} authority) — {args.day} ===")
    if not trades:
        print("NO TRADES")
        return
    tot = 0.0
    for t in trades:
        e = f"{t['entry_min']//60:02d}:{t['entry_min']%60:02d}"
        x = f"{t['exit_min']//60:02d}:{t['exit_min']%60:02d}"
        tot += t["pts"]
        print(f"  {e}->{x}  {t['side']} {t['symbol']}  entry {t['entry']:.2f}  exit {t['exit']:.2f}  "
              f"{t['pts']:+.2f} pts  ({t['rs']:+,d} rs)  {t['reason']}  lev={t['level']:.1f}")
    wins = [t for t in trades if t["pts"] > 0]
    print(f"  total {len(trades)} trades, WR {len(wins)/len(trades)*100:.0f}%, "
          f"net {tot:+.2f} pts ({round(tot*LOT_SIZE):+,d} rs), "
          f"after fees {round(tot*LOT_SIZE - FEES_PER_TRADE*len(trades)):+,d} rs")


if __name__ == "__main__":
    main()