"""CLI for the ATR16K backtest harness.

    python run.py matrix
    python run.py day 2026-11-06
    python run.py range 2026-11-02 2026-11-06
"""
import os
import re
import sys
from datetime import datetime

import bt

EXP = re.compile(r"^NIFTY(\d{2}[A-Z]{3}\d{2})")


def parse_exp(tok: str):
    try:
        return datetime.strptime(tok, "%d%b%y").date()
    except ValueError:
        return None


def nearest_map(data):
    """For each trading day, the nearest expiry present in the data."""
    exps = set()
    for s in data.df.symbol.unique():
        m = EXP.match(s)
        if m:
            e = parse_exp(m.group(1))
            if e:
                exps.add(e)
    out = {}
    for d in data.days:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        later = sorted(e for e in exps if e >= dt)
        e = later[0] if later else (sorted(exps)[-1] if exps else None)
        out[d] = f"NIFTY{e:%d%b%y}".upper() if e else ""
    return out


def main():
    data = bt.Data()
    if not data.days:
        print("no data in", bt.DATA_DIR); return
    nm = nearest_map(data)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "matrix"

    if cmd == "day":
        d = sys.argv[2]
        days = [x for x in data.days if x == d]
    elif cmd == "range":
        a, b = sys.argv[2], sys.argv[3]
        days = [x for x in data.days if a <= x <= b]
    else:
        days = data.days[-12:]

    all_days = bt.run_days(data, days, nm)
    for d, kept in all_days:
        pts = sum(t["pts"] for t in kept)
        print(f"{d}: {len(kept):2d} tr {pts:+8.2f}")
        if cmd == "day":
            for t in kept:
                print(f"   {t['j']//60:02d}:{t['j']%60:02d} {t['sym']} {t['nm']}={t['entry']} "
                      f"SL {t['sl']} ({t['risk']:.2f}) -> {t['why']} {t['pts']:+.2f}")
    s = bt.summarise(all_days)
    print(f"\n{s['trades']} trades | WR {s['wr']}% | {s['pts']:+.2f} pts | "
          f"Rs {s['rs']:+,} | maxDD {s['dd']:.2f} | Calmar {s['calmar']}")


if __name__ == "__main__":
    main()
