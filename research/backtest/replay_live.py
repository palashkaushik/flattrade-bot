"""Replay a recorded live-bar session through the engine (exact live tape).

    python replay_live.py bars_2026-11-09.csv nov09

Priors come from the option CSVs in data/ (strike-first symbols); the live bar
file uses the broker's side-first tsym, so both are normalised to strike-first.
"""
import os
import re
import sys
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from flattrade_bot.strategies.atr16k_engine import ATR16KContractState  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def canon(sym: str) -> str:
    """NIFTY09NOV26C24800 -> NIFTY09NOV2624800CE (CSV convention)."""
    m = re.match(r"^NIFTY(\d{2}[A-Z]{3}\d{2})([CP])(\d+)$", sym)
    if m:
        return f"NIFTY{m.group(1)}{m.group(3)}{'CE' if m.group(2) == 'C' else 'PE'}"
    return sym


def load_dir(day_from=None):
    frames = []
    for f in sorted(os.listdir(DATA)):
        if not f.endswith(".csv") or "index" in f or f.startswith("bars_"):
            continue
        df = pd.read_csv(os.path.join(DATA, f))
        df["symbol"] = df["symbol"].map(canon)
        df["min"] = df["time"].map(lambda t: int(t[:2]) * 60 + int(t[3:5]))
        frames.append(df.drop_duplicates(["symbol", "date", "time"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(bars_path, tag):
    day = tag
    bars = pd.read_csv(bars_path)
    bars["symbol"] = bars["symbol"].map(canon)
    bars["min"] = bars["time"].map(lambda t: int(t[:2]) * 60 + int(t[3:5]))
    priors = load_dir()

    idx = pd.read_csv(os.path.join(DATA, f"{tag}_index.csv"))
    idx["min"] = (idx["date"].astype(str).str.slice(11, 13).astype(int) * 60
                  + idx["date"].astype(str).str.slice(14, 16).astype(int))
    spot = {int(m): float(c) for m, c in zip(idx["min"], idx.close)}

    def atm_at(d, minute):
        ks = [k for k in spot if k <= minute]
        return int(round(spot[max(ks)] / 50.0) * 50) if ks else None

    day_str = bars["date"].iloc[0]
    total = 0
    for sym in sorted(bars.symbol.unique()):
        g = bars[bars.symbol == sym].sort_values("min")
        prior = priors[(priors.symbol == sym) & (priors.date < day_str)].sort_values(["date", "min"]).tail(2000)
        if not len(prior) or len(g) < 3:
            continue
        prev = prior[prior.date == prior.date.max()].sort_values("min")
        cs = ATR16KContractState(symbol=sym, token="x",
                                 side="CE" if sym.endswith("CE") else "PE",
                                 strike=int(re.search(r"(\d+)(CE|PE)$", sym).group(1)))
        cs.set_day_levels(float(prev.high.max()), float(prev.low.min()),
                          float(prev.close.iloc[-1]))
        cs.seed_bars([SimpleNamespace(minute=int(r.min), open=float(r.open), high=float(r.high),
                                      low=float(r.low), close=float(r.close), volume=float(r.volume),
                                      timestamp=datetime.strptime(r.date, "%Y-%m-%d"))
                      for r in prior.itertuples()], session_date=day_str)
        cs.atm_fn = atm_at
        hits = []
        for r in g.itertuples():
            s = cs.on_bar(SimpleNamespace(minute=int(r.min), open=float(r.open), high=float(r.high),
                                          low=float(r.low), close=float(r.close), volume=float(r.volume),
                                          timestamp=datetime.strptime(r.date, "%Y-%m-%d")))
            if s:
                hits.append(s)
        if hits:
            print(f"\n{sym}")
            for s in hits:
                total += 1
                print(f"  {s['bar'].minute // 60:02d}:{s['bar'].minute % 60:02d} "
                      f"{s['trigger']} {s['level']}={s['level_px']} entry {s['entry']} "
                      f"SL {s['sl']} risk {s['sl_points']} atm_w={s['atm_w']}")
    print(f"\n{total} signals on the recorded live bars")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
