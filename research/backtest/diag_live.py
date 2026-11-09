import re
import sys
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
from flattrade_bot.strategies.atr16k_engine import ATR16KContractState, pat_any  # noqa: E402
from replay_live import canon, load_dir, DATA  # noqa: E402

bars = pd.read_csv(r"C:\Websites\FLATTRADE BOT\research\backtest\data\bars_2026-11-09.csv")
bars["symbol"] = bars["symbol"].map(canon)
bars["min"] = bars["time"].map(lambda t: int(t[:2]) * 60 + int(t[3:5]))
priors = load_dir()
sym = "NIFTY09NOV2624800CE"
g = bars[bars.symbol == sym].sort_values("min")
prior = priors[(priors.symbol == sym) & (priors.date < "2026-11-09")].sort_values(["date", "min"]).tail(2000)
print("live bars", len(g), "first", g.time.iloc[0] if len(g) else "-", "prior", len(prior),
      "prior dates", sorted(prior.date.unique())[-3:] if len(prior) else [])

idx = pd.read_csv(f"{DATA}/nov09_index.csv")
idx["min"] = (idx["date"].astype(str).str.slice(11, 13).astype(int) * 60
              + idx["date"].astype(str).str.slice(14, 16).astype(int))
spot = {int(m): float(c) for m, c in zip(idx["min"], idx.close)}
atm = lambda d, m: int(round(spot[max(k for k in spot if k <= m)] / 50.0) * 50)

prev = prior[prior.date == prior.date.max()].sort_values("min")
cs = ATR16KContractState(symbol=sym, token="x", side="CE", strike=24800)
cs.set_day_levels(float(prev.high.max()), float(prev.low.min()), float(prev.close.iloc[-1]))
cs.seed_bars([SimpleNamespace(minute=int(r.min), open=float(r.open), high=float(r.high),
                              low=float(r.low), close=float(r.close), volume=float(r.volume),
                              timestamp=datetime.strptime(r.date, "%Y-%m-%d"))
              for r in prior.itertuples()], session_date="2026-11-09")
cs.atm_fn = atm
hits = []
for r in g.itertuples():
    s = cs.on_bar(SimpleNamespace(minute=int(r.min), open=float(r.open), high=float(r.high),
                                  low=float(r.low), close=float(r.close), volume=float(r.volume),
                                  timestamp=datetime.strptime(r.date, "%Y-%m-%d")))
    if s:
        hits.append(s)
print("signals", [(h["bar"].minute, h["trigger"], h["level"], h["level_px"], h["entry"]) for h in hits])
print("ok_ctx", [(c.get("w_min"), c.get("b_conf"), c.get("c_conf"), c.get("atm")) for c in cs.ok_ctx][-6:])
j = len(cs._bd) - 1
print("last comb", cs._bd[j])
print("EMA200 level", cs._levels.get("EMA200"), "EMA200_5", cs._levels.get("EMA200_5"))
for k in range(max(0, j - 3), j + 1):
    b = cs._bd[k]
    print(f"  {b['t']//60:02d}:{b['t']%60:02d} O{b['o']:.2f} H{b['h']:.2f} L{b['l']:.2f} C{b['c']:.2f} pat={pat_any(cs._bd, k)}")

