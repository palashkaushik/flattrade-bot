import sys

import bt
from flattrade_bot.strategies.atr16k_engine import pat_any, twin_candles

d = bt.Data()
cs = bt.precompute(d, "2026-11-06", "NIFTY09NOV26")
c = [x for x in cs if x["sym"] == "NIFTY09NOV2624500CE"][0]
t0 = c["t0"]
for x in c["ctxs"]:
    print("ctx wave", x["w_min"], "win", x["bc"] + 1, "..", x["cc"])
    for gj in range(max(x["bc"] + 1, t0), x["cc"] + 1):
        ti = gj - t0
        if ti >= len(c["rows"]):
            break
        b = c["comb"][gj]
        p = pat_any(c["comb"], gj)
        tw = twin_candles(c["comb"], gj, bt.TWIN_TOL)
        lv = dict(c["base"])
        for nm, arr in c["per_ti"].items():
            if arr[ti] is not None:
                lv[nm] = arr[ti]
        inside = [(nm, round(v, 2), b["o"] >= v) for nm, v in lv.items() if b["l"] <= v <= b["h"]]
        print(f'   {b["t"]//60:02d}:{b["t"]%60:02d} O{b["o"]:.1f} H{b["h"]:.1f} '
              f'L{b["l"]:.1f} C{b["c"]:.1f} pat={p} tw={tw} inside={inside[:3]}')
