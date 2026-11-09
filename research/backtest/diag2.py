import bt
from flattrade_bot.strategies.atr16k_engine import pat_any, twin_candles

d = bt.Data()
cs = bt.precompute(d, "2026-11-06", "NIFTY09NOV26")
c = [x for x in cs if x["sym"] == "NIFTY09NOV2624800PE"][0]
print("P24800 t0", c["t0"], "rows", len(c["rows"]))
for x in c["ctxs"]:
    print("  ctx wave", x["w_min"], "atm", x["atm"], "win", x["bc"] + 1, "..", x["cc"])
t0 = c["t0"]
for gj in range(t0, len(c["comb"])):
    b = c["comb"][gj]
    if 780 <= b["t"] <= 790:
        ti = gj - t0
        p = pat_any(c["comb"], gj)
        tw = twin_candles(c["comb"], gj, bt.TWIN_TOL)
        lv = dict(c["base"])
        for nm, arr in c["per_ti"].items():
            if arr[ti] is not None:
                lv[nm] = arr[ti]
        inside = [(nm, round(v, 2), b["o"] >= v) for nm, v in lv.items() if b["l"] <= v <= b["h"]]
        print(f'  {b["t"]//60:02d}:{b["t"]%60:02d} O{b["o"]:.1f} H{b["h"]:.1f} '
              f'L{b["l"]:.1f} C{b["c"]:.1f} pat={p} tw={tw} inside={inside}')
