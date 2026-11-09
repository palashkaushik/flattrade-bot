import bt
from flattrade_bot.strategies.atr16k_engine import pat_any, twin_candles

d = bt.Data()
cs = bt.precompute(d, "2026-11-09", "NIFTY09NOV26")
print("contract-days:", len(cs))
for c in cs:
    if c["sym"] in ("NIFTY09NOV2624800CE", "NIFTY09NOV2625050PE"):
        print(f"\n{c['sym']} ctxs {[(x['w_min'], x['bc'], x['cc'], x['atm']) for x in c['ctxs']]}")
        print("  PDVWAP", c["base"].get("PDVWAP"), "PDH", c["base"].get("PDH"), "PDL", c["base"].get("PDL"))
        t0 = c["t0"]
        for gj in range(t0, len(c["comb"])):
            b = c["comb"][gj]
            if 605 <= b["t"] <= 615:
                ti = gj - t0
                p = pat_any(c["comb"], gj)
                tw = twin_candles(c["comb"], gj, bt.TWIN_TOL)
                lv = dict(c["base"])
                for nm, arr in c["per_ti"].items():
                    if arr[ti] is not None:
                        lv[nm] = arr[ti]
                hits = [(nm, round(v, 2), b["o"] >= v) for nm, v in lv.items()
                        if b["l"] <= v <= b["h"]]
                print(f'   {b["t"]//60:02d}:{b["t"]%60:02d} O{b["o"]:.1f} H{b["h"]:.1f} '
                      f'L{b["l"]:.1f} C{b["c"]:.1f} pat={p} tw={tw} inside={hits}')
