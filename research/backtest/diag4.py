import bt
from flattrade_bot.strategies.atr16k_engine import pat_any, twin_candles

d = bt.Data()
cs = bt.precompute(d, "2026-11-05", "NIFTY09NOV26")
for c in cs:
    out = bt.evaluate([c])
    print(c["sym"], "ctxs", len(c["ctxs"]), "-> evaluate signals", len(out))
    if out:
        for s in out:
            print("   ", s)

# trace the first contract's first window in full
c = cs[0]
t0 = c["t0"]
x = c["ctxs"][0]
print(f"\ntrace {c['sym']} window {x['bc']+1}..{x['cc']} t0 {t0}")
for gj in range(max(x["bc"] + 1, t0), x["cc"] + 1):
    ti = gj - t0
    if ti < 0 or ti >= len(c["rows"]):
        continue
    b = c["comb"][gj]
    p = pat_any(c["comb"], gj)
    tw = twin_candles(c["comb"], gj, bt.TWIN_TOL)
    lv = dict(c["base"])
    for n2, arr in c["per_ti"].items():
        if arr[ti] is not None:
            lv[n2] = arr[ti]
    any_touch = not p
    hits = []
    for n2, v in lv.items():
        if any_touch:
            ok = b["l"] <= v <= b["h"]
        else:
            ok = b["l"] <= v <= b["h"] and b["o"] >= v
        if ok:
            hits.append((n2, round(v, 2)))
    print(f'  {b["t"]//60:02d}:{b["t"]%60:02d} pat={p} tw={tw} hits={hits}')
