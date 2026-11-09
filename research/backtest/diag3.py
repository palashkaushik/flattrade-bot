import bt
from flattrade_bot.strategies.atr16k_engine import pat_any, twin_candles

d = bt.Data()
print("days", d.days)
for day in ("2026-11-04", "2026-11-05", "2026-11-06"):
    nm = "NIFTY09NOV26"
    cs = bt.precompute(d, day, nm)
    print(f"\n{day}: {len(cs)} contract-days, {sum(len(c['ctxs']) for c in cs)} ctxs")
    tot_win = 0
    for c in cs:
        t0 = c["t0"]
        for x in c["ctxs"]:
            lo = max(x["bc"] + 1, t0)
            hi = x["cc"]
            tot_win += max(0, hi - lo + 1)
    print("  window bars examined:", tot_win)
    # count triggers and level passes across all windows
    trig = lvl = 0
    for c in cs:
        t0 = c["t0"]
        for x in c["ctxs"]:
            for gj in range(max(x["bc"] + 1, t0), x["cc"] + 1):
                ti = gj - t0
                if ti < 0 or ti >= len(c["rows"]):
                    continue
                b = c["comb"][gj]
                p = pat_any(c["comb"], gj)
                tw = twin_candles(c["comb"], gj, bt.TWIN_TOL)
                if not p and not tw:
                    continue
                trig += 1
                lv = dict(c["base"])
                for n2, arr in c["per_ti"].items():
                    if arr[ti] is not None:
                        lv[n2] = arr[ti]
                any_touch = not p
                for n2, v in lv.items():
                    if any_touch:
                        ok = b["l"] <= v <= b["h"]
                    else:
                        ok = b["l"] <= v <= b["h"] and b["o"] >= v
                    if ok:
                        lvl += 1
                        break
    print("  trigger bars:", trig, " with level pass:", lvl)
