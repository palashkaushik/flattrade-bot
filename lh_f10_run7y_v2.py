"""F10 CERTIFIED 7-YEAR RUN v2 — Full10 v5 (ARM_WIN=1, no EMA200, no dormancy)
+ geometry x daily-cap sweep on the patched base recordings.

Parity status (weekly gate, Sep 2/3/4/7 2026):
  - engine parity: 100% in-window (99/99 aligned; 6 trades post-15:00 excluded
    = documented 15:00 certified end vs 15:15 geoms end)
  - cap=30 parity: 100% in-window
Run: python lh_f10_run7y_v2.py [--caps 0,30] [--geoms G4,SYM20,SYM25,SYM30]
"""
import sys, os, glob, time, json
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
import numpy as np
import lh_f10_replay as R

BASE = r"C:\Users\user\AppData\Local\Temp\opencode\lh_events_f10\base"

def run(geom, cap):
    files = sorted(glob.glob(os.path.join(BASE, "ev_*.npz")))
    tot_n = 0; tot_w = 0; net = 0.0
    yearly = {}
    day_nets = []
    t0 = time.time()
    for f in files:
        z = np.load(f)
        d = str(int(z["date"]))
        trs = R.numba_day(z, geom=geom, cap=cap)
        dnet = 0.0
        for tr in trs:
            if tr["pnl"] is None:
                continue
            yr = d[:4]
            yearly.setdefault(yr, [0, 0, 0.0])
            yearly[yr][0] += 1; yearly[yr][1] += (tr["pnl"] > 0); yearly[yr][2] += tr["pnl"]
            tot_n += 1; tot_w += (tr["pnl"] > 0); net += tr["pnl"]; dnet += tr["pnl"]
        day_nets.append((d, dnet))
    print(f"\n=== {geom} cap={cap or 'off'}: {len(files)} days | {tot_n} trades | "
          f"WR {100*tot_w/max(tot_n,1):.1f}% | NET {net:+,.0f} rs | {time.time()-t0:.0f}s ===")
    for yr in sorted(yearly):
        n, w, p = yearly[yr]
        print(f"  {yr}: {n:>5} trades  {w:>4}W ({100*w/max(n,1):.0f}%)  net {p:>+12,.0f}")
    eq = np.cumsum([x[1] for x in day_nets])
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    print(f"  maxDD {dd.min():+,.0f} | final {eq[-1]:+,.0f} | Calmar {net/max(abs(dd.min()),1):.2f}")
    dn = sorted(day_nets, key=lambda x: x[1])
    print("  worst 3 days:", [(d, round(p)) for d, p in dn[:3]])
    print("  best 3 days:", [(d, round(p)) for d, p in dn[-3:]])
    return net

if __name__ == "__main__":
    geoms = ["G4", "SYM20", "SYM25", "SYM30"]
    caps = [None, 30.0, 25.0, 20.0]
    for a in sys.argv[1:]:
        if a.startswith("--geoms"): geoms = a.split("=")[1].split(",")
        if a.startswith("--caps"): caps = [None if x == "0" else float(x) for x in a.split("=")[1].split(",")]
    summary = []
    for g in geoms:
        for c in caps:
            net = run(g, c)
            summary.append((g, c, net))
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY (net rs, 2020-2026)")
    print("=" * 60)
    for g, c, net in summary:
        print(f"  {g:<7} cap={str(c or 'off'):<5} net {net:>+12,.0f}")
    with open(r"C:\Websites\FLATTRADE BOT\lh_f10_7y_v2_results.json", "w") as f:
        json.dump([dict(geom=g, cap=c, net=net) for g, c, net in summary], f, indent=1)
