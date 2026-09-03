"""DUAL-GATE DYNAMIC-STRIKE SWEEP — full history, trade-time 2nd-ITM.

Gates:  'ema20' (§43 champion gate)  vs  'full10' (CPR/Cam/PDHL/EMA20/200/VWAP)
Grid:   arm{5,10,15,20} x mult{1.0,1.25,1.5,2.0} x be{0.4,0.5,0.6,0.7} x tb{0.0,0.5}
        = 64 configs, per gate. Chunked batches (B=16 configs per sim pass).
"""
import sys, os, time, csv, itertools
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
os.environ['LH_BIAS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

SMOKE = '--smoke' in sys.argv
if SMOKE:
    os.environ['DYN_DAY_FIRST'] = '2025-09-01'
    os.environ['DYN_DAY_LAST'] = '2025-09-30'

import importlib
import dyn_strike_engine as DYN

D = DYN.D
print(f"[sweep] dyn engine loaded | D={D} days={DYN.days[0]}..{DYN.days[-1]}")

ARMS = [10, 15] if SMOKE else [5, 10, 15, 20]
MULTS = [1.0, 1.5] if SMOKE else [1.0, 1.25, 1.5, 2.0]
BES = [0.5, 0.6] if SMOKE else [0.4, 0.5, 0.6, 0.7]
TBS = [0.0] if SMOKE else [0.0, 0.5]

def metrics(m):
    return (f"net Rs {m['net']:>11,.0f} | dd {m['max_dd']:>7,.0f} | worst {m['worst_day']:>8,.0f} | "
            f"calmar {m['calmar']:>6.1f} | wr {m['wr']:>5.1f}% | tr {m['trades']:>6}")

rows_out = []
for gate in ("ema20", "full10"):
    print("\n" + "=" * 104)
    print(f"GATE: {gate.upper()}")
    print("=" * 104)
    results = []
    t_g = time.time()
    for tb in TBS:
        # batch ALL (arm, mult, be) combos in ONE dyn_sim call (B=64)
        cfgs = [dict(arm_window=arm, atr_mult=mult, be_trigger=be, be_buffer=1.0)
                for arm in ARMS for mult in MULTS for be in BES]
        t_b = time.time()
        dp, tc, wc = DYN.dyn_sim(cfgs, gate=gate, tb=tb)
        print(f"  [batch] gate={gate} tb={tb} B={len(cfgs)} in {time.time()-t_b:.0f}s")
        for bi, cfg in enumerate(cfgs):
            m = DYN.metrics(dp[bi], tc[bi], wc[bi])
            m.update(gate=gate, arm=cfg['arm_window'], mult=cfg['atr_mult'],
                     be=cfg['be_trigger'], tb=tb)
            results.append(m)
            print(f"  {gate:6s} arm{m['arm']:>2} x{m['mult']:<4} be{m['be']:<3} tb{tb:<3} | "
                  f"net Rs {m['net']:>11,.0f} | dd {m['max_dd']:>7,.0f} | worst {m['worst_day']:>8,.0f} | "
                  f"calmar {m['calmar']:>6.1f} | wr {m['wr']:>5.1f}% | tr {m['trades']:>6}")
    print(f"[{gate}] {len(results)} configs in {time.time()-t_g:.0f}s")
    best_net = max(results, key=lambda x: x['net'])
    best_cal = max((r for r in results if r['trades'] >= (50 if SMOKE else 3000)),
                   key=lambda x: x['calmar'])
    print(f"\n  BEST-NET  {gate:6s}: arm{best_net['arm']:>2} x{best_net['mult']:<4} be{best_net['be']:<3} tb{best_net['tb']:<3} | {metrics(best_net)}")
    print(f"  BEST-CAL {gate:6s}: arm{best_cal['arm']:>2} x{best_cal['mult']:<4} be{best_cal['be']:<3} tb{best_cal['tb']:<3} | {metrics(best_cal)}")
    rows_out.extend(results)

with open('dyn_sweep_results.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['gate', 'arm', 'mult', 'be', 'tb', 'trades', 'wr', 'net', 'max_dd', 'worst_day', 'calmar'])
    for m in rows_out:
        w.writerow([m['gate'], m['arm'], m['mult'], m['be'], m['tb'],
                    m['trades'], round(m['wr'], 2), round(m['net'], 2),
                    round(m['max_dd'], 2), round(m['worst_day'], 2), round(m['calmar'], 3)])
print(f"\n[saved] dyn_sweep_results.csv | total {len(rows_out)} configs")
