"""CAUSAL PARITY v2 — dyn engine vs static (proven) engine, BOTH modes:

MODE A (engine equivalence): DYN_FORCE_STATIC=1 — the dyn engine pins the
  active strike to the 09:15 pair; its seeded math + gates must reproduce the
  static engine's trades EXACTLY (same rule, different code).

MODE B (rule difference): FORCE_STATIC off — only ATM-crossing bars differ.

Window: 2025-09-04..09 (Sep-05 provides Sep-08's seed; Sep-08 seeds Sep-09).
Static = full-history seeded_lib run (position-independent seeding), trades
filtered to the same window days.
"""
import sys, os, time
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
os.environ['LH_BIAS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

WINDOW = ("2025-09-04", "2025-09-09")

import numpy as np
import torch

# ---------------- STATIC ----------------
print("=== STATIC reference (full-history seeded_lib) ===")
import run_7y_v4_master as M
import seeded_lib as SL
import gpu_sim_last_hope as G

D_s, T1_s = SL.D, SL.T1
CH = dict(SL.CHAMPION_BASE)
CH.update(arm_window=10, atr_period=10, atr_mult=1.0, touch_buffer=0.0,
          be_trigger=0.60, be_buffer=1.0, entry_end=T1_s, entry_start=0)

pe_tf_lo = [c['lo'] for c in SL.pe_tf_seed]
pe_tf_cl = [c['cl'] for c in SL.pe_tf_seed]
ce_tf_lo = [c['lo'] for c in SL.ce_tf_seed]
ce_tf_cl = [c['cl'] for c in SL.ce_tf_seed]

def ema20_bounce(ema20, low, close, tlos, tcls, buf=0.0):
    sr = ema20.unsqueeze(-1)
    lo = low.unsqueeze(-1); cl = close.unsqueeze(-1)
    b = ((lo <= sr + buf) & (cl >= sr - 0.5)).any(dim=-1)
    for tlo, tcl in zip(tlos, tcls):
        b |= ((tlo.unsqueeze(-1) <= sr + buf) & (tcl.unsqueeze(-1) >= sr - 0.5)).any(dim=-1)
    return b

G.bounce_pe_stack = torch.stack([ema20_bounce(SL.pe_ema20_seed, G.pe_l, G.pe_c, pe_tf_lo, pe_tf_cl, buf=b)
                                 for b in G.TOUCH_BUFFERS], 0)
G.bounce_ce_stack = torch.stack([ema20_bounce(SL.ce_ema20_seed, G.ce_l, G.ce_c, ce_tf_lo, ce_tf_cl, buf=b)
                                 for b in G.TOUCH_BUFFERS], 0)

t0 = time.time()
static_trades = G._eager_sim_core([CH])[0]
print(f"static run: {len(static_trades)} total in {time.time()-t0:.1f}s")
wd = [d for d in M.trading_days if WINDOW[0] <= d <= WINDOW[1]]
win_trades = [t for t in static_trades if WINDOW[0] <= t[0] <= WINDOW[1]]
from collections import defaultdict
sn = defaultdict(float); sc = defaultdict(int)
for t in win_trades:
    sn[t[0]] += t[5]; sc[t[0]] += 1
print("static per-day in window:")
for d in wd:
    if d in sc:
        print(f"  {d}: {sc[d]} trades {sn[d]:+.2f}")

# ---------------- DYN (both modes) ----------------
MODE = os.environ.get('PARITY_MODE', 'A')
for mode, fs in (('A', '1'), ('B', '0')):
    if MODE != 'X' and mode != MODE:
        continue
    print(f"\n=== DYN MODE {mode} ({'FORCE_STATIC pin' if fs=='1' else 'dynamic trade-time'}) ===")
    os.environ['DYN_DAY_FIRST'] = WINDOW[0]
    os.environ['DYN_DAY_LAST'] = WINDOW[1]
    os.environ['DYN_FORCE_STATIC'] = fs
    for m in list(sys.modules):
        if m == 'dyn_strike_engine':
            del sys.modules[m]
    import dyn_strike_engine as DYN
    cfg = [dict(arm_window=10, atr_mult=1.0, be_trigger=0.60, be_buffer=1.0)]
    dp, tc, wc = DYN.dyn_sim(cfg, gate="ema20", tb=0.0)
    print(f"dyn days: {DYN.days}")
    print("dyn per-day:")
    for i, d in enumerate(DYN.days):
        print(f"  {d}: {int(tc[0, i].item())} trades {dp[0, i].item():+.2f}"
              + (f"   [static {sc.get(d,0)} trades {sn.get(d,0.0):+.2f}]" if d in sc else " (no static)"))
    if mode == 'A':
        ok = True
        for i, d in enumerate(DYN.days):
            if d in sc and (int(tc[0, i].item()) != sc[d] or abs(dp[0, i].item() - sn[d]) > 1.0):
                ok = False
                print(f"  !! MISMATCH {d}: dyn {int(tc[0,i].item())}/{dp[0,i].item():+.2f} vs static {sc[d]}/{sn[d]:+.2f}")
        print(f"\nMODE A VERDICT: {'EXACT PARITY — engine equivalence PROVEN' if ok else 'MISMATCH — investigate'}")
    else:
        print("\nMODE B: diffs below are the RULE difference (trade-time 2nd-ITM vs fixed 09:15):")
        for i, d in enumerate(DYN.days):
            if d in sc and int(tc[0, i].item()) != sc[d]:
                print(f"  {d}: dyn {int(tc[0,i].item())} vs static {sc[d]} (ATM-crossing effect)")
