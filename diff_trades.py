"""Trade-level diff: dyn (pinned) vs actual static engine on 2025-09-08.
Prints both engines' trade sequences with entry-bar reconstruction for exact diff.
"""
import sys, os
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
os.environ['LH_BIAS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import torch

DAY = "2025-09-08"

import run_7y_v4_master as M
import seeded_lib as SL
import gpu_sim_last_hope as G

pe_tf_lo = [c['lo'] for c in SL.pe_tf_seed]
pe_tf_cl = [c['cl'] for c in SL.pe_tf_seed]
ce_tf_lo = [c['lo'] for c in SL.ce_tf_seed]
ce_tf_cl = [c['cl'] for c in SL.ce_tf_seed]
def ema20_bounce2(ema, low, close, tlos, tcls, buf=0.0):
    b = (low <= ema + buf) & (close >= ema - 0.5)
    for tlo, tcl in zip(tlos, tcls):
        b |= (tlo <= ema + buf) & (tcl >= ema - 0.5)
    return b
G.bounce_pe_stack = torch.stack([ema20_bounce2(SL.pe_ema20_seed, G.pe_l, G.pe_c, pe_tf_lo, pe_tf_cl, buf=b)
                                  for b in G.TOUCH_BUFFERS], 0)
G.bounce_ce_stack = torch.stack([ema20_bounce2(SL.ce_ema20_seed, G.ce_l, G.ce_c, ce_tf_lo, ce_tf_cl, buf=b)
                                  for b in G.TOUCH_BUFFERS], 0)
CH = dict(SL.CHAMPION_BASE)
CH.update(arm_window=10, atr_period=10, atr_mult=1.0, touch_buffer=0.0,
          be_trigger=0.60, be_buffer=1.0, entry_end=SL.T1, entry_start=0)
static_trades = G._eager_sim_core([CH])[0]
st = [t for t in static_trades if t[0] == DAY]
print(f"STATIC {DAY}: {len(st)} trades")
for t in st:
    print(f"  {t[1]:2s} {t[2]:2s} entry {t[3]:>8.2f} exit {t[4]:>8.2f} pnl {t[5]:>+8.2f}")

# dyn pinned
os.environ['DYN_DAY_FIRST'] = "2025-09-05"
os.environ['DYN_DAY_LAST'] = "2025-09-09"
os.environ['DYN_FORCE_STATIC'] = '1'
import dyn_strike_engine as DYN
# instrument: modify dyn_sim? No — rerun and dump per-day counts; for trade list,
# re-run with a hook: capture via monkeypatched _record-like accumulation is not
# available; instead re-run sim and print per-day
cfg = [dict(arm_window=10, atr_mult=1.0, be_trigger=0.60, be_buffer=1.0)]
os.environ['DYN_TRADE_LOG'] = '1'
dp, tc, wc, tlog = DYN.dyn_sim(cfg, gate="ema20", tb=0.0)
i = DYN.days.index(DAY)
print(f"\nDYN pinned {DAY}: {int(tc[0, i].item())} trades, net {dp[0, i].item():+.2f}")
print(f"STATIC same day: {len(st)} trades, net {sum(t[5] for t in st):+.2f}")
print("\nDYN trade list (side 1=PE 2=CE, bars are exit-bar):")
for tr in tlog[0]:
    if tr[0] == i:
        s = 'PE' if tr[1] == 1 else 'CE'
        print(f"  {s:2s} {tr[2]:2s} entry {tr[3]:>8.2f} exit {tr[4]:>8.2f} pnl {tr[5]:>+8.2f} @bar {tr[6]}")
