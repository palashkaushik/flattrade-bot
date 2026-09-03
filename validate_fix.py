"""Validate the be_done-reset fix: patched static engine vs the hand-replay
(ground truth semantics) vs dyn engine on 2025-09-08 (pinned strikes).

Expected after fix: static 20 trades matching replay/dyn trade-for-trade.
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
print(f"PATCHED STATIC {DAY}: {len(st)} trades, net {sum(t[5] for t in st):+.2f}")
for t in st:
    print(f"  {t[1]:2s} {t[2]:2s} entry {t[3]:>8.2f} exit {t[4]:>8.2f} pnl {t[5]:>+8.2f}")

# full-history net for the §43 champion with the fix (vs pre-fix 2,832,706)
allt = [t for t in static_trades]
net = sum(t[5] for t in allt)
wins = sum(1 for t in allt if t[5] > 0)
daily = {}
for t in allt:
    daily[t[0]] = daily.get(t[0], 0.0) + t[5]
cum = peak = 0.0; dd = 0.0
for d in sorted(daily):
    cum += daily[d]; peak = max(peak, cum); dd = max(dd, peak - cum)
print(f"\n§43 champion (PATCHED, 7y): trades {len(allt)} | WR {100*wins/len(allt):.1f}% | "
      f"net Rs {net:,.0f} | maxDD Rs {dd:,.0f} | worst-day Rs {min(daily.values()):,.0f}")
print("PRE-FIX reference: trades 19,701 | WR 78.5% | net Rs 2,832,706 | maxDD Rs 1,963")
