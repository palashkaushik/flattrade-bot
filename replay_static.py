"""Instrument the STATIC engine's BE for the divergent trade by replicating
its exact inner-loop operations (vectorized semantics, single config, single
day-row) bar by bar, using the engine's OWN tensor values.
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

di = M.trading_days.index(DAY)
ce_h = G.ce_h[di].cpu().numpy()
ce_l = G.ce_l[di].cpu().numpy()

# Walk the static engine's EXACT loop for a single (b,d) cell, single config:
# arm_window=10, atr_sl=True, atr_mult=1.0, atr_period=10, be_trigger=0.6,
# be_buffer=1.0, touch_buffer=0.0, tp=15, sl=7, EMA20 gate.
atr = SL.ce_atr_seed[10][di].cpu().numpy()
s1 = SL.ce_s1_seed[di].cpu().numpy()
m6 = G.ce_m6_full[di].cpu().numpy().astype(bool)
sup = G.ce_super_full[di].cpu().numpy().astype(bool)
ema20 = SL.ce_ema20_seed[di].cpu().numpy()
c = G.ce_c[di].cpu().numpy()
bnc = (ce_l <= ema20) & (c >= ema20 - 0.5)
for cc in SL.ce_tf_seed:
    tlo = cc['lo'][di].cpu().numpy(); tcl = cc['cl'][di].cpu().numpy()
    bnc |= (tlo <= ema20) & (tcl >= ema20 - 0.5)
# PE side too (state machine interleaves)
pe_h = G.pe_h[di].cpu().numpy(); pe_l = G.pe_l[di].cpu().numpy(); pe_c = G.pe_c[di].cpu().numpy()
pe_atr = SL.pe_atr_seed[10][di].cpu().numpy()
pe_s1 = SL.pe_s1_seed[di].cpu().numpy()
pe_m6 = G.pe_m6_full[di].cpu().numpy().astype(bool)
pe_sup = G.pe_super_full[di].cpu().numpy().astype(bool)
pe_ema = SL.pe_ema20_seed[di].cpu().numpy()
pe_bnc = (pe_l <= pe_ema) & (pe_c >= pe_ema - 0.5)
for cc in SL.pe_tf_seed:
    tlo = cc['lo'][di].cpu().numpy(); tcl = cc['cl'][di].cpu().numpy()
    pe_bnc |= (tlo <= pe_ema) & (tcl >= pe_ema - 0.5)

ARM, MULT, BE, BEBUF, TP_CAP = 10, 1.0, 0.60, 1.0, 15.0
in_pos = False; pos = 0
entry = sl = tp = 0.0; be_done = False
pe_a = ce_a = False; pe_at = ce_at = -999
log = []
for t in range(1, 345):
    if in_pos:
        l_arr = pe_l if pos == 1 else ce_l
        h_arr = pe_h if pos == 1 else ce_h
        sl_hit = l_arr[t] <= sl
        tp_hit = (not sl_hit) and (h_arr[t] >= tp)
        if sl_hit:
            log.append((t, f"{'PE' if pos==1 else 'CE'} SL exit {sl:.2f} (entry {entry:.2f})"))
            in_pos = False; pos = 0; be_done = False
            pe_a = ce_a = False
        elif tp_hit:
            log.append((t, f"{'PE' if pos==1 else 'CE'} TP exit {tp:.2f} (entry {entry:.2f})"))
            in_pos = False; pos = 0; be_done = False
            pe_a = ce_a = False
    # arming
    if (not in_pos) and pe_s1[t] <= 25.0:
        pe_a = True; pe_at = t
    if (not in_pos) and s1[t] <= 25.0:
        ce_a = True; ce_at = t
    pe_a = pe_a and (t - pe_at <= ARM)
    ce_a = ce_a and (t - ce_at <= ARM)
    # BE
    if in_pos and not be_done:
        h_arr = pe_h if pos == 1 else ce_h
        d0 = entry - sl
        bepx = entry + BE * d0
        if h_arr[t] >= bepx:
            sl = entry + BEBUF
            be_done = True
            log.append((t, f"BE RATCHET -> SL {sl:.2f} (bepx {bepx:.2f})"))
    # entries (PE first — matches engine's PE-then-CE order)
    if not in_pos:
        if pe_a and (pe_m6[t] or pe_sup[t]) and pe_bnc[t]:
            entry = pe_c[t]; d = max(min(pe_atr[t]*MULT, TP_CAP), 2.0)
            sl = entry - d; tp = entry + d
            pos = 1; in_pos = True; be_done = False; pe_a = False
            log.append((t, f"ENTRY-PE @ {entry:.2f} d {d:.2f}"))
        elif ce_a and (m6[t] or sup[t]) and bnc[t]:
            entry = c[t]; d = max(min(atr[t]*MULT, TP_CAP), 2.0)
            sl = entry - d; tp = entry + d
            pos = 2; in_pos = True; be_done = False; ce_a = False
            log.append((t, f"ENTRY-CE @ {entry:.2f} d {d:.2f}"))

print(f"static-semantics replay: {len([x for x in log if 'ENTRY' in x[1]])} entries")
for t, msg in log:
    print(f"  bar {t}: {msg}")
