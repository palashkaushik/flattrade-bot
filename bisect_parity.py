"""Bisect Mode-A mismatch: compare per-bar trigger/bounce masks between the
static (proven) engine and the dyn engine on a pinned day + strike.

Static: seeded_lib masks (pe_s1, pe_m6_full, pe_super_full, bounce @tb0).
Dyn:    the same strike-day row's s1_1m, m6, super_m, and ema20-bounce.
Both on 2025-09-08, PE 2nd-ITM (ATM0915=24850 -> PE 24950).
"""
import sys, os, time
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
os.environ['LH_BIAS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import torch

DAY = "2025-09-08"

# ---------- static ----------
import run_7y_v4_master as M
import seeded_lib as SL
import gpu_sim_last_hope as G

# 09:15 ATM for that day
idx_atm = {d: int(round(M.spot_by_day[d][555] / 50) * 50) for d in [DAY]}
atm = idx_atm[DAY]
pe_k = atm + 100
ce_k = atm - 100
print(f"DAY {DAY} ATM={atm} CE={ce_k} PE={pe_k}")

di = M.trading_days.index(DAY)

# static seeded masks
pe_s1_st = SL.pe_s1_seed[di].cpu().numpy()          # (T1,)
pe_m6_st = G.pe_m6_seed[di].cpu().numpy() if hasattr(G, 'pe_m6_seed') else None
# seeded_lib patches G masks:
pe_m6_st = G.pe_m6_full[di].cpu().numpy()
pe_sup_st = G.pe_super_full[di].cpu().numpy()

# static bounce (tb=0.0) on PE for that day — ELEMENTWISE (the earlier
# (T1,) <= (T1,1) broadcast silently compared EVERY bar to EVERY EMA —
# a bug in this bisect script, not in the engines)
pe_tf_lo = [c['lo'][di] for c in SL.pe_tf_seed]
pe_tf_cl = [c['cl'][di] for c in SL.pe_tf_seed]
ema20_st = SL.pe_ema20_seed[di]
lo_st = G.pe_l[di]; cl_st = G.pe_c[di]
b_st = (lo_st <= ema20_st) & (cl_st >= ema20_st - 0.5)
for tlo, tcl in zip(pe_tf_lo, pe_tf_cl):
    b_st |= (tlo <= ema20_st) & (tcl >= ema20_st - 0.5)
b_st = b_st.cpu().numpy()

# ---------- dyn ----------
os.environ['DYN_DAY_FIRST'] = "2025-09-05"
os.environ['DYN_DAY_LAST'] = "2025-09-09"
os.environ['DYN_FORCE_STATIC'] = '1'
import dyn_strike_engine as DYN

# find the PE strike-day row for (day di_dyn, strike pe_k, side, day's token)
di_dyn = DYN.days.index(DAY)
tok = DYN.day_exp[DAY]
row = DYN.sd_row.get((di_dyn, pe_k, "PE", tok))
print(f"dyn row for ({di_dyn}, {pe_k}, PE, {tok}) = {row}")
if row is None:
    # try any token for that strike+day
    for (d_, k_, s_, t_), r in DYN.sd_row.items():
        if d_ == di_dyn and k_ == pe_k and s_ == "PE":
            row = r; tok = t_
            print(f"  fallback row {r} (token {t_})")
assert row is not None

s1_dy = DYN.tensors["PE"]["s1_1m"][row].cpu().numpy()
m6_dy = DYN.tensors["PE"]["m6"][row].cpu().numpy()
sup_dy = DYN.tensors["PE"]["super_m"][row].cpu().numpy()
# dyn bounce gate (ema20, tb=0)
T = DYN.tensors["PE"]
lo_dy = T["l_s"][row]; cl_dy = T["c_s"][row]; ema_dy = T["ema20"][row]
cond = ((lo_dy <= ema_dy) & (cl_dy >= ema_dy - 0.5))
for tf in (2, 3, 5):
    tlo, tcl = T[f"lo_{tf}"][row], T[f"cl_{tf}"][row]
    cond = cond | ((tlo <= ema_dy) & (tcl >= ema_dy - 0.5))
b_dy = cond.cpu().numpy()

# TF-bucket convention check: static engine's seeded TF lo (extended-axis
# chunking) vs live-bot convention (session-aligned buckets from 09:15)
print("\n=== TF CONVENTION CHECK (PE lo, 5m, first 20 session bars) ===")
st5_lo = SL.pe_tf_seed[3]['lo'][di].cpu().numpy()   # 5m is TF_LIST[-1]
print("static 5m lo (extended-axis):", np.round(st5_lo[:20], 2))
dy5_lo = DYN.tensors["PE"]["lo_5"][row].cpu().numpy()
print("dyn    5m lo (session-align):", np.round(dy5_lo[:20], 2))
lo_st_raw = G.pe_l[di].cpu().numpy()
print("1m raw lows (bars 0-9)      :", np.round(lo_st_raw[:10], 2))

# how many bars does 1m-bounce-ONLY differ? (isolate the TF component)
b_1m_st = (lo_st <= ema20_st) & (cl_st >= ema20_st - 0.5)
b_1m_st = b_1m_st.cpu().numpy() if torch.is_tensor(b_1m_st) else b_1m_st
b_1m_dy = ((lo_dy <= ema_dy) & (cl_dy >= ema_dy - 0.5)).cpu().numpy()
d1m = int((b_1m_st != b_1m_dy).sum())
print(f"\n1m-only bounce diff: {d1m}/345")

# ---------- compare ----------
def cmp(name, a, b):
    d = int((a != b).sum())
    print(f"{name:10s} differing bars: {d}/345" + (f"  at t={np.where(a != b)[0][:12]}" if d else ""))
    return d

print("\n=== MASK COMPARISON (static vs dyn, same day+strike) ===")
cmp("S1(1m)<=25", (pe_s1_st <= 25.0), (s1_dy <= 25.0))
cmp("M6(FLAG)", pe_m6_st.astype(bool), m6_dy.astype(bool))
cmp("SUPER", pe_sup_st.astype(bool), sup_dy.astype(bool))
cmp("BOUNCE", b_st.astype(bool), b_dy.astype(bool))
print("\nsample S1 values (first 15 session bars):")
print("  static:", np.round(pe_s1_st[:15], 2))
print("  dyn   :", np.round(s1_dy[:15], 2))
print("sample EMA20 (bars 0,50,100,200,344):")
print("  static:", [round(float(ema20_st[i]), 3) for i in (0, 50, 100, 200, 344)])
print("  dyn   :", [round(float(ema_dy[i]), 3) for i in (0, 50, 100, 200, 344)])
