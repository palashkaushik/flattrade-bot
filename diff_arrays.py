"""Full session-array diff (h/l/c) static vs dyn for CE 24700 on 2025-09-08."""
import sys, os
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
os.environ['LH_BIAS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import torch

DAY = "2025-09-08"
import run_7y_v4_master as M
import gpu_sim_last_hope as G

di = M.trading_days.index(DAY)
atm = int(round(M.spot_by_day[DAY][555] / 50) * 50)
ce_k = atm - 100

os.environ['DYN_DAY_FIRST'] = "2025-09-05"
os.environ['DYN_DAY_LAST'] = "2025-09-09"
os.environ['DYN_FORCE_STATIC'] = '1'
import dyn_strike_engine as DYN

di_dyn = DYN.days.index(DAY)
tok = DYN.day_exp[DAY]
row = DYN.sd_row[(di_dyn, ce_k, "CE", tok)]

for nm, st, dy in (("h", G.ce_h[di], DYN.tensors["CE"]["h_s"][row]),
                   ("l", G.ce_l[di], DYN.tensors["CE"]["l_s"][row]),
                   ("c", G.ce_c[di], DYN.tensors["CE"]["c_s"][row])):
    a = st.cpu().numpy(); b = dy.cpu().numpy()
    bad = np.where(~np.isclose(a, b, atol=1e-4))[0]
    print(f"{nm}: differing bars {len(bad)}/345" + (f" -> {bad[:15]}" if len(bad) else ""))
    if len(bad):
        for t in bad[:10]:
            print(f"   bar {t}: static {a[t]:.2f} vs dyn {b[t]:.2f}")

# PE too
pe_k = atm + 100
rowp = DYN.sd_row.get((di_dyn, pe_k, "PE", tok))
for nm, st, dy in (("h", G.pe_h[di], DYN.tensors["PE"]["h_s"][rowp]),
                   ("l", G.pe_l[di], DYN.tensors["PE"]["l_s"][rowp]),
                   ("c", G.pe_c[di], DYN.tensors["PE"]["c_s"][rowp])):
    a = st.cpu().numpy(); b = dy.cpu().numpy()
    bad = np.where(~np.isclose(a, b, atol=1e-4))[0]
    print(f"PE {nm}: differing bars {len(bad)}/345" + (f" -> {bad[:15]}" if len(bad) else ""))
