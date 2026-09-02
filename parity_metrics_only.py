"""PARITY CHECK: metrics_only vs trade-list metrics must match exactly."""
import sys, time, os
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
os.environ['LH_BIAS'] = '0'

import numpy as np
import torch

import seeded_lib as SL
import gpu_sim_last_hope as G

def metrics_from_trades(trades):
    if not trades:
        return dict(trades=0, wr=0.0, net=0.0, max_dd=0.0, worst_day=0.0)
    n = len(trades)
    wins = sum(1 for t in trades if t[5] > 0)
    net = sum(t[5] for t in trades)
    daily = {}
    for t in trades:
        daily[t[0]] = daily.get(t[0], 0.0) + t[5]
    cum = peak = 0.0; max_dd = 0.0
    for d in sorted(daily):
        cum += daily[d]; peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    worst = min(daily.values())
    return dict(trades=n, wr=100.0 * wins / n, net=net, max_dd=max_dd, worst_day=worst)

# Champion config x 4 gates (incl. the §42 gate)
CHAMP = dict(SL.CHAMPION_BASE)
CHAMP.update(arm_window=15, atr_period=10, atr_mult=1.5, touch_buffer=0.0,
             be_trigger=0.50, be_buffer=1.0, entry_end=SL.T1, entry_start=0)

# §42 gate via the module stack (bounce_idx=None uses TOUCH_BUFFERS mapping
# with tb=0.0 -> stack row 0 = the prebuilt champion bounce at tb 0.0)
cfgs = [dict(CHAMP), dict(CHAMP, arm_window=10, be_trigger=0.7),
        dict(CHAMP, atr_mult=1.25), dict(CHAMP, arm_window=20, be_trigger=0.4)]

# Legacy path
t0 = time.time()
trades_list = G._eager_sim_core(cfgs)
t_legacy = time.time() - t0

# Sweep path
t0 = time.time()
day_pnl, tc, wc = G._eager_sim_core(cfgs, metrics_only=True)
t_metrics = time.time() - t0

dp = day_pnl.cpu().numpy(); tcn = tc.cpu().numpy(); wcn = wc.cpu().numpy()

print(f"{'#':<3} {'legacy net':>14} {'sweep net':>14} {'legacy tr':>10} {'sweep tr':>10} "
      f"{'legacy wr':>10} {'sweep wr':>10} {'legacy dd':>12} {'sweep dd':>12}")
ok = True
for i, tr in enumerate(trades_list):
    m = metrics_from_trades(tr)
    n_tr = int(tcn[i].sum())
    wins = int(wcn[i].sum())
    s_net = float(dp[i].sum())
    s_wr = 100.0 * wins / n_tr if n_tr else 0.0
    cum = peak = 0.0; s_dd = 0.0
    for v in dp[i]:
        cum += v; peak = max(peak, cum); s_dd = max(s_dd, peak - cum)
    match = (abs(m['net'] - s_net) < 0.5 and m['trades'] == n_tr
             and abs(m['wr'] - s_wr) < 0.01 and abs(m['max_dd'] - s_dd) < 0.5)
    ok = ok and match
    print(f"{i:<3} {m['net']:>14,.0f} {s_net:>14,.0f} {m['trades']:>10} {n_tr:>10} "
          f"{m['wr']:>10.2f} {s_wr:>10.2f} {m['max_dd']:>12,.0f} {s_dd:>12,.0f}  {'MATCH' if match else '*** MISMATCH ***'}")

print(f"\nlegacy path: {t_legacy:.1f}s | metrics_only: {t_metrics:.1f}s")
print("PARITY OK" if ok else "PARITY FAILED")
