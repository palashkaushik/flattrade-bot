"""Sweep SR touch-buffer on the Last Hope engine.

Per user request: rerun the sweep with different SR "touch buffers" ---
0.5, 1, 2, 3, 4, 5 option points to the S/R level --- and find the
buffer that maximises net points / win rate on the established winners.

Sweeps touch_buffer on:
  (a) the FLAT base winner (be_trigger=0, be_buffer=0)
  (b) the BREAKEVEN winner (be_trigger=0.70, be_buffer=1.0)

All other params held at the winning config:
  kind='B', sl=15, tp=15, arm_window=10, use_elder=False, use_rsi=False,
  reversal=False, atr_sl=True, atr_mult=1.5, atr_period=10, cap=0,
  use_bias=False, tp_frac=1.0 (entry_start=0 / entry_end=345 defaults).
"""
import sys, time, csv, statistics, itertools
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
import run_7y_v4_master as M
import gpu_sim_last_hope as G

BUFFERS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def enriched(trades):
    base = M._metrics_from_trades(trades)
    n = base['trades']
    if n == 0:
        return {**base, 'profit_factor': 0.0, 'sharpe': 0.0, 'sortino': 0.0, 'calmar': 0.0}
    pnls = [t[5] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp = sum(wins)
    gl = -sum(losses)
    pf = gp / gl if gl > 0 else float('inf')
    daily = {}
    for t in trades:
        daily[t[0]] = daily.get(t[0], 0.0) + t[5]
    dvals = list(daily.values())
    md = statistics.mean(dvals)
    sd = statistics.stdev(dvals) if len(dvals) > 1 else 0.0
    sharpe = (md / sd) * (252 ** 0.5) if sd > 0 else 0.0
    downside = [min(0.0, d) for d in dvals]
    dd_dev = (sum(d * d for d in downside) / len(dvals)) ** 0.5
    sortino = (md / dd_dev) * (252 ** 0.5) if dd_dev > 0 else 0.0
    calmar = base['net_rs'] / base['max_dd'] if base['max_dd'] > 0 else float('inf')
    return {**base, 'profit_factor': pf, 'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar}


def base_cfg(buf, be):
    c = dict(kind='B', sl=15, tp=15, arm_window=10, use_elder=False, use_rsi=False,
             reversal=False, atr_sl=True, atr_mult=1.5, atr_period=10, cap=0,
             use_bias=False, tp_frac=1.0, touch_buffer=buf)
    c['be_trigger'] = 0.70 if be else 0.0
    c['be_buffer'] = 1.0 if be else 0.0
    return c


labels = []
param_cfgs = []
for b in BUFFERS:
    labels.append(('FLAT', b)); param_cfgs.append(base_cfg(b, False))
    labels.append(('BE', b)); param_cfgs.append(base_cfg(b, True))

print(f"##### TOUCH-BUFFER SWEEP (7y) | {len(param_cfgs)} configs #####", flush=True)
t0 = time.time()
tl = G.gpu_sim_batch(param_cfgs)
print(f"  sim done in {time.time()-t0:.0f}s", flush=True)

rows = []
for (mode, buf), c, tr in zip(labels, param_cfgs, tl):
    m = enriched(tr)
    rows.append((mode, buf, m))
    print(f"  {mode:>4} buf={buf:>4} | net_rs={m['net_rs']:>14,.2f} wr={m['wr']:>6.2f}% "
          f"trades={m['trades']:>6} dd={m['max_dd']:>10,.2f} pf={m['profit_factor']:>6.3f} "
          f"sharpe={m['sharpe']:>6.2f} sortino={m['sortino']:>6.2f} calmar={m['calmar']:>6.2f}",
          flush=True)

cols = ['mode', 'touch_buffer', 'be_trigger', 'be_buffer', 'trades', 'wr', 'net_rs',
        'max_dd', 'profit_factor', 'sharpe', 'sortino', 'calmar', 'avg_sl', 'avg_tp', 'avg_trades_day']
with open(r'C:\Websites\FLATTRADE BOT\sweep_touch_buffer.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(cols)
    for (mode, buf, m) in rows:
        def num(x):
            return 9999.0 if x == float('inf') else round(x, 4)
        w.writerow([mode, buf, (0.70 if mode == 'BE' else 0.0), (1.0 if mode == 'BE' else 0.0),
                    m['trades'], round(m['wr'], 4), round(m['net_rs'], 2), round(m['max_dd'], 2),
                    num(m['profit_factor']), round(m['sharpe'], 3), round(m['sortino'], 3),
                    num(m['calmar']), round(m['avg_sl'], 3), round(m['avg_tp'], 3),
                    round(m['avg_trades_day'], 3)])

print("\nDONE -> sweep_touch_buffer.csv", flush=True)
