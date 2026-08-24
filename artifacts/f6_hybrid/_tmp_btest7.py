import itertools, optimized_gpu_backtest as m

GRID = {"ut_key": [0.5,0.6,0.7,0.8],"atr_period":[10,14,20],"sl_m":[1.115,1.25],"tp_m":[0.29,0.0],"direction":["both","bull","bear"],"allow_friday":[False,True],"daily_loss_pts":[20,30,50],"daily_profit_pts":[30,50,80]}
ks=list(GRID)
def gen_combos():
    for vals in itertools.product(*(GRID[k] for k in ks)):
        yield {k:v for k,v in zip(ks,vals)}
combos=list(gen_combos())
p = {'ut_key':0.6,'atr_period':10,'sl_m':1.25,'tp_m':0.29,'direction':'both','allow_friday':False,'daily_loss_pts':20,'daily_profit_pts':30}
p_idx = combos.index(p)
print("p at combos index:", p_idx, "-> batch", p_idx//100, "local", p_idx%100)

BATCH=100
nb=(len(combos)+BATCH-1)//BATCH
rows=[]
for bi in range(nb):
    batch=combos[bi*BATCH:(bi+1)*BATCH]
    rf=m.evaluate_batch("B09",batch,None)
    ri=m.evaluate_batch("B09",batch,m.d_is_mask)
    ro=m.evaluate_batch("B09",batch,m.d_oos_mask)
    for j,pp in enumerate(batch):
        rows.append({**pp,"full_net":round(rf[j]['net_rs'],1),"is_net":round(ri[j]['net_rs'],1),"oos_net":round(ro[j]['net_rs'],1),"tr":rf[j]['trades']})

rp=[r for r in rows if r['ut_key']==0.6 and r['atr_period']==10 and r['sl_m']==1.25 and r['tp_m']==0.29 and r['direction']=='both' and r['allow_friday']==False and r['daily_loss_pts']==20 and r['daily_profit_pts']==30]
print("FULL-SWEEP-LOOP end-of-process value for p:", rp[0]['full_net'], "tr", rp[0]['tr'])
print("count of p in rows:", len(rp))
