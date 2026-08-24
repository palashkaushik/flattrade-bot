import itertools, optimized_gpu_backtest as m

GRID = {"ut_key": [0.5,0.6,0.7,0.8],"atr_period":[10,14,20],"sl_m":[1.115,1.25],"tp_m":[0.29,0.0],"direction":["both","bull","bear"],"allow_friday":[False,True],"daily_loss_pts":[20,30,50],"daily_profit_pts":[30,50,80]}
ks=list(GRID); combos=[dict(zip(ks,v)) for v in itertools.product(*(GRID[k] for k in ks))]
p = {'ut_key':0.6,'atr_period':10,'sl_m':1.25,'tp_m':0.29,'direction':'both','allow_friday':False,'daily_loss_pts':20,'daily_profit_pts':30}
batch = [p] + [c for c in combos if c != p][:99]
rf=m.evaluate_batch("B09",batch,None)
ri=m.evaluate_batch("B09",batch,m.d_is_mask)
ro=m.evaluate_batch("B09",batch,m.d_oos_mask)
idx=[i for i,c in enumerate(batch) if c==p][0]
print("batch index of p:", idx)
print("rf[idx] net/trades:", rf[idx]['net_rs'], rf[idx]['trades'])
print("ri[idx] net/trades:", ri[idx]['net_rs'], ri[idx]['trades'])
print("ro[idx] net/trades:", ro[idx]['net_rs'], ro[idx]['trades'])
print("rf is ri?", rf[idx] is ri[idx], "| ri is ro?", ri[idx] is ro[idx], "| rf is ro?", rf[idx] is ro[idx])
print("type(rf[idx]['net_rs']):", type(rf[idx]['net_rs']))
# standalone
s = m.evaluate_batch('B09',[p],None)[0]
print("standalone net/trades:", s['net_rs'], s['trades'])
