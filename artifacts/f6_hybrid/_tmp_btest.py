import optimized_gpu_backtest as m
base = {'atr_period':10,'sl_m':1.25,'tp_m':0.29,'direction':'both','allow_friday':False,'daily_loss_pts':20,'daily_profit_pts':30}
combos = [dict(base, ut_key=k) for k in [0.5,0.6,0.7,0.8]]
single = {k: m.evaluate_batch('B09',[dict(base, ut_key=k)], None)[0]['net_rs'] for k in [0.5,0.6,0.7,0.8]}
batch = m.evaluate_batch('B09', combos, None)
print('key | single      batch')
for i,k in enumerate([0.5,0.6,0.7,0.8]):
    print(k, round(single[k],1), round(batch[i]['net_rs'],1))
