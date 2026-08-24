import optimized_gpu_backtest as m

def mk(uk, sl):
    return {'ut_key':uk,'atr_period':10,'sl_m':sl,'tp_m':0.29,'direction':'both','allow_friday':False,'daily_loss_pts':20,'daily_profit_pts':30}

batch = [mk(uk, sl) for uk in [0.5,0.6,0.7,0.8] for sl in [1.115,1.25]]
singles = { (uk,sl): m.evaluate_batch('B09',[mk(uk,sl)],None)[0]['net_rs'] for uk,sl in [(uk,sl) for uk in [0.5,0.6,0.7,0.8] for sl in [1.115,1.25]] }
br = m.evaluate_batch('B09', batch, None)
print("uk   sl     single      batch")
for i,(uk,sl) in enumerate([(uk,sl) for uk in [0.5,0.6,0.7,0.8] for sl in [1.115,1.25]]):
    print(f"{uk}  {sl}  {singles[(uk,sl)]:10.1f} {br[i]['net_rs']:10.1f}")
