import optimized_gpu_backtest as m
import itertools

GRID = {
    "ut_key": [0.5, 0.6, 0.7, 0.8],
    "atr_period": [10, 14, 20],
    "sl_m": [1.115, 1.25],
    "tp_m": [0.29, 0.0],
    "direction": ["both", "bull", "bear"],
    "allow_friday": [False, True],
    "daily_loss_pts": [20, 30, 50],
    "daily_profit_pts": [30, 50, 80],
}
ks = list(GRID.keys())
combos = [dict(zip(ks, v)) for v in itertools.product(*(GRID[k] for k in ks))]

# sweep batch 700:800
batch = combos[700:800]
target = {'ut_key':0.6,'atr_period':10,'sl_m':1.25,'tp_m':0.29,'direction':'both','allow_friday':False,'daily_loss_pts':20,'daily_profit_pts':30}
tidx = batch.index(target)   # should be 56
standalone = m.evaluate_batch('B09',[target],None)[0]['net_rs']
r_full = m.evaluate_batch("B09", batch, None)
print("target idx in batch:", tidx)
print("standalone:", round(standalone,1))
print("batch[tidx] full_net:", round(r_full[tidx]['net_rs'],1))
# also check: is the batch[tidx] result actually for a DIFFERENT combo?
print("batch[tidx] params:", batch[tidx])
