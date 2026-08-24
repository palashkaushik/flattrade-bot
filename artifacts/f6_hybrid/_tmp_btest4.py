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

# find key0.6 sl1.25 ap10 both Fri=F dl20 dp30 in full list
target = {'ut_key':0.6,'atr_period':10,'sl_m':1.25,'tp_m':0.29,'direction':'both','allow_friday':False,'daily_loss_pts':20,'daily_profit_pts':30}
full_idx = combos.index(target)
batch = combos[full_idx:full_idx+100]
tidx = batch.index(target)
standalone = m.evaluate_batch('B09',[target],None)[0]['net_rs']

# Reproduce sweep: 3 sequential calls
r_full = m.evaluate_batch("B09", batch, None)
r_is   = m.evaluate_batch("B09", batch, m.d_is_mask)
r_oos  = m.evaluate_batch("B09", batch, m.d_oos_mask)
print("standalone key0.6 sl1.25:", round(standalone,1))
print("batch r_full[tidx]:", round(r_full[tidx]['net_rs'],1))
print("batch r_is[tidx]:", round(r_is[tidx]['net_rs'],1))
print("batch r_oos[tidx]:", round(r_oos[tidx]['net_rs'],1))
print("r_full[tidx] after all 3 calls:", round(r_full[tidx]['net_rs'],1))
