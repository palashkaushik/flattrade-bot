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
print("total combos", len(combos))

batch = combos[:100]
br = m.evaluate_batch('B09', batch, None)
mismatch = 0
for j, p in enumerate(batch):
    s = m.evaluate_batch('B09', [p], None)[0]['net_rs']
    b = br[j]['net_rs']
    if abs(s - b) > 1.0:
        mismatch += 1
        if mismatch <= 10:
            print("MISMATCH", p, "single", round(s,1), "batch", round(b,1))
print("mismatches in first 100:", mismatch)
