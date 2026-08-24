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

BATCH = 100
nb = (len(combos) + BATCH - 1) // BATCH
all_rows = []
for bi in range(nb):
    batch = combos[bi*BATCH:(bi+1)*BATCH]
    r_full = m.evaluate_batch("B09", batch, None)
    r_is = m.evaluate_batch("B09", batch, m.d_is_mask)
    r_oos = m.evaluate_batch("B09", batch, m.d_oos_mask)
    for j, p in enumerate(batch):
        rf, ri, ro = r_full[j], r_is[j], r_oos[j]
        all_rows.append({**p,
            "full_net": round(rf["net_rs"], 1),
            "is_net": round(ri["net_rs"], 1),
            "oos_net": round(ro["net_rs"], 1)})

# check key0.6 sl1.25 both Fri=F dl20 dp30 (which is in batch 7)
for r in all_rows:
    if r['ut_key']==0.6 and r['sl_m']==1.25 and r['tp_m']==0.29 and r['direction']=='both' and r['allow_friday']==False and r['daily_loss_pts']==20 and r['daily_profit_pts']==30:
        print("CUMULATIVE ROW:", r)
# sanity: count rows
print("total rows:", len(all_rows))
