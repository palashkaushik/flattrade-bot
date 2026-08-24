import csv
rows = list(csv.DictReader(open('b09_marni_core_results.csv')))
print("total rows:", len(rows))
exact = [r for r in rows if r['ut_key']=='0.6' and r['atr_period']=='10' and r['sl_m']=='1.25' and r['tp_m']=='0.29' and r['direction']=='both' and r['allow_friday']=='False' and r['daily_loss_pts']=='20' and r['daily_profit_pts']=='30']
print("exact-match rows for key0.6 ap10 sl1.25 dl20 dp30:", len(exact))
for r in exact:
    print("  full", r['full_net'], "is", r['is_net'], "oos", r['oos_net'], "tr", r['full_trades'])
# count how many rows have full_net == 14512.6
import collections
c = collections.Counter(r['ut_key']+','+r['atr_period']+','+r['sl_m']+','+r['tp_m']+','+r['direction']+','+r['allow_friday']+','+r['daily_loss_pts']+','+r['daily_profit_pts'] for r in rows)
dups = [k for k,v in c.items() if v>1]
print("num param-combos appearing >1 time:", len(dups))
if dups: print("examples:", dups[:5])
