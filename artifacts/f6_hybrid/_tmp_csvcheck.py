import csv
rows = list(csv.DictReader(open('b09_marni_core_results.csv')))
for key, sl in [('0.6','1.25'), ('0.6','1.115'), ('0.7','1.25')]:
    for r in rows:
        if (r['ut_key']==key and r['atr_period']=='10' and r['sl_m']==sl and r['tp_m']=='0.29'
                and r['direction']=='both' and r['allow_friday']=='False'
                and r['daily_loss_pts']=='20' and r['daily_profit_pts']=='30'):
            print('key', key, 'sl', sl, '| full', r['full_net'], 'is', r['is_net'],
                  'oos', r['oos_net'], 'tr', r['full_trades'], 'oos_tr', r['oos_trades'])
