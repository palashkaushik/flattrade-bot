import csv
rows = list(csv.DictReader(open('b09_marni_core_results.csv')))
def f(x): return float(x)
pooled = [r for r in rows if f(r['full_trades']) >= 30]

print("=== NON-WALK-FORWARD TOP 8 by full_net ===")
for r in sorted(pooled, key=lambda r: f(r['full_net']), reverse=True)[:8]:
    print(f"  net {f(r['full_net']):10.1f} PF {r['full_pf']:>5} WR {r['full_wr']:>5} tr {r['full_trades']:>4} | IS {f(r['is_net']):9.1f} OOS {f(r['oos_net']):9.1f} OOSpf {r['oos_pf']} | key{r['ut_key']} ap{r['atr_period']} sl{r['sl_m']} tp{r['tp_m']} {r['direction'][:1]} f{r['allow_friday']} dl{r['daily_loss_pts']} dp{r['daily_profit_pts']}")

print("\n=== WALK-FORWARD TOP 8 by oos_net ===")
fw = [r for r in pooled if f(r['oos_trades']) >= 10]
for r in sorted(fw, key=lambda r: f(r['oos_net']), reverse=True)[:8]:
    print(f"  OOS {f(r['oos_net']):10.1f} OOSpf {r['oos_pf']:>5} OOSwr {r['oos_wr']:>5} OOStr {r['oos_trades']:>4} | full {f(r['full_net']):10.1f} IS {f(r['is_net']):9.1f} | key{r['ut_key']} ap{r['atr_period']} sl{r['sl_m']} tp{r['tp_m']} {r['direction'][:1]} f{r['allow_friday']} dl{r['daily_loss_pts']} dp{r['daily_profit_pts']}")
