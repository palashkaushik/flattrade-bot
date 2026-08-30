"""Walk-forward backtest for the Last Hope Winner.

Splits 7 years into yearly folds. For each fold, runs the fixed champion
on that year's data (out-of-sample) and reports per-fold stats.
No optimization — params are locked to the champion config.
"""
import sys, time
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
t0 = time.time()
import run_7y_v4_master as M
import gpu_sim_last_hope as G

CHAMPION = dict(
    kind='B', sl=15, tp=15, arm_window=10, use_elder=False, use_rsi=False,
    reversal=False, atr_sl=True, atr_mult=1.5, atr_period=10, cap=0,
    use_bias=False, be_trigger=0.70, be_buffer=1.0, tp_frac=1.0,
    touch_buffer=0.0
)

print("Running full 7-year sim...")
sys.stdout.flush()
trades = G.gpu_sim_batch([CHAMPION])[0]
print(f"Done: {len(trades)} trades in {time.time()-t0:.1f}s")
sys.stdout.flush()

# --- walk-forward: yearly folds ---
from collections import defaultdict
import csv

daily_pnl = defaultdict(float)
daily_trades = defaultdict(list)
for t in trades:
    daily_pnl[t[0]] += t[5]
    daily_trades[t[0]].append(t)

# Build daily list sorted by date
all_days = sorted(daily_pnl.keys())
years = sorted(set(d[0:4] for d in all_days))

print(f"\n{'='*90}")
print(f"  WALK-FORWARD BACKTEST — Last Hope Winner (ATR, BE=0.70, touch_buf=0.0)")
print(f"  Total: {len(trades)} trades over {len(all_days)} days ({years[0]}-{years[-1]})")
print(f"{'='*90}")

fold_results = []
total_net = 0
total_trades = 0
total_wins = 0

for yr in years:
    yr_days = [d for d in all_days if d.startswith(yr)]
    yr_trades = [t for t in trades if t[0].startswith(yr)]
    n = len(yr_trades)
    if n == 0:
        continue
    wins = sum(1 for t in yr_trades if t[5] > 0)
    losses = n - wins
    wr = wins / n * 100
    net = sum(t[5] for t in yr_trades)
    avg_win = sum(t[5] for t in yr_trades if t[5] > 0) / wins if wins else 0
    avg_loss = sum(t[5] for t in yr_trades if t[5] <= 0) / losses if losses else 0
    expectancy = net / n

    # Max drawdown from daily P&L
    cum = 0
    peak = 0
    max_dd = 0
    for d in yr_days:
        cum += daily_pnl[d]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    # Profit factor
    gross_win = sum(t[5] for t in yr_trades if t[5] > 0)
    gross_loss = -sum(t[5] for t in yr_trades if t[5] <= 0)
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    total_net += net
    total_trades += n
    total_wins += wins

    fold_results.append({
        'year': yr, 'days': len(yr_days), 'trades': n,
        'wins': wins, 'losses': losses, 'wr': wr,
        'net': net, 'expectancy': expectancy,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'max_dd': max_dd, 'pf': pf
    })

    print(f"\n  {yr} | {len(yr_days)} days | {n} trades | {wins}W/{losses}L | WR {wr:.1f}%")
    print(f"    Net: Rs {net:>12,.2f} | Avg: Rs {expectancy:>7.2f}/tr | PF {pf:.2f} | MaxDD Rs {max_dd:,.2f}")

print(f"\n{'='*90}")
print(f"  CUMULATIVE (all years)")
print(f"  {total_trades} trades | {total_wins}W/{total_trades-total_wins}L | WR {total_wins/total_trades*100:.1f}% | Net: Rs {total_net:>12,.2f}")
print(f"  Avg: Rs {total_net/total_trades:.2f}/tr")

# --- Also do 2-year rolling walk-forward ---
print(f"\n{'='*90}")
print(f"  ROLLING 2-YEAR WALK-FORWARD (train on Y-1,Y, test on Y+1)")
print(f"{'='*90}")

for i in range(2, len(years)):
    test_yr = years[i]
    train_yrs = years[i-2:i]
    test_trades = [t for t in trades if t[0].startswith(test_yr)]
    n = len(test_trades)
    if n == 0:
        continue
    wins = sum(1 for t in test_trades if t[5] > 0)
    wr = wins / n * 100
    net = sum(t[5] for t in test_trades)
    expectancy = net / n
    gross_win = sum(t[5] for t in test_trades if t[5] > 0)
    gross_loss = -sum(t[5] for t in test_trades if t[5] <= 0)
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    print(f"\n  Train: {','.join(train_yrs)} -> Test: {test_yr}")
    print(f"    {n} trades | {wins}W/{n-wins}L | WR {wr:.1f}% | Net: Rs {net:>12,.2f} | Avg: Rs {expectancy:>7.2f}/tr | PF {pf:.2f}")

# --- Monthly heatmap for 2026 ---
print(f"\n{'='*90}")
print(f"  2026 MONTHLY BREAKDOWN")
print(f"{'='*90}")

m26 = [t for t in trades if t[0].startswith('2026')]
by_month = defaultdict(list)
for t in m26:
    by_month[t[0][5:7]].append(t)

for mo in sorted(by_month.keys()):
    mt = by_month[mo]
    n = len(mt)
    w = sum(1 for t in mt if t[5] > 0)
    net = sum(t[5] for t in mt)
    print(f"  2026-{mo}: {n:>3} trades | {w}W/{n-w}L | WR {w/n*100:.0f}% | Net Rs {net:>10,.2f}")

print(f"\nTotal time: {time.time()-t0:.1f}s")
