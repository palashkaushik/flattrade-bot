"""Smoke-test runner that prints every trade for verification.

Usage:
  python smoke_show_trades.py                 # latest 3 available days
  python smoke_show_trades.py 2026-04-28 2026-04-29 2026-04-30
"""
import sys
import test_f6_champion_s3exit as m

spot = m.load_spot()
files = m.option_files("2020-01-01", "2027-12-31")
days = sorted(set(files) & set(spot))
avail = set(days)

req = sys.argv[1:]
sel = [d for d in req if d in avail]
if not sel:
    if req:
        print(f"NOTE: requested {req} not in data (dataset ends {days[-1]}). "
              f"Using latest available 3 days instead.")
    sel = days[-3:]

m.init_worker_local(spot)
tasks = [(d, str(files[d]), str(files[days[days.index(d) - 1]]) if days.index(d) > 0 else "", m.CHAMPION)
         for d in sel]
allt = []
for t in tasks:
    allt += m.process_day(t)

print()
hdr = f"{'DATE':12}{'ENT':>5}{'EX':>5} {'SIDE':>2} {'SYMBOL':>14} {'ENTRY':>8}{'EXIT':>8} {'PTS':>7}{'RS':>8}  {'REASON':>18} {'DUR':>4}"
print(hdr)
print("-" * 92)
for tr in allt:
    print(f"{tr['date']:12}{tr['entry_min']:>5}{tr['exit_min']:>5} {tr['side']:>2} "
          f"{tr['symbol']:>14} {tr['entry']:>8.2f}{tr['exit']:>8.2f} {tr['pts']:>7.2f}{tr['rs']:>8}  "
          f"{tr['reason']:>18} {tr['duration_min']:>4}")
print("-" * 92)
st, _ = m.stats_for(allt)
print(f"Days: {sel}")
print(f"Trades {st['trades']} | WR {st['wr']:.1f}% | Net Rs {st['rs']:+,d} | PF {st['pf']:.2f}")
