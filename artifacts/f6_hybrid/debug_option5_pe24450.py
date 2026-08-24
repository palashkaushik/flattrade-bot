"""Debug: what setup does option5 produce for PE 24450 on Aug 14 (warmed from Aug 13)?"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from artifacts.flattrade_day_cache import load_day_cache
from artifacts.f6_hybrid.marni_fib_backtest import SymbolFibFeed
import artifacts.f6_hybrid.marni_fib_flattrade_cache as mc

cache_dir = Path("artifacts/flattrade_day_cache")
d14 = date(2026, 8, 14)
d13 = date(2026, 8, 13)
c14 = load_day_cache(cache_dir, d14)
c13 = load_day_cache(cache_dir, d13)

def parse(r):
    from datetime import datetime
    p = datetime.strptime(r["time"], "%d-%m-%Y %H:%M:%S")
    return {"minute": p.hour*60+p.minute, "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]), "time": r["time"]}

prev_rows = [parse(r) for r in c13["contracts"]["PE:24450"]["rows"]]
cur_rows = [parse(r) for r in c14["contracts"]["PE:24450"]["rows"]]

feed = SymbolFibFeed("option5", strict_confirmation=False)
feed.warmup(prev_rows, reset_session=False)

print(f"Aug13 prev rows: {len(prev_rows)} | Aug14 cur rows: {len(cur_rows)}")
print("Completed setups on Aug14 (fib_high/fib_low/entry_level/direction/orientation):")
for r in cur_rows:
    for ev in feed.push(r):
        el = ev["entry_level"]
        print(f"  {r['time']} dir={ev['direction']} orient={ev['orientation']} "
              f"hi={ev['fib_high']:.1f} lo={ev['fib_low']:.1f} entry={el:.1f} tf={ev['timeframe']}")
