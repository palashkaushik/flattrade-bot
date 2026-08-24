import sys
from pathlib import Path
from datetime import date

ROOT = Path("C:/Websites/FLATTRADE BOT")
sys.path.insert(0, str(ROOT))
from artifacts.flattrade_day_cache import load_day_cache
import artifacts.f6_hybrid.marni_fib_backtest as mb

d14 = date(2026, 8, 14)
d13 = date(2026, 8, 13)
c14 = load_day_cache(Path("artifacts/flattrade_day_cache"), d14)
c13 = load_day_cache(Path("artifacts/flattrade_day_cache"), d13)

def parse(row):
    from datetime import datetime
    t = datetime.strptime(row["time"], "%d-%m-%Y %H:%M:%S")
    return {"time": row["time"], "minute": t.hour*60+t.minute,
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"])}

# PE 24450 rows
key = "PE:24450"
r13 = [parse(r) for r in c13["contracts"][key]["rows"]]
r14 = [parse(r) for r in c14["contracts"][key]["rows"]]

# User's range: high at 15:14 Aug13, low at 09:32 Aug14
print("=== PE 24450 premium around the described range ===")
for r in r13:
    if 900 <= r["minute"] <= 920:   # 15:00-15:20
        print(f"  Aug13 {r['minute']//60:02d}:{r['minute']%60:02d}  H={r['high']:.1f} L={r['low']:.1f} C={r['close']:.1f}")
for r in r14:
    if 560 <= r["minute"] <= 580:   # 09:20-09:40
        print(f"  Aug14 {r['minute']//60:02d}:{r['minute']%60:02d}  H={r['high']:.1f} L={r['low']:.1f} C={r['close']:.1f}")
    if 600 <= r["minute"] <= 605:   # 10:00-10:05
        print(f"  Aug14 {r['minute']//60:02d}:{r['minute']%60:02d}  H={r['high']:.1f} L={r['low']:.1f} C={r['close']:.1f}")

# Simple 0.786 of (max premium before 15:14 Aug13, min premium around 09:32 Aug14)
hi = max(r["high"] for r in r13 if r["minute"] <= 914)
lo = min(r["low"] for r in r14 if 560 <= r["minute"] <= 575)
print(f"\nRange high (<=15:14 Aug13) = {hi:.1f} | low (09:20-09:35 Aug14) = {lo:.1f}")
for orient in ("high_to_low", "low_to_high"):
    lvl = mb.fib_price(hi, lo, 0.786, orient)
    print(f"  0.786 ({orient}) = {lvl:.1f}")

# What does the detector actually emit for PE 24450 on Aug14 with overnight carry?
prev = r13
feed = mb.SymbolFibFeed("option", strict_confirmation=False)
feed.warmup(prev, reset_session=False)
print("\n=== Detector events for PE 24450 (Aug14, overnight carry) ===")
for r in r14:
    for ev in feed.push(r):
        print(f"  {ev['timeframe']} dir={ev['direction']} entry_lvl={ev['entry_level']:.1f} "
              f"fibH={ev['fib_high']:.1f} fibL={ev['fib_low']:.1f} min={r['minute']}")
