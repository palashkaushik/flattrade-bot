import sys, json, gzip
from pathlib import Path
from datetime import date

ROOT = Path("C:/Websites/FLATTRADE BOT")
sys.path.insert(0, str(ROOT))
from artifacts.flattrade_day_cache import load_day_cache

d14 = date(2026, 8, 14)
d13 = date(2026, 8, 13)
c14 = load_day_cache(Path("artifacts/flattrade_day_cache"), d14)
c13 = load_day_cache(Path("artifacts/flattrade_day_cache"), d13)

print("=== Aug14 cache: contract keys & how many PREVIOUS-day (Aug13) rows each carries ===")
for key, info in c14["contracts"].items():
    rows = info["rows"]
    prev = [r for r in rows if r["time"].split(" ")[0] != "14-08-2026"]
    cur = [r for r in rows if r["time"].split(" ")[0] == "14-08-2026"]
    if key.startswith("PE"):
        closes = [float(r["close"]) for r in cur]
        print(f"{key:<14} total={len(rows):<4} aug13_rows={len(prev):<4} aug14_rows={len(cur):<4} "
              f"premium min={min(closes):.1f} max={max(closes):.1f}")

print()
print("=== Does Aug14 cache even contain Aug13 spot rows (for index warmup)? ===")
print("spot_rows dates present:", sorted(set(r['time'].split(' ')[0] for r in c14['spot_rows'])))
print("=== Aug13 cache spot present? ===", c13 is not None)
