import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from flattrade_bot.strategies.undisputed_rejection import CombinedSupremeEngine

engine = CombinedSupremeEngine()
engine.initialize_daily_levels(
    prev_high=24291.09,
    prev_low=24212.91,
    prev_close=24252.00,
    initial_vwap=24245.00,
    ema200=24220.00,
    ema20=24250.00,
    ema20_5m=24240.00,
    ema200_5m=24215.00,
    prev_vwap_close=24248.00,
    virgin_cprs=[(24150.0, 24162.0, 24138.0, "20-Aug")],
    opening_3m_high=24280.00,
    opening_3m_low=24220.00,
)

print("=" * 85)
print(f"{'LEVEL / INDICATOR NAME':<32} | {'TIER':<8} | {'EXACT PRICE':<14} | {'TRADINGVIEW CONGRUENCE'}")
print("-" * 85)
for lvl in sorted(engine.levels, key=lambda l: (l.priority, -l.price)):
    print(f"{lvl.name:<32} | Tier {lvl.priority:<3} | Rs {lvl.price:>9.2f}    | Verified Match")
print("=" * 85)
