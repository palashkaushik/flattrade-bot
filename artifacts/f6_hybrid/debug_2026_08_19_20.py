"""Debug: count raw F6 signals vs Marny-15m-gated events for 08-19/08-20."""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import opt_futures_quad as source
import grid_optimize_f6_atr as f6_eng
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.f6_champion_marny_15m_filter_backtest import (
    CHAMPION_12_50, Option15mHTFBias,
)
from smoke_2026_08_19_20 import load_index_spot, build_opt_map

SESSION_START, SESSION_END = 560, 900


def probe():
    spot_all = load_index_spot()
    opt_map = build_opt_map()
    for day in ("2026-08-19", "2026-08-20"):
        spot = spot_all[day]
        opt_path = opt_map.get(day)
        print(f"\n=== {day} ===")
        if opt_path is None:
            print("  no option file")
            continue
        rec = source.cached_option(str(opt_path))
        df, groups, prefix = rec
        spot_mins = spot["min"]
        raw = 0
        gated = 0
        by_min = defaultdict(lambda: {"raw": 0, "gated": 0, "ut": "", "bull": False, "ha": 0.0, "lin": 0.0})
        for side in ("CE", "PE"):
            atm_strikes = set()
            for m in range(SESSION_START, SESSION_END + 1):
                idx = np.searchsorted(spot_mins, m, side="right") - 1
                if idx >= 0:
                    spot_px = float(spot["close"][idx])
                    atm = int(round(spot_px / 50) * 50)
                    strike = atm - 100 if side == "CE" else atm + 100
                    atm_strikes.add(strike)
            for strike in atm_strikes:
                sym = f"{prefix}{strike}{side}"
                sl = source.make_slice(df, groups, sym)
                if sl is None or len(sl["times"]) < 15:
                    continue
                marny = Option15mHTFBias()
                tracker = f6_eng.MTFTracker(CHAMPION_12_50)
                for j in range(len(sl["times"])):
                    m = int(sl["times"][j])
                    c = Candle(float(sl["open"][j]), float(sl["high"][j]),
                               float(sl["low"][j]), float(sl["close"][j]), minute=m)
                    marny.update_1m(c)
                    bull = marny.snapshot()
                    trigs = tracker.push_1m(c)
                    for trig in trigs:
                        raw += 1
                        by_min[m]["raw"] += 1
                        if bull:
                            gated += 1
                            by_min[m]["gated"] += 1
        print(f"  raw F6 signals: {raw} | gated (15m bullish): {gated}")
        for m in sorted(by_min):
            if by_min[m]["raw"]:
                print(f"  {m//60:02d}:{m%60:02d}  raw={by_min[m]['raw']} gated={by_min[m]['gated']}")


if __name__ == "__main__":
    probe()