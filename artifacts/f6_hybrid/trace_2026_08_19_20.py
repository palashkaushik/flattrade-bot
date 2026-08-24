"""Trace gated F6 events for 08-19/08-20 with prev-day warmup (entry minutes + prices)."""
import sys
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
DAYS = ["2026-08-18", "2026-08-19", "2026-08-20"]


def trace():
    spot_all = load_index_spot()
    opt_map = build_opt_map()
    for day in DAYS:
        spot = spot_all.get(day)
        opt_path = opt_map.get(day)
        if spot is None or opt_path is None:
            print(f"{day}: no data")
            continue
        rec = source.cached_option(str(opt_path))
        df, groups, prefix = rec
        spot_mins = spot["min"]
        print(f"\n=== {day} ===")
        for side in ("CE", "PE"):
            atm_strikes = set()
            for m in range(SESSION_START, SESSION_END + 1):
                idx = np.searchsorted(spot_mins, m, side="right") - 1
                if idx >= 0:
                    spot_px = float(spot["close"][idx])
                    atm = int(round(spot_px / 50) * 50)
                    strike = atm - 100 if side == "CE" else atm + 100
                    atm_strikes.add(strike)
            for strike in sorted(atm_strikes):
                sym = f"{prefix}{strike}{side}"
                sl = source.make_slice(df, groups, sym)
                if sl is None or len(sl["times"]) < 15:
                    continue
                marny = Option15mHTFBias()
                tracker = f6_eng.MTFTracker(CHAMPION_12_50)
                # prev-day warmup (same as engine)
                prev_day = {"2026-08-18": None, "2026-08-19": "2026-08-18", "2026-08-20": "2026-08-19"}[day]
                if prev_day:
                    prev_path = opt_map.get(prev_day)
                    if prev_path:
                        pr = source.cached_option(str(prev_path))
                        p_df, p_groups, p_prefix = pr
                        p_sl = source.make_slice(p_df, p_groups, sym)
                        if p_sl is not None:
                            for pj in range(len(p_sl["times"])):
                                pc = Candle(float(p_sl["open"][pj]), float(p_sl["high"][pj]),
                                            float(p_sl["low"][pj]), float(p_sl["close"][pj]),
                                            minute=int(p_sl["times"][pj]))
                                marny.update_1m(pc)
                                tracker.push_1m(pc)
                for j in range(len(sl["times"])):
                    m = int(sl["times"][j])
                    c = Candle(float(sl["open"][j]), float(sl["high"][j]),
                               float(sl["low"][j]), float(sl["close"][j]), minute=m)
                    marny.update_1m(c)
                    bull = marny.snapshot()
                    trigs = tracker.push_1m(c)
                    for tf, is_rev, stype, px, atr_val in trigs:
                        print(f"  {m//60:02d}:{m%60:02d} {side} {sym} {stype:>12} "
                              f"close={c.close:.2f} bull={bull} atr={atr_val}")


if __name__ == "__main__":
    trace()