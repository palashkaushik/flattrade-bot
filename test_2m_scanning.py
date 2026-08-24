"""Test Multi-Timeframe (1m + 2m) Option Scanning Engine."""

import sys
from pathlib import Path
import pandas as pd

from run_today_backtest import load_today_data, SESSION_START, SESSION_END, CE_OFFSET, PE_OFFSET
from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.divergence import DivergenceEngine


def build_2m_candles(minute_map: dict) -> dict:
    """Aggregates 1m candles into 2m candles (keyed by completion minute)."""
    mins = sorted(minute_map.keys())
    res = {}
    for i in range(1, len(mins)):
        m1 = mins[i-1]
        m2 = mins[i]
        if m2 == m1 + 1 and m2 % 2 == 1:  # Completed on odd minutes (09:17, 09:19, 09:21...)
            c1 = minute_map[m1]
            c2 = minute_map[m2]
            res[m2] = (
                c1[0],                   # open
                max(c1[1], c2[1]),       # high
                min(c1[2], c2[2]),       # low
                c2[3],                   # close
            )
    return res


def main():
    spot_map, opts_groups = load_today_data()
    all_minutes = sorted(spot_map.keys())

    print("Evaluating 2-Minute Option Candles today (2026-08-05)...")

    # Build 2m option candles for every symbol
    opts_2m = {key: build_2m_candles(m_map) for key, m_map in opts_groups.items()}

    pin_bars_2m = []
    
    for (side, strike), m_map in opts_2m.items():
        stoch = QuadStochastics()
        div = DivergenceEngine()
        pending_pin = None

        for m in sorted(m_map.keys()):
            o_px, h_px, l_px, c_px = m_map[m]
            candle = Candle(open=o_px, high=h_px, low=l_px, close=c_px, minute=m)
            t_str = f"{m // 60:02d}:{m % 60:02d}"

            stoch_vals = stoch.push(h_px, l_px, c_px)
            s1 = stoch_vals.get("s1d")
            div.update(c_px, s1)

            if BullishPinBarDetector.is_bullish_pin_bar(candle):
                pin_bars_2m.append((t_str, side, strike, c_px))

    print(f"\n2-Minute Bullish Pin Bars Found Today: {len(pin_bars_2m)}")
    for t_str, side, strike, px in pin_bars_2m[:15]:
        print(f"   [{t_str}] 2m {side} {strike} Pin Bar @ Rs {px:.2f}")


if __name__ == "__main__":
    main()
