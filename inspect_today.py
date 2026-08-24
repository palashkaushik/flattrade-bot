"""Inspect Today's (2026-08-05) Stochastics, Pin Bars & Divergences."""

from run_today_backtest import load_today_data, SESSION_START, SESSION_END, CE_OFFSET, PE_OFFSET
from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.divergence import DivergenceEngine

def main():
    spot_map, opts_groups = load_today_data()
    minutes = sorted(spot_map.keys())

    print(f"Inspecting today's 2026-08-05 intraday session ({len(minutes)} mins)...")
    
    pin_bars_found = []
    flags_found = []
    super_found = []
    bull_divs_found = []

    trackers = {
        "CE": {"stoch": QuadStochastics(), "div": DivergenceEngine()},
        "PE": {"stoch": QuadStochastics(), "div": DivergenceEngine()}
    }

    for minute in minutes:
        if minute < SESSION_START or minute > SESSION_END:
            continue

        spot_px = spot_map[minute]
        atm = int(round(spot_px / 50.0) * 50)
        active_strikes = {"CE": atm + CE_OFFSET, "PE": atm + PE_OFFSET}
        t_str = f"{minute // 60:02d}:{minute % 60:02d}"

        for side in ("CE", "PE"):
            strike = active_strikes[side]
            candle_data = opts_groups.get((side, strike), {}).get(minute)
            if candle_data is None:
                continue

            o_px, h_px, l_px, c_px = candle_data
            candle = Candle(open=o_px, high=h_px, low=l_px, close=c_px, minute=minute)
            
            # Pin Bar
            if BullishPinBarDetector.is_bullish_pin_bar(candle):
                pin_bars_found.append((t_str, side, strike, c_px))

            # Stochastics & Divergence
            stoch = trackers[side]["stoch"]
            div = trackers[side]["div"]
            stoch_vals = stoch.push(h_px, l_px, c_px)
            s1, s2, s3, s4 = (stoch_vals.get(k) for k in ("s1d", "s2d", "s3d", "s4d"))

            if any(v is None for v in (s1, s2, s3, s4)):
                continue

            div.update(c_px, s1)
            has_bull_div = div.has_bullish_trough_divergence()
            if has_bull_div:
                bull_divs_found.append((t_str, side, strike, c_px, s1))

            is_flag = s4 >= 79.5 and s1 <= 20.5
            is_super = all(v <= 20.5 for v in (s1, s2, s3, s4))

            if is_flag:
                flags_found.append((t_str, side, strike, c_px, s1, s4))
            if is_super:
                super_found.append((t_str, side, strike, c_px, s1, s4))

    print(f"\n1. Bullish Pin Bars Found Today: {len(pin_bars_found)}")
    for t_str, side, strike, px in pin_bars_found[:10]:
        print(f"   [{t_str}] {side} {strike} Pin Bar @ Rs {px:.2f}")

    print(f"\n2. Quad Flag Setups Found Today: {len(flags_found)}")
    for t_str, side, strike, px, s1, s4 in flags_found[:10]:
        print(f"   [{t_str}] {side} {strike} Flag Setup | S1: {s1:.1f}, S4: {s4:.1f}")

    print(f"\n3. Quad SuperSignal Setups Found Today: {len(super_found)}")
    for t_str, side, strike, px, s1, s4 in super_found[:10]:
        print(f"   [{t_str}] {side} {strike} SuperSignal | S1: {s1:.1f}, S4: {s4:.1f}")

    print(f"\n4. Bullish Trough Divergences Active Today: {len(bull_divs_found)}")
    for t_str, side, strike, px, s1 in bull_divs_found[:10]:
        print(f"   [{t_str}] {side} {strike} Trough Divergence | S1: {s1:.1f}")

if __name__ == "__main__":
    main()
