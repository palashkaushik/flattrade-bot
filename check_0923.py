"""Check 09:23 AM Stochastics, Divergence, Pin Bar and SuperSignal with 09:15 AM Warm-Up."""

from run_today_backtest import load_today_data, CE_OFFSET, PE_OFFSET
from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.divergence import DivergenceEngine

def main():
    spot_map, opts_groups = load_today_data()
    minutes = sorted(spot_map.keys())

    print(f"Loaded {len(minutes)} spot minutes. Warming up stochastics from 09:15 AM (minute 555)...")

    trackers = {
        "CE": {"stoch": QuadStochastics(), "div": DivergenceEngine()},
        "PE": {"stoch": QuadStochastics(), "div": DivergenceEngine()}
    }

    for minute in minutes:
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
            
            stoch = trackers[side]["stoch"]
            div = trackers[side]["div"]
            stoch_vals = stoch.push(h_px, l_px, c_px)
            s1, s2, s3, s4 = (stoch_vals.get(k) for k in ("s1d", "s2d", "s3d", "s4d"))

            div.update(c_px, s1)
            has_bull_div = div.has_bullish_trough_divergence()
            is_pin = BullishPinBarDetector.is_bullish_pin_bar(candle)

            is_flag = False if s4 is None or s1 is None else (s4 >= 79.5 and s1 <= 20.5)
            is_super = False if any(v is None for v in (s1, s2, s3, s4)) else all(v <= 20.5 for v in (s1, s2, s3, s4))

            if minute in (561, 562, 563, 564, 565):  # 09:21 to 09:25
                print(f"[{t_str}] {side} {strike} | Close: {c_px:.2f} | S1: {s1} | S2: {s2} | S3: {s3} | S4: {s4}")
                print(f"      -> Flag: {is_flag} | SuperSignal: {is_super} | BullDiv: {has_bull_div} | PinBar: {is_pin}")

if __name__ == "__main__":
    main()
