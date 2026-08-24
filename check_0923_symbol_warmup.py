"""Check 09:23 AM Stochastics and Divergence by pre-warming per symbol."""

from run_today_backtest import load_today_data
from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.divergence import DivergenceEngine

def main():
    spot_map, opts_groups = load_today_data()
    minutes = sorted(spot_map.keys())

    print("Pre-warming stochastics for all option symbols...")
    
    # Precompute stochastics for every (side, strike) symbol series
    symbol_stochs = {}
    symbol_divs = {}

    for (side, strike), minute_map in opts_groups.items():
        stoch = QuadStochastics()
        div = DivergenceEngine()
        stoch_out = {}
        div_out = {}

        for m in sorted(minute_map.keys()):
            o_px, h_px, l_px, c_px = minute_map[m]
            vals = stoch.push(h_px, l_px, c_px)
            stoch_out[m] = vals
            s1 = vals.get("s1d")
            div.update(c_px, s1)
            div_out[m] = div.has_bullish_trough_divergence()

        symbol_stochs[(side, strike)] = stoch_out
        symbol_divs[(side, strike)] = div_out

    # Now inspect 09:23 AM for all strikes around ATM (24600 / 24650)
    target_min = 563  # 09:23 AM
    spot_px = spot_map[target_min]
    atm = int(round(spot_px / 50.0) * 50)
    print(f"\nAt 09:23 AM (Minute 563): Spot = {spot_px:.2f}, ATM = {atm}")
    print("=" * 95)
    print(f"{'SYMBOL':15s} | {'CLOSE':8s} | {'S1':7s} | {'S2':7s} | {'S3':7s} | {'S4':7s} | {'BULL_DIV'} | {'QUAD_STATUS'}")
    print("=" * 95)

    for (side, strike), minute_map in sorted(opts_groups.items()):
        if target_min in minute_map:
            o_px, h_px, l_px, c_px = minute_map[target_min]
            s_vals = symbol_stochs[(side, strike)].get(target_min, {})
            s1, s2, s3, s4 = (s_vals.get(k) for k in ("s1d", "s2d", "s3d", "s4d"))
            has_div = symbol_divs[(side, strike)].get(target_min, False)

            s1_str = f"{s1:.1f}" if s1 is not None else "None"
            s2_str = f"{s2:.1f}" if s2 is not None else "None"
            s3_str = f"{s3:.1f}" if s3 is not None else "None"
            s4_str = f"{s4:.1f}" if s4 is not None else "None"

            is_flag = False if any(v is None for v in (s1, s4)) else (s4 >= 79.5 and s1 <= 20.5)
            is_super = False if any(v is None for v in (s1, s2, s3, s4)) else all(v <= 20.5 for v in (s1, s2, s3, s4))

            status = "SUPERSIGNAL" if is_super else ("FLAG" if is_flag else "NORMAL")
            sym_name = f"{side} {strike}"
            print(f"{sym_name:15s} | {c_px:8.2f} | {s1_str:>7s} | {s2_str:>7s} | {s3_str:>7s} | {s4_str:>7s} | {str(has_div):8s} | {status}")

if __name__ == "__main__":
    main()
