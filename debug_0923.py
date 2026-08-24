"""Deep Debugger for 09:23 AM Setup, Divergence & Trigger."""

from run_today_backtest import load_today_data
from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.divergence import DivergenceEngine

def main():
    spot_map, opts_groups = load_today_data()
    
    print("Evaluating 09:15 to 09:30 AM candles for all strikes...")
    
    for (side, strike), minute_map in sorted(opts_groups.items()):
        stoch = QuadStochastics()
        div = DivergenceEngine()
        pending_pin = None
        setup_active = False

        for m in sorted(minute_map.keys()):
            if m > 570: # Stop at 09:30 AM
                break
                
            o_px, h_px, l_px, c_px = minute_map[m]
            candle = Candle(open=o_px, high=h_px, low=l_px, close=c_px, minute=m)
            t_str = f"{m // 60:02d}:{m % 60:02d}"

            stoch_vals = stoch.push(h_px, l_px, c_px)
            s1, s2, s3, s4 = (stoch_vals.get(k) for k in ("s1d", "s2d", "s3d", "s4d"))

            div.update(c_px, s1)
            has_bull_div = div.has_bullish_trough_divergence()
            is_pin = BullishPinBarDetector.is_bullish_pin_bar(candle)

            is_flag = False if any(v is None for v in (s1, s4)) else (s4 >= 79.5 and s1 <= 20.5)
            is_super = False if any(v is None for v in (s1, s2, s3, s4)) else all(v <= 20.5 for v in (s1, s2, s3, s4))

            if (is_flag or is_super) and has_bull_div:
                setup_active = True

            triggered = False
            if setup_active:
                if pending_pin is not None:
                    if BullishPinBarDetector.is_breakout_confirmed(pending_pin, candle):
                        triggered = True
                        setup_active = False
                        pending_pin = None
                    else:
                        pending_pin = None

                if is_pin:
                    pending_pin = candle

            if m in range(560, 566): # 09:20 to 09:25
                print(f"[{t_str}] {side} {strike} | O:{o_px:.2f} H:{h_px:.2f} L:{l_px:.2f} C:{c_px:.2f}")
                print(f"   S1:{s1} S2:{s2} S3:{s3} S4:{s4}")
                print(f"   BullDiv:{has_bull_div} | PinBar:{is_pin} | SetupActive:{setup_active} | Triggered:{triggered}")

if __name__ == "__main__":
    main()
