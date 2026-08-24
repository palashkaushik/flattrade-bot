"""Detailed Trade Reason Inspector for 1m Stochastics Session."""

from test_1m_stoch_all import load_day_dataset, SPOT_YESTERDAY, OPTS_YESTERDAY, SPOT_TODAY, OPTS_TODAY
from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.divergence import DivergenceEngine

def main():
    _, yest_opts = load_day_dataset(SPOT_YESTERDAY, OPTS_YESTERDAY)
    today_spot, today_opts = load_day_dataset(SPOT_TODAY, OPTS_TODAY)

    targets = [
        ("PE", 24750, 562),  # 09:22
        ("PE", 24750, 664),  # 11:04
        ("CE", 24450, 721),  # 12:01
        ("CE", 24450, 758),  # 12:38
        ("CE", 24450, 783),  # 13:03
    ]

    for side, strike, entry_m in targets:
        stoch = QuadStochastics()
        div = DivergenceEngine()
        history = []

        # Warm up yesterday
        for m in sorted(yest_opts[(side, strike)].keys()):
            o, h, l, c = yest_opts[(side, strike)][m]
            stoch.push(h, l, c)

        # Today
        today_map = today_opts[(side, strike)]
        armed_m = None
        armed_type = ""
        armed_s1, armed_s4 = None, None
        pin_m = None
        pin_high = None

        print("\n" + "=" * 110)
        print(f"TRADE REASON ANALYSIS FOR {side} {strike} @ ENTRY {entry_m // 60:02d}:{entry_m % 60:02d}")
        print("=" * 110)

        for m in sorted(today_map.keys()):
            if m > entry_m:
                break
            o, h, l, c = today_map[m]
            c1m = Candle(open=o, high=h, low=l, close=c, minute=m)
            history.append(c1m)

            vals = stoch.push(h, l, c)
            s1, s2, s3, s4 = (vals.get(k) for k in ("s1d", "s2d", "s3d", "s4d"))
            div.update(c, s1)
            has_div = div.has_bullish_trough_divergence()

            is_flag = False if any(v is None for v in (s1, s4)) else (s4 >= 79.5 and s1 <= 20.5)
            is_super = False if any(v is None for v in (s1, s2, s3, s4)) else all(v <= 20.5 for v in (s1, s2, s3, s4))

            if (is_flag or is_super) and has_div:
                armed_m = m
                armed_type = "SuperSignal" if is_super else "Quad Flag"
                armed_s1, armed_s4 = s1, s4

            if BullishPinBarDetector.is_bullish_pin_bar(c1m):
                pin_m = m
                pin_high = h

        e_t = f"{entry_m // 60:02d}:{entry_m % 60:02d}"
        arm_t = f"{armed_m // 60:02d}:{armed_m % 60:02d}" if armed_m is not None else "N/A"
        pin_t = f"{pin_m // 60:02d}:{pin_m % 60:02d}" if pin_m is not None else "N/A"
        entry_c = today_map[entry_m]

        s1_s = f"{armed_s1:.1f}" if armed_s1 is not None else "N/A"
        s4_s = f"{armed_s4:.1f}" if armed_s4 is not None else "N/A"
        ph_s = f"{pin_high:.2f}" if pin_high is not None else "N/A"

        print(f"1. Setup Armed Time      : [{arm_t}] ({armed_type} | S1={s1_s}, S4={s4_s} + Bullish Trough Div)")
        print(f"2. Vicinity Pin Bar Time : [{pin_t}] (High = Rs {ph_s})")
        print(f"3. Entry Trigger Time    : [{e_t}] (Close = Rs {entry_c[3]:.2f} > Pin Bar High Rs {ph_s})")

if __name__ == "__main__":
    main()
