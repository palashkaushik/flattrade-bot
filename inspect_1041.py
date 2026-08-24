"""Inspect 10:41 AM 2m Trade on CE 24500."""

from run_today_backtest import load_day_dataset, SPOT_YESTERDAY, OPTS_YESTERDAY, SPOT_TODAY, OPTS_TODAY
from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.divergence import DivergenceEngine

def main():
    _, yest_opts = load_day_dataset(SPOT_YESTERDAY, OPTS_YESTERDAY)
    today_spot, today_opts = load_day_dataset(SPOT_TODAY, OPTS_TODAY)

    # 1. Warm up CE 24500 stochastics using yesterday's data
    stoch_2m = QuadStochastics()
    div_2m = DivergenceEngine()
    buf_1m = []
    history_2m = []

    # Yesterday 2m candles
    yest_ce = yest_opts[("CE", 24500)]
    for m in sorted(yest_ce.keys()):
        o, h, l, c = yest_ce[m]
        c1m = Candle(open=o, high=h, low=l, close=c, minute=m)
        buf_1m.append(c1m)
        if len(buf_1m) == 2:
            c1, c2 = buf_1m
            c2m = Candle(open=c1.open, high=max(c1.high, c2.high), low=min(c1.low, c2.low), close=c2.close, minute=c2.minute)
            buf_1m = []
            vals = stoch_2m.push(c2m.high, c2m.low, c2m.close)
            div_2m.update(c2m.close, vals.get("s1d"))

    # Today 2m candles up to 10:45 AM (minute 645)
    today_ce = today_opts[("CE", 24500)]
    print("=" * 110)
    print(f"2-MINUTE TIMEFRAME INSPECTION ON CE 24500 (10:20 AM to 10:45 AM)")
    print("=" * 110)
    print(f"{'TIME':5s} | {'OPEN':7s} | {'HIGH':7s} | {'LOW':7s} | {'CLOSE':7s} | {'S1':6s} | {'S2':6s} | {'S3':6s} | {'S4':6s} | {'BULL_DIV'} | {'PIN_BAR'}")
    print("-" * 110)

    for m in sorted(today_ce.keys()):
        if m > 645:
            break
        o, h, l, c = today_ce[m]
        c1m = Candle(open=o, high=h, low=l, close=c, minute=m)
        buf_1m.append(c1m)
        if len(buf_1m) == 2:
            c1, c2 = buf_1m
            c2m = Candle(open=c1.open, high=max(c1.high, c2.high), low=min(c1.low, c2.low), close=c2.close, minute=c2.minute)
            buf_1m = []
            history_2m.append(c2m)
            vals = stoch_2m.push(c2m.high, c2m.low, c2m.close)
            s1, s2, s3, s4 = (vals.get(k) for k in ("s1d", "s2d", "s3d", "s4d"))
            div_2m.update(c2m.close, s1)
            has_div = div_2m.has_bullish_trough_divergence()
            is_pin = BullishPinBarDetector.is_bullish_pin_bar(c2m)

            if m >= 620: # 10:20 AM onwards
                t_str = f"{m // 60:02d}:{m % 60:02d}"
                s1_s = f"{s1:.1f}" if s1 else "None"
                s2_s = f"{s2:.1f}" if s2 else "None"
                s3_s = f"{s3:.1f}" if s3 else "None"
                s4_s = f"{s4:.1f}" if s4 is not None else "None"
                print(f"{t_str:5s} | {c2m.open:7.2f} | {c2m.high:7.2f} | {c2m.low:7.2f} | {c2m.close:7.2f} | {s1_s:>6s} | {s2_s:>6s} | {s3_s:>6s} | {s4_s:>6s} | {str(has_div):8s} | {str(is_pin)}")

if __name__ == "__main__":
    main()
