"""Test 1m Stochastics at 10:29 AM on CE 24500."""

from run_today_backtest import load_day_dataset, SPOT_YESTERDAY, OPTS_YESTERDAY, SPOT_TODAY, OPTS_TODAY
from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle
from flattrade_bot.indicators.divergence import DivergenceEngine

def main():
    _, yest_opts = load_day_dataset(SPOT_YESTERDAY, OPTS_YESTERDAY)
    _, today_opts = load_day_dataset(SPOT_TODAY, OPTS_TODAY)

    stoch_1m = QuadStochastics()

    # Yesterday 1m candles for CE 24500
    yest_ce = yest_opts[("CE", 24500)]
    for m in sorted(yest_ce.keys()):
        o, h, l, c = yest_ce[m]
        stoch_1m.push(h, l, c)

    # Today 1m candles up to 10:35 AM
    today_ce = today_opts[("CE", 24500)]
    print("=" * 95)
    print("1-MINUTE STOCHASTICS ON CE 24500 (10:20 AM to 10:35 AM)")
    print("=" * 95)
    print(f"{'TIME':5s} | {'CLOSE':7s} | {'S1 (9,3)':10s} | {'S2 (14,3)':10s} | {'S3 (40,4)':10s} | {'S4 (60,10)':10s}")
    print("-" * 95)

    for m in sorted(today_ce.keys()):
        if m > 635:
            break
        o, h, l, c = today_ce[m]
        vals = stoch_1m.push(h, l, c)
        if m >= 620:
            t_str = f"{m // 60:02d}:{m % 60:02d}"
            s1, s2, s3, s4 = (vals.get(k) for k in ("s1d", "s2d", "s3d", "s4d"))
            print(f"{t_str:5s} | {c:7.2f} | {s1:10.2f} | {s2:10.2f} | {s3:10.2f} | {s4:10.2f}")

if __name__ == "__main__":
    main()
