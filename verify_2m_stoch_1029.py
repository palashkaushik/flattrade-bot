"""Verify 2m Stochastics at 10:29 AM on CE 24500."""

from run_today_backtest import load_day_dataset, SPOT_YESTERDAY, OPTS_YESTERDAY, SPOT_TODAY, OPTS_TODAY
from flattrade_bot.indicators.stochastic import QuadStochastics
from flattrade_bot.indicators.patterns import Candle

def main():
    _, yest_opts = load_day_dataset(SPOT_YESTERDAY, OPTS_YESTERDAY)
    _, today_opts = load_day_dataset(SPOT_TODAY, OPTS_TODAY)

    # Combine yesterday 1m + today 1m for CE 24500
    yest_ce = yest_opts[("CE", 24500)]
    today_ce = today_opts[("CE", 24500)]

    stoch_2m = QuadStochastics()

    # Feed all 1m candles chronologically, aggregating every 2 1m candles into 1 2m candle
    all_1m = []
    for m in sorted(yest_ce.keys()):
        o, h, l, c = yest_ce[m]
        all_1m.append((m, o, h, l, c))
    for m in sorted(today_ce.keys()):
        o, h, l, c = today_ce[m]
        all_1m.append((m + 1440, o, h, l, c))  # offset today by 1440 mins to keep chronological

    buf = []
    print("=" * 95)
    print("2-MINUTE STOCHASTICS ON CE 24500 TODAY (10:20 AM to 10:35 AM)")
    print("=" * 95)
    print(f"{'TIME':5s} | {'CLOSE':7s} | {'S1 (9,3)':10s} | {'S2 (14,3)':10s} | {'S3 (40,4)':10s} | {'S4 (60,10)':10s}")
    print("-" * 95)

    for item in all_1m:
        m_raw, o, h, l, c = item
        c1m = Candle(open=o, high=h, low=l, close=c, minute=m_raw)
        buf.append(c1m)
        if len(buf) == 2:
            c1, c2 = buf
            c2m = Candle(open=c1.open, high=max(c1.high, c2.high), low=min(c1.low, c2.low), close=c2.close, minute=c2.minute)
            buf = []
            vals = stoch_2m.push(c2m.high, c2m.low, c2m.close)
            
            m_today = c2m.minute - 1440
            if 620 <= m_today <= 635:
                t_str = f"{m_today // 60:02d}:{m_today % 60:02d}"
                s1, s2, s3, s4 = (vals.get(k) for k in ("s1d", "s2d", "s3d", "s4d"))
                print(f"{t_str:5s} | {c2m.close:7.2f} | {s1:10.2f} | {s2:10.2f} | {s3:10.2f} | {s4:10.2f}")

if __name__ == "__main__":
    main()
