"""Verify engine 15m bias (HA + LinReg-SMA11 + UT Bot) against TradingView.

Uses the raw NSE:NIFTY 15m OHLCV pulled from TradingView (regular candles).
TradingView indicator values (Humble LinReg Candles): HA-close + Plot(LinReg).
We replicate the engine's Option15mHTFBias logic exactly and print per-bar
bias for 2026-08-19 and 2026-08-20, then compare the last bar's HA-close and
LinReg plot to TradingView's displayed 24,228.583 / 24,239.9275.
"""
from collections import deque

# Raw NSE:NIFTY 15m bars from TradingView (time, open, high, low, close)
# slice starts 2026-08-18 09:15 IST to warm up UT(10)/LinReg(11) before 08-19.
TV_BARS = [
    (1786938300, 24354.65, 24357.6, 24282.8, 24306.35),
    (1786939200, 24306.05, 24312.45, 24280.85, 24301.0),
    (1786940100, 24301.35, 24309.25, 24283.85, 24283.85),
    (1786941000, 24283.95, 24295.65, 24276.85, 24281.5),
    (1786941900, 24281.45, 24283.9, 24249.95, 24260.3),
    (1786942800, 24260.45, 24261.9, 24235.45, 24238.75),
    (1786943700, 24239.2, 24247.45, 24228.15, 24238.2),
    (1786944600, 24237.55, 24246.6, 24231.45, 24236.45),
    (1786945500, 24236.2, 24243.6, 24226.95, 24240.0),
    (1786946400, 24239.85, 24248.35, 24234.25, 24244.7),
    (1786947300, 24244.75, 24275.25, 24242.1, 24271.7),
    (1786948200, 24271.65, 24275.65, 24256.55, 24270.75),
    (1786949100, 24270.6, 24292.0, 24269.8, 24290.8),
    (1786950000, 24289.95, 24306.75, 24286.1, 24303.5),
    (1786950900, 24303.2, 24336.7, 24302.4, 24333.95),
    (1786951800, 24334.0, 24354.3, 24332.5, 24346.35),
    (1786952700, 24346.6, 24356.15, 24342.1, 24351.85),
    (1786953600, 24352.95, 24359.3, 24339.8, 24353.25),
    (1786954500, 24353.65, 24359.6, 24332.45, 24337.75),
    (1786955400, 24338.3, 24346.75, 24329.5, 24331.0),
    (1786956300, 24331.15, 24343.05, 24322.55, 24326.15),
    (1786957200, 24326.75, 24337.9, 24317.9, 24334.35),
    (1786958100, 24335.4, 24346.05, 24330.95, 24338.2),
    (1786959000, 24337.7, 24360.1, 24330.7, 24341.9),
    (1786959900, 24339.6, 24339.8, 24287.65, 24287.65),
    # ---- 2026-08-19 session ----
    (1787024700, 24225.15, 24269.65, 24211.1, 24245.1),
    (1787025600, 24246.4, 24249.2, 24220.65, 24224.55),
    (1787026500, 24226.15, 24241.35, 24214.85, 24234.5),
    (1787027400, 24234.1, 24234.6, 24213.0, 24220.2),
    (1787028300, 24221.1, 24231.05, 24208.0, 24225.85),
    (1787029200, 24227.0, 24233.4, 24210.05, 24214.8),
    (1787030100, 24216.15, 24216.4, 24187.3, 24200.25),
    (1787031000, 24201.95, 24211.5, 24194.85, 24202.35),
    (1787031900, 24203.75, 24209.15, 24182.3, 24182.65),
    (1787032800, 24181.8, 24201.9, 24174.45, 24198.4),
    (1787033700, 24198.5, 24214.4, 24195.8, 24206.8),
    (1787034600, 24208.3, 24220.8, 24201.85, 24212.95),
    (1787035500, 24213.2, 24217.4, 24203.65, 24212.8),
    (1787036400, 24213.25, 24220.35, 24196.0, 24198.05),
    (1787037300, 24198.0, 24210.7, 24194.45, 24199.05),
    (1787038200, 24200.0, 24204.9, 24185.6, 24202.6),
    (1787039100, 24204.5, 24210.7, 24198.65, 24207.85),
    (1787040000, 24208.35, 24209.0, 24190.1, 24202.0),
    (1787040900, 24202.65, 24207.6, 24187.9, 24190.8),
    (1787041800, 24192.95, 24209.5, 24189.2, 24195.55),
    (1787042700, 24196.95, 24203.5, 24181.75, 24199.1),
    (1787043600, 24201.1, 24207.5, 24191.85, 24205.2),
    (1787044500, 24206.0, 24221.1, 24198.9, 24214.3),
    (1787045400, 24216.6, 24230.75, 24165.45, 24165.8),
    (1787046300, 24166.35, 24166.35, 24154.9, 24154.9),
    # ---- 2026-08-20 session ----
    (1787197500, 24225.2, 24225.2, 24184.55, 24190.8),
    (1787198400, 24190.95, 24212.85, 24185.1, 24210.95),
    (1787199300, 24211.05, 24215.05, 24195.95, 24203.8),
    (1787200200, 24204.05, 24221.9, 24199.7, 24221.85),
    (1787201100, 24220.75, 24226.85, 24207.25, 24210.15),
    (1787202000, 24209.65, 24215.35, 24201.25, 24211.85),
    (1787202900, 24211.45, 24218.1, 24203.9, 24212.55),
    (1787203800, 24212.5, 24223.55, 24202.5, 24204.4),
    (1787204700, 24204.75, 24207.7, 24194.2, 24201.0),
    (1787205600, 24200.75, 24217.4, 24200.1, 24212.35),
    (1787206500, 24212.1, 24220.5, 24209.95, 24216.6),
    (1787207400, 24216.55, 24219.8, 24210.15, 24214.75),
    (1787208300, 24214.55, 24223.0, 24208.15, 24213.75),
    (1787209200, 24214.45, 24221.85, 24207.4, 24217.2),
    (1787210100, 24216.9, 24265.15, 24216.05, 24244.25),
    (1787211000, 24244.15, 24251.7, 24239.05, 24246.8),
    (1787211900, 24247.15, 24251.75, 24237.15, 24247.55),
    (1787212800, 24246.85, 24252.15, 24238.4, 24246.05),
    (1787213700, 24245.1, 24249.85, 24231.2, 24232.95),
    (1787214600, 24231.7, 24241.65, 24227.25, 24238.45),
    (1787215500, 24238.05, 24246.3, 24223.3, 24245.8),
    (1787216400, 24244.85, 24249.55, 24236.0, 24237.0),
    (1787217300, 24236.95, 24240.9, 24231.15, 24238.1),
    (1787218200, 24238.05, 24240.7, 24205.3, 24212.35),
    (1787219100, 24211.6, 24231.85, 24211.6, 24231.85),
]

IST_HOUR_OFFSET = 5 * 3600 + 30 * 60  # UTC->IST


def ist_str(ts):
    lt = ts + IST_HOUR_OFFSET
    from datetime import datetime, timezone
    return datetime.fromtimestamp(lt, tz=timezone.utc).strftime("%m-%d %H:%M")


class IncATR:
    def __init__(self, period=10):
        self.period = period
        self._buf = deque(maxlen=period)
        self.atr = None
        self.prev_close = None
        self._n = 0

    def update(self, h, l, c):
        tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close)) if self.prev_close else h - l
        self._buf.append(tr)
        self._n += 1
        self.prev_close = c
        if self._n < self.period:
            self.atr = None
        elif self._n == self.period:
            self.atr = sum(self._buf) / self.period
        else:
            self.atr = (self.atr * (self.period - 1) + tr) / self.period
        return self.atr


class UTBot:
    def __init__(self, key=1.0, period=10):
        self.key = key
        self.atr = IncATR(period)
        self.trailing_stop = 0.0
        self.previous_source = None
        self.position = 0

    def update(self, o, h, l, c):
        atr = self.atr.update(h, l, c)
        prev_src = self.previous_source
        prev_stop = self.trailing_stop
        self.previous_source = c
        if atr is None or prev_src is None:
            return "green" if self.position == 1 else "red"
        loss = self.key * atr
        if c > prev_stop and prev_src > prev_stop:
            self.trailing_stop = max(prev_stop, c - loss)
        elif c < prev_stop and prev_src < prev_stop:
            self.trailing_stop = min(prev_stop, c + loss)
        elif c > prev_stop:
            self.trailing_stop = c - loss
        else:
            self.trailing_stop = c + loss
        if self.position == 0:
            self.position = 1 if c > self.trailing_stop else -1
        elif prev_src < prev_stop and c > prev_stop:
            self.position = 1
        elif prev_src > prev_stop and c < prev_stop:
            self.position = -1
        return "green" if self.position == 1 else "red"


def heikin_ashi(prev, o, h, l, c):
    ha_c = (o + h + l + c) / 4.0
    if prev is None:
        ha_o = (o + c) / 2.0
    else:
        ha_o = (prev[0] + prev[1]) / 2.0
    ha_h = max(h, ha_o, ha_c)
    ha_l = min(l, ha_o, ha_c)
    return (ha_o, ha_c, ha_h, ha_l)


def main():
    ut = UTBot(key=1.0, period=10)
    ha_prev = None
    ha_closes = deque(maxlen=11)
    # For the engine the LinReg plot = SMA(11) of HA closes. Compare that and
    # a true OLS regression line value at the last bar.
    print(f"{'IST':<12} {'HAclose':>10} {'LinRegSMA11':>12} {'UT':>6} {'HA>Lin':>6} {'BULLISH':>8}")
    for ts, o, h, l, c in TV_BARS:
        ha_o, ha_c, ha_h, ha_l = heikin_ashi(ha_prev, o, h, l, c)
        ha_prev = (ha_o, ha_c)
        ha_closes.append(ha_c)
        utcol = ut.update(o, h, l, c)
        lin = sum(ha_closes) / len(ha_closes) if len(ha_closes) >= 11 else None
        bull = (lin is not None and ha_c > lin and utcol == "green")
        ts_str = ist_str(ts)
        lin_s = f"{lin:.3f}" if lin is not None else "-"
        gt_s = str(ha_c > lin) if lin is not None else "-"
        print(f"{ts_str:<12} {ha_c:>10.3f} {lin_s:>12} {utcol:>6} {gt_s:>6} {str(bull):>8}")

    # TradingView displayed values for the last completed bar:
    print("\nTradingView study (Humble LinReg Candles) at last bar:")
    print("  LinReg Candles (HA close) = 24,228.583")
    print("  Plot (LinReg)              = 24,239.9275")


if __name__ == "__main__":
    main()