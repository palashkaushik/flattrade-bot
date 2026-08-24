import sys
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
import opt_futures_quad as source
from artifacts.ew_ob.ew_ob_runner import _augment, make_option_resolver, resample_tf
from artifacts.ew_ob.ew_ob_engine import (
    Bar, SESSION_START, SESSION_END, RED, GREEN, WaveDetector, EWOBEngine,
)

class BacktrackDetector(WaveDetector):
    """WaveDetector that, on condition-1 failure, retries later W1 starts
    inside the failed attempt's range instead of flooring past its W5 end."""

    def __init__(self):
        super().__init__()
        self._log: list[Bar] = []

    def feed(self, bar: Bar):
        self._log.append(bar)
        super().feed(bar)

    def _retry_w1(self, f_w1_start_gi: int):
        bars = [b for b in self._log if b.gi >= f_w1_start_gi - 1]
        self.last_consumed_gi = f_w1_start_gi - 2
        self.pos = 0
        self.waves = []
        self.origin = None
        self.impulse = None
        self.cur = list(bars)
        self._w1_scan_idx = 2

    def _advance(self):
        while True:
            if self.pos == 0:
                res = self._find_wave(self._w1_scan_idx, RED)
                if res is None:
                    return
                s, e = res
                if s < 1:
                    self._w1_scan_idx = s + 1
                    continue
                self.origin = self.cur[s - 1]
                self.waves = [self._wave_from(self.cur[s:e + 1])]
                self.impulse_start_gi = self.cur[s].gi
                self.last_consumed_gi = self.cur[e].gi
                self.cur = self.cur[e + 1:]
                self.pos = 1
            else:
                res = self._find_wave(0, self._boundary(self.pos))
                if res is None:
                    return
                s, e = res
                wave = self._wave_from(self.cur[s:e + 1])
                self.waves.append(wave)
                self.last_consumed_gi = self.cur[e].gi
                self.cur = self.cur[e + 1:]
                self.pos += 1
                if self.pos == 5:
                    direction = self._check_condition()
                    if direction is None:
                        f_w1 = self.waves[0].start_gi
                        self._retry_w1(f_w1)
                        continue
                    self.impulse = Impulse(
                        direction=direction,
                        start_gi=self.impulse_start_gi,
                        end_gi=wave.end_gi,
                        w1=self.waves[0], w2=self.waves[1], w3=self.waves[2],
                        w4=self.waves[3], w5=self.waves[4],
                    )
                elif self.pos == 7:
                    if self.impulse is not None:
                        self.armed = True
                        self.armed_gi = wave.end_gi
                        self.armed_impulse = self.impulse
                    self._reset()
        # end while


from artifacts.ew_ob.ew_ob_engine import Impulse  # noqa: E402

spot_all = source.load_spot()
opt_map = source.option_day_files("2026-08-19", "2026-08-20")
opt_map, spot_all = _augment(opt_map, spot_all)

class BacktrackEngine(EWOBEngine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.wave = BacktrackDetector()

eng = BacktrackEngine(tol=0.5)
eng.resolve_option = make_option_resolver(opt_map)
last_armed_gi = None
gi = 0
for day in ["2026-08-19", "2026-08-20"]:
    spot = spot_all[day]
    day_start_gi = gi
    for tf in (1, 2, 3, 5):
        highs, lows, gis = resample_tf(spot, tf)
        eng.obs.feed_tf_bars(tf, highs, lows, gis + day_start_gi)
    for i in range(len(spot["min"])):
        minute = int(spot["min"][i])
        if minute < SESSION_START or minute > SESSION_END:
            continue
        b = Bar(gi=gi, day=day, minute=minute,
                open=float(spot["open"][i]), high=float(spot["high"][i]),
                low=float(spot["low"][i]), close=float(spot["close"][i]))
        eng.feed(b)
        gi += 1
        ag = eng.wave.armed_gi
        if ag is not None and ag != last_armed_gi:
            last_armed_gi = ag
            imp = eng.wave.armed_impulse
            print(f"ARM day={day} min={minute} arm_gi={b.gi} armed_gi={ag} "
                  f"dir={getattr(imp, 'direction', '?')} imp=[{getattr(imp, 'start_gi', '?')}..{getattr(imp, 'end_gi', '?')}]")
    eng.close_day()
print("TRADES:")
for t in eng.trades:
    print(f"  {t['date']} {t['exit_reason']:>3} {t['side']} entry {t['entry_min']}@{t['entry']} exit {t['exit_min']} pts {t['pts_net']:+.1f}")