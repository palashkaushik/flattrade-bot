"""Elder Impulse fib-leg engine for Smart Fib (research -> live port).

Installs over artifacts.f6_hybrid.marni_fib_core_combo_cache so BOTH the
research backtests and the live B17 strategy run the IDENTICAL patched
classes (single source of truth for parity).

Leg rule (user spec):
  - Bars colored by Elder Impulse (green/red/blue) instead of UT Bot.
  - Leg = contiguous run of >= 4 same-color candles (green OR red only);
    no bookend requirement; boundary candles excluded from the fib range.
  - Span (run high-low) >= 20 pts (ELDER_MIN_SPAN).
"""

from flattrade_bot.indicators.elder import IncrementalElderImpulse

ELDER_MIN_SPAN = 20.0
ELDER_MIN_RUN = 4


class ElderColorState:
    """Elder Impulse colorer with Pine-style always-a-color semantics."""

    def __init__(self, use_heikin_ashi=False):
        self.imp = IncrementalElderImpulse()

    def update(self, candle):
        return self.imp.update(float(candle.close))


class ElderRunPattern:
    """Contiguous same-color run detector (green/red only).

    Wave top/bottom anchored at the swing extreme including the
    immediate boundary candle (user spec: 09:40 top / 09:46 bottom
    for the 5-red run 09:42-09:46) so the fib spans the visual
    swing, not just the monochrome run.
    """

    def __init__(self, name, middle, orientation, min_middle=ELDER_MIN_RUN):
        self.name = name
        self.middle = middle
        self.orientation = orientation
        self.min_middle = min_middle
        self.run = []
        self.recent = []  # last 2 candles before run (covers 09:40 top for 09:42-09:46 red impulse per user)
        self.run_prev_high = None
        self.run_prev_low = None
        self.run_prev_min = None
        self._queued = None  # dual 0.618-1 top+bottom for single wave

    def update(self, candle, color):
        if self._queued is not None:
            q = self._queued
            self._queued = None
            self.recent.append(candle)
            if len(self.recent) > 2:
                self.recent.pop(0)
            if color == self.middle and not self.run:
                recent_high = max(c.high for c in self.recent) if self.recent else None
                recent_low = min(c.low for c in self.recent) if self.recent else None
                recent_min = min(c.minute for c in self.recent) if self.recent else None
                self.run_prev_high = recent_high
                self.run_prev_low = recent_low
                self.run_prev_min = recent_min
                self.run.append(candle)
            return q
        completed = None
        if color == self.middle:
            if not self.run:
                recent_high = max(c.high for c in self.recent) if self.recent else None
                recent_low = min(c.low for c in self.recent) if self.recent else None
                recent_min = min(c.minute for c in self.recent) if self.recent else None
                self.run_prev_high = recent_high
                self.run_prev_low = recent_low
                self.run_prev_min = recent_min
            self.run.append(candle)
        else:
            if len(self.run) >= self.min_middle:
                run_high = max(c.high for c in self.run)
                run_low = min(c.low for c in self.run)
                fib_high = max(run_high, self.run_prev_high) if self.run_prev_high is not None else run_high
                fib_low = min(run_low, self.run_prev_low) if self.run_prev_low is not None else run_low
                span = fib_high - fib_low
                if span >= ELDER_MIN_SPAN:
                    completed = {
                        "pattern": self.name,
                        "start_minute": self.run_prev_min if self.run_prev_min is not None else self.run[0].minute,
                        "fib_high": fib_high,
                        "fib_low": fib_low,
                        "orientation": self.orientation,
                        "completion_minute": candle.minute,
                    }
                    opposite = "low_to_high" if self.orientation == "high_to_low" else "high_to_low"
                    self._queued = {
                        "pattern": self.name + "_dual",
                        "start_minute": completed["start_minute"],
                        "fib_high": fib_high,
                        "fib_low": fib_low,
                        "orientation": opposite,
                        "completion_minute": candle.minute,
                    }
            self.run = []
            self.run_prev_high = None
            self.run_prev_low = None
            self.run_prev_min = None
        self.recent.append(candle)
        if len(self.recent) > 2:
            self.recent.pop(0)
        return completed


def install_elder_engine():
    """Patches smart_core.UTSwingFeed in-place; idempotent."""
    import artifacts.f6_hybrid.marni_fib_core_combo_cache as smart_core

    if getattr(smart_core.UTSwingFeed, "__name__", "") == "ElderSwingFeed":
        return smart_core.UTSwingFeed

    class ElderSwingFeed(smart_core.UTSwingFeed):
        """UTSwingFeed with the Elder colorer and run-based leg patterns."""

        def __init__(self, patterns, **kwargs):
            kwargs.pop("patterns", None)
            super().__init__([], **kwargs)
            self.ut = ElderColorState()
            self.patterns = [
                ElderRunPattern("elder_bull", "green", "low_to_high"),
                ElderRunPattern("elder_bear", "red", "high_to_low"),
            ]

    smart_core.UTSwingFeed = ElderSwingFeed
    return ElderSwingFeed
