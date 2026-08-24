"""Validate UT Bot run-extreme anchors vs the user's Aug 20 wave drawings.

Feeds the Pine-exact UT Bot (key=1.0, period=10, src=close, regular candles)
over ALL AMMU days sequentially (state continuous across days), tracks color
runs, and prints each run's extreme point for Aug 19-20.

Expected Aug 20 anchor minutes (from the drawings):
  L 573 H 586 L 593 H 597 L 603 H 616 L 621 H 623 L 638 H 660
  L 681 H 693 L 701 H 707 L 715 H 716 L 718 H 737 L 746
"""
import sys
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
import opt_futures_quad as source
from artifacts.ew_ob.ew_ob_engine import SESSION_START, SESSION_END, UTBot
from artifacts.ew_ob.ew_ob_runner import _augment

EXPECTED = {
    "2026-08-20": [(573, "L"), (586, "H"), (593, "L"), (597, "H"), (603, "L"),
                   (616, "H"), (621, "L"), (623, "H"), (638, "L"), (660, "H"),
                   (681, "L"), (693, "H"), (701, "L"), (707, "H"), (715, "L"),
                   (716, "H"), (718, "L"), (737, "H"), (746, "L")],
}


def main():
    spot_all = source.load_spot()
    _, spot_all = _augment({}, spot_all)
    ut = UTBot()
    run_color = None
    run_extreme = None      # (minute, price) for the run in progress
    runs = {}               # day -> list[(start_min, color, ext_min, ext_px)]
    for day in sorted(spot_all):
        spot = spot_all[day]
        day_runs = []
        run_color = None
        run_extreme = None
        for i in range(len(spot["min"])):
            minute = int(spot["min"][i])
            if minute < SESSION_START or minute > SESSION_END:
                continue
            h = float(spot["high"][i]); l = float(spot["low"][i])
            c = float(spot["close"][i])
            col = ut.update_close(c, h, l)
            if col == "none":
                continue
            if col == run_color:
                if col == "red" and l < run_extreme[1]:
                    run_extreme = (minute, l)
                    day_runs[-1][2] = minute
                    day_runs[-1][3] = l
                elif col == "green" and h > run_extreme[1]:
                    run_extreme = (minute, h)
                    day_runs[-1][2] = minute
                    day_runs[-1][3] = h
            else:
                run_color = col
                run_extreme = (minute, l if col == "red" else h)
                day_runs.append([minute, col, minute, run_extreme[1]])
        if run_extreme is not None and day_runs:
            day_runs[-1][2] = run_extreme[0]
            day_runs[-1][3] = run_extreme[1]
        runs[day] = day_runs

    for day, exp in EXPECTED.items():
        print(f"\n=== {day} (expected anchors: {exp}) ===")
        got = []
        for start, col, emin, epx in runs.get(day, []):
            kind = "L" if col == "red" else "H"
            got.append((emin, kind))
            print(f"  run {col:>5} start {start:>4}  extreme min={emin:>4} price={epx:.2f}")
        print(f"  -> anchors: {got}")
        missing = [e for e in exp if e not in got]
        extra = [g for g in got if g not in exp]
        print(f"  missing: {missing}  extra: {extra}")


if __name__ == "__main__":
    main()