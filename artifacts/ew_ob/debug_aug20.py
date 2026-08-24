import sys
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
import opt_futures_quad as source
from artifacts.ew_ob.ew_ob_runner import _augment, make_option_resolver, resample_tf
from artifacts.ew_ob.ew_ob_engine import EWOBEngine, Bar, SESSION_START, SESSION_END

spot_all = source.load_spot()
opt_map = source.option_day_files("2026-08-19", "2026-08-20")
opt_map, spot_all = _augment(opt_map, spot_all)

eng = EWOBEngine(tol=0.5)
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
            w = eng.wave.armed_impulse
            print(f"ARM  day={day} arm_min={minute} arm_gi={b.gi} armed_gi={ag} "
                  f"dir={getattr(w, 'direction', '(consumed)')} "
                  f"imp_gi=[{getattr(w, 'start_gi', '?')}..{getattr(w, 'end_gi', '?')}] "
                  f"pos={len(eng.positions)} armed={eng.armed}")
            for c in eng.candidates:
                print(f"      tf={c.ob.tf} zone=({round(c.ob.lo,2)},{round(c.ob.hi,2)}) "
                      f"utop={round(c.untouched_top,2)} ubot={round(c.untouched_bot,2)} dead={c.dead}")
            if eng.pos is not None:
                print(f"      >>> open positions while armed: {[(p.entry_min, p.side) for p in eng.positions]}")
    eng.close_day()
print("TRADES:")
for t in eng.trades:
    print(f"  {t['date']} {t['exit_reason']:>3} {t['side']} entry {t['entry_min']}@{t['entry']} exit {t['exit_min']} pts {t['pts_net']:+.1f}")