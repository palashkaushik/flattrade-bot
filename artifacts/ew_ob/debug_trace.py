import sys
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
import opt_futures_quad as source
from artifacts.ew_ob.ew_ob_runner import _augment, make_option_resolver, resample_tf
from artifacts.ew_ob.ew_ob_engine import EWOBEngine, Bar, SESSION_START, SESSION_END, RED, GREEN

spot_all = source.load_spot()
opt_map = source.option_day_files("2026-08-19", "2026-08-20")
opt_map, spot_all = _augment(opt_map, spot_all)

eng = EWOBEngine(tol=0.5)
eng.resolve_option = make_option_resolver(opt_map)

det = eng.wave
orig_find = det._find_wave
def traced_find(start_idx, boundary):
    res = orig_find(start_idx, boundary)
    if res is not None and det.pos in (0, 1, 2, 3, 4, 5, 6):
        print(f"    [W{det.pos+1 if det.pos < 5 else ('A' if det.pos==5 else 'B')}] found {res} boundary={'G' if boundary==GREEN else 'R'}")
    return res
det._find_wave = traced_find

orig_check = det._check_condition
def traced_check():
    res = orig_check()
    w = det.waves
    o = det.origin
    o_low = o.low if o else None
    o_high = o.high if o else None
    print(f"    [cond] -> {res} "
          f"W1p {w[0].peak} W3p {w[2].peak} W5p {w[4].peak} "
          f"W1t {w[0].trough} W3t {w[2].trough} W5t {w[4].trough} "
          f"o_low {o_low} o_high {o_high} "
          f"W2t {w[1].trough} W4t {w[3].trough} W2p {w[1].peak} W4p {w[3].peak}")
    return res
det._check_condition = traced_check

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
        if day == "2026-08-20" and 610 <= minute <= 770:
            print(f"min={minute} gi={b.gi} O={b.open} H={b.high} L={b.low} C={b.close} "
                  f"col={'G' if b.color==GREEN else 'R'} pos={det.pos} nwaves={len(det.waves)}")
        eng.feed(b)
        gi += 1
    eng.close_day()