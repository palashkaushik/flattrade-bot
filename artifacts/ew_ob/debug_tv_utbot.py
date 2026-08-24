"""Definitive UT Bot verification: run the Pine-exact port on TradingView's OWN
Aug 20 1m data (fetched via MCP), compare run extremes vs the user's anchors,
and diff TV vs AMMU OHLC for the anchor bars."""
import json
import sys
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
import opt_futures_quad as source
from artifacts.ew_ob.ew_ob_engine import SESSION_START, SESSION_END, UTBot
from artifacts.ew_ob.ew_ob_runner import _augment

TV_FILE = r"C:\Users\user\.local\share\opencode\tool-output\tool_020a8a624001XRrvAHdaejdYgV"
EXPECTED = [(573, "L"), (586, "H"), (593, "L"), (597, "H"), (603, "L"),
            (616, "H"), (621, "L"), (623, "H"), (638, "L"), (660, "H"),
            (681, "L"), (693, "H"), (701, "L"), (707, "H"), (715, "L"),
            (716, "H"), (718, "L"), (737, "H"), (746, "L")]
DAY0 = 1787197500  # 2026-08-20 09:15 IST


def minute_of(t):
    return 555 + (t - DAY0) // 60


def main():
    with open(TV_FILE, encoding="utf-8") as f:
        tv = json.load(f)
    bars = [b for b in tv["bars"] if b["time"] >= DAY0]

    ut = UTBot()
    runs = []
    run_color = None
    run_extreme = None
    for b in bars:
        mn = minute_of(b["time"])
        col = ut.update_close(b["close"], b["high"], b["low"])
        if col == "none":
            continue
        if col == run_color:
            if col == "red" and b["low"] < run_extreme[1]:
                run_extreme = (mn, b["low"])
                runs[-1][2], runs[-1][3] = mn, b["low"]
            elif col == "green" and b["high"] > run_extreme[1]:
                run_extreme = (mn, b["high"])
                runs[-1][2], runs[-1][3] = mn, b["high"]
        else:
            run_color = col
            run_extreme = (mn, b["low"] if col == "red" else b["high"])
            runs.append([mn, col, mn, run_extreme[1]])
    if runs:
        runs[-1][2], runs[-1][3] = run_extreme

    print("=== TradingView data -> UT Bot runs (Aug 20) ===")
    got = []
    for start, col, emin, epx in runs:
        kind = "L" if col == "red" else "H"
        got.append((emin, kind))
        print(f"  run {col:>5} start {start:>4}  extreme min={emin:>4} price={epx:.2f}")
    print(f"  -> anchors: {got}")
    missing = [e for e in EXPECTED if e not in got]
    extra = [g for g in got if g not in EXPECTED]
    print(f"  missing: {missing}  extra: {extra}")

    # TV vs AMMU diff at the anchor-adjacent bars
    spot_all = source.load_spot()
    _, spot_all = _augment({}, spot_all)
    spot = spot_all["2026-08-20"]
    ammu = {}
    for i in range(len(spot["min"])):
        m = int(spot["min"][i])
        ammu[m] = (float(spot["open"][i]), float(spot["high"][i]),
                   float(spot["low"][i]), float(spot["close"][i]))
    tvb = {minute_of(b["time"]): (b["open"], b["high"], b["low"], b["close"]) for b in bars}

    print("\n=== AMMU vs TV OHLC (anchor windows) ===")
    for mn in sorted(set(list(ammu) + list(tvb))):
        if mn < 555 or mn > 930:
            continue
        a = ammu.get(mn)
        t = tvb.get(mn)
        if a is None or t is None:
            continue
        if max(abs(a[k] - t[k]) for k in range(4)) > 0.01:
            flag = ""
            if mn in (564, 573, 616, 617, 638, 660, 681, 693, 701, 707, 715, 716, 718, 737, 746, 757):
                flag = "  <<< anchor bar"
            print(f"  min {mn:>4}  AMMU o{max(a[0],0):9.2f} h{a[1]:9.2f} l{a[2]:9.2f} c{a[3]:9.2f}  "
                  f"TV o{t[0]:9.2f} h{t[1]:9.2f} l{t[2]:9.2f} c{t[3]:9.2f}{flag}")


if __name__ == "__main__":
    main()