"""Debug packed spike at minute 698 on 2020-01-02 for CHAMPION params."""

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files
from f6_hybrid.raw_features import (
    base_key_for,
    build_day_base_state,
    materialize_signal_state,
)
from f6_hybrid.incremental import simulate_day_signal_state
from f6_hybrid.packed import _DayBundle, _pack_signals, run_bundle_candidates

REF_MINUTE = 698


def main():
    spot_all = load_spot()
    files = option_files("2020-01-01", "2020-01-07")
    days = sorted(set(files) & set(spot_all))[:5]
    day, fpath = days[1], files[days[1]]
    fprev = files[days[0]]
    spot = spot_all[day]
    params = dict(grid.CHAMPION)

    key = base_key_for(day, fprev, params)
    base = build_day_base_state(day, str(fpath), str(fprev), params, spot)
    state = materialize_signal_state(base, params["f6_s4_thresh"], params["f6_s1_thresh"])

    ref_trades = simulate_day_signal_state(state, params)
    print("reference trades:", len(ref_trades), "first:", ref_trades[0] if ref_trades else None)

    sig_at_ref = state.pmtrig.get(REF_MINUTE, [])
    print(f"pmtrig[{REF_MINUTE}] count:", len(sig_at_ref))
    for index, signal in enumerate(sig_at_ref):
        side, strike, symbol, px, is_rev, tf, sl_pts, tp_pts, atr = signal
        spot_px = grid.latest_spot(spot, REF_MINUTE)
        atm = int(round(spot_px / 50) * 50)
        atk = atm + (-100 if side == "CE" else 100)
        print(
            f"  sig[{index}] side={side} strike={strike} sym={symbol} atk={atk} "
            f"spot_px={spot_px} atm={atm} flags=({'SPOT_MATCH' if atk == strike else 'NO'}, "
            f"{'SYM_IN_SLICES' if symbol in state.slices else 'MISSING'})"
        )

    bundle = _DayBundle(state, base)
    minutes, sides, strikes, revs, tfs, sls, tps, atrs = _pack_signals(state)
    print("\npacked signals at", REF_MINUTE, ":")
    for index in range(len(minutes)):
        if minutes[index] == REF_MINUTE:
            print(
                f"  sig[{index}] side={sides[index]} strike={strikes[index]} "
                f"rev={revs[index]} atr={atrs[index]}"
            )

    packed = run_bundle_candidates(bundle, [state], [params], False)[0]
    print("\npacked trades:", len(packed), "first:", packed[0] if packed else None)

    print("\n=== full sequence diff (day", day, ") ===")
    longer = max(len(ref_trades), len(packed))
    for index in range(longer):
        ref = ref_trades[index] if index < len(ref_trades) else None
        pck = packed[index] if index < len(packed) else None
        marker = "  " if ref == pck else "**"
        print(marker, "idx", index, "\n    ref:", ref, "\n    pck:", pck)

    target = f"{state.prefix}{strikes[0]}{'CE' if sides[0] == 0 else 'PE'}" if len(sides) else ""
    probe = strikes[0] if len(strikes) else 0
    rel = (probe - bundle.base_strike) // 50
    print(
        f"\nslot probe: strike={probe} rel={rel} side={sides[0] if len(sides) else '?'} "
        f"slot={bundle.slot_map[rel * 2 + (sides[0] if len(sides) else 0)]}"
    )
    print("base_strike", bundle.base_strike, "n_strikes", bundle.n_strikes)


if __name__ == "__main__":
    main()