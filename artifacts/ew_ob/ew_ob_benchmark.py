"""Seven-year causal/non-walk-forward EW-OB risk benchmark.

The walk-forward path uses expanding folds:
  train 2020-2022 -> OOS 2023
  train 2020-2023 -> OOS 2024
  train 2020-2024 -> OOS 2025
  train 2020-2025 -> OOS 2026

Each selected fold configuration is replayed from the beginning of its
training window through the OOS year with the same sequential engine used by
the live-parity runner. No OOS trades are used to select parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source
from artifacts.ew_ob.ew_ob_engine import RISK_MODE_ATR
from artifacts.ew_ob.ew_ob_runner import _augment, run_engine, summarize

START = "2020-01-01"
END = "2026-08-20"
OUT = Path("artifacts/ew_ob") / "seven_year_risk_benchmark.json"
MULTIPLIERS = (1.0, 2.0, 3.0, 4.0, 5.0)
FOLDS = ("2023", "2024", "2025", "2026")


def load_data():
    spot = source.load_spot()
    options = source.option_day_files(START, END)
    options, spot = _augment(options, spot)
    days = sorted(d for d in set(spot) & set(options) if START <= d <= END)
    return spot, options, days


def run_config(spot, options, days, sl_mult, tp_mult):
    trades = run_engine(
        spot,
        options,
        days,
        tol=0.5,
        sl_mult=sl_mult,
        tp_pts=60.0,
        risk_mode=RISK_MODE_ATR,
        tp_atr_mult=tp_mult,
    )
    return trades, summarize(trades)


def grid_rows(spot, options, days):
    rows = []
    for sl_mult in MULTIPLIERS:
        for tp_mult in MULTIPLIERS:
            started = time.time()
            trades, stats = run_config(spot, options, days, sl_mult, tp_mult)
            rows.append({
                "sl_atr": sl_mult,
                "tp_atr": tp_mult,
                "stats": stats,
                "seconds": round(time.time() - started, 2),
                "trades": trades,
            })
            print(
                f"NW sl={sl_mult:.0f} tp={tp_mult:.0f} "
                f"trades={stats['trades']} pts={stats['pts']:+.2f} "
                f"rs={stats['rs']:+.2f}",
                flush=True,
            )
    return rows


def choose_train(rows):
    return max(rows, key=lambda row: (row["stats"]["pts"], row["stats"]["rs"]))


def smoke_gate(spot, options, days):
    """Run every grid point on the short anchors before any long job."""
    anchor_days = [d for d in days if d >= "2026-08-18"][:3]
    reference_days = [d for d in days if d.startswith("2020")][:5]
    if not anchor_days or not reference_days:
        raise RuntimeError("smoke gate data windows are incomplete")

    checks = []
    for sl_mult in MULTIPLIERS:
        for tp_mult in MULTIPLIERS:
            _, anchor_stats = run_config(spot, options, anchor_days, sl_mult, tp_mult)
            _, reference_stats = run_config(spot, options, reference_days, sl_mult, tp_mult)
            if anchor_stats["trades"] == 0 or reference_stats["trades"] == 0:
                raise RuntimeError(
                    f"smoke gate failed for ATR {sl_mult:.0f}/{tp_mult:.0f}: "
                    "no trades in an anchor window"
                )
            checks.append({
                "sl_atr": sl_mult,
                "tp_atr": tp_mult,
                "anchor": anchor_stats,
                "reference": reference_stats,
            })
    print(
        f"SMOKE GATE passed: {len(checks)} ATR combinations, "
        f"anchor={anchor_days[0]}..{anchor_days[-1]}, "
        f"reference={reference_days[0]}..{reference_days[-1]}",
        flush=True,
    )
    return checks


def walk_forward(spot, options, days):
    folds = []
    stitched_trades = []
    for oos_year in FOLDS:
        train_days = [d for d in days if d[:4] < oos_year]
        oos_days = [d for d in days if d[:4] == oos_year]
        if not train_days or not oos_days:
            continue

        train_rows = []
        for sl_mult in MULTIPLIERS:
            for tp_mult in MULTIPLIERS:
                started = time.time()
                trades, stats = run_config(spot, options, train_days, sl_mult, tp_mult)
                train_rows.append({
                    "sl_atr": sl_mult,
                    "tp_atr": tp_mult,
                    "stats": stats,
                    "seconds": round(time.time() - started, 2),
                    "trades": trades,
                })
        selected = choose_train(train_rows)
        sl_mult = selected["sl_atr"]
        tp_mult = selected["tp_atr"]

        # Replay from the beginning of the training window through OOS. This
        # preserves detector, queue, OB, and position state at the fold edge.
        replay_days = train_days + oos_days
        replay_trades, _ = run_config(spot, options, replay_days, sl_mult, tp_mult)
        oos_trades = [t for t in replay_trades if t["date"][:4] == oos_year]
        oos_stats = summarize(oos_trades)
        folds.append({
            "oos_year": oos_year,
            "train_start": train_days[0],
            "train_end": train_days[-1],
            "oos_start": oos_days[0],
            "oos_end": oos_days[-1],
            "selected_sl_atr": sl_mult,
            "selected_tp_atr": tp_mult,
            "train_stats": selected["stats"],
            "oos_stats": oos_stats,
            "oos_trades": oos_trades,
        })
        stitched_trades.extend(oos_trades)
        print(
            f"WF oos={oos_year} selected={sl_mult:.0f}/{tp_mult:.0f} "
            f"train_pts={selected['stats']['pts']:+.2f} "
            f"oos_pts={oos_stats['pts']:+.2f} oos_rs={oos_stats['rs']:+.2f}",
            flush=True,
        )

    return {
        "folds": folds,
        "stitched_oos": summarize(stitched_trades),
        "trades": stitched_trades,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("nonwf", "wf", "both"), default="both")
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    started = time.time()
    spot, options, days = load_data()
    smoke = smoke_gate(spot, options, days)
    if args.smoke_only:
        return
    result = {
        "config": {
            "start": START,
            "end": END,
            "days": len(days),
            "multipliers": list(MULTIPLIERS),
            "risk_mode": RISK_MODE_ATR,
            "selection_metric": "net_points_then_net_rs",
            "causal_live_parity": True,
        },
        "non_walk_forward": None,
        "walk_forward": None,
        "smoke": smoke,
    }

    if args.mode in ("nonwf", "both"):
        rows = grid_rows(spot, options, days)
        best = choose_train(rows)
        result["non_walk_forward"] = {
            "grid": rows,
            "best": best,
        }
        print(
            f"NW BEST sl={best['sl_atr']:.0f} tp={best['tp_atr']:.0f} "
            f"pts={best['stats']['pts']:+.2f} rs={best['stats']['rs']:+.2f}",
            flush=True,
        )

    if args.mode in ("wf", "both"):
        result["walk_forward"] = walk_forward(spot, options, days)

    result["seconds"] = round(time.time() - started, 2)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
