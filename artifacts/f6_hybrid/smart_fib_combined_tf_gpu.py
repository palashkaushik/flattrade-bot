"""Combined-TF Smart Fib champion backtest.

Union-of-TF signal stream: ANY of the 4 timeframes (1m/2m/3m/5m, bias filter
= 5x TF) firing a champion signal enters one merged stream per day; a single
global position trades the merged stream (simulate timeframe_mode="combined",
the canonical CPU oracle). Exits = champion (0.786/1.13) for the non-WF run;
the expanding WFO (4 folds, mirroring smart_fib_finesweep_fixed40) re-selects
exits per fold from a 9-config grid. TF attribution: each trade is tagged by
the source TF of its entry event; "trades at non-1m minutes" = entries at
minutes with no 1m signal at all (the true incremental value of the union).

Validation: --timeframes 1 (full window) must reproduce the 1m champion
byte-identically (12,380 tr / 70.93% / +25,799.35 pts / Rs 1,181,757.75).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid import marni_fib_core_combo_cache as smart_core
from artifacts.f6_hybrid import smart_fib_optimus_gpu as optimus
from artifacts.f6_hybrid import smart_fib_optimus_grid_gpu as grid
from artifacts.f6_hybrid.marni_fib_backtest import LOT_SIZE, simulate

CHAMPION = dict(
    min_span=15.0,
    touch_buffer=0.5,
    setup_max_age=45,
    zone_start=0.5,
    zone_end=0.786,
    s1_k_period=12,
    s1_d_period=4,
)
FIXED_COST = 40.0
TARGETS = (0.618, 0.786, 1.0)
STOPS = (1.05, 1.13, 1.272)
EXIT_GRID = [
    {
        "target_level": float(t),
        "stop_level": float(s),
        "fallback_target_level": 0.0,
        "option_point_threshold": 5.0,
    }
    for t in TARGETS
    for s in STOPS
]
CHAMPION_EXIT_INDEX = 4  # (0.786, 1.13)
CHAMPION_EXITS = EXIT_GRID[CHAMPION_EXIT_INDEX]

FOLDS = (
    ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2020-01-01", "2024-12-31", "2025-01-01", "2026-05-05"),
)

KNOWN_1M_SMOKE = dict(trades=18, win_rate=83.33, net_points=24.35, net_rs=862.75)
KNOWN_1M_FULL = dict(
    trades=12380,
    win_rate=70.93,
    net_points=25799.35,
    net_rs=1181757.75,
    max_drawdown_points=127.21,
    profit_factor=18.7351,
)


def _extract_combined_day(task: tuple) -> dict:
    """Worker: extract champion events for every TF, merge, simulate all exits."""
    data_root, start, end, day, timeframes = task
    adapter = grid.PolarsHistoricalDataAdapter(
        data_root,
        start=start,
        end=end,
        cache_days=8,
    )
    per_tf_selected = {}
    per_tf_events = {}
    merged_payload = None
    for tf in timeframes:
        payload = smart_core.extract_day_events(
            day,
            cache_loader=adapter.load_day_cache,
            bar_minutes=tf,
            filter_period=5 * tf,
            debug=False,
            **CHAMPION,
        )
        if not payload:
            raise RuntimeError(f"no Smart Fib cache payload for {day} tf={tf}")
        selected, raw_count = optimus._select_day_events(payload)
        per_tf_selected[tf] = len(selected)
        per_tf_events[tf] = selected
        if tf == 1:
            merged_payload = payload

    merged = []
    for tf in timeframes:
        for signal in per_tf_events[tf]:
            merged.append((signal["minute"], tf, signal))
    merged.sort(key=lambda item: (item[0], item[1]))
    merged_events = []
    source_tf_by_minute = {}
    seen = set()
    for minute, tf, signal in merged:
        key = (minute, signal["side"], signal["symbol"])
        if key in seen:
            continue
        seen.add(key)
        merged_events.append({**signal, "timeframe": "combined"})
        source_tf_by_minute.setdefault(minute, tf)

    trades_by_cfg = {}
    for index, exits in enumerate(EXIT_GRID):
        trades_by_cfg[index] = simulate(
            merged_events,
            merged_payload["bars"],
            merged_payload["index_bars"],
            merged_payload["spot"],
            "combined",
            exits["target_level"],
            exits["stop_level"],
            concurrent=False,
            option_point_threshold=exits["option_point_threshold"],
            fallback_target_level=exits["fallback_target_level"],
            fixed_cost_per_trade=FIXED_COST,
        )
    return {
        "day": day,
        "per_tf_selected": per_tf_selected,
        "merged_count": len(merged_events),
        "trades_by_cfg": trades_by_cfg,
        "source_tf_by_minute": source_tf_by_minute,
    }


def _stats(trades: list[dict]) -> dict:
    wins = sum(1 for t in trades if t["rs_net"] > 0)
    losses = sum(1 for t in trades if t["rs_net"] <= 0)
    gross_win_pts = sum(t["points"] for t in trades if t["points"] > 0)
    gross_loss_pts = abs(sum(t["points"] for t in trades if t["points"] <= 0))
    return {
        "trades": len(trades),
        "wins": wins,
        "win_rate": round(100.0 * wins / len(trades), 2) if trades else 0.0,
        "net_points": round(sum(t["points"] for t in trades), 2),
        "net_rs": round(sum(t["rs_net"] for t in trades), 2),
        "fees_rs": round(sum(t["fee"] for t in trades), 2),
        "profit_factor": round(gross_win_pts / gross_loss_pts, 4) if gross_loss_pts else (float("inf") if gross_win_pts else 0.0),
    }


def _daily_series(trades: list[dict], days: list[str]) -> dict[str, dict]:
    by_day = {day: [] for day in days}
    for t in trades:
        day = t.get("day", "?")
        if day in by_day:
            by_day[day].append(t)
    series = {}
    for day, day_trades in by_day.items():
        cum = 0.0
        peak = 0.0
        trough = 0.0
        for t in sorted(day_trades, key=lambda x: (x["exit_min"], x["entry_min"])):
            cum += t["rs_net"]
            peak = max(peak, cum)
            trough = min(trough, cum)
        series[day] = {
            "net_rs": cum,
            "peak_rs": peak,
            "trough_rs": trough,
        }
    return series


def _stitch_drawdown(daily_series: dict[str, dict]) -> float:
    equity = 0.0
    global_peak = 0.0
    max_dd = 0.0
    for day in daily_series:
        series = daily_series[day]
        day_start = equity
        global_peak = max(global_peak, day_start + series["peak_rs"])
        max_dd = max(max_dd, global_peak - (day_start + series["trough_rs"]))
        equity += series["net_rs"]
    return max_dd


def _evaluate_window(
    day_results: list[dict],
    days: list[str],
    config_index: int,
) -> dict:
    trades = []
    for result in day_results:
        day = result["day"]
        for t in result["trades_by_cfg"][config_index]:
            t["day"] = day
        trades.extend(result["trades_by_cfg"][config_index])
    stats = _stats(trades)
    daily = _daily_series(trades, days)
    stats["max_drawdown_points"] = round(_stitch_drawdown(daily) / LOT_SIZE, 2)
    stats["daily"] = daily
    return stats


def _attribution(day_results: list[dict], config_index: int) -> dict:
    source = {1: 0, 2: 0, 3: 0, 5: 0}
    non_1m_minutes = 0
    total = 0
    for result in day_results:
        for t in result["trades_by_cfg"][config_index]:
            total += 1
            tf = result["source_tf_by_minute"].get(t["entry_min"], 0)
            source[tf] = source.get(tf, 0) + 1
            if tf != 1:
                non_1m_minutes += 1
    return {"trades": total, "by_source_tf": source, "at_non_1m_minutes": non_1m_minutes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=r"C:\Users\user\Desktop\nifty50 data")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--timeframes", default="1,2,3,5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dd-penalty", type=float, default=0.50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        default="artifacts/f6_hybrid/smart_fib_combined_tf_full.json",
    )
    args = parser.parse_args(argv)

    timeframes = tuple(int(x) for x in args.timeframes.split(","))
    if not timeframes or any(tf not in (1, 2, 3, 5) for tf in timeframes):
        raise SystemExit(f"invalid --timeframes: {args.timeframes}")

    adapter = grid.PolarsHistoricalDataAdapter(args.data_root, start=args.start, end=args.end)
    days = adapter.available_days(args.start, args.end)
    if not days:
        raise SystemExit("no days available")
    if args.smoke:
        days = days[:5]

    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tasks = [
        (str(adapter.data_root), adapter.start, adapter.end, day, timeframes)
        for day in days
    ]
    workers = max(1, min(8, args.workers))
    day_results: list[dict] = []
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=grid._init_grid_worker,
            initargs=(str(adapter.data_root), adapter.start, adapter.end),
        ) as pool:
            for count, raw in enumerate(pool.map(_extract_combined_day, tasks, chunksize=1), start=1):
                day_results.append(raw)
                if count == 1 or count % 25 == 0 or count == len(tasks):
                    print(f"[COMBINED PREP] {count}/{len(tasks)} last={raw['day']}", flush=True)
    else:
        for count, task in enumerate(tasks, start=1):
            raw = _extract_combined_day(task)
            day_results.append(raw)
            if count == 1 or count % 25 == 0 or count == len(tasks):
                print(f"[COMBINED PREP] {count}/{len(tasks)} last={raw['day']}", flush=True)

    prep_seconds = time.perf_counter() - started
    print(
        f"[COMBINED PREP] complete {prep_seconds:.3f}s N={len(day_results)} "
        f"events_per_day={sum(r['merged_count'] for r in day_results) / max(1, len(day_results)):.1f}",
        flush=True,
    )

    non_wf = _evaluate_window(day_results, days, CHAMPION_EXIT_INDEX)
    print(
        f"[NON-WF champion exits {CHAMPION_EXITS['target_level']}/{CHAMPION_EXITS['stop_level']}] "
        f"trades={non_wf['trades']} WR={non_wf['win_rate']:.2f} net={non_wf['net_points']:+.2f} "
        f"net_rs={non_wf['net_rs']:+,.2f} DD={non_wf['max_drawdown_points']:.2f} "
        f"PF={non_wf['profit_factor']:.4f}",
        flush=True,
    )

    if args.timeframes == "1" and len(days) == 5:
        ok = (
            non_wf["trades"] == KNOWN_1M_SMOKE["trades"]
            and abs(non_wf["net_rs"] - KNOWN_1M_SMOKE["net_rs"]) < 0.01
        )
        print(f"[VALIDATION 1m smoke] {'PASS' if ok else 'FAIL'} vs known {KNOWN_1M_SMOKE}")
        if not ok:
            raise SystemExit("1m smoke validation FAILED")

    full_table = []
    for index, exits in enumerate(EXIT_GRID):
        stats = _evaluate_window(day_results, days, index)
        stats["config"] = exits
        stats["attribution"] = _attribution(day_results, index)
        full_table.append(stats)

    if args.timeframes == "1" and not args.smoke:
        champion_row = full_table[CHAMPION_EXIT_INDEX]
        ok = (
            champion_row["trades"] == KNOWN_1M_FULL["trades"]
            and abs(champion_row["net_points"] - KNOWN_1M_FULL["net_points"]) < 0.01
            and abs(champion_row["net_rs"] - KNOWN_1M_FULL["net_rs"]) < 0.01
            and abs(champion_row["max_drawdown_points"] - KNOWN_1M_FULL["max_drawdown_points"]) < 0.01
        )
        print(f"[VALIDATION 1m full] {'PASS' if ok else 'FAIL'} vs known {KNOWN_1M_FULL}")
        if not ok:
            raise SystemExit("1m full validation FAILED")

    stitched = None
    fold_results = []
    if not args.smoke:
        best_train = [None] * len(FOLDS)
        for fold_index, (train_start, train_end, val_start, val_end) in enumerate(FOLDS):
            train_days = [d for d in days if train_start <= d <= train_end]
            train_results = [r for r in day_results if train_start <= r["day"] <= train_end]
            winners = []
            for index, exits in enumerate(EXIT_GRID):
                stats = _evaluate_window(train_results, train_days, index)
                stats["score"] = round(
                    stats["net_points"] - args.dd_penalty * stats["max_drawdown_points"], 4
                )
                winners.append((stats["score"], stats["net_points"], index, stats))
            winners.sort(key=lambda row: (row[0], row[1]), reverse=True)
            winner = winners[0]
            best_train[fold_index] = dict(
                fold=fold_index + 1,
                config=EXIT_GRID[winner[2]],
                score=winner[0],
                stats=winners[0][3],
            )
            val_days = [d for d in days if val_start <= d <= val_end]
            val_results = [r for r in day_results if val_start <= r["day"] <= val_end]
            validation = _evaluate_window(val_results, val_days, winner[2])
            validation["fold"] = fold_index + 1
            validation["config"] = EXIT_GRID[winner[2]]
            validation["train_selection"] = best_train[fold_index]
            validation["attribution"] = _attribution(val_results, winner[2])
            fold_results.append(validation)
            print(
                f"[WF VALIDATE] fold={fold_index + 1} {val_start}..{val_end} "
                f"cfg={EXIT_GRID[winner[2]]['target_level']}/{EXIT_GRID[winner[2]]['stop_level']} "
                f"trades={validation['trades']} WR={validation['win_rate']:.2f} "
                f"net={validation['net_points']:+.2f} net_rs={validation['net_rs']:+,.2f} "
                f"DD={validation['max_drawdown_points']:.2f}",
                flush=True,
            )

        stitched = {
            "trades": 0,
            "wins": 0,
            "fees_rs": 0.0,
            "net_points": 0.0,
            "net_rs": 0.0,
            "max_fold_drawdown_points": 0.0,
        }
        stitched_daily = {}
        for fold_index, (_, _, val_start, val_end) in enumerate(FOLDS):
            winner = best_train[fold_index]
            val_days = [d for d in days if val_start <= d <= val_end]
            val_results = [r for r in day_results if val_start <= r["day"] <= val_end]
            validation = _evaluate_window(val_results, val_days, EXIT_GRID.index(winner["config"]))
            for day, series in validation["daily"].items():
                stitched_daily[day] = series
            for key in ("trades", "wins"):
                stitched[key] += int(validation[key])
            for key in ("fees_rs", "net_points", "net_rs"):
                stitched[key] += float(validation[key])
            stitched["max_fold_drawdown_points"] = max(
                stitched["max_fold_drawdown_points"],
                float(validation["max_drawdown_points"]),
            )
        stitched["win_rate"] = round(100.0 * stitched["wins"] / stitched["trades"], 2) if stitched["trades"] else 0.0
        stitched["max_drawdown_points"] = round(_stitch_drawdown(stitched_daily) / LOT_SIZE, 2)
        stitched["score"] = round(
            stitched["net_points"] - args.dd_penalty * stitched["max_drawdown_points"], 4
        )
        print(
            f"[STITCHED OOS] trades={stitched['trades']} WR={stitched['win_rate']:.2f} "
            f"net={stitched['net_points']:+.2f} net_rs={stitched['net_rs']:+,.2f} "
            f"DD={stitched['max_drawdown_points']:.2f}",
            flush=True,
        )

    result = {
        "engine": "smart_fib_combined_tf",
        "mode": "union_of_tf_streams",
        "data_root": str(Path(args.data_root).resolve()),
        "start": days[0],
        "end": days[-1],
        "days": len(days),
        "timeframes": list(timeframes),
        "champion_variant": CHAMPION,
        "fixed_cost_per_trade": FIXED_COST,
        "exit_grid": EXIT_GRID,
        "champion_exit_index": CHAMPION_EXIT_INDEX,
        "non_wf_champion_exits": non_wf,
        "full_table": full_table,
        "folds": fold_results,
        "stitched_oos": stitched,
        "per_day_merged_counts": [
            {"day": r["day"], "per_tf_selected": r["per_tf_selected"], "merged": r["merged_count"]}
            for r in day_results
        ],
        "prep_seconds": round(prep_seconds, 3),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())