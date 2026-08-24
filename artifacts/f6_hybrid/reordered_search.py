"""Staged, batched Optuna search for the factorized F6 runner.

This is an orchestration artifact. ``grid_optimize_f6_atr`` remains the
reference engine and ``run_factorized_candidates`` remains the exact evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import optuna

# Direct ``python artifacts/f6_hybrid/reordered_search.py`` execution puts the
# script directory, not the repository root, first on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, apply_costs, net_stats
from f6_hybrid.raw_features import FactorizedCandidatePool, run_factorized_candidates


TRADE_FIELDS = (
    "date", "entry_min", "exit_min", "side", "symbol", "entry", "exit",
    "pts", "rs", "sl_pts", "tp_pts", "reason", "duration_min", "tf",
)
CSV_FIELDS = (
    "run_id", "trial", "resource", "state", "score", "trades", "wr",
    "net_rs", "net_pts", "fees", "pf", "elapsed_s", "base_builds", "signal_builds", "s1_k",
    "s4_k", "atr_period", "atr_sl_mult", "atr_tp_mult", "f6_s4_thresh",
    "f6_s1_thresh", "consec_loss",
)


def build_stage_resources(days: Sequence[str]) -> list[int]:
    """Return unique cumulative day counts for staged screening."""
    available = len(days)
    resources = [min(limit, available) for limit in (5, 20, 60, available)]
    return list(dict.fromkeys(resource for resource in resources if resource > 0))


def stage_day_block(
    days: Sequence[str], previous_resource: int, resource: int
) -> tuple[list[str], list[str]]:
    """Return new stage days plus one prior day for indicator warmup."""
    days = list(days)
    block = days[previous_resource:resource]
    if previous_resource == 0:
        return block, block
    return [days[previous_resource - 1], *block], block


def canonical_trade(trade: dict) -> tuple:
    """Return the reference trade fields used for exact parity checks."""
    return tuple(trade.get(field) for field in TRADE_FIELDS)


def params_from_trial(trial, search_space=None) -> dict:
    search_space = search_space or grid.SEARCH_SPACE
    params = {
        name: trial.suggest_categorical(name, values)
        for name, values in search_space.items()
    }
    params["s1_d"] = 3
    return params


def _stats_payload(stats: dict) -> dict:
    return {
        "trades": int(stats["trades"]),
        "wr": float(stats["wr"]),
        "net_pts": float(stats.get("pts", 0.0)),
        "net_rs": int(stats["rs"]),
        "pf": float(stats["pf"]),
        "fees": float(stats.get("fees", 0.0)),
    }


def score_candidate(
    trades: list[dict],
    cost_aware: bool = False,
    slippage_points: float | None = None,
) -> tuple[float, dict]:
    """Return the ranking score and stats using raw or configured net P&L."""
    if cost_aware:
        net_trades = [dict(trade) for trade in trades]
        apply_costs(
            net_trades,
            BROKERAGE_PER_ORDER,
            slippage_pts=slippage_points,
        )
        stats = net_stats(net_trades)
    else:
        stats, _ = grid.stats_for(trades)
    return float(grid.composite_score(stats)), stats


def _run_id(smoke: bool) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"smoke_{stamp}" if smoke else stamp


def _output_paths(output_prefix: str | Path, run_id: str) -> tuple[Path, Path]:
    base = Path(output_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    stem = base.name
    return (
        base.parent / f"{stem}_{run_id}.csv",
        base.parent / f"{stem}_{run_id}.json",
    )


def _csv_row(
    run_id: str,
    trial_number: int,
    resource: int,
    state: str,
    score: float,
    stats: dict,
    elapsed_s: float,
    base_builds: int,
    signal_builds: int,
    params: dict,
) -> dict:
    return {
        "run_id": run_id,
        "trial": trial_number,
        "resource": resource,
        "state": state,
        "score": round(float(score), 6),
        "trades": int(stats["trades"]),
        "wr": round(float(stats["wr"]), 4),
        "net_rs": int(stats["rs"]),
        "net_pts": round(float(stats.get("pts", 0.0)), 4),
        "fees": round(float(stats.get("fees", 0.0)), 2),
        "pf": round(float(stats["pf"]), 6),
        "elapsed_s": round(float(elapsed_s), 3),
        "base_builds": int(base_builds),
        "signal_builds": int(signal_builds),
        **{name: params[name] for name in grid.SEARCH_SPACE},
    }


def run_search(
    days: Sequence[str],
    files: dict[str, str],
    spot_all: dict,
    n_trials: int = 40,
    batch_size: int = 8,
    workers: int = grid.WORKERS,
    output_prefix: str | Path = "artifacts/f6_hybrid/reordered_search",
    fixed_candidates: list[dict] | None = None,
    smoke: bool = False,
    search_space: dict[str, list] | None = None,
    cost_aware: bool = False,
    slippage_points: float | None = None,
    use_divergence: bool = True,
) -> dict:
    """Run staged candidate batches and persist an auditable result ledger."""
    days = list(days)
    workers = max(1, min(8, int(workers)))
    search_space = search_space or grid.SEARCH_SPACE
    resources = build_stage_resources(days)
    if not resources:
        raise ValueError("search requires at least one available day")
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if fixed_candidates is not None and len(fixed_candidates) != n_trials:
        raise ValueError("fixed_candidates length must equal n_trials")

    run_id = _run_id(smoke)
    csv_path, json_path = _output_paths(output_prefix, run_id)
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=resources[0],
        max_resource=resources[-1],
        reduction_factor=3,
    )
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rows = []
    full_results = []
    stage_timings = []
    pruned_count = 0
    completed_count = 0
    trial_number = 0
    started = time.time()

    with FactorizedCandidatePool(spot_all, workers=workers) as evaluator, csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        while trial_number < n_trials:
            count = min(batch_size, n_trials - trial_number)
            active = []
            for _ in range(count):
                trial = study.ask()
                params = (
                    dict(fixed_candidates[trial_number])
                    if fixed_candidates is not None
                    else params_from_trial(trial, search_space)
                )
                params["s1_d"] = 3
                params["use_divergence"] = use_divergence
                active.append((trial, params))
                trial_number += 1

            cumulative_trades = [[] for _ in active]
            previous_resource = 0
            for resource in resources:
                if not active:
                    break
                stage_started = time.time()
                eval_days, block_days = stage_day_block(
                    days, previous_resource, resource
                )
                stage_trades, base_builds, signal_builds = evaluator.run(
                    [params for _, params in active],
                    eval_days,
                    files,
                )
                if len(stage_trades) != len(active):
                    raise RuntimeError(
                        f"stage {resource} returned {len(stage_trades)} results "
                        f"for {len(active)} candidates"
                    )
                stage_elapsed = time.time() - stage_started
                next_active = []
                stage_pruned = 0
                stage_completed = 0

                block_set = set(block_days)
                next_cumulative = []
                for index, ((trial, params), trades) in enumerate(zip(active, stage_trades)):
                    cumulative_trades[index].extend(
                        trade for trade in trades if trade.get("date") in block_set
                    )
                    score, stats = score_candidate(
                        cumulative_trades[index],
                        cost_aware=cost_aware,
                        slippage_points=slippage_points,
                    )
                    trial.report(score, step=resource)
                    is_final = resource == resources[-1]
                    should_prune = not is_final and trial.should_prune()
                    state = "PRUNED" if should_prune else (
                        "COMPLETE" if is_final else "ACTIVE"
                    )
                    row = _csv_row(
                        run_id,
                        trial.number,
                        resource,
                        state,
                        score,
                        stats,
                        stage_elapsed,
                        base_builds,
                        signal_builds,
                        params,
                    )
                    rows.append(row)
                    writer.writerow(row)

                    if should_prune:
                        study.tell(
                            trial,
                            state=optuna.trial.TrialState.PRUNED,
                        )
                        pruned_count += 1
                        stage_pruned += 1
                    elif is_final:
                        study.tell(trial, score)
                        completed_count += 1
                        stage_completed += 1
                        full_results.append({
                            "trial": trial.number,
                            "score": float(score),
                            "stats": _stats_payload(stats),
                            "params": params,
                        })
                    else:
                        next_active.append((trial, params))
                        next_cumulative.append(cumulative_trades[index])

                csv_file.flush()
                stage_timings.append({
                    "resource": resource,
                    "active": len(active),
                    "survivors": len(next_active),
                    "pruned": stage_pruned,
                    "completed": stage_completed,
                    "seconds": round(stage_elapsed, 3),
                    "base_builds": base_builds,
                    "signal_builds": signal_builds,
                })
                active = next_active
                cumulative_trades = next_cumulative
                previous_resource = resource

    full_results.sort(key=lambda item: item["score"], reverse=True)
    summary = {
        "run_id": run_id,
        "trials_requested": n_trials,
        "trial_count": trial_number,
        "completed_count": completed_count,
        "pruned_count": pruned_count,
        "batch_size": batch_size,
        "workers": workers,
        "days": days,
        "resources": resources,
        "stage_timings": stage_timings,
        "wall_seconds": round(time.time() - started, 3),
        "best_full_fidelity": full_results[:20],
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "rows_written": len(rows),
        "search_space": search_space,
        "cost_aware": cost_aware,
        "slippage_points": slippage_points,
        "use_divergence": use_divergence,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _smoke_reference_parity(
    champion: dict,
    days: list[str],
    files: dict[str, str],
    spot_all: dict,
    workers: int,
) -> tuple[bool, int]:
    factorized, _, _ = run_factorized_candidates(
        [champion], days, files, spot_all, workers=workers
    )
    with grid.Pool(
        processes=workers,
        initializer=grid.init_worker_local,
        initargs=(spot_all,),
    ) as pool:
        reference = grid.run_days(pool, champion, days, files, spot_all)
    actual = [canonical_trade(trade) for trade in factorized[0]]
    expected = [canonical_trade(trade) for trade in reference]
    return actual == expected, len(actual)


def smoke_test(
    output_prefix: str | Path,
    cost_aware: bool = False,
    slippage_points: float | None = None,
    use_divergence: bool = True,
) -> dict:
    spot_all = load_spot()
    files = option_files("2020-01-01", "2020-01-07")
    days = sorted(set(files) & set(spot_all))[:5]
    if len(days) < 5:
        raise RuntimeError(f"smoke requires five days, found {len(days)}")

    champion = dict(grid.CHAMPION)
    alternate = dict(champion)
    alternate.update(atr_sl_mult=3.0, atr_tp_mult=6.0, consec_loss=8)
    champion["use_divergence"] = use_divergence
    alternate["use_divergence"] = use_divergence
    parity_ok, champion_trades = _smoke_reference_parity(
        champion, days, files, spot_all, workers=2
    )
    if not parity_ok:
        raise SystemExit("SMOKE FAIL: champion trade parity mismatch")
    if not 15 <= champion_trades <= 40:
        raise SystemExit(
            f"SMOKE FAIL: suspicious champion trade count {champion_trades}"
        )

    summary = run_search(
        days,
        files,
        spot_all,
        n_trials=2,
        batch_size=2,
        workers=2,
        output_prefix=output_prefix,
        fixed_candidates=[champion, alternate],
        smoke=True,
        cost_aware=cost_aware,
        slippage_points=slippage_points,
        use_divergence=use_divergence,
    )
    if summary["trial_count"] != 2 or summary["completed_count"] != 2:
        raise SystemExit("SMOKE FAIL: candidate result dropped")
    summary["parity_ok"] = True
    summary["champion_trade_count"] = champion_trades
    Path(summary["json_path"]).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"SMOKE TEST OK | days={len(days)} | champion_trades={champion_trades} "
        f"| parity=True | outputs={summary['json_path']}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=grid.WORKERS)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--f6-s4-thresh", type=float, default=79.5)
    parser.add_argument("--f6-s1-thresh", type=float, default=20.5)
    parser.add_argument("--cost-aware", action="store_true")
    parser.add_argument("--slippage", type=float, default=1.0)
    parser.add_argument("--no-divergence", action="store_true")
    parser.add_argument(
        "--output-prefix",
        default="artifacts/f6_hybrid/reordered_search",
    )
    args = parser.parse_args()

    if args.smoke:
        smoke_test(
            f"{args.output_prefix}_smoke",
            cost_aware=args.cost_aware,
            slippage_points=args.slippage,
            use_divergence=not args.no_divergence,
        )
        return

    spot_all = load_spot()
    files = option_files(args.start, args.end)
    days = sorted(set(files) & set(spot_all))
    if not days:
        raise SystemExit("no overlapping option and spot days in search window")
    resources = build_stage_resources(days)
    search_space = {
        **grid.SEARCH_SPACE,
        "f6_s4_thresh": [args.f6_s4_thresh],
        "f6_s1_thresh": [args.f6_s1_thresh],
    }
    print(
        f"REORDERED SEARCH | days={len(days)} | resources={resources} | "
        f"trials={args.trials} | batch={args.batch_size} | workers={args.workers} | "
        f"f6={args.f6_s4_thresh}/{args.f6_s1_thresh} | "
        f"slippage={args.slippage} | divergence={not args.no_divergence}",
        flush=True,
    )
    summary = run_search(
        days,
        files,
        spot_all,
        n_trials=args.trials,
        batch_size=args.batch_size,
        workers=args.workers,
        output_prefix=args.output_prefix,
        search_space=search_space,
        cost_aware=args.cost_aware,
        slippage_points=args.slippage,
        use_divergence=not args.no_divergence,
    )
    print(
        f"SEARCH COMPLETE | completed={summary['completed_count']} | "
        f"pruned={summary['pruned_count']} | seconds={summary['wall_seconds']:.1f} "
        f"| json={summary['json_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
