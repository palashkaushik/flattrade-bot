"""Shortcut Backtest Optuna search over the causal live-parity engine."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from multiprocessing import Pool
from pathlib import Path
from statistics import mean

import optuna

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_5y_optimized import load_spot, option_files
from causal_live_parity_research import EXIT_POLICIES, LOT_SIZE, init_worker, process_day


SEARCH_SPACE = {
    "s1_k": [7, 9, 12, 14],
    "s4_k": [50, 60, 75],
    "atr_period": [10, 14, 20],
    "atr_sl_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
    "atr_tp_mult": [2.0, 3.0, 4.0, 5.0, 6.0],
    "f6_s4_thresh": [75.0, 79.5, 85.0],
    "f6_s1_thresh": [15.0, 20.5, 25.0],
    "consec_loss": [4, 6, 8],
}


def choose_days(days: list[str], limit: int) -> list[str]:
    if limit >= len(days):
        return days
    years = {}
    for day in days:
        years.setdefault(day[:4], []).append(day)
    selected = []
    per_year = max(1, limit // max(1, len(years)))
    for year_days in years.values():
        if len(year_days) <= per_year:
            selected.extend(year_days)
            continue
        indexes = [round(index * (len(year_days) - 1) / max(1, per_year - 1)) for index in range(per_year)]
        selected.extend(year_days[index] for index in indexes)
    return sorted(set(selected))


def tasks_for(
    days: list[str],
    all_days: list[str],
    files: dict[str, str],
    params: dict,
    mode: str,
    max_daily_profit_points: float | None = None,
    max_daily_loss_points: float | None = None,
    costs_enabled: bool = True,
    fixed_tp_points: float | None = None,
    exit_policy: str = "static",
    breakout_mode: str = "first_break_high",
):
    positions = {day: index for index, day in enumerate(all_days)}
    return [
        (
            day,
            str(files[day]),
            str(files[all_days[positions[day] - 1]]) if positions[day] else "",
            params,
            mode,
            costs_enabled,
            max_daily_profit_points,
            max_daily_loss_points,
            fixed_tp_points,
            exit_policy,
            breakout_mode,
        )
        for day in days
    ]


def collect(
    pool,
    days,
    all_days,
    files,
    params,
    mode,
    max_daily_profit_points=None,
    max_daily_loss_points=None,
    costs_enabled=True,
    fixed_tp_points=None,
    exit_policy="static",
    breakout_mode="first_break_high",
):
    trades = []
    tasks = tasks_for(
        days,
        all_days,
        files,
        params,
        mode,
        max_daily_profit_points,
        max_daily_loss_points,
        costs_enabled,
        fixed_tp_points,
        exit_policy,
        breakout_mode,
    )
    for result in pool.imap(process_day, tasks):
        trades.extend(result)
    return trades


def metrics(trades, day_count):
    ordered = sorted(trades, key=lambda trade: (trade["date"], trade["exit_min"]))
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in ordered:
        equity += trade["rs_net"] / LOT_SIZE
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    wins = [trade["rs_net"] for trade in trades if trade["rs_net"] > 0]
    losses = [trade["rs_net"] for trade in trades if trade["rs_net"] <= 0]
    loss_total = abs(sum(losses))
    return {
        "trades": len(trades),
        "net_points": round(sum(trade["rs_net"] for trade in trades) / LOT_SIZE, 2),
        "net_rs": round(sum(trade["rs_net"] for trade in trades)),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(sum(wins) / loss_total, 4) if loss_total else float("inf"),
        "max_drawdown_points": round(max_drawdown, 2),
        "avg_trades_per_day": round(len(trades) / day_count, 3) if day_count else 0.0,
        "avg_sl_points": round(mean(trade["sl_points"] for trade in trades), 3) if trades else 0.0,
        "avg_tp_points": round(mean(trade["tp_points"] for trade in trades), 3) if trades else 0.0,
        "fees_rs": round(sum(trade["fee"] for trade in trades), 2),
    }


def suggest(trial, params, fixed_tp_points=None, exit_policy="static", fixed_consec_loss=None):
    params.update({
        "s1_k": trial.suggest_categorical("s1_k", SEARCH_SPACE["s1_k"]),
        "s1_d": 3,
        "s4_k": trial.suggest_categorical("s4_k", SEARCH_SPACE["s4_k"]),
        "atr_period": trial.suggest_categorical("atr_period", SEARCH_SPACE["atr_period"]),
        "atr_sl_mult": trial.suggest_categorical("atr_sl_mult", SEARCH_SPACE["atr_sl_mult"]),
        "f6_s4_thresh": trial.suggest_categorical("f6_s4_thresh", SEARCH_SPACE["f6_s4_thresh"]),
        "f6_s1_thresh": trial.suggest_categorical("f6_s1_thresh", SEARCH_SPACE["f6_s1_thresh"]),
        "consec_loss": (
            trial.suggest_categorical("consec_loss", SEARCH_SPACE["consec_loss"])
            if fixed_consec_loss is None
            else fixed_consec_loss
        ),
    })
    params["atr_tp_mult"] = (
        trial.suggest_categorical("atr_tp_mult", SEARCH_SPACE["atr_tp_mult"])
        if fixed_tp_points is None
        else None
    )
    params["exit_policy"] = (
        trial.suggest_categorical("exit_policy", EXIT_POLICIES)
        if exit_policy == "matrix"
        else exit_policy
    )
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("no_divergence", "new_divergence", "previous_divergence"), default="new_divergence")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dd-penalty", type=float, default=0.25)
    parser.add_argument("--max-daily-profit-points", type=float, default=None)
    parser.add_argument("--max-daily-loss-points", type=float, default=None)
    parser.add_argument("--no-costs", action="store_true")
    parser.add_argument("--fixed-tp-points", type=float, default=None)
    parser.add_argument("--consec-loss", type=int, default=None, help="Keep max consecutive losses fixed")
    parser.add_argument("--breakout-mode", choices=("legacy_high_break", "first_break_high"), default="first_break_high")
    parser.add_argument("--exit-policy", choices=("static", *EXIT_POLICIES, "matrix"), default="static")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/shortcut_optuna_result.json")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    spot = load_spot()
    files = option_files("2020-01-01", "2026-05-05")
    all_days = sorted(set(files) & set(spot))
    train_days = [day for day in all_days if "2020" <= day[:4] <= "2022"]
    validation_days = [day for day in all_days if "2023" <= day[:4] <= "2024"]
    blind_days = [day for day in all_days if day[:4] >= "2025"]
    if args.smoke:
        stages = [train_days[:5]]
        validation_days = validation_days[:5]
        blind_days = blind_days[:5]
    else:
        stages = [choose_days(train_days, limit) for limit in (20, 60, 200, len(train_days))]

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True, seed=42)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=max(1, min(20, len(stages[0]))),
        max_resource=len(stages[-1]),
        reduction_factor=3,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    if args.exit_policy == "matrix" and not args.smoke:
        for policy in EXIT_POLICIES:
            study.enqueue_trial({"exit_policy": policy})

    with Pool(max(1, min(8, args.workers)), initializer=init_worker, initargs=(spot,)) as pool:
        def objective(trial):
            params = suggest(
                trial,
                {},
                args.fixed_tp_points,
                args.exit_policy,
                args.consec_loss,
            )
            latest = None
            for step, stage_days in enumerate(stages):
                trades = collect(
                    pool,
                    stage_days,
                    train_days,
                    files,
                    params,
                    args.mode,
                    args.max_daily_profit_points,
                    args.max_daily_loss_points,
                    not args.no_costs,
                    args.fixed_tp_points,
                    params["exit_policy"],
                    args.breakout_mode,
                )
                latest = metrics(trades, len(stage_days))
                score = latest["net_points"] - args.dd_penalty * latest["max_drawdown_points"]
                trial.report(score, len(stage_days))
                if step < len(stages) - 1 and trial.should_prune():
                    raise optuna.TrialPruned()
            trial.set_user_attr("metrics", latest)
            return score

        study.optimize(objective, n_trials=min(args.trials, 2) if args.smoke else args.trials)
        best_params = dict(study.best_trial.params)
        best_params["s1_d"] = 3
        if args.consec_loss is not None:
            best_params["consec_loss"] = args.consec_loss
        best_params.setdefault("atr_tp_mult", None if args.fixed_tp_points is not None else 6.0)
        best_params.setdefault("exit_policy", args.exit_policy if args.exit_policy != "matrix" else "static")
        train_eval_days = stages[-1] if args.smoke else train_days
        train_trades = collect(
            pool,
            train_eval_days,
            train_days,
            files,
            best_params,
            args.mode,
            args.max_daily_profit_points,
            args.max_daily_loss_points,
            not args.no_costs,
            args.fixed_tp_points,
            best_params["exit_policy"],
            args.breakout_mode,
        )
        validation_trades = collect(
            pool,
            validation_days,
            all_days,
            files,
            best_params,
            args.mode,
            args.max_daily_profit_points,
            args.max_daily_loss_points,
            not args.no_costs,
            args.fixed_tp_points,
            best_params["exit_policy"],
            args.breakout_mode,
        )
        blind_trades = collect(
            pool,
            blind_days,
            all_days,
            files,
            best_params,
            args.mode,
            args.max_daily_profit_points,
            args.max_daily_loss_points,
            not args.no_costs,
            args.fixed_tp_points,
            best_params["exit_policy"],
            args.breakout_mode,
        )

    if args.smoke and not train_trades:
        raise RuntimeError("SMOKE FAIL: no training trades were produced")

    result = {
        "mode": args.mode,
        "smoke": args.smoke,
        "trials": args.trials,
        "max_daily_profit_points": args.max_daily_profit_points,
        "max_daily_loss_points": args.max_daily_loss_points,
        "fixed_tp_points": args.fixed_tp_points,
        "consec_loss": args.consec_loss,
        "breakout_mode": args.breakout_mode,
        "costs_enabled": not args.no_costs,
        "exit_policy": args.exit_policy,
        "search_space": {
            key: value
            for key, value in SEARCH_SPACE.items()
            if not (key == "atr_tp_mult" and args.fixed_tp_points is not None)
            and not (key == "consec_loss" and args.consec_loss is not None)
        },
        "best_params": best_params,
        "best_train_score": study.best_value,
        "train_evaluation_days": len(train_eval_days),
        "train": metrics(train_trades, len(train_eval_days)),
        "validation": metrics(validation_trades, len(validation_days)),
        "blind": metrics(blind_trades, len(blind_days)),
        "study_trials": [
            {"number": trial.number, "state": trial.state.name, "value": trial.value, "params": trial.params}
            for trial in study.trials
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
