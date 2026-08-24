"""Optuna study over the flip order-block exit geometry (target/stop).

Optimizes  net_points - dd_penalty * max_drawdown_points  on the cached
flip tensor dataset (single GPU evaluation per trial via the canonical
evaluate_variant_grid path).  Search space is custom and centered on the
tight-stop family that won the asymmetric sweep (target -0.272 / stop
0.618).

Usage:
  python artifacts/f6_hybrid/order_block_optuna.py --allow-expensive \
      --n-trials 250 --output artifacts/f6_hybrid/order_block_optuna_flip.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.smart_fib_cuda_bootstrap import configure_cuda_toolkit

configure_cuda_toolkit()

import torch
import optuna

import artifacts.f6_hybrid.smart_fib_optimus_gpu as optimus
import artifacts.f6_hybrid.smart_fib_optimus_grid_gpu as grid
from artifacts.f6_hybrid import order_block_optimus_gpu as ob

DEFAULT_OUTPUT = ROOT / "artifacts/f6_hybrid/order_block_optuna_flip.json"


def _probe(
    evaluator: optimus.GpuEvaluator,
    gpu_variant: grid.VariantGpuDataset,
    days: list[str],
    target: float,
    stop: float,
) -> dict:
    configs, _ = ob.evaluate_video_grid(
        ob.ObVariant(flavor="flip"),
        gpu_variant,
        evaluator,
        days,
        targets=[target],
        fallback_targets=[0.29],
        thresholds=[15.0],
        stops=[stop],
    )
    return configs[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=grid.historical.DEFAULT_DATA_ROOT)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--timeframes", default="1,2,3,5")
    parser.add_argument("--flavor", default="flip", help="order-block flavor (flip or turn)")
    parser.add_argument("--n-trials", type=int, default=250)
    parser.add_argument("--timeout", type=int, default=None,
                        help="study wall-clock budget in seconds")
    parser.add_argument("--target-low", type=float, default=-0.9)
    parser.add_argument("--target-high", type=float, default=0.0)
    parser.add_argument("--stop-low", type=float, default=0.5)
    parser.add_argument("--stop-high", type=float, default=1.0)
    parser.add_argument("--dd-penalty", type=float, default=0.2,
                        help="drawdown penalty weight in the objective (runner default 0.2)")
    parser.add_argument("--min-trades", type=int, default=3000,
                        help="reject trials below this trade count (0 disables)")
    parser.add_argument("--min-win-rate", type=float, default=40.0,
                        help="reject trials below this win rate in PERCENT (0 disables)")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-expensive", action="store_true")
    parser.add_argument("--tensor-cache-dir", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    if args.n_trials <= 0:
        raise SystemExit("--n-trials must be positive")
    if args.target_low >= args.target_high or args.stop_low >= args.stop_high:
        raise SystemExit("empty search space: target/stop low must be below high")
    if not args.smoke and not args.allow_expensive:
        raise SystemExit("refusing a non-smoke study; use --smoke or --allow-expensive")
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")

    timeframes = tuple(int(piece) for piece in args.timeframes.split(",") if piece)
    if any(tf not in (1, 2, 3, 5) for tf in timeframes):
        raise SystemExit(f"--timeframes must be a comma list of 1,2,3,5: {args.timeframes}")

    device = optimus.require_device(False)
    print(f"[GPU INIT] device={device} allocator={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)

    adapter = grid.PolarsHistoricalDataAdapter(args.data_root, start=args.start, end=args.end)
    available = adapter.available_days(args.start, args.end)
    if not available:
        raise SystemExit(f"no overlapping index/options days in {args.start}..{args.end}")
    days = available[:5] if args.smoke else available
    n_days = len(days)
    print(f"[STUDY] days={n_days} range {days[0]}..{days[-1]} flavor={args.flavor}", flush=True)

    variant = ob.ObVariant(flavor=args.flavor)
    bundle = ob.collect_ob_cpu_dataset(
        adapter, days, variant, timeframes,
        workers=max(1, min(8, os.cpu_count() or 1)),
        parity_days=days[:3],
        tensor_cache_dir=Path(args.tensor_cache_dir) if args.tensor_cache_dir else None,
    )
    gpu_variant = grid.to_gpu_dataset(bundle.dataset, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluator = optimus.GpuEvaluator(
        gpu_variant.engine,
        16,
        brokerage_per_order=0.0,
        fixed_cost_per_trade=ob.FIXED_COST_PER_TRADE,
    )

    probes = [
        ("flip winner", -0.272, 0.618),
        ("video base", 0.0, 1.155),
    ]
    for label, tgt, stop in probes:
        rec = _probe(evaluator, gpu_variant, days, tgt, stop)
        print(f"[PROBE] {label}: target={tgt} stop={stop} trades={rec['trades']} "
              f"WR={rec['win_rate']:.1f}% net={rec['net_points']:+.2f}pts "
              f"DD={rec['max_drawdown_points']:.2f}pts PF={rec['profit_factor']:.2f}", flush=True)

    min_trades = args.min_trades if not args.smoke else max(10, n_days * 4)
    min_wr = args.min_win_rate
    dd_penalty = args.dd_penalty

    def objective(trial: optuna.Trial) -> float:
        target = trial.suggest_float("target_level", args.target_low, args.target_high)
        stop = trial.suggest_float("stop_level", args.stop_low, args.stop_high)
        rec = _probe(evaluator, gpu_variant, days, target, stop)
        trades = int(rec["trades"])
        wr = float(rec["win_rate"])
        net = float(rec["net_points"])
        dd = float(rec["max_drawdown_points"])
        pf = float(rec["profit_factor"])
        for name, value in (
            ("trades", trades), ("win_rate", wr), ("net_points", net),
            ("max_drawdown_points", dd), ("profit_factor", pf),
            ("net_rs", float(rec["net_rs"])), ("fees_rs", float(rec["fees_rs"])),
        ):
            trial.set_user_attr(name, value)
        if (args.min_trades and trades < min_trades) or (min_wr and wr < min_wr):
            return -1.0e9
        return net - dd_penalty * dd

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, n_startup_trials=25)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=5),
        study_name=f"ob_{args.flavor}_exit",
    )
    print(f"[STUDY] objective = net_points - {dd_penalty}*max_drawdown_points | "
          f"space target [{args.target_low},{args.target_high}] stop [{args.stop_low},{args.stop_high}] | "
          f"constraints trades>={min_trades} WR>={min_wr}", flush=True)
    started = time.perf_counter()
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=False)
    wall = time.perf_counter() - started

    best = study.best_trial
    top_rows = []
    for trial in sorted(study.trials, key=lambda t: t.value if t.value is not None else -1.0e18, reverse=True)[:5]:
        top_rows.append({
            "target_level": trial.params["target_level"],
            "stop_level": trial.params["stop_level"],
            "value": trial.value,
            "trades": trial.user_attrs.get("trades"),
            "win_rate": trial.user_attrs.get("win_rate"),
            "net_points": trial.user_attrs.get("net_points"),
            "net_rs": trial.user_attrs.get("net_rs"),
            "max_drawdown_points": trial.user_attrs.get("max_drawdown_points"),
            "profit_factor": trial.user_attrs.get("profit_factor"),
            "fees_rs": trial.user_attrs.get("fees_rs"),
        })
    print(f"[BEST] target={best.params['target_level']:.4f} stop={best.params['stop_level']:.4f} "
          f"value={best.value:+.2f} net={best.user_attrs['net_points']:+.2f}pts "
          f"DD={best.user_attrs['max_drawdown_points']:.2f}pts trades={best.user_attrs['trades']} "
          f"WR={best.user_attrs['win_rate']:.1f}% PF={best.user_attrs['profit_factor']:.2f}", flush=True)
    print(f"[TOP-5]")
    for i, row in enumerate(top_rows, start=1):
        print(f"  {i}. target={row['target_level']:.4f} stop={row['stop_level']:.4f} "
              f"value={row['value']:+.2f} net={row['net_points']:+.2f}pts DD={row['max_drawdown_points']:.2f}pts "
              f"trades={row['trades']} WR={row['win_rate']:.1f}% PF={row['profit_factor']:.2f}", flush=True)

    payload = {
        "study": f"ob_{args.flavor}_exit",
        "flavor": args.flavor,
        "days": days,
        "n_days": n_days,
        "n_trials_completed": len(study.trials),
        "wall_seconds": round(wall, 1),
        "objective": f"net_points - {dd_penalty} * max_drawdown_points",
        "search_space": {
            "target_level": [args.target_low, args.target_high],
            "stop_level": [args.stop_low, args.stop_high],
        },
        "constraints": {"min_trades": min_trades, "min_win_rate_percent": min_wr},
        "sampler_seed": args.sampler_seed,
        "best": {
            "target_level": best.params["target_level"],
            "stop_level": best.params["stop_level"],
            "value": best.value,
            **{k: v for k, v in best.user_attrs.items()},
        },
        "top_5": top_rows,
        "probes": {
            "flip_winner_minus0p272_0p618": _probe(evaluator, gpu_variant, days, -0.272, 0.618),
            "video_0_1p155": _probe(evaluator, gpu_variant, days, 0.0, 1.155),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"JSON: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
