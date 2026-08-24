"""Exact 3-date CPU/GPU parity for the new frequency-cap params.

Tests max_trades_per_day + daily_loss_limit_rs on the champion config
(0.786/1.13, fallback 0, threshold 5, fixed Rs40) over the first three
available days. Mirrors validate_variant_parity with zero tolerance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

from artifacts.f6_hybrid.smart_fib_cuda_bootstrap import configure_cuda_toolkit

configure_cuda_toolkit()

import torch

from artifacts.f6_hybrid import smart_fib_optimus_gpu as optimus
from artifacts.f6_hybrid import smart_fib_optimus_grid_gpu as grid

DATA_ROOT = r"C:\Users\user\Desktop\nifty50 data"
CHAMPION = grid._parse_variant("12:4:15:45:0.5:0.5:0.786")

CAP_CONFIGS = [
    {},
    {"max_trades_per_day": 4},
    {"max_trades_per_day": 5},
    {"daily_loss_limit_rs": 5000.0},
    {"max_trades_per_day": 4, "daily_loss_limit_rs": 5000.0},
    {"max_trades_per_day": 8, "daily_loss_limit_rs": 10000.0},
]

BASE_PARAMS = {
    "stop_level": 1.13,
    "target_level": 0.786,
    "fallback_target_level": 0.0,
    "option_point_threshold": 5.0,
}


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    adapter = grid.PolarsHistoricalDataAdapter(DATA_ROOT, start="2020-01-01", end="2026-05-05")
    days = adapter.available_days("2020-01-01", "2026-05-05")
    if len(days) < 500:
        raise SystemExit("full 2020-2026 data is required")
    print(f"[SCAN] {len(days)} days for high-volume window", flush=True)
    dataset, _, _ = grid._cached_variant_dataset(
        adapter.data_root, days, CHAMPION, Path("artifacts/f6_hybrid/smart_fib_grid_cache_full_float64")
    )
    resident = grid.to_gpu_dataset(dataset, device)
    evaluator = optimus.GpuEvaluator(
        resident.engine, 1, brokerage_per_order=0.0, fixed_cost_per_trade=40.0
    )
    scan_result = evaluator.evaluate([dict(BASE_PARAMS)], None)
    daily = scan_result[0]["daily_trades"]
    top_indices = sorted(range(len(days)), key=lambda i: -int(daily[i]))[:3]
    parity_days = [days[i] for i in sorted(top_indices)]
    print(f"[PARITY DAYS] {parity_days} "
          f"trades={[int(daily[i]) for i in sorted(top_indices)]}", flush=True)

    bundle = grid.collect_variant_cpu_dataset(
        adapter,
        parity_days,
        CHAMPION,
        workers=4,
        parity_days=parity_days,
        tensor_cache_dir=None,
    )
    gpu_variant = grid.to_gpu_dataset(bundle.dataset, device)
    evaluator = optimus.GpuEvaluator(
        gpu_variant.engine,
        100,
        brokerage_per_order=0.0,
        fixed_cost_per_trade=40.0,
    )

    mask = torch.tensor(
        [day in parity_days for day in bundle.dataset.days],
        dtype=torch.bool,
        device=device,
    )
    params = [dict(BASE_PARAMS, **cap) for cap in CAP_CONFIGS]
    gpu_results = evaluator.evaluate(params, mask)

    payloads = dict(bundle.parity_payloads)
    failed = False
    for index, (cap, gpu_result) in enumerate(zip(CAP_CONFIGS, gpu_results)):
        label = cap or {"(no caps)": None}
        print(f"\n=== {label} ===", flush=True)
        for day in parity_days:
            payload = payloads[day]
            cpu_trades = grid._cpu_variant_trades(
                payload,
                BASE_PARAMS["target_level"],
                BASE_PARAMS["fallback_target_level"],
                BASE_PARAMS["option_point_threshold"],
                BASE_PARAMS["stop_level"],
                0.0,
                40.0,
                max_trades_per_day=cap.get("max_trades_per_day"),
                daily_loss_limit_rs=cap.get("daily_loss_limit_rs"),
            )
            cpu = grid._cpu_stats(cpu_trades)
            day_index = bundle.dataset.days.index(day)
            gpu = {
                "trades": int(gpu_result["daily_trades"][day_index]),
                "net_points": round(float(gpu_result["daily_net_points"][day_index]), 2),
                "net_rs": round(float(gpu_result["daily_net_rs"][day_index]), 2),
                "dd_points": round(
                    float(gpu_result["daily_drawdown_rs"][day_index]) / 65.0, 2
                ),
            }
            ok = (
                cpu["trades"] == gpu["trades"]
                and abs(cpu["net_points"] - gpu["net_points"]) < 1e-9
                and abs(cpu["net_rs"] - gpu["net_rs"]) < 1e-9
                and abs(cpu["max_drawdown_points"] - gpu["dd_points"]) < 1e-9
            )
            failed = failed or not ok
            print(
                f"  {day} CPU trades={cpu['trades']} pts={cpu['net_points']:+.2f} "
                f"rs={cpu['net_rs']:+.2f} dd={cpu['max_drawdown_points']:.2f}",
                flush=True,
            )
            print(
                f"  {day} GPU trades={gpu['trades']} pts={gpu['net_points']:+.2f} "
                f"rs={gpu['net_rs']:+.2f} dd={gpu['dd_points']:.2f} -> "
                f"{'PASS' if ok else 'FAIL'}",
                flush=True,
            )

    if failed:
        raise SystemExit("CAP PARITY FAILED")
    print("\nCAP PARITY PASS (all configs x days exact)", flush=True)


if __name__ == "__main__":
    main()