"""Build GPU tensor caches for stricter signal variants (background prep).

Variants (all S1=12/4, buffer=0.5):
  V2: span=20 age=45 zone=(0.5, 0.786)
  V3: span=25 age=45 zone=(0.618, 0.786)
  V4: span=20 age=30 zone=(0.618, 0.786)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

from artifacts.f6_hybrid.smart_fib_cuda_bootstrap import configure_cuda_toolkit

configure_cuda_toolkit()

import torch

from artifacts.f6_hybrid import smart_fib_optimus_grid_gpu as grid

DATA_ROOT = r"C:\Users\user\Desktop\nifty50 data"
CACHE_DIR = Path("artifacts/f6_hybrid/smart_fib_grid_cache_full_float64")

VARIANTS = {
    "V2": "12:4:20:45:0.5:0.5:0.786",
    "V3": "12:4:20:45:0.5:0.618:0.786",
    "V4": "12:4:20:30:0.5:0.618:0.786",
}


def main() -> None:
    device = torch.device("cuda")
    adapter = grid.PolarsHistoricalDataAdapter(DATA_ROOT, start="2020-01-01", end="2026-05-05")
    days = adapter.available_days("2020-01-01", "2026-05-05")
    print(f"[DAYS] {len(days)}", flush=True)
    for label, spec in VARIANTS.items():
        variant = grid._parse_variant(spec)
        existing = grid._variant_cache_path(CACHE_DIR, days, variant)
        if existing.exists():
            print(f"[SKIP] {label} {variant.variant_id} cache exists: {existing}", flush=True)
            continue
        started = time.perf_counter()
        print(f"[BUILD] {label} {variant.variant_id} ...", flush=True)
        bundle = grid.collect_variant_cpu_dataset(
            adapter,
            days,
            variant,
            workers=8,
            parity_days=days[:3],
            tensor_cache_dir=CACHE_DIR,
        )
        elapsed = time.perf_counter() - started
        print(
            f"[DONE] {label} {variant.variant_id} events={int(bundle.dataset.event_mask.sum())} "
            f"prep={bundle.prep_seconds:.1f}s wall={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()