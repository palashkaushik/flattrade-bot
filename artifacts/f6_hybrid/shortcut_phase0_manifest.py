"""Create the reproducibility manifest for Shortcut Backtest Phase 0."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(r"C:\Websites\ammu")
OPTION_ROOT = DATA_ROOT / "nifty_options"
SPOT_FILE = DATA_ROOT / "index" / "NIFTY 50_minute.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    files = [SPOT_FILE, *sorted(OPTION_ROOT.rglob("*.csv"))]
    manifest = []
    for path in files:
        if path.exists():
            manifest.append({
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })

    params_path = ROOT / "artifacts" / "f6_hybrid" / "backtest_f6_16l_champion_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))["params"]
    output = ROOT / "artifacts" / "f6_hybrid" / "shortcut_phase0_manifest.json"
    result = {
        "name": "Shortcut Backtest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in ("numpy", "pandas", "optuna", "numba", "polars", "pyarrow")
        },
        "data": {
            "root": str(DATA_ROOT),
            "file_count": len(manifest),
            "files": manifest,
        },
        "canonical_baseline": {
            "params": params,
            "pinbar": {
                "lower_shadow_min_ratio": 0.45,
                "body_max_ratio": 0.45,
                "upper_shadow_max_ratio": 0.25,
            },
            "timeframes": ["1m", "2m", "3m", "5m"],
            "divergence_profiles": ["no_divergence", "current_pivot", "legacy_rolling"],
            "slippage_points_per_side": 1.0,
            "brokerage_per_order": 0.0,
            "lot_size": 65,
            "daily_loss_rs": -2000.0,
        },
        "validation_windows": {
            "train": ["2020-01-01", "2022-12-31"],
            "validate": ["2023-01-01", "2024-12-31"],
            "blind": ["2025-01-01", "2026-05-05"],
        },
        "live_or_paper_changes": False,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Manifest: {output}")
    print(f"Files hashed: {len(manifest)}")
    print(f"Git revision: {result['git_revision']}")


if __name__ == "__main__":
    main()
