"""Prepare an opt-in Parquet cache for F6 daily option files."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backtest_5y_optimized import option_files
from f6_hybrid.columnar import cache_path_for, convert_csv_to_parquet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2020-01-07")
    parser.add_argument(
        "--output",
        default="artifacts/f6_hybrid/columnar_cache",
    )
    args = parser.parse_args()
    output = Path(args.output)
    files = option_files(args.start, args.end)
    created = 0
    skipped = 0
    for day, source in sorted(files.items()):
        target = cache_path_for(Path(source), output)
        if target.exists():
            skipped += 1
            continue
        convert_csv_to_parquet(Path(source), target)
        created += 1
        print(f"converted {day}: {target}", flush=True)
    print(f"created={created} skipped={skipped} output={output}")


if __name__ == "__main__":
    main()
