"""Arrow/Parquet day cache for the F6 option-file hot path."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from backtest_5y_optimized import to_minutes


COLUMNS = ("time", "symbol", "open", "high", "low", "close")


def cache_path_for(source: Path, cache_dir: Path) -> Path:
    """Return a stable cache path that cannot collide across source folders."""
    source_id = str(source.resolve()).encode("utf-8")
    digest = sha256(source_id).hexdigest()[:16]
    return cache_dir / f"{source.stem}-{digest}.parquet"


def convert_csv_to_parquet(source: Path, target: Path) -> Path:
    """Convert only the six engine columns, sorted for pointer-friendly reads."""
    convert_options = pacsv.ConvertOptions(
        include_columns=list(COLUMNS),
        column_types={
            "time": pa.string(),
            "symbol": pa.string(),
            "open": pa.float64(),
            "high": pa.float64(),
            "low": pa.float64(),
            "close": pa.float64(),
        },
    )
    table = pacsv.read_csv(
        str(source),
        read_options=pacsv.ReadOptions(use_threads=True),
        convert_options=convert_options,
    )
    table = table.select(list(COLUMNS)).sort_by(
        [("symbol", "ascending"), ("time", "ascending")]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        str(target),
        compression="snappy",
        use_dictionary=["symbol"],
        write_statistics=True,
    )
    return target


def _numpy_column(table: pa.Table, name: str) -> np.ndarray:
    return table[name].combine_chunks().to_numpy(zero_copy_only=False)


def load_parquet_day(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load sorted Parquet rows into the reference engine's symbol mapping."""
    table = pq.read_table(str(path), columns=list(COLUMNS), use_threads=True)
    if table.num_rows == 0:
        return {}

    symbols = np.asarray(table["symbol"].to_pylist(), dtype=str)
    minutes = np.fromiter(
        (to_minutes(value) for value in table["time"].to_pylist()),
        dtype=np.int32,
        count=table.num_rows,
    )
    columns = {
        name: _numpy_column(table, name).astype(np.float64, copy=False)
        for name in ("open", "high", "low", "close")
    }
    boundaries = np.flatnonzero(symbols[1:] != symbols[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(symbols)]

    result = {}
    for start, end in zip(starts, ends):
        symbol = str(symbols[start])
        result[symbol] = {
            "min": minutes[start:end],
            "open": columns["open"][start:end],
            "high": columns["high"][start:end],
            "low": columns["low"][start:end],
            "close": columns["close"][start:end],
        }
    return result
