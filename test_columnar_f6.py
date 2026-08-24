import os
import tempfile
import unittest
from pathlib import Path

import grid_optimize_f6_atr as grid
from f6_hybrid.columnar import (
    cache_path_for,
    convert_csv_to_parquet,
    load_parquet_day,
)


class ColumnarCacheTests(unittest.TestCase):
    def test_converts_and_loads_sorted_symbol_arrays(self):
        csv_text = (
            "time,symbol,open,high,low,close\n"
            "09:21:00,100CE,2,3,1,2.5\n"
            "09:20:00,200PE,4,5,3,4.5\n"
            "09:20:00,100CE,1,2,0,1.5\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "options.csv"
            cache_dir = Path(directory) / "cache"
            source.write_text(csv_text)
            target = cache_path_for(source, cache_dir)

            convert_csv_to_parquet(source, target)
            loaded = load_parquet_day(target)

        self.assertEqual(sorted(loaded), ["100CE", "200PE"])
        self.assertEqual(loaded["100CE"]["min"].tolist(), [560, 561])
        self.assertEqual(loaded["100CE"]["close"].tolist(), [1.5, 2.5])
        self.assertEqual(loaded["200PE"]["min"].tolist(), [560])

    def test_cache_path_is_stable_and_scoped_by_source_path(self):
        source = Path("C:/data/day.csv")
        first = cache_path_for(source, Path("cache"))
        second = cache_path_for(source, Path("cache"))

        self.assertEqual(first, second)
        self.assertEqual(first.parent, Path("cache"))

    def test_reference_cache_uses_existing_columnar_file_when_enabled(self):
        csv_text = (
            "time,symbol,open,high,low,close\n"
            "09:20:00,100CE,1,2,0,1.5\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "options.csv"
            cache_dir = Path(directory) / "cache"
            source.write_text(csv_text)
            target = cache_path_for(source, cache_dir)
            convert_csv_to_parquet(source, target)
            source.unlink()
            old_value = os.environ.get("F6_COLUMNAR_CACHE_DIR")
            os.environ["F6_COLUMNAR_CACHE_DIR"] = str(cache_dir)
            try:
                grid.GLOBAL_CACHE = {}
                loaded = grid.cached_day(str(source))
            finally:
                if old_value is None:
                    os.environ.pop("F6_COLUMNAR_CACHE_DIR", None)
                else:
                    os.environ["F6_COLUMNAR_CACHE_DIR"] = old_value

        self.assertEqual(loaded["100CE"]["min"].tolist(), [560])


if __name__ == "__main__":
    unittest.main()
