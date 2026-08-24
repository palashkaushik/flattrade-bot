"""Parameterized historical Smart Fib evaluator and optimizer scaffold.

The signal path is intentionally delegated to
``marni_fib_core_combo_cache.process_day``.  This module only adapts the
ammu/index-options CSV layout to the cache-shaped payload expected by that
engine and provides bounded evaluation, masks, WFO hooks, and ranking.

No historical sweep runs on import or by default.  The CLI requires
``--smoke`` for a bounded run or ``--trials N --allow-expensive`` for an
explicit optimization request.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid import marni_fib_core_combo_cache as smart_core
from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS


DEFAULT_DATA_ROOT = Path(os.environ.get("AMMU_DATA_ROOT", r"C:\Websites\ammu"))
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-05-05"
DEFAULT_IS_END = "2023-12-31"
DEFAULT_OOS_START = "2024-01-01"
LOT_SIZE = smart_core.LOT_SIZE
HISTORICAL_OPTION_RE = re.compile(
    r"^nifty_options_(?P<day>\d{2})_(?P<month>\d{2})_(?P<year>\d{4})\.csv$",
    re.IGNORECASE,
)
CONTRACT_RE = re.compile(
    r"^(?P<prefix>NIFTY\d{2}[A-Z]{3}\d{2})(?P<strike>\d+)(?P<side>CE|PE)$"
)


# Tunable signal/exit fields are deliberately small and economically bounded.
# Execution fields below are fixed by the imported Smart Fib simulator.
SMART_FIB_PARAMETER_SCHEMA: dict[str, dict[str, Any]] = {
    "timeframe": {
        "kind": "categorical",
        "choices": ["1m", "2m", "3m", "5m", "combined"],
        "default": "combined",
    },
    "target_level": {
        "kind": "categorical",
        "choices": [0.0, 0.29],
        "default": 0.29,
    },
    "stop_level": {
        "kind": "categorical",
        "choices": [1.079, 1.155, 1.25],
        "default": 1.155,
    },
    "fallback_target_level": {
        "kind": "categorical",
        "choices": [0.0, 0.29],
        "default": 0.0,
    },
    "option_point_threshold": {
        "kind": "float",
        "low": 5.0,
        "high": 25.0,
        "step": 5.0,
        "default": 10.0,
    },
    "min_span": {
        "kind": "float",
        "low": 15.0,
        "high": 30.0,
        "step": 5.0,
        "default": 15.0,
    },
    "touch_buffer": {
        "kind": "float",
        "low": 0.0,
        "high": 2.0,
        "step": 0.5,
        "default": 0.0,
    },
    "setup_max_age": {
        "kind": "int",
        "low": 15,
        "high": 75,
        "step": 15,
        "default": 45,
    },
}

FIXED_EXECUTION_SCHEMA: dict[str, Any] = {
    "lot_size": LOT_SIZE,
    "slippage_points_per_side": SLIPPAGE_PTS,
    "brokerage_per_order": BROKERAGE_PER_ORDER,
    "option_delta_metadata": smart_core.OPTION_DELTA,
    "concurrent_positions": False,
    "strike_selection": "dynamic ATM/ITM candidates from trigger-minute spot",
    "contract_source": "actual option CSV symbol rows",
}


@dataclass(frozen=True)
class SmartFibParams:
    """One causal Smart Fib configuration."""

    timeframe: str = "combined"
    target_level: float = 0.29
    stop_level: float = 1.155
    fallback_target_level: float = 0.0
    option_point_threshold: float = 10.0
    min_span: float = 15.0
    touch_buffer: float = 0.0
    setup_max_age: int = 45

    def validate(self) -> "SmartFibParams":
        if self.timeframe not in SMART_FIB_PARAMETER_SCHEMA["timeframe"]["choices"]:
            raise ValueError(f"unsupported timeframe: {self.timeframe}")
        if not 0.0 <= self.target_level <= 1.0:
            raise ValueError("target_level must be between 0 and 1")
        if not 0.0 <= self.fallback_target_level <= 1.0:
            raise ValueError("fallback_target_level must be between 0 and 1")
        if self.fallback_target_level > self.target_level:
            raise ValueError("fallback_target_level cannot exceed target_level")
        if self.stop_level <= 1.0:
            raise ValueError("stop_level must be an extension above 1.0")
        if self.option_point_threshold < 0.0:
            raise ValueError("option_point_threshold must be non-negative")
        if self.min_span <= 0.0:
            raise ValueError("min_span must be positive")
        if self.touch_buffer < 0.0:
            raise ValueError("touch_buffer must be non-negative")
        if self.setup_max_age < 0:
            raise ValueError("setup_max_age must be non-negative")
        return self

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SmartFibParams":
        fields = {
            field: values[field]
            for field in cls.__dataclass_fields__
            if field in values
        }
        return cls(**fields).validate()


def _day_key(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _time_text(target_day: str, minute: int) -> str:
    target = date.fromisoformat(target_day)
    return datetime.combine(
        target, dt_time(minute // 60, minute % 60)
    ).strftime("%d-%m-%Y %H:%M:%S")


def _timestamp_text(value: Any) -> str:
    # Kept local to the adapter so core.parse_row continues to use its exact
    # broker-cache timestamp format.
    if isinstance(value, str):
        return datetime.fromisoformat(value).strftime("%d-%m-%Y %H:%M:%S")
    import pandas as pd

    return pd.Timestamp(value).strftime("%d-%m-%Y %H:%M:%S")


class CsvHistoricalDataAdapter:
    """Adapt ammu CSVs to the Flattrade day-cache contract.

    The adapter loads the index once, discovers option files by date, and reads
    only the current day's dynamically reachable contracts plus the preceding
    available trading day's rows needed to warm the causal Smart Fib state.
    ``spot_by_day`` and ``option_files_by_day`` allow an equivalent repository
    loader to be injected without changing the evaluator.
    """

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        *,
        start: str = DEFAULT_START,
        end: str = DEFAULT_END,
        index_path: str | Path | None = None,
        option_files_by_day: Mapping[Any, str | Path] | None = None,
        spot_by_day: Mapping[Any, Mapping[str, Any]] | None = None,
        cache_days: int = 8,
    ) -> None:
        self.data_root = Path(data_root)
        self.start = _day_key(start)
        self.end = _day_key(end)
        self.index_path = Path(index_path) if index_path else self.data_root / "index" / "NIFTY 50_minute.csv"
        self.options_root = self.data_root / "nifty_options"
        self.cache_days = max(0, int(cache_days))
        self._day_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._option_files = (
            {_day_key(day): Path(path) for day, path in option_files_by_day.items()}
            if option_files_by_day is not None
            else self._discover_option_files()
        )
        self._spot_by_day = (
            self._normalize_spot_loader(spot_by_day)
            if spot_by_day is not None
            else self._load_spot_csv()
        )
        self._available_all = sorted(
            set(self._spot_by_day).intersection(self._option_files)
        )

    @classmethod
    def from_repository_loaders(
        cls,
        spot_loader: Callable[[], Mapping[Any, Mapping[str, Any]]],
        option_file_loader: Callable[[str, str], Mapping[Any, str | Path]],
        *,
        start: str = DEFAULT_START,
        end: str = DEFAULT_END,
        **kwargs: Any,
    ) -> "CsvHistoricalDataAdapter":
        """Build an adapter from existing repository loader functions."""
        return cls(
            start=start,
            end=end,
            spot_by_day=spot_loader(),
            option_files_by_day=option_file_loader(start, end),
            **kwargs,
        )

    def _load_spot_csv(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.exists():
            raise FileNotFoundError(f"index CSV not found: {self.index_path}")
        import pandas as pd

        frame = pd.read_csv(
            self.index_path,
            usecols=["date", "open", "high", "low", "close"],
            parse_dates=["date"],
        )
        frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
        frame["day"] = frame["date"].dt.strftime("%Y-%m-%d")
        lower = min(self._option_files, default=self.start)
        frame = frame[(frame["day"] >= lower) & (frame["day"] <= self.end)]
        output: dict[str, dict[str, Any]] = {}
        for day, group in frame.groupby("day", sort=False):
            output[day] = {
                "time": group["date"].to_numpy(),
                "open": group["open"].to_numpy(dtype=float),
                "high": group["high"].to_numpy(dtype=float),
                "low": group["low"].to_numpy(dtype=float),
                "close": group["close"].to_numpy(dtype=float),
            }
        return output

    @staticmethod
    def _normalize_spot_loader(
        spot_by_day: Mapping[Any, Mapping[str, Any]],
    ) -> dict[str, dict[str, list[Any]]]:
        output: dict[str, dict[str, list[Any]]] = {}
        required = ("min", "open", "high", "low", "close")
        for raw_day, series in spot_by_day.items():
            missing = [name for name in required if name not in series]
            if missing:
                raise ValueError(
                    f"spot loader for {_day_key(raw_day)} lacks OHLC fields: {missing}"
                )
            day = _day_key(raw_day)
            output[day] = {
                "time": [
                    _time_text(day, int(minute))
                    for minute in series["min"]
                ],
                "open": [float(value) for value in series["open"]],
                "high": [float(value) for value in series["high"]],
                "low": [float(value) for value in series["low"]],
                "close": [float(value) for value in series["close"]],
            }
        return output

    def _discover_option_files(self) -> dict[str, Path]:
        if not self.options_root.exists():
            raise FileNotFoundError(f"options directory not found: {self.options_root}")
        output: dict[str, Path] = {}
        for path in sorted(self.options_root.rglob("*.csv")):
            match = HISTORICAL_OPTION_RE.match(path.name)
            if not match:
                continue
            day = (
                f"{match['year']}-{match['month']}-{match['day']}"
            )
            output[day] = path
        return output

    def available_days(self, start: str | None = None, end: str | None = None) -> list[str]:
        lower = _day_key(start or self.start)
        upper = _day_key(end or self.end)
        return [day for day in self._available_all if lower <= day <= upper]

    def _previous_available_day(self, day: str) -> str | None:
        candidates = [candidate for candidate in self._available_all if candidate < day]
        return candidates[-1] if candidates else None

    def _spot_rows(self, day: str) -> list[dict[str, Any]]:
        series = self._spot_by_day.get(day)
        if series is None:
            return []
        return [
            {
                "time": _timestamp_text(timestamp),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
            }
            for timestamp, open_, high, low, close in zip(
                series["time"],
                series["open"],
                series["high"],
                series["low"],
                series["close"],
            )
        ]

    @staticmethod
    def _candidate_keys(spot_rows: Sequence[Mapping[str, Any]]) -> set[str]:
        keys: set[str] = set()
        for row in spot_rows:
            price = float(row["close"])
            atm = int(round(price / 50.0) * 50)
            for side, offsets in (("CE", (0, -50, -100)), ("PE", (0, 50, 100))):
                keys.update(f"{side}:{atm + offset}" for offset in offsets)
        return keys

    @staticmethod
    def _read_option_rows(
        path: Path,
        *,
        candidate_keys: set[str] | None = None,
        symbols: set[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        try:
            import polars as pl

            frame = pl.read_csv(
                path,
                columns=["date", "time", "symbol", "open", "high", "low", "close"],
                schema_overrides={"symbol": pl.Utf8},
                try_parse_dates=False,
            )
            if frame.is_empty():
                return {}
            frame = frame.with_columns(pl.col("symbol").cast(pl.Utf8).alias("_symbol"))
            if symbols is not None:
                frame = frame.filter(pl.col("_symbol").is_in(list(symbols)))
            frame = frame.with_columns([
                pl.col("_symbol").str.extract(CONTRACT_RE.pattern, 2).alias("_strike"),
                pl.col("_symbol").str.extract(CONTRACT_RE.pattern, 3).alias("_side"),
            ]).with_columns(
                (pl.col("_side") + ":" + pl.col("_strike")).alias("_key")
            )
            if candidate_keys is not None:
                frame = frame.filter(pl.col("_key").is_in(list(candidate_keys)))
            if frame.is_empty():
                return {}
            frame = frame.with_columns([
                pl.concat_str([
                    pl.col("date").cast(pl.Utf8),
                    pl.lit(" "),
                    pl.col("time").cast(pl.Utf8),
                ]).alias("_timestamp_text"),
                pl.concat_str([
                    pl.col("date").cast(pl.Utf8),
                    pl.lit(" "),
                    pl.col("time").cast(pl.Utf8),
                ]).str.to_datetime(
                    format="%Y-%m-%d %H:%M:%S", strict=False
                ).alias("_timestamp"),
            ]).filter(pl.col("_timestamp").is_not_null()).sort(
                ["_symbol", "_timestamp"]
            )
            rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
            for group in frame.partition_by("_symbol", maintain_order=True):
                symbol = str(group["_symbol"][0])
                rows = []
                for row in group.select(
                    ["_timestamp_text", "open", "high", "low", "close"]
                ).iter_rows(named=True):
                    values = (row["open"], row["high"], row["low"], row["close"])
                    if any(value is None for value in values):
                        continue
                    rows.append({
                        "time": _timestamp_text(row["_timestamp_text"]),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                    })
                if rows:
                    rows_by_symbol[symbol] = rows
            return rows_by_symbol
        except ImportError:
            import pandas as pd

            frame = pd.read_csv(
                path,
                usecols=["date", "time", "symbol", "open", "high", "low", "close"],
                dtype={"symbol": "string"},
            )
            if frame.empty:
                return {}
            frame["_symbol"] = frame["symbol"].astype(str)
            if symbols is not None:
                frame = frame[frame["_symbol"].isin(symbols)]
            if frame.empty:
                return {}
            parts = frame["_symbol"].str.extract(CONTRACT_RE)
            frame["_key"] = parts["side"] + ":" + parts["strike"]
            if candidate_keys is not None:
                frame = frame[frame["_key"].isin(candidate_keys)]
            if frame.empty:
                return {}
            frame["_timestamp"] = pd.to_datetime(
                frame["date"].astype(str) + " " + frame["time"].astype(str),
                errors="coerce",
            )
            frame = frame.dropna(subset=["_timestamp"])
            rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
            for symbol, group in frame.groupby("_symbol", sort=False):
                rows = []
                for timestamp, open_, high, low, close in group[
                    ["_timestamp", "open", "high", "low", "close"]
                ].sort_values("_timestamp").itertuples(index=False, name=None):
                    values = (open_, high, low, close)
                    if any(pd.isna(value) for value in values):
                        continue
                    rows.append({
                        "time": _timestamp_text(timestamp),
                        "open": float(open_),
                        "high": float(high),
                        "low": float(low),
                        "close": float(close),
                    })
                if rows:
                    rows_by_symbol[str(symbol)] = rows
            return rows_by_symbol

    def _build_day_cache(self, resolved_day: str) -> dict[str, Any] | None:
        spot_rows = self._spot_rows(resolved_day)
        option_path = self._option_files.get(resolved_day)
        if not spot_rows or option_path is None:
            return None

        candidate_keys = self._candidate_keys(spot_rows)
        current = self._read_option_rows(
            option_path,
            candidate_keys=candidate_keys,
        )
        if not current:
            return None

        previous_day = self._previous_available_day(resolved_day)
        previous: dict[str, list[dict[str, Any]]] = {}
        if previous_day is not None:
            previous_path = self._option_files.get(previous_day)
            if previous_path is not None:
                previous = self._read_option_rows(
                    previous_path,
                    symbols=set(current),
                )

        contracts: dict[str, dict[str, Any]] = {}
        for symbol, current_rows in current.items():
            match = CONTRACT_RE.match(symbol)
            if match is None:
                continue
            key = f"{match['side']}:{int(match['strike'])}"
            contracts[key] = {
                "tsym": symbol,
                "rows": previous.get(symbol, []) + current_rows,
            }
        if not contracts:
            return None
        return {
            "schema_version": 1,
            "date": resolved_day,
            "interval": "1",
            "spot_rows": spot_rows,
            "contracts": contracts,
        }

    def load_day_cache(self, _cache_dir: Path | str, target: date) -> dict[str, Any] | None:
        """Callback matching ``artifacts.flattrade_day_cache.load_day_cache``."""
        requested_day = target.isoformat()
        resolved_day = (
            requested_day
            if requested_day in self._available_all
            else self._previous_available_day(requested_day)
        )
        if resolved_day is None:
            return None
        cached = self._day_cache.get(resolved_day)
        if cached is not None:
            self._day_cache.move_to_end(resolved_day)
            return cached
        payload = self._build_day_cache(resolved_day)
        if payload is None:
            return None
        if self.cache_days:
            self._day_cache[resolved_day] = payload
            self._day_cache.move_to_end(resolved_day)
            while len(self._day_cache) > self.cache_days:
                self._day_cache.popitem(last=False)
        return payload

    def clear_cache(self) -> None:
        self._day_cache.clear()


def _range_mask(
    days: Sequence[str],
    start: str,
    end: str,
) -> tuple[bool, ...]:
    lower, upper = _day_key(start), _day_key(end)
    return tuple(lower <= day <= upper for day in days)


def nw_mask(
    days: Sequence[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> tuple[bool, ...]:
    """Return the non-walk-forward/full historical window mask."""
    return _range_mask(days, start, end)


def is_mask(
    days: Sequence[str],
    end: str = DEFAULT_IS_END,
) -> tuple[bool, ...]:
    return _range_mask(days, DEFAULT_START, end)


def oos_mask(
    days: Sequence[str],
    start: str = DEFAULT_OOS_START,
    end: str = DEFAULT_END,
) -> tuple[bool, ...]:
    return _range_mask(days, start, end)


def select_days(
    days: Sequence[str],
    mask: Sequence[bool] | set[str] | frozenset[str] | Callable[[str], bool] | None,
) -> list[str]:
    if mask is None:
        return list(days)
    if callable(mask):
        return [day for day in days if mask(day)]
    if len(mask) == len(days) and all(isinstance(value, bool) for value in mask):
        return [day for day, include in zip(days, mask) if include]
    allowed = set(mask)
    return [day for day in days if day in allowed]


@dataclass(frozen=True)
class WFOFold:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str


DEFAULT_WFO_FOLDS: tuple[WFOFold, ...] = (
    WFOFold("2021", "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    WFOFold("2022", "2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    WFOFold("2023", "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    WFOFold("2024", "2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    WFOFold("2025", "2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    WFOFold("2026", "2020-01-01", "2025-12-31", "2026-01-01", DEFAULT_END),
)


def rolling_wfo_folds(
    folds: Sequence[WFOFold] = DEFAULT_WFO_FOLDS,
    *,
    end: str = DEFAULT_END,
) -> tuple[WFOFold, ...]:
    """Return documented folds clipped to the available historical end date."""
    upper = _day_key(end)
    return tuple(
        WFOFold(
            fold.name,
            fold.train_start,
            fold.train_end,
            fold.validation_start,
            min(fold.validation_end, upper),
        )
        for fold in folds
        if fold.validation_start <= upper and fold.train_start <= upper
    )


def _trade_stats(trades: Sequence[Mapping[str, Any]], days_count: int) -> dict[str, Any]:
    ordered = sorted(
        trades,
        key=lambda trade: (
            str(trade.get("date", "")),
            int(trade.get("entry_min", 0)),
            int(trade.get("exit_min", 0)),
        ),
    )
    wins = [trade for trade in ordered if float(trade.get("rs_net", 0.0)) > 0.0]
    losses = [trade for trade in ordered if float(trade.get("rs_net", 0.0)) <= 0.0]
    net_rs = sum(float(trade.get("rs_net", 0.0)) for trade in ordered)
    net_points = sum(float(trade.get("points", 0.0)) for trade in ordered)
    gross_wins = sum(float(trade.get("rs_net", 0.0)) for trade in wins)
    gross_losses = abs(sum(float(trade.get("rs_net", 0.0)) for trade in losses))
    equity = 0.0
    peak = 0.0
    max_dd_rs = 0.0
    for trade in ordered:
        equity += float(trade.get("rs_net", 0.0))
        peak = max(peak, equity)
        max_dd_rs = max(max_dd_rs, peak - equity)
    profit_factor = gross_wins / gross_losses if gross_losses else (float("inf") if gross_wins else 0.0)
    return {
        "trades": len(ordered),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(ordered) * 100.0, 2) if ordered else 0.0,
        "net_rs": round(net_rs, 2),
        "net_points": round(net_points, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else profit_factor,
        "max_drawdown_rs": round(max_dd_rs, 2),
        "max_drawdown_points": round(max_dd_rs / LOT_SIZE, 2),
        "fees_rs": round(sum(float(trade.get("fee", 0.0)) for trade in ordered), 2),
        "avg_trades_per_day": round(len(ordered) / days_count, 3) if days_count else 0.0,
    }


@dataclass
class DayEvaluation:
    day: str
    params: SmartFibParams
    trades: list[dict[str, Any]]
    stats: dict[str, Any]

    def as_dict(self, include_trades: bool = True) -> dict[str, Any]:
        output = {
            "day": self.day,
            "params": asdict(self.params),
            "stats": self.stats,
        }
        if include_trades:
            output["trades"] = self.trades
        return output


@dataclass
class HistoryEvaluation:
    params: SmartFibParams
    days: tuple[str, ...]
    trades: list[dict[str, Any]]
    stats: dict[str, Any]
    score: float = 0.0

    def as_dict(self, include_trades: bool = False) -> dict[str, Any]:
        output = {
            "params": asdict(self.params),
            "days": list(self.days),
            "stats": self.stats,
            "score": self.score,
        }
        if include_trades:
            output["trades"] = self.trades
        return output


def evaluate_day(
    adapter: CsvHistoricalDataAdapter,
    day: str,
    params: SmartFibParams | Mapping[str, Any] = SmartFibParams(),
) -> DayEvaluation:
    """Evaluate exactly one day through the unchanged Smart Fib core path."""
    params = params if isinstance(params, SmartFibParams) else SmartFibParams.from_mapping(params)
    params.validate()
    output = smart_core.process_day(
        day,
        (params.timeframe,),
        (params.target_level,),
        (params.stop_level,),
        cache_loader=adapter.load_day_cache,
        min_span=params.min_span,
        touch_buffer=params.touch_buffer,
        setup_max_age=params.setup_max_age,
        option_point_threshold=params.option_point_threshold,
        fallback_target_level=params.fallback_target_level,
        debug=False,
    )
    key = f"smart-fib|{params.timeframe}|tp{params.target_level}|sl{params.stop_level}"
    trades = list(output.get(key, []))
    return DayEvaluation(day, params, trades, _trade_stats(trades, 1))


def evaluate_history(
    adapter: CsvHistoricalDataAdapter,
    days: Sequence[str],
    params: SmartFibParams | Mapping[str, Any] = SmartFibParams(),
    *,
    mask: Sequence[bool] | set[str] | frozenset[str] | Callable[[str], bool] | None = None,
) -> HistoryEvaluation:
    """Evaluate a bounded day set; no implicit full-period expansion occurs."""
    params = params if isinstance(params, SmartFibParams) else SmartFibParams.from_mapping(params)
    selected = select_days(list(days), mask)
    trades: list[dict[str, Any]] = []
    for day in selected:
        trades.extend(evaluate_day(adapter, day, params).trades)
    stats = _trade_stats(trades, len(selected))
    return HistoryEvaluation(params, tuple(selected), trades, stats)


def composite_score(
    stats: Mapping[str, Any],
    *,
    drawdown_penalty: float = 0.20,
    min_trades: int = 1,
) -> float:
    """Rank net points while charging a linear max-drawdown penalty."""
    if int(stats.get("trades", 0)) < min_trades:
        return float("-inf")
    return float(stats.get("net_points", 0.0)) - drawdown_penalty * float(
        stats.get("max_drawdown_points", 0.0)
    )


def rank_results(
    results: Sequence[HistoryEvaluation],
    *,
    drawdown_penalty: float = 0.20,
    min_trades: int = 1,
) -> list[HistoryEvaluation]:
    for result in results:
        result.score = composite_score(
            result.stats,
            drawdown_penalty=drawdown_penalty,
            min_trades=min_trades,
        )
    return sorted(results, key=lambda result: result.score, reverse=True)


def suggest_parameters(trial: Any) -> SmartFibParams:
    """Optuna-compatible parameter suggestion hook; does not start a study."""
    target_level = trial.suggest_categorical("target_level", [0.0, 0.29])
    return SmartFibParams(
        timeframe=trial.suggest_categorical("timeframe", ["1m", "2m", "3m", "5m", "combined"]),
        target_level=target_level,
        stop_level=trial.suggest_categorical("stop_level", [1.079, 1.155, 1.25]),
        fallback_target_level=trial.suggest_categorical(
            "fallback_target_level",
            [level for level in [0.0, 0.29] if level <= target_level],
        ),
        option_point_threshold=trial.suggest_float("option_point_threshold", 5.0, 25.0, step=5.0),
        min_span=trial.suggest_float("min_span", 15.0, 30.0, step=5.0),
        touch_buffer=trial.suggest_float("touch_buffer", 0.0, 2.0, step=0.5),
        setup_max_age=trial.suggest_int("setup_max_age", 15, 75, step=15),
    ).validate()


def top_trial_summaries(study: Any, limit: int = 5) -> list[dict[str, Any]]:
    """Return completed trials ordered by the configured composite score."""
    completed = [
        trial
        for trial in study.trials
        if getattr(getattr(trial, "state", None), "name", "") == "COMPLETE"
        and trial.value is not None
    ]
    completed.sort(key=lambda trial: float(trial.value), reverse=True)
    return [
        {
            "trial": trial.number,
            "score": round(float(trial.value), 4),
            "params": dict(trial.params),
            "stats": trial.user_attrs.get("stats", {}),
        }
        for trial in completed[:limit]
    ]


def optimize(
    adapter: CsvHistoricalDataAdapter,
    train_days: Sequence[str],
    *,
    n_trials: int,
    drawdown_penalty: float = 0.20,
    min_trades: int = 10,
    seed: int = 42,
    oos_start: str | None = None,
) -> tuple[Any, HistoryEvaluation]:
    """Run an explicit Optuna study on the supplied training days only."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive; use --smoke for a bounded evaluation")
    if oos_start is not None and any(day >= _day_key(oos_start) for day in train_days):
        raise ValueError(
            f"optimizer training days must be strictly before OOS start {_day_key(oos_start)}"
        )
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is required only when an explicit sweep is requested") from exc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    def objective(trial: Any) -> float:
        params = suggest_parameters(trial)
        result = evaluate_history(adapter, train_days, params)
        score = composite_score(
            result.stats,
            drawdown_penalty=drawdown_penalty,
            min_trades=min_trades,
        )
        trial.set_user_attr("stats", result.stats)
        return score

    study.optimize(objective, n_trials=n_trials)
    best_params = SmartFibParams.from_mapping(study.best_trial.params)
    best_result = evaluate_history(adapter, train_days, best_params)
    best_result.score = composite_score(
        best_result.stats,
        drawdown_penalty=drawdown_penalty,
        min_trades=min_trades,
    )
    return study, best_result


def optimize_wfo(
    adapter: CsvHistoricalDataAdapter,
    *,
    n_trials: int,
    folds: Sequence[WFOFold] = DEFAULT_WFO_FOLDS,
    drawdown_penalty: float = 0.20,
    min_trades: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Select parameters on each fold's train window and stitch OOS only."""
    fold_results: list[dict[str, Any]] = []
    stitched_oos: list[dict[str, Any]] = []
    for fold in rolling_wfo_folds(folds, end=adapter.end):
        train_days = adapter.available_days(fold.train_start, fold.train_end)
        validation_days = adapter.available_days(
            fold.validation_start, fold.validation_end
        )
        if not train_days or not validation_days:
            continue
        study, train_result = optimize(
            adapter,
            train_days,
            n_trials=n_trials,
            drawdown_penalty=drawdown_penalty,
            min_trades=min_trades,
            seed=seed + len(fold_results),
            oos_start=fold.validation_start,
        )
        selected_params = train_result.params
        validation_result = evaluate_history(adapter, validation_days, selected_params)
        validation_result.score = composite_score(
            validation_result.stats,
            drawdown_penalty=drawdown_penalty,
            min_trades=1,
        )
        stitched_oos.extend(validation_result.trades)
        fold_results.append(
            {
                "fold": asdict(fold),
                "train_days": [train_days[0], train_days[-1]],
                "validation_days": [validation_days[0], validation_days[-1]],
                "selected_params": asdict(selected_params),
                "train": train_result.as_dict(),
                "validation": validation_result.as_dict(),
                "top_trials": top_trial_summaries(study),
            }
        )
    oos_days = sorted({str(trade.get("date")) for trade in stitched_oos})
    oos_stats = _trade_stats(stitched_oos, len(oos_days))
    return {
        "folds": fold_results,
        "stitched_oos": {
            "days": oos_days,
            "stats": oos_stats,
            "score": composite_score(oos_stats, drawdown_penalty=drawdown_penalty),
        },
    }


def evaluate_wfo(
    adapter: CsvHistoricalDataAdapter,
    base_params: SmartFibParams | Mapping[str, Any] = SmartFibParams(),
    *,
    folds: Sequence[WFOFold] = DEFAULT_WFO_FOLDS,
    parameter_selector: Callable[[list[str], WFOFold], SmartFibParams | Mapping[str, Any]] | None = None,
    drawdown_penalty: float = 0.20,
) -> dict[str, Any]:
    """Walk-forward hook with train-only selection and OOS stitching.

    ``parameter_selector`` may call ``optimize`` on the fold's training days.
    With no selector this is a fixed-parameter parity harness, not a hidden
    re-fit. OOS days are never passed to the selector.
    """
    base_params = base_params if isinstance(base_params, SmartFibParams) else SmartFibParams.from_mapping(base_params)
    fold_results: list[dict[str, Any]] = []
    stitched_oos: list[dict[str, Any]] = []
    for fold in rolling_wfo_folds(folds, end=adapter.end):
        train_days = adapter.available_days(fold.train_start, fold.train_end)
        validation_days = adapter.available_days(fold.validation_start, fold.validation_end)
        selected_params = (
            parameter_selector(train_days, fold)
            if parameter_selector is not None
            else base_params
        )
        selected_params = (
            selected_params
            if isinstance(selected_params, SmartFibParams)
            else SmartFibParams.from_mapping(selected_params)
        )
        train_result = evaluate_history(adapter, train_days, selected_params)
        validation_result = evaluate_history(adapter, validation_days, selected_params)
        stitched_oos.extend(validation_result.trades)
        fold_results.append(
            {
                "fold": asdict(fold),
                "params": asdict(selected_params),
                "train": train_result.as_dict(),
                "validation": validation_result.as_dict(),
            }
        )
    oos_day_count = len({str(trade.get("date")) for trade in stitched_oos})
    oos_stats = _trade_stats(stitched_oos, oos_day_count)
    return {
        "folds": fold_results,
        "stitched_oos": {
            "days": sorted({str(trade.get("date")) for trade in stitched_oos}),
            "stats": oos_stats,
            "score": composite_score(oos_stats, drawdown_penalty=drawdown_penalty),
        },
    }


def _build_params(args: argparse.Namespace) -> SmartFibParams:
    return SmartFibParams(
        timeframe=args.timeframe,
        target_level=args.target_level,
        stop_level=args.stop_level,
        fallback_target_level=args.fallback_target_level,
        option_point_threshold=args.option_point_threshold,
        min_span=args.min_span,
        touch_buffer=args.touch_buffer,
        setup_max_age=args.setup_max_age,
    ).validate()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--timeframe", choices=["1m", "2m", "3m", "5m", "combined"], default="combined")
    parser.add_argument("--target-level", type=float, default=0.29)
    parser.add_argument("--stop-level", type=float, default=1.155)
    parser.add_argument("--fallback-target-level", type=float, default=0.0)
    parser.add_argument("--option-point-threshold", type=float, default=10.0)
    parser.add_argument("--min-span", type=float, default=15.0)
    parser.add_argument("--touch-buffer", type=float, default=0.0)
    parser.add_argument("--setup-max-age", type=int, default=45)
    parser.add_argument("--smoke", action="store_true", help="evaluate only the first five available days")
    parser.add_argument("--smoke-days", type=int, default=5)
    parser.add_argument("--trials", type=int, default=0, help="explicit Optuna trial count")
    parser.add_argument(
        "--wfo",
        action="store_true",
        help="run annual rolling train-only selection and stitched OOS evaluation",
    )
    parser.add_argument("--allow-expensive", action="store_true", help="confirm an explicit historical sweep")
    parser.add_argument("--drawdown-penalty", type=float, default=0.20)
    parser.add_argument("--min-trades", type=int, default=10)
    args = parser.parse_args(argv)

    if not args.smoke and args.trials <= 0:
        parser.error("refusing an unbounded run; use --smoke or --trials N --allow-expensive")
    if args.trials > 0 and not args.allow_expensive:
        parser.error("--trials requires --allow-expensive; no full sweep is run by default")
    if args.wfo and args.trials <= 0:
        parser.error("--wfo requires --trials N --allow-expensive")

    adapter = CsvHistoricalDataAdapter(
        args.data_root,
        start=args.start,
        end=args.end,
    )
    days = adapter.available_days(args.start, args.end)
    if not days:
        parser.error(f"no overlapping index/options days in {args.start}..{args.end}")
    params = _build_params(args)

    if args.wfo:
        wfo = optimize_wfo(
            adapter,
            n_trials=args.trials,
            drawdown_penalty=args.drawdown_penalty,
            min_trades=args.min_trades,
        )
        print(json.dumps({
            "mode": "walk_forward",
            "trials_per_fold": args.trials,
            "result": wfo,
            "fixed_execution": FIXED_EXECUTION_SCHEMA,
        }, indent=2, default=str))
        return 0

    if args.trials > 0:
        train_days = adapter.available_days(args.start, args.end)
        if not train_days:
            parser.error(f"no optimization days in {args.start}..{args.end}")
        study, result = optimize(
            adapter,
            train_days,
            n_trials=args.trials,
            drawdown_penalty=args.drawdown_penalty,
            min_trades=args.min_trades,
        )
        print(json.dumps({
            "mode": "optimize",
            "trials": args.trials,
            "train_days": [train_days[0], train_days[-1]],
            "best_trial": study.best_trial.number,
            "top_trials": top_trial_summaries(study),
            "result": result.as_dict(),
            "fixed_execution": FIXED_EXECUTION_SCHEMA,
        }, indent=2, default=str))
        return 0

    smoke_days = days[: min(5, max(1, args.smoke_days))]
    result = evaluate_history(adapter, smoke_days, params)
    result.score = composite_score(
        result.stats,
        drawdown_penalty=args.drawdown_penalty,
        min_trades=1,
    )
    print(json.dumps({
        "mode": "smoke",
        "days": smoke_days,
        "result": result.as_dict(),
        "fixed_execution": FIXED_EXECUTION_SCHEMA,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
