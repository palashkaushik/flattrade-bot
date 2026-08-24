"""B17 Smart Fib Combined 4-TF live strategy (1m/2m/3m/5m union).

Ports the B17 combined-timeframe champion exactly as validated in
``artifacts/f6_hybrid/smart_fib_combined_tf_gpu.py``:

  - Per-TF causal event extraction via ``extract_day_events`` with
    ``bar_minutes=tf`` and ``filter_period=5*tf`` (champion params).
  - One event per minute per TF (global-slot ordering), merged across
    TFs sorted by (minute, tf-priority 1 < 2 < 3 < 5) and deduped by
    (minute, side, symbol).
  - Exit levels: target 0.786 / stop 1.13 fib extensions of the event
    swing, monitored on the index (index-source events) or the option
    (option-source events), with ``price_profit_on_rise`` direction.
  - Session window 09:20 - 15:00, single global position, EOD at 15:00.

The live cache loader synthesizes the day-cache payload from TPSeries
spot + option rows, so the extraction code path is byte-identical to the
backtest.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from artifacts.f6_hybrid.marni_fib_core_combo_cache import (
    active_strike_candidates,
    extract_day_events,
)
from artifacts.f6_hybrid.marni_fib_backtest import fib_price

logger = logging.getLogger(__name__)

# B17 champion constants (mirror smart_fib_combined_tf_gpu.py)
CHAMPION = dict(
    min_span=15.0,
    touch_buffer=0.5,
    setup_max_age=45,
    zone_start=0.5,
    zone_end=0.786,
    s1_k_period=12,
    s1_d_period=4,
)
TIMEFRAMES: Tuple[int, ...] = (1, 2, 3, 5)
TARGET_LEVEL = 0.786
STOP_LEVEL = 1.13
SESSION_START_MINUTE = 560   # 09:20
SESSION_END_MINUTE = 900     # 15:00


def _session_events(payload: Dict[str, Any]) -> List[dict]:
    """One reachable event per bar (mirrors optimus._select_day_events).

    Tolerant variant: skips signals without an actual option row instead
    of raising, which is the live-data equivalent of the backtest's
    ParityError (the backtest cache guarantees rows; live feeds may lag).
    """
    selected: List[dict] = []
    seen_minutes: set = set()
    bars = payload["bars"]
    for signal in payload["signals"]:
        minute = int(signal["minute"])
        if minute in seen_minutes:
            continue
        if not SESSION_START_MINUTE <= minute < SESSION_END_MINUTE:
            continue
        key = (str(signal["side"]), int(signal["strike"]))
        if key not in bars or minute not in bars[key]:
            continue
        seen_minutes.add(minute)
        selected.append(dict(signal))
    return selected


def _row_minute(row: Dict[str, Any]) -> int:
    """Minute-of-day of a row whose time is 'DD-MM-YYYY HH:MM:SS'."""
    try:
        dt = datetime.strptime(str(row.get("time", "")), "%d-%m-%Y %H:%M:%S")
        return dt.hour * 60 + dt.minute
    except (ValueError, TypeError):
        return 0


class LiveSmartFibCombinedStrategy:
    """Causal live B17 event stream over growing spot + option row buffers."""

    def __init__(self, timeframes: Sequence[int] = TIMEFRAMES):
        self.timeframes = tuple(sorted(timeframes))
        self.today: Optional[date] = None
        self.spot_rows: Dict[str, List[dict]] = {}       # date -> parsed rows
        self.contract_rows: Dict[str, Dict[str, Any]] = {}  # "side:strike" -> {"rows":[...], "tsym":..., "token":...}
        self.last_evaluated_minute: int = -1
        self._slice_minute: int = 10_000
        self.per_tf_counts: Dict[int, int] = {}
        self.source_tf_by_minute: Dict[int, int] = {}
        self._evaluations = 0

    # ── Row ingestion ────────────────────────────────────────────────

    def set_today(self, day: date) -> None:
        self.today = day

    def _row_key(self, row: Dict[str, Any]) -> str:
        """ISO date of a row whose time is 'DD-MM-YYYY HH:MM:SS'."""
        day = str(row.get("time", "")).split(" ")[0]
        try:
            return datetime.strptime(day, "%d-%m-%Y").date().isoformat()
        except (ValueError, TypeError):
            return day

    def add_spot_rows(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Adds parsed spot rows (time '%d-%m-%Y %H:%M:%S'); returns count added."""
        added = 0
        for row in rows:
            day = self._row_key(row)
            bucket = self.spot_rows.setdefault(day, [])
            times = {r["time"] for r in bucket}
            if row.get("time") not in times:
                bucket.append(dict(row))
                added += 1
        return added

    def add_contract_rows(
        self,
        side: str,
        strike: int,
        tsym: str,
        token: str,
        rows: Sequence[Dict[str, Any]],
    ) -> int:
        """Adds option rows for a contract; dedupes by row time."""
        key = f"{side}:{int(strike)}"
        entry = self.contract_rows.setdefault(
            key, {"rows": [], "tsym": tsym, "token": token}
        )
        entry["tsym"] = tsym
        entry["token"] = token
        added = 0
        known = {r["time"] for r in entry["rows"]}
        for row in rows:
            if row.get("time") not in known:
                entry["rows"].append(dict(row))
                known.add(row["time"])
                added += 1
        return added

    def candidate_strikes(self, spot_close: float, minute: int) -> Tuple[int, ...]:
        """CE + PE candidate strikes for the current spot snapshot."""
        if not self.today or not self.spot_rows.get(self.today.isoformat()):
            atm = int(round(spot_close / 50.0) * 50)
            return (atm - 50, atm - 100, atm - 150, atm + 50, atm + 100, atm + 150)
        synthetic = {"min": [], "close": []}
        for row in self.spot_rows[self.today.isoformat()]:
            m = _row_minute(row)
            if m <= minute:
                synthetic["min"].append(m)
                synthetic["close"].append(float(row["close"]))
        if minute not in synthetic["min"]:
            atm = int(round(spot_close / 50.0) * 50)
            return (atm - 50, atm - 100, atm - 150, atm + 50, atm + 100, atm + 150)
        ce = active_strike_candidates(synthetic, minute, "CE")
        pe = active_strike_candidates(synthetic, minute, "PE")
        return tuple(ce) + tuple(pe)

    # ── Live cache loader ────────────────────────────────────────────

    def _payload_for(self, day: date) -> Optional[Dict[str, Any]]:
        iso = day.isoformat()
        spot_rows = [
            r for r in self.spot_rows.get(iso, [])
            if _row_minute(r) <= self._slice_minute
        ]
        contracts = {}
        for key, entry in self.contract_rows.items():
            rows = [
                r for r in entry["rows"]
                if self._row_key(r) == iso and _row_minute(r) <= self._slice_minute
            ]
            contracts[key] = {
                "rows": rows,
                "tsym": entry["tsym"],
                "token": entry["token"],
            }
        if not spot_rows:
            return None
        return {"spot_rows": spot_rows, "contracts": contracts}

    def _loader(self, cache_root: Any, day: date) -> Optional[Dict[str, Any]]:
        return self._payload_for(day)

    # ── Event evaluation ─────────────────────────────────────────────

    def evaluate(self, minute: int) -> List[dict]:
        """Runs the B17 4-TF extraction over current buffers and returns
        events whose signal minute is newer than the last evaluation.

        Rows are sliced to ``minute`` (causal): only bars that have closed
        by the evaluated minute are served to the engine, matching the
        backtest's per-minute knowledge exactly.
        """
        if self.today is None:
            return []
        iso = self.today.isoformat()
        if minute <= self.last_evaluated_minute:
            return []

        self._slice_minute = minute
        merged: List[Tuple[int, int, dict]] = []
        per_tf_counts: Dict[int, int] = {}
        for tf in self.timeframes:
            try:
                payload = extract_day_events(
                    iso,
                    cache_loader=self._loader,
                    bar_minutes=tf,
                    filter_period=5 * tf,
                    debug=False,
                    **CHAMPION,
                )
            except Exception:
                logger.exception("B17 extraction failed for tf=%sm", tf)
                continue
            if not payload:
                per_tf_counts[tf] = 0
                continue
            selected = _session_events(payload)
            per_tf_counts[tf] = len(selected)
            for signal in selected:
                merged.append((int(signal["minute"]), tf, signal))

        merged.sort(key=lambda item: (item[0], item[1]))
        events: List[dict] = []
        seen: set = set()
        source_tf_by_minute: Dict[int, int] = {}
        for sig_minute, tf, signal in merged:
            key = (sig_minute, signal["side"], signal["symbol"])
            if key in seen:
                continue
            seen.add(key)
            event = dict(signal)
            event["timeframe"] = "combined"
            events.append(event)
            source_tf_by_minute.setdefault(sig_minute, tf)

        self.per_tf_counts = per_tf_counts
        self.source_tf_by_minute = source_tf_by_minute
        new_events = [e for e in events if int(e["minute"]) > self.last_evaluated_minute]
        self.last_evaluated_minute = minute
        self._evaluations += 1
        return new_events

    def exit_levels(self, event: dict) -> Tuple[float, float, bool, str]:
        """Returns (sl_level, tp_level, price_rise, monitor_source).

        monitor_source is 'index' for index-source events (exit checked on
        Nifty spot) or 'option' (exit checked on the option premium).
        """
        sl_level = fib_price(
            float(event["fib_high"]),
            float(event["fib_low"]),
            STOP_LEVEL,
            event.get("orientation", "high_to_low"),
        )
        tp_level = fib_price(
            float(event["fib_high"]),
            float(event["fib_low"]),
            TARGET_LEVEL,
            event.get("orientation", "high_to_low"),
        )
        price_rise = bool(event.get("price_profit_on_rise", event["side"] == "CE"))
        monitor = "index" if event.get("fib_source") == "index" else "option"
        return sl_level, tp_level, price_rise, monitor

    def get_summary(self) -> Dict[str, Any]:
        today = self.today.isoformat() if self.today else None
        counts = {f"{tf}m": self.per_tf_counts.get(tf, 0) for tf in self.timeframes}
        return {
            "timeframe": "combined",
            "today": today,
            "per_tf_counts": counts,
            "total_events": sum(self.per_tf_counts.values()),
            "last_evaluated_minute": self.last_evaluated_minute,
            "evaluations": self._evaluations,
            "tracked_contracts": len(self.contract_rows),
            "spot_rows": len(self.spot_rows.get(today, [])) if today else 0,
        }
