"""Research-only F6 replay using the pre-pivot divergence semantics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grid_optimize_f6_atr as grid
import backtest_monthly_ramp as ramp
from backtest_5y_optimized import load_spot, option_files


class LegacyDivergenceEngine:
    """The former rolling-close-extrema divergence implementation."""

    def __init__(self, max_history=40, min_lookback=3, max_lookback=30):
        self.max_history = max_history
        self.min_lookback = min_lookback
        self.max_lookback = max_lookback
        self.price_history = deque(maxlen=max_history)
        self.s1_history = deque(maxlen=max_history)

    def update(self, close_price: float, s1_val: Optional[float], **_kwargs):
        if s1_val is not None:
            self.price_history.append(close_price)
            self.s1_history.append(s1_val)

    def _find_troughs(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        prices = list(self.price_history)
        s1_vals = list(self.s1_history)
        count = len(prices)
        if count < self.min_lookback + 3:
            return None, None
        recent_window = min(10, count)
        t2_index = min(range(count - recent_window, count), key=prices.__getitem__)
        end_index = max(0, t2_index - self.min_lookback + 1)
        start_index = max(0, t2_index - self.max_lookback)
        if end_index <= start_index:
            return None, None
        prior = prices[start_index:end_index]
        if not prior:
            return None, None
        t1_index = start_index + min(range(len(prior)), key=prior.__getitem__)
        return (
            (prices[t1_index], s1_vals[t1_index]),
            (prices[t2_index], s1_vals[t2_index]),
        )

    def has_bullish_trough_divergence(self) -> bool:
        first, second = self._find_troughs()
        if first is None or second is None:
            return False
        return second[0] < first[0] and second[1] > first[1]

    def bullish_divergence_id(self):
        # The legacy engine had no pair identity, so a continuing condition
        # could re-arm on later bars. Use the history length to preserve that
        # behavior inside the current tracker adapter.
        return (len(self.price_history),) if self.has_bullish_trough_divergence() else None

    def _find_peaks(self) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        prices = list(self.price_history)
        s1_vals = list(self.s1_history)
        count = len(prices)
        if count < self.min_lookback + 3:
            return None, None
        recent_window = min(10, count)
        p2_index = max(range(count - recent_window, count), key=prices.__getitem__)
        end_index = max(0, p2_index - self.min_lookback + 1)
        start_index = max(0, p2_index - self.max_lookback)
        if end_index <= start_index:
            return None, None
        prior = prices[start_index:end_index]
        if not prior:
            return None, None
        p1_index = start_index + max(range(len(prior)), key=prior.__getitem__)
        return (
            (prices[p1_index], s1_vals[p1_index]),
            (prices[p2_index], s1_vals[p2_index]),
        )

    def has_bearish_peak_divergence(self) -> bool:
        first, second = self._find_peaks()
        if first is None or second is None:
            return False
        return second[0] > first[0] and second[1] < first[1]


def init_worker(spot, params):
    grid.GLOBAL_SPOT = spot
    grid.GLOBAL_CACHE = {}
    grid.DivergenceEngine = LegacyDivergenceEngine
    ramp.ACTIVE_PARAMS = dict(params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-divergence", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/legacy_divergence_research.json")
    args = parser.parse_args()
    params = json.loads(Path(args.params_file).read_text(encoding="utf-8"))["params"]
    params["use_divergence"] = not args.no_divergence
    spot = load_spot()
    files = option_files(args.start, args.end)
    days = sorted(set(files) & set(spot))
    if args.smoke:
        days = days[:5]
    tasks = [
        (day, str(files[day]), str(files[days[index - 1]]) if index else "")
        for index, day in enumerate(days)
    ]
    with Pool(max(1, min(8, args.workers)), initializer=init_worker, initargs=(spot, params)) as pool:
        trades = [trade for result in pool.imap(ramp.process_day, tasks) for trade in result]
    result = {
        "runner": "backtest_monthly_ramp with legacy rolling-close divergence",
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "smoke": args.smoke,
        "params": params,
        "stats": grid.summarize(trades),
        "yearly": {
            year: grid.summarize([trade for trade in trades if trade["date"].startswith(year)])
            for year in sorted({trade["date"][:4] for trade in trades})
        },
    }
    output = Path(args.output)
    if args.smoke:
        output = output.with_name(output.stem + "_smoke" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
