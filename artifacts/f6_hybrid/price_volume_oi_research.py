"""Research-only F6 filter using price, option-chain volume, and OI."""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import oi_interval_research as base
from backtest_5y_optimized import load_spot, option_files


OI_RADIUS = 4


def market_sentiment(groups: dict, spot: dict, interval: int, strict: bool) -> dict[int, str]:
    """Samples chain volume/OI and spot price at the requested candle interval."""
    if interval <= 0:
        return {}

    snapshots = {}
    previous = None
    for minute in range(base.SESSION_START - 30, base.DAY_LAST + 1):
        if minute % interval != 0:
            continue
        spot_price = base.latest_spot(spot, minute)
        if spot_price is None:
            continue
        atm = int(round(spot_price / 50.0) * 50)
        strikes = {atm + offset * 50 for offset in range(-OI_RADIUS, OI_RADIUS + 1)}
        total_volume = 0.0
        total_oi = 0.0
        for symbol, group in groups.items():
            match = base.SYM_RE.match(symbol)
            if not match or int(match.group(2)) not in strikes:
                continue
            volume = base.value_at(group, minute, "volume")
            oi = base.value_at(group, minute, "oi")
            if volume is not None:
                total_volume += volume
            if oi is not None:
                total_oi += oi

        current = (spot_price, total_volume, total_oi)
        regime = "NEUTRAL"
        if previous is not None:
            price_up = current[0] > previous[0]
            price_down = current[0] < previous[0]
            volume_up = current[1] > previous[1]
            volume_down = current[1] < previous[1]
            oi_up = current[2] > previous[2]
            oi_down = current[2] < previous[2]
            if price_up and volume_up and oi_up:
                regime = "BULLISH"
            elif price_up and volume_down and oi_down and not strict:
                regime = "BULLISH"
            elif price_down and volume_up and oi_up:
                regime = "BEARISH"
            elif price_down and volume_down and oi_down and not strict:
                regime = "BEARISH"
        snapshots[minute] = regime
        previous = current

    carried = "NEUTRAL"
    output = {}
    for minute in range(base.SESSION_START, base.DAY_LAST + 1):
        if minute in snapshots:
            carried = snapshots[minute]
        output[minute] = carried
    return output


def init_worker(spot):
    base.GLOBAL_SPOT = spot


def process_day(args):
    day, path, previous_path, params, intervals, strict = args
    spot = base.GLOBAL_SPOT.get(day)
    if spot is None:
        return {interval: [] for interval in intervals}
    groups = base.load_groups(path)
    previous = base.load_groups(previous_path) if previous_path else {}
    if not groups:
        return {interval: [] for interval in intervals}
    triggers, state, prefix = base.build_signals(groups, previous, params, spot)
    if not state.get("slices"):
        return {interval: [] for interval in intervals}
    state["oi_groups"] = groups
    return {
        interval: base.simulate(
            day,
            spot,
            triggers,
            state,
            prefix,
            params,
            interval,
            sentiment_override=market_sentiment(groups, spot, interval, strict),
        )
        for interval in intervals
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--intervals", default="0,1,2,3,5,15")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-divergence", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Allow only strong all-three-direction regimes")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/price_volume_oi_research.json")
    args = parser.parse_args()

    params = json.loads(Path(args.params_file).read_text(encoding="utf-8"))["params"]
    params["use_divergence"] = not args.no_divergence
    intervals = [int(value) for value in args.intervals.split(",")]
    spot_all = load_spot()
    files = option_files(args.start, args.end)
    days = sorted(set(files) & set(spot_all))
    if args.smoke:
        days = days[:5]
    tasks = [
        (day, str(files[day]), str(files[days[index - 1]]) if index else "", params, intervals, args.strict)
        for index, day in enumerate(days)
    ]
    aggregate = {interval: [] for interval in intervals}
    with Pool(max(1, min(8, args.workers)), initializer=init_worker, initargs=(spot_all,)) as pool:
        for day_results in pool.imap(process_day, tasks):
            for interval, trades in day_results.items():
                aggregate[interval].extend(trades)
    result = {
        "start": args.start,
        "end": args.end,
        "days": len(days),
        "workers": max(1, min(8, args.workers)),
        "timeframes": sorted(base.EXECUTION_TIMEFRAMES),
        "filter": "price_volume_oi",
        "strict": args.strict,
        "use_divergence": not args.no_divergence,
        "params": params,
        "results": {str(interval): base.summarize(trades) for interval, trades in aggregate.items()},
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
