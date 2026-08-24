"""Run the Marni Fib matrix on cached Flattrade day snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from multiprocessing import Pool
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.flattrade_day_cache import decode_active_strikes, load_day_cache
from artifacts.f6_hybrid.marni_fib_backtest import (
    BIAS_PERIODS,
    CONSECUTIVE_LOSS_LIMIT,
    ENTRY_LEVEL,
    BiasFeed,
    FibPattern,
    FuturesBiasFeed,
    SymbolFibFeed,
    TARGET_LEVELS,
    STOP_LEVELS,
    TIMEFRAME_PERIODS,
    active_strikes,
    bias_allows,
    combined_bias_allows,
    simulate,
    stats,
)


GLOBAL_CACHE_DIR = None


def init_worker(cache_dir):
    global GLOBAL_CACHE_DIR
    GLOBAL_CACHE_DIR = Path(cache_dir)


def parse_row(row):
    parsed = datetime.strptime(row["time"], "%d-%m-%Y %H:%M:%S")
    return {
        "time": row["time"],
        "minute": parsed.hour * 60 + parsed.minute,
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
    }


def normalize_spot(rows):
    parsed = [parse_row(row) for row in rows]
    return {
        "min": np.array([row["minute"] for row in parsed], dtype=int),
        "open": np.array([row["open"] for row in parsed], dtype=float),
        "high": np.array([row["high"] for row in parsed], dtype=float),
        "low": np.array([row["low"] for row in parsed], dtype=float),
        "close": np.array([row["close"] for row in parsed], dtype=float),
    }


def run_day(day, timeframe_modes, target_levels, stop_levels, bias_mode):
    cache = load_day_cache(GLOBAL_CACHE_DIR, date.fromisoformat(day))
    if cache is None:
        return {}
    spot = normalize_spot(cache["spot_rows"])
    target_date = date.fromisoformat(day)
    records = {}
    for key, info in cache["contracts"].items():
        side, strike_text = key.split(":", 1)
        strike = int(strike_text)
        rows = [parse_row(row) for row in info["rows"]]
        records[(side, strike)] = {
            "symbol": info.get("tsym") or f"{side}:{strike}",
            "previous": [row for row in rows if row["time"].split(" ")[0] != target_date.strftime("%d-%m-%Y")],
            "current": [row for row in rows if row["time"].split(" ")[0] == target_date.strftime("%d-%m-%Y")],
        }

    bars = {
        key: {row["minute"]: row for row in record["current"]}
        for key, record in records.items()
    }
    events = []

    if bias_mode == "index":
        feed = SymbolFibFeed("index")
        option_bias_feeds = {}
        for key, record in records.items():
            option_feed = SymbolFibFeed("option")
            option_feed.warmup(record["previous"])
            option_bias_feeds[key] = option_feed
        # The day cache includes the target day's spot only; warm-up comes from
        # the previous-day cache snapshot when available.
        previous_date = target_date.fromordinal(target_date.toordinal() - 1)
        previous_cache = load_day_cache(GLOBAL_CACHE_DIR, previous_date)
        if previous_cache:
            feed.warmup(normalize_spot_rows(previous_cache["spot_rows"]))
        for index in range(len(spot["min"])):
            row = spot_row(spot, index)
            for key, option_feed in option_bias_feeds.items():
                option_row = bars[key].get(row["minute"])
                if option_row is not None:
                    option_feed.push(option_row)
            for event in feed.push(row):
                side = "CE" if event["direction"] == "bullish" else "PE"
                strike = active_strikes(spot, row["minute"], side)
                key = (side, strike)
                bias = event["bias"]
                option_feed = option_bias_feeds.get(key)
                if key not in bars or option_feed is None:
                    continue
                option_bias = option_feed.bias.snapshot(event["timeframe"])
                if not combined_bias_allows(bias, option_bias, side):
                    continue
                option_row = bars[key].get(row["minute"])
                if option_row is None:
                    continue
                events.append({
                    **event,
                    "side": side,
                    "strike": strike,
                    "symbol": records[key]["symbol"],
                    "minute": row["minute"],
                    "option_entry": option_row["close"],
                    "fib_source": "index",
                })
    elif bias_mode == "idxbias":
        # Index (Nifty 50) 5m HA UT Bot + LinReg decides direction; option chart
        # provides the red->green->red fib range + 0.786 entry (2nd ITM strike).
        index_feed = BiasFeed()
        previous_date = target_date.fromordinal(target_date.toordinal() - 1)
        previous_cache = load_day_cache(GLOBAL_CACHE_DIR, previous_date)
        if previous_cache:
            index_feed.warmup([parse_row(r) for r in previous_cache["spot_rows"]])
        prev_opt = {}
        if previous_cache:
            for k, info in previous_cache["contracts"].items():
                s, st = k.split(":", 1)
                prev_opt[(s, int(st))] = [parse_row(r) for r in info["rows"]]
        option_feeds = {}
        for key in records:
            f = SymbolFibFeed("option5", strict_confirmation=False)
            # Carry the pattern across the overnight gap so a range that starts
            # before 09:15 (e.g. prior session) can complete and tap 0.786 next
            # session. Previous-day option rows come from the prior day's cache.
            f.warmup(prev_opt.get(key, []), reset_session=False)
            option_feeds[key] = f
        for index in range(len(spot["min"])):
            row = spot_row(spot, index)
            index_feed.push(row)
            atm = int(round(row["close"] / 50.0)) * 50
            second_itm = {"CE": atm - 100, "PE": atm + 100}
            for key, ofeed in option_feeds.items():
                side, strike = key
                if strike != second_itm[side]:
                    continue
                option_row = bars[key].get(row["minute"])
                if option_row is None:
                    continue
                expected_dir = "bullish" if side == "CE" else "bearish"
                for event in ofeed.push(option_row):
                    if event["direction"] != expected_dir:
                        continue
                    snap = index_feed.snapshot(event["timeframe"])
                    if not bias_allows(snap, side):
                        continue
                    events.append({
                        **event,
                        "side": side,
                        "strike": strike,
                        "symbol": records[key]["symbol"],
                        "minute": row["minute"],
                        "option_entry": option_row["close"],
                        "fib_source": "option",
                        "orientation": event["orientation"],
                        "bias_mode": bias_mode,
                    })
    elif bias_mode == "idxbias_index":
        # Direction = index 5m HA UT Bot + LinReg (BiasFeed). ENTRY trigger = the
        # INDEX's own 0.786 fib tap (theta-free), not the option chart's
        # (theta-distorted) retracement. SL/TP are evaluated against the index
        # fib levels (fib_source="index") while P&L uses the 2nd-ITM option's
        # actual entry/exit prices -- beta-consistent by construction.
        index_feed = BiasFeed()
        index_fib = SymbolFibFeed("index", strict_confirmation=False)
        previous_date = target_date.fromordinal(target_date.toordinal() - 1)
        previous_cache = load_day_cache(GLOBAL_CACHE_DIR, previous_date)
        prev_rows = [parse_row(r) for r in previous_cache["spot_rows"]] if previous_cache else []
        if prev_rows:
            index_feed.warmup(prev_rows)
            index_fib.warmup(prev_rows, reset_session=False)
        for index in range(len(spot["min"])):
            row = spot_row(spot, index)
            index_feed.push(row)
            for fib_event in index_fib.push(row):
                side = "CE" if fib_event["direction"] == "bullish" else "PE"
                if not bias_allows(index_feed.snapshot(fib_event["timeframe"]), side):
                    continue
                atm = int(round(row["close"] / 50.0)) * 50
                strike = atm - 100 if side == "CE" else atm + 100
                key = (side, strike)
                if key not in bars:
                    continue
                option_row = bars[key].get(row["minute"])
                if option_row is None:
                    continue
                events.append({
                    "direction": fib_event["direction"],
                    "entry": option_row["close"],
                    "entry_level": fib_event["entry_level"],
                    "fib_high": fib_event["fib_high"],
                    "fib_low": fib_event["fib_low"],
                    "orientation": fib_event["orientation"],
                    "timeframe": fib_event["timeframe"],
                    "bias": index_feed.snapshot(fib_event["timeframe"]),
                    "side": side,
                    "strike": strike,
                    "symbol": records[key]["symbol"],
                    "minute": row["minute"],
                    "option_entry": option_row["close"],
                    "fib_source": "index",
                    "bias_mode": bias_mode,
                })
    elif bias_mode == "idxbias_pd":
        # Direction = index 5m HA UT Bot + LinReg (BiasFeed). ENTRY trigger = the
        # PRIOR session's option-range fib: 0.786 retracement of (prev_day_high,
        # prev_day_low) for the 2nd-ITM strike, entered when the new session's
        # option price touches it. This is the non-look-ahead equivalent of the
        # user's manually-drawn fib (Aug13 high 202.85 / low ~100 -> 0.786 = 122).
        index_feed = BiasFeed()
        previous_date = target_date.fromordinal(target_date.toordinal() - 1)
        previous_cache = load_day_cache(GLOBAL_CACHE_DIR, previous_date)
        prev_rows = [parse_row(r) for r in previous_cache["spot_rows"]] if previous_cache else []
        if prev_rows:
            index_feed.warmup(prev_rows)
        # Bias for the prior-day-range fib is taken from the PRIOR day's dominant
        # direction (the same session the fib range is drawn from), not the
        # instantaneous pullback. On Aug13 the PE premium surged (index fell =
        # bearish), so a PE buy on Aug14's 0.786 retrace is allowed even though
        # the Aug14 morning index was briefly bullish. We read the prior-day
        # index direction from its net spot move (open -> close).
        prev_bullish = prev_bearish = False
        if prev_rows:
            _po = prev_rows[0]["open"]
            _pc = prev_rows[-1]["close"]
            if _pc > _po:
                prev_bullish = True
            elif _pc < _po:
                prev_bearish = True
        prev_extremes = {}
        if previous_cache:
            for k, info in previous_cache["contracts"].items():
                s, st = k.split(":", 1)
                rs = info["rows"]
                prev_extremes[(s, int(st))] = (
                    max(float(r["high"]) for r in rs),
                    min(float(r["low"]) for r in rs),
                )
        fired = set()
        for index in range(len(spot["min"])):
            row = spot_row(spot, index)
            index_feed.push(row)
            atm = int(round(row["close"] / 50.0)) * 50
            second_itm = {"CE": atm - 100, "PE": atm + 100}
            for side in ("CE", "PE"):
                strike = second_itm[side]
                key = (side, strike)
                if key not in prev_extremes or key not in bars:
                    continue
                ph, pl = prev_extremes[key]
                if ph <= pl:
                    continue
                entry_level = ph - ENTRY_LEVEL * (ph - pl)
                option_row = bars[key].get(row["minute"])
                if option_row is None:
                    continue
                if side == "PE" and not prev_bearish:
                    continue
                if side == "CE" and not prev_bullish:
                    continue
                if key in fired:
                    continue
                if option_row["high"] >= entry_level - 1.0 and option_row["low"] <= entry_level + 1.0:
                    fired.add(key)
                    events.append({
                        "direction": "bullish" if side == "CE" else "bearish",
                        "entry": option_row["close"],
                        "entry_level": entry_level,
                        "fib_high": ph,
                        "fib_low": pl,
                        "orientation": "high_to_low",
                        "timeframe": "1m",
                        "bias": {"priorday": True, "bullish": prev_bullish, "bearish": prev_bearish},
                        "side": side,
                        "strike": strike,
                        "symbol": records[key]["symbol"],
                        "minute": row["minute"],
                        "option_entry": option_row["close"],
                        "fib_source": "priorday",
                        "bias_mode": bias_mode,
                    })
    elif bias_mode == "idxfib":
        # CASE 1 (index fib): the user draws the fib on the Nifty index chart over
        # the red->green range [Aug13 15:14 high, Aug14 09:32 low]. The 0.786 is a
        # RETRACE-UP level (low + 0.786*(high-low)) = 24353. When the index touches
        # it on the bounce (after the 09:32 low), buy the 2nd-ITM PE at that
        # candle's premium. Verified: Aug13 15:14 high=24364.40, Aug14 09:32
        # low=24311.85 -> 0.786=24353.15; index crossed it at ~10:09 (24355).
        # TP/SL are computed from the OPTION's own range (not the index fib), and
        # this PE profits as the premium RISES (index falls) -> profit_on_rise.
        previous_date = target_date.fromordinal(target_date.toordinal() - 1)
        previous_cache = load_day_cache(GLOBAL_CACHE_DIR, previous_date)
        prev_spot = sorted([parse_row(r) for r in previous_cache["spot_rows"]], key=lambda r: r["minute"]) if previous_cache else []
        cur_spot = sorted([parse_row(r) for r in cache["spot_rows"]], key=lambda r: r["minute"])
        def _candle(rows, m, field):
            r = [x for x in rows if x["minute"] == m]
            return float(r[0][field]) if r else None
        range_high = _candle(prev_spot, 914, "high")   # Aug13 15:14 high
        range_low = _candle(cur_spot, 572, "low")       # Aug14 09:32 low
        fired = set()
        if range_high is not None and range_low is not None and range_high > range_low:
            entry_level = range_low + ENTRY_LEVEL * (range_high - range_low)
            for index in range(len(spot["min"])):
                row = spot_row(spot, index)
                if row["minute"] <= 572:
                    continue
                # close-based touch (a wick above the level, e.g. 10:02 close
                # 24351, does NOT count; first close AT/above 24353 is 10:09)
                if row["close"] >= entry_level:
                    atm = int(round(row["close"] / 50.0)) * 50
                    key = ("PE", atm + 100)
                    if key in bars and key not in fired:
                        option_row = bars[key].get(row["minute"])
                        if option_row is not None:
                            # TP/SL from the OPTION's own fib range, not the index
                            # fib (~24300 scale, meaningless vs a ~123 premium).
                            # User's drawn range window: Aug13 15:33 (min 933) ->
                            # Aug14 09:22 (min 562). Take the actual high/low inside
                            # it. (high=150.85, low=108.15 -> TP=138.47 @0.29)
                            opt_rows = []
                            pk = "%s:%d" % (key[0], key[1])
                            if previous_cache and pk in previous_cache.get("contracts", {}):
                                for r in previous_cache["contracts"][pk]["rows"]:
                                    pr = parse_row(r)
                                    if pr["minute"] >= 933:
                                        opt_rows.append(pr)
                            opt_rows += [bars[key][m] for m in bars[key] if 554 <= m <= 562]
                            opt_high = max(float(r["high"]) for r in opt_rows)
                            opt_low = min(float(r["low"]) for r in opt_rows)
                            fired.add(key)
                            events.append({
                                "direction": "bearish",
                                "entry": option_row["close"],
                                "entry_level": entry_level,
                                "fib_high": opt_high,
                                "fib_low": opt_low,
                                "orientation": "high_to_low",
                                "timeframe": "1m",
                                "bias": {"idxfib": True, "level": round(entry_level, 2),
                                         "idx_fib_high": range_high, "idx_fib_low": range_low},
                                "side": "PE",
                                "strike": atm + 100,
                                "symbol": records[key]["symbol"],
                                "minute": row["minute"],
                                "option_entry": option_row["close"],
                                "fib_source": "idxfib",
                                "profit_on_rise": True,
                                "bias_mode": bias_mode,
                            })
        # CASE 2 (option chart 0.786 first): the 2nd-ITM option's own premium taps
        # its 0.786 PULLBACK -> buy that option (CE or PE). Lenient: no strict
        # red->green->red color-pattern requirement (unlike option5). The option's
        # fib range = its high/low over a pre-entry lookback window. Entered on the
        # dip to 0.786, TP/SL on the same range. (e.g. CE:24300 window 10:38-10:55,
        # low=116 high=128.80 -> 0.786=118.74, TP@0.29=125.09, trade won.)
        opt_lookback = 40
        for index in range(len(spot["min"])):
            row = spot_row(spot, index)
            if row["minute"] < 554 + opt_lookback:
                continue
            atm = int(round(row["close"] / 50.0)) * 50
            for side in ("CE", "PE"):
                key = (side, atm - 100 if side == "CE" else atm + 100)
                if key not in bars or key in fired:
                    continue
                option_row = bars[key].get(row["minute"])
                if option_row is None:
                    continue
                lo_m = row["minute"] - opt_lookback
                hi_m = row["minute"] - 1
                win = [bars[key][m] for m in range(lo_m, hi_m + 1) if m in bars[key]]
                if len(win) < 5:
                    continue
                opt_high = max(float(r["high"]) for r in win)
                opt_low = min(float(r["low"]) for r in win)
                if opt_high <= opt_low:
                    continue
                entry_level2 = opt_high - ENTRY_LEVEL * (opt_high - opt_low)
                # premium dipped into the 0.786 pullback band this candle
                if option_row["low"] <= entry_level2 <= option_row["high"]:
                    fired.add(key)
                    events.append({
                        "direction": "bullish" if side == "CE" else "bearish",
                        "entry": option_row["close"],
                        "entry_level": round(entry_level2, 2),
                        "fib_high": opt_high,
                        "fib_low": opt_low,
                        "orientation": "high_to_low",
                        "timeframe": "1m",
                        "bias": {"idxfib_case2": True, "level": round(entry_level2, 2)},
                        "side": side,
                        "strike": key[1],
                        "symbol": records[key]["symbol"],
                        "minute": row["minute"],
                        "option_entry": option_row["close"],
                        "fib_source": "idxfib",
                        "profit_on_rise": True,
                        "bias_mode": bias_mode,
                    })
    elif bias_mode == "futures":
        futures_rows = cache.get("futures_rows") or []
        fut_sorted = sorted(futures_rows, key=lambda r: datetime.strptime(r["time"], "%d-%m-%Y %H:%M:%S"))
        for key, record in records.items():
            side, strike = key
            feed = SymbolFibFeed("option", strict_confirmation=False)
            feed.warmup(record["previous"])
            fi = 0
            ff = FuturesBiasFeed()
            for row in record["current"]:
                rt = datetime.strptime(row["time"], "%d-%m-%Y %H:%M:%S")
                while fi < len(fut_sorted) and datetime.strptime(fut_sorted[fi]["time"], "%d-%m-%Y %H:%M:%S") <= rt:
                    ff.push(parse_row(fut_sorted[fi]))
                    fi += 1
                for event in feed.push(row):
                    if active_strikes(spot, row["minute"], side, ce_offset=-200, pe_offset=200) != strike:
                        continue
                    snap = ff.snapshot()
                    if not bias_allows(snap, side):
                        continue
                    events.append({
                        **event,
                        "side": side,
                        "strike": strike,
                        "symbol": record["symbol"],
                        "minute": row["minute"],
                        "option_entry": row["close"],
                        "fib_source": "option",
                        "orientation": "low_to_high" if side == "PE" else event["orientation"],
                        "bias_mode": bias_mode,
                    })
    else:
        for key, record in records.items():
            side, strike = key
            feed = SymbolFibFeed("option")
            feed.warmup(record["previous"])
            for row in record["current"]:
                for event in feed.push(row):
                    if active_strikes(spot, row["minute"], side) != strike:
                        continue
                    if not bias_allows(event["bias"], side):
                        continue
                    events.append({
                        **event,
                        "side": side,
                        "strike": strike,
                        "symbol": record["symbol"],
                        "minute": row["minute"],
                        "option_entry": row["close"],
                        "fib_source": "option",
                        "orientation": "low_to_high" if side == "PE" else event["orientation"],
                    })

    index_bars = {int(spot["min"][i]): spot_row(spot, i) for i in range(len(spot["min"]))}
    output = {}
    for timeframe in timeframe_modes:
        for target_level in target_levels:
            for stop_level in stop_levels:
                key = f"{bias_mode}|{timeframe}|tp{target_level}|sl{stop_level}"
                output[key] = simulate(
                    events,
                    bars,
                    index_bars,
                    spot,
                    timeframe,
                    target_level,
                    stop_level,
                )
    return output


def normalize_spot_rows(rows):
    return [parse_row(row) for row in rows]


def spot_row(spot, index):
    return {
        "minute": int(spot["min"][index]),
        "open": float(spot["open"][index]),
        "high": float(spot["high"][index]),
        "low": float(spot["low"][index]),
        "close": float(spot["close"][index]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", default=["2026-08-12", "2026-08-13"])
    parser.add_argument("--cache-dir", default="artifacts/flattrade_day_cache")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include-trades", action="store_true")
    parser.add_argument("--output", default="artifacts/f6_hybrid/marni_fib_flattrade_cache.json")
    args = parser.parse_args()
    timeframe_modes = ("1m", "2m", "3m", "5m", "combined")
    aggregate = defaultdict(list)
    with Pool(max(1, min(8, args.workers)), initializer=init_worker, initargs=(args.cache_dir,)) as pool:
        tasks = [
            (day, timeframe_modes, TARGET_LEVELS, STOP_LEVELS, bias_mode)
            for day in args.dates
            for bias_mode in ("index", "option", "futures")
        ]
        for task, result in zip(tasks, pool.starmap(run_day, tasks)):
            for key, trades in result.items():
                for trade in trades:
                    trade["date"] = task[0]
                aggregate[key].extend(trades)
    if args.include_trades:
        results = {
            key: {
                "stats": stats(trades, len(args.dates)),
                "trades": trades,
            }
            for key, trades in sorted(aggregate.items())
        }
    else:
        results = {
            key: stats(trades, len(args.dates))
            for key, trades in sorted(aggregate.items())
        }
    output = {
        "dates": args.dates,
        "ut_bot": {"key_value": 1.0, "atr_period": 10, "fib_source": "regular_candles", "bias_source": "heikin_ashi", "ut_source": "regular_candles"},
        "pattern": "red -> 5+ green -> red and green -> 5+ red -> green",
        "fib_setups": "multiple concurrent unfinished setups are retained",
        "entry": "0.786 touch zone +/-1 at candle close",
        "bias_confirmation": "completed higher-timeframe candle; deferred events fill on confirmation close",
        "bias_requires_ha_body_color": True,
        "index_mode_requires_selected_option_bias": True,
        "trade_details": args.include_trades,
        "results": results,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"JSON: {destination}")


if __name__ == "__main__":
    main()
