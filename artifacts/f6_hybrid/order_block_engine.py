"""Order-block signal engine — "Fibb Block Strategy" entry condition.

Entry logic is taken from the "Fibb Block Strategy" video (Sustainable
Trading, https://www.youtube.com/watch?v=9QozaXooCYc) as mapped by the user:

* No time-window rule (the video's 13:00-20:00 GMT+2 window is outside Nifty
  hours); instead use the NEAREST completed order block when a setup range
  fires.
* A block is a consolidation (squeeze): on the bar stream, a trailing window
  of ``block_window`` bars whose range stays within ``squeeze_ratio`` of the
  range of the preceding ``lookback`` bars (floor 1 point), and whose own
  range is at least ``min_block_span`` points. Contiguous squeeze bars merge
  into one block {start_minute, end_minute, low, high}.
* A block is completed when price breaks out of it (close beyond the high for
  an up breakout, beyond the low for a down breakout) within
  ``breakout_window`` bars after the squeeze run ends. Blocks without a
  breakout are never triggered.
* Two entry flavors, run as SEPARATE variants (user request):
    - "flip": 80% continuation rule. Price breaks the block, RETESTS it
      (returns inside at any bar after the breakout) and flips back through
      the block edge at the trigger bar. CE: close(m) > block_high after a
      retest below/at block_high; PE symmetric.
    - "turn": price comes STRAIGHT DOWN into the block (a recent bar closed
      beyond the edge — the penetration bar opened at/above the edge — and
      the trigger bar closes inside the block) and S1 turns up inside the
      block. CE: last beyond-edge close is within ``straight_lookback`` bars
      before the trigger and the bar that entered the block opened at/above
      block_high; PE symmetric.
* The setup itself is the standard Smart Fib setup (UT swing pattern +
  S1 turn + 5x-TF index bias filter); the order-block condition REPLACES the
  fib-zone entry condition. Exit geometry is unchanged: targets/stops are
  fib levels of the setup swing (video exits: target 0 / -0.055, stops
  1.079 / 1.155 / 1.25 are evaluated by the GPU engine, not this module).
* One event per minute (first matching setup consumes the minute, as the
  GPU dataset keeps a single slot per minute). A block is consumed per side
  (one entry per block per side), and only the most recent (nearest)
  matching block is used at a trigger bar.

The returned payload mirrors ``extract_day_events`` so the existing Optimus
GPU dataset machinery consumes it unchanged.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid import marni_fib_core_combo_cache as smart_core
from artifacts.f6_hybrid.marni_fib_core_combo_cache import (
    OPTION_DELTA,
    _normalize_s1_periods,
    _resample_rows,
    active_strike_candidates,
    build_index_filter,
    build_s1_snapshots,
    clean_terminal_quotes,
    index_filter_allows,
    normalize_spot,
    parse_row,
    spot_row,
)

# ---------------------------------------------------------------- blocks ---

BLOCK_WINDOW = 12          # bars in the squeeze window
SQUEEZE_RATIO = 0.5        # window range <= ratio x preceding lookback range
BLOCK_LOOKBACK = 40        # bars used as the "swing" baseline
MIN_BLOCK_SPAN = 3.0       # points; ignore micro-consolidations
BREAKOUT_WINDOW = 12       # bars after the squeeze run to confirm a breakout
MAX_BLOCK_AGE = 60         # bars; breakout must be this recent at the trigger
STRAIGHT_LOOKBACK = 5      # bars; "came straight down/up" recency for flavor turn
SQUEEZE_FLOOR = 1.0        # points; absolute floor for the squeeze threshold


def detect_completed_blocks(
    minutes: Sequence[int],
    rows_by_minute: Mapping[int, Mapping[str, Any]],
    *,
    window: int = BLOCK_WINDOW,
    squeeze_ratio: float = SQUEEZE_RATIO,
    lookback: int = BLOCK_LOOKBACK,
    min_block_span: float = MIN_BLOCK_SPAN,
    breakout_window: int = BREAKOUT_WINDOW,
) -> list[dict[str, Any]]:
    """Detect completed consolidation blocks on one bar stream (causal).

    Only fully closed bars are used for the squeeze decision (the window ends
    at the previous bar). Each returned block has ``start_minute``,
    ``end_minute``, ``low``, ``high`` and a ``breakout_minute``/``breakout_dir``
    (``"up"`` or ``"down"``) once price has left the block.
    """
    count = len(minutes)
    if count < window + lookback + 1:
        return []

    def row(index: int) -> Mapping[str, Any]:
        return rows_by_minute[minutes[index]]

    squeezed: list[bool] = [False] * count
    for index in range(window - 1, count):
        win_high = max(row(offset)["high"] for offset in range(index - window + 1, index + 1))
        win_low = min(row(offset)["low"] for offset in range(index - window + 1, index + 1))
        win_range = win_high - win_low
        if index < window + lookback - 1:
            continue
        prior_high = max(
            row(offset)["high"]
            for offset in range(index - window - lookback + 1, index - window + 1)
        )
        prior_low = min(
            row(offset)["low"]
            for offset in range(index - window - lookback + 1, index - window + 1)
        )
        prior_range = max(1e-9, prior_high - prior_low)
        squeezed[index] = (
            win_range >= min_block_span
            and win_range <= max(squeeze_ratio * prior_range, SQUEEZE_FLOOR)
        )

    blocks: list[dict[str, Any]] = []
    index = 0
    while index < count:
        if not squeezed[index]:
            index += 1
            continue
        start = index
        while index + 1 < count and squeezed[index + 1]:
            index += 1
        block_minutes = [minutes[offset] for offset in range(start, index + 1)]
        block = {
            "start_minute": block_minutes[0],
            "end_minute": block_minutes[-1],
            "low": min(row(offset)["low"] for offset in range(start, index + 1)),
            "high": max(row(offset)["high"] for offset in range(start, index + 1)),
            "breakout_minute": None,
            "breakout_dir": None,
        }
        for offset in range(index + 1, min(count, index + 1 + breakout_window)):
            close = row(offset)["close"]
            if close > block["high"]:
                block["breakout_minute"] = minutes[offset]
                block["breakout_dir"] = "up"
                break
            if close < block["low"]:
                block["breakout_minute"] = minutes[offset]
                block["breakout_dir"] = "down"
                break
        if block["breakout_minute"] is not None:
            blocks.append(block)
        index += 1
    return blocks


def match_block(
    blocks: Sequence[Mapping[str, Any]],
    minute: int,
    side: str,
    flavor: str,
    minutes: Sequence[int],
    rows_by_minute: Mapping[int, Mapping[str, Any]],
    minute_index: Mapping[int, int],
    consumed: set[tuple[str, int]],
    max_block_age: int = MAX_BLOCK_AGE,
    straight_lookback: int = STRAIGHT_LOOKBACK,
) -> dict[str, Any] | None:
    """Return the nearest (most recent) matching block for the trigger bar.

    Blocks are scanned newest-first by breakout minute; the first block that
    satisfies the flavor's conditions at ``minute`` (and is not consumed) is
    returned. Conditions use only bars strictly before ``minute`` except the
    current bar's open/high/close (entry is at the close of ``minute``, the
    same convention as the Smart Fib engine).
    """
    direction = "up" if side == "CE" else "down"
    ordered = sorted(
        blocks,
        key=lambda block: int(block["breakout_minute"]),
        reverse=True,
    )
    for block in ordered:
        breakout = int(block["breakout_minute"])
        if breakout >= minute:
            continue
        if minute - breakout > max_block_age:
            continue
        key = (side, int(block["start_minute"]))
        if key in consumed:
            continue
        if block["breakout_dir"] != direction:
            continue
        high = float(block["high"])
        low = float(block["low"])
        row = rows_by_minute[minute]
        close = float(row["close"])
        if flavor == "flip":
            if direction == "up":
                retested = any(
                    float(rows_by_minute[minutes[offset]]["close"]) <= high
                    for offset in range(
                        minute_index[breakout] + 1,
                        minute_index[minute],
                    )
                    if minutes[offset] in rows_by_minute
                )
                if retested and close > high:
                    return dict(block)
            else:
                retested = any(
                    float(rows_by_minute[minutes[offset]]["close"]) >= low
                    for offset in range(
                        minute_index[breakout] + 1,
                        minute_index[minute],
                    )
                    if minutes[offset] in rows_by_minute
                )
                if retested and close < low:
                    return dict(block)
        elif flavor == "turn":
            trigger_index = minute_index[minute]
            last_beyond = None
            for offset in range(minute_index[breakout], trigger_index):
                candle_close = float(rows_by_minute[minutes[offset]]["close"])
                if direction == "up" and candle_close > high:
                    last_beyond = offset
                elif direction == "down" and candle_close < low:
                    last_beyond = offset
            if last_beyond is None:
                continue
            if trigger_index - last_beyond > straight_lookback:
                continue
            penetration = last_beyond + 1
            if penetration > trigger_index:
                continue
            penetration_row = rows_by_minute[minutes[penetration]]
            if direction == "up":
                penetrated = float(penetration_row["open"]) >= high
            else:
                penetrated = float(penetration_row["open"]) <= low
            if not penetrated:
                continue
            if not (low <= close <= high):
                continue
            return dict(block)
        else:
            raise ValueError(f"unsupported order-block flavor: {flavor}")
    return None


# ------------------------------------------------------------ extraction ---

def extract_order_block_events(
    day,
    *,
    cache_loader=None,
    min_span=smart_core.MIN_SPAN,
    touch_buffer=0.0,
    setup_max_age=smart_core.SETUP_MAX_AGE,
    s1_k_period=smart_core.S1_K_PERIOD,
    s1_d_period=smart_core.S1_D_PERIOD,
    bar_minutes=1,
    filter_period=smart_core.FILTER_PERIOD,
    flavor="flip",
    block_window=BLOCK_WINDOW,
    squeeze_ratio=SQUEEZE_RATIO,
    lookback=BLOCK_LOOKBACK,
    min_block_span=MIN_BLOCK_SPAN,
    breakout_window=BREAKOUT_WINDOW,
    max_block_age=MAX_BLOCK_AGE,
    straight_lookback=STRAIGHT_LOOKBACK,
    debug=False,
):
    """Build the causal order-block event stream for one day.

    Same payload contract as ``marni_fib_core_combo_cache.extract_day_events``
    (``signals``/``events``/``bars``/``index_bars``/``spot``/``records``/
    ``filtered``) so the Optimus GPU dataset builder consumes it unchanged.
    Extra keys: ``ob_blocks`` (all completed blocks of the day, for
    attribution) and per-signal ``trigger`` = ``"ob_flip"``/``"ob_turn"``
    plus ``ob_*`` attribution fields.
    """
    s1_k_period, s1_d_period = _normalize_s1_periods(s1_k_period, s1_d_period)
    load_cache = smart_core.load_day_cache if cache_loader is None else cache_loader
    cache = load_cache(smart_core.GLOBAL_CACHE_DIR, date.fromisoformat(day))
    if cache is None:
        return {}
    target_date = date.fromisoformat(day)
    target_text = target_date.strftime("%d-%m-%Y")
    current_spot_rows = clean_terminal_quotes(
        [parse_row(row) for row in cache["spot_rows"]]
    )
    spot = normalize_spot(current_spot_rows)

    records = {}
    for key, info in cache["contracts"].items():
        side, strike_text = key.split(":", 1)
        strike = int(strike_text)
        rows = [parse_row(row) for row in info["rows"]]
        records[(side, strike)] = {
            "symbol": info.get("tsym") or f"{side}:{strike}",
            "previous": sorted(
                [r for r in rows if r["time"].split(" ")[0] != target_text],
                key=lambda row: (row["time"].split(" ")[0], row["minute"]),
            ),
            "current": sorted(
                [r for r in rows if r["time"].split(" ")[0] == target_text],
                key=lambda row: row["minute"],
            ),
        }

    bars = {
        key: {row["minute"]: row for row in rec["current"]}
        for key, rec in records.items()
    }

    events = []
    prev_date = target_date - timedelta(days=1)
    prev_cache = load_cache(smart_core.GLOBAL_CACHE_DIR, prev_date)
    previous_spot_rows = (
        clean_terminal_quotes([parse_row(row) for row in prev_cache["spot_rows"]])
        if prev_cache
        else []
    )
    index_filter = build_index_filter(
        previous_spot_rows, current_spot_rows, period=filter_period
    )
    tf_current_rows = _resample_rows(current_spot_rows, bar_minutes)
    tf_previous_rows = _resample_rows(previous_spot_rows, bar_minutes)
    tf_rows_by_minute = {row["minute"]: row for row in tf_current_rows}
    tf_close_minutes = set(tf_rows_by_minute)

    minutes = sorted(tf_rows_by_minute)
    minute_index = {minute: index for index, minute in enumerate(minutes)}
    blocks = detect_completed_blocks(
        minutes,
        tf_rows_by_minute,
        window=block_window,
        squeeze_ratio=squeeze_ratio,
        lookback=lookback,
        min_block_span=min_block_span,
        breakout_window=breakout_window,
    )

    filtered = []
    signals = []
    index_retraced = False
    retraced_at = 9999
    consumed_index_setups = set()
    consumed_blocks: set[tuple[str, int]] = set()

    index_feed = smart_core.UTSwingFeed(
        [
            ("bullish", "red", "green", "red", "high_to_low"),
            ("bearish", "green", "red", "green", "low_to_high"),
        ],
        emit_touches=False,
        replace_setups=False,
        min_span=min_span,
        max_setup_age=setup_max_age,
        touch_buffer=touch_buffer,
    )
    for row in tf_previous_rows:
        index_feed.push(row)
    index_feed.clear_setups()

    index_s1 = build_s1_snapshots(
        tf_previous_rows,
        tf_current_rows,
        k_period=s1_k_period,
        d_period=s1_d_period,
    )

    for row in current_spot_rows:
        m = row["minute"]
        if bar_minutes > 1 and m not in tf_close_minutes:
            continue
        index_feed.push(tf_rows_by_minute[m])
        snapshot = index_filter.get(m)

        s1_snapshot = index_s1.get(m)
        if s1_snapshot is None or s1_snapshot["turn"] is None:
            continue

        latest_index_setups = {}
        for candidate in index_feed.setups:
            current = latest_index_setups.get(candidate["pattern"])
            if current is None or candidate["completion_minute"] > current["completion_minute"]:
                latest_index_setups[candidate["pattern"]] = candidate

        for setup in sorted(
            latest_index_setups.values(),
            key=lambda item: item["completion_minute"],
        ):
            setup_id = (setup["pattern"], setup["completion_minute"])
            if setup_id in consumed_index_setups:
                continue
            if setup["completion_minute"] >= m:
                continue
            if setup["fib_high"] - setup["fib_low"] < min_span:
                continue

            side = "CE" if setup["pattern"] == "bullish" else "PE"
            expected_turn = "up" if side == "CE" else "down"
            if s1_snapshot["turn"] != expected_turn:
                continue

            block = match_block(
                blocks,
                m,
                side,
                flavor,
                minutes,
                tf_rows_by_minute,
                minute_index,
                consumed_blocks,
                max_block_age=max_block_age,
                straight_lookback=straight_lookback,
            )
            if block is None:
                continue

            if not index_filter_allows(snapshot, side):
                filtered.append({
                    "minute": m,
                    "side": side,
                    "strike": None,
                    "trigger": f"ob_{flavor}",
                    "zone_start": 0.0,
                    "zone_end": 1.0,
                    "snapshot": snapshot,
                })
                continue

            consumed_index_setups.add(setup_id)
            consumed_blocks.add((side, int(block["start_minute"])))

            candidates = [
                ((side, strike), strike)
                for strike in active_strike_candidates(spot, m, side)
                if (side, strike) in bars and m in bars[(side, strike)]
            ]
            if not candidates:
                continue

            index_retraced = True
            retraced_at = m
            for key, strike in candidates:
                signals.append({
                    "side": side,
                    "strike": strike,
                    "symbol": records[key]["symbol"],
                    "minute": m,
                    "signal_minute": m,
                    "option_entry": bars[key][m]["close"],
                    "fib_source": "index",
                    "trigger": f"ob_{flavor}",
                    "fib_high": setup["fib_high"],
                    "fib_low": setup["fib_low"],
                    "orientation": setup["orientation"],
                    "profit_on_rise": True,
                    "price_profit_on_rise": side == "CE",
                    "dynamic_target": True,
                    "option_delta": OPTION_DELTA,
                    "s1_value": s1_snapshot["value"],
                    "s1_turn": s1_snapshot["turn"],
                    "zone_low": block["low"],
                    "zone_high": block["high"],
                    "zone_start": 0.0,
                    "zone_end": 1.0,
                    "ob_flavor": flavor,
                    "ob_block_start": block["start_minute"],
                    "ob_block_end": block["end_minute"],
                    "ob_breakout_minute": block["breakout_minute"],
                    "ob_breakout_dir": block["breakout_dir"],
                })
            break

    index_bars = {int(spot["min"][i]): spot_row(spot, i) for i in range(len(spot["min"]))}

    if debug:
        print(
            f"[debug {day}] signals={len(signals)} blocks={len(blocks)} "
            f"flavor={flavor} index_retraced={index_retraced} retraced_at={retraced_at}",
            flush=True,
        )
        for signal in signals:
            print(
                f"  sig {signal['minute']:04d} {signal['side']} {signal['strike']} "
                f"trig={signal['trigger']} entry={signal['option_entry']:.2f} "
                f"fib=({signal['fib_high']:.2f},{signal['fib_low']:.2f}) "
                f"block=({signal['zone_low']},{signal['zone_high']}) "
                f"bout={signal['ob_breakout_minute']} "
                f"s1={signal.get('s1_value')} turn={signal.get('s1_turn')}",
                flush=True,
            )
        for item in filtered:
            snapshot = item["snapshot"] or {}
            print(
                f"  filtered {item['minute']:04d} {item['side']} "
                f"trig={item['trigger']} ut={snapshot.get('ut_color')} "
                f"ha={snapshot.get('ha_close')} plot={snapshot.get('linreg_plot')}",
                flush=True,
            )

    return {
        "day": day,
        "signals": signals,
        "events": list(signals),
        "bars": bars,
        "index_bars": index_bars,
        "spot": spot,
        "records": records,
        "filtered": filtered,
        "current_spot_rows": current_spot_rows,
        "previous_spot_rows": previous_spot_rows,
        "zone_start": 0.0,
        "zone_end": 1.0,
        "ob_blocks": blocks,
        "ob_flavor": flavor,
    }
