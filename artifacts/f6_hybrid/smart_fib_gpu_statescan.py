"""GPU state-machine scan for Smart Fib Optimus preprocessing.

Goal: move the sequential per-series state machines (Wilder ATR, UT Bot
trailing-stop *color*, and the S1 stochastic) fully onto the GPU so that the
event-extraction preprocessing no longer runs on the CPU oracle.

Design (web-research driven, see SMART_FIB_OPTIMUS_RESULTS_2020_2026.md):
  * The recurrences are *per-series* and inherently sequential in time, but
    they are fully independent across the ~100k series. We therefore run a
    batched time-scan: a Python loop over the time axis where every step is a
    vectorized op over the (B, *) batch of series. All tensors live on CUDA.
  * No atomic reductions are used (arxiv 2606.16059: GPU atomics break
    bit-level reproducibility required for audited backtests). Each series is an
    independent strand, so results are deterministic and exactly reproducible.
  * Tensors are padded with NaN; invalid (padded) bars carry the previous state
    forward and emit NaN, so true-length series are handled correctly.

The numeric logic is a line-for-line port of:
  * ``artifacts.f6_hybrid.causal_live_parity_research.IncrementalATR``
  * ``artifacts.f6_hybrid.marni_fib_core_combo_cache.UTColorState``
  * ``flattrade_bot.indicators.stochastic.IncrementalStochastic``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.smart_fib_cuda_bootstrap import configure_cuda_toolkit

configure_cuda_toolkit()

import torch


@torch.no_grad()
def gpu_wilder_atr(open_, high, low, close, period: int = 10, device=None,
                   dtype=torch.float64):
    """Batched Wilder ATR with device-side state updates.

    Inputs are ``(B, T)`` tensors with NaN padding. The only Python loop is
    over time; all series state is updated together on the selected device.
    This preserves the CPU oracle's warmup and missing-bar behavior without
    ``.item()``, ``tolist()``, or per-series dispatch.
    """
    del open_  # ATR uses high, low, and close only.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    high = high.to(device=device, dtype=dtype)
    low = low.to(device=device, dtype=dtype)
    close = close.to(device=device, dtype=dtype)
    B, T = high.shape
    valid = ~torch.isnan(close)

    prev_close = torch.full((B,), float("nan"), device=device, dtype=dtype)
    tr_buf = torch.zeros(B, period, device=device, dtype=dtype)
    write_idx = torch.zeros(B, dtype=torch.long, device=device)
    count = torch.zeros(B, dtype=torch.long, device=device)
    atr_val = torch.full((B,), float("nan"), device=device, dtype=dtype)
    atr_out = torch.full((B, T), float("nan"), device=device, dtype=dtype)
    nan_values = torch.full((B,), float("nan"), device=device, dtype=dtype)

    for t in range(T):
        v = valid[:, t]
        h = high[:, t]
        l = low[:, t]
        c = close[:, t]
        first = ~torch.isfinite(prev_close)
        tr = torch.where(
            first,
            h - l,
            torch.maximum(
                h - l,
                torch.maximum(torch.abs(h - prev_close), torch.abs(l - prev_close)),
            ),
        )

        slot = write_idx.unsqueeze(1)
        old_tr = tr_buf.gather(1, slot).squeeze(1)
        tr_buf.scatter_(1, slot, torch.where(v, tr, old_tr).unsqueeze(1))

        next_count = count + v.to(torch.long)
        next_atr = torch.where(
            next_count == period,
            tr_buf.sum(dim=1) / period,
            torch.where(
                next_count > period,
                (atr_val * (period - 1) + tr) / period,
                atr_val,
            ),
        )
        atr_val = torch.where(v, next_atr, atr_val)
        count = next_count
        write_idx = torch.where(v, (write_idx + 1) % period, write_idx)
        prev_close = torch.where(v, c, prev_close)
        atr_out[:, t] = torch.where(v, atr_val, nan_values)

    return atr_out


@torch.no_grad()
def gpu_ut_color(open_, high, low, close, use_heikin_ashi=False,
                 key: float = 1.0, atr_period: int = 10, device=None,
                 dtype=torch.float64):
    """Batched UT Bot trailing-stop color. Inputs (B, T), NaN-padded.

    Returns (B, T) int8: 1 = green, 0 = red, -1 = invalid/padded.
    Reproduces ``UTColorState.update`` (key=1.0, atr_period=10 by default).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    high = high.to(device=device, dtype=dtype)
    low = low.to(device=device, dtype=dtype)
    close = close.to(device=device, dtype=dtype)
    open_ = open_.to(device=device, dtype=dtype)
    B, T = high.shape
    valid = ~torch.isnan(close)

    atr = gpu_wilder_atr(open_, high, low, close, period=atr_period, device=device,
                        dtype=dtype)

    prev_source = torch.full((B,), float("nan"), device=device, dtype=dtype)
    trailing_stop = torch.zeros(B, device=device, dtype=dtype)
    color_out = torch.full((B, T), -1, dtype=torch.int8, device=device)
    invalid_colors = torch.full((B,), -1, dtype=torch.int8, device=device)

    for t in range(T):
        v = valid[:, t]
        o = open_[:, t]
        c = close[:, t]
        src = (o + high[:, t] + low[:, t] + c) / 4.0 if use_heikin_ashi else c
        atr_t = atr[:, t]
        warm = v & (~torch.isfinite(atr_t) | ~torch.isfinite(prev_source))
        loss = key * atr_t
        candidate = torch.where(
            (src > trailing_stop) & (prev_source > trailing_stop),
            torch.maximum(trailing_stop, src - loss),
            torch.where(
                (src < trailing_stop) & (prev_source < trailing_stop),
                torch.minimum(trailing_stop, src + loss),
                torch.where(src > trailing_stop, src - loss, src + loss),
            ),
        )
        next_stop = torch.where(v & ~warm, candidate, trailing_stop)
        color_out[:, t] = torch.where(
            v,
            (src > next_stop).to(torch.int8),
            invalid_colors,
        )
        trailing_stop = next_stop
        prev_source = torch.where(v, src, prev_source)
    return color_out


@torch.no_grad()
def gpu_stochastic(open_, high, low, close, k_period: int = 12,
                   d_period: int = 3, device=None, dtype=torch.float64):
    """Batched S1 stochastic %D. Inputs (B, T), NaN-padded.

    Returns (B, T) float (NaN until warmup). Reproduces
    ``IncrementalStochastic``. Defaults to float64 to match the float64 CPU
    oracle exactly.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    high = high.to(device=device, dtype=dtype)
    low = low.to(device=device, dtype=dtype)
    close = close.to(device=device, dtype=dtype)
    B, T = high.shape
    valid = ~torch.isnan(close)

    high_buf = torch.zeros(B, k_period, device=device, dtype=dtype)
    low_buf = torch.zeros(B, k_period, device=device, dtype=dtype)
    k_buf = torch.zeros(B, d_period, device=device, dtype=dtype)
    hw = torch.zeros(B, dtype=torch.long, device=device)
    lw = torch.zeros(B, dtype=torch.long, device=device)
    kw = torch.zeros(B, dtype=torch.long, device=device)
    hc = torch.zeros(B, dtype=torch.long, device=device)
    lc = torch.zeros(B, dtype=torch.long, device=device)
    kc = torch.zeros(B, dtype=torch.long, device=device)
    out = torch.full((B, T), float("nan"), device=device, dtype=dtype)
    nan_values = torch.full((B,), float("nan"), device=device, dtype=dtype)

    for t in range(T):
        v = valid[:, t]
        h = high[:, t]
        l = low[:, t]
        c = close[:, t]

        slot_h = hw.unsqueeze(1)
        slot_l = lw.unsqueeze(1)
        old_h = high_buf.gather(1, slot_h).squeeze(1)
        old_l = low_buf.gather(1, slot_l).squeeze(1)
        high_buf.scatter_(1, slot_h, torch.where(v, h, old_h).unsqueeze(1))
        low_buf.scatter_(1, slot_l, torch.where(v, l, old_l).unsqueeze(1))

        next_hc = hc + v.to(torch.long)
        next_lc = lc + v.to(torch.long)
        window_ready = v & (next_hc >= k_period) & (next_lc >= k_period)
        hh = high_buf.max(dim=1).values
        ll = low_buf.min(dim=1).values
        raw_k = torch.where(
            hh == ll,
            torch.full_like(hh, 50.0),
            (c - ll) / (hh - ll) * 100.0,
        )

        k_slot = kw.unsqueeze(1)
        old_k = k_buf.gather(1, k_slot).squeeze(1)
        k_buf.scatter_(1, k_slot, torch.where(window_ready, raw_k, old_k).unsqueeze(1))
        next_kc = kc + window_ready.to(torch.long)
        d_ready = window_ready & (next_kc >= d_period)
        out[:, t] = torch.where(
            d_ready,
            k_buf.sum(dim=1) / d_period,
            nan_values,
        )

        hw = torch.where(v, (hw + 1) % k_period, hw)
        lw = torch.where(v, (lw + 1) % k_period, lw)
        kw = torch.where(window_ready, (kw + 1) % d_period, kw)
        hc = next_hc
        lc = next_lc
        kc = next_kc
    return out


@torch.no_grad()
def gpu_fused_state_scan(
    open_,
    high,
    low,
    close,
    k_period: int = 12,
    d_period: int = 3,
    key: float = 1.0,
    atr_period: int = 10,
    use_heikin_ashi: bool = False,
    device=None,
    dtype=torch.float64,
):
    """Compute ATR, UT color, and stochastic in one device time-scan.

    This is the preferred entry point when all three outputs are required. It
    reads each OHLC bar once per time step and keeps all recurrence state in
    device tensors. The separate public functions remain available for callers
    that need only one indicator.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    open_ = open_.to(device=device, dtype=dtype)
    high = high.to(device=device, dtype=dtype)
    low = low.to(device=device, dtype=dtype)
    close = close.to(device=device, dtype=dtype)
    B, T = high.shape
    valid = ~torch.isnan(close)
    nan_values = torch.full((B,), float("nan"), device=device, dtype=dtype)
    invalid_colors = torch.full((B,), -1, device=device, dtype=torch.int8)

    prev_close = nan_values.clone()
    tr_buf = torch.zeros(B, atr_period, device=device, dtype=dtype)
    atr_write = torch.zeros(B, dtype=torch.long, device=device)
    atr_count = torch.zeros(B, dtype=torch.long, device=device)
    atr_value = nan_values.clone()
    atr_out = torch.full((B, T), float("nan"), device=device, dtype=dtype)

    prev_source = nan_values.clone()
    trailing_stop = torch.zeros(B, device=device, dtype=dtype)
    color_out = torch.full((B, T), -1, device=device, dtype=torch.int8)

    high_buf = torch.zeros(B, k_period, device=device, dtype=dtype)
    low_buf = torch.zeros(B, k_period, device=device, dtype=dtype)
    k_buf = torch.zeros(B, d_period, device=device, dtype=dtype)
    stoch_write = torch.zeros(B, dtype=torch.long, device=device)
    stoch_k_write = torch.zeros(B, dtype=torch.long, device=device)
    stoch_count = torch.zeros(B, dtype=torch.long, device=device)
    stoch_k_count = torch.zeros(B, dtype=torch.long, device=device)
    s1_out = torch.full((B, T), float("nan"), device=device, dtype=dtype)

    for t in range(T):
        v = valid[:, t]
        o = open_[:, t]
        h = high[:, t]
        l = low[:, t]
        c = close[:, t]

        # Wilder ATR state.
        first = ~torch.isfinite(prev_close)
        tr = torch.where(
            first,
            h - l,
            torch.maximum(
                h - l,
                torch.maximum(torch.abs(h - prev_close), torch.abs(l - prev_close)),
            ),
        )
        atr_slot = atr_write.unsqueeze(1)
        old_tr = tr_buf.gather(1, atr_slot).squeeze(1)
        tr_buf.scatter_(1, atr_slot, torch.where(v, tr, old_tr).unsqueeze(1))
        next_atr_count = atr_count + v.to(torch.long)
        next_atr = torch.where(
            next_atr_count == atr_period,
            tr_buf.sum(dim=1) / atr_period,
            torch.where(
                next_atr_count > atr_period,
                (atr_value * (atr_period - 1) + tr) / atr_period,
                atr_value,
            ),
        )
        atr_value = torch.where(v, next_atr, atr_value)
        atr_count = next_atr_count
        atr_write = torch.where(v, (atr_write + 1) % atr_period, atr_write)
        prev_close = torch.where(v, c, prev_close)
        atr_out[:, t] = torch.where(v, atr_value, nan_values)

        # UT Bot color state consumes the ATR value from this same bar.
        source = (o + h + l + c) / 4.0 if use_heikin_ashi else c
        warm = v & (~torch.isfinite(atr_value) | ~torch.isfinite(prev_source))
        loss = key * atr_value
        candidate = torch.where(
            (source > trailing_stop) & (prev_source > trailing_stop),
            torch.maximum(trailing_stop, source - loss),
            torch.where(
                (source < trailing_stop) & (prev_source < trailing_stop),
                torch.minimum(trailing_stop, source + loss),
                torch.where(source > trailing_stop, source - loss, source + loss),
            ),
        )
        trailing_stop = torch.where(v & ~warm, candidate, trailing_stop)
        color_out[:, t] = torch.where(
            v,
            (source > trailing_stop).to(torch.int8),
            invalid_colors,
        )
        prev_source = torch.where(v, source, prev_source)

        # S1 rolling high/low and %D state.
        stoch_slot = stoch_write.unsqueeze(1)
        old_high = high_buf.gather(1, stoch_slot).squeeze(1)
        old_low = low_buf.gather(1, stoch_slot).squeeze(1)
        high_buf.scatter_(1, stoch_slot, torch.where(v, h, old_high).unsqueeze(1))
        low_buf.scatter_(1, stoch_slot, torch.where(v, l, old_low).unsqueeze(1))
        next_stoch_count = stoch_count + v.to(torch.long)
        window_ready = v & (next_stoch_count >= k_period)
        highest = high_buf.max(dim=1).values
        lowest = low_buf.min(dim=1).values
        raw_k = torch.where(
            highest == lowest,
            torch.full_like(highest, 50.0),
            (c - lowest) / (highest - lowest) * 100.0,
        )
        k_slot = stoch_k_write.unsqueeze(1)
        old_k = k_buf.gather(1, k_slot).squeeze(1)
        k_buf.scatter_(1, k_slot, torch.where(window_ready, raw_k, old_k).unsqueeze(1))
        next_k_count = stoch_k_count + window_ready.to(torch.long)
        d_ready = window_ready & (next_k_count >= d_period)
        s1_out[:, t] = torch.where(
            d_ready,
            k_buf.sum(dim=1) / d_period,
            nan_values,
        )
        stoch_write = torch.where(v, (stoch_write + 1) % k_period, stoch_write)
        stoch_k_write = torch.where(
            window_ready,
            (stoch_k_write + 1) % d_period,
            stoch_k_write,
        )
        stoch_count = next_stoch_count
        stoch_k_count = next_k_count

    return atr_out, color_out, s1_out


@torch.no_grad()
def gpu_index5m_filter_closed(
    open_,
    high,
    low,
    close,
    valid=None,
    device=None,
    dtype=torch.float64,
):
    """GPU scan for closed 5m HA + UT + LinReg filter snapshots.

    ``open_``, ``high``, ``low``, and ``close`` are already causally aggregated
    5-minute bars with shape ``(B, M)``. Bucket formation and date-boundary
    handling intentionally stay outside this function because the CPU oracle
    treats forming bars, partial buckets, and date changes specially. Invalid
    closed-bar slots do not advance state and emit NaN/-1 outputs.

    Returned tensors are ``(B, M)``:

    ``ha_open``, ``ha_high``, ``ha_low``, ``ha_close``, ``linreg_plot`` are
    float tensors; ``ut_color`` is int8 (1 green, 0 red, -1 before a snapshot).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    open_ = open_.to(device=device, dtype=dtype)
    high = high.to(device=device, dtype=dtype)
    low = low.to(device=device, dtype=dtype)
    close = close.to(device=device, dtype=dtype)
    B, M = high.shape
    if valid is None:
        valid = ~torch.isnan(close)
    else:
        valid = valid.to(device=device, dtype=torch.bool)
    nan_values = torch.full((B,), float("nan"), device=device, dtype=dtype)

    ha_open = torch.full((B, M), float("nan"), device=device, dtype=dtype)
    ha_high = torch.full((B, M), float("nan"), device=device, dtype=dtype)
    ha_low = torch.full((B, M), float("nan"), device=device, dtype=dtype)
    ha_close = torch.full((B, M), float("nan"), device=device, dtype=dtype)
    previous_ha_open = torch.full((B,), float("nan"), device=device, dtype=dtype)
    previous_ha_close = torch.full((B,), float("nan"), device=device, dtype=dtype)

    # Heikin-Ashi is itself causal and must be completed before UT sees the HA
    # close/high/low series.
    for t in range(M):
        v = valid[:, t]
        raw_open = open_[:, t]
        raw_high = high[:, t]
        raw_low = low[:, t]
        raw_close = close[:, t]
        first = ~torch.isfinite(previous_ha_open)
        current_close = (raw_open + raw_high + raw_low + raw_close) / 4.0
        current_open = torch.where(
            first,
            (raw_open + raw_close) / 2.0,
            (previous_ha_open + previous_ha_close) / 2.0,
        )
        current_high = torch.maximum(
            raw_high,
            torch.maximum(current_open, current_close),
        )
        current_low = torch.minimum(
            raw_low,
            torch.minimum(current_open, current_close),
        )
        previous_ha_open = torch.where(v, current_open, previous_ha_open)
        previous_ha_close = torch.where(v, current_close, previous_ha_close)
        ha_open[:, t] = torch.where(v, current_open, nan_values)
        ha_high[:, t] = torch.where(v, current_high, nan_values)
        ha_low[:, t] = torch.where(v, current_low, nan_values)
        ha_close[:, t] = torch.where(v, current_close, nan_values)

    raw_ut_color = gpu_ut_color(
        ha_open,
        ha_high,
        ha_low,
        ha_close,
        use_heikin_ashi=False,
        device=device,
        dtype=dtype,
    )
    ut_color = torch.full((B, M), -1, device=device, dtype=torch.int8)
    previous_color = torch.full((B,), -1, device=device, dtype=torch.int8)
    for t in range(M):
        v = valid[:, t]
        previous_color = torch.where(v, raw_ut_color[:, t], previous_color)
        ut_color[:, t] = previous_color

    linreg_plot = torch.full((B, M), float("nan"), device=device, dtype=dtype)
    raw_buffer = torch.zeros(B, 11, device=device, dtype=dtype)
    raw_write = torch.zeros(B, dtype=torch.long, device=device)
    raw_count = torch.zeros(B, dtype=torch.long, device=device)
    linreg_buffer = torch.zeros(B, 11, device=device, dtype=dtype)
    linreg_write = torch.zeros(B, dtype=torch.long, device=device)
    linreg_count = torch.zeros(B, dtype=torch.long, device=device)
    current_plot = torch.full((B,), float("nan"), device=device, dtype=dtype)
    weights = torch.arange(11, device=device, dtype=dtype).view(1, 11)
    x_sum = 55.0
    denominator = 1210.0

    for t in range(M):
        v = valid[:, t]
        slot = raw_write.unsqueeze(1)
        old_value = raw_buffer.gather(1, slot).squeeze(1)
        raw_buffer.scatter_(1, slot, torch.where(v, ha_close[:, t], old_value).unsqueeze(1))
        next_count = raw_count + v.to(torch.long)
        ready = v & (next_count >= 11)
        raw_write = torch.where(v, (raw_write + 1) % 11, raw_write)
        raw_count = next_count

        order = (raw_write.unsqueeze(1) + torch.arange(11, device=device)) % 11
        ordered = raw_buffer.gather(1, order)
        y_sum = ordered.sum(dim=1)
        xy_sum = (ordered * weights).sum(dim=1)
        slope = (11.0 * xy_sum - x_sum * y_sum) / denominator
        intercept = (y_sum - slope * x_sum) / 11.0
        bclose = intercept + slope * 10.0

        linreg_slot = linreg_write.unsqueeze(1)
        old_bclose = linreg_buffer.gather(1, linreg_slot).squeeze(1)
        linreg_buffer.scatter_(
            1,
            linreg_slot,
            torch.where(ready, bclose, old_bclose).unsqueeze(1),
        )
        next_linreg_count = linreg_count + ready.to(torch.long)
        plot_ready = ready & (next_linreg_count >= 11)
        current_plot = torch.where(
            plot_ready,
            linreg_buffer.sum(dim=1) / 11.0,
            current_plot,
        )
        linreg_write = torch.where(
            ready,
            (linreg_write + 1) % 11,
            linreg_write,
        )
        linreg_count = next_linreg_count
        linreg_plot[:, t] = current_plot

    return {
        "ha_open": ha_open,
        "ha_high": ha_high,
        "ha_low": ha_low,
        "ha_close": ha_close,
        "linreg_plot": linreg_plot,
        "ut_color": ut_color,
    }


@torch.no_grad()
def gpu_ut_swing_patterns(
    open_,
    high,
    low,
    close,
    color=None,
    minutes=None,
    min_middle: int = 5,
    device=None,
    dtype=torch.float64,
):
    """Detect the Smart Fib bullish/bearish UT swing sequences on GPU.

    The fixed patterns match ``extract_day_events``:

    - bullish: red -> at least ``min_middle`` green bars -> red;
    - bearish: green -> at least ``min_middle`` red bars -> green.

    ``color`` may be supplied as int8 ``1=green, 0=red, -1=invalid``. If it is
    omitted, regular-close UT colors are computed first. The returned tensors
    contain completion records only; pending setup expiry, touch checks, and
    global event ordering remain explicit downstream operations.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    high = high.to(device=device, dtype=dtype)
    low = low.to(device=device, dtype=dtype)
    close = close.to(device=device, dtype=dtype)
    open_ = open_.to(device=device, dtype=dtype)
    B, T = high.shape
    if color is None:
        color = gpu_ut_color(open_, high, low, close, device=device, dtype=dtype)
    else:
        color = color.to(device=device, dtype=torch.int8)
    valid = color >= 0
    if minutes is None:
        minutes = torch.arange(T, device=device, dtype=torch.long).view(1, T).expand(B, -1)
    elif minutes.ndim == 1:
        minutes = minutes.to(device=device, dtype=torch.long).view(1, T).expand(B, -1)
    else:
        minutes = minutes.to(device=device, dtype=torch.long)

    previous_color = torch.full((B,), -1, device=device, dtype=torch.int8)
    previous_high = torch.full((B,), float("nan"), device=device, dtype=dtype)
    previous_low = torch.full((B,), float("nan"), device=device, dtype=dtype)
    previous_minute = torch.full((B,), -1, device=device, dtype=torch.long)
    previous_valid = torch.zeros(B, device=device, dtype=torch.bool)

    bull_active = torch.zeros(B, device=device, dtype=torch.bool)
    bull_count = torch.zeros(B, device=device, dtype=torch.long)
    bull_start = torch.full((B,), -1, device=device, dtype=torch.long)
    bull_high = torch.full((B,), float("nan"), device=device, dtype=dtype)
    bull_low = torch.full((B,), float("nan"), device=device, dtype=dtype)

    bear_active = torch.zeros(B, device=device, dtype=torch.bool)
    bear_count = torch.zeros(B, device=device, dtype=torch.long)
    bear_start = torch.full((B,), -1, device=device, dtype=torch.long)
    bear_high = torch.full((B,), float("nan"), device=device, dtype=dtype)
    bear_low = torch.full((B,), float("nan"), device=device, dtype=dtype)

    bull_completed = torch.zeros((B, T), device=device, dtype=torch.bool)
    bear_completed = torch.zeros((B, T), device=device, dtype=torch.bool)
    bull_start_out = torch.full((B, T), -1, device=device, dtype=torch.long)
    bear_start_out = torch.full((B, T), -1, device=device, dtype=torch.long)
    bull_high_out = torch.full((B, T), float("nan"), device=device, dtype=dtype)
    bull_low_out = torch.full((B, T), float("nan"), device=device, dtype=dtype)
    bear_high_out = torch.full((B, T), float("nan"), device=device, dtype=dtype)
    bear_low_out = torch.full((B, T), float("nan"), device=device, dtype=dtype)

    for t in range(T):
        v = valid[:, t]
        col = color[:, t]
        h = high[:, t]
        l = low[:, t]
        minute = minutes[:, t]

        # Bullish pattern: red -> green* -> red.
        bull_middle = bull_active & v & (col == 1)
        bull_final = bull_active & v & (col == 0)
        bull_done = bull_final & (bull_count >= min_middle)
        bull_reset = bull_active & v & ~bull_middle
        bull_high_at_done = torch.maximum(bull_high, h)
        bull_low_at_done = torch.minimum(bull_low, l)
        bull_started = (
            ~bull_active
            & v
            & (col == 1)
            & previous_valid
            & (previous_color == 0)
        )
        bull_active = torch.where(
            bull_middle,
            torch.ones_like(bull_active),
            torch.where(bull_reset, torch.zeros_like(bull_active), bull_active),
        )
        bull_active = torch.where(bull_started, torch.ones_like(bull_active), bull_active)
        bull_count = torch.where(
            bull_middle,
            bull_count + 1,
            torch.where(bull_started, torch.ones_like(bull_count), bull_count),
        )
        bull_high = torch.where(
            bull_middle,
            torch.maximum(bull_high, h),
            torch.where(bull_started, torch.maximum(previous_high, h), bull_high),
        )
        bull_low = torch.where(
            bull_middle,
            torch.minimum(bull_low, l),
            torch.where(bull_started, torch.minimum(previous_low, l), bull_low),
        )
        bull_start = torch.where(bull_started, previous_minute, bull_start)
        bull_completed[:, t] = bull_done
        bull_start_out[:, t] = torch.where(bull_done, bull_start, torch.full_like(bull_start, -1))
        bull_high_out[:, t] = torch.where(bull_done, bull_high_at_done, torch.full_like(bull_high_at_done, float("nan")))
        bull_low_out[:, t] = torch.where(bull_done, bull_low_at_done, torch.full_like(bull_low_at_done, float("nan")))
        bull_count = torch.where(bull_done | bull_reset, torch.zeros_like(bull_count), bull_count)
        bull_high = torch.where(bull_done | bull_reset, torch.full_like(bull_high, float("nan")), bull_high)
        bull_low = torch.where(bull_done | bull_reset, torch.full_like(bull_low, float("nan")), bull_low)
        bull_start = torch.where(bull_done | bull_reset, torch.full_like(bull_start, -1), bull_start)

        # Bearish pattern: green -> red* -> green.
        bear_middle = bear_active & v & (col == 0)
        bear_final = bear_active & v & (col == 1)
        bear_done = bear_final & (bear_count >= min_middle)
        bear_reset = bear_active & v & ~bear_middle
        bear_high_at_done = torch.maximum(bear_high, h)
        bear_low_at_done = torch.minimum(bear_low, l)
        bear_started = (
            ~bear_active
            & v
            & (col == 0)
            & previous_valid
            & (previous_color == 1)
        )
        bear_active = torch.where(
            bear_middle,
            torch.ones_like(bear_active),
            torch.where(bear_reset, torch.zeros_like(bear_active), bear_active),
        )
        bear_active = torch.where(bear_started, torch.ones_like(bear_active), bear_active)
        bear_count = torch.where(
            bear_middle,
            bear_count + 1,
            torch.where(bear_started, torch.ones_like(bear_count), bear_count),
        )
        bear_high = torch.where(
            bear_middle,
            torch.maximum(bear_high, h),
            torch.where(bear_started, torch.maximum(previous_high, h), bear_high),
        )
        bear_low = torch.where(
            bear_middle,
            torch.minimum(bear_low, l),
            torch.where(bear_started, torch.minimum(previous_low, l), bear_low),
        )
        bear_start = torch.where(bear_started, previous_minute, bear_start)
        bear_completed[:, t] = bear_done
        bear_start_out[:, t] = torch.where(bear_done, bear_start, torch.full_like(bear_start, -1))
        bear_high_out[:, t] = torch.where(bear_done, bear_high_at_done, torch.full_like(bear_high_at_done, float("nan")))
        bear_low_out[:, t] = torch.where(bear_done, bear_low_at_done, torch.full_like(bear_low_at_done, float("nan")))
        bear_count = torch.where(bear_done | bear_reset, torch.zeros_like(bear_count), bear_count)
        bear_high = torch.where(bear_done | bear_reset, torch.full_like(bear_high, float("nan")), bear_high)
        bear_low = torch.where(bear_done | bear_reset, torch.full_like(bear_low, float("nan")), bear_low)
        bear_start = torch.where(bear_done | bear_reset, torch.full_like(bear_start, -1), bear_start)

        previous_color = torch.where(v, col, previous_color)
        previous_high = torch.where(v, h, previous_high)
        previous_low = torch.where(v, l, previous_low)
        previous_minute = torch.where(v, minute, previous_minute)
        previous_valid = previous_valid | v

    return {
        "bullish_completed": bull_completed,
        "bearish_completed": bear_completed,
        "bullish_start_minute": bull_start_out,
        "bearish_start_minute": bear_start_out,
        "bullish_fib_high": bull_high_out,
        "bullish_fib_low": bull_low_out,
        "bearish_fib_high": bear_high_out,
        "bearish_fib_low": bear_low_out,
    }


@torch.no_grad()
def gpu_swing_pending_snapshots(
    high,
    low,
    minutes,
    patterns,
    current_mask=None,
    max_setup_age: int | None = 45,
    replace_setups: bool = False,
    device=None,
    dtype=torch.float64,
):
    """Track latest valid pending Fib setups from GPU pattern completions.

    ``patterns`` is the result of :func:`gpu_ut_swing_patterns`. The default
    ``replace_setups=False`` matches the index feed used by Smart Fib. Setup
    state is filtered causally at each bar for age and swing-base invalidation;
    the newest valid completion for each pattern is returned.

    This function intentionally does not perform touch/zone checks. Those checks
    depend on the strategy variant and belong in the later event-selection
    stage, where candidate ordering and option-bar reachability are explicit.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    high = high.to(device=device, dtype=dtype)
    low = low.to(device=device, dtype=dtype)
    minutes = minutes.to(device=device, dtype=torch.long)
    if minutes.ndim == 1:
        B, T = high.shape
        minutes = minutes.view(1, T).expand(B, -1)
    else:
        B, T = minutes.shape
    if current_mask is None:
        current_mask = torch.ones((B, T), device=device, dtype=torch.bool)
    else:
        current_mask = current_mask.to(device=device, dtype=torch.bool)

    def one_pattern(prefix: str, breaks_above: bool):
        completed = patterns[f"{prefix}_completed"].to(device=device, dtype=torch.bool)
        setup_high = torch.full((B, T), float("nan"), device=device, dtype=dtype)
        setup_low = torch.full((B, T), float("nan"), device=device, dtype=dtype)
        setup_start = torch.full((B, T), -1, device=device, dtype=torch.long)
        setup_completion = torch.full((B, T), -1, device=device, dtype=torch.long)
        setup_valid = torch.zeros((B, T), device=device, dtype=torch.bool)
        latest_valid = torch.zeros((B, T), device=device, dtype=torch.bool)
        latest_high = torch.full((B, T), float("nan"), device=device, dtype=dtype)
        latest_low = torch.full((B, T), float("nan"), device=device, dtype=dtype)
        latest_start = torch.full((B, T), -1, device=device, dtype=torch.long)
        latest_completion = torch.full((B, T), -1, device=device, dtype=torch.long)

        completed_high = patterns[f"{prefix}_fib_high"].to(device=device, dtype=dtype)
        completed_low = patterns[f"{prefix}_fib_low"].to(device=device, dtype=dtype)
        completed_start = patterns[f"{prefix}_start_minute"].to(device=device, dtype=torch.long)

        for t in range(T):
            now = minutes[:, t].unsqueeze(1)
            age_ok = (now - setup_completion) >= 0
            if max_setup_age is not None:
                age_ok &= (now - setup_completion) <= max_setup_age
            if breaks_above:
                base_ok = low[:, t].unsqueeze(1) >= setup_low
            else:
                base_ok = high[:, t].unsqueeze(1) <= setup_high
            setup_valid = setup_valid & age_ok & base_ok

            insert = completed[:, t] & current_mask[:, t]
            if replace_setups:
                setup_valid = torch.where(
                    insert.unsqueeze(1),
                    torch.zeros_like(setup_valid),
                    setup_valid,
                )
            setup_valid[:, t] = insert
            setup_high[:, t] = torch.where(insert, completed_high[:, t], setup_high[:, t])
            setup_low[:, t] = torch.where(insert, completed_low[:, t], setup_low[:, t])
            setup_start[:, t] = torch.where(insert, completed_start[:, t], setup_start[:, t])
            setup_completion[:, t] = torch.where(insert, minutes[:, t], setup_completion[:, t])

            latest_time = torch.where(setup_valid, setup_completion, torch.full_like(setup_completion, -1))
            latest_idx = latest_time.argmax(dim=1)
            has_latest = setup_valid.any(dim=1)
            latest_valid[:, t] = has_latest
            latest_high[:, t] = torch.where(
                has_latest,
                setup_high.gather(1, latest_idx.unsqueeze(1)).squeeze(1),
                torch.full((B,), float("nan"), device=device, dtype=dtype),
            )
            latest_low[:, t] = torch.where(
                has_latest,
                setup_low.gather(1, latest_idx.unsqueeze(1)).squeeze(1),
                torch.full((B,), float("nan"), device=device, dtype=dtype),
            )
            latest_start[:, t] = torch.where(
                has_latest,
                setup_start.gather(1, latest_idx.unsqueeze(1)).squeeze(1),
                torch.full((B,), -1, device=device, dtype=torch.long),
            )
            latest_completion[:, t] = torch.where(
                has_latest,
                setup_completion.gather(1, latest_idx.unsqueeze(1)).squeeze(1),
                torch.full((B,), -1, device=device, dtype=torch.long),
            )

        return {
            "valid": latest_valid,
            "fib_high": latest_high,
            "fib_low": latest_low,
            "start_minute": latest_start,
            "completion_minute": latest_completion,
        }

    return {
        "bullish": one_pattern("bullish", breaks_above=True),
        "bearish": one_pattern("bearish", breaks_above=False),
    }


def demo_parity(day="2020-01-01", device=None):
    """Smoke: compare GPU UT color + S1 against the CPU oracle on one day."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from datetime import date
    from artifacts.flattrade_day_cache import load_day_cache
    from artifacts.f6_hybrid.marni_fib_core_combo_cache import (
        UTColorState, parse_row, normalize_spot,
    )
    from flattrade_bot.indicators.stochastic import IncrementalStochastic

    cache = load_day_cache(Path("artifacts/flattrade_day_cache"), date.fromisoformat(day))
    spot = normalize_spot([parse_row(r) for r in cache["spot_rows"]])
    length = int(spot["min"].max() - spot["min"].min() + 1)

    # Drive the CPU oracle over the SAME ordered present-bar list the GPU
    # tensor is built from, so index alignment is exact (no minute-range gaps).
    n = len(spot["open"])
    cpu_ut = UTColorState(use_heikin_ashi=False)
    cpu_ut_colors = []
    cpu_s1 = IncrementalStochastic(12, 3)
    cpu_s1_vals = []
    for i in range(n):
        candle = type("C", (), {"open": spot["open"][i], "high": spot["high"][i],
                                "low": spot["low"][i], "close": spot["close"][i],
                                "minute": int(spot["min"][i])})()
        cpu_ut_colors.append(cpu_ut.update(candle))
        cpu_s1_vals.append(cpu_s1.push(spot["high"][i], spot["low"][i], spot["close"][i]))

    o = torch.tensor(spot["open"], dtype=torch.float64, device=device)
    h = torch.tensor(spot["high"], dtype=torch.float64, device=device)
    l = torch.tensor(spot["low"], dtype=torch.float64, device=device)
    c = torch.tensor(spot["close"], dtype=torch.float64, device=device)
    g_color = gpu_ut_color(o.unsqueeze(0), h.unsqueeze(0), l.unsqueeze(0),
                           c.unsqueeze(0), device=device)[0].cpu().numpy()
    g_s1 = gpu_stochastic(o.unsqueeze(0), h.unsqueeze(0), l.unsqueeze(0),
                          c.unsqueeze(0), device=device)[0].cpu().numpy()

    color_ok = True
    for i in range(n):
        cpu = cpu_ut_colors[i]
        if cpu is None:
            continue
        if int(g_color[i]) != (1 if cpu == "green" else 0):
            color_ok = False
            print(f"  COLOR MISMATCH bar {i}: cpu={cpu} gpu={int(g_color[i])}")
            break

    s1_ok = True
    for i in range(n):
        cpu = cpu_s1_vals[i]
        if cpu is None:
            continue
        if abs(cpu - g_s1[i]) > 1e-9:
            s1_ok = False
            print(f"  S1 MISMATCH bar {i}: cpu={cpu} gpu={g_s1[i]}")
            break

    print(f"day={day} length={length} bars={n} color_parity={color_ok} s1_parity={s1_ok}")
    return color_ok and s1_ok


if __name__ == "__main__":
    import sys
    days = sys.argv[1:] or ["2020-01-01", "2020-01-02", "2020-01-03"]
    ok = all(demo_parity(d) for d in days)
    print("ALL_PARITY_OK" if ok else "PARITY_FAILED")
