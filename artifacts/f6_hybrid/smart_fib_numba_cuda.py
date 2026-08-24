"""Optional fused Numba-CUDA state scan for Smart Fib preprocessing.

This module is deliberately separate from the PyTorch reference scans. It
provides the real compiled-kernel path once the CUDA Toolkit/NVVM toolchain is
installed, while callers can continue using the tested PyTorch implementation
as the fallback.

The kernel maps one CUDA thread to one independent OHLC series and loops over
time inside that thread. Inputs are converted once to contiguous ``[T, S]``
time-major arrays so neighboring threads read neighboring series values. All
state remains device-side; there are no scalar readbacks or Python operations
inside the time loop.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.smart_fib_cuda_bootstrap import configure_cuda_toolkit

configure_cuda_toolkit()

import torch

try:
    from numba import cuda, float64
    from numba.cuda.cudadrv import nvvm

    NUMBA_CUDA_AVAILABLE = bool(cuda.is_available() and nvvm.is_available())
    NUMBA_CUDA_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on no-CUDA hosts
    cuda = None
    float64 = None
    nvvm = None
    NUMBA_CUDA_AVAILABLE = False
    NUMBA_CUDA_ERROR = f"{type(exc).__name__}: {exc}"


MAX_ATR_PERIOD = 64
MAX_K_PERIOD = 64
MAX_D_PERIOD = 32
DEFAULT_THREADS_PER_BLOCK = 128


if cuda is not None:

    @cuda.jit
    def _fused_state_scan_kernel(
        open_tb,
        high_tb,
        low_tb,
        close_tb,
        atr_tb,
        color_tb,
        s1_tb,
        atr_period,
        k_period,
        d_period,
        key,
        use_heikin_ashi,
    ):
        """One causal ATR + UT color + S1 scan per series."""
        series = cuda.grid(1)
        series_count = high_tb.shape[1]
        if series >= series_count:
            return

        atr_buffer = cuda.local.array(MAX_ATR_PERIOD, dtype=float64)
        high_buffer = cuda.local.array(MAX_K_PERIOD, dtype=float64)
        low_buffer = cuda.local.array(MAX_K_PERIOD, dtype=float64)
        k_buffer = cuda.local.array(MAX_D_PERIOD, dtype=float64)

        atr_write = 0
        atr_count = 0
        previous_close = math.nan
        atr_value = math.nan

        previous_source = math.nan
        trailing_stop = 0.0

        price_write = 0
        price_count = 0
        k_write = 0
        k_count = 0

        for t in range(high_tb.shape[0]):
            o = open_tb[t, series]
            h = high_tb[t, series]
            l = low_tb[t, series]
            c = close_tb[t, series]

            if math.isnan(c):
                atr_tb[t, series] = math.nan
                color_tb[t, series] = -1
                s1_tb[t, series] = math.nan
                continue

            # Wilder ATR, matching IncrementalATR.
            if math.isnan(previous_close):
                true_range = h - l
            else:
                true_range = max(h - l, abs(h - previous_close), abs(l - previous_close))
            atr_buffer[atr_write] = true_range
            atr_count += 1
            next_atr_write = (atr_write + 1) % atr_period
            if atr_count == atr_period:
                tr_sum = 0.0
                for j in range(MAX_ATR_PERIOD):
                    if j < atr_period:
                        tr_sum += atr_buffer[(next_atr_write + j) % atr_period]
                atr_value = tr_sum / atr_period
            elif atr_count > atr_period:
                atr_value = (atr_value * (atr_period - 1) + true_range) / atr_period
            atr_write = next_atr_write
            previous_close = c
            atr_tb[t, series] = atr_value

            # UTColorState: update source every valid bar, but not the stop
            # during ATR/previous-source warmup.
            if use_heikin_ashi:
                source = (o + h + l + c) / 4.0
            else:
                source = c
            previous_stop = trailing_stop
            if math.isnan(atr_value) or math.isnan(previous_source):
                color = 1 if source > previous_stop else 0
            else:
                loss = key * atr_value
                if source > previous_stop and previous_source > previous_stop:
                    trailing_stop = max(previous_stop, source - loss)
                elif source < previous_stop and previous_source < previous_stop:
                    trailing_stop = min(previous_stop, source + loss)
                elif source > previous_stop:
                    trailing_stop = source - loss
                else:
                    trailing_stop = source + loss
                color = 1 if source > trailing_stop else 0
            previous_source = source
            color_tb[t, series] = color

            # Incremental stochastic %D, matching IncrementalStochastic.
            high_buffer[price_write] = h
            low_buffer[price_write] = l
            price_write = (price_write + 1) % k_period
            price_count += 1
            if price_count < k_period:
                s1_tb[t, series] = math.nan
                continue

            highest = high_buffer[0]
            lowest = low_buffer[0]
            for j in range(MAX_K_PERIOD):
                if j < k_period:
                    highest = max(highest, high_buffer[j])
                    lowest = min(lowest, low_buffer[j])
            if highest == lowest:
                raw_k = 50.0
            else:
                raw_k = ((c - lowest) / (highest - lowest)) * 100.0
            k_buffer[k_write] = raw_k
            next_k_write = (k_write + 1) % d_period
            k_write = next_k_write
            k_count += 1
            if k_count < d_period:
                s1_tb[t, series] = math.nan
                continue

            k_sum = 0.0
            for j in range(MAX_D_PERIOD):
                if j < d_period:
                    k_sum += k_buffer[(next_k_write + j) % d_period]
            s1_tb[t, series] = k_sum / d_period


def numba_cuda_status() -> dict[str, object]:
    """Return diagnostic state without throwing on an unconfigured machine."""
    status: dict[str, object] = {
        "available": NUMBA_CUDA_AVAILABLE,
        "error": NUMBA_CUDA_ERROR,
        "cuda_home": os.environ.get("CUDA_HOME"),
        "nvvm_available": bool(nvvm is not None and nvvm.is_available()),
    }
    if cuda is not None:
        try:
            device = cuda.get_current_device()
            name = device.name
            status["device"] = name.decode() if isinstance(name, bytes) else str(name)
            status["compute_capability"] = device.compute_capability
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            status["device_error"] = f"{type(exc).__name__}: {exc}"
    return status


@torch.no_grad()
def numba_fused_state_scan(
    open_,
    high,
    low,
    close,
    atr_period: int = 10,
    k_period: int = 12,
    d_period: int = 3,
    key: float = 1.0,
    use_heikin_ashi: bool = False,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
):
    """Run the compiled fused state scan and return PyTorch CUDA tensors.

    Inputs and outputs use the public module's ``(B, T)`` convention. The
    internal kernel uses contiguous ``(T, B)`` buffers. This function is
    intentionally explicit; callers should catch ``RuntimeError`` and fall
    back to ``gpu_fused_state_scan`` until parity is approved.
    """
    if not NUMBA_CUDA_AVAILABLE:
        raise RuntimeError(
            "Numba CUDA is unavailable. Run numba_cuda_status() first. "
            f"{NUMBA_CUDA_ERROR or 'CUDA Toolkit/NVVM is not configured.'}"
        )
    if not (1 <= atr_period <= MAX_ATR_PERIOD):
        raise ValueError(f"atr_period must be 1..{MAX_ATR_PERIOD}")
    if not (1 <= k_period <= MAX_K_PERIOD):
        raise ValueError(f"k_period must be 1..{MAX_K_PERIOD}")
    if not (1 <= d_period <= MAX_D_PERIOD):
        raise ValueError(f"d_period must be 1..{MAX_D_PERIOD}")
    if threads_per_block <= 0 or threads_per_block % 32:
        raise ValueError("threads_per_block must be a positive multiple of 32")
    if not all(isinstance(value, torch.Tensor) for value in (open_, high, low, close)):
        raise TypeError("OHLC inputs must be torch tensors")
    if not all(value.is_cuda for value in (open_, high, low, close)):
        raise ValueError("OHLC inputs must already be on CUDA")
    if not all(value.dtype == torch.float64 for value in (open_, high, low, close)):
        raise ValueError("The audit kernel currently requires float64 OHLC tensors")
    if len({value.shape for value in (open_, high, low, close)}) != 1:
        raise ValueError("OHLC tensors must have identical shapes")

    open_tb = open_.transpose(0, 1).contiguous()
    high_tb = high.transpose(0, 1).contiguous()
    low_tb = low.transpose(0, 1).contiguous()
    close_tb = close.transpose(0, 1).contiguous()
    time_count, series_count = high_tb.shape
    atr_tb = torch.empty_like(high_tb)
    s1_tb = torch.empty_like(high_tb)
    color_tb = torch.empty((time_count, series_count), dtype=torch.int8, device=high.device)

    d_open = cuda.as_cuda_array(open_tb)
    d_high = cuda.as_cuda_array(high_tb)
    d_low = cuda.as_cuda_array(low_tb)
    d_close = cuda.as_cuda_array(close_tb)
    d_atr = cuda.as_cuda_array(atr_tb)
    d_color = cuda.as_cuda_array(color_tb)
    d_s1 = cuda.as_cuda_array(s1_tb)
    blocks = (series_count + threads_per_block - 1) // threads_per_block
    _fused_state_scan_kernel[blocks, threads_per_block](
        d_open,
        d_high,
        d_low,
        d_close,
        d_atr,
        d_color,
        d_s1,
        atr_period,
        k_period,
        d_period,
        float(key),
        use_heikin_ashi,
    )
    cuda.synchronize()
    return (
        atr_tb.transpose(0, 1).contiguous(),
        color_tb.transpose(0, 1).contiguous(),
        s1_tb.transpose(0, 1).contiguous(),
    )


if __name__ == "__main__":
    print(numba_cuda_status())
