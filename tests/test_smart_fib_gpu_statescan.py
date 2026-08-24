from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from artifacts.f6_hybrid.causal_live_parity_research import IncrementalATR
from artifacts.f6_hybrid.marni_fib_5y_fast import Candle, HeikinAshiState
from artifacts.f6_hybrid.marni_fib_core_combo_cache import (
    Index5mFilter,
    UTColorState,
    UTSwingPattern,
)
from artifacts.f6_hybrid.smart_fib_numba_cuda import (
    NUMBA_CUDA_AVAILABLE,
    numba_fused_state_scan,
)
import torch
from artifacts.f6_hybrid.smart_fib_optimus_gpu import _parameter_tensors
from artifacts.f6_hybrid.smart_fib_gpu_statescan import (
    gpu_fused_state_scan,
    gpu_index5m_filter_closed,
    gpu_stochastic,
    gpu_swing_pending_snapshots,
    gpu_ut_swing_patterns,
    gpu_ut_color,
    gpu_wilder_atr,
)
from flattrade_bot.indicators.stochastic import IncrementalStochastic


def test_parameter_controls_follow_market_dtype():
    params = [{
        "stop_level": 1.155,
        "target_level": 0.29,
        "fallback_target_level": 0.0,
        "option_point_threshold": 10.0,
    }]
    controls = _parameter_tensors(params, torch.device("cpu"), dtype=torch.float64)
    assert all(value.dtype == torch.float64 for value in controls.values())


@pytest.mark.skipif(
    not torch.cuda.is_available() or not NUMBA_CUDA_AVAILABLE,
    reason="CUDA Toolkit/NVVM is required",
)
def test_numba_fused_scan_matches_cpu_with_gaps():
    rng = np.random.default_rng(20260819)
    batch, bars = 4, 72
    close = 150.0 + np.cumsum(rng.normal(0.0, 0.25, (batch, bars)), axis=1)
    high = close + rng.uniform(0.05, 0.35, (batch, bars))
    low = close - rng.uniform(0.05, 0.35, (batch, bars))
    open_ = close + rng.normal(0.0, 0.05, (batch, bars))
    valid = np.ones((batch, bars), dtype=bool)
    valid[0, 0:2] = False
    valid[1, 31:34] = False
    valid[2, 60:] = False
    valid[3, 11] = False
    for values in (open_, high, low, close):
        values[~valid] = np.nan

    device = torch.device("cuda")
    tensors = [torch.tensor(values, dtype=torch.float64, device=device)
               for values in (open_, high, low, close)]
    numba_atr, numba_color, numba_s1 = numba_fused_state_scan(*tensors)
    numba_atr = numba_atr.cpu().numpy()
    numba_color = numba_color.cpu().numpy()
    numba_s1 = numba_s1.cpu().numpy()

    cpu_atr = np.full((batch, bars), np.nan, dtype=np.float64)
    cpu_color = np.full((batch, bars), -1, dtype=np.int8)
    cpu_s1 = np.full((batch, bars), np.nan, dtype=np.float64)
    for row in range(batch):
        atr = IncrementalATR(10)
        ut = UTColorState(use_heikin_ashi=False)
        stochastic = IncrementalStochastic(12, 3)
        for index in range(bars):
            if not valid[row, index]:
                continue
            atr_value = atr.update(high[row, index], low[row, index], close[row, index])
            if atr_value is not None:
                cpu_atr[row, index] = atr_value
            candle = SimpleNamespace(
                open=open_[row, index], high=high[row, index],
                low=low[row, index], close=close[row, index],
            )
            cpu_color[row, index] = 1 if ut.update(candle) == "green" else 0
            s1_value = stochastic.push(high[row, index], low[row, index], close[row, index])
            if s1_value is not None:
                cpu_s1[row, index] = s1_value

    np.testing.assert_allclose(numba_atr, cpu_atr, rtol=0.0, atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(numba_color, cpu_color)
    np.testing.assert_allclose(numba_s1, cpu_s1, rtol=0.0, atol=1e-9, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vectorized_state_scans_match_cpu_with_gaps():
    rng = np.random.default_rng(20260817)
    batch, bars = 5, 96
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.35, (batch, bars)), axis=1)
    high = close + rng.uniform(0.05, 0.45, (batch, bars))
    low = close - rng.uniform(0.05, 0.45, (batch, bars))
    open_ = close + rng.normal(0.0, 0.08, (batch, bars))

    valid = np.ones((batch, bars), dtype=bool)
    valid[0, 0:3] = False
    valid[1, 41:44] = False
    valid[2, 72:] = False
    valid[3, 7] = False
    valid[4, 20:25] = False
    for values in (open_, high, low, close):
        values[~valid] = np.nan

    device = torch.device("cuda")
    tensors = [torch.tensor(values, dtype=torch.float64, device=device)
               for values in (open_, high, low, close)]
    gpu_atr = gpu_wilder_atr(*tensors, device=device).cpu().numpy()
    gpu_color = gpu_ut_color(*tensors, device=device).cpu().numpy()
    gpu_s1 = gpu_stochastic(*tensors, device=device).cpu().numpy()
    fused_atr, fused_color, fused_s1 = gpu_fused_state_scan(*tensors, device=device)
    fused_atr = fused_atr.cpu().numpy()
    fused_color = fused_color.cpu().numpy()
    fused_s1 = fused_s1.cpu().numpy()

    cpu_atr = np.full((batch, bars), np.nan, dtype=np.float64)
    cpu_color = np.full((batch, bars), -1, dtype=np.int8)
    cpu_s1 = np.full((batch, bars), np.nan, dtype=np.float64)
    for row in range(batch):
        atr = IncrementalATR(10)
        ut = UTColorState(use_heikin_ashi=False)
        stochastic = IncrementalStochastic(12, 3)
        for index in range(bars):
            if not valid[row, index]:
                continue
            atr_value = atr.update(high[row, index], low[row, index], close[row, index])
            if atr_value is not None:
                cpu_atr[row, index] = atr_value
            candle = SimpleNamespace(
                open=open_[row, index],
                high=high[row, index],
                low=low[row, index],
                close=close[row, index],
            )
            cpu_color[row, index] = 1 if ut.update(candle) == "green" else 0
            s1_value = stochastic.push(high[row, index], low[row, index], close[row, index])
            if s1_value is not None:
                cpu_s1[row, index] = s1_value

    np.testing.assert_allclose(gpu_atr, cpu_atr, rtol=0.0, atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(gpu_color, cpu_color)
    np.testing.assert_allclose(gpu_s1, cpu_s1, rtol=0.0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(fused_atr, cpu_atr, rtol=0.0, atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(fused_color, cpu_color)
    np.testing.assert_allclose(fused_s1, cpu_s1, rtol=0.0, atol=1e-9, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_closed_5m_filter_scan_matches_cpu_oracle():
    rng = np.random.default_rng(20260818)
    batch, bars = 3, 24
    close = 200.0 + np.cumsum(rng.normal(0.0, 0.5, (batch, bars)), axis=1)
    high = close + rng.uniform(0.1, 0.8, (batch, bars))
    low = close - rng.uniform(0.1, 0.8, (batch, bars))
    open_ = close + rng.normal(0.0, 0.12, (batch, bars))
    valid = np.ones((batch, bars), dtype=bool)
    valid[1, :2] = False
    valid[1, 13] = False
    valid[2, 18:] = False
    for values in (open_, high, low, close):
        values[~valid] = np.nan

    device = torch.device("cuda")
    tensors = [torch.tensor(values, dtype=torch.float64, device=device)
               for values in (open_, high, low, close)]
    gpu = gpu_index5m_filter_closed(*tensors, valid=torch.tensor(valid, device=device))
    gpu = {name: value.cpu().numpy() for name, value in gpu.items()}

    expected = {
        name: np.full((batch, bars), np.nan, dtype=np.float64)
        for name in ("ha_open", "ha_high", "ha_low", "ha_close", "linreg_plot")
    }
    expected_color = np.full((batch, bars), -1, dtype=np.int8)
    for row in range(batch):
        ha = HeikinAshiState()
        ut = UTColorState(use_heikin_ashi=False)
        raw_closes = deque(maxlen=11)
        linreg_values = deque(maxlen=11)
        for index in range(bars):
            if not valid[row, index]:
                continue
            aggregate = Candle(
                open_[row, index], high[row, index], low[row, index],
                close[row, index], minute=555 + index * 5,
            )
            snapshot = Index5mFilter._calculate(
                aggregate, ha, ut, raw_closes, linreg_values, forming=False,
            )
            for name in expected:
                if snapshot[name] is not None:
                    expected[name][row, index] = snapshot[name]
            expected_color[row, index] = 1 if snapshot["ut_color"] == "green" else 0

    for name in expected:
        np.testing.assert_allclose(
            gpu[name][valid], expected[name][valid], rtol=0.0, atol=1e-9,
        )
    np.testing.assert_array_equal(gpu["ut_color"][valid], expected_color[valid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ut_swing_pattern_scan_matches_cpu_oracle():
    colors = np.array([
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    ], dtype=np.int8)
    batch, bars = colors.shape
    minutes = np.arange(100, 100 + bars, dtype=np.int64)
    close = np.tile(np.linspace(100.0, 111.0, bars), (batch, 1))
    open_ = close - 0.2
    high = close + np.arange(bars, dtype=np.float64) * 0.1 + 0.5
    low = close - np.arange(bars, dtype=np.float64) * 0.1 - 0.5

    device = torch.device("cuda")
    tensors = [torch.tensor(values, dtype=torch.float64, device=device)
               for values in (open_, high, low, close)]
    gpu = gpu_ut_swing_patterns(
        *tensors,
        color=torch.tensor(colors, dtype=torch.int8, device=device),
        minutes=torch.tensor(minutes, dtype=torch.long, device=device),
        min_middle=5,
        device=device,
    )
    gpu = {name: value.cpu().numpy() for name, value in gpu.items()}

    expected = {
        "bullish_completed": np.zeros((batch, bars), dtype=bool),
        "bearish_completed": np.zeros((batch, bars), dtype=bool),
        "bullish_start_minute": np.full((batch, bars), -1, dtype=np.int64),
        "bearish_start_minute": np.full((batch, bars), -1, dtype=np.int64),
        "bullish_fib_high": np.full((batch, bars), np.nan),
        "bullish_fib_low": np.full((batch, bars), np.nan),
        "bearish_fib_high": np.full((batch, bars), np.nan),
        "bearish_fib_low": np.full((batch, bars), np.nan),
    }
    for row in range(batch):
        patterns = [
            UTSwingPattern("bullish", "red", "green", "red", "high_to_low", 5),
            UTSwingPattern("bearish", "green", "red", "green", "low_to_high", 5),
        ]
        for index in range(bars):
            candle = SimpleNamespace(
                open=open_[row, index], high=high[row, index],
                low=low[row, index], close=close[row, index],
                minute=int(minutes[index]),
            )
            color = "green" if colors[row, index] else "red"
            for pattern in patterns:
                completed = pattern.update(candle, color)
                if completed is None:
                    continue
                prefix = "bullish" if completed["pattern"] == "bullish" else "bearish"
                expected[f"{prefix}_completed"][row, index] = True
                expected[f"{prefix}_start_minute"][row, index] = completed["start_minute"]
                expected[f"{prefix}_fib_high"][row, index] = completed["fib_high"]
                expected[f"{prefix}_fib_low"][row, index] = completed["fib_low"]

    np.testing.assert_array_equal(gpu["bullish_completed"], expected["bullish_completed"])
    np.testing.assert_array_equal(gpu["bearish_completed"], expected["bearish_completed"])
    np.testing.assert_array_equal(gpu["bullish_start_minute"], expected["bullish_start_minute"])
    np.testing.assert_array_equal(gpu["bearish_start_minute"], expected["bearish_start_minute"])
    for name in ("bullish_fib_high", "bullish_fib_low", "bearish_fib_high", "bearish_fib_low"):
        np.testing.assert_allclose(gpu[name], expected[name], rtol=0.0, atol=1e-9, equal_nan=True)

    pending = gpu_swing_pending_snapshots(
        torch.tensor(high, dtype=torch.float64, device=device),
        torch.tensor(low, dtype=torch.float64, device=device),
        torch.tensor(minutes, dtype=torch.long, device=device),
        {name: torch.tensor(value, device=device) for name, value in gpu.items()},
        max_setup_age=45,
        device=device,
    )
    pending = {
        side: {name: value.cpu().numpy() for name, value in values.items()}
        for side, values in pending.items()
    }
    expected_pending = {
        side: {
            "valid": np.zeros((batch, bars), dtype=bool),
            "fib_high": np.full((batch, bars), np.nan),
            "fib_low": np.full((batch, bars), np.nan),
            "start_minute": np.full((batch, bars), -1, dtype=np.int64),
            "completion_minute": np.full((batch, bars), -1, dtype=np.int64),
        }
        for side in ("bullish", "bearish")
    }
    for row in range(batch):
        active = {"bullish": [], "bearish": []}
        for index in range(bars):
            now = int(minutes[index])
            for side in active:
                active[side] = [
                    setup for setup in active[side]
                    if now - setup["completion_minute"] <= 45
                    and (low[row, index] >= setup["fib_low"] if side == "bullish"
                         else high[row, index] <= setup["fib_high"])
                ]
                if expected[f"{side}_completed"][row, index]:
                    active[side].append({
                        "fib_high": expected[f"{side}_fib_high"][row, index],
                        "fib_low": expected[f"{side}_fib_low"][row, index],
                        "start_minute": expected[f"{side}_start_minute"][row, index],
                        "completion_minute": now,
                    })
                if active[side]:
                    setup = max(active[side], key=lambda item: item["completion_minute"])
                    expected_pending[side]["valid"][row, index] = True
                    expected_pending[side]["fib_high"][row, index] = setup["fib_high"]
                    expected_pending[side]["fib_low"][row, index] = setup["fib_low"]
                    expected_pending[side]["start_minute"][row, index] = setup["start_minute"]
                    expected_pending[side]["completion_minute"][row, index] = setup["completion_minute"]

    for side in expected_pending:
        np.testing.assert_array_equal(pending[side]["valid"], expected_pending[side]["valid"])
        np.testing.assert_array_equal(pending[side]["start_minute"], expected_pending[side]["start_minute"])
        np.testing.assert_array_equal(pending[side]["completion_minute"], expected_pending[side]["completion_minute"])
        np.testing.assert_allclose(
            pending[side]["fib_high"], expected_pending[side]["fib_high"],
            rtol=0.0, atol=1e-9, equal_nan=True,
        )
        np.testing.assert_allclose(
            pending[side]["fib_low"], expected_pending[side]["fib_low"],
            rtol=0.0, atol=1e-9, equal_nan=True,
        )
