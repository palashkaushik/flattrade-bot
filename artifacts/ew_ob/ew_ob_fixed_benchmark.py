"""Fixed-entry EW-OB SL/TP benchmark with an optional CUDA exit matrix.

The stateful EW-OB engine runs once to create the live-parity entry stream.
Only exits are batched across risk configurations, so parameter changes cannot
alter wave detection, order-block selection, queue order, or entries.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import opt_futures_quad as source

from artifacts.ew_ob.ew_ob_engine import (
    LOT_SIZE,
    RISK_MODE_ATR,
    SLIPPAGE_PTS,
    trade_cost,
)
from artifacts.ew_ob.ew_ob_runner import _augment, make_option_resolver, run_engine, summarize

START = "2020-01-01"
END = "2026-08-20"
MULTIPLIERS = (1.0, 2.0, 3.0, 4.0, 5.0)
FOLDS = ("2023", "2024", "2025", "2026")
SESSION_BARS = 375


def load_data():
    spot = source.load_spot()
    options = source.option_day_files(START, END)
    options, spot = _augment(options, spot)
    days = sorted(d for d in set(spot) & set(options) if START <= d <= END)
    return spot, options, days


def smoke_gate(spot, options, days, workers):
    anchor = [d for d in days if d >= "2026-08-18"][:3]
    reference = [d for d in days if d.startswith("2020")][:5]
    if not anchor or not reference:
        raise RuntimeError("fixed-entry smoke windows are incomplete")
    for sample in (anchor, reference):
        trades = run_engine(spot, options, sample, 0.5, 3.0, 60.0, "ob_w5", None, workers)
        if not trades:
            raise RuntimeError(f"fixed-entry smoke failed for {sample[0]}..{sample[-1]}")
    print(
        f"SMOKE GATE passed: baseline entries on {anchor[0]}..{anchor[-1]} "
        f"and {reference[0]}..{reference[-1]}",
        flush=True,
    )
    return anchor, reference


def pack_spot(spot, days):
    day_index = {day: i for i, day in enumerate(days)}
    high = np.zeros((len(days), SESSION_BARS), dtype=np.float32)
    low = np.zeros_like(high)
    close = np.zeros_like(high)
    for day, index in day_index.items():
        sp = spot[day]
        for i, minute in enumerate(sp["min"]):
            bar = int(minute) - 555
            if 0 <= bar < SESSION_BARS:
                high[index, bar] = float(sp["high"][i])
                low[index, bar] = float(sp["low"][i])
                close[index, bar] = float(sp["close"][i])
    return high, low, close, day_index


def fixed_entry_arrays(trades, day_index):
    rows = []
    for trade in trades:
        rows.append({
            "day_idx": day_index[trade["date"]],
            "entry_idx": trade["entry_min"] - 555,
            "direction": 1 if trade["direction"] == "bull" else -1,
            "entry": float(trade["entry"]),
            "atr": float(trade["atr_entry"]),
            "entry_prem": float(trade["entry_prem"]),
            "side": trade["side"],
            "strike": trade["strike"],
            "date": trade["date"],
            "entry_min": trade["entry_min"],
            "timeframe": trade["timeframe"],
            "zero": trade["wave_zero_minute"],
            "entry_ob_lo": trade["entry_ob_lo"],
            "entry_ob_hi": trade["entry_ob_hi"],
            "same_tf_ob_lo": trade["same_tf_ob_lo"],
            "same_tf_ob_hi": trade["same_tf_ob_hi"],
        })
    return rows


def config_label(config):
    return config["label"]


def build_configs():
    configs = [
        {"label": "baseline_ob_w5", "mode": "ob_w5"},
        {"label": "ob_same_tf", "mode": "ob_same_tf"},
        {"label": "atr_3_3", "mode": "atr", "sl": 3.0, "tp": 3.0},
        {"label": "atr_5_5", "mode": "atr", "sl": 5.0, "tp": 5.0},
    ]
    for sl in MULTIPLIERS:
        for tp in MULTIPLIERS:
            configs.append({
                "label": f"atr_{sl:.0f}_{tp:.0f}",
                "mode": "atr",
                "sl": sl,
                "tp": tp,
            })
    seen = set()
    unique = []
    for config in configs:
        if config["label"] not in seen:
            seen.add(config["label"])
            unique.append(config)
    return unique


def prices_for_config(rows, config):
    sl = np.zeros(len(rows), dtype=np.float32)
    tp = np.zeros(len(rows), dtype=np.float32)
    for i, row in enumerate(rows):
        if config["mode"] == "atr":
            sl_mult = config["sl"]
            tp_mult = config["tp"]
            if row["direction"] == 1:
                sl[i] = row["entry"] - sl_mult * row["atr"]
                tp[i] = row["entry"] + tp_mult * row["atr"]
            else:
                sl[i] = row["entry"] + sl_mult * row["atr"]
                tp[i] = row["entry"] - tp_mult * row["atr"]
        elif config["mode"] == "ob_same_tf":
            lo = row["same_tf_ob_lo"]
            hi = row["same_tf_ob_hi"]
            if lo is None or hi is None:
                lo = row["entry_ob_lo"]
                hi = row["entry_ob_hi"]
            sl[i] = lo if row["direction"] == 1 else hi
            tp[i] = row["entry"]  # replaced by the stored baseline W5 target below
        else:
            sl[i] = row["entry_ob_lo"] if row["direction"] == 1 else row["entry_ob_hi"]
            tp[i] = row["entry"]  # baseline W5 target is not in the compact row
    return sl, tp


def cpu_exits(rows, config, spot, options, day_index, high_mat=None, low_mat=None, close_mat=None):
    resolver = make_option_resolver(options)
    output = []
    for row in rows:
        if config["mode"] == "atr":
            sl, tp = prices_for_config([row], config)
            sl, tp = float(np.float32(sl[0])), float(np.float32(tp[0]))
        else:
            sl = float(np.float32(row["entry_ob_lo"] if row["direction"] == 1 else row["entry_ob_hi"]))
            # The fixed-entry baseline carries its W5 TP in the source trade.
            tp = float(np.float32(row["baseline_tp"]))
            if config["mode"] == "ob_same_tf":
                if row["same_tf_ob_lo"] is not None:
                    sl = float(np.float32(row["same_tf_ob_lo"] if row["direction"] == 1 else row["same_tf_ob_hi"]))
        if high_mat is not None and low_mat is not None and close_mat is not None:
            day_idx = row["day_idx"]
            high_row = high_mat[day_idx]
            low_row = low_mat[day_idx]
            close_row = close_mat[day_idx]
        else:
            sp = spot[row["date"]]
            high_row = sp["high"]
            low_row = sp["low"]
            close_row = sp["close"]
            # sp arrays may be sparse for truncated sessions; map via minute
            if len(high_row) != SESSION_BARS:
                # fallback to packed mats if available, else search by minute
                high_row = None
                low_row = None
                close_row = None
        entry_index = row["entry_idx"]
        exit_index = None
        reason = "EOD"
        for index in range(entry_index + 1, SESSION_BARS):
            if high_row is not None and low_row is not None:
                high = float(high_row[index])
                low = float(low_row[index])
                if high == 0 and low == 0:
                    continue
            else:
                # sparse fallback: find bar by minute
                sp = spot[row["date"]]
                pos = int(np.searchsorted(sp["min"], 555 + index))
                if pos >= len(sp["min"]) or int(sp["min"][pos]) != 555 + index:
                    continue
                high = float(sp["high"][pos])
                low = float(sp["low"][pos])
            if row["direction"] == 1:
                if low <= sl:
                    exit_index, reason = index, "SL"
                    break
                if high >= tp:
                    exit_index, reason = index, "TP"
                    break
            else:
                if high >= sl:
                    exit_index, reason = index, "SL"
                    break
                if low <= tp:
                    exit_index, reason = index, "TP"
                    break
        if exit_index is None:
            if high_row is not None and low_row is not None:
                for idx in range(SESSION_BARS - 1, entry_index, -1):
                    if float(high_row[idx]) != 0 or float(low_row[idx]) != 0:
                        exit_index = idx
                        break
                else:
                    exit_index = entry_index
            else:
                exit_index = SESSION_BARS - 1
        exit_min = 555 + exit_index
        if high_mat is not None and low_mat is not None and close_mat is not None:
            exit_close = float(close_mat[row["day_idx"], exit_index])
            if exit_close == 0:
                sp2 = spot[row["date"]]
                pos2 = int(np.searchsorted(sp2["min"], exit_min))
                if 0 <= pos2 < len(sp2["min"]) and int(sp2["min"][pos2]) == exit_min:
                    exit_close = float(sp2["close"][pos2])
                else:
                    exit_close = float(row["entry"])
        else:
            exit_close = float(sp["close"][exit_index])
        exit_prem = resolver(row["date"], row["side"], exit_min, exit_close, row["strike"])
        if exit_prem is None:
            exit_prem = row["entry_prem"]
        points = (exit_close - row["entry"]) if row["direction"] == 1 else (row["entry"] - exit_close)
        points_net = points - 2 * SLIPPAGE_PTS
        fee = trade_cost(row["entry_prem"], exit_prem)
        output.append({
            "date": row["date"],
            "entry_min": row["entry_min"],
            "exit_min": exit_min,
            "timeframe": row["timeframe"],
            "direction": "bull" if row["direction"] == 1 else "bear",
            "side": row["side"],
            "strike": row["strike"],
            "entry": row["entry"],
            "exit": exit_close,
            "sl": sl,
            "tp": tp,
            "exit_reason": reason,
            "pts": round(points, 2),
            "pts_net": round(points_net, 2),
            "entry_prem": row["entry_prem"],
            "exit_prem": exit_prem,
            "fee": fee,
            "rs_net": round((exit_prem - row["entry_prem"]) * LOT_SIZE - fee, 2),
        })
    return output


def torch_exits(rows, configs, high, low, close):
    """Batch index exit detection on CUDA; CPU fallback preserves the same API."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    day_idx = torch.tensor([r["day_idx"] for r in rows], dtype=torch.long, device=device)
    entry_idx = torch.tensor([r["entry_idx"] for r in rows], dtype=torch.long, device=device)
    direction = torch.tensor([r["direction"] for r in rows], dtype=torch.int8, device=device)
    entries = torch.tensor([r["entry"] for r in rows], dtype=torch.float32, device=device)
    atr = torch.tensor([r["atr"] for r in rows], dtype=torch.float32, device=device)
    h = torch.tensor(high, dtype=torch.float32, device=device)
    l = torch.tensor(low, dtype=torch.float32, device=device)
    c = torch.tensor(close, dtype=torch.float32, device=device)
    offsets = torch.arange(SESSION_BARS, dtype=torch.long, device=device)
    positions = entry_idx[:, None] + 1 + offsets[None, :]
    valid = positions < SESSION_BARS
    clipped = positions.clamp(max=SESSION_BARS - 1)
    future_h = h[day_idx[:, None], clipped]
    future_l = l[day_idx[:, None], clipped]
    future_close = c[day_idx[:, None], clipped]
    valid = valid & (future_h != 0) & (future_l != 0)
    sl_rows = []
    tp_rows = []
    directions = [row["direction"] for row in rows]
    for config in configs:
        if config["mode"] == "atr":
            sl_rows.append([
                row["entry"] - config["sl"] * row["atr"] if row["direction"] == 1
                else row["entry"] + config["sl"] * row["atr"]
                for row in rows
            ])
            tp_rows.append([
                row["entry"] + config["tp"] * row["atr"] if row["direction"] == 1
                else row["entry"] - config["tp"] * row["atr"]
                for row in rows
            ])
        else:
            sl_rows.append([
                (row["entry_ob_lo"] if direction_i == 1 else row["entry_ob_hi"])
                if config["mode"] == "ob_w5" else
                ((row["same_tf_ob_lo"] if direction_i == 1 else row["same_tf_ob_hi"])
                 if row["same_tf_ob_lo"] is not None
                 else (row["entry_ob_lo"] if direction_i == 1 else row["entry_ob_hi"]))
                for row, direction_i in zip(rows, directions)
            ])
            tp_rows.append([row["baseline_tp"] for row in rows])

    sl_matrix = torch.tensor(sl_rows, dtype=torch.float32, device=device)
    tp_matrix = torch.tensor(tp_rows, dtype=torch.float32, device=device)
    direction_matrix = direction[None, :, None] == 1
    sl_hits = torch.where(
        direction_matrix,
        future_l[None, :, :] <= sl_matrix[:, :, None],
        future_h[None, :, :] >= sl_matrix[:, :, None],
    ) & valid[None, :, :]
    tp_hits = torch.where(
        direction_matrix,
        future_h[None, :, :] >= tp_matrix[:, :, None],
        future_l[None, :, :] <= tp_matrix[:, :, None],
    ) & valid[None, :, :]
    sl_any = sl_hits.any(dim=2)
    tp_any = tp_hits.any(dim=2)
    no_hit_index = torch.full_like(entry_idx, SESSION_BARS)
    sl_index = torch.where(sl_any, sl_hits.to(torch.int8).argmax(dim=2), no_hit_index[None, :])
    tp_index = torch.where(tp_any, tp_hits.to(torch.int8).argmax(dim=2), no_hit_index[None, :])
    use_sl = sl_index <= tp_index
    no_hit = (~sl_any) & (~tp_any)
    exit_offset = torch.where(no_hit, no_hit_index[None, :], torch.where(use_sl, sl_index, tp_index))
    last_valid_pos = torch.where(valid, positions, torch.full_like(positions, -1)).max(dim=1).values
    last_valid_pos = torch.where(last_valid_pos < 0, entry_idx, last_valid_pos)
    absolute_exit = torch.where(
        no_hit,
        last_valid_pos[None, :],
        (entry_idx[None, :] + 1 + exit_offset).clamp(max=SESSION_BARS - 1),
    )
    exit_px = c[day_idx[None, :], absolute_exit]
    points = torch.where(direction[None, :] == 1, exit_px - entries[None, :], entries[None, :] - exit_px)
    points_net = points - 2 * SLIPPAGE_PTS
    if device.type == "cuda":
        torch.cuda.synchronize()
    results = []
    for index, config in enumerate(configs):
        results.append({
            "config": config,
            "exit_indices": absolute_exit[index].detach().cpu().numpy(),
            "exit_prices": exit_px[index].detach().cpu().numpy(),
            "reasons": np.where(
                no_hit[index].detach().cpu().numpy(),
                "EOD",
                np.where(use_sl[index].detach().cpu().numpy(), "SL", "TP"),
            ),
            "points": points_net[index].detach().cpu().numpy(),
            "device": str(device),
        })
    return results


def with_baseline_tp(rows, baseline_trades):
    for row, trade in zip(rows, baseline_trades):
        row["baseline_tp"] = float(trade["tp"])
    return rows


def verify_gpu_parity(spot, options, days, workers):
    trades = run_engine(spot, options, days, 0.5, 3.0, 60.0, "ob_w5", None, workers)
    high, low, close, day_index = pack_spot(spot, days)
    rows = with_baseline_tp(fixed_entry_arrays(trades, day_index), trades)
    configs = [
        {"label": "smoke_atr_3_3", "mode": "atr", "sl": 3.0, "tp": 3.0},
        {"label": "smoke_atr_5_5", "mode": "atr", "sl": 5.0, "tp": 5.0},
    ]
    gpu_results = torch_exits(rows, configs, high, low, close)
    failures = []
    for result in gpu_results:
        cpu = cpu_exits(rows, result["config"], spot, options, day_index, high, low, close)
        for i, trade in enumerate(cpu):
            gpu_exit = 555 + int(result["exit_indices"][i])
            gpu_reason = str(result["reasons"][i])
            if (trade["exit_min"] != gpu_exit
                    or trade["exit_reason"] != gpu_reason
                    or abs(trade["pts_net"] - float(result["points"][i])) > 0.01):
                failures.append((result["config"]["label"], i, trade, gpu_exit, gpu_reason))
    if failures:
        raise RuntimeError(f"GPU smoke parity failed: {failures[:3]}")
    print(
        f"SMOKE GPU PARITY passed: {len(configs)} exit configurations "
        f"checked on {len(trades)} fixed entries ({gpu_results[0]['device']})",
        flush=True,
    )


def fixed_walk_forward(all_results):
    folds = []
    for year in ("2023", "2024", "2025", "2026"):
        candidates = []
        for label, result in all_results.items():
            train = [t for t in result["trades"] if t["date"][:4] < year]
            candidates.append((summarize(train), label))
        selected_stats, selected_label = max(candidates, key=lambda item: (item[0]["pts"], item[0]["rs"]))
        oos = [t for t in all_results[selected_label]["trades"] if t["date"][:4] == year]
        folds.append({
            "oos_year": year,
            "selected": selected_label,
            "train_stats": selected_stats,
            "oos_stats": summarize(oos),
            "oos_trades": oos,
        })
    stitched = [t for fold in folds for t in fold["oos_trades"]]
    return {"folds": folds, "stitched_oos": summarize(stitched), "trades": stitched}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("nonwf", "wf", "both"), default="both")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--entry-cache", default="artifacts/ew_ob/fixed_entry_stream.json")
    parser.add_argument("--rebuild-entry-cache", action="store_true")
    parser.add_argument("--output", default="artifacts/ew_ob/fixed_entry_benchmark.json")
    args = parser.parse_args()
    started = time.time()
    spot, options, days = load_data()
    anchor_days, _ = smoke_gate(spot, options, days, args.workers)
    verify_gpu_parity(spot, options, anchor_days, args.workers)
    if args.smoke_only:
        return

    cache_path = Path(args.entry_cache)
    if cache_path.exists() and not args.rebuild_entry_cache:
        print(f"PHASE CPU_ENTRY_STREAM cache_load path={cache_path}", flush=True)
        baseline_trades = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"PHASE CPU_ENTRY_STREAM cache_entries={len(baseline_trades)}", flush=True)
    else:
        print("PHASE CPU_ENTRY_STREAM start", flush=True)
        baseline_trades = run_engine(
            spot, options, days, 0.5, 3.0, 60.0, "ob_w5", None,
            args.workers, progress=True,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(baseline_trades), encoding="utf-8")
        print(f"PHASE CPU_ENTRY_STREAM complete entries={len(baseline_trades)}", flush=True)
    high, low, close, day_index = pack_spot(spot, days)
    rows = with_baseline_tp(fixed_entry_arrays(baseline_trades, day_index), baseline_trades)
    configs = build_configs()
    print("PHASE GPU_EXIT_BATCH start", flush=True)
    gpu_results = torch_exits(rows, configs, high, low, close)
    print(f"PHASE GPU_EXIT_BATCH complete device={gpu_results[0]['device'] if gpu_results else 'none'}", flush=True)
    resolver = make_option_resolver(options)
    parity_failures = []
    all_results = {}
    for result in gpu_results:
        config = result["config"]
        cpu_trades = cpu_exits(rows, config, spot, options, day_index, high, low, close)
        gpu_exit = result["exit_indices"]
        gpu_reason = result["reasons"]
        for index, cpu_trade in enumerate(cpu_trades):
            expected_reason = str(gpu_reason[index])
            if (cpu_trade["exit_min"] != 555 + int(gpu_exit[index])
                    or cpu_trade["exit_reason"] != expected_reason
                    or abs(cpu_trade["pts_net"] - float(result["points"][index])) > 0.01):
                parity_failures.append({
                    "config": config["label"],
                    "index": index,
                    "cpu": (cpu_trade["exit_min"], cpu_trade["exit_reason"], cpu_trade["pts_net"]),
                    "gpu": (555 + int(gpu_exit[index]), expected_reason, float(result["points"][index])),
                })
        trades = []
        for row, exit_index, exit_px, reason in zip(
                rows, result["exit_indices"], result["exit_prices"], result["reasons"]):
            exit_min = 555 + int(exit_index)
            exit_prem = resolver(row["date"], row["side"], exit_min, float(exit_px), row["strike"])
            if exit_prem is None:
                exit_prem = row["entry_prem"]
            points = (float(exit_px) - row["entry"]) if row["direction"] == 1 else (row["entry"] - float(exit_px))
            fee = trade_cost(row["entry_prem"], exit_prem)
            trades.append({
                "date": row["date"], "entry_min": row["entry_min"], "exit_min": exit_min,
                "timeframe": row["timeframe"], "side": row["side"], "direction": "bull" if row["direction"] == 1 else "bear",
                "entry": row["entry"], "exit": float(exit_px), "exit_reason": str(reason),
                "pts_net": round(points - 2 * SLIPPAGE_PTS, 2),
                "rs_net": round((exit_prem - row["entry_prem"]) * LOT_SIZE - fee, 2),
            })
        all_results[config_label(config)] = {"config": config, "stats": summarize(trades), "trades": trades}
        print(config_label(config), all_results[config_label(config)]["stats"], flush=True)

    if parity_failures:
        raise RuntimeError(f"GPU/CPU fixed-entry parity failed: {parity_failures[:5]}")

    result = {
        "causal_live_parity": True,
        "fixed_entries": len(rows),
        "device": gpu_results[0]["device"] if gpu_results else "none",
        "non_walk_forward": all_results,
        "walk_forward": fixed_walk_forward(all_results),
        "parity": {"passed": True, "checked": len(gpu_results) * len(rows)},
        "seconds": round(time.time() - started, 2),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
