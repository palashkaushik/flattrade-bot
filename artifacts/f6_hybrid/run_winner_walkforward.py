"""Multi-Fold Walk-Forward & Drawdown Deep Dive for Winner Configurations.

Evaluates:
  1. 4-Fold Rolling Walk-Forward (2023, 2024, 2025, 2026 OOS)
  2. Anchored In-Sample (2020-2023) vs Out-of-Sample (2024-2026)
  3. Detailed Year-by-Year Max Drawdown, Drawdown Duration, and Recovery
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.f6_hybrid.optimus_vix_atr_gpu import (
    load_gpu_all_data,
    compute_quad_stochastics_gpu,
    compute_atr_gpu,
    LOT_SIZE,
    FEE,
    BASE_SESSION_START,
    BASE_SESSION_END,
)

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_h, d_l, d_c, d_o, d_vix, days, d_is_mask, d_oos_mask = load_gpu_all_data()
N_DAYS = len(days)

s1, s2, s3, s4 = compute_quad_stochastics_gpu(d_h, d_l, d_c)
d_atr = compute_atr_gpu(d_h, d_l, d_c, period=14)

prev_s1 = F.pad(s1[:, :-1], (1, 0), mode="replicate")
s1_turn_up = (s1 > prev_s1)
super_setup = (s1 <= 20.5) & (s2 <= 20.5) & (s3 <= 20.5) & (s4 <= 20.5) & s1_turn_up
flag_setup = (s4 >= 79.5) & (s1 <= 20.5) & s1_turn_up
entries_mask = super_setup | flag_setup


@torch.inference_mode()
def evaluate_winner_detailed(
    entries_mask: torch.Tensor,
    sl_mult: float,
    tp_mult: float,
    trail_trigger_atr: float | None,
    trail_dist_atr: float,
    gamma: float = 0.0,
    day_indices_subset: np.ndarray | None = None,
):
    coords = torch.nonzero(entries_mask, as_tuple=False)
    d_idx = coords[:, 0]
    b_idx = coords[:, 1]
    ep = d_c[d_idx, b_idx]
    base_atr = d_atr[d_idx, b_idx].clamp(min=5.0, max=25.0)
    trade_vix = d_vix[d_idx, b_idx].clamp(min=8.0, max=80.0)

    max_future = 345 - BASE_SESSION_START - 1
    col_offsets = torch.arange(max_future, device=device).unsqueeze(0)
    col_idx = (b_idx + 1).unsqueeze(1) + col_offsets
    valid = (col_idx < BASE_SESSION_END) & (col_idx < 375)
    col_safe = col_idx.clamp(max=374)

    d_exp = d_idx.unsqueeze(1).expand(-1, max_future)
    fut_h = d_h[d_exp, col_safe]
    fut_l = d_l[d_exp, col_safe]

    fut_h_m = torch.where(valid, fut_h, torch.tensor(-1e9, device=device))
    fut_l_m = torch.where(valid, fut_l, torch.tensor(1e9, device=device))
    eod_px = d_c[d_idx, BASE_SESSION_END - 1]

    fut_h_clean = torch.where(valid, fut_h, ep.unsqueeze(1))
    running_peaks = torch.cummax(fut_h_clean, dim=1).values

    ep_exp = ep.unsqueeze(1)

    vix_scale = torch.pow(trade_vix / 15.0, gamma).clamp(min=0.6, max=2.0)
    eff_atr = base_atr * vix_scale
    sl_d = (sl_mult * eff_atr).clamp(min=5.0, max=30.0)
    tp_d = (tp_mult * eff_atr).clamp(min=8.0, max=60.0)

    init_sl = ep_exp - sl_d.unsqueeze(1)
    tp_barrier = ep_exp + tp_d.unsqueeze(1)

    if trail_trigger_atr is not None:
        trig_gain = (trail_trigger_atr * eff_atr).unsqueeze(1)
        trail_d = (trail_dist_atr * eff_atr).unsqueeze(1)
        gains = running_peaks - ep_exp
        is_trailing = gains >= trig_gain
        trailing_sl_level = running_peaks - trail_d
        dyn_sl_barrier = torch.where(is_trailing, torch.maximum(init_sl, trailing_sl_level), init_sl)
    else:
        dyn_sl_barrier = init_sl.expand(-1, max_future)

    hit_sl = (fut_l_m <= dyn_sl_barrier)
    hit_tp = (fut_h_m >= tp_barrier)

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)

    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)

    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    exit_sl_px = dyn_sl_barrier.gather(1, sl_first.clamp(max=max_future-1).unsqueeze(1)).squeeze(1)
    exit_tp_px = tp_barrier.squeeze(1)
    exit_px = torch.where(sl_exits, exit_sl_px, torch.where(tp_exits, exit_tp_px, eod_px))

    pts = (exit_px - ep) * 0.50
    rs_net = pts * LOT_SIZE - FEE

    rs_cpu = rs_net.cpu().numpy()
    pts_cpu = pts.cpu().numpy()
    d_idx_cpu = d_idx.cpu().numpy()
    b_idx_cpu = b_idx.cpu().numpy()

    # Filter subset if requested
    if day_indices_subset is not None:
        subset_set = set(day_indices_subset)
        mask = np.isin(d_idx_cpu, list(subset_set))
        rs_cpu = rs_cpu[mask]
        pts_cpu = pts_cpu[mask]
        d_idx_cpu = d_idx_cpu[mask]
        b_idx_cpu = b_idx_cpu[mask]

    order = np.lexsort((b_idx_cpu, d_idx_cpu))
    rs_sorted = rs_cpu[order]
    pts_sorted = pts_cpu[order]
    days_sorted = d_idx_cpu[order]

    n_t = len(rs_sorted)
    if n_t == 0:
        return {
            "trades": 0, "win_rate": 0.0, "net_points": 0.0, "net_rs": 0.0,
            "profit_factor": 0.0, "max_drawdown_rs": 0.0, "calmar_ratio": 0.0,
            "max_dd_duration_trades": 0,
        }

    wins = [r for r in rs_sorted if r > 0]
    losses = [r for r in rs_sorted if r <= 0]
    win_tot = sum(wins)
    loss_tot = abs(sum(losses))
    net_rs_tot = float(sum(rs_sorted))
    net_pts_tot = float(sum(pts_sorted))
    wr = len(wins) / n_t * 100.0
    pf = win_tot / loss_tot if loss_tot > 0 else (99.0 if win_tot > 0 else 0.0)

    # Detailed Drawdown & Peak-to-Trough Duration
    eq = np.cumsum(rs_sorted)
    peak = np.maximum.accumulate(eq)
    dd_curve = peak - eq
    max_dd = float(np.max(dd_curve))

    # Max Drawdown Duration (in trades)
    cur_dd_len = 0
    max_dd_len = 0
    for dd in dd_curve:
        if dd > 0:
            cur_dd_len += 1
            if cur_dd_len > max_dd_len:
                max_dd_len = cur_dd_len
        else:
            cur_dd_len = 0

    calmar = net_rs_tot / max_dd if max_dd > 0 else 0.0

    return {
        "trades": n_t,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr, 2),
        "net_points": round(net_pts_tot, 2),
        "net_rs": round(net_rs_tot, 2),
        "profit_factor": round(pf, 3),
        "max_drawdown_rs": round(max_dd, 2),
        "max_dd_duration_trades": max_dd_len,
        "calmar_ratio": round(calmar, 3),
        "fees_rs": round(n_t * FEE, 2),
    }


def main():
    print("=" * 135)
    print("WALK-FORWARD VALIDATION & DETAILED DRAWDOWN AUDIT (2020–2026)")
    print("=" * 135)

    winners = [
        {
            "name": "Champion Low-Drawdown (Trail=0.75x/0.50x, SL=1.5x, TP=3.0x)",
            "sl_mult": 1.5, "tp_mult": 3.0, "trail_trigger_atr": 0.75, "trail_dist_atr": 0.5, "gamma": 0.0,
        },
        {
            "name": "Champion Max-Profit (Trail=1.25x/0.50x, SL=1.5x, TP=3.5x)",
            "sl_mult": 1.5, "tp_mult": 3.5, "trail_trigger_atr": 1.25, "trail_dist_atr": 0.5, "gamma": 0.0,
        },
        {
            "name": "Champion VIX-Enhanced (Power g=1.0, SL=3.0x, TP=6.0x)",
            "sl_mult": 3.0, "tp_mult": 6.0, "trail_trigger_atr": None, "trail_dist_atr": 0.5, "gamma": 1.0,
        },
    ]

    years = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
    folds = [
        {"name": "Fold 1", "is_years": ["2020", "2021", "2022"], "oos_year": "2023"},
        {"name": "Fold 2", "is_years": ["2021", "2022", "2023"], "oos_year": "2024"},
        {"name": "Fold 3", "is_years": ["2022", "2023", "2024"], "oos_year": "2025"},
        {"name": "Fold 4", "is_years": ["2023", "2024", "2025"], "oos_year": "2026"},
    ]

    all_reports = {}

    for w in winners:
        print(f"\n{'#'*35} STRATEGY: {w['name'].upper()} {'#'*35}")

        # 1. Full 7-Year Overall Stats
        overall = evaluate_winner_detailed(entries_mask, w["sl_mult"], w["tp_mult"], w["trail_trigger_atr"], w["trail_dist_atr"], w["gamma"])
        print(f"\n>>> 7-YEAR NON-WALK-FORWARD OVERVIEW (2020–2026):")
        print(f"  Total Trades: {overall['trades']:,} | Win Rate: {overall['win_rate']}% | Profit Factor: {overall['profit_factor']}")
        print(f"  Net Points: {overall['net_points']:+,.2f} | Net Realized PnL: Rs {overall['net_rs']:+,.2f}")
        print(f"  Max Drawdown: Rs {overall['max_drawdown_rs']:,.2f} | Max DD Duration: {overall['max_dd_duration_trades']} trades | Calmar Ratio: {overall['calmar_ratio']}")

        # 2. Year-by-Year Breakdown with Max Drawdown per Year
        print(f"\n>>> YEAR-BY-YEAR DETAILED METRICS & LOCAL MAX DRAWDOWN:")
        print(f"{'Year':6s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Yearly Max DD':14s} | {'DD Duration':11s}")
        print("-" * 105)
        yearly_stats = {}
        for y in years:
            y_indices = [i for i, d in enumerate(days) if d.startswith(y)]
            st = evaluate_winner_detailed(entries_mask, w["sl_mult"], w["tp_mult"], w["trail_trigger_atr"], w["trail_dist_atr"], w["gamma"], day_indices_subset=y_indices)
            yearly_stats[y] = st
            print(f"{y:6s} | {st['trades']:7d} | {st['win_rate']:7.1f}% | {st['net_points']:+10.2f} | Rs {st['net_rs']:+12.2f} | {st['profit_factor']:6.3f} | Rs {st['max_drawdown_rs']:11.2f} | {st['max_dd_duration_trades']:5d} trades")

        # 3. 4-Fold Rolling Walk-Forward Out-Of-Sample Validation
        print(f"\n>>> 4-FOLD ROLLING WALK-FORWARD VALIDATION (OOS YEARS: 2023, 2024, 2025, 2026):")
        print(f"{'Fold':8s} | {'IS Period':12s} | {'IS PnL':14s} | {'IS PF':6s} | {'OOS Year':9s} | {'OOS Trades':10s} | {'OOS WR':7s} | {'OOS PnL':14s} | {'OOS PF':7s} | {'OOS Max DD':12s} | {'WFE':6s}")
        print("-" * 135)
        fold_results = []
        oos_all_trades_indices = []

        for f in folds:
            is_indices = [i for i, d in enumerate(days) if any(d.startswith(y) for y in f["is_years"])]
            oos_indices = [i for i, d in enumerate(days) if d.startswith(f["oos_year"])]
            oos_all_trades_indices.extend(oos_indices)

            is_st = evaluate_winner_detailed(entries_mask, w["sl_mult"], w["tp_mult"], w["trail_trigger_atr"], w["trail_dist_atr"], w["gamma"], day_indices_subset=is_indices)
            oos_st = evaluate_winner_detailed(entries_mask, w["sl_mult"], w["tp_mult"], w["trail_trigger_atr"], w["trail_dist_atr"], w["gamma"], day_indices_subset=oos_indices)

            # Walk-Forward Efficiency (Annualized OOS / Annualized IS)
            wfe = (oos_st["net_rs"] / 1.0) / (is_st["net_rs"] / 3.0) if is_st["net_rs"] > 0 else 0.0
            fold_results.append({"fold": f["name"], "is": is_st, "oos": oos_st, "wfe": wfe})
            is_yr_str = f"{f['is_years'][0]}-{f['is_years'][-1]}"
            print(f"{f['name']:8s} | {is_yr_str:12s} | Rs {is_st['net_rs']:+11.2f} | {is_st['profit_factor']:6.2f} | {f['oos_year']:9s} | {oos_st['trades']:10d} | {oos_st['win_rate']:6.1f}% | Rs {oos_st['net_rs']:+11.2f} | {oos_st['profit_factor']:7.2f} | Rs {oos_st['max_drawdown_rs']:9.2f} | {wfe:6.2f}")

        # 4. Stitched 4-Year Out-Of-Sample (2023–2026 Continuous)
        stitched_oos = evaluate_winner_detailed(entries_mask, w["sl_mult"], w["tp_mult"], w["trail_trigger_atr"], w["trail_dist_atr"], w["gamma"], day_indices_subset=oos_all_trades_indices)
        print(f"\n>>> STITCHED OUT-OF-SAMPLE TOTAL (2023–2026 Continuous Unseen Market):")
        print(f"  OOS Trades: {stitched_oos['trades']:,} | OOS Win Rate: {stitched_oos['win_rate']}% | OOS Profit Factor: {stitched_oos['profit_factor']}")
        print(f"  OOS Net PnL: Rs {stitched_oos['net_rs']:+,.2f} | OOS Max Drawdown: Rs {stitched_oos['max_drawdown_rs']:,.2f} | OOS Calmar: {stitched_oos['calmar_ratio']}")

        all_reports[w["name"]] = {
            "overall": overall,
            "yearly": yearly_stats,
            "folds": fold_results,
            "stitched_oos": stitched_oos,
        }

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "winner_walkforward_drawdown_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_reports, indent=2, default=float), encoding="utf-8")
    print(f"\n[Saved Walk-Forward JSON Ledger]: {out_file}")
    print("=" * 135)


if __name__ == "__main__":
    main()
