"""Full 7-Year Non-Walk-Forward (NWF) Master Comparison (2020–2026).

Compares across all 1,588 trading days:
  1. Baseline (No HTF Filter, Trailing Stop 0.75x/0.40x ATR)
  2. Rule 1 Only (TradingView 15m HTF Filter, Trailing Stop 0.75x/0.40x ATR)
  3. Rule 2 Only (No HTF Filter, 2:1 Positive R:R -7/+14 pts, BE +5 pts)
  4. Combined (TradingView 15m HTF Filter + 2:1 Positive R:R)
  5. Pocket Money Hybrid Scalp (TV 15m HTF Filter + SL=-7.0 pts, TP=+10.0 pts, BE=+5.0 pts)

Outputs:
  - 7-Year Overall Master Table
  - Detailed Year-by-Year Breakdown (2020, 2021, 2022, 2023, 2024, 2025, 2026)
  - Annual Max Drawdowns & Consistency Metrics
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_5y_optimized import SYM_RE, latest_spot, option_files
from flattrade_bot.indicators.patterns import Candle
from artifacts.f6_hybrid.pocket_money_backtest import (
    build_index_filter, filter_allows,
)
from artifacts.f6_hybrid.test_super_only_aug import extend_with_august, ParamStoch, IncrementalATR, bslice, to_hhmm, LOT_SIZE, FEE
from artifacts.f6_hybrid.compare_rules_1_and_2 import load_full_ohlc_spot, DualTracker, simulate_day_rule_engine
import grid_optimize_f6_atr as grid


def run_full_7y_nwf():
    print("=" * 145)
    print("FULL 7-YEAR NON-WALK-FORWARD (NWF) MASTER COMPARISON (2020–2026)")
    print("Dataset: 1,588 Trading Days | Nifty 50 Options 1-Minute Bars | Flat Rs 40.00 / Trade Fee")
    print("=" * 145)

    spot_all = load_full_ohlc_spot()
    opt_map = option_files("2020-01-01", "2026-05-05")
    opt_map, spot_all = extend_with_august(opt_map, spot_all)
    grid.GLOBAL_SPOT = spot_all
    all_cal = sorted(opt_map.keys())
    cal_idx = {d: i for i, d in enumerate(all_cal)}
    N_DAYS = len(all_cal)

    years = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]

    configs = [
        {
            "id": "baseline",
            "name": "1. BASELINE (No HTF, Trail=0.75x/0.40x ATR)",
            "use_htf": False, "exit_mode": "trailing",
            "sl_mult": 1.50, "tp_mult": 3.00, "trail_trig": 0.75, "trail_dist": 0.40,
            "fixed_sl_pts": 7.0, "fixed_tp_pts": 14.0, "be_pts": 5.0, "max_tr": 3,
        },
        {
            "id": "rule_1",
            "name": "2. RULE 1 ONLY (TV 15m HTF Filter, Trail=0.75x/0.40x ATR)",
            "use_htf": True, "exit_mode": "trailing",
            "sl_mult": 1.50, "tp_mult": 3.00, "trail_trig": 0.75, "trail_dist": 0.40,
            "fixed_sl_pts": 7.0, "fixed_tp_pts": 14.0, "be_pts": 5.0, "max_tr": 3,
        },
        {
            "id": "rule_2",
            "name": "3. RULE 2 ONLY (No HTF, 2:1 R:R -7/+14 pts, BE +5 pts)",
            "use_htf": False, "exit_mode": "positive_rr",
            "sl_mult": 1.50, "tp_mult": 3.00, "trail_trig": 0.75, "trail_dist": 0.40,
            "fixed_sl_pts": 7.0, "fixed_tp_pts": 14.0, "be_pts": 5.0, "max_tr": 3,
        },
        {
            "id": "combined",
            "name": "4. COMBINED (TV 15m HTF Filter + 2:1 R:R -7/+14 pts)",
            "use_htf": True, "exit_mode": "positive_rr",
            "sl_mult": 1.50, "tp_mult": 3.00, "trail_trig": 0.75, "trail_dist": 0.40,
            "fixed_sl_pts": 7.0, "fixed_tp_pts": 14.0, "be_pts": 5.0, "max_tr": 3,
        },
        {
            "id": "pocket_money_scalp",
            "name": "5. POCKET MONEY SCALP (TV 15m HTF + SL=-7.0 / TP=+10.0 pts)",
            "use_htf": True, "exit_mode": "positive_rr",
            "sl_mult": 1.50, "tp_mult": 3.00, "trail_trig": 0.75, "trail_dist": 0.40,
            "fixed_sl_pts": 7.0, "fixed_tp_pts": 10.0, "be_pts": 5.0, "max_tr": 3,
        },
    ]

    all_config_results = {}

    for cfg in configs:
        t0 = time.time()
        print(f"\nEvaluating: {cfg['name']} across {N_DAYS} days...", flush=True)

        trades_all = []
        for d in all_cal:
            trs = simulate_day_rule_engine(
                d, opt_map, all_cal, cal_idx, spot_all,
                use_htf=cfg["use_htf"],
                exit_mode=cfg["exit_mode"],
                sl_mult=cfg["sl_mult"], tp_mult=cfg["tp_mult"],
                trail_trig=cfg["trail_trig"], trail_dist=cfg["trail_dist"],
                fixed_sl_pts=cfg["fixed_sl_pts"], fixed_tp_pts=cfg["fixed_tp_pts"],
                be_trigger_pts=cfg["be_pts"],
                max_trades_day=cfg["max_tr"],
            )
            trades_all.extend(trs)

        n_t = len(trades_all)
        wins = [t for t in trades_all if t["rs_net"] > 0]
        losses = [t for t in trades_all if t["rs_net"] <= 0]
        net_rs = sum(t["rs_net"] for t in trades_all)
        net_pts = sum(t["pts"] for t in trades_all)
        wr = len(wins) / n_t * 100 if n_t > 0 else 0.0
        win_tot = sum(t["rs_net"] for t in wins)
        loss_tot = abs(sum(t["rs_net"] for t in losses))
        pf = win_tot / loss_tot if loss_tot > 0 else (99.0 if win_tot > 0 else 0.0)

        eq = np.cumsum([t["rs_net"] for t in trades_all]) if trades_all else np.array([0.0])
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max(peak - eq)) if len(eq) > 0 else 0.0
        calmar = net_rs / max_dd if max_dd > 0 else 0.0

        # Yearly breakdown
        df_tr = pd.DataFrame(trades_all)
        yearly_map = {}
        if not df_tr.empty:
            df_tr["year"] = df_tr["date"].str[:4]
            for y in years:
                sub = df_tr[df_tr["year"] == y]
                if not sub.empty:
                    y_w = sub[sub["rs_net"] > 0]
                    y_l = sub[sub["rs_net"] <= 0]
                    y_net = sub["rs_net"].sum()
                    y_pts = sub["pts"].sum()
                    y_wr = len(y_w) / len(sub) * 100.0
                    y_pf = y_w["rs_net"].sum() / abs(y_l["rs_net"].sum()) if abs(y_l["rs_net"].sum()) > 0 else 99.0
                    y_eq = np.cumsum(sub["rs_net"].to_numpy())
                    y_dd = float(np.max(np.maximum.accumulate(y_eq) - y_eq))
                    yearly_map[y] = {
                        "trades": len(sub), "win_rate": round(y_wr, 1),
                        "net_points": round(y_pts, 2), "net_rs": round(y_net, 2),
                        "profit_factor": round(y_pf, 3), "max_dd": round(y_dd, 2),
                    }
                else:
                    yearly_map[y] = {"trades": 0, "win_rate": 0.0, "net_points": 0.0, "net_rs": 0.0, "profit_factor": 0.0, "max_dd": 0.0}

        all_config_results[cfg["id"]] = {
            "name": cfg["name"],
            "trades": n_t,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(wr, 2),
            "net_points": round(net_pts, 2),
            "net_rs": round(net_rs, 2),
            "profit_factor": round(pf, 3),
            "max_drawdown": round(max_dd, 2),
            "calmar_ratio": round(calmar, 3),
            "avg_trade_rs": round(net_rs / n_t, 2) if n_t > 0 else 0.0,
            "yearly": yearly_map,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        print(f"  Done in {time.time()-t0:.2f}s: {n_t} trades | Win Rate: {wr:.1f}% | Net Rs: Rs {net_rs:+,.2f} | PF: {pf:.3f} | Max DD: Rs {max_dd:,.2f}", flush=True)

    # 1. Print Master Overview Table
    print("\n" + "=" * 145)
    print(">>> 7-YEAR MASTER OVERVIEW TABLE (2020–2026):")
    print("=" * 145)
    print(f"{'Rank':4s} | {'Configuration':58s} | {'Trades':7s} | {'Win Rate':9s} | {'Net Points':11s} | {'Net Rs':15s} | {'PF':6s} | {'Max DD':12s} | {'Calmar':7s}")
    print("-" * 145)
    for r, (cid, res) in enumerate(all_config_results.items(), 1):
        print(f"{r:4d} | {res['name']:58s} | {res['trades']:7d} | {res['win_rate']:7.1f}% | {res['net_points']:+10.2f} | Rs {res['net_rs']:+12.2f} | {res['profit_factor']:6.3f} | Rs {res['max_drawdown']:9.2f} | {res['calmar_ratio']:7.3f}")

    # 2. Print Year-by-Year Table for Each Configuration
    print("\n" + "=" * 145)
    print(">>> YEAR-BY-YEAR REALIZED P&L (Rs) COMPARISON (2020–2026):")
    print("=" * 145)
    header = f"{'Configuration':58s} | " + " | ".join(f"{y:10s}" for y in years) + f" | {'Total (Rs)':14s}"
    print(header)
    print("-" * 155)
    for cid, res in all_config_results.items():
        y_strs = []
        for y in years:
            ystat = res["yearly"].get(y, {})
            val = ystat.get("net_rs", 0.0)
            y_strs.append(f"Rs {val:+7.0f}")
        print(f"{res['name']:58s} | " + " | ".join(y_strs) + f" | Rs {res['net_rs']:+11.2f}")

    # 3. Print Year-by-Year Win Rate & Trade Count
    print("\n" + "=" * 145)
    print(">>> YEAR-BY-YEAR WIN RATE (%) COMPARISON:")
    print("=" * 145)
    header = f"{'Configuration':58s} | " + " | ".join(f"{y:10s}" for y in years) + f" | {'Overall WR':10s}"
    print(header)
    print("-" * 155)
    for cid, res in all_config_results.items():
        y_strs = []
        for y in years:
            ystat = res["yearly"].get(y, {})
            wr = ystat.get("win_rate", 0.0)
            tr = ystat.get("trades", 0)
            y_strs.append(f"{wr:5.1f}% ({tr:3d})")
        print(f"{res['name']:58s} | " + " | ".join(y_strs) + f" | {res['win_rate']:7.1f}%")

    # Save to JSON
    out_file = ROOT / "artifacts" / "f6_hybrid" / "full_7y_nwf_all_rules_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_config_results, indent=2, default=float), encoding="utf-8")
    print("\n" + "=" * 145)
    print(f"[Saved 7-Year NWF JSON Ledger]: {out_file}")
    print("=" * 145)


if __name__ == "__main__":
    run_full_7y_nwf()
