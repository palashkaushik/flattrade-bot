"""Optimus GPU sweep for B08 — Opening-Range Breakout (book translation).

Runs the EXACT 3D-fused GPU engine (optimized_gpu_backtest.evaluate_breakout_all).
Full 7-year (2020-2026) + IS/OOS walk-forward, honest robustness reporting.

Usage:
    $env:PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
    python optimus_breakout_sweep.py
"""
import csv
import time
import itertools

import torch

import optimized_gpu_backtest as m

BATCH = 100

# Pruned, book-faithful grid (≈18k combos). Fixed daily caps (not swept).
GRID = {
    "open_candles": [2, 3],
    "range_mode": ["body", "wick"],
    "break_mode": ["close", "high_low"],
    "break_buf": [0, 2],
    "entry_until": [6, 10],          # in 5-min candles -> bars 30 / 50 (~10:00 / 10:40)
    "otm_strikes": [2, 3],
    "sl_mode": ["opposite", "cpr_respect"],
    "sl_buf": [0, 3],
    "target_mode": ["level_ride", "pct"],
    "ride_frac": [0.6, 0.7, 0.75],
    "pct_target": [0.06, 0.10],
    "direction": ["both", "bull", "bear"],
    "allow_friday": [False],
}


def gen_combos():
    ks = list(GRID.keys())
    for vals in itertools.product(*(GRID[k] for k in ks)):
        p = {k: v for k, v in zip(ks, vals)}
        p["daily_loss_pts"] = 50          # ₹3,250/day cap (rarely hit: ≤1 trade/day)
        p["daily_profit_pts"] = 120       # ₹7,800/day cap
        yield p


def main():
    combos = list(gen_combos())
    print(f"[B08 sweep] {len(combos)} combos | GPU {m.device} | N={m.N_DAYS} T={m.T_BARS}", flush=True)
    t0 = time.time()

    rows = []
    nb = (len(combos) + BATCH - 1) // BATCH
    for bi in range(nb):
        batch = combos[bi * BATCH:(bi + 1) * BATCH]
        out = m.evaluate_breakout_all(batch)
        for j, p in enumerate(batch):
            rfull, ris, roos = out["full"][j], out["is"][j], out["oos"][j]
            rows.append({
                **p,
                "full_trades": rfull["trades"], "full_wr": round(rfull["win_rate"], 2),
                "full_net": round(rfull["net_rs"], 1), "full_pf": round(rfull["pf"], 3),
                "full_dd": round(rfull["max_dd"], 1),
                "is_net": round(ris["net_rs"], 1), "is_pf": round(ris["pf"], 3),
                "oos_trades": roos["trades"], "oos_wr": round(roos["win_rate"], 2),
                "oos_net": round(roos["net_rs"], 1), "oos_pf": round(roos["pf"], 3),
                "oos_dd": round(roos["max_dd"], 1),
            })
        if (bi + 1) % 20 == 0 or bi == nb - 1:
            print(f"  batch {bi+1}/{nb}  ({time.time()-t0:.1f}s)", flush=True)

    # Robustness over the whole pool (honest, unfiltered).
    pooled = [r for r in rows if r["full_trades"] >= 30]
    oos_pos = [r for r in pooled if r["oos_net"] > 0]
    print(f"\nPool: {len(pooled)} combos with >=30 trades | "
          f"OOS-positive: {len(oos_pos)} ({100*len(oos_pos)/max(1,len(pooled)):.1f}%)", flush=True)

    # Top by full net, then show IS/OOS consistency.
    top = sorted(pooled, key=lambda r: r["full_net"], reverse=True)[:25]
    print("\nTOP 25 by FULL 7y net (OTM-capture adjusted):")
    print(f"{'net':>10} {'PF':>5} {'WR':>5} {'trades':>6} | {'IS_net':>9} {'OOS_net':>9} {'OOS_PF':>6} | params")
    for r in top:
        wfe = r["oos_net"] / r["full_net"] if r["full_net"] else 0
        print(f"{r['full_net']:>10.0f} {r['full_pf']:>5.2f} {r['full_wr']:>5.1f} {r['full_trades']:>6} | "
              f"{r['is_net']:>9.0f} {r['oos_net']:>9.0f} {r['oos_pf']:>6.2f} | "
              f"WFE={wfe:>4.2f} oc={r['open_candles']} {r['range_mode'][:1]}{r['break_mode'][:1]} "
              f"buf{r['break_buf']} eu{r['entry_until']} otm{r['otm_strikes']} "
              f"{r['sl_mode'][:4]} sb{r['sl_buf']} {r['target_mode'][:1]} "
              f"rf{r['ride_frac']} pc{r['pct_target']} {r['direction'][:1]} f{r['allow_friday']}")

    # Save full results.
    fn = "b08_breakout_results.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {fn} ({len(rows)} rows) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
