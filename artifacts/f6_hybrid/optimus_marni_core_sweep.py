"""Optimus GPU sweep for B09 — Marni Core 15m-HA + UT Bot color signal.

Runs the EXACT 3D-fused GPU engine (optimized_gpu_backtest.evaluate_marni_core_batch).
Full 7-year (2020-2026) + IS/OOS walk-forward, honest robustness reporting.

Usage:
    $env:PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync"
    python optimus_marni_core_sweep.py
"""
import csv
import time
import itertools

import torch

import optimized_gpu_backtest as m

BATCH = 1

# Custom parameter list for the Marni Core 15m-HA UT Bot signal.
#   3-phase UT Bot color RANGE (GREEN-RED-GREEN / RED-GREEN-RED) -> Fibonacci:
#   entry = 0.786 retracement; SL = 1.115/1.25 extension beyond range; TP = 0.29/0 extension.
# NOTE: on 15m HA, UT Bot key >=1.0 is too "sticky" to form 3-phase setups on Nifty
# (<=7 setups / 7y). The responsive UT Bot key range is 0.5-0.8, so we sweep that.
GRID = {
    "ut_key": [0.5, 0.6, 0.7, 0.8],
    "atr_period": [10, 14, 20],
    "sl_m": [1.115, 1.25],
    "tp_m": [0.29, 0.0],
    "direction": ["both", "bull", "bear"],
    "allow_friday": [False, True],
    "daily_loss_pts": [20, 30, 50],
    "daily_profit_pts": [30, 50, 80],
}


def gen_combos():
    ks = list(GRID.keys())
    for vals in itertools.product(*(GRID[k] for k in ks)):
        yield {k: v for k, v in zip(ks, vals)}


def _materialize(rs):
    """Force every metric to a native Python scalar NOW, before any later
    evaluate_batch call can recycle the underlying GPU buffer."""
    out = []
    for r in rs:
        d = {}
        for k, v in r.items():
            if hasattr(v, "item"):  # torch tensor
                d[k] = v.item()
            elif hasattr(v, "numpy"):
                d[k] = float(v.numpy())
            elif isinstance(v, (int, float)):
                d[k] = v
            else:
                d[k] = v
        out.append(d)
    return out


def main():
    combos = list(gen_combos())
    print(f"[B09 sweep] {len(combos)} combos | GPU {m.device} | N={m.N_DAYS} T={m.T_BARS}", flush=True)
    t0 = time.time()

    rows = []
    nb = (len(combos) + BATCH - 1) // BATCH
    for bi in range(nb):
        batch = combos[bi * BATCH:(bi + 1) * BATCH]
        r_full = _materialize(m.evaluate_batch("B09", batch, None))
        r_is = _materialize(m.evaluate_batch("B09", batch, m.d_is_mask))
        r_oos = _materialize(m.evaluate_batch("B09", batch, m.d_oos_mask))
        for j, p in enumerate(batch):
            rf, ri, ro = r_full[j], r_is[j], r_oos[j]
            rows.append({
                **p,
                "full_trades": rf["trades"], "full_wr": round(rf["win_rate"], 2),
                "full_net": round(rf["net_rs"], 1), "full_pf": round(rf["pf"], 3),
                "full_dd": round(rf["max_dd"], 1),
                "is_net": round(ri["net_rs"], 1), "is_pf": round(ri["pf"], 3),
                "oos_trades": ro["trades"], "oos_wr": round(ro["win_rate"], 2),
                "oos_net": round(ro["net_rs"], 1), "oos_pf": round(ro["pf"], 3),
                "oos_dd": round(ro["max_dd"], 1),
            })
        if (bi + 1) % 20 == 0 or bi == nb - 1:
            print(f"  batch {bi+1}/{nb}  ({time.time()-t0:.1f}s)", flush=True)

    # Non-walk-forward winner = best on FULL 7y.
    pooled = [r for r in rows if r["full_trades"] >= 30]
    oos_pos = [r for r in pooled if r["oos_net"] > 0]
    print(f"\nPool: {len(pooled)} combos with >=30 trades | "
          f"OOS-positive: {len(oos_pos)} ({100*len(oos_pos)/max(1,len(pooled)):.1f}%)", flush=True)

    fn = "b09_marni_core_results.csv"
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved {fn} ({len(rows)} rows) in {time.time()-t0:.1f}s", flush=True)

    # Winner tables read back from the written CSV (authoritative, type-stable).
    crows = []
    for r in csv.DictReader(open(fn)):
        for k in ("ut_key", "sl_m", "tp_m"):
            r[k] = float(r[k])
        for k in ("atr_period", "daily_loss_pts", "daily_profit_pts", "full_trades", "oos_trades"):
            r[k] = int(r[k])
        for k in ("full_net", "full_wr", "full_pf", "full_dd", "is_net", "is_pf",
                  "oos_net", "oos_wr", "oos_pf", "oos_dd"):
            r[k] = float(r[k])
        r["allow_friday"] = (r["allow_friday"] == "True")
        crows.append(r)

    print("\n=== NON-WALK-FORWARD (full 7y) TOP 15 by net ===")
    print(f"{'net':>10} {'PF':>5} {'WR':>5} {'tr':>5} | {'IS_net':>9} {'OOS_net':>9} {'OOS_PF':>6} | params")
    for r in sorted(crows, key=lambda r: r["full_net"], reverse=True)[:15]:
        wfe = r["oos_net"] / r["full_net"] if r["full_net"] else 0
        print(f"{r['full_net']:>10.0f} {r['full_pf']:>5.2f} {r['full_wr']:>5.1f} {r['full_trades']:>5} | "
              f"{r['is_net']:>9.0f} {r['oos_net']:>9.0f} {r['oos_pf']:>6.2f} | "
              f"WFE={wfe:>4.2f} key{r['ut_key']} ap{r['atr_period']} sl{r['sl_m']} tp{r['tp_m']} "
              f"{r['direction'][:1]} f{r['allow_friday']} dl{r['daily_loss_pts']} dp{r['daily_profit_pts']}")

    print(f"\n=== WALK-FORWARD (OOS / forward) TOP 15 by OOS net ===")
    print(f"{'OOS_net':>9} {'OOS_PF':>6} {'OOS_WR':>6} {'OOS_tr':>6} | {'full_net':>10} {'IS_net':>9} | params")
    for r in sorted([c for c in crows if c["oos_trades"] >= 10], key=lambda r: r["oos_net"], reverse=True)[:15]:
        print(f"{r['oos_net']:>9.0f} {r['oos_pf']:>6.2f} {r['oos_wr']:>6.1f} {r['oos_trades']:>6} | "
              f"{r['full_net']:>10.0f} {r['is_net']:>9.0f} | "
              f"key{r['ut_key']} ap{r['atr_period']} sl{r['sl_m']} tp{r['tp_m']} "
              f"{r['direction'][:1]} f{r['allow_friday']} dl{r['daily_loss_pts']} dp{r['daily_profit_pts']}")


if __name__ == "__main__":
    main()
