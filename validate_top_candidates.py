"""Phase 2 + 3 — Full 5-Year Validation, Walk-Forward & Sensitivity Analysis.

Reads optuna_results.csv (Phase 1), takes the top N unique candidates and:

  Phase 2 — Full 5Y (2020-2024) backtest on each:
      - all 5 years profitable (mandatory)
      - WR >= 42%, PF > 1.4, trades >= 4,000 (statistical significance)
      - compare vs champion (ATR×2/×4 Unlimited + F6 = +1,030,642 / 48.0% / 1.45)

  Phase 3 — Walk-Forward robustness (no re-optimization per window):
      Rolling 3Y-in-sample -> 1Y-out-of-sample windows, same candidate params:
        [2020-22 -> 2023], [2021-23 -> 2024], [2020-21 -> 2022]
      Candidate must be profitable in EVERY OOS year.

  Phase 3b — Sensitivity: top 3 candidates, perturb each axis +/-1 step,
      full 5Y net profit must stay within +-10% of the unperturbed run.

Usage:
  python validate_top_candidates.py [--top 15] [--smoke]
"""

import argparse
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from backtest_5y_optimized import load_spot, option_files, summarize, print_yearly_breakdown
import grid_optimize_f6_atr as eng

CHAMPION = dict(eng.CHAMPION)
CHAMPION_5Y = {"rs": 1030642, "wr": 48.0, "pf": 1.45, "trades": 7843}

AXES = ["s1_k", "s4_k", "atr_period", "atr_sl_mult", "atr_tp_mult",
        "f6_s4_thresh", "f6_s1_thresh", "consec_loss"]

WF_WINDOWS = [
    ("2020", "2022", "2023"),   # IS 2020-22 -> OOS 2023
    ("2021", "2023", "2024"),   # IS 2021-23 -> OOS 2024
    ("2020", "2021", "2022"),   # IS 2020-21 -> OOS 2022
]


def run_candidates(candidates, spot_all, files, days, label):
    """Run a list of param dicts over `days`, returns list of (params, trades)."""
    results = []
    with Pool(processes=eng.WORKERS, initializer=eng.init_worker_local,
              initargs=(spot_all,)) as pool:
        for p in candidates:
            t0 = time.time()
            trades = eng.run_days(pool, p, days, files, spot_all)
            results.append((p, trades, time.time() - t0))
    return results


def yearly_stats(trades):
    """Returns {year: summarize} for all years present."""
    out = {}
    for t in trades:
        out.setdefault(int(t["date"][:4]), []).append(t)
    return {y: summarize(ts) for y, ts in out.items()}


def check_all_years_profitable(trades):
    ys = yearly_stats(trades)
    bad = [y for y, st in ys.items() if st["rs"] <= 0]
    return len(bad) == 0, bad, ys


def phase2(candidates, spot_all, files, days_all):
    print(f"\n{'='*120}")
    print(f"PHASE 2 — FULL 5-YEAR VALIDATION ({len(days_all)} days, 2020-2024)")
    print(f"{'='*120}")
    results = run_candidates(candidates, spot_all, files, days_all, "5Y")

    rows = []
    for p, trades, elapsed in results:
        st = summarize(trades)
        ok_years, bad, ys = check_all_years_profitable(trades)
        meets = (st["wr"] >= 42.0 and st["pf"] > 1.4 and st["trades"] >= 4000 and ok_years)
        rows.append({"p": p, "st": st, "ok_years": ok_years, "bad_years": bad,
                     "meets": meets, "ys": ys, "elapsed": elapsed})

    rows.sort(key=lambda r: r["st"]["rs"], reverse=True)
    print(f"\n{'RANK':>4} | {'NET PROFIT':>12} | {'WR%':>6} | {'PF':>5} | {'TRADES':>7} | {'5Y OK':>6} | params")
    print("-" * 120)
    for i, r in enumerate(rows, 1):
        flag = "PASS" if r["meets"] else "FAIL"
        print(f"{i:4d} | Rs {r['st']['rs']:>9,d} | {r['st']['wr']:5.1f}% | {r['st']['pf']:5.2f} | "
              f"{r['st']['trades']:7,d} | {flag:>6} | "
              f"s1={r['p']['s1_k']} s4={r['p']['s4_k']} atr={r['p']['atr_period']} "
              f"sl={r['p']['atr_sl_mult']} tp={r['p']['atr_tp_mult']} "
              f"f6s4={r['p']['f6_s4_thresh']} f6s1={r['p']['f6_s1_thresh']} cl={r['p']['consec_loss']}")
    return rows


def phase3_walk_forward(rows, spot_all, files, days_all, max_candidates=5):
    print(f"\n{'='*120}")
    print("PHASE 3 — WALK-FORWARD VALIDATION (top candidates on unseen OOS years)")
    print(f"{'='*120}")
    days_by_year = {y: [d for d in days_all if d.startswith(str(y))] for y in range(2020, 2025)}
    total_days = sum(len(v) for v in days_by_year.values())

    for wf_is_start, wf_is_end, wf_oos in WF_WINDOWS:
        is_days = [d for y in range(int(wf_is_start), int(wf_is_end) + 1) for d in days_by_year[y]]
        oos_days = days_by_year[int(wf_oos)]
        print(f"\n--- IS {wf_is_start}-{wf_is_end} ({len(is_days)}d) -> OOS {wf_oos} ({len(oos_days)}d) ---")
        cands = [r["p"] for r in rows[:max_candidates]]
        with Pool(processes=eng.WORKERS, initializer=eng.init_worker_local,
                  initargs=(spot_all,)) as pool:
            for p in cands:
                trades_oos = eng.run_days(pool, p, oos_days, files, spot_all)
                st = summarize(trades_oos)
                mark = "PASS" if st["rs"] > 0 else "FAIL"
                print(f"  OOS {wf_oos} | Rs {st['rs']:+10,d} | WR {st['wr']:5.1f}% | PF {st['pf']:5.2f} | "
                      f"trades {st['trades']:5,d} | {mark} | "
                      f"s1={p['s1_k']} s4={p['s4_k']} atr={p['atr_period']} sl={p['atr_sl_mult']} "
                      f"tp={p['atr_tp_mult']} f6s4={p['f6_s4_thresh']} f6s1={p['f6_s1_thresh']} cl={p['consec_loss']}")


def neighbor_values(name):
    vals = list(eng.SEARCH_SPACE[name])
    return vals


def phase3b_sensitivity(rows, spot_all, files, days_all, max_candidates=3):
    print(f"\n{'='*120}")
    print("PHASE 3b — PARAMETER SENSITIVITY (top candidates, +/-1 step per axis, full 5Y)")
    print(f"{'='*120}")
    top = rows[:max_candidates]
    for r in top:
        base = r["st"]["rs"]
        base_p = r["p"]
        print(f"\nCandidate: s1={base_p['s1_k']} s4={base_p['s4_k']} atr={base_p['atr_period']} "
              f"sl={base_p['atr_sl_mult']} tp={base_p['atr_tp_mult']} "
              f"f6s4={base_p['f6_s4_thresh']} f6s1={base_p['f6_s1_thresh']} cl={base_p['consec_loss']} "
              f"-> 5Y Rs {base:+,d}")
        perturbs = []
        for axis in AXES:
            vals = neighbor_values(axis)
            idx = vals.index(base_p[axis])
            for delta in (-1, 1):
                j = idx + delta
                if 0 <= j < len(vals):
                    np_ = dict(base_p)
                    np_[axis] = vals[j]
                    perturbs.append((axis, vals[j], np_))
        with Pool(processes=eng.WORKERS, initializer=eng.init_worker_local,
                  initargs=(spot_all,)) as pool:
            worst = 0.0
            for axis, newval, np_ in perturbs:
                trades = eng.run_days(pool, np_, days_all, files, spot_all)
                st = summarize(trades)
                pct = (st["rs"] - base) / abs(base) * 100
                worst = min(worst, pct)
                print(f"  {axis:14s} = {newval:<6} -> Rs {st['rs']:+10,d} ({pct:+6.1f}%)")
        verdict = "ROBUST" if worst > -10.0 else "FRAGILE"
        print(f"  WORST perturbation: {worst:+.1f}% -> {verdict} (must stay above -10%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not Path(eng.RESULTS_CSV).exists():
        print(f"{eng.RESULTS_CSV} not found — run Phase 1 first")
        return

    df = pd.read_csv(eng.RESULTS_CSV)
    cols = list(eng.SEARCH_SPACE.keys())
    df = df[df["pruned"] == 0]
    uniq = df.drop_duplicates(subset=cols)
    top = uniq.sort_values("score", ascending=False).head(args.top)
    candidates = [dict(zip(cols, [float(r[c]) if c in ("atr_sl_mult", "atr_tp_mult", "f6_s4_thresh", "f6_s1_thresh")
                                   else int(r[c]) for c in cols]))
                  for _, r in top.iterrows()]
    for c in candidates:
        c["s1_d"] = 3

    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days_all = sorted(set(files.keys()) & set(spot_all.keys()))

    print(f"Loaded {len(days_all)} days. Top {len(candidates)} unique candidates from Phase 1.")

    if args.smoke:
        days5 = days_all[:5]
        print(f"=== SMOKE TEST — {len(days5)} DAYS, top candidate ===")
        with Pool(processes=eng.WORKERS, initializer=eng.init_worker_local,
                  initargs=(spot_all,)) as pool:
            trades = eng.run_days(pool, candidates[0], days5, files, spot_all)
        st = summarize(trades)
        print(f"Trades: {st['trades']} | WR: {st['wr']:.1f}% | Rs: {st['rs']:+,d}")
        print("SMOKE OK" if 15 <= st["trades"] <= 40 else "SMOKE SUSPICIOUS")
        return

    print("\nReference champion (full 5Y):", CHAMPION_5Y)

    rows = phase2(candidates, spot_all, files, days_all)
    phase3_walk_forward(rows, spot_all, files, days_all, max_candidates=5)
    phase3b_sensitivity(rows, spot_all, files, days_all, max_candidates=3)

    passed = [r for r in rows if r["meets"]]
    print(f"\n{'='*120}")
    print(f"SUMMARY: {len(passed)}/{len(rows)} candidates passed ALL Phase-2 gates "
          f"(all-years profitable, WR>=42%, PF>1.4, trades>=4000)")
    for i, r in enumerate(rows, 1):
        m = "PASS" if r["meets"] else "FAIL"
        print(f"  #{i:2d} {m} | Rs {r['st']['rs']:>10,d} | WR {r['st']['wr']:5.1f}% | PF {r['st']['pf']:5.2f} | "
              f"trades {r['st']['trades']:6,d}")


if __name__ == "__main__":
    main()