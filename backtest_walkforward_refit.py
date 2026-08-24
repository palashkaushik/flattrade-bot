"""Rolling Walk-Forward REFIT Backtest — re-optimize per IS window, test on OOS.

Unlike backtest_walkforward_fees.py (fixed champion params), this refits
Optuna on each IS window and evaluates the freshly-fit params on the OOS year.
Answer it targets: is the strategy TRAINABLE (edge survives refitting on data
the refit never saw), or was the champion just a lucky in-sample draw?

Windows (same as fees script):
    IS 2020-22 -> OOS 2023   (true OOS)
    IS 2021-23 -> OOS 2024   (true OOS)
    IS 2020-21 -> OOS 2022   (pseudo-OOS: skip with --only-true)

Per window:
  1. Optuna TPE study (same seed 42, MedianPruner, same 8-axis search space
     as the original Phase 1) on the IS window — prune on the 1st IS year.
  2. Best IS params -> run OOS year -> deduct fees+slippage per trade
     (same cost model as backtest_walkforward_fees.py).
  3. A/B: fixed champion params run on the SAME OOS year with same costs.
  4. Only TRUE-OOS years (2023+2024) are stitched into the monthly ramp.

Usage:
  python backtest_walkforward_refit.py --smoke
  python backtest_walkforward_refit.py --trials 40
  python backtest_walkforward_refit.py --trials 40 --only-true
  python backtest_walkforward_refit.py --trials 40 --only-true --incremental
  python backtest_walkforward_refit.py --trials 40 --capital 20000 --brokerage 0
"""

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import optuna

import grid_optimize_f6_atr as grid
from backtest_5y_optimized import load_spot, option_files
from backtest_monthly_ramp import CHAMPION_PARAMS
from backtest_walkforward_fees import (
    WF_WINDOWS, apply_costs, net_stats, summarize_net,
)
from f6_hybrid.raw_features import run_factorized_candidates

WORKERS = grid.WORKERS
RESULTS_DIR = Path("refit_results")
RESULTS_DIR.mkdir(exist_ok=True)


def slice_by_year(days, y0, y1):
    return [d for d in days if y0 <= d[:4] <= y1]


def run_refit_window(pool, window, days, files, spot_all, n_trials, brokerage):
    """Run one Optuna study on the IS window, return best params + stats."""
    is_start, is_end, oos_year = window
    is_days = slice_by_year(days, is_start, is_end)
    year1 = is_start
    year1_days = [d for d in is_days if d.startswith(f"{year1}-")]
    rest_days = [d for d in is_days if not d.startswith(f"{year1}-")]
    t_win = time.time()

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0,
                                         n_min_trials=2)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    csv_path = RESULTS_DIR / f"study_{oos_year}.csv"

    def objective(trial):
        params = {name: trial.suggest_categorical(name, values)
                  for name, values in grid.SEARCH_SPACE.items()}
        params["s1_d"] = 3
        t0 = time.time()
        trades_y1 = grid.run_days(pool, params, year1_days, files, spot_all)
        st_y1, _ = grid.stats_for(trades_y1)
        trial.report(st_y1["rs"], step=1)
        if trial.should_prune():
            _log_trial(csv_path, trial.number, -1e9, params, st_y1,
                       st_y1["rs"], round(time.time() - t0, 1), 1)
            print(f"Trial {trial.number:3d} | PRUNED (year1 rs={st_y1['rs']:+10,d}) "
                  f"| {time.time()-t0:5.0f}s", flush=True)
            raise optuna.TrialPruned()
        trades_rest = grid.run_days(pool, params, rest_days, files, spot_all)
        all_trades = trades_y1 + trades_rest
        st, _ = grid.stats_for(all_trades)
        score = grid.composite_score(st)
        elapsed = time.time() - t0
        _log_trial(csv_path, trial.number, score, params, st, st_y1["rs"],
                   elapsed, 0)
        print(f"Trial {trial.number:3d} | score={score:12.0f} | "
              f"rs={st['rs']:+12,d} | wr={st['wr']:5.1f}% | pf={st['pf']:5.2f} | "
              f"trades={st['trades']:6,d} | {elapsed:5.0f}s", flush=True)
        return score

    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best["s1_d"] = 3
    trades_best = grid.run_days(pool, best, is_days, files, spot_all)
    st_best, _ = grid.stats_for(trades_best)
    print(f"\n  Window {is_start}-{is_end}->{oos_year} | {n_trials} trials | "
          f"{time.time()-t_win:.0f}s | best IS: {summarize_stats(st_best)}", flush=True)
    return best, st_best


def run_refit_window_incremental(window, days, files, spot_all, n_trials,
                                  batch_size=8):
    """Refit in small ask/tell batches so signal state is reused safely."""
    is_start, is_end, oos_year = window
    is_days = slice_by_year(days, is_start, is_end)
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    csv_path = RESULTS_DIR / f"study_{oos_year}_incremental.csv"
    t_win = time.time()

    for start in range(0, n_trials, batch_size):
        count = min(batch_size, n_trials - start)
        trials = [study.ask() for _ in range(count)]
        params_batch = []
        for trial in trials:
            params = {name: trial.suggest_categorical(name, values)
                      for name, values in grid.SEARCH_SPACE.items()}
            params["s1_d"] = 3
            params_batch.append(params)

        trade_lists, base_builds, signal_builds = run_factorized_candidates(
            params_batch, is_days, files, spot_all, workers=WORKERS
        )
        print(f"  Batch {start + 1}-{start + count}: {base_builds} base / "
              f"{signal_builds} signal builds",
              flush=True)
        for trial, params, trades in zip(trials, params_batch, trade_lists):
            st, _ = grid.stats_for(trades)
            score = grid.composite_score(st)
            study.tell(trial, score)
            _log_trial(csv_path, trial.number, score, params, st, st["rs"],
                       0.0, 0)
            print(f"Trial {trial.number:3d} | score={score:12.0f} | "
                  f"rs={st['rs']:+12,d} | wr={st['wr']:5.1f}% | "
                  f"pf={st['pf']:5.2f} | trades={st['trades']:6,d}", flush=True)

    best = study.best_params
    best["s1_d"] = 3
    best_trades, _, _ = run_factorized_candidates(
        [best], is_days, files, spot_all, workers=WORKERS
    )
    st_best, _ = grid.stats_for(best_trades[0])
    print(f"\n  Incremental window {is_start}-{is_end}->{oos_year} | "
          f"{n_trials} trials | {time.time()-t_win:.0f}s | "
          f"best IS: {summarize_stats(st_best)}", flush=True)
    return best, st_best


def _log_trial(csv_path, number, score, params, st, year1_rs, elapsed, pruned):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["trial", "score", "s1_k", "s4_k", "atr_period",
                        "atr_sl_mult", "atr_tp_mult", "f6_s4_thresh",
                        "f6_s1_thresh", "consec_loss", "trades", "wr", "net_rs",
                        "pf", "year1_rs", "elapsed_s", "pruned"])
        w.writerow([number, round(score), params["s1_k"], params["s4_k"],
                    params["atr_period"], params["atr_sl_mult"],
                    params["atr_tp_mult"], params["f6_s4_thresh"],
                    params["f6_s1_thresh"], params["consec_loss"], st["trades"],
                    round(st["wr"], 2), st["rs"], round(st["pf"], 4),
                    round(year1_rs), elapsed, pruned])


def summarize_stats(st):
    return (f"trades {st['trades']:5,d} | WR {st['wr']:5.1f}% | "
            f"Rs {st['rs']:+12,d} | PF {st['pf']:5.2f}")


def pretty_params(p):
    return (f"s1={p['s1_k']} s4={p['s4_k']} atr={p['atr_period']} "
            f"sl={p['atr_sl_mult']} tp={p['atr_tp_mult']} "
            f"f6s4={p['f6_s4_thresh']} f6s1={p['f6_s1_thresh']} cl={p['consec_loss']}")


def main():
    ap = argparse.ArgumentParser(description="Rolling Walk-Forward REFIT Backtest")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--capital", type=float, default=20000.0)
    ap.add_argument("--increment", type=float, default=40000.0)
    ap.add_argument("--brokerage", type=float, default=0.0)
    ap.add_argument("--only-true", action="store_true",
                    help="skip the pseudo-OOS 2022 window")
    ap.add_argument("--incremental", action="store_true",
                    help="batch candidates and reuse signal state")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    windows = WF_WINDOWS
    if args.only_true:
        windows = [w for w in windows if w[2] in ("2023", "2024")]

    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    print("=== ROLLING WALK-FORWARD REFIT | Optuna per IS window | "
          f"{args.trials} trials/window | fees+slippage on every OOS trade ===", flush=True)
    print(f"Capital {args.capital:,.0f} | ramp {args.increment:,.0f}/lot | "
          f"workers {WORKERS} | windows: {[w[2] for w in windows]}", flush=True)

    oos_collect = []
    summary = {"windows": [], "champion_ab": {}}
    for window in windows:
        is_start, is_end, oos_year = window
        oos_days = [d for d in days if d.startswith(f"{oos_year}-")]
        if args.smoke:
            is_days_smoke = slice_by_year(days, is_start, is_end)[:5]
            oos_days = oos_days[:5]
            n_trials = 2
        else:
            is_days_smoke = None
            n_trials = args.trials

        print(f"\n{'='*104}")
        print(f"WINDOW {is_start}-{is_end} -> OOS {oos_year} "
              f"{'[TRUE OOS]' if oos_year in ('2023','2024') else '[pseudo-OOS]'}")
        print(f"{'='*104}", flush=True)

        if args.incremental:
            if args.smoke:
                best = CHAMPION_PARAMS.copy()
                best["s1_d"] = 3
                is_trades, _, _ = run_factorized_candidates(
                    [best], is_days_smoke, files, spot_all, workers=WORKERS
                )
                st_best, _ = grid.stats_for(is_trades[0])
                print(f"  INCREMENTAL SMOKE IS {is_start}-{is_end} (5d): "
                      f"{summarize_stats(st_best)}")
            else:
                best, st_best = run_refit_window_incremental(
                    window, days, files, spot_all, n_trials,
                    batch_size=args.batch_size,
                )
            t0 = time.time()
            best_oos, _, _ = run_factorized_candidates(
                [best], oos_days, files, spot_all, workers=WORKERS
            )
            raw_elapsed = time.time() - t0
            trades_oos = best_oos[0]
            champion_oos, _, _ = run_factorized_candidates(
                [CHAMPION_PARAMS], oos_days, files, spot_all, workers=WORKERS
            )
            trades_champ = champion_oos[0]
        else:
            with Pool(processes=WORKERS, initializer=grid.init_worker_local,
                      initargs=(spot_all,)) as pool:
                if args.smoke:
                    best = CHAMPION_PARAMS.copy()
                    best["s1_d"] = 3
                    st_best = {}
                    tr = grid.run_days(pool, best, is_days_smoke, files, spot_all)
                    st_best, _ = grid.stats_for(tr)
                    print(f"  SMOKE IS {is_start}-{is_end} (5d): {summarize_stats(st_best)}")
                else:
                    best, st_best = run_refit_window(pool, window, days, files,
                                                     spot_all, n_trials,
                                                     args.brokerage)
                t0 = time.time()
                trades_oos = grid.run_days(pool, best, oos_days, files, spot_all)
                raw_elapsed = time.time() - t0
                trades_champ = grid.run_days(pool, CHAMPION_PARAMS, oos_days,
                                             files, spot_all)

        apply_costs(trades_oos, args.brokerage)
        apply_costs(trades_champ, args.brokerage)
        st_net = net_stats(trades_oos)
        st_champ_net = net_stats(trades_champ)

        print(f"\n  REFIT best ({pretty_params(best)})")
        print(f"    OOS {oos_year} (raw, {len(oos_days)}d, {raw_elapsed:.1f}s): "
              f"{summarize_net(st_net)}")
        print(f"  CHAMPION fix  ({pretty_params(CHAMPION_PARAMS)})")
        print(f"    OOS {oos_year} (raw, {raw_elapsed:.1f}s): "
              f"{summarize_net(st_champ_net)}")

        summary["windows"].append({
            "window": f"{is_start}-{is_end}->{oos_year}",
            "true_oos": oos_year in ("2023", "2024"),
            "best_params": best,
            "is": {k: v for k, v in st_best.items()} if not args.smoke else None,
            "oos_net": st_net,
            "champion_oos_net": st_champ_net,
        })
        summary["champion_ab"][oos_year] = st_champ_net
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)

        if args.smoke:
            ok = 15 <= st_net["trades"] <= 40
            print(f"SMOKE {oos_year}: {st_net['trades']} trades "
                  f"(expect 15-40) -> {'OK' if ok else 'SUSPICIOUS'}")
            continue
        if oos_year in ("2023", "2024"):
            oos_collect.extend(trades_oos)

    if args.smoke:
        sys.exit(0)

    print(f"\n{'='*104}")
    print(f"STITCHED TRUE-OOS REFIT STREAM (2023 + 2024) — {len(oos_collect)} trades")
    print(f"{'='*104}")
    st = net_stats(oos_collect)
    print(summarize_net(st))
    print("\nMonthly lot ramp (capital refreshes at stitch start):")
    from backtest_monthly_ramp import (
        MARGIN_PER_LOT, apply_monthly_ramp, print_ramp_table, print_yearly_ramp,
    )
    rows, _ = apply_monthly_ramp(oos_collect, args.capital, args.increment,
                                 MARGIN_PER_LOT)
    print_ramp_table(rows)
    print_yearly_ramp(rows, args.capital)
    print(f"\nJSON summary: {RESULTS_DIR / 'summary.json'} | studies: {RESULTS_DIR}/study_*.csv")


if __name__ == "__main__":
    main()
