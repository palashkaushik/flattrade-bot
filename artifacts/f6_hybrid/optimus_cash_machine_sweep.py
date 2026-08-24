"""
OPTIMUS CASH-MACHINE SWEEP  —  MAX PROFIT / LEAST DD / CONSISTENT
=================================================================
Fixed-component meta-confirmation ensemble over the project's best strategies
(top-5 of every phase + top-10 leaderboard) as stochastic-vote components.
Only the ensemble risk/exit dials are optimized.

NEW OBJECTIVE (per user):  maximize NET POINTS with LEAST DRAWDOWN, CONSISTENTLY
  - search score  = net_rs - 20 * max_dd   (profit primary, DD as cost)
  - HARD FILTER   = profitable in EVERY calendar year 2020..2026  (consistency)
  - report TOP 5  by net_rs, showing per-year + OOS + DD.

Built on cross_strategy_ensemble_gpu.evaluate_ensemble_batch.
Run:  HALF=1 BATCH=100 TRIALS=2000 python optimus_cash_machine_sweep.py
"""

import os, sys, json, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler

import optimized_gpu_backtest as base
import cross_strategy_ensemble_gpu as ens

device = base.device
N_DAYS = base.N_DAYS
LOT_SIZE = base.LOT_SIZE
N_TOTAL = N_DAYS  # 1574 (7Y)

# ── fixed component library: stochastic-vote specs (CE: S4>=s4_ob & S1<=s1_os) ──
def _comp(tf, s1k, s4k, s4ob, s1os, tag):
    return {"strat_id": "B07", "timeframe": tf, "s1_k": s1k, "s4_k": s4k,
            "s1_os": s1os, "s4_ob": s4ob, "atr_p": 10,
            "sl_m": 2.0, "tp_m": 4.0, "daily_loss_pts": 10, "daily_profit_pts": 50,
            "moneyness": 0.5, "max_trade_loss_rs": 9999,
            "sess_start_off": 5, "sess_end_off": 45, "sess_end": 300, "tag": tag}

COMPONENTS = [
    _comp(1, 30, 70, 70.0, 40.0, "B02_1m"),
    _comp(2, 30, 70, 70.0, 40.0, "B03_2m"),
    _comp(3, 30, 70, 70.0, 40.0, "B04_3m"),
    _comp(5, 30, 70, 70.0, 40.0, "B05_5m"),
    _comp(3, 30, 70, 70.0, 40.0, "B07_3m"),
    _comp(1, 16, 80, 77.5, 17.5, "F01_C02_1m"),
    _comp(1,  7, 60, 80.0, 25.0, "GPU_Optuna_1m"),
    _comp(1, 12, 50, 79.5, 25.0, "MarniF6_1m"),
    _comp(1,  9, 70, 79.5, 25.0, "S1TurnUp_1m"),
    _comp(1, 11, 75, 75.0, 25.0, "Elder_1m"),
    _comp(1, 24, 95, 77.5, 17.5, "F05_DblStoch_1m"),
]
N_COMP = len(COMPONENTS)
print(f"Components loaded: {N_COMP}  (days={N_DAYS}, lot={LOT_SIZE})", flush=True)

# per-year masks for the consistency test
YEARS = sorted(set(d[:4] for d in base.all_days))
YMASK = [(y, torch.tensor([d.startswith(y) for d in base.all_days], dtype=torch.bool, device=device)) for y in YEARS]
print(f"Consistency years: {YEARS}", flush=True)


def suggest_ens(trial):
    return {
        "timeframe": 1, "s1_k": 13, "s4_k": 70, "s1_os": 15.0, "s4_ob": 77.5,
        "atr_p": trial.suggest_int("atr_p", 10, 35),
        "sl_m": trial.suggest_float("sl_m", 1.0, 5.0, step=0.1),
        "tp_m": trial.suggest_float("tp_m", 2.0, 12.0, step=0.25),
        "daily_loss_pts": trial.suggest_int("daily_loss_pts", 5, 40, step=5),
        "daily_profit_pts": 60,
        "moneyness": trial.suggest_categorical("moneyness", [0.5, 0.6, 0.7]),
        "max_trade_loss_rs": 9999,
        "sess_start_off": trial.suggest_int("sess_start_off", 0, 30, step=5),
        "sess_end_off": trial.suggest_int("sess_end_off", 30, 75, step=15),
        "sess_end": 345 - trial.suggest_int("sess_end_off", 30, 75, step=15),
        "confirm_k": trial.suggest_int("confirm_k", 1, N_COMP),
        "band_relax": trial.suggest_float("band_relax", 0.0, 15.0, step=1.0),
        "reentry": trial.suggest_categorical("reentry", [False, True]),
    }


def _ens_from_params(p):
    return {"timeframe": 1, "s1_k": 13, "s4_k": 70, "s1_os": 15.0, "s4_ob": 77.5,
            "atr_p": int(p["atr_p"]), "sl_m": float(p["sl_m"]), "tp_m": float(p["tp_m"]),
            "daily_loss_pts": int(p["daily_loss_pts"]), "daily_profit_pts": 60,
            "moneyness": float(p["moneyness"]), "max_trade_loss_rs": 9999,
            "sess_start_off": int(p["sess_start_off"]), "sess_end_off": int(p["sess_end_off"]),
            "sess_end": 345 - int(p["sess_end_off"]), "confirm_k": int(p["confirm_k"]),
            "band_relax": float(p["band_relax"]), "reentry": bool(p["reentry"])}


def profit_score(r):
    """Maximize net points; drawdown is a cost; non-degenerate + profitable."""
    net = float(r.get("net_rs", 0.0))
    dd = float(r.get("max_dd", 0.0))
    tr = int(r.get("trades", 0))
    if net <= 0 or tr < 400:           # >~0.25 trades/day, must be profitable
        return -1e9
    return net - 20.0 * dd


def run():
    n_trials = int(os.environ.get("TRIALS", "2000"))
    bs = int(os.environ.get("BATCH", "100"))
    print(f"\n=== MAX-PROFIT / LEAST-DD / CONSISTENT STUDY: {n_trials} trials (NW) ===", flush=True)
    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=42, constant_liar=True, multivariate=True))
    n_batches = max(1, n_trials // bs)
    for _ in range(n_batches):
        batch = [study.ask() for _ in range(bs)]
        pairs, keep = [], []
        for t in batch:
            e = suggest_ens(t)
            if e["tp_m"] < 1.5 * e["sl_m"]:
                study.tell(t, -1e9); continue
            pairs.append((COMPONENTS, e)); keep.append(t)
        if not pairs:
            continue
        res_list = ens.evaluate_ensemble_batch(pairs, None)
        for t, res in zip(keep, res_list):
            for k, v in res.items():
                t.set_user_attr(k, float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in t.params.items():
                t.set_user_attr("p_" + k, v)
            study.tell(t, profit_score(res))
    print(f"NW study done in {time.time()-t0:.0f}s", flush=True)

    # ── consistency pool: top by profit + top by WR (high-WR→fewer losers→more all-month-green)
    trials = [t for t in study.trials if t.value is not None and t.value > -1e8]
    by_profit = sorted(trials, key=lambda t: t.value, reverse=True)[:250]
    by_wr = sorted(trials, key=lambda t: t.user_attrs.get("win_rate", 0), reverse=True)[:250]
    seen, pool = set(), []
    for t in by_profit + by_wr:
        if id(t) not in seen:
            seen.add(id(t)); pool.append(t)
    print(f"Consistency pool: {len(pool)} candidates; checking EVERY MONTH 2020-01..2026-05...", flush=True)

    pairs = [(COMPONENTS, _ens_from_params(t.params)) for t in pool]
    MONTHS = sorted(set(d[:7] for d in base.all_days))
    MMASK = [(mo, torch.tensor([d.startswith(mo) for d in base.all_days], dtype=torch.bool, device=device)) for mo in MONTHS]
    per_month = [dict() for _ in pool]
    CH = 100
    for mo, m in MMASK:
        for s in range(0, len(pairs), CH):
            rl = ens.evaluate_ensemble_batch(pairs[s:s + CH], m)
            for j, r in enumerate(rl):
                per_month[s + j][mo] = float(r.get("net_rs", 0.0))

    consistent = []
    for i, t in enumerate(pool):
        nets = per_month[i]
        n_neg = sum(1 for v in nets.values() if v <= 0)
        ok = n_neg == 0
        consistent.append((i, t, nets, ok, n_neg))
    n_allpos = sum(1 for c in consistent if c[3])
    print(f"  candidates profitable EVERY month: {n_allpos}/{len(consistent)}", flush=True)

    # rank: all-months-green first, then by NW net, then least DD
    consistent.sort(key=lambda c: (not c[3], c[4], -c[1].user_attrs["net_rs"], c[1].user_attrs["max_dd"]))
    top5 = consistent[:5]

    print("\n==========  TOP 5  CASH MACHINES  (max net / least DD / positive every month)  ==========")
    out_rows = []
    for rank, (i, t, nets, ok, n_neg) in enumerate(top5, 1):
        a = t.user_attrs
        p = t.params
        yr = {}
        for mo, v in nets.items():
            yr.setdefault(mo[:4], 0.0)
            yr[mo[:4]] += v
        tag = "[ALL-MONTHS +]" if ok else f"[{n_neg} NEG MONTHS]"
        row = (f"\n#{rank}  {tag}\n"
               f"   NW net=Rs{a['net_rs']:+,.0f}  DD=Rs{a['max_dd']:,.0f}  PF={a['pf']:.2f}  "
               f"T={int(a['trades'])} ({a['trades']/N_TOTAL:.2f}/day)  WR={a['win_rate']:.1f}%\n"
               f"   params: ck={p['confirm_k']} sl={p['sl_m']:.1f} tp={p['tp_m']:.1f} atr={p['atr_p']} "
               f"dl={p['daily_loss_pts']} mn={p['moneyness']} re={p['reentry']} relax={p['band_relax']:.0f} "
               f"sess=[+{p['sess_start_off']},-{p['sess_end_off']}]\n"
               f"   per-year net: " + "  ".join(f"{y}:Rs{v:,.0f}" for y, v in sorted(yr.items())))
        print(row, flush=True)
        oos = ens.evaluate_ensemble_batch([(COMPONENTS, _ens_from_params(p))], base.d_oos_mask)[0]
        print(f"   OOS(2024-26): net=Rs{oos['net_rs']:+,.0f}  WR={oos['win_rate']:.1f}%  "
              f"PF={oos['pf']:.2f}  DD=Rs{oos['max_dd']:,.0f}  T={oos['trades']}", flush=True)
        out_rows.append({"rank": rank, "all_months_positive": ok, "neg_months": n_neg,
                      "nw": {k: a.get(k) for k in ["net_rs", "max_dd", "pf", "trades", "win_rate"]},
                      "params": dict(p), "per_year_net": yr,
                      "oos": {k: oos.get(k) for k in ["net_rs", "win_rate", "pf", "max_dd", "trades"]}})

    json.dump({"objective": "max net_rs - 20*max_dd; require profitable every month 2020-01..2026-05",
               "n_trials": n_trials, "n_components": N_COMP, "n_months": len(MONTHS),
               "n_all_months_positive": n_allpos, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "top5": out_rows},
              open(HERE / "optimus_cash_machine.json", "w"), indent=2, default=str)
    print(f"\nSaved: {HERE / 'optimus_cash_machine.json'}", flush=True)


if __name__ == "__main__":
    run()
