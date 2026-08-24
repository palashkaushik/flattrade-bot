"""
OPTIMUS HFT CASH-MACHINE SWEEP  —  HIGH FREQUENCY, MAX NET / LEAST DD / ALL MONTHS +
===================================================================================
Re-engineers the project's BEST strategies into HFT scalping mechanisms (web-research
derived) and backtests them on the Optimus ensemble engine.

HFT mechanisms applied (from web research):
  - reentry=True ............... multiple entries per day (scalp, 5-10+/session)
  - wide band_relax ............ divergence fires far more often
  - fast 1m stochastic comps ... %K 5-9 (scalping settings)
  - tight SL + small TP ........ R:R ~2 (SL 1.5xATR / TP 3xATR)
  - session time filter ........ skip 9:15-9:50 open & last 30 min (volatility traps)
  - daily_loss_pts cap ......... bounds bleed on a high-frequency day
  - cross-TF component voting ... 1/2/3/5m = higher-timeframe confirmation filter

OBJECTIVE: maximize NET POINTS with LEAST DRAWDOWN, CONSISTENT (positive EVERY month),
subject to a HARD minimum-frequency gate (>= MIN_PER_DAY trades/day = real HFT).
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
N_TOTAL = N_DAYS
MIN_PER_DAY = int(os.environ.get("MIN_PER_DAY", "5"))   # target trades/day (relaxable)
MIN_TRADES = MIN_PER_DAY * N_TOTAL
TREND = int(os.environ.get("TREND", "0"))   # 0=none, 5=5m HA UT Bot, 15=15m HA UT Bot
TREND_LABEL = {0: "NO FILTER (baseline)", 5: "5m HA UT Bot", 15: "15m HA UT Bot"}[TREND]
USE_VOL = int(os.environ.get("USE_VOL", "1"))  # 1=vol-regime gate ON, 0=OFF (ablation)
USE_MARNI = int(os.environ.get("USE_MARNI", "0"))  # 1=tune UT Bot ATR mult / HA period / linlen

def _comp(tf, s1k, s4k, s4ob, s1os, tag):
    return {"strat_id": "B07", "timeframe": tf, "s1_k": s1k, "s4_k": s4k,
            "s1_os": s1os, "s4_ob": s4ob, "atr_p": 10,
            "sl_m": 2.0, "tp_m": 4.0, "daily_loss_pts": 10, "daily_profit_pts": 50,
            "moneyness": 0.5, "max_trade_loss_rs": 9999,
            "sess_start_off": 5, "sess_end_off": 45, "sess_end": 300, "tag": tag}

# PROVEN performers (kept) + HFT-fast 1m scalper variants (fast %K, looser thresholds)
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
    # HFT-fast scalper variants (derived from the winners)
    _comp(1,  9, 20, 72.0, 28.0, "HFT_A_fast"),
    _comp(1,  5, 20, 68.0, 32.0, "HFT_B_ultra"),
    _comp(1, 12, 30, 75.0, 25.0, "HFT_C_fast"),
    _comp(1,  7, 30, 70.0, 30.0, "HFT_D_bfast"),
]
N_COMP = len(COMPONENTS)
print(f"Components loaded: {N_COMP} (proven+HFT-fast)  (days={N_DAYS}, min_trades={MIN_TRADES})", flush=True)

YEARS = sorted(set(d[:4] for d in base.all_days))
YMASK = [(y, torch.tensor([d.startswith(y) for d in base.all_days], dtype=torch.bool, device=device)) for y in YEARS]
MONTHS = sorted(set(d[:7] for d in base.all_days))
MMASK = [(mo, torch.tensor([d.startswith(mo) for d in base.all_days], dtype=torch.bool, device=device)) for mo in MONTHS]
print(f"Consistency: {len(YEARS)} yrs, {len(MONTHS)} months", flush=True)


def suggest_ens(trial):
    return {
        "timeframe": 1, "s1_k": 13, "s4_k": 70, "s1_os": 15.0, "s4_ob": 77.5,
        "atr_p": trial.suggest_int("atr_p", 8, 30),
        "sl_m": trial.suggest_float("sl_m", 1.0, 4.0, step=0.1),     # tight scalp SL
        "tp_m": trial.suggest_float("tp_m", 2.0, 8.0, step=0.25),     # small scalp TP (R:R~2)
        "daily_loss_pts": trial.suggest_int("daily_loss_pts", 5, 30, step=5),
        "daily_profit_pts": 60,
        "moneyness": trial.suggest_categorical("moneyness", [0.5, 0.6]),
        "max_trade_loss_rs": 9999,
        "sess_start_off": trial.suggest_int("sess_start_off", 15, 45, step=5),   # skip 9:15-9:50
        "sess_end_off": trial.suggest_int("sess_end_off", 30, 60, step=15),      # end by ~3:00
        "sess_end": 375 - trial.suggest_int("sess_end_off", 30, 60, step=15),
        "confirm_k": trial.suggest_int("confirm_k", 1, N_COMP),
        "band_relax": trial.suggest_float("band_relax", 0.0, 25.0, step=1.0),    # wide bands = freq
        "reentry": True,                                                       # HFT: re-enter same day
        "trend_filter": TREND,                                                 # Marni HA UT Bot gate
        "marni": None if (USE_MARNI == 0 or TREND == 0) else {
            "key": trial.suggest_float("marni_key", 0.6, 2.0, step=0.1),    # UT Bot ATR stop multiple
            "period": trial.suggest_int("marni_period", 5, 20),             # Wilder ATR period
            "linlen": trial.suggest_int("marni_linlen", 5, 20),             # HA linreg length
        },
        # Volatility-regime gate (web-validated whipsaw guard): trade only when the
        # 1-min ATR percentile rank in [lo,hi]; exclude chop (low) and exhaustion (high)
        "vol_filter": None if USE_VOL == 0 else {
            "atr_p": trial.suggest_int("vf_atrp", 14, 40),
            "lookback": trial.suggest_int("vf_lb", 30, 120, step=10),
            "lo": trial.suggest_float("vf_lo", 0.0, 40.0, step=5.0),
            "hi": trial.suggest_float("vf_hi", 60.0, 100.0, step=5.0),
        },
    }


def _ens_from_params(p):
    vf = p.get("vol_filter")
    if vf is not None:
        vf = {"atr_p": int(vf["atr_p"]), "lookback": int(vf["lookback"]),
              "lo": float(vf["lo"]), "hi": float(vf["hi"])}
    return {"timeframe": 1, "s1_k": 13, "s4_k": 70, "s1_os": 15.0, "s4_ob": 77.5,
            "atr_p": int(p["atr_p"]), "sl_m": float(p["sl_m"]), "tp_m": float(p["tp_m"]),
            "daily_loss_pts": int(p["daily_loss_pts"]), "daily_profit_pts": 60,
            "moneyness": float(p["moneyness"]), "max_trade_loss_rs": 9999,
            "sess_start_off": int(p["sess_start_off"]), "sess_end_off": int(p["sess_end_off"]),
            "sess_end": 375 - int(p["sess_end_off"]),             "confirm_k": int(p["confirm_k"]),
            "band_relax": float(p["band_relax"]), "reentry": True,
            "trend_filter": TREND, "vol_filter": vf, "marni": (p.get("marni") if p.get("marni") else None)}


WR_FLOOR = float(os.environ.get("WR_FLOOR", "55.0"))  # (web: scalping needs >=55%; relax if edge is thinner)
def profit_score(r):
    net = float(r.get("net_rs", 0.0))
    dd = float(r.get("max_dd", 0.0))
    tr = int(r.get("trades", 0))
    wr = float(r.get("win_rate", 0.0))
    if net <= 0 or tr < MIN_TRADES or wr < WR_FLOOR:   # profitable + HFT + high-WR
        return -1e9
    return net - 20.0 * dd


TRIAL_CSV = HERE / f"optimus_trials_trend{TREND}_vol{USE_VOL}.csv"
_TRIAL_FIELDS = ["trial", "objective", "net_rs", "max_dd", "pf", "trades", "win_rate",
                "oos_net", "oos_wr",
                "confirm_k", "sl_m", "tp_m", "atr_p", "daily_loss_pts", "moneyness",
                "band_relax", "sess_start_off", "sess_end_off", "reentry",
                "trend_filter", "vf_atrp", "vf_lb", "vf_lo", "vf_hi"]


def _log_trial(t, res, objective=None, oos=None):
    new = not TRIAL_CSV.exists()
    import csv as _csv
    with open(TRIAL_CSV, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=_TRIAL_FIELDS)
        if new:
            w.writeheader()
        p = t.params
        row = {"trial": t.number, "objective": objective,
               "net_rs": res.get("net_rs"), "max_dd": res.get("max_dd"),
               "pf": res.get("pf"), "trades": res.get("trades"), "win_rate": res.get("win_rate"),
               "oos_net": (oos or {}).get("net_rs"), "oos_wr": (oos or {}).get("win_rate"),
               "confirm_k": p.get("confirm_k"), "sl_m": p.get("sl_m"), "tp_m": p.get("tp_m"),
               "atr_p": p.get("atr_p"), "daily_loss_pts": p.get("daily_loss_pts"),
               "moneyness": p.get("moneyness"), "band_relax": p.get("band_relax"),
               "sess_start_off": p.get("sess_start_off"), "sess_end_off": p.get("sess_end_off"),
               "reentry": p.get("reentry"), "trend_filter": p.get("trend_filter"),
               "vf_atrp": (p.get("vol_filter") or {}).get("atr_p"),
               "vf_lb": (p.get("vol_filter") or {}).get("lookback"),
               "vf_lo": (p.get("vol_filter") or {}).get("lo"),
               "vf_hi": (p.get("vol_filter") or {}).get("hi")}
        w.writerow(row)


def run():
    n_trials = int(os.environ.get("TRIALS", "2500"))
    bs = int(os.environ.get("BATCH", "100"))
    print(f"\n=== HFT STUDY [{TREND_LABEL}]: {n_trials} trials (NW) ===", flush=True)
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
            val = profit_score(res)
            study.tell(t, val)
            _log_trial(t, res, objective=val)
    hft = [t for t in study.trials if t.value is not None and t.value > -1e8]
    print(f"NW study done in {time.time()-t0:.0f}s; HFT-passing candidates: {len(hft)}", flush=True)

    # pool: top by profit + top by WR (high-WR helps survive HFT chop)
    by_profit = sorted(hft, key=lambda t: t.value, reverse=True)[:300]
    by_wr = sorted(hft, key=lambda t: t.user_attrs.get("win_rate", 0), reverse=True)[:300]
    seen, pool = set(), []
    for t in by_profit + by_wr:
        if id(t) not in seen:
            seen.add(id(t)); pool.append(t)
    print(f"Consistency pool: {len(pool)}; checking EVERY MONTH...", flush=True)

    pairs = [(COMPONENTS, _ens_from_params(t.params)) for t in pool]
    per_month = [dict() for _ in pool]
    CH = 80
    for mo, m in MMASK:
        for s in range(0, len(pairs), CH):
            rl = ens.evaluate_ensemble_batch(pairs[s:s + CH], m)
            for j, r in enumerate(rl):
                per_month[s + j][mo] = float(r.get("net_rs", 0.0))

    consistent = []
    for i, t in enumerate(pool):
        nets = per_month[i]
        n_neg = sum(1 for v in nets.values() if v <= 0)
        consistent.append((i, t, nets, n_neg == 0, n_neg))
    n_allpos = sum(1 for c in consistent if c[3])
    print(f"  candidates positive EVERY month: {n_allpos}/{len(consistent)}", flush=True)

    # OOS pass for the WHOLE pool (robustness: IS vs OOS consistency / WFC)
    pool_pairs = [(COMPONENTS, _ens_from_params(t.params)) for t in pool]
    pool_oos = [None] * len(pool)
    for s in range(0, len(pool_pairs), CH):
        rl = ens.evaluate_ensemble_batch(pool_pairs[s:s + CH], base.d_oos_mask)
        for j, r in enumerate(rl):
            pool_oos[s + j] = r
    import csv as _csv
    oos_csv = HERE / f"optimus_pool_oos_trend{TREND}_vol{USE_VOL}.csv"
    with open(oos_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["trial", "is_net", "oos_net", "is_wr", "oos_wr"])
        for t, r in zip(pool, pool_oos):
            w.writerow([t.number, t.user_attrs.get("net_rs"), r.get("net_rs"),
                        t.user_attrs.get("win_rate"), r.get("win_rate")])
    print(f"  Saved pool OOS consistency: {oos_csv}", flush=True)

    consistent.sort(key=lambda c: (not c[3], c[4], -c[1].user_attrs["net_rs"], c[1].user_attrs["max_dd"]))
    top5 = consistent[:5]

    print(f"\n======  TOP 5  [{TREND_LABEL}]  (max net / least DD / all months +)  ======")
    out_rows = []
    for rank, (i, t, nets, ok, n_neg) in enumerate(top5, 1):
        a = t.user_attrs; p = t.params
        yr = {}
        for mo, v in nets.items():
            yr[mo[:4]] = yr.get(mo[:4], 0.0) + v
        tag = "[ALL-MONTHS +]" if ok else f"[{n_neg} NEG MONTHS]"
        print(f"\n#{rank}  {tag}\n"
              f"   NW net=Rs{a['net_rs']:+,.0f}  DD=Rs{a['max_dd']:,.0f}  PF={a['pf']:.2f}  "
              f"T={int(a['trades'])} ({a['trades']/N_TOTAL:.2f}/day)  WR={a['win_rate']:.1f}%\n"
              f"   params: ck={p['confirm_k']} sl={p['sl_m']:.1f} tp={p['tp_m']:.1f} atr={p['atr_p']} "
              f"dl={p['daily_loss_pts']} mn={p['moneyness']} relax={p['band_relax']:.0f} "
                f"sess=[+{p['sess_start_off']},-{p['sess_end_off']}]", flush=True)
        if p.get("vol_filter"):
            vf = p["vol_filter"]
            print(f"   vol=[atrp={vf['atr_p']} lb={vf['lookback']} "
                  f"{vf['lo']:.0f}-{vf['hi']:.0f}]", flush=True)
        print(f"   per-year net: " + "  ".join(f"{y}:Rs{v:,.0f}" for y, v in sorted(yr.items())), flush=True)
        oos = pool_oos[i]
        print(f"   OOS(2024-26): net=Rs{oos['net_rs']:+,.0f}  WR={oos['win_rate']:.1f}%  "
              f"PF={oos['pf']:.2f}  DD=Rs{oos['max_dd']:,.0f}  T={oos['trades']} "
              f"({oos['trades']/611:.1f}/day)", flush=True)
        out_rows.append({"rank": rank, "all_months_positive": ok, "neg_months": n_neg,
                      "nw": {k: a.get(k) for k in ["net_rs", "max_dd", "pf", "trades", "win_rate"]},
                      "params": dict(p), "per_year_net": yr,
                      "oos": {k: oos.get(k) for k in ["net_rs", "win_rate", "pf", "max_dd", "trades"]}})

    json.dump({"objective": f"max net_rs - 20*max_dd; WR>={WR_FLOOR}; min {MIN_PER_DAY}/day; all months positive",
               "trend_filter": TREND_LABEL, "use_vol": USE_VOL, "n_trials": n_trials, "n_components": N_COMP,
               "min_per_day": MIN_PER_DAY, "wr_floor": WR_FLOOR,
               "n_months": len(MONTHS), "n_all_months_positive": n_allpos,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "top5": out_rows},
              open(HERE / f"optimus_hft_trend{TREND}_vol{USE_VOL}_f{MIN_PER_DAY}_wr{WR_FLOOR}.json", "w"), indent=2, default=str)
    print(f"\nSaved: {HERE / f'optimus_hft_trend{TREND}_vol{USE_VOL}_f{MIN_PER_DAY}_wr{WR_FLOOR}.json'}", flush=True)


if __name__ == "__main__":
    run()
