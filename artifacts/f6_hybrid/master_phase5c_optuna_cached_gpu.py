"""
PHASE 5c — OPTUNA TPE + PRE-CACHED INDICATORS + 3000 TRIALS/STRATEGY
=====================================================================
Combines:
  - Optuna TPE sampler (smart search, not random) from Phase 5
  - Pre-computed indicator lookup tables from Phase 5b (no recomputation)
  - 3000 trials per strategy (20K+ total)
  - 7 strategy families (B01-B07)
  - Per-trade SL cap + bidirectional CE+PE + multi-TF
"""

import json, sys, time
from pathlib import Path
import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opt_futures_quad as source

torch.set_float32_matmul_precision("high")
LOT_SIZE = 65
FEE = 30.0
BASE_SESSION_START = 5
BASE_SESSION_END = 345
TRIALS_PER_STRATEGY = 3000
BATCH_SIZE = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optuna.logging.set_verbosity(optuna.logging.WARNING)
print(f"CUDA: {device} ({torch.cuda.get_device_name(0)})", flush=True)

# ─── Load data ──────────────────────────────────────────────────────────────
def load_gpu_data():
    spot_all = source.load_spot()
    opt_map = source.option_day_files("2020-01-01", "2026-05-05")
    days = sorted(set(opt_map.keys()) & set(spot_all.keys()))
    N = len(days)
    arr_h = np.zeros((N, 375), dtype=np.float32)
    arr_l = np.zeros((N, 375), dtype=np.float32)
    arr_c = np.zeros((N, 375), dtype=np.float32)
    for i, d in enumerate(days):
        sp = spot_all[d]
        for idx, m in enumerate(sp["min"]):
            b = int(m) - 555
            if 0 <= b < 375:
                arr_h[i, b] = float(sp["high"][idx])
                arr_l[i, b] = float(sp["low"][idx])
                arr_c[i, b] = float(sp["close"][idx])
    d_h = torch.tensor(arr_h, dtype=torch.float32, device=device)
    d_l = torch.tensor(arr_l, dtype=torch.float32, device=device)
    d_c = torch.tensor(arr_c, dtype=torch.float32, device=device)
    is_mask = torch.tensor([d < "2024-01-01" for d in days], dtype=torch.bool, device=device)
    oos_mask = torch.tensor([d >= "2024-01-01" for d in days], dtype=torch.bool, device=device)
    return d_h, d_l, d_c, days, is_mask, oos_mask

print("Loading data...", flush=True)
t0 = time.time()
d_high, d_low, d_close, all_days, d_is_mask, d_oos_mask = load_gpu_data()
N_DAYS, T_BARS = d_close.shape
prev_c = F.pad(d_close[:, :-1], (1, 0), mode="replicate")
d_tr = torch.maximum(torch.maximum(d_high - d_low, torch.abs(d_high - prev_c)), torch.abs(d_low - prev_c))
print(f"Loaded {N_DAYS} days in {time.time()-t0:.1f}s", flush=True)

# ─── Multi-TF aggregation ──────────────────────────────────────────────────
@torch.no_grad()
def aggregate_tf(k):
    if k == 1:
        return d_high, d_low, d_close, d_tr
    N, T = d_high.shape
    pad = (k - T % k) % k
    h_r = F.pad(d_high, (0, pad), mode="replicate").reshape(N, -1, k).max(dim=2).values
    l_r = F.pad(d_low, (0, pad), mode="replicate").reshape(N, -1, k).min(dim=2).values
    c_r = F.pad(d_close, (0, pad), mode="replicate").reshape(N, -1, k)[:, :, -1]
    pc = F.pad(c_r[:, :-1], (1, 0), mode="replicate")
    tr_r = torch.maximum(torch.maximum(h_r - l_r, torch.abs(h_r - pc)), torch.abs(l_r - pc))
    return h_r, l_r, c_r, tr_r

TF_DATA = {}
for tf in [1, 2, 3, 5]:
    TF_DATA[tf] = aggregate_tf(tf)

# ─── Pre-compute indicator lookup tables ────────────────────────────────────
print("Pre-computing indicators...", flush=True)
t1 = time.time()
STOCH_CACHE = {}
ATR_CACHE = {}

@torch.no_grad()
def get_stoch(tf, period):
    key = (tf, period)
    if key not in STOCH_CACHE:
        h, l, c, _ = TF_DATA[tf]
        h_pad = F.pad(h.unsqueeze(1), (period - 1, 0), mode="replicate")
        l_pad = F.pad(l.unsqueeze(1), (period - 1, 0), mode="replicate")
        max_h = F.max_pool1d(h_pad, kernel_size=period, stride=1).squeeze(1)
        min_l = -F.max_pool1d(-l_pad, kernel_size=period, stride=1).squeeze(1)
        denom = (max_h - min_l).clamp(min=1e-6)
        STOCH_CACHE[key] = ((c - min_l) / denom) * 100.0
    return STOCH_CACHE[key]

@torch.no_grad()
def get_atr(tf, period):
    key = (tf, period)
    if key not in ATR_CACHE:
        _, _, _, tr = TF_DATA[tf]
        tr_pad = F.pad(tr.unsqueeze(1), (period - 1, 0), mode="replicate")
        ATR_CACHE[key] = F.avg_pool1d(tr_pad, kernel_size=period, stride=1).squeeze(1)
    return ATR_CACHE[key]

# Pre-warm all periods
for tf in [1, 2, 3, 5]:
    for sk in range(5, 31):
        get_stoch(tf, sk)
    for sk in range(20, 121, 5):
        get_stoch(tf, sk)
    for ap in range(8, 36):
        get_atr(tf, ap)
print(f"  Cached {len(STOCH_CACHE)+len(ATR_CACHE)} tensors in {time.time()-t1:.1f}s", flush=True)

# ─── Simulation Engine ─────────────────────────────────────────────────────
@torch.no_grad()
def simulate_direction(entries_mask, sl_tensor, tp_tensor, direction, max_daily_loss, sess_end, day_mask=None):
    if day_mask is not None:
        entries_mask = entries_mask & day_mask.unsqueeze(1)

    coords = torch.nonzero(entries_mask, as_tuple=False)
    if coords.shape[0] == 0:
        return {"trades": 0, "win_rate": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}
    coords = coords[:8000]
    d_idx = coords[:, 0]; b_idx = coords[:, 1]
    ep = d_close[d_idx, b_idx]

    max_future = sess_end - BASE_SESSION_START
    col_off = torch.arange(max_future, device=device).unsqueeze(0)
    col_idx = (b_idx + 1).unsqueeze(1) + col_off
    valid = (col_idx < sess_end) & (col_idx < 375)
    col_safe = col_idx.clamp(max=374)
    d_exp = d_idx.unsqueeze(1).expand(-1, max_future)

    fut_h = d_high[d_exp, col_safe]
    fut_l = d_low[d_exp, col_safe]
    INF = 1e9
    fut_h_m = torch.where(valid, fut_h, -INF)
    fut_l_m = torch.where(valid, fut_l, INF)

    sl_p = sl_tensor[d_idx, b_idx]; tp_p = tp_tensor[d_idx, b_idx]

    if direction == "CE":
        hit_sl = fut_l_m <= sl_p.unsqueeze(1)
        hit_tp = fut_h_m >= tp_p.unsqueeze(1)
    else:
        hit_sl = fut_h_m >= sl_p.unsqueeze(1)
        hit_tp = fut_l_m <= tp_p.unsqueeze(1)

    BIG = 999999
    sl_any = hit_sl.any(dim=1); tp_any = hit_tp.any(dim=1)
    sl_first = torch.where(sl_any, torch.argmax(hit_sl.int(), dim=1), BIG)
    tp_first = torch.where(tp_any, torch.argmax(hit_tp.int(), dim=1), BIG)
    sl_exits = sl_any & (sl_first <= tp_first)
    tp_exits = tp_any & (~sl_exits)

    eod_bar = min(sess_end - 1, 374)
    eod_px = d_close[d_idx, eod_bar]

    if direction == "CE":
        exit_px = torch.where(sl_exits, sl_p, torch.where(tp_exits, tp_p, eod_px))
        raw_pts = (exit_px - ep) * 0.50
    else:
        exit_px = torch.where(sl_exits, sl_p, torch.where(tp_exits, tp_p, eod_px))
        raw_pts = (ep - exit_px) * 0.50

    has_future = (b_idx + 1) < sess_end
    raw_pts = raw_pts[has_future]; d_idx_v = d_idx[has_future]
    if raw_pts.shape[0] == 0:
        return {"trades": 0, "win_rate": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    all_rs = raw_pts * LOT_SIZE - FEE
    all_rs_cpu = all_rs.cpu().numpy(); d_idx_cpu = d_idx_v.cpu().numpy()

    daily_pnl = {}
    keep = np.ones(len(all_rs_cpu), dtype=bool)
    for k in range(len(all_rs_cpu)):
        di = int(d_idx_cpu[k])
        cum = daily_pnl.get(di, 0.0)
        if cum <= -max_daily_loss:
            keep[k] = False; continue
        daily_pnl[di] = cum + all_rs_cpu[k]
    final_rs = all_rs_cpu[keep]
    if len(final_rs) == 0:
        return {"trades": 0, "win_rate": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    wins = int((final_rs > 0).sum()); n = len(final_rs)
    pos = float(final_rs[final_rs > 0].sum())
    neg = float(abs(final_rs[final_rs <= 0].sum()))
    pf = pos / neg if neg > 0 else 0.0
    eq = np.cumsum(final_rs); pk = np.maximum.accumulate(eq)
    dd = float(np.max(pk - eq))
    return {"trades": n, "win_rate": round(wins/n*100, 2), "net_rs": round(float(final_rs.sum()), 2),
            "pf": round(pf, 2), "max_dd": round(dd, 2)}


def merge_results(ce, pe):
    t = ce["trades"] + pe["trades"]
    if t == 0:
        return {"trades": 0, "win_rate": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0,
                "ce_trades": 0, "pe_trades": 0, "ce_pnl": 0.0, "pe_pnl": 0.0}
    net = ce["net_rs"] + pe["net_rs"]
    ce_w = int(ce["trades"] * ce["win_rate"] / 100)
    pe_w = int(pe["trades"] * pe["win_rate"] / 100)
    wr = (ce_w + pe_w) / t * 100.0
    ce_pos = max(ce["net_rs"], 0) * (ce["pf"]/(1+ce["pf"]) if ce["pf"] > 0 else 1)
    pe_pos = max(pe["net_rs"], 0) * (pe["pf"]/(1+pe["pf"]) if pe["pf"] > 0 else 1)
    tot_pos = ce_pos + pe_pos; tot_neg = tot_pos - net
    pf = tot_pos / tot_neg if tot_neg > 0 else 0.0
    dd = max(ce["max_dd"], pe["max_dd"])
    return {"trades": t, "win_rate": round(wr, 2), "net_rs": round(net, 2),
            "pf": round(pf, 2), "max_dd": round(dd, 2),
            "ce_trades": ce["trades"], "pe_trades": pe["trades"],
            "ce_pnl": ce["net_rs"], "pe_pnl": pe["net_rs"]}


# ─── Strategy Builder ──────────────────────────────────────────────────────
def build_and_eval(strat_id, trial, day_mask=None):
    daily_loss_pts = trial.suggest_int("daily_loss_pts", 3, 20)
    daily_loss_rs = float(daily_loss_pts) * LOT_SIZE
    max_trade_loss = trial.suggest_categorical("max_trade_loss_rs", [500, 1000, 1500, 2000, 3000, 5000, 9999])
    sess_start_off = trial.suggest_int("sess_start_off", 0, 30, step=5)
    sess_end_off = trial.suggest_int("sess_end_off", 30, 75, step=15)
    sess_end = BASE_SESSION_END - sess_end_off

    if strat_id in ("B01", "B02", "B06"):
        tf = 1
    elif strat_id == "B03":
        tf = 2
    elif strat_id == "B04":
        tf = 3
    elif strat_id == "B05":
        tf = 5
    elif strat_id == "B07":
        tf = trial.suggest_categorical("timeframe", [1, 2, 3, 5])
    else:
        tf = 1

    s1_k = trial.suggest_int("s1_k", 5, 30)
    s4_k = trial.suggest_int("s4_k", 20, 120, step=5)
    s1_os = trial.suggest_float("s1_os", 10.0, 40.0, step=2.5)
    s4_ob = trial.suggest_float("s4_ob", 65.0, 90.0, step=2.5)
    atr_p = trial.suggest_int("atr_p", 8, 35)
    sl_m = trial.suggest_float("sl_m", 1.0, 5.0, step=0.1)
    tp_m = trial.suggest_float("tp_m", 2.0, 10.0, step=0.25)
    if tp_m < 1.5 * sl_m:
        raise optuna.TrialPruned("R:R constraint")

    # Lookup pre-cached indicators (NO recomputation!)
    S1 = get_stoch(tf, s1_k)
    S4 = get_stoch(tf, s4_k)
    ATR = get_atr(tf, atr_p)

    # Map to 1m
    if tf > 1:
        S1_1m = S1.repeat_interleave(tf, dim=1)[:, :T_BARS]
        S4_1m = S4.repeat_interleave(tf, dim=1)[:, :T_BARS]
        ATR_1m = ATR.repeat_interleave(tf, dim=1)[:, :T_BARS]
    else:
        S1_1m = S1; S4_1m = S4; ATR_1m = ATR

    vw = torch.zeros((N_DAYS, T_BARS), dtype=torch.bool, device=device)
    vw[:, BASE_SESSION_START + sess_start_off:sess_end] = True

    # Trade SL cap filter
    sl_dist_rs = ATR_1m * sl_m * 0.50 * LOT_SIZE
    trade_ok = sl_dist_rs <= max_trade_loss

    # CE entries
    ce_ent = (S4_1m >= s4_ob) & (S1_1m <= s1_os) & vw & trade_ok
    ce_sl = d_close - ATR_1m * sl_m
    ce_tp = d_close + ATR_1m * tp_m
    ce_res = simulate_direction(ce_ent, ce_sl, ce_tp, "CE", daily_loss_rs, sess_end, day_mask)

    is_bidir = strat_id != "B01"
    if is_bidir:
        pe_s4_os = 100.0 - s4_ob
        pe_s1_ob = 100.0 - s1_os
        pe_ent = (S4_1m <= pe_s4_os) & (S1_1m >= pe_s1_ob) & vw & trade_ok
        pe_sl = d_close + ATR_1m * sl_m
        pe_tp = d_close - ATR_1m * tp_m
        pe_res = simulate_direction(pe_ent, pe_sl, pe_tp, "PE", daily_loss_rs, sess_end, day_mask)
        res = merge_results(ce_res, pe_res)
    else:
        res = ce_res
        res["ce_trades"] = res["trades"]; res["pe_trades"] = 0
        res["ce_pnl"] = res["net_rs"]; res["pe_pnl"] = 0.0

    n_tr = res["trades"]
    if n_tr < 50 or res["net_rs"] <= 0:
        return -999.0, res

    pf_comp = res["pf"] * (res["win_rate"] / 40.0)
    dd_pen = 0.50 * (res["max_dd"] / max(res["net_rs"], 1.0))
    freq = min(n_tr / 500.0, 1.0) * 0.10
    score = pf_comp - dd_pen + freq
    if strat_id in ("B06", "B07") and res["max_dd"] > 50000:
        score -= (res["max_dd"] - 50000) / 50000 * 0.5
    return score, res


# ─── Strategy Definitions ──────────────────────────────────────────────────
STRATS = [
    ("B01", "B01: 1m CE-Only (Baseline)"),
    ("B02", "B02: 1m CE+PE Bidirectional"),
    ("B03", "B03: 2m CE+PE Bidirectional"),
    ("B04", "B04: 3m CE+PE Bidirectional"),
    ("B05", "B05: 5m CE+PE Bidirectional"),
    ("B06", "B06: 1m CE+PE Tight DD"),
    ("B07", "B07: Best-TF CE+PE DD Target"),
]


def run_strategy(sid, sname, idx, total):
    print(f"\n[{idx:02d}/{total}] {sname}", flush=True)
    t0 = time.time()

    # NW
    study_nw = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42, constant_liar=True))
    for _ in range(TRIALS_PER_STRATEGY // BATCH_SIZE):
        batch = [study_nw.ask() for _ in range(BATCH_SIZE)]
        for trial in batch:
            try:
                sc, res = build_and_eval(sid, trial, day_mask=None)
            except optuna.TrialPruned:
                sc = -999.0; res = {}
            for k, v in res.items():
                trial.set_user_attr(k, v)
            study_nw.tell(trial, sc)
    t_nw = time.time() - t0

    # WF IS
    t1 = time.time()
    study_wf = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42, constant_liar=True))
    for _ in range(TRIALS_PER_STRATEGY // BATCH_SIZE):
        batch = [study_wf.ask() for _ in range(BATCH_SIZE)]
        for trial in batch:
            try:
                sc, res = build_and_eval(sid, trial, day_mask=d_is_mask)
            except optuna.TrialPruned:
                sc = -999.0; res = {}
            for k, v in res.items():
                trial.set_user_attr(k, v)
            study_wf.tell(trial, sc)
    t_wf = time.time() - t1

    # OOS eval
    fixed = optuna.trial.FixedTrial(study_wf.best_trial.params)
    try:
        _, oos = build_and_eval(sid, fixed, day_mask=d_oos_mask)
    except:
        oos = {"trades": 0, "win_rate": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0,
               "ce_trades": 0, "pe_trades": 0, "ce_pnl": 0.0, "pe_pnl": 0.0}

    nw = study_nw.best_trial.user_attrs
    is_ann = study_wf.best_trial.user_attrs.get("net_rs", 0) / 4.0
    oos_ann = oos.get("net_rs", 0) / 2.35
    wfe = round(oos_ann / is_ann, 2) if is_ann > 0 else 0.0

    print(f"  NW {t_nw:.0f}s: PnL=Rs {nw.get('net_rs',0):+,.0f} WR={nw.get('win_rate',0):.1f}% PF={nw.get('pf',0):.2f} DD=Rs {nw.get('max_dd',0):,.0f} T={nw.get('trades',0)} CE={nw.get('ce_trades',0)} PE={nw.get('pe_trades',0)}", flush=True)
    print(f"  WF {t_wf:.0f}s: IS=Rs {study_wf.best_trial.user_attrs.get('net_rs',0):+,.0f} -> OOS=Rs {oos['net_rs']:+,.0f} PF={oos['pf']:.2f} WR={oos['win_rate']:.1f}% DD=Rs {oos['max_dd']:,.0f} WFE={wfe:.2f}", flush=True)
    print(f"  NW params: {study_nw.best_trial.params}", flush=True)
    print(f"  WF params: {study_wf.best_trial.params}", flush=True)

    return {
        "id": sid, "name": sname,
        "non_wf": {"best_params": study_nw.best_trial.params, **{k: nw.get(k, 0) for k in
                    ["win_rate","pf","net_rs","max_dd","trades","ce_trades","pe_trades","ce_pnl","pe_pnl"]}},
        "walk_forward": {"is_params": study_wf.best_trial.params,
                         "is_net_rs": study_wf.best_trial.user_attrs.get("net_rs", 0),
                         "oos_net_rs": oos["net_rs"], "oos_pf": oos["pf"],
                         "oos_wr": oos["win_rate"], "oos_max_dd": oos["max_dd"],
                         "oos_trades": oos.get("trades", 0),
                         "oos_ce_trades": oos.get("ce_trades", 0), "oos_pe_trades": oos.get("pe_trades", 0),
                         "oos_ce_pnl": oos.get("ce_pnl", 0), "oos_pe_pnl": oos.get("pe_pnl", 0),
                         "wfe": wfe},
    }


def main():
    total = len(STRATS)
    total_trials = total * TRIALS_PER_STRATEGY * 2
    print(f"\n{'='*140}", flush=True)
    print(f"PHASE 5c: OPTUNA TPE + CACHED INDICATORS | {total} strategies x {TRIALS_PER_STRATEGY} trials x 2 = {total_trials:,} GPU evals", flush=True)
    print(f"{'='*140}", flush=True)

    t_start = time.time()
    results = [run_strategy(sid, sname, i+1, total) for i, (sid, sname) in enumerate(STRATS)]
    total_time = time.time() - t_start

    by_nw = sorted(results, key=lambda x: x["non_wf"]["net_rs"], reverse=True)
    by_oos = sorted(results, key=lambda x: x["walk_forward"]["oos_net_rs"], reverse=True)

    print(f"\n{'='*160}", flush=True)
    print(f"PHASE 5c LEADERBOARD ({total_trials:,} TRIALS IN {total_time:.1f}s = {total_trials/total_time:.0f} trials/s)", flush=True)
    print(f"{'='*160}", flush=True)

    nw_rank = {r["id"]: i+1 for i, r in enumerate(by_nw)}
    oos_rank = {r["id"]: i+1 for i, r in enumerate(by_oos)}

    print(f"\n{'NW#':4s} {'OOS#':5s} {'Strategy':42s} {'NW PnL':>14s} {'NW PF':>7s} {'NW WR':>7s} {'NW DD':>10s} {'OOS PnL':>14s} {'OOS PF':>7s} {'OOS WR':>7s} {'OOS DD':>10s} {'WFE':>5s} {'CE/PE':>12s}", flush=True)
    print("-" * 160, flush=True)
    for r in by_nw:
        nw = r["non_wf"]; wf = r["walk_forward"]
        nr = nw_rank[r["id"]]; orr = oos_rank[r["id"]]
        cep = f"{wf.get('oos_ce_trades',0)}/{wf.get('oos_pe_trades',0)}"
        star = " ***" if orr <= 2 else ""
        print(f"[{nr:2d}] [{orr:2d}]  {r['name']:42s} Rs {nw['net_rs']:+11,.0f} {nw['pf']:6.2f} {nw['win_rate']:5.1f}% Rs {nw['max_dd']:7,.0f} Rs {wf['oos_net_rs']:+11,.0f} {wf['oos_pf']:6.2f} {wf['oos_wr']:5.1f}% Rs {wf['oos_max_dd']:7,.0f} {wf['wfe']:4.2f} {cep:>12s}{star}", flush=True)
    print("-" * 160, flush=True)

    print(f"\nDD COMPARISON:", flush=True)
    for r in by_oos:
        wf = r["walk_forward"]
        dd_ratio = wf["oos_max_dd"] / max(wf["oos_net_rs"], 1) * 100 if wf["oos_net_rs"] > 0 else 999
        print(f"  {r['name']:42s}: OOS DD=Rs {wf['oos_max_dd']:>7,.0f} ({dd_ratio:.0f}% of PnL) CE=Rs {wf.get('oos_ce_pnl',0):>+10,.0f} PE=Rs {wf.get('oos_pe_pnl',0):>+10,.0f}", flush=True)

    out = ROOT / "artifacts" / "f6_hybrid" / "master_phase5c_optuna_cached.json"
    with open(out, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "total_time_s": round(total_time, 2),
                   "total_trials": total_trials,
                   "results": by_oos}, f, indent=2)
    print(f"\nSaved to: {out}", flush=True)

if __name__ == "__main__":
    main()
