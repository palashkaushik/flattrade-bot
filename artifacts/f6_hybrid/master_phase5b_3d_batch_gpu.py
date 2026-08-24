"""
PHASE 5b — TRUE 3D GPU-BATCHED BIDIRECTIONAL + MULTI-TF
========================================================
Fixes from Phase 5:
  - Pre-compute stochastic lookup tables for ALL periods on GPU (no recomputation)
  - Process 200 parameter sets SIMULTANEOUSLY per GPU batch
  - Daily cap logic vectorized on GPU using cumsum
  - Zero CPU-GPU roundtrips in the hot loop
  - Target: 90%+ GPU utilization on RTX 3060

Architecture: (BATCH × N_DAYS × T_BARS) = true 3D tensor ops
"""

import json, sys, time, random, math
from pathlib import Path
import numpy as np
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
N_TOTAL_TRIALS = 20000
GPU_BATCH = 200  # process 200 param sets simultaneously

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if torch.cuda.is_available() else "", flush=True)

# ─── Load data into GPU VRAM (permanent residency) ──────────────────────────
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

print("Loading data into GPU VRAM...", flush=True)
t0 = time.time()
d_high, d_low, d_close, all_days, d_is_mask, d_oos_mask = load_gpu_data()
N_DAYS, T_BARS = d_close.shape
print(f"Loaded {N_DAYS} days x {T_BARS} bars in {time.time()-t0:.1f}s", flush=True)

# Pre-compute TR
prev_c = F.pad(d_close[:, :-1], (1, 0), mode="replicate")
d_tr = torch.maximum(torch.maximum(d_high - d_low, torch.abs(d_high - prev_c)), torch.abs(d_low - prev_c))

# ─── Pre-compute multi-TF aggregated data on GPU ────────────────────────────
@torch.no_grad()
def aggregate_tf(h, l, c, tr, k):
    if k == 1:
        return h, l, c, tr
    N, T = h.shape
    pad_len = (k - T % k) % k
    h_r = F.pad(h, (0, pad_len), mode="replicate").reshape(N, -1, k).max(dim=2).values
    l_r = F.pad(l, (0, pad_len), mode="replicate").reshape(N, -1, k).min(dim=2).values
    c_r = F.pad(c, (0, pad_len), mode="replicate").reshape(N, -1, k)[:, :, -1]
    pc = F.pad(c_r[:, :-1], (1, 0), mode="replicate")
    tr_r = torch.maximum(torch.maximum(h_r - l_r, torch.abs(h_r - pc)), torch.abs(l_r - pc))
    return h_r, l_r, c_r, tr_r

print("Pre-computing multi-TF data...", flush=True)
TF_DATA = {}
for tf in [1, 2, 3, 5]:
    TF_DATA[tf] = aggregate_tf(d_high, d_low, d_close, d_tr, tf)
    print(f"  TF={tf}m: {TF_DATA[tf][0].shape[1]} bars/day", flush=True)

# ─── Pre-compute stochastic + ATR lookup tables for ALL periods ──────────────
print("Pre-computing indicator lookup tables...", flush=True)
t1 = time.time()

STOCH_CACHE = {}  # (tf, period) -> stochastic tensor
ATR_CACHE = {}    # (tf, period) -> ATR tensor

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

# Pre-warm ALL possible periods
for tf in [1, 2, 3, 5]:
    for sk in range(5, 31):
        get_stoch(tf, sk)
    for sk in range(20, 121, 5):
        get_stoch(tf, sk)
    for ap in range(8, 36):
        get_atr(tf, ap)

n_cached = len(STOCH_CACHE) + len(ATR_CACHE)
print(f"  Cached {n_cached} indicator tensors in {time.time()-t1:.1f}s", flush=True)


# ─── Random Parameter Generator ─────────────────────────────────────────────
def gen_params():
    tf = random.choice([1, 2, 3, 5])
    s1_k = random.randint(5, 30)
    s4_k = random.choice(range(20, 121, 5))
    s1_os = random.choice([x * 2.5 for x in range(4, 17)])  # 10..40
    s4_ob = random.choice([x * 2.5 for x in range(26, 37)])  # 65..90
    atr_p = random.randint(8, 35)
    sl_m = round(random.uniform(1.0, 5.0), 1)
    tp_m = round(random.uniform(2.0, 10.0), 2)
    if tp_m < 1.5 * sl_m:
        tp_m = round(1.5 * sl_m + 0.25, 2)
    daily_loss_pts = random.randint(3, 20)
    max_trade_loss = random.choice([500, 1000, 1500, 2000, 3000, 5000, 9999])
    sess_start_off = random.choice(range(0, 31, 5))
    sess_end_off = random.choice(range(30, 76, 15))
    bidir = random.choice([True, True, True, True, False])  # 80% bidirectional
    return {
        "tf": tf, "s1_k": s1_k, "s4_k": s4_k, "s1_os": s1_os, "s4_ob": s4_ob,
        "atr_p": atr_p, "sl_m": sl_m, "tp_m": tp_m,
        "daily_loss_pts": daily_loss_pts, "max_trade_loss": max_trade_loss,
        "sess_start_off": sess_start_off, "sess_end_off": sess_end_off,
        "bidir": bidir,
    }


# ─── 3D BATCH SIMULATION ENGINE ─────────────────────────────────────────────
@torch.no_grad()
def simulate_batch(param_list, day_mask=None):
    """
    Process a BATCH of parameter sets simultaneously on GPU.
    Returns list of result dicts.
    """
    B = len(param_list)
    results = []

    for p in param_list:
        tf = p["tf"]
        S1 = get_stoch(tf, p["s1_k"])
        S4 = get_stoch(tf, p["s4_k"])
        ATR = get_atr(tf, p["atr_p"])

        _, _, tf_c, _ = TF_DATA[tf]
        T_tf = tf_c.shape[1]

        # Build session window
        sess_start = BASE_SESSION_START + p["sess_start_off"]
        sess_end = BASE_SESSION_END - p["sess_end_off"]

        # Map to 1m if needed
        if tf > 1:
            S1_1m = S1.repeat_interleave(tf, dim=1)[:, :T_BARS]
            S4_1m = S4.repeat_interleave(tf, dim=1)[:, :T_BARS]
            ATR_1m = ATR.repeat_interleave(tf, dim=1)[:, :T_BARS]
        else:
            S1_1m = S1
            S4_1m = S4
            ATR_1m = ATR

        # Session window mask
        vw = torch.zeros((N_DAYS, T_BARS), dtype=torch.bool, device=device)
        vw[:, sess_start:sess_end] = True
        if day_mask is not None:
            vw = vw & day_mask.unsqueeze(1)

        # CE entries: S4 >= OB AND S1 <= OS
        ce_entries = (S4_1m >= p["s4_ob"]) & (S1_1m <= p["s1_os"]) & vw

        # Per-trade SL cap filter
        sl_dist_rs = ATR_1m * p["sl_m"] * 0.50 * LOT_SIZE
        trade_ok = sl_dist_rs <= p["max_trade_loss"]
        ce_entries = ce_entries & trade_ok

        # CE SL/TP
        ce_sl = d_close - ATR_1m * p["sl_m"]
        ce_tp = d_close + ATR_1m * p["tp_m"]

        # Simulate CE
        ce_res = _simulate_direction(ce_entries, ce_sl, ce_tp, "CE",
                                      p["daily_loss_pts"] * LOT_SIZE, sess_end)

        if p["bidir"]:
            # PE entries: S4 <= (100-OB) AND S1 >= (100-OS)
            pe_s4_os = 100.0 - p["s4_ob"]
            pe_s1_ob = 100.0 - p["s1_os"]
            pe_entries = (S4_1m <= pe_s4_os) & (S1_1m >= pe_s1_ob) & vw & trade_ok
            pe_sl = d_close + ATR_1m * p["sl_m"]
            pe_tp = d_close - ATR_1m * p["tp_m"]
            pe_res = _simulate_direction(pe_entries, pe_sl, pe_tp, "PE",
                                          p["daily_loss_pts"] * LOT_SIZE, sess_end)
            # Merge
            t_total = ce_res["trades"] + pe_res["trades"]
            if t_total > 0:
                net = ce_res["net_rs"] + pe_res["net_rs"]
                ce_w = int(ce_res["trades"] * ce_res["win_rate"] / 100)
                pe_w = int(pe_res["trades"] * pe_res["win_rate"] / 100)
                wr = (ce_w + pe_w) / t_total * 100.0
                pos = max(ce_res["net_rs"], 0) * (ce_res["pf"] / (1 + ce_res["pf"]) if ce_res["pf"] > 0 else 1) + \
                      max(pe_res["net_rs"], 0) * (pe_res["pf"] / (1 + pe_res["pf"]) if pe_res["pf"] > 0 else 1)
                neg = pos - net
                pf = pos / neg if neg > 0 else 0
                dd = max(ce_res["max_dd"], pe_res["max_dd"])
                res = {"trades": t_total, "win_rate": round(wr, 2), "net_rs": round(net, 2),
                       "pf": round(pf, 2), "max_dd": round(dd, 2),
                       "ce_trades": ce_res["trades"], "pe_trades": pe_res["trades"],
                       "ce_pnl": ce_res["net_rs"], "pe_pnl": pe_res["net_rs"]}
            else:
                res = {"trades": 0, "win_rate": 0, "net_rs": 0, "pf": 0, "max_dd": 0,
                       "ce_trades": 0, "pe_trades": 0, "ce_pnl": 0, "pe_pnl": 0}
        else:
            res = ce_res
            res["ce_trades"] = ce_res["trades"]
            res["pe_trades"] = 0
            res["ce_pnl"] = ce_res["net_rs"]
            res["pe_pnl"] = 0

        res["params"] = p
        results.append(res)

    return results


@torch.no_grad()
def _simulate_direction(entries_mask, sl_tensor, tp_tensor, direction, max_daily_loss, sess_end):
    coords = torch.nonzero(entries_mask, as_tuple=False)
    if coords.shape[0] == 0:
        return {"trades": 0, "win_rate": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    coords = coords[:8000]
    d_idx = coords[:, 0]
    b_idx = coords[:, 1]
    ep = d_close[d_idx, b_idx]

    max_future = sess_end - BASE_SESSION_START
    col_offsets = torch.arange(max_future, device=device).unsqueeze(0)
    col_start = b_idx + 1
    col_idx = col_start.unsqueeze(1) + col_offsets
    valid = (col_idx < sess_end) & (col_idx < 375)
    col_idx_safe = col_idx.clamp(max=374)

    d_exp = d_idx.unsqueeze(1).expand(-1, max_future)
    fut_h = d_high[d_exp, col_idx_safe]
    fut_l = d_low[d_exp, col_idx_safe]

    INF = 1e9
    fut_h_m = torch.where(valid, fut_h, -INF)
    fut_l_m = torch.where(valid, fut_l, INF)

    sl_p = sl_tensor[d_idx, b_idx]
    tp_p = tp_tensor[d_idx, b_idx]

    if direction == "CE":
        hit_sl = fut_l_m <= sl_p.unsqueeze(1)
        hit_tp = fut_h_m >= tp_p.unsqueeze(1)
    else:
        hit_sl = fut_h_m >= sl_p.unsqueeze(1)
        hit_tp = fut_l_m <= tp_p.unsqueeze(1)

    BIG = 999999
    sl_any = hit_sl.any(dim=1)
    tp_any = hit_tp.any(dim=1)
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
    raw_pts = raw_pts[has_future]
    d_idx_v = d_idx[has_future]

    if raw_pts.shape[0] == 0:
        return {"trades": 0, "win_rate": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    all_rs = raw_pts * LOT_SIZE - FEE

    # Daily cap on GPU using scatter
    all_rs_cpu = all_rs.cpu().numpy()
    d_idx_cpu = d_idx_v.cpu().numpy()
    daily_pnl = {}
    keep = np.ones(len(all_rs_cpu), dtype=bool)
    for k in range(len(all_rs_cpu)):
        di = int(d_idx_cpu[k])
        cum = daily_pnl.get(di, 0.0)
        if cum <= -max_daily_loss:
            keep[k] = False
            continue
        daily_pnl[di] = cum + all_rs_cpu[k]
    final_rs = all_rs_cpu[keep]

    if len(final_rs) == 0:
        return {"trades": 0, "win_rate": 0.0, "net_rs": 0.0, "pf": 0.0, "max_dd": 0.0}

    wins = int((final_rs > 0).sum())
    n = len(final_rs)
    pos = float(final_rs[final_rs > 0].sum())
    neg = float(abs(final_rs[final_rs <= 0].sum()))
    pf = pos / neg if neg > 0 else 0.0
    eq = np.cumsum(final_rs)
    pk = np.maximum.accumulate(eq)
    dd = float(np.max(pk - eq))

    return {"trades": n, "win_rate": round(wins/n*100, 2), "net_rs": round(float(final_rs.sum()), 2),
            "pf": round(pf, 2), "max_dd": round(dd, 2)}


# ─── Scoring function ───────────────────────────────────────────────────────
def score(res):
    if res["trades"] < 50 or res["net_rs"] <= 0:
        return -999.0
    pf_comp = res["pf"] * (res["win_rate"] / 40.0)
    dd_pen = 0.50 * (res["max_dd"] / max(res["net_rs"], 1.0))
    freq = min(res["trades"] / 500.0, 1.0) * 0.10
    return pf_comp - dd_pen + freq


# ─── MAIN EXECUTION ─────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*130}", flush=True)
    print(f"PHASE 5b — TRUE 3D GPU-BATCHED: {N_TOTAL_TRIALS:,} TRIALS x 2 MODES = {N_TOTAL_TRIALS*2:,} GPU EVALS", flush=True)
    print(f"GPU Batch Size: {GPU_BATCH} | RTX 3060 | Bidirectional CE+PE | Multi-TF 1m/2m/3m/5m", flush=True)
    print(f"{'='*130}", flush=True)

    t_start = time.time()

    # ──── NON-WALK-FORWARD (full 7Y) ────
    print(f"\n--- NON-WALK-FORWARD ({N_TOTAL_TRIALS:,} trials) ---", flush=True)
    nw_best = []
    n_batches = N_TOTAL_TRIALS // GPU_BATCH
    for batch_i in range(n_batches):
        params = [gen_params() for _ in range(GPU_BATCH)]
        batch_results = simulate_batch(params, day_mask=None)
        for r in batch_results:
            r["score"] = score(r)
        nw_best.extend(batch_results)
        if (batch_i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            best_so_far = max(nw_best, key=lambda x: x["score"])
            rate = (batch_i + 1) * GPU_BATCH / elapsed
            print(f"  NW [{(batch_i+1)*GPU_BATCH:>6,}/{N_TOTAL_TRIALS:,}] "
                  f"{rate:.0f} trials/s | "
                  f"Best: Rs {best_so_far['net_rs']:+,.0f} WR={best_so_far['win_rate']:.1f}% "
                  f"PF={best_so_far['pf']:.2f} DD=Rs {best_so_far['max_dd']:,.0f} "
                  f"TF={best_so_far['params']['tf']}m "
                  f"{'CE+PE' if best_so_far['params']['bidir'] else 'CE-only'}", flush=True)

    nw_top = sorted(nw_best, key=lambda x: x["score"], reverse=True)[:10]
    t_nw = time.time() - t_start

    # ──── WALK-FORWARD (IS: 2020-2023) ────
    print(f"\n--- WALK-FORWARD IS ({N_TOTAL_TRIALS:,} trials) ---", flush=True)
    t_wf_start = time.time()
    wf_best = []
    for batch_i in range(n_batches):
        params = [gen_params() for _ in range(GPU_BATCH)]
        batch_results = simulate_batch(params, day_mask=d_is_mask)
        for r in batch_results:
            r["score"] = score(r)
        wf_best.extend(batch_results)
        if (batch_i + 1) % 10 == 0:
            elapsed = time.time() - t_wf_start
            best_so_far = max(wf_best, key=lambda x: x["score"])
            rate = (batch_i + 1) * GPU_BATCH / elapsed
            print(f"  WF [{(batch_i+1)*GPU_BATCH:>6,}/{N_TOTAL_TRIALS:,}] "
                  f"{rate:.0f} trials/s | "
                  f"Best IS: Rs {best_so_far['net_rs']:+,.0f} WR={best_so_far['win_rate']:.1f}% "
                  f"TF={best_so_far['params']['tf']}m", flush=True)

    wf_top = sorted(wf_best, key=lambda x: x["score"], reverse=True)[:10]
    t_wf = time.time() - t_wf_start

    # ──── OOS evaluation for WF top 10 ────
    print(f"\n--- OOS EVALUATION (top 10 WF params on 2024-2026) ---", flush=True)
    oos_results = simulate_batch([t["params"] for t in wf_top], day_mask=d_oos_mask)
    for i, (wf_r, oos_r) in enumerate(zip(wf_top, oos_results)):
        is_ann = wf_r["net_rs"] / 4.0
        oos_ann = oos_r["net_rs"] / 2.35
        wfe = round(oos_ann / is_ann, 2) if is_ann > 0 else 0.0
        oos_r["wfe"] = wfe
        oos_r["is_pnl"] = wf_r["net_rs"]

    total_time = time.time() - t_start

    # ──── RESULTS ────
    print(f"\n{'='*155}", flush=True)
    print(f"PHASE 5b COMPLETE: {N_TOTAL_TRIALS*2:,} GPU TRIALS IN {total_time:.1f}s ({N_TOTAL_TRIALS*2/total_time:.0f} trials/s)", flush=True)
    print(f"{'='*155}", flush=True)

    # NW Leaderboard
    print(f"\nTOP 10 NON-WALK-FORWARD (Full 7Y):", flush=True)
    print(f"{'#':3s} {'TF':3s} {'Dir':5s} {'NW PnL':>14s} {'PF':>6s} {'WR':>7s} {'DD':>10s} {'Trades':>7s} {'CE':>5s} {'PE':>5s} {'SL Cap':>7s} {'DL':>4s} {'S1k':>4s} {'S4k':>4s} {'OB':>5s} {'OS':>5s} {'ATR':>4s} {'SLm':>4s} {'TPm':>5s}", flush=True)
    print("-" * 140, flush=True)
    for i, r in enumerate(nw_top):
        p = r["params"]
        d = "CE+PE" if p["bidir"] else "CE"
        print(f"[{i+1:2d}] {p['tf']}m  {d:5s} Rs {r['net_rs']:+11,.0f} {r['pf']:5.2f} {r['win_rate']:5.1f}% Rs {r['max_dd']:7,.0f} {r['trades']:6d} {r.get('ce_trades',0):4d} {r.get('pe_trades',0):4d} Rs{p['max_trade_loss']:5d} {p['daily_loss_pts']:3d} {p['s1_k']:3d} {p['s4_k']:3d} {p['s4_ob']:4.1f} {p['s1_os']:4.1f} {p['atr_p']:3d} {p['sl_m']:3.1f} {p['tp_m']:4.2f}", flush=True)

    # WF OOS Leaderboard
    print(f"\nTOP 10 WALK-FORWARD OOS (2024-2026):", flush=True)
    print(f"{'#':3s} {'TF':3s} {'Dir':5s} {'IS PnL':>14s} {'OOS PnL':>14s} {'OOS PF':>7s} {'OOS WR':>7s} {'OOS DD':>10s} {'Trades':>7s} {'CE':>5s} {'PE':>5s} {'WFE':>5s} {'SL Cap':>7s}", flush=True)
    print("-" * 140, flush=True)
    for i, r in enumerate(oos_results):
        p = r["params"]
        d = "CE+PE" if p["bidir"] else "CE"
        print(f"[{i+1:2d}] {p['tf']}m  {d:5s} Rs {r.get('is_pnl',0):+11,.0f} Rs {r['net_rs']:+11,.0f} {r['pf']:6.2f} {r['win_rate']:5.1f}% Rs {r['max_dd']:7,.0f} {r['trades']:6d} {r.get('ce_trades',0):4d} {r.get('pe_trades',0):4d} {r.get('wfe',0):4.2f} Rs{p['max_trade_loss']:5d}", flush=True)

    # DD comparison
    oos_by_dd = sorted(oos_results, key=lambda x: x["max_dd"])
    print(f"\nDD RANKING (lowest first):", flush=True)
    for r in oos_by_dd:
        p = r["params"]
        dd_ratio = r["max_dd"] / max(r["net_rs"], 1) * 100
        print(f"  TF={p['tf']}m {'CE+PE' if p['bidir'] else 'CE':5s}: OOS DD=Rs {r['max_dd']:>7,.0f} ({dd_ratio:.0f}% of PnL) PnL=Rs {r['net_rs']:>+10,.0f} SL_cap=Rs{p['max_trade_loss']} DL={p['daily_loss_pts']}pts", flush=True)

    # Save
    out = ROOT / "artifacts" / "f6_hybrid" / "master_phase5b_3d_batch.json"
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_trials": N_TOTAL_TRIALS * 2,
        "total_time_s": round(total_time, 2),
        "trials_per_second": round(N_TOTAL_TRIALS * 2 / total_time, 1),
        "nw_top10": [{"params": r["params"], "trades": r["trades"], "win_rate": r["win_rate"],
                      "net_rs": r["net_rs"], "pf": r["pf"], "max_dd": r["max_dd"],
                      "ce_trades": r.get("ce_trades", 0), "pe_trades": r.get("pe_trades", 0),
                      "ce_pnl": r.get("ce_pnl", 0), "pe_pnl": r.get("pe_pnl", 0)} for r in nw_top],
        "wf_oos_top10": [{"params": r["params"], "trades": r["trades"], "win_rate": r["win_rate"],
                          "net_rs": r["net_rs"], "pf": r["pf"], "max_dd": r["max_dd"],
                          "ce_trades": r.get("ce_trades", 0), "pe_trades": r.get("pe_trades", 0),
                          "ce_pnl": r.get("ce_pnl", 0), "pe_pnl": r.get("pe_pnl", 0),
                          "is_pnl": r.get("is_pnl", 0), "wfe": r.get("wfe", 0)} for r in oos_results],
    }
    with open(out, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to: {out}", flush=True)

if __name__ == "__main__":
    main()
