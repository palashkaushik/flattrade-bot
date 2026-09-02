"""
FINAL COMBINATORIAL SWEEP — MEGA-BATCHED (GPU-saturated) EDITION.

Improvements over the first draft (smoke-test findings):
 1. ALL 256 gate bounce tensors are precomputed ONCE and stacked into the
    module-level bounce stacks; a single _eager_sim_core call evaluates a
    CHUNK of (gate, config) rows via the new bounce_idx parameter — the GPU
    runs one 345-step sequence per chunk at B=4096+ rows (full saturation)
    instead of 512 separate small passes.
 2. Level-list fill vectorized: per-day Python loops replaced by padded
    tensors built with numpy once (no torch tensor-per-day loop).
 3. Per-step bounce gather (B,D) from the (n_gate, D, T1) stack — no
    (B,D,T1) materialization (17GB saved at B=8192).
 4. Chunked at 4096 rows (VRAM ~2-3GB incl. temporaries; 12GB card safe).

Search space unchanged: gates = EMA20 + subsets of {EMA200,VWAP,CPR,Cam,
PDH/PDL,Fib,PrevVWAP,VirginCPR} (256) x configs = arm{5,10,15,20} x
mult{1.0,1.25,1.5,2.0} x be{0.4,0.5,0.6,0.7} x tb{0.0,0.5} (64)
= 16,384 runs, ~4-8 min total.

Pareto (net up, maxDD down, worst-day down), plateau stability, causal parity.
"""
import sys, time, os, itertools, csv
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
os.environ['LH_BIAS'] = '0'
os.environ['NUMBA_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['POLARS_MAX_THREADS'] = '8'

import numpy as np
import torch
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SMOKE = '--smoke' in sys.argv

t0 = time.time()
import seeded_lib as SL
import gpu_sim_last_hope as G
import run_7y_v4_master as M

D, T1 = SL.D, SL.T1
days = list(M.trading_days)

def metrics_from_trades(trades):
    if not trades:
        return dict(trades=0, wr=0.0, net=0.0, max_dd=0.0, worst_day=0.0, calmar=0.0)
    n = len(trades)
    wins = sum(1 for t in trades if t[5] > 0)
    net = sum(t[5] for t in trades)
    daily = {}
    for t in trades:
        daily[t[0]] = daily.get(t[0], 0.0) + t[5]
    cum = peak = 0.0; max_dd = 0.0
    for d in sorted(daily):
        cum += daily[d]; peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    worst = min(daily.values()) if daily else 0.0
    return dict(trades=n, wr=100.0 * wins / n, net=net, max_dd=max_dd,
                worst_day=worst, calmar=net / max_dd if max_dd > 1e-9 else float('inf'))

# ---------------------------------------------------------------------------
# Families (vectorized per-day level lists via numpy)
# ---------------------------------------------------------------------------
hi_p, lo_p, cl_p = np.asarray(M.pe_h).max(axis=1), np.asarray(M.pe_l).min(axis=1), np.asarray(M.pe_c)[:, -1]
hi_c, lo_c, cl_c = np.asarray(M.ce_h).max(axis=1), np.asarray(M.ce_l).min(axis=1), np.asarray(M.ce_c)[:, -1]
vwap_p_final = np.asarray(M.pe_vwap)[:, -1]
vwap_c_final = np.asarray(M.ce_vwap)[:, -1]
Dn = len(days)

def _bands(hi, lo, cl):
    bands = [None] * Dn
    for j in range(1, Dn):
        H, L, C = hi[j - 1], lo[j - 1], cl[j - 1]
        p = (H + L + C) / 3.0; bc = (H + L) / 2.0; tc = 2.0 * p - bc
        bands[j] = (min(bc, tc), max(bc, tc), bc, tc)
    return bands

def _virgin(hi, lo, bands):
    out = [[] for _ in range(Dn)]; active = []
    for d in range(Dn):
        if bands[d] is not None:
            active.append((d, bands[d]))
        for (_j, b) in [(j, b) for (j, b) in active if j < d][-5:]:
            out[d] += [b[2], b[3]]
        active = [(j, b) for (j, b) in active if not (lo[d] <= b[1] and hi[d] >= b[0])]
    return out

bands_p, bands_c = _bands(hi_p, lo_p, cl_p), _bands(hi_c, lo_c, cl_c)
vir_p, vir_c = _virgin(hi_p, lo_p, bands_p), _virgin(hi_c, lo_c, bands_c)

def _cpr(hi, lo, cl):
    out = [[] for _ in range(Dn)]
    for d in range(1, Dn):
        H, L, C = hi[d - 1], lo[d - 1], cl[d - 1]
        p = (H + L + C) / 3.0; bc = (H + L) / 2.0
        out[d] = [bc, p, 2.0 * p - bc]
    return out

def _cam(hi, lo, cl):
    out = [[] for _ in range(Dn)]
    for d in range(1, Dn):
        H, L, C = hi[d - 1], lo[d - 1], cl[d - 1]; rng = H - L
        out[d] = [C + rng * 1.1 / 4.0, C - rng * 1.1 / 4.0]
    return out

def _pdhl(hi, lo):
    return [[hi[d - 1], lo[d - 1]] if d >= 1 else [] for d in range(Dn)]

def _fib(hi, lo, cl):
    out = [[] for _ in range(Dn)]
    for d in range(1, Dn):
        H, L, C = hi[d - 1], lo[d - 1], cl[d - 1]
        p = (H + L + C) / 3.0; rng = H - L
        out[d] = [p + rng, p - rng]
    return out

DYN = {"EMA20": (SL.pe_ema20_seed, SL.ce_ema20_seed),
       "EMA200": (SL.pe_ema200_seed, SL.ce_ema200_seed),
       "VWAP": (SL.pe_vwap_seed, SL.ce_vwap_seed)}
FAM_STATIC = {
    "CPR":       (_cpr(hi_p, lo_p, cl_p), _cpr(hi_c, lo_c, cl_c)),
    "Cam":       (_cam(hi_p, lo_p, cl_p), _cam(hi_c, lo_c, cl_c)),
    "PDH/PDL":   (_pdhl(hi_p, lo_p), _pdhl(hi_c, lo_c)),
    "Fib":       (_fib(hi_p, lo_p, cl_p), _fib(hi_c, lo_c, cl_c)),
    "PrevVWAP":  ([[float(vwap_p_final[d - 1])] if d >= 1 else [] for d in range(Dn)],
                  [[float(vwap_c_final[d - 1])] if d >= 1 else [] for d in range(Dn)]),
    "VirginCPR": (vir_p, vir_c),
}
OPT_FAMS = ["EMA200", "VWAP"] + list(FAM_STATIC.keys())
ALL_SUBSETS = [tuple(c) for r in range(0, 9) for c in itertools.combinations(OPT_FAMS, r)]

pe_tf_lo = [c['lo'] for c in SL.pe_tf_seed]
pe_tf_cl = [c['cl'] for c in SL.pe_tf_seed]
ce_tf_lo = [c['lo'] for c in SL.ce_tf_seed]
ce_tf_cl = [c['cl'] for c in SL.ce_tf_seed]

def build_bounce_tensor(dyn_fams, static_fams, buf):
    """Vectorized (D, T1) bool bounce for one gate.

    Static levels: (D, T1, max_n) broadcast across time.
    Dynamic levels: (D, T1) each, appended as extra columns.
    """
    use_dyn = ["EMA20"] + [f for f in dyn_fams if f != "EMA20"]
    pe_lists = [sum((FAM_STATIC[f][0][d] for f in static_fams), []) for d in range(Dn)]
    ce_lists = [sum((FAM_STATIC[f][1][d] for f in static_fams), []) for d in range(Dn)]
    max_n = max(max((len(x) for x in pe_lists), default=0),
                max((len(x) for x in ce_lists), default=0), 1)
    # (D, max_n) per-day static values (pad = +inf so never bounces)
    st_pe = np.full((Dn, max_n), np.inf, dtype=np.float32)
    st_ce = np.full((Dn, max_n), np.inf, dtype=np.float32)
    for d in range(Dn):
        if pe_lists[d]:
            st_pe[d, :len(pe_lists[d])] = pe_lists[d]
        if ce_lists[d]:
            st_ce[d, :len(ce_lists[d])] = ce_lists[d]
    # broadcast across time -> (D, T1, max_n)
    st_pe_t = torch.tensor(st_pe, device=DEVICE).unsqueeze(1).expand(-1, T1, -1)
    st_ce_t = torch.tensor(st_ce, device=DEVICE).unsqueeze(1).expand(-1, T1, -1)
    # dynamic columns: (D, T1, n_dyn)
    dyn_pe = torch.stack([DYN[nm][0] for nm in use_dyn], dim=2) if len(use_dyn) > 1 else \
             DYN[use_dyn[0]][0].unsqueeze(2) if use_dyn else torch.empty(D, T1, 0, device=DEVICE)
    dyn_ce = torch.stack([DYN[nm][1] for nm in use_dyn], dim=2) if len(use_dyn) > 1 else \
             DYN[use_dyn[0]][1].unsqueeze(2) if use_dyn else torch.empty(D, T1, 0, device=DEVICE)
    sr_pe = torch.cat([st_pe_t, dyn_pe], dim=2)   # (D, T1, max_n + n_dyn)
    sr_ce = torch.cat([st_ce_t, dyn_ce], dim=2)
    def one(low, close, sr, tlos, tcls):
        lo = low.unsqueeze(-1); cl = close.unsqueeze(-1)
        b = ((lo <= sr + buf) & (cl >= sr - 0.5)).any(dim=-1)
        for tlo, tcl in zip(tlos, tcls):
            b |= ((tlo.unsqueeze(-1) <= sr + buf) & (tcl.unsqueeze(-1) >= sr - 0.5)).any(dim=-1)
        return b
    return one(G.pe_l, G.pe_c, sr_pe, pe_tf_lo, pe_tf_cl), \
           one(G.ce_l, G.ce_c, sr_ce, ce_tf_lo, ce_tf_cl)

# ---------------------------------------------------------------------------
# Precompute ALL gate bounces (256) x tb (2) -> stacks (512, D, T1)
# ---------------------------------------------------------------------------
t_pre = time.time()
TBS = [0.0, 0.5] if not SMOKE else [0.0]
gate_names = []
stack_pe_list, stack_ce_list = [], []
for tb in TBS:
    for subset in ALL_SUBSETS:
        dyn_fams = [f for f in subset if f in DYN]
        static_fams = [f for f in subset if f in FAM_STATIC]
        b_pe, b_ce = build_bounce_tensor(dyn_fams, static_fams, tb)
        stack_pe_list.append(b_pe)
        stack_ce_list.append(b_ce)
        gate_names.append(("+".join(("EMA20",) + subset), tb))
n_bounce = len(stack_pe_list)
stack_pe = torch.stack(stack_pe_list, 0)   # (n_bounce, D, T1)
stack_ce = torch.stack(stack_ce_list, 0)
del stack_pe_list, stack_ce_list
print(f"[bounces] {n_bounce} gate-tb tensors in {time.time()-t_pre:.1f}s "
      f"(mem {n_bounce * D * T1 * 1 / 1e9:.2f} GB bool)")

# Point the module stacks at our mega-stack (indices are OUR row ids now)
G.bounce_pe_stack = stack_pe
G.bounce_ce_stack = stack_ce

# ---------------------------------------------------------------------------
# Build the mega (gate, config) row list
# ---------------------------------------------------------------------------
ARMS = [10, 15] if SMOKE else [5, 10, 15, 20]
MULTS = [1.25, 1.5] if SMOKE else [1.0, 1.25, 1.5, 2.0]
BES = [0.5, 0.7] if SMOKE else [0.4, 0.5, 0.6, 0.7]

def mk_cfg(arm, mult, be, tb):
    c = dict(SL.CHAMPION_BASE)
    c.update(arm_window=arm, atr_period=10, atr_mult=mult, touch_buffer=tb,
             be_trigger=be, be_buffer=1.0, entry_end=T1, entry_start=0)
    return c

rows = []   # (gate_idx, cfg_tuple)
for gi, (gname, tb) in enumerate(gate_names):
    for arm in ARMS:
        for mult in MULTS:
            for be in BES:
                rows.append((gi, (arm, mult, be, tb)))
print(f"[rows] {len(rows)} (gate, config) combos")

# ---------------------------------------------------------------------------
# Mega-batch execution in chunks (metrics_only mode: GPU-side accumulation,
# NO trade-list materialization — the 16GB-RAM freeze fix; and zero per-step
# host syncs — the WDDM desktop-freeze fix)
# ---------------------------------------------------------------------------
def metrics_from_day_pnl(day_pnl_row, trade_counts_row, win_counts_row):
    """Host-side metrics from per-day arrays for ONE config."""
    tc = trade_counts_row.astype(np.float64)
    wc = win_counts_row.astype(np.float64)
    dp = day_pnl_row.astype(np.float64)
    n = int(tc.sum())
    if n == 0:
        return dict(trades=0, wr=0.0, net=0.0, max_dd=0.0, worst_day=0.0, calmar=0.0)
    net = float(dp.sum())
    wins = int(wc.sum())
    cum = peak = 0.0; max_dd = 0.0
    for v in dp:
        cum += v
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    worst = float(dp.min())
    return dict(trades=n, wr=100.0 * wins / n, net=net, max_dd=max_dd,
                worst_day=worst, calmar=net / max_dd if max_dd > 1e-9 else float('inf'))

CHUNK = 256 if SMOKE else 1024
results = []
t_sweep = time.time()
for s in range(0, len(rows), CHUNK):
    chunk_rows = rows[s:s + CHUNK]
    cfgs = [mk_cfg(*r[1]) for r in chunk_rows]
    bidx = torch.tensor([r[0] for r in chunk_rows], dtype=torch.long)
    day_pnl, trade_counts, win_counts = G._eager_sim_core(
        cfgs, bounce_idx=bidx, metrics_only=True)
    # ONE host transfer per chunk: (B, D) x3 floats — ~6 MB at B=1024
    dp_np = day_pnl.cpu().numpy()
    tc_np = trade_counts.cpu().numpy()
    wc_np = win_counts.cpu().numpy()
    for bi, (gi, key) in enumerate(chunk_rows):
        m = metrics_from_day_pnl(dp_np[bi], tc_np[bi], wc_np[bi])
        m['gate'] = gate_names[gi][0]
        m['tb'] = gate_names[gi][1]
        m['cfg'] = key
        results.append(m)
    done = s + len(chunk_rows)
    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {done}/{len(rows)} rows | {time.time()-t_sweep:.0f}s | GPU-mem {mem_gb:.1f}GB")

print(f"[sweep] done: {len(results)} rows in {time.time()-t_sweep:.0f}s")

# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------
FLOOR_TRADES = 50 if SMOKE else 5000
valid = [m for m in results if m['trades'] >= FLOOR_TRADES and m['net'] > 0]
print(f"[pareto] {len(valid)} valid of {len(results)}")

def dominates(a, b):
    ge = (a['net'] >= b['net']) and (a['max_dd'] <= b['max_dd']) and (a['worst_day'] >= b['worst_day'])
    gt = (a['net'] > b['net']) or (a['max_dd'] < b['max_dd']) or (a['worst_day'] > b['worst_day'])
    return ge and gt

front = []
for i, a in enumerate(valid):
    if not any(dominates(b, a) for j, b in enumerate(valid) if i != j):
        front.append(a)
print(f"[pareto] front size {len(front)}")

print("\n" + "=" * 118)
print(f"TOP 20 BY NET (Pareto members, trades>={FLOOR_TRADES})")
print("=" * 118)
for m in sorted(front, key=lambda x: -x['net'])[:20]:
    print(f"net Rs {m['net']:>12,.0f} | dd Rs {m['max_dd']:>7,.0f} | worst Rs {m['worst_day']:>7,.0f} | "
          f"calmar {m['calmar']:>6.1f} | wr {m['wr']:>5.1f}% | tr {m['trades']:>6} | "
          f"arm{m['cfg'][0]:>2} x{m['cfg'][1]:<4} be{m['cfg'][2]:<3} tb{m['tb']:<3} | {m['gate']}")

print("\n" + "=" * 118)
print("TOP 20 BY CALMAR (Pareto members)")
print("=" * 118)
for m in sorted(front, key=lambda x: -x['calmar'])[:20]:
    print(f"net Rs {m['net']:>12,.0f} | dd Rs {m['max_dd']:>7,.0f} | worst Rs {m['worst_day']:>7,.0f} | "
          f"calmar {m['calmar']:>6.1f} | wr {m['wr']:>5.1f}% | tr {m['trades']:>6} | "
          f"arm{m['cfg'][0]:>2} x{m['cfg'][1]:<4} be{m['cfg'][2]:<3} tb{m['tb']:<3} | {m['gate']}")

# ---------------------------------------------------------------------------
# Plateau / stability (neighbors one axis-step away)
# ---------------------------------------------------------------------------
CONFIGS_ALL = [(a, m, b, t) for a in ARMS for m in MULTS for b in BES for t in TBS]
res_idx = {}
for m in results:
    res_idx[(m['gate'], m['tb'], m['cfg'])] = m

def neighbors(k):
    out = []
    for k2 in CONFIGS_ALL:
        diff = (k2[0] != k[0]) + (k2[1] != k[1]) + (k2[2] != k[2]) + (k2[3] != k[3])
        if diff == 1:
            out.append(k2)
    return out

print("\n" + "=" * 118)
print("PLATEAU / STABILITY (top-10 by net: worst neighbor drop)")
print("=" * 118)
with open('plateau_report.txt', 'w') as pf:
    for m in sorted(front, key=lambda x: -x['net'])[:10]:
        nbrs = [res_idx[(m['gate'], m['tb'], nk)] for nk in neighbors(m['cfg'])
                if (m['gate'], m['tb'], nk) in res_idx]
        if not nbrs:
            continue
        worst_drop = min((nb['net'] - m['net']) / max(abs(m['net']), 1) for nb in nbrs)
        stable = "STABLE (plateau)" if worst_drop > -0.10 else "FRAGILE (isolated peak)"
        line = (f"{m['gate']:<55} arm{m['cfg'][0]:>2} x{m['cfg'][1]:<4} be{m['cfg'][2]:<3} tb{m['tb']:<3} | "
                f"net {m['net']:>12,.0f} | worst neighbor {worst_drop*100:>+6.1f}% | {stable}")
        print(line); pf.write(line + "\n")

with open('pareto_front.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['gate', 'arm', 'mult', 'be', 'tb', 'trades', 'wr', 'net', 'max_dd', 'worst_day', 'calmar', 'pareto'])
    front_ids = set(id(x) for x in front)
    for m in results:
        w.writerow([m['gate'], m['cfg'][0], m['cfg'][1], m['cfg'][2], m['tb'],
                    m['trades'], round(m['wr'], 2), round(m['net'], 2),
                    round(m['max_dd'], 2), round(m['worst_day'], 2),
                    round(m['calmar'], 3), int(id(m) in front_ids)])
print("\n[saved] pareto_front.csv, plateau_report.txt")
print(f"[total] {time.time()-t0:.0f}s")
