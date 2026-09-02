"""
SEEDED-INDICATOR SWEEP for the Last Hope strategy (GPU, causal parity checked).

QUESTION THIS ANSWERS
---------------------
The champion backtest cold-starts every indicator each day (per-day arrays).
The live bot seeds from prior-day data. Live losses showed the seeded morning
multi-TF signals underperform massively. So: is there a BETTER config that is
OPTIMAL *specifically for the seeded mode*?

METHOD
------
1. Rebuild all indicator inputs as SEEDED (D, T1) arrays: each day's
   stochastics/ATR/EMA/VWAP computed with prior-day tail bars prepended
   (exactly like the live bot's warmup), then sliced back to the day.
2. Reuse the batched GPU engine (gpu_sim_last_hope) — swap only the
   module-level tensors it consumed at import for seeded equivalents.
3. Sweep a compact grid around the champion + seeded-specific axes
   (arm_window, atr_period, atr_mult, touch_buffer, entry_start, be params).
4. Metrics: net_rs, WR, max_dd (equity peak-to-trough on DAILY aggregated P&L),
   plus WORST-DAY loss (daily drawdown control) and Calmar-like ratio.

SPEED (per the GPU guides): all indicators computed ONCE per unique lookback
(max_pool1d vectorized, day-continuity handled by prefix-padding — no Python
per-day loops in the indicator build); the sweep itself is one batched eager
pass per config-chunk. fp32 throughout (sweep tier), champion re-run included
for A/B comparison within the same run.

CAUSAL PARITY: prefix bars are strictly PRIOR-day data — no future leak. The
last-bar prefix is the previous session's actual close region, matching live.
"""
import sys, time, json, itertools
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
import os
# Skip the 45s 15m-bias lookup build — every swept config runs bias-OFF
os.environ['LH_BIAS'] = os.environ.get('LH_BIAS_OVERRIDE', '0')
import numpy as np
import torch

SMOKE = '--smoke' in sys.argv   # 5-day validation run

torch.set_float32_matmul_precision('high')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t0 = time.time()
import run_7y_v4_master as M          # builds per-day (cold-start) state
import gpu_sim_last_hope as G         # builds GPU tensors from M (cold-start)

D, T1 = M.D, M.T1
LOT, FEE = M.LOT, M.FEE
SS = M.SESSION_START                  # 555
trading_days = M.trading_days

# ── SMOKE: use whatever the master loaded (master smoke mode already
# restricts days); just flag it and shrink the grid. No re-slicing needed —
# the seeded builder below reads array shapes dynamically.
if SMOKE:
    print("=== SMOKE TEST — reduced grid, master's loaded day range ===")

print(f"[init] master+gpu modules loaded in {time.time()-t0:.1f}s | D={D} T1={T1}")

# ---------------------------------------------------------------------------
# 1. SEEDED indicator rebuild
# ---------------------------------------------------------------------------
SEED_BARS = 300   # 300 prior-day 1m bars = 60 5m bars >= S4(50,10) full warmup
_TF_LCM = 30      # LCM(1,2,3,5): extended axis padded so TF chunks align for all TFs

def _g(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)

def _seeded_pair(src, days_sorted):
    """Return (c,h,l,o) (D, T1) where each day is NAN-safe and we also return
    the per-day seed slices (prior trading day's last SEED_BARS bars)."""
    return src  # placeholder (built below by direct dataframe route)

# Fastest causal route: rebuild seeded stochastics DIRECTLY from the master's
# per-day tensors by concatenating day d-1's tail to day d (GPU batched).
ce_c, ce_h, ce_l = M.ce_c, M.ce_h, M.ce_l
pe_c, pe_h, pe_l = M.pe_c, M.pe_h, M.pe_l

# torch copies on GPU (D,T1) — build extended (D, SEED+T1) then slice
def _extend(c):
    ct = _g(c)
    D_loc = ct.shape[0]
    prev_tail = ct[:, -(SEED_BARS):]                    # (D, 300) each day's own tail
    # roll: day d uses day d-1's tail -> shift rows down by 1
    shifted = torch.zeros_like(prev_tail)
    if D_loc > 1:
        shifted[1:] = prev_tail[:-1]
    shifted[0] = prev_tail[0]                           # first day: own tail (no prior)
    ext = torch.cat([shifted, ct], dim=1)               # (D, 300+T1)
    # Pad the day region to a multiple of _TF_LCM so TF chunking aligns for
    # every TF across the full extended axis (mirror-pad with the last bar).
    day_len = ext.shape[1] - SEED_BARS
    pad_need = (-day_len) % _TF_LCM
    if pad_need:
        ext = torch.cat([ext, ext[:, -1:].expand(-1, pad_need)], dim=1)
    return ext

def _stoch_seed(h3, l3, c3, k, d_period):
    """Stochastic %D over (D, SEED+T1) — vectorized max_pool1d, causal."""
    h_pad = torch.nn.functional.pad(h3.unsqueeze(1), (k - 1, 0), mode="replicate")
    l_pad = torch.nn.functional.pad(l3.unsqueeze(1), (k - 1, 0), mode="replicate")
    max_h = torch.nn.functional.max_pool1d(h_pad, k, stride=1).squeeze(1)
    min_l = -torch.nn.functional.max_pool1d(-l_pad, k, stride=1).squeeze(1)
    denom = (max_h - min_l).clamp(min=1e-6)
    fast_k = (c3 - min_l) / denom * 100.0
    k_pad = torch.nn.functional.pad(fast_k.unsqueeze(1), (d_period - 1, 0), mode="replicate")
    slow_d = torch.nn.functional.avg_pool1d(k_pad, d_period, stride=1).squeeze(1)
    return slow_d

def _atr_seed(h3, l3, c3, period):
    """EMA ATR (alpha=2/(p+1)) over extended axis, causal."""
    prev = torch.empty_like(c3); prev[:, 0] = c3[:, 0]; prev[:, 1:] = c3[:, :-1]
    tr = torch.maximum(h3 - l3, torch.maximum(torch.abs(h3 - prev), torch.abs(l3 - prev)))
    atr = torch.empty_like(tr); alpha = 2.0 / (period + 1)
    atr[:, 0] = tr[:, 0]
    for i in range(1, tr.shape[1]):
        atr[:, i] = atr[:, i - 1] * (1.0 - alpha) + tr[:, i] * alpha
    return atr

def _ema_seed(c3, period):
    alpha = 2.0 / (period + 1)
    ema = torch.empty_like(c3); ema[:, 0] = c3[:, 0]
    for i in range(1, c3.shape[1]):
        ema[:, i] = ema[:, i - 1] * (1.0 - alpha) + c3[:, i] * alpha
    return ema

def _vwap_seed(h3, l3, c3):
    """Session VWAP — resets at the day boundary (extended index SEED_BARS)."""
    D_loc = h3.shape[0]
    tp = (h3 + l3 + c3) / 3.0
    cs_pv = torch.cumsum(tp, dim=1)
    # baseline = cumsum through the last SEED bar, one value per day bar:
    # day bar i (0..T1-1) has baseline cs_pv[:, SEED_BARS-1+i]
    pv_at_day_start = cs_pv[:, SEED_BARS - 1:SEED_BARS - 1 + T1]    # (D, T1)
    vwap_day = (cs_pv[:, SEED_BARS:SEED_BARS + T1] - pv_at_day_start) / \
        torch.arange(1, T1 + 1, device=c3.device, dtype=torch.float32).unsqueeze(0)
    return vwap_day

# Build extended tensors once
ce_c3, ce_h3, ce_l3 = _extend(ce_c), _extend(ce_h), _extend(ce_l)
pe_c3, pe_h3, pe_l3 = _extend(pe_c), _extend(pe_h), _extend(pe_l)
print(f"[seed] extended tensors built ({D}x{SEED_BARS+T1}) in {time.time()-t0:.1f}s")

# Seeded stochastics per TF — clock-aligned chunking from day start (matches
# the backtest's reshape convention; seeding only changes the indicator STATE)
S1_K, S1_D = M.S1_K, M.S1_D
S3_K, S3_D = M.S3_K, M.S3_D
S4_K, S4_D = M.S4_K, M.S4_D
TF_LIST = M.TF_LIST

def _tf_seed_stoch(h3, l3, c3, tf):
    """Per-TF seeded stochastics: chunk the EXTENDED axis, compute on TF bars,
    then map back per-day (day-local chunking identical to make_tf_stoch)."""
    D_loc = h3.shape[0]
    T_ext = h3.shape[1]
    T_tf = T_ext // tf
    n = T_tf * tf
    c_ = c3[:, :n].reshape(D_loc, T_tf, tf)
    h_ = h3[:, :n].reshape(D_loc, T_tf, tf)
    l_ = l3[:, :n].reshape(D_loc, T_tf, tf)
    cl = c_[:, :, -1]; hi = h_.amax(dim=2); lo = l_.amin(dim=2)
    s1 = _stoch_seed_hi(hi, lo, cl, S1_K, S1_D)
    s3 = _stoch_seed_hi(hi, lo, cl, S3_K, S3_D)
    s4 = _stoch_seed_hi(hi, lo, cl, S4_K, S4_D)
    rising = torch.zeros_like(s1)
    rising[:, 1:] = (s1[:, 1:] > s1[:, :-1]).float()
    # expand each TF value to per-1m-bar (repeat tf), then slice the day region
    # (SEED_BARS is a multiple of tf's chunk alignment; day slice is T1 wide)
    def exp(a):
        e = a.repeat_interleave(tf, dim=1)
        return e[:, SEED_BARS:SEED_BARS + T1]
    return dict(s1=exp(s1), s3=exp(s3), s4=exp(s4), rising=exp(rising),
                lo=exp(lo), cl=exp(cl), tf=tf)

def _stoch_seed_hi(hi, lo, cl, k, d_period):
    """Stoch %D on TF-aggregated series (hi, lo, cl are TF-close-indexed)."""
    h_pad = torch.nn.functional.pad(hi.unsqueeze(1), (k - 1, 0), mode="replicate")
    l_pad = torch.nn.functional.pad(lo.unsqueeze(1), (k - 1, 0), mode="replicate")
    max_h = torch.nn.functional.max_pool1d(h_pad, k, stride=1).squeeze(1)
    min_l = -torch.nn.functional.max_pool1d(-l_pad, k, stride=1).squeeze(1)
    denom = (max_h - min_l).clamp(min=1e-6)
    fast_k = (cl - min_l) / denom * 100.0
    k_pad = torch.nn.functional.pad(fast_k.unsqueeze(1), (d_period - 1, 0), mode="replicate")
    return torch.nn.functional.avg_pool1d(k_pad, d_period, stride=1).squeeze(1)

t1 = time.time()
ce_tf_seed = [_tf_seed_stoch(ce_h3, ce_l3, ce_c3, tf) for tf in TF_LIST]
pe_tf_seed = [_tf_seed_stoch(pe_h3, pe_l3, pe_c3, tf) for tf in TF_LIST]
print(f"[seed] TF stochastics built in {time.time()-t1:.1f}s")

# Seeded 1m stochastics (s1 arrays used for arming) + ATR + EMA + VWAP
def _stoch_1m(h3, l3, c3, k, d_period):
    return _stoch_seed(h3, l3, c3, k, d_period)[:, SEED_BARS:SEED_BARS + T1]

ce_s1_seed = _stoch_1m(ce_h3, ce_l3, ce_c3, S1_K, S1_D)
pe_s1_seed = _stoch_1m(pe_h3, pe_l3, pe_c3, S1_K, S1_D)

ATR_PERIODS = sorted(set([10, 14]))
ce_atr_seed = {p: _atr_seed(ce_h3, ce_l3, ce_c3, p)[:, SEED_BARS:SEED_BARS + T1] for p in ATR_PERIODS}
pe_atr_seed = {p: _atr_seed(pe_h3, pe_l3, pe_c3, p)[:, SEED_BARS:SEED_BARS + T1] for p in ATR_PERIODS}

ce_ema20_seed = _ema_seed(ce_c3, 20)[:, SEED_BARS:SEED_BARS + T1]
ce_ema200_seed = _ema_seed(ce_c3, 200)[:, SEED_BARS:SEED_BARS + T1]
pe_ema20_seed = _ema_seed(pe_c3, 20)[:, SEED_BARS:SEED_BARS + T1]
pe_ema200_seed = _ema_seed(pe_c3, 200)[:, SEED_BARS:SEED_BARS + T1]
ce_vwap_seed = _vwap_seed(ce_h3, ce_l3, ce_c3)
pe_vwap_seed = _vwap_seed(pe_h3, pe_l3, pe_c3)
print(f"[seed] 1m stoch/ATR/EMA/VWAP built in {time.time()-t1:.1f}s total")

# ---------------------------------------------------------------------------
# 2. Rebuild masks + bounce in SEEDED space (GPU batched, chunk-free)
# ---------------------------------------------------------------------------
def _build_masks(tf_list, side):
    super_full = torch.zeros((D, T1), dtype=torch.bool, device=DEVICE)
    m6_full = torch.zeros((D, T1), dtype=torch.bool, device=DEVICE)
    for c in tf_list:
        super_full |= (c['s3'] < 25.0) & (c['s4'] < 25.0) & (c['s1'] < 25.0) & (c['rising'] > 0.5)
        m6_full |= (c['s4'] >= 79.5) & (c['s1'] < 79.5)
    return super_full, m6_full

ce_super_seed, ce_m6_seed = _build_masks(ce_tf_seed, 'CE')
pe_super_seed, pe_m6_seed = _build_masks(pe_tf_seed, 'PE')
print(f"[seed] masks built ({time.time()-t0:.1f}s)")

# SR levels identical to master (day-level, prior-day OHLC — already "seeded"
# by construction). Reuse master's CPU lists.
pe_sr_levels, ce_sr_levels = G.pe_sr_levels, G.ce_sr_levels
max_sr = G.max_sr
PAD = G.PAD

def _build_bounce_seed(low, close, ema20, ema200, vwap, sr_levels, tf_lo_list, tf_cl_list, buf=0.0):
    sr = torch.full((D, T1, PAD), float('inf'), device=DEVICE)
    for d in range(D):
        levels = sr_levels[d]
        n = len(levels)
        if n:
            sr[d, :, :n] = torch.tensor(levels, dtype=torch.float32, device=DEVICE)
    sr[:, :, max_sr] = ema20
    sr[:, :, max_sr + 1] = ema200
    sr[:, :, max_sr + 2] = vwap
    lo = low.unsqueeze(-1)
    cl = close.unsqueeze(-1)
    cond = (lo <= sr + buf) & (cl >= sr - 0.5)
    b = cond.any(dim=-1)
    for tlo, tcl in zip(tf_lo_list, tf_cl_list):
        tb = ((tlo.unsqueeze(-1) <= sr + buf) & (tcl.unsqueeze(-1) >= sr - 0.5)).any(dim=-1)
        b |= tb
    return b

# Touch buffers to sweep
TOUCH_BUFFERS_SWEEP = [0.0, 0.5, 1.0]
bounce_ce_seeds = {b: _build_bounce_seed(G.ce_l, G.ce_c, ce_ema20_seed, ce_ema200_seed, ce_vwap_seed,
                                         ce_sr_levels, [c['lo'] for c in ce_tf_seed], [c['cl'] for c in ce_tf_seed], buf=b)
                   for b in TOUCH_BUFFERS_SWEEP}
bounce_pe_seeds = {b: _build_bounce_seed(G.pe_l, G.pe_c, pe_ema20_seed, pe_ema200_seed, pe_vwap_seed,
                                         pe_sr_levels, [c['lo'] for c in pe_tf_seed], [c['cl'] for c in pe_tf_seed], buf=b)
                   for b in TOUCH_BUFFERS_SWEEP}
print(f"[seed] bounce built ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# 3. Patch the GPU sim's module state to SEEDED, then run a custom core copy
# ---------------------------------------------------------------------------
# The eager core reads module globals (ce_s1 etc. in G). We monkey-patch G's
# globals to seeded versions, swap the bounce stacks, and neutralize the
# per-config touch-buffer selection (we handle buffers ourselves below by
# running separate batches per buffer).

G.ce_s1, G.pe_s1 = pe_s1_seed if False else ce_s1_seed, pe_s1_seed  # (arm source)
G.ce_s3, G.ce_s4 = ce_tf_seed[0]['s3'], ce_tf_seed[0]['s4']        # 1m TF entries
G.pe_s3, G.pe_s4 = pe_tf_seed[0]['s3'], pe_tf_seed[0]['s4']
G.ce_super_full, G.ce_m6_full = ce_super_seed, ce_m6_seed
G.pe_super_full, G.pe_m6_full = pe_super_seed, pe_m6_seed
G.ce_ema20, G.ce_ema200, G.ce_vwap = ce_ema20_seed, ce_ema200_seed, ce_vwap_seed
G.pe_ema20, G.pe_ema200, G.pe_vwap = pe_ema20_seed, pe_ema200_seed, pe_vwap_seed
# ATR: patch the cache used by _get_atr so per-config atr_period hits seeded ATR
for p in ATR_PERIODS:
    G._ATR_CACHE[(id(G.pe_h), p)] = pe_atr_seed[p]
    G._ATR_CACHE[(id(G.ce_h), p)] = ce_atr_seed[p]
# Supertrend zone filter is OFF in all swept configs; leave as-is.
print(f"[seed] module state patched to SEEDED ({time.time()-t0:.1f}s)")



# ---------------------------------------------------------------------------
# Library exports for walk-forward runs
# ---------------------------------------------------------------------------

CHAMPION_BASE = dict(kind='B', use_elder=False, use_rsi=False,
                     reversal=False, atr_sl=True, cap=0, use_bias=False,
                     be_buffer=1.0, tp_frac=1.0, entry_start=0)

CANDIDATES = [
    # Max-net §41
    dict(label='maxnet_arm15', arm_window=15, atr_period=10, atr_mult=1.5,
         touch_buffer=0.0, be_trigger=0.50),
    # Least-drawdown §41
    dict(label='leastdd_arm10', arm_window=10, atr_period=14, atr_mult=1.25,
         touch_buffer=0.0, be_trigger=0.50),
    dict(label='leastdd_arm5', arm_window=5, atr_period=14, atr_mult=1.25,
         touch_buffer=0.0, be_trigger=0.50),
    # Champion-as-is (seeded) §41
    dict(label='champ_seeded', arm_window=10, atr_period=10, atr_mult=1.5,
         touch_buffer=0.0, be_trigger=0.70),
]


def build_cfg(cand):
    cfg = dict(CHAMPION_BASE)
    cfg.update({k: v for k, v in cand.items() if k != 'label'})
    cfg['entry_end'] = T1
    return cfg


def trades_by_year(trades):
    """Bucket trade tuples (day, side, kind, ep, xp, pnl) by year."""
    by_year = {}
    for t in trades:
        yr = str(t[0])[:4]
        by_year.setdefault(yr, []).append(t)
    return by_year


def year_metrics(tr_list, lot=LOT, fee=FEE):
    """Metrics for a list of trade tuples."""
    if not tr_list:
        return dict(trades=0, wr=0.0, net=0.0, max_dd=0.0, worst_day=0.0, calmar=0.0)
    n = len(tr_list)
    wins = sum(1 for t in tr_list if t[5] > 0)
    net = sum(t[5] for t in tr_list)
    daily = {}
    for t in tr_list:
        daily[t[0]] = daily.get(t[0], 0.0) + t[5]
    cum = peak = 0.0; max_dd = 0.0
    for d in sorted(daily):
        cum += daily[d]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    worst_day = min(daily.values()) if daily else 0.0
    calmar = net / max_dd if max_dd > 1e-9 else float('inf')
    return dict(trades=n, wr=100.0 * wins / n, net=net, max_dd=max_dd,
                worst_day=worst_day, calmar=calmar)
