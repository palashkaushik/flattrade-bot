"""
DYNAMIC-STRIKE SEEDED ENGINE v2 — trade-time 2nd-ITM (user-verified rule).

RULE (matches live bot):
  ATM(t) = round(index_close(t) / 50) * 50; active CE = ATM-100, PE = ATM+100.
  Indicators are PER-STRIKE seeded chains (prior day's last 300 bars of that
  strike's own front-weekly contract, then today's bars).
  One position per (config, day) across both sides (engine parity: pos_side).
  A position keeps ITS OWN strike until exit (live holds the bought contract).
  Gates: 'ema20' (§43) or 'full10' (CPR/Cam/PDHL statics from the strike's own
  prior-day OHLC + EMA20/EMA200/VWAP).

Structure: build per-strike-day tensors + seeded indicators ONCE; the sim core
runs a single time loop with per-(b,d) state, gathering active-strike values.
"""
import sys, time, os
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
os.environ['LH_BIAS'] = '0'
os.environ['LH_BIAS_OVERRIDE'] = '0'

import numpy as np
import pandas as pd
import polars as pl
import torch

torch.set_float32_matmul_precision('high')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMOKE = '--smoke' in sys.argv

PARQ = os.environ.get("DYN_PARQ", r"C:\Users\user\AppData\Local\Temp\opencode\data\nifty50_options_master.parquet")
IDX = os.environ.get("DYN_IDX", r"C:\Users\user\AppData\Local\Temp\opencode\data\NIFTY 50_minute.csv")
SS, SE = 555, 900
T1 = SE - SS
SEED_BARS = 300
TF_LIST = [1, 2, 3, 5]
_TF_LCM = 30
LOT, FEE = 65, 45
S1_K, S1_D = 12, 3
S3_K, S3_D = 40, 4
S4_K, S4_D = 50, 10
ARM_S1 = 25.0
TP_CAP = 15.0
MO = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}

t0 = time.time()

# ---------------------------------------------------------------------------
# Data loading (index + front-weekly)
# ---------------------------------------------------------------------------
print("[dyn] loading index 1m...")
idx = pd.read_csv(IDX)
idx["dt"] = pd.to_datetime(idx["date"])
idx["d"] = idx["dt"].dt.strftime("%Y-%m-%d")
idx["m"] = idx["dt"].dt.hour * 60 + idx["dt"].dt.minute
ses = idx[(idx["m"] >= SS) & (idx["m"] < SE)]

print("[dyn] loading parquet (front-weekly)...")
df = (pl.scan_parquet(PARQ).filter(pl.col("side").is_in(["CE", "PE"]))
      .select(["day", "minute", "symbol", "strike", "side", "open", "high", "low", "close"])
      .collect())
df = df.with_columns(pl.col("symbol").str.extract(r"NIFTY(\d{2}[A-Z]{3}\d{2})", 1).alias("exp_token"))

def tok_date(tok):
    return pd.Timestamp(2000 + int(tok[5:7]), MO[tok[2:5]], int(tok[0:2]))

toks = [t for t in df.select("exp_token").unique().to_series().to_list() if t]
tok_dates = {t: tok_date(t) for t in toks}
days = sorted(set(df["day"]) & set(ses["d"]))
_f = os.environ.get('DYN_DAY_FIRST', '2020-01-01')
_l = os.environ.get('DYN_DAY_LAST', '2026-08-27')
days = [d for d in days if _f <= d <= _l]
if SMOKE:
    days = days[-5:]
D = len(days)
day_i = {d: i for i, d in enumerate(days)}
day_exp = {}
for d in days:
    dd = pd.Timestamp(d).date()
    cands = [t for t, dt in tok_dates.items() if dt.date() >= dd]
    day_exp[d] = min(cands, key=lambda t: tok_dates[t]) if cands else None
df = df.with_columns(pl.col("day").replace(day_exp).alias("te"))
fw = df.filter((pl.col("exp_token") == pl.col("te")) & pl.col("day").is_in(days)).to_pandas()
fw["di"] = fw["day"].map(day_i)
fw["strike"] = fw["strike"].astype(int)
fw = fw[(fw["minute"] >= SS - SEED_BARS) & (fw["minute"] < SE)]
print(f"[dyn] D={D} rows={len(fw):,}")

pv = ses.pivot_table(index="d", columns="m", values="close").reindex(days).ffill(axis=1).bfill(axis=1)
close_arr = pv.values.astype(np.float64)
atm_dyn = np.round(close_arr / 50.0) * 50.0

# ---------------------------------------------------------------------------
# Strike-day tensors — keyed by (di, strike, side, WEEKLY-TOKEN).
# SEEDED-LIVE PARITY (user directive): the live bot warms each contract token
# with the SAME weekly contract's prior-day LAST 300 session bars (10:00-15:00
# = the final 300 of 345), matching what TradingView shows for that contract.
# A brand-new weekly token has no prior-day tape -> COLD indicators that day
# (TV shows the same for a newly listed contract).
# ---------------------------------------------------------------------------
sd_keys = sorted(set(zip(fw["di"], fw["strike"], fw["side"], fw["te"])))
sd_row = {k: i for i, k in enumerate(sd_keys)}
N = len(sd_keys)
print(f"[dyn] strike-days N={N:,} (keyed with weekly token)")

T_ext = SEED_BARS + T1
pad_need = (-T_ext) % _TF_LCM
T_ext_p = T_ext + pad_need

def build_side(side):
    o = np.full((N, T_ext_p), np.nan, dtype=np.float32)
    h = np.full((N, T_ext_p), np.nan, dtype=np.float32)
    l = np.full((N, T_ext_p), np.nan, dtype=np.float32)
    c = np.full((N, T_ext_p), np.nan, dtype=np.float32)
    sub = fw[fw["side"] == side]
    # VECTORIZED row lookup (was: 25M-row Python itertuples loop — the #1
    # bottleneck of the 443s data phase). Map (di, strike, token) -> row via
    # a dict probe on numpy arrays; ~40x faster, identical result.
    di_arr = sub["di"].to_numpy()
    st_arr = sub["strike"].to_numpy()
    tk_arr = sub["te"].to_numpy()
    # VECTORIZED row lookup (was: 25M-row Python itertuples loop — the #1
    # bottleneck of the 443s data phase). Pure-dict probe over zip of numpy
    # columns skips ALL pandas attribute overhead; identical result.
    r_idx = np.fromiter(
        (sd_row.get((d, s, side, t), -1) for d, s, t in zip(di_arr, st_arr, tk_arr)),
        dtype=np.int64, count=len(sub))
    keep = r_idx >= 0
    sub = sub[keep]; r_idx = r_idx[keep]
    pos = sub["minute"].to_numpy() - (SS - SEED_BARS)
    # DIRECT ASSIGNMENT (np.add.at was a silent NO-OP: nan + price = nan —
    # every session bar "landed" as NaN, killing all downstream state)
    o[r_idx, pos] = sub["open"].to_numpy(np.float32)
    h[r_idx, pos] = sub["high"].to_numpy(np.float32)
    l[r_idx, pos] = sub["low"].to_numpy(np.float32)
    c[r_idx, pos] = sub["close"].to_numpy(np.float32)
    return o, h, l, c

def attach_prior(o, h, l, c):
    """Seeds row i's [0,300) with row j's LAST 300 session bars, where j is the
    PRIOR trading day, SAME strike, SAME side, SAME weekly token (TV parity).
    Rows with no same-token prior (new weeklies) stay seed-empty -> the ffill
    in fbfill backfills from the first session bar = COLD start (TV behavior
    for a newly listed contract)."""
    by_key = {}
    for i, k in enumerate(sd_keys):
        by_key[k[:2] + (k[2], k[3])] = i      # (di, strike, side, token) -> row
    o2, h2, l2, c2 = o.copy(), h.copy(), l.copy(), c.copy()
    n_fill = n_cold = 0
    # VECTORIZED (was 234k-row Python loop — the #2 bottleneck of the 443s
    # data phase). Precompute prior-day row map, gather seeds in bulk.
    import numpy as _np
    rows = _np.array([by_key.get((k[0] - 1, k[1], k[2], k[3]), -1) for k in sd_keys], dtype=_np.int64)
    has_prior = rows >= 0
    # prior-day seed validity: prior row's [345,645) close must be non-NaN
    valid = _np.zeros(N, dtype=bool)
    r_ok = rows[has_prior]
    if len(r_ok):
        valid[has_prior] = ~_np.isnan(c[r_ok, 345:645]).all(axis=1)
    fill_mask = has_prior & valid
    r_fill = rows[fill_mask]
    d_fill = _np.nonzero(fill_mask)[0]
    if len(d_fill):
        o2[d_fill, :SEED_BARS] = o[r_fill, 345:645]
        h2[d_fill, :SEED_BARS] = h[r_fill, 345:645]
        l2[d_fill, :SEED_BARS] = l[r_fill, 345:645]
        c2[d_fill, :SEED_BARS] = c[r_fill, 345:645]
    n_fill = int(fill_mask.sum()); n_cold = N - n_fill
    print(f"  seed: {n_fill:,} rows seeded (same-token prior tail) | "
          f"{n_cold:,} rows COLD (new weekly / no prior tape — TV parity)")
    return o2, h2, l2, c2

def fbfill(x):
    t = torch.tensor(x, device=DEVICE)
    nan_mask = torch.isnan(t)
    if nan_mask.any():
        idx = torch.arange(t.shape[1], device=DEVICE, dtype=torch.float32).unsqueeze(0).expand_as(t)
        idx = torch.where(nan_mask, torch.full_like(idx, -1e9), idx)
        ff = idx.cummax(dim=1).values.long().clamp(min=0)
        t = torch.where(nan_mask, t.gather(1, ff), t)
        nan_mask = torch.isnan(t)
        if nan_mask.any():
            first_valid = (~torch.isnan(t)).float().argmax(dim=1, keepdim=True)
            t = torch.where(nan_mask, t.gather(1, first_valid).expand_as(t), t)
    return t.contiguous()

CACHE_DIR = os.environ.get('DYN_CACHE', r'C:\Websites\FLATTRADE BOT\.dyn_cache')

def _cache_key():
    import hashlib
    h = hashlib.sha1()
    for p in (PARQ, IDX):
        h.update(str(os.path.getmtime(p)).encode()); h.update(str(os.path.getsize(p)).encode())
    h.update(f"{SEED_BARS}|{SS}|{SE}|{S1_K},{S1_D}|{S3_K},{S3_D}|{S4_K},{S4_D}".encode())
    h.update(os.environ.get('DYN_DAY_FIRST', '').encode())
    h.update(os.environ.get('DYN_DAY_LAST', '').encode())
    h.update(os.environ.get('DYN_M6_S1', '79.5').encode())
    h.update(b"|smoke=" + (b"1" if SMOKE else b"0"))  # POISONED-CACHE FIX: the
    # 2026-09-04 incident — a --smoke run (5 days) SAVED under the full-range
    # key; the next full run loaded 5 days of 2026 and "swept" in 2s.
    h.update(b"v2")  # schema version — bump on build-logic changes
    return h.hexdigest()[:16]

def _cache_path():
    return os.path.join(CACHE_DIR, f"dyn_{_cache_key()}.pt")

tensors = {}
# DISK CACHE IS OPT-IN (DYN_CACHE_OUT=1) — default OFF. The 2026-09-04
# incident: 12-20GB .pt caches filled the user's disk. Normal runs build
# in-memory in ~3-4 min (vectorized); nobody needs the disk cache anymore.
_use_cache = (os.environ.get('DYN_CACHE_OUT') == '1'
              and os.environ.get('DYN_NO_CACHE') != '1'
              and os.environ.get('DYN_FORCE_STATIC') != '1')
if _use_cache and os.path.exists(_cache_path()):
    print(f"[dyn] CACHE HIT — loading prebuilt tensors from {_cache_path()}")
    _bundle = torch.load(_cache_path(), map_location=DEVICE, weights_only=False)
    tensors = _bundle["tensors"]
    sd_keys = _bundle["sd_keys"]; N = len(sd_keys)
    sd_row = _bundle["sd_row"]
    days = _bundle["days"]; day_i = _bundle["day_i"]; day_exp = _bundle["day_exp"]
    atm_dyn = _bundle["atm_dyn"]
    D = len(days)
    _act_cache = _bundle.get("act")
    print(f"[dyn] cache restore N={N:,} D={D} ({time.time()-t0:.1f}s)")
else:
    _act_cache = None
    for side in ("CE", "PE"):
        o, h, l, c = build_side(side)
        o, h, l, c = attach_prior(o, h, l, c)
        h_t = fbfill(h); l_t = fbfill(l); c_t = fbfill(c)
        tensors[side] = dict(h=h_t, l=l_t, c=c_t,
                             h_s=h_t[:, SEED_BARS:SEED_BARS + T1].contiguous(),
                             l_s=l_t[:, SEED_BARS:SEED_BARS + T1].contiguous(),
                             c_s=c_t[:, SEED_BARS:SEED_BARS + T1].contiguous())
    print(f"[dyn] tensors ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Seeded indicators per strike-day
# ---------------------------------------------------------------------------
def stoch_d(h3, l3, c3, k, d_period):
    h_pad = torch.nn.functional.pad(h3.unsqueeze(1), (k - 1, 0), mode="replicate")
    l_pad = torch.nn.functional.pad(l3.unsqueeze(1), (k - 1, 0), mode="replicate")
    max_h = torch.nn.functional.max_pool1d(h_pad, k, stride=1).squeeze(1)
    min_l = -torch.nn.functional.max_pool1d(-l_pad, k, stride=1).squeeze(1)
    denom = (max_h - min_l).clamp(min=1e-6)
    fk = (c3 - min_l) / denom * 100.0
    k_pad = torch.nn.functional.pad(fk.unsqueeze(1), (d_period - 1, 0), mode="replicate")
    return torch.nn.functional.avg_pool1d(k_pad, d_period, stride=1).squeeze(1)

def ema_scan(c3, period):
    alpha = 2.0 / (period + 1)
    ema = torch.empty_like(c3); ema[:, 0] = c3[:, 0]
    for i in range(1, c3.shape[1]):
        ema[:, i] = ema[:, i - 1] * (1.0 - alpha) + c3[:, i] * alpha
    return ema

def atr_scan(h3, l3, c3, period):
    prev = torch.empty_like(c3); prev[:, 0] = c3[:, 0]; prev[:, 1:] = c3[:, :-1]
    tr = torch.maximum(h3 - l3, torch.maximum(torch.abs(h3 - prev), torch.abs(l3 - prev)))
    atr = torch.empty_like(tr); alpha = 2.0 / (period + 1)
    atr[:, 0] = tr[:, 0]
    for i in range(1, tr.shape[1]):
        atr[:, i] = atr[:, i - 1] * (1.0 - alpha) + tr[:, i] * alpha
    return atr

def vwap_s(h_t, l_t, c_t):
    """Session VWAP — resets at the day boundary (index SEED_BARS)."""
    tp = (h_t + l_t + c_t) / 3.0
    cs = torch.cumsum(tp, dim=1)
    base = cs[:, SEED_BARS - 1:SEED_BARS - 1 + T1]
    vwap = (cs[:, SEED_BARS:SEED_BARS + T1] - base) / \
        torch.arange(1, T1 + 1, device=c_t.device, dtype=torch.float32).unsqueeze(0)
    return vwap

# CHUNKED indicator build — N=234k rows x 660 bars in fp32 blows 12GB VRAM if
# done whole (pool temporaries x4 TFs). Stream row-chunks, accumulate into
# preallocated (N, T1) outputs, free intermediates per chunk.
IND_CHUNK = 32768

def build_indicators_chunked(side):
    h_t, l_t, c_t = tensors[side]["h"], tensors[side]["l"], tensors[side]["c"]
    print(f"[dyn] indicators {side} (chunked, N={N})...")
    out = dict(
        s1_1m=torch.zeros(N, T1, device=DEVICE),
        atr10=torch.zeros(N, T1, device=DEVICE),
        ema20=torch.zeros(N, T1, device=DEVICE),
        ema200=torch.zeros(N, T1, device=DEVICE),
        vwap=torch.zeros(N, T1, device=DEVICE),
        super_m=torch.zeros(N, T1, dtype=torch.bool, device=DEVICE),
        m6=torch.zeros(N, T1, dtype=torch.bool, device=DEVICE),
        p_h=torch.zeros(N, device=DEVICE),
        p_l=torch.zeros(N, device=DEVICE),
        p_c=torch.zeros(N, device=DEVICE),
        lo_2=torch.zeros(N, T1, device=DEVICE), cl_2=torch.zeros(N, T1, device=DEVICE),
        lo_3=torch.zeros(N, T1, device=DEVICE), cl_3=torch.zeros(N, T1, device=DEVICE),
        lo_5=torch.zeros(N, T1, device=DEVICE), cl_5=torch.zeros(N, T1, device=DEVICE),
    )
    for s0 in range(0, N, IND_CHUNK):
        s1_, s2_ = s0, min(s0 + IND_CHUNK, N)
        hc, lc, cc = h_t[s1_:s2_], l_t[s1_:s2_], c_t[s1_:s2_]
        sup_c = torch.zeros(s2_ - s1_, T1, dtype=torch.bool, device=DEVICE)
        m6_c = torch.zeros(s2_ - s1_, T1, dtype=torch.bool, device=DEVICE)
        for tf in TF_LIST:
            n = (T_ext_p // tf) * tf
            c_ = cc[:, :n].reshape(-1, n // tf, tf)
            h_ = hc[:, :n].reshape(-1, n // tf, tf)
            l_ = lc[:, :n].reshape(-1, n // tf, tf)
            cl = c_[:, :, -1]; hi = h_.amax(dim=2); lo = l_.amin(dim=2)
            t1_ = stoch_d(hi, lo, cl, S1_K, S1_D)
            t3_ = stoch_d(hi, lo, cl, S3_K, S3_D)
            t4_ = stoch_d(hi, lo, cl, S4_K, S4_D)
            rising = torch.zeros_like(t1_)
            rising[:, 1:] = (t1_[:, 1:] > t1_[:, :-1]).float()
            def exp(a):
                return a.repeat_interleave(tf, dim=1)[:, SEED_BARS:SEED_BARS + T1]
            e1, e3, e4, er = exp(t1_), exp(t3_), exp(t4_), exp(rising)
            sup_c |= (e3 < 25.0) & (e4 < 25.0) & (e1 < 25.0) & (er > 0.5)
            # FLAG S1 leg: 79.5 (§44 reference) or 25.0 (KNOWLEDGE SWEEP Sep-4:
            # "S1<25 FLAG" — user variant). Env-selectable at import time.
            _m6_s1 = float(os.environ.get('DYN_M6_S1', '79.5'))
            m6_c |= (e4 >= 79.5) & (e1 < _m6_s1)
            # TF bucket lo/cl (BOUNCE GATE PARITY: the §43/live gate accepts a
            # touch on the 1m bar OR any completed 2m/3m/5m bucket; tf=1's
            # "bucket" IS the 1m bar itself -> already in l_s/c_s)
            if tf > 1:
                # SESSION-ALIGNED buckets (live TFTracker / make_tf_stoch
                # convention): chunk the SESSION region only, from 09:15 (bar
                # SEED_BARS of the extended axis). The extended-axis chunking
                # aligned buckets to the SEED boundary — wrong contents at
                # session start (straddling seed+session bars) -> 199-bar
                # bounce divergence caught by parity bisection.
                n_s = (T1 // tf) * tf
                cs_ = cc[:, SEED_BARS:SEED_BARS + T1][:, :n_s].reshape(-1, T1 // tf, tf)
                hs_ = hc[:, SEED_BARS:SEED_BARS + T1][:, :n_s].reshape(-1, T1 // tf, tf)
                ls_ = lc[:, SEED_BARS:SEED_BARS + T1][:, :n_s].reshape(-1, T1 // tf, tf)
                blo = ls_.amin(dim=2)          # (chunk, T1//tf) bucket lo
                bcl = cs_[:, :, -1]            # bucket close
                def exp_s(a):                  # repeat bucket value over its bars
                    e = a.repeat_interleave(tf, dim=1)
                    if e.shape[1] < T1:
                        e = torch.nn.functional.pad(e, (0, T1 - e.shape[1]), mode="replicate")
                    return e
                out[f"lo_{tf}"][s1_:s2_] = exp_s(blo)
                out[f"cl_{tf}"][s1_:s2_] = exp_s(bcl)
                del cs_, hs_, ls_, blo, bcl
            del c_, h_, l_, cl, hi, lo, t1_, t3_, t4_, rising, e1, e3, e4, er
        out["s1_1m"][s1_:s2_] = stoch_d(hc, lc, cc, S1_K, S1_D)[:, SEED_BARS:SEED_BARS + T1]
        out["atr10"][s1_:s2_] = atr_scan(hc, lc, cc, 10)[:, SEED_BARS:SEED_BARS + T1]
        out["ema20"][s1_:s2_] = ema_scan(cc, 20)[:, SEED_BARS:SEED_BARS + T1]
        out["ema200"][s1_:s2_] = ema_scan(cc, 200)[:, SEED_BARS:SEED_BARS + T1]
        # session VWAP (chunk-local)
        tp = (hc + lc + cc) / 3.0
        cs = torch.cumsum(tp, dim=1)
        base = cs[:, SEED_BARS - 1:SEED_BARS - 1 + T1]
        out["vwap"][s1_:s2_] = (cs[:, SEED_BARS:SEED_BARS + T1] - base) / \
            torch.arange(1, T1 + 1, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        out["p_h"][s1_:s2_] = hc[:, :SEED_BARS].amax(dim=1)
        out["p_l"][s1_:s2_] = lc[:, :SEED_BARS].amin(dim=1)
        out["p_c"][s1_:s2_] = cc[:, SEED_BARS - 1]
        out["super_m"][s1_:s2_] = sup_c
        out["m6"][s1_:s2_] = m6_c
        del hc, lc, cc, sup_c, m6_c, tp, cs, base
        if s2_ % (IND_CHUNK * 4) == 0 or s2_ == N:
            print(f"    rows {s2_:,}/{N:,} ({time.time()-t0:.0f}s)")
    for k in ("s1_1m", "atr10", "ema20", "ema200", "vwap", "super_m", "m6",
              "p_h", "p_l", "p_c", "lo_2", "cl_2", "lo_3", "cl_3", "lo_5", "cl_5"):
        out[k] = out[k].contiguous()
    return out

_cache_hit = None  # (cache-hit path skips builds via 'm6 in tensors' checks)

for side in ("CE", "PE"):
    if "m6" in tensors[side]:
        continue  # cache hit: indicators already present
    tensors[side].update(build_indicators_chunked(side))
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
print(f"[dyn] indicators done ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Active-strike maps (D, T1)
# ---------------------------------------------------------------------------
strike_of_row = {}
for i, (di, strike, side, tok) in enumerate(sd_keys):
    strike_of_row.setdefault((di, side), {})[strike] = i
act = {}
FORCE_STATIC = os.environ.get('DYN_FORCE_STATIC') == '1'
if _act_cache is not None and not FORCE_STATIC:
    act = _act_cache
    print("[dyn] act maps restored from cache")
elif FORCE_STATIC:
    # PARITY MODE: pin the active strike to the 09:15 2nd-ITM pair for the
    # whole day — the dyn engine then degenerates EXACTLY to the static
    # engine's contract choice (engine-equivalence verification).
    print("[dyn] FORCE_STATIC: active strike pinned to 09:15 2nd-ITM (parity mode)")
for side, sign in (("CE", -100), ("PE", +100)):
    a = np.full((D, T1), -1, dtype=np.int64)
    for d in range(D):
        m = strike_of_row.get((d, side), {})
        atm = atm_dyn[d]
        if FORCE_STATIC:
            # 09:15 (bar 0) ATM defines the pair for the entire day
            a0 = atm[0] if not np.isnan(atm[0]) else None
            if a0 is not None:
                r = m.get(int(a0) + sign, -1)
                if r >= 0:
                    a[d, :] = r
        else:
            for t in range(T1):
                if np.isnan(atm[t]):
                    continue
                r = m.get(int(atm[t]) + sign, -1)
                if r >= 0:
                    a[d, t] = r
    act[side] = torch.tensor(a, device=DEVICE)
print(f"[dyn] coverage CE {100*(act['CE']>=0).float().mean():.2f}% PE {100*(act['PE']>=0).float().mean():.2f}%")

# ---- persist the full prebuilt state (tensors + indicators + maps) ----
# OPT-IN ONLY (DYN_CACHE_OUT=1) — see the note at the cache-load block.
if _use_cache and not os.path.exists(_cache_path()):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        torch.save({
            "tensors": tensors,
            "sd_keys": sd_keys, "sd_row": sd_row,
            "days": days, "day_i": day_i, "day_exp": day_exp,
            "atm_dyn": atm_dyn,
            "act": act,
        }, _cache_path())
        print(f"[dyn] cache SAVED -> {_cache_path()} ({time.time()-t0:.1f}s)")
    except Exception as _e:
        print(f"[dyn] cache save failed (continuing): {_e}")

# ---------------------------------------------------------------------------
# Bounce gates per strike-day
# ---------------------------------------------------------------------------
def bounce_gate(side, gate, tb, levels=None):
    """§43/live gate parity: a touch counts if the 1m bar (low/close) OR any
    completed 2m/3m/5m bucket (tf-lo/tf-cl) pierces the level and reclaims it.
    levels=None -> 'ema20'/'full10' legacy behaviour. levels=('EMA20','VWAP')
    -> pairwise-union gates (KNOWLEDGE SWEEP Sep-4 2026)."""
    T = tensors[side]
    lo, cl = T["l_s"], T["c_s"]
    cond = torch.zeros_like(lo, dtype=torch.bool)
    def add(lev):
        nonlocal cond  # `cond |= x` REBINDS (Python augmented assignment on closure)
        cond = cond | ((lo <= lev + tb) & (cl >= lev - 0.5))
        for tf in (2, 3, 5):
            tlo, tcl = T[f"lo_{tf}"], T[f"cl_{tf}"]
            cond = cond | ((tlo <= lev + tb) & (tcl >= lev - 0.5))
    if levels is not None:
        # ---- KNOWLEDGE SWEEP: named pairwise unions with EMA20 ----
        ph, pl_, pc_ = T["p_h"], T["p_l"], T["p_c"]
        pivot = (ph + pl_ + pc_) / 3.0
        bc = (ph + pl_) / 2.0
        tc = 2.0 * pivot - bc
        rng = ph - pl_
        for name in levels:
            if name == "EMA20":
                add(T["ema20"])
            elif name == "VWAP":
                add(T["vwap"])
            elif name == "EMA200":
                add(T["ema200"])
            elif name == "PIVOT":
                add(pivot.unsqueeze(1))
            elif name == "CPR":
                add(bc.unsqueeze(1)); add(pivot.unsqueeze(1)); add(tc.unsqueeze(1))
            elif name == "CAM_S3R3":  # Camarilla 1.1/4 R3/S3
                add((pc_ + rng * 1.1 / 4.0).unsqueeze(1))
                add((pc_ - rng * 1.1 / 4.0).unsqueeze(1))
            elif name == "PDH_PDL":
                add(ph.unsqueeze(1)); add(pl_.unsqueeze(1))
        return cond
    if gate in ("ema20", "full10"):
        add(T["ema20"])
    if gate == "full10":
        # statics from the strike's own prior-day OHLC
        ph, pl_, pc_ = T["p_h"], T["p_l"], T["p_c"]
        pivot = (ph + pl_ + pc_) / 3.0
        bc = (ph + pl_) / 2.0
        tc = 2.0 * pivot - bc
        rng = ph - pl_
        for lev in (bc, pivot, tc,
                    pc_ + rng * 1.1 / 4.0, pc_ - rng * 1.1 / 4.0,
                    ph, pl_):
            add(lev.unsqueeze(1))
        add(T["ema200"]); add(T["vwap"])
    return cond

# ---------------------------------------------------------------------------
# SIM CORE — single time loop, one pos per (b,d), position keeps its strike
# ---------------------------------------------------------------------------
def dyn_sim(configs, gate="ema20", tb=0.0, require_trig=True, gate_levels=None):
    B = len(configs)
    arm_b = torch.tensor([float(c['arm_window']) for c in configs], device=DEVICE)
    mult_b = torch.tensor([float(c['atr_mult']) for c in configs], device=DEVICE)
    be_b = torch.tensor([float(c['be_trigger']) for c in configs], device=DEVICE)
    be_buf_b = torch.tensor([float(c.get('be_buffer', 1.0)) for c in configs], device=DEVICE)
    bcol = lambda x: x[:, None]

    bnc = {s: bounce_gate(s, gate, tb, levels=gate_levels) for s in ("CE", "PE")}
    # active-value tensors (D, T1) per side (gathered from strike-day rows)
    AV = {}
    for s in ("CE", "PE"):
        T = tensors[s]; a = act[s].clamp(min=0); valid = act[s] >= 0
        g = lambda x: torch.gather(x, 0, a) * valid
        AV[s] = dict(
            s1=g(T["s1_1m"]), sup=g(T["super_m"].float()).bool(), m6=g(T["m6"].float()).bool(),
            atr=g(T["atr10"]), l=g(T["l_s"]), c=g(T["c_s"]), bnc=g(bnc[s].float()).bool(),
            valid=valid,
        )

    day_pnl = torch.zeros(B, D, device=DEVICE)
    tr_ct = torch.zeros(B, D, device=DEVICE)
    win_ct = torch.zeros(B, D, device=DEVICE)
    trade_log = [[] for _ in range(B)]  # (day_idx, side, kind, entry, exit, pnl, entry_bar, exit_bar)

    pos_open = torch.zeros(B, D, dtype=torch.bool, device=DEVICE)   # any position open
    pos_side = torch.zeros(B, D, dtype=torch.int8, device=DEVICE)   # 0 flat, 1 PE, 2 CE
    pos_row = torch.zeros(B, D, dtype=torch.long, device=DEVICE)   # strike-day row of position
    entry_px = torch.zeros(B, D, device=DEVICE)
    sl_px = torch.zeros(B, D, device=DEVICE)
    tp_px = torch.zeros(B, D, device=DEVICE)
    be_done = torch.zeros(B, D, dtype=torch.bool, device=DEVICE)
    # PER-SIDE arming (static engine has pe_flag_armed/ce_flag_armed INDEPENDENT;
    # the live bot has per-contract state). A SINGLE shared arm made the sides
    # clear each other's arming every bar via the arm_row check (dyn 11 vs
    # static 19 trades on the parity day — root cause).
    arm = {s: torch.zeros(B, D, dtype=torch.bool, device=DEVICE) for s in ("PE", "CE")}
    arm_t = {s: torch.full((B, D), -999, dtype=torch.int32, device=DEVICE) for s in ("PE", "CE")}
    arm_row = {s: torch.zeros(B, D, dtype=torch.long, device=DEVICE) for s in ("PE", "CE")}

    for t in range(1, T1):
        # ---- EXITS on the POSITION strike's prices ----
        prow = pos_row.clamp(min=0)
        pe_pos = pos_side == 1
        ce_pos = pos_side == 2
        pl_pos = torch.where(pe_pos, tensors["PE"]["l_s"][prow, t], tensors["CE"]["l_s"][prow, t])
        ph_pos = torch.where(pe_pos, tensors["PE"]["h_s"][prow, t], tensors["CE"]["h_s"][prow, t])
        sl_hit = pos_open & (pl_pos <= sl_px)
        tp_hit = pos_open & (~sl_hit) & (ph_pos >= tp_px)
        do_sl = sl_hit
        do_tp = pos_open & (~sl_hit) & tp_hit
        for mask, xp in ((do_sl, sl_px), (do_tp, tp_px)):
            pnl = (xp - entry_px) * LOT - FEE
            safe = torch.where(mask, torch.nan_to_num(pnl, nan=0.0), torch.zeros_like(pnl))
            day_pnl.add_(safe)
            tr_ct.add_(mask.float())
            win_ct.add_((mask & (safe > 0)).float())
            if mask.any():
                idx = mask.nonzero(as_tuple=False)
                kind = 'SL' if mask is do_sl else 'TP'
                for b_i, d_i in idx.tolist():
                    trade_log[b_i].append((d_i, int(pos_side[b_i, d_i].item()),
                                           kind, float(entry_px[b_i, d_i].item()),
                                           float(xp[b_i, d_i].item()),
                                           float(safe[b_i, d_i].item()), t))
        done = do_sl | do_tp
        pos_open &= ~done
        pos_side[done] = 0
        pos_row[done] = 0
        be_done &= ~done
        for s in ("PE", "CE"):
            # reference semantics (gpu_sim:371-374): ANY exit clears BOTH
            # sides' arming for that (b,d)
            arm[s] &= ~done
        # ---- BE ratchet ----
        dist0 = entry_px - sl_px
        be_px = entry_px + bcol(be_b) * dist0
        trig = (bcol(be_b) > 0) & (~be_done) & pos_open & (ph_pos >= be_px)
        sl_px = torch.where(trig, entry_px + bcol(be_buf_b), sl_px)
        be_done |= trig
        # ---- ENTRIES per side (INDEPENDENT per-side arming) ----
        for s, code in (("PE", 1), ("CE", 2)):
            AVs = AV[s]
            a_t = act[s][:, t]                      # (D,) active strike row (-1 if none)
            flat = pos_side == 0
            a_ok = a_t >= 0
            a_cl = a_t.clamp(min=0)
            # arming persists while the SAME strike stays active (per-contract
            # state); a strike switch resets that side's arming
            same_row = (arm_row[s] == a_cl.unsqueeze(0).expand(B, -1))
            arm[s] = arm[s] & same_row
            # arm when the ACTIVE strike's S1(1m) <= 25, flat only
            pa_col = (AVs["s1"][:, t] <= ARM_S1) & a_ok               # (D,)
            arm_new = pa_col.unsqueeze(0) & flat & (~pos_open)
            arm[s] = arm[s] | arm_new
            arm_t[s] = torch.where(arm_new, t, arm_t[s])
            arm_row[s] = torch.where(arm_new, a_cl.unsqueeze(0).expand(B, -1), arm_row[s])
            arm[s] = arm[s] & ((t - arm_t[s]) <= bcol(arm_b))
            # require_trig=False (KNOWLEDGE SWEEP, Sep-4 2026): the 10:08→10:15
            # case — armed, gate passed (VWAP touch+reclaim), but the stochastic
            # trigger had evaporated (deep crash tanks S4). This variant enters
            # on ARM + BOUNCE + FLAT alone: "when S4 is dead, still take it".
            # Default True = §44 semantics (trigger mandatory) — unchanged.
            if require_trig:
                trig = arm[s] & (AVs["m6"][:, t] | AVs["sup"][:, t]) & AVs["bnc"][:, t] & flat & a_ok
            else:
                trig = arm[s] & AVs["bnc"][:, t] & flat & a_ok
            ent = trig
            if bool(ent.any()):
                ep = AVs["c"][:, t].unsqueeze(0).expand(B, -1)
                dist = torch.clamp(torch.min(AVs["atr"][:, t].unsqueeze(0).expand(B, -1) * bcol(mult_b),
                                             torch.tensor(TP_CAP, device=DEVICE)), min=2.0)
                entry_px = torch.where(ent, ep, entry_px)
                sl_px = torch.where(ent, ep - dist, sl_px)
                tp_px = torch.where(ent, ep + dist, tp_px)
                pos_open |= ent
                pos_side = torch.where(ent, torch.tensor(code, dtype=torch.int8, device=DEVICE), pos_side)
                pos_row = torch.where(ent, a_cl.unsqueeze(0).expand(B, -1), pos_row)
                be_done &= ~ent
                arm[s] &= ~ent
    if os.environ.get('DYN_TRADE_LOG') == '1':
        return day_pnl, tr_ct, win_ct, trade_log
    return day_pnl, tr_ct, win_ct

def metrics(dp_row, tc_row, wc_row):
    tc = tc_row.cpu().numpy(); wc = wc_row.cpu().numpy(); dp = dp_row.cpu().numpy()
    n = int(tc.sum())
    if n == 0:
        return dict(trades=0, wr=0.0, net=0.0, max_dd=0.0, worst_day=0.0, calmar=0.0)
    net = float(dp.sum()); wins = int(wc.sum())
    cum = peak = 0.0; max_dd = 0.0
    for v in dp:
        cum += v
        if cum > peak: peak = cum
        max_dd = max(max_dd, peak - cum)
    return dict(trades=n, wr=100.0 * wins / n, net=net, max_dd=max_dd,
                worst_day=float(dp.min()), calmar=net / max_dd if max_dd > 1e-9 else float('inf'))

if __name__ == "__main__":
    # SMOKE: champion §43 config on 5 days
    cfg = [dict(arm_window=10, atr_mult=1.0, be_trigger=0.60, be_buffer=1.0)]
    t1 = time.time()
    dp, tc, wc = dyn_sim(cfg, gate="ema20", tb=0.0)
    m = metrics(dp[0], tc[0], wc[0])
    print(f"\n[smoke §43 EMA20 dyn] {m}  ({time.time()-t1:.1f}s)")
    t2 = time.time()
    dp, tc, wc = dyn_sim(cfg, gate="full10", tb=0.0)
    m = metrics(dp[0], tc[0], wc[0])
    print(f"[smoke full10 dyn] {m}  ({time.time()-t2:.1f}s)")
    print(f"[total build+sim {time.time()-t0:.1f}s]")
