"""Master V4 Engine: Exact Rule-Refined 7-Year Backtest + Walk-Forward Analysis.
Rules:
  1. Non-concurrent (1 trade at a time per account).
  2. Flag (M6): S1 armed <= 25.0 -> 5 to 10 bar window -> S4 >= 79.5 (S1 < 79.5) -> SR Bounce -> Buy same side.
  3. Super Signal: S1 armed <= 25.0 -> 5 to 10 bar window -> S3 < 25 & S4 < 25 + S1 turn-up -> SR Bounce -> Buy same side.
  4. Cross-Side Reversal: S4 embedded <= 20.5 on Chart X for >= 14 bars + Super on Chart X -> Instant buy Chart Y ONLY IF Chart Y has SR bounce on exact bar (NO arming / delay).
  5. Exact SL = entry - dist, Exact TP = entry + dist (dist = min(ATR(10)*1.5, 15.0), floor 2.0).
  6. Lot = 65, Fee = 45 per trade.
  7. Option chart SR suite (10 levels: CPR BC/Pivot/TC, Camarilla H3/L3, PDH, PDL, EMA20, EMA200, VWAP).
  8. Full GPU TF32 acceleration with Causal Parity.
"""
import time, sys, os
from bias15m import build_bias_lookup, build_bias_lookup_tf
import numpy as np, polars as pl, pyarrow.parquet as pq, torch, pandas as pd

torch.set_float32_matmul_precision('high')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Master V4 Engine | Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) | TF32: high")

# Non-synced working cache. Desktop AND the repo are OneDrive-synced; reading
# the 472MB parquet from either stalls on the sync/AV scan (hangs 90s+).
# Keep canonical source at Desktop; this temp copy is a working cache.
_LOCAL = r"C:\Users\user\AppData\Local\Temp\opencode\data"
PARQUET = _LOCAL + r"\nifty50_options_master.parquet"
IDX_PATH = _LOCAL + r"\NIFTY 50_minute.csv"

ARM_WINDOW = 5  # user refinement 2026-08-26: was 1, now 5 to match 26Aug multitf
USE_RSI = False  # RSI(14) on 3m underlying (futures) chart — entry gate: CE needs rsi>60, PE needs rsi<40
# 15m Marni Fib bias: bullish->CE, bearish->PE, else none.
# The 1M-min-bar bias lookup costs ~45s to build; bias-OFF sweeps can skip it
# by setting the env var (default behavior unchanged).
USE_BIAS = os.environ.get("LH_BIAS", "1") != "0"

# Parameters
S1_K, S1_D = 12, 3
S3_K, S3_D = 40, 4
S4_K, S4_D = 50, 10

ARM_S1 = 25.0
M6_S4 = 79.5
M6_S1 = 79.5
REV_EMBED = 14

SL_PTS = 7.0
TP_PTS = 15.0
LOT, FEE = 65, 45
SESSION_START = 555  # 09:15
SESSION_END = 900    # 15:00
T1 = SESSION_END - SESSION_START

# 1. Load Index for ATM Strikes
idx_df = pd.read_csv(IDX_PATH)
idx_df['date_dt'] = pd.to_datetime(idx_df['date'])
idx_df['day'] = idx_df['date_dt'].dt.strftime('%Y-%m-%d')
idx_df['minute'] = idx_df['date_dt'].dt.hour * 60 + idx_df['date_dt'].dt.minute

spot_by_day = {}
for _, r in idx_df.iterrows():
    spot_by_day.setdefault(r["day"], {})[int(r["minute"])] = float(r["close"])

tbl = pq.read_table(PARQUET, columns=["day"])
all_days = sorted(pd.Series(tbl.column("day").to_pandas()).unique())
trading_days = [d for d in all_days if "2020-01-01" <= d <= "2026-08-27" and d in spot_by_day]

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, default=0)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--walkforward", action="store_true")
ap.add_argument("--sl", type=float, default=7.0, help="Stop-loss in points (LTP distance)")
ap.add_argument("--tp", type=float, default=15.0, help="Take-profit in points (LTP distance)")
ap.add_argument("--reversal", action="store_true", help="Enable cross-side Reversal entries (S4<=20.5 x14 bars + Super on opposite chart)")
ap.add_argument("--atr_sl", action="store_true", help="Use ATR-based SL (capped at TP) instead of fixed SL_PTS")
ap.add_argument("--atr_mult", type=float, default=1.0, help="ATR multiplier for SL (SL = min(ATR*mult, TP))")
ap.add_argument("--atr_period", type=int, default=14, help="ATR lookback period")
ap.add_argument("--bias_tf", type=str, default="15m", choices=["5m","15m"], help="Bias timeframe: 5m or 15m (Marni Fib, INDEX UT-on-HA TV parity)")
ap.add_argument("--arm_window", type=int, default=5, help="Flag/Super arm window length in bars (USER PARAM: ARM_WINDOW)")
ap.add_argument("--use_elder", type=lambda x: str(x).lower() in ("1","true","yes","on"), default=True,
                help="Enable INDEX Elder Impulse bias filter (True/False)")
ap.add_argument("--use_rsi", type=lambda x: str(x).lower() in ("1","true","yes","on"), default=False,
                help="Enable option-chart RSI(14) entry gate: CE entry requires ce_rsi>60, PE entry requires pe_rsi<40")
ap.add_argument("--profile", action="store_true", help="Print per-phase timing + GPU util for bottleneck analysis")
args = ap.parse_args()
SL_PTS = args.sl
TP_PTS = args.tp
ARM_WINDOW = args.arm_window
USE_ELDER = args.use_elder
USE_RSI = args.use_rsi
RSI_CE_HI = 60.0   # CE entry requires ce_rsi > 60
RSI_PE_LO = 40.0   # PE entry requires pe_rsi < 40

_PROF = {"t": time.time(), "marks": []}
def _mark(name):
    if not getattr(args, "profile", False): return
    now = time.time(); _PROF["marks"].append((name, now - _PROF["t"])); _PROF["t"] = now
    print(f"  [PROFILE] {name}: {_PROF['marks'][-1][1]*1000:.1f} ms")

if args.smoke:
    # Causal parity smoke: 5 days INCLUDING 26 Aug (last 5 trading days)
    if "2026-08-26" in trading_days:
        # last 5 days that include 26 Aug for parity with prototype
        smoke_days = [d for d in trading_days if d >= "2026-08-20"][-5:]
        # ensure 26 Aug is included; fallback to last 5
        if "2026-08-26" not in smoke_days:
            smoke_days = trading_days[-5:]
        trading_days = smoke_days
    else:
        trading_days = trading_days[:5]
    print(f"SMOKE TEST MODE (5-day, incl. 26 Aug for parity, 8 workers max): {trading_days}")

D = len(trading_days)
print(f"Total Trading Days: {D}")

# --- Elder Impulse (index, full-history warmup) directional filter ---
# NOTE: futures 1m history is only available for Aug 2026; the index spans
# 2015-2026, so it is the available long-history underlying for the 7y filter.
USE_ELDER = args.use_elder
def ema_py(arr, p):
    a = np.asarray(arr, dtype=float); e = np.zeros_like(a); k = 2.0 / (p + 1); e[0] = a[0]
    for i in range(1, len(a)): e[i] = a[i] * k + e[i - 1] * (1 - k)
    return e
def _resample_idx(df, freq):
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df.columns: agg['volume'] = 'sum'
    return df.set_index('dt').resample(freq).agg(agg).dropna().reset_index()
_IDX_FULL = None
def build_elder_idx():
    global _IDX_FULL
    if _IDX_FULL is None:
        d = pd.read_csv(IDX_PATH); d['dt'] = pd.to_datetime(d['date'])
        _IDX_FULL = d.sort_values('dt').reset_index(drop=True)
    i3 = _resample_idx(_IDX_FULL, '3min'); c3 = i3['close'].values.astype(float)
    e12 = ema_py(c3, 12); e26 = ema_py(c3, 26); e13 = ema_py(c3, 13)
    macd = e12 - e26; sig = ema_py(macd, 9); hist = macd - sig
    buckets = i3['dt'].dt.strftime('%Y-%m-%d %H:%M:%S').values
    b2i = {b: i for i, b in enumerate(buckets)}
    iclose = {dt.strftime('%Y-%m-%d %H:%M:%S'): float(c) for dt, c in zip(_IDX_FULL['dt'], _IDX_FULL['close'].values)}
    return dict(c3=c3, e12=e12, e26=e26, e13=e13, sig=sig, hist=hist, bucket_to_bi=b2i, idx_close=iclose)
def elder_color_at_idx(E, ts_str):
    t = pd.Timestamp(ts_str); bucket = t.floor('3min').strftime('%Y-%m-%d %H:%M:%S')
    bi = E['bucket_to_bi'].get(bucket)
    if bi is None or bi == 0: return 'blue'
    cl = E['idx_close'].get(ts_str)
    if cl is None: cl = E['c3'][bi]
    k12 = 2/13; k26 = 2/27; k13 = 2/14; k9 = 2/10
    e12i = cl*k12 + E['e12'][bi-1]*(1-k12); e26i = cl*k26 + E['e26'][bi-1]*(1-k26)
    e13i = cl*k13 + E['e13'][bi-1]*(1-k13)
    macdi = e12i - e26i; sigi = macdi*k9 + E['sig'][bi-1]*(1-k9); histi = macdi - sigi
    if e13i > E['e13'][bi-1] and histi > E['hist'][bi-1]: return 'green'
    if e13i < E['e13'][bi-1] and histi < E['hist'][bi-1]: return 'red'
    return 'blue'

def build_rsi_idx():
    i3 = _resample_idx(_IDX_FULL, '3min'); c3 = i3['close'].values.astype(float)
    n = len(c3); p = 14
    delta = np.zeros(n); delta[1:] = c3[1:] - c3[:-1]
    gain = np.where(delta > 0, delta, 0.0); loss = np.where(delta < 0, -delta, 0.0)
    ag = np.zeros(n); al = np.zeros(n)
    if n > p:
        ag[p] = gain[1:p+1].mean(); al[p] = loss[1:p+1].mean()
        for i in range(p+1, n):
            ag[i] = (ag[i-1]*(p-1) + gain[i])/p; al[i] = (al[i-1]*(p-1) + loss[i])/p
    buckets = i3['dt'].dt.strftime('%Y-%m-%d %H:%M:%S').values
    b2i = {b: i for i, b in enumerate(buckets)}
    iclose = {dt.strftime('%Y-%m-%d %H:%M:%S'): float(c) for dt, c in zip(_IDX_FULL['dt'], _IDX_FULL['close'].values)}
    return dict(c3=c3, ag=ag, al=al, period=p, bucket_to_bi=b2i, idx_close=iclose)
def rsi_at_idx(E, ts_str):
    if E is None: return 50.0
    t = pd.Timestamp(ts_str); bucket = t.floor('3min').strftime('%Y-%m-%d %H:%M:%S')
    bi = E['bucket_to_bi'].get(bucket)
    if bi is None or bi < E['period']: return 50.0
    cl = E['idx_close'].get(ts_str)
    if cl is None: cl = E['c3'][bi]
    d = cl - E['c3'][bi-1]; g = d if d > 0 else 0.0; l = -d if d < 0 else 0.0
    p = E['period']; ag = (E['ag'][bi-1]*(p-1) + g)/p; al = (E['al'][bi-1]*(p-1) + l)/p
    if al == 0: return 100.0
    return 100 - 100/(1 + ag/al)

elder_E = build_elder_idx() if USE_ELDER else None
elder_lookup = {}
if USE_ELDER:
    for d in trading_days:
        for m in range(SESSION_START, SESSION_END):
            hh = m // 60; mm = m % 60
            ts = f"{d} {hh:02d}:{mm:02d}:00"
            elder_lookup[ts] = elder_color_at_idx(elder_E, ts)
    print(f"Elder lookup built: {len(elder_lookup)} min-bars (USE_ELDER={USE_ELDER})")

# RSI(14) on the 3m underlying (futures) chart, per-minute lookup (matches Elder's chart).
# User rule: CE entry requires rsi > 60, PE entry requires rsi < 40.
if _IDX_FULL is None:
    _IDX_FULL = pd.read_csv(IDX_PATH); _IDX_FULL['dt'] = pd.to_datetime(_IDX_FULL['date'])
    _IDX_FULL = _IDX_FULL.sort_values('dt').reset_index(drop=True)
rsi_E = build_rsi_idx()
rsi_lookup = {}
for d in trading_days:
    for m in range(SESSION_START, SESSION_END):
        hh = m // 60; mm = m % 60
        ts = f"{d} {hh:02d}:{mm:02d}:00"
        rsi_lookup[ts] = rsi_at_idx(rsi_E, ts)
print(f"RSI(14) 3m-underlying lookup built: {len(rsi_lookup)} min-bars (USE_RSI={USE_RSI})")


bias_tf_map = {"15m": "15min", "5m": "5min"}
bias_tf_full = bias_tf_map.get(args.bias_tf, "15min")
# Full-history INDEX bias (TV parity). Vectorized build is ~2s, so smoke == full run (causal parity).
bias_lookup, bias_bars = build_bias_lookup_tf(_IDX_FULL, timeframe=bias_tf_full, ut_on_ha=True) if USE_BIAS else (None, None)
bias_ut_lookup = {}
bias_lr_lookup = {}
if USE_BIAS:
    print(f"{args.bias_tf} Marni Fib bias lookup built: {len(bias_lookup)} min-bars (INDEX UT-on-HA TV parity, USE_BIAS={USE_BIAS})")
    # Decompose the combined UT-on-HA bias into its two raw components (15m -> 1m carried):
    #  - UT colour only : green -> +1 (CE/bull), red -> -1 (PE/bear)
    #  - LinReg only    : HA close above the double-smoothed LinReg signal -> +1, below -> -1
    bars_df2 = pd.DataFrame([{'t_eff': pd.Timestamp(ts),
        'ut_int': 1 if d['ut'] == 'green' else (-1 if d['ut'] == 'red' else 0),
        'ha_close': d['ha_close'], 'linreg': d['linreg']} for ts, d in bias_bars])
    df1 = _IDX_FULL[['dt']].copy().sort_values('dt').reset_index(drop=True)
    m2 = pd.merge_asof(df1, bars_df2.sort_values('t_eff'), left_on='dt', right_on='t_eff', direction='backward')
    ts2 = m2['dt'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    bias_ut_lookup = {t: int(u) for t, u in zip(ts2, m2['ut_int'].fillna(0).astype(int))}
    for t, hc, lin in zip(ts2, m2['ha_close'].tolist(), m2['linreg'].tolist()):
        if hc is None or lin is None or (isinstance(lin, float) and np.isnan(lin)):
            bias_lr_lookup[t] = 0
        elif hc > lin:
            bias_lr_lookup[t] = 1
        elif hc < lin:
            bias_lr_lookup[t] = -1
        else:
            M.bias_lr_lookup[t] = 0

# 2. Load Option Data
t0 = time.time()
print(f"Loading 7-Year Parquet Data (D={D}, T1={T1})...")
lazy = pl.scan_parquet(PARQUET).filter(
    (pl.col("day") >= trading_days[0]) & (pl.col("day") <= trading_days[-1]) &
    (pl.col("side").is_in(["PE", "CE"]))
)
df_all = lazy.select(["day", "minute", "symbol", "strike", "side", "close", "high", "low", "open"]).collect()

# 2b. Select FRONT WEEKLY expiry per day (avoid monthly/weekly contamination).
#     Strategy trades the current weekly expiry -> nearest expiry date >= trading day.
import re as _re, datetime as _dt
_MONTHS = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
def _tok_date(tok):
    return _dt.date(2000 + int(tok[5:7]), _MONTHS[tok[2:5]], int(tok[0:2]))
df_all = df_all.with_columns(pl.col("symbol").str.extract(r"NIFTY(\d{2}[A-Z]{3}\d{2})", 1).alias("exp_token"))
_toks = [t for t in df_all.select("exp_token").unique().to_series().to_list() if t]
_tok_dates = {t: _tok_date(t) for t in _toks}
_day_exp = {}
for d in trading_days:
    dd = pd.Timestamp(d).date()
    cands = [t for t, dt in _tok_dates.items() if dt >= dd]
    _day_exp[d] = min(cands, key=lambda t: _tok_dates[t]) if cands else max(_tok_dates, key=lambda t: _tok_dates[t])
df_all = df_all.with_columns(pl.col("day").replace(_day_exp).alias("te"))
df_all = df_all.filter(pl.col("exp_token") == pl.col("te"))
print(f"Front-weekly expiry per day (e.g. 26 Aug -> {_day_exp.get('2026-08-26')}); "
      f"all selected: {sorted(set(_day_exp.values()))}")

day_to_idx = {d: i for i, d in enumerate(trading_days)}
atm_by_day = {d: int(round(spot_by_day.get(d, {}).get(555, 25000) / 50) * 50) for d in trading_days}
# 2nd ITM: CE = ATM-100 (2 strikes ITM), PE = ATM+100
target_map = {d: {"PE": atm_by_day[d] + 100, "CE": atm_by_day[d] - 100} for d in trading_days}
target_df = pl.DataFrame({
    "day": trading_days,
    "pe_target": [target_map[d]["PE"] for d in trading_days],
    "ce_target": [target_map[d]["CE"] for d in trading_days],
})

df_pe = df_all.filter(pl.col("side") == "PE").join(target_df.select("day", "pe_target").rename({"pe_target": "target"}), on="day", how="inner").filter(pl.col("strike") == pl.col("target"))
df_ce = df_all.filter(pl.col("side") == "CE").join(target_df.select("day", "ce_target").rename({"ce_target": "target"}), on="day", how="inner").filter(pl.col("strike") == pl.col("target"))

# Vectorized 1m Tensor Builder
def build_tensors_1m(df_side, D, T, day_to_idx):
    c = np.full((D, T), np.nan, dtype=np.float32); h = np.full((D, T), np.nan, dtype=np.float32)
    l = np.full((D, T), np.nan, dtype=np.float32); o = np.full((D, T), np.nan, dtype=np.float32)
    day_idx = df_side["day"].replace(day_to_idx).cast(pl.Int32).to_numpy()
    t_idx = (df_side["minute"] - SESSION_START).to_numpy().astype(np.int32)
    valid = (t_idx >= 0) & (t_idx < T)
    day_idx = day_idx[valid]; t_idx = t_idx[valid]
    c[day_idx, t_idx] = df_side["close"].to_numpy()[valid].astype(np.float32)
    h[day_idx, t_idx] = df_side["high"].to_numpy()[valid].astype(np.float32)
    l[day_idx, t_idx] = df_side["low"].to_numpy()[valid].astype(np.float32)
    o[day_idx, t_idx] = df_side["open"].to_numpy()[valid].astype(np.float32)
    c = pd.DataFrame(c).ffill(axis=1).bfill(axis=1).values
    h = pd.DataFrame(h).ffill(axis=1).bfill(axis=1).values
    l = pd.DataFrame(l).ffill(axis=1).bfill(axis=1).values
    o = pd.DataFrame(o).ffill(axis=1).bfill(axis=1).values
    return c, h, l, o

pe_c, pe_h, pe_l, pe_o = build_tensors_1m(df_pe, D, T1, day_to_idx)
ce_c, ce_h, ce_l, ce_o = build_tensors_1m(df_ce, D, T1, day_to_idx)
_mark("data_load+tensors")

# 3. GPU Causal Indicators
@torch.no_grad()
def gpu_stoch(high, low, close, k, d):
    ht = torch.tensor(high, device=DEVICE, dtype=torch.float32)
    lt = torch.tensor(low, device=DEVICE, dtype=torch.float32)
    ct = torch.tensor(close, device=DEVICE, dtype=torch.float32)
    h_pad = torch.nn.functional.pad(ht.unsqueeze(1), (k - 1, 0), mode="replicate")
    l_pad = torch.nn.functional.pad(lt.unsqueeze(1), (k - 1, 0), mode="replicate")
    max_h = torch.nn.functional.max_pool1d(h_pad, k, stride=1).squeeze(1)
    min_l = -torch.nn.functional.max_pool1d(-l_pad, k, stride=1).squeeze(1)
    denom = (max_h - min_l).clamp(min=1e-6)
    fast_k = (ct - min_l) / denom * 100.0
    k_pad = torch.nn.functional.pad(fast_k.unsqueeze(1), (d - 1, 0), mode="replicate")
    slow_d = torch.nn.functional.avg_pool1d(k_pad, d, stride=1).squeeze(1)
    return slow_d.cpu().numpy()

@torch.no_grad()
def gpu_ema(close, period):
    ct = torch.tensor(close, device=DEVICE, dtype=torch.float32)
    alpha = 2 / (period + 1)
    ema = torch.zeros_like(ct); ema[:, 0] = ct[:, 0]
    for t in range(1, ct.shape[1]): ema[:, t] = ema[:, t - 1] * (1 - alpha) + ct[:, t] * alpha
    return ema.cpu().numpy()

@torch.no_grad()
def gpu_vwap(high, low, close):
    ht = torch.tensor(high, device=DEVICE, dtype=torch.float32)
    lt = torch.tensor(low, device=DEVICE, dtype=torch.float32)
    ct = torch.tensor(close, device=DEVICE, dtype=torch.float32)
    vol = torch.full_like(ct, 100.0); hlc3 = (ht + lt + ct) / 3.0
    cum_pv = torch.cumsum(hlc3 * vol.clamp(min=10), dim=1)
    cum_v = torch.cumsum(vol.clamp(min=10), dim=1)
    return (cum_pv / cum_v.clamp(min=1)).cpu().numpy()

_mark("->gpu_indicators")
print("Calculating GPU indicators (S1, S3, S4, EMA20, EMA200, VWAP)...")
pe_s1 = gpu_stoch(pe_h, pe_l, pe_c, S1_K, S1_D)
pe_s3 = gpu_stoch(pe_h, pe_l, pe_c, S3_K, S3_D)
pe_s4 = gpu_stoch(pe_h, pe_l, pe_c, S4_K, S4_D)

ce_s1 = gpu_stoch(ce_h, ce_l, ce_c, S1_K, S1_D)
ce_s3 = gpu_stoch(ce_h, ce_l, ce_c, S3_K, S3_D)
ce_s4 = gpu_stoch(ce_h, ce_l, ce_c, S4_K, S4_D)

pe_ema20 = gpu_ema(pe_c, 20); pe_ema200 = gpu_ema(pe_c, 200); pe_vwap = gpu_vwap(pe_h, pe_l, pe_c)
ce_ema20 = gpu_ema(ce_c, 20); ce_ema200 = gpu_ema(ce_c, 200); ce_vwap = gpu_vwap(ce_h, ce_l, ce_c)

# ATR (True Range, EMA-smoothed) on option LTP — for volatility-scaled SL (capped at TP).
def compute_atr(high, low, close, period=14):
    prev = np.empty_like(close); prev[:, 0] = close[:, 0]
    prev[:, 1:] = close[:, :-1]
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    atr = np.empty_like(tr, dtype=np.float32)
    alpha = 2.0 / (period + 1)
    atr[:, 0] = tr[:, 0]
    for t in range(1, tr.shape[1]):
        atr[:, t] = atr[:, t - 1] * (1 - alpha) + tr[:, t] * alpha
    return atr

ce_atr = compute_atr(ce_h, ce_l, ce_c, args.atr_period if hasattr(args, 'atr_period') else 14)
pe_atr = compute_atr(pe_h, pe_l, pe_c, args.atr_period if hasattr(args, 'atr_period') else 14)
_mark("gpu_indicators+atr")

# 3b. Combined multi-TF stochastic (2m/3m/5m) expanded to 1m resolution.
#     Faithful port of run_26aug_multitf COMBINED: a TF bar's stoch is applied to
#     every 1m bar inside that TF candle; a trade fires if ANY TF's signal triggers.
TF_LIST = [1, 2, 3, 5]
def make_tf_stoch(c1, h1, l1):
    cache = []
    for tf in TF_LIST:
        D, T1 = c1.shape; T_tf = T1 // tf; n = T_tf * tf
        c_ = c1[:, :n].reshape(D, T_tf, tf); h_ = h1[:, :n].reshape(D, T_tf, tf); l_ = l1[:, :n].reshape(D, T_tf, tf)
        cl = c_[:, :, -1]; hi = h_.max(2); lo = l_.min(2)
        s1 = gpu_stoch(hi, lo, cl, S1_K, S1_D)
        s3 = gpu_stoch(hi, lo, cl, S3_K, S3_D)
        s4 = gpu_stoch(hi, lo, cl, S4_K, S4_D)
        rising = np.concatenate([np.zeros((D, 1), dtype=np.float32),
                                 (s1[:, 1:] > s1[:, :-1]).astype(np.float32)], 1)

        def exp(a):
            e = np.repeat(a, tf, axis=1)
            if e.shape[1] < T1:
                e = np.pad(e, ((0, 0), (0, T1 - e.shape[1])), mode='edge')
            return e
        cache.append(dict(s1=exp(s1), s3=exp(s3), s4=exp(s4), rising=exp(rising),
                          lo=exp(lo), cl=exp(cl), tf=tf))
    return cache

ce_tf = make_tf_stoch(ce_c, ce_h, ce_l)
pe_tf = make_tf_stoch(pe_c, pe_h, pe_l)

ce_super_full = np.zeros((D, T1), dtype=bool)
ce_m6_full = np.zeros((D, T1), dtype=bool)
pe_super_full = np.zeros((D, T1), dtype=bool)
pe_m6_full = np.zeros((D, T1), dtype=bool)
for c in ce_tf:
    ce_super_full |= (c['s3'] < 25.0) & (c['s4'] < 25.0) & (c['s1'] < 25.0) & (c['rising'] > 0.5)
    ce_m6_full |= (c['s4'] >= 79.5) & (c['s1'] < 79.5)
for c in pe_tf:
    pe_super_full |= (c['s3'] < 25.0) & (c['s4'] < 25.0) & (c['s1'] < 25.0) & (c['rising'] > 0.5)
    pe_m6_full |= (c['s4'] >= 79.5) & (c['s1'] < 79.5)

# Reversal entries (optional): Cross-Side Reversal per design note.
# S4 <= 20.5 on Chart X for >= 14 consecutive bars + Super on Chart X ->
# instant buy Chart Y (opposite) ONLY IF Chart Y has SR bounce on that exact bar.
def _sustain_runs(mask):
    out = np.zeros(mask.shape, dtype=np.int32)
    for i in range(mask.shape[0]):
        run = 0; row = mask[i]
        for t in range(mask.shape[1]):
            run = run + 1 if row[t] else 0
            out[i, t] = run
    return out

ce_rev_on = (_sustain_runs(ce_s4 <= 20.5) >= 14) & ce_super_full
pe_rev_on = (_sustain_runs(pe_s4 <= 20.5) >= 14) & pe_super_full
ce_rev_off = np.zeros((D, T1), dtype=bool)
pe_rev_off = np.zeros((D, T1), dtype=bool)
ce_rev_full = ce_rev_on if args.reversal else ce_rev_off
pe_rev_full = pe_rev_on if args.reversal else pe_rev_off
print(f"Reversal entries: CE-rev bars={int(ce_rev_on.sum())}, PE-rev bars={int(pe_rev_on.sum())}")
_mark("tf_masks+reversal")

# 4. Option Chart SR Levels per Day
def compute_option_sr(df_side, trading_days):
    daily = {}
    for day in trading_days:
        sub = df_side.filter(df_side["day"] == day)
        if len(sub) > 0:
            daily[day] = {"h": float(sub["high"].max()), "l": float(sub["low"].min()), "c": float(sub["close"][-1])}
    sorted_days = sorted(daily.keys())
    sr_by_day = {}
    for i in range(1, len(sorted_days)):
        day = sorted_days[i]; prev = sorted_days[i - 1]
        ph, pl_, pc = daily[prev]["h"], daily[prev]["l"], daily[prev]["c"]
        pivot = (ph + pl_ + pc) / 3.0; bc = (ph + pl_) / 2.0; tc = 2.0 * pivot - bc
        rng = ph - pl_
        sr_by_day[day] = [
            ("CPR_BC", bc), ("CPR_Pivot", pivot), ("CPR_TC", tc),
            ("Cam_H3", pc + rng * 1.1 / 4.0), ("Cam_L3", pc - rng * 1.1 / 4.0),
            ("PDH", ph), ("PDL", pl_)
        ]
    return sr_by_day

pe_option_sr = compute_option_sr(df_pe, trading_days)
ce_option_sr = compute_option_sr(df_ce, trading_days)
_mark("sr_levels")

# 5. Core Signal Processing Engine
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_days_chunk(idxs, cap_val=0):
    Cd = len(idxs)
    
    in_pos = np.zeros(Cd, dtype=bool)
    pos_side = np.full(Cd, '  ', dtype='<U2')
    entry_price = np.full(Cd, np.nan, dtype=np.float32)
    sl_price = np.full(Cd, np.nan, dtype=np.float32)
    tp_price = np.full(Cd, np.nan, dtype=np.float32)
    
    ce_s4_embed = np.zeros(Cd, dtype=np.int32)
    pe_s4_embed = np.zeros(Cd, dtype=np.int32)
    
    ce_flag_armed = np.zeros(Cd, dtype=bool); ce_flag_arm_t = np.full(Cd, -999, dtype=np.int32)
    pe_flag_armed = np.zeros(Cd, dtype=bool); pe_flag_arm_t = np.full(Cd, -999, dtype=np.int32)
    
    ce_super_armed = np.zeros(Cd, dtype=bool); ce_super_arm_t = np.full(Cd, -999, dtype=np.int32)
    pe_super_armed = np.zeros(Cd, dtype=bool); pe_super_arm_t = np.full(Cd, -999, dtype=np.int32)
    
    daily_pts = np.zeros(Cd, dtype=np.float32)
    trades = []
    
    for t in range(1, T1):
        cap_hit = (cap_val != 0) & (np.abs(daily_pts) >= cap_val)
        
        # Update S4 oversold embed counters
        ce_s4_embed = np.where(ce_s4[idxs, t] <= 20.5, ce_s4_embed + 1, 0)
        pe_s4_embed = np.where(pe_s4[idxs, t] <= 20.5, pe_s4_embed + 1, 0)
        
        # Position Management
        active = np.where(in_pos)[0]
        if len(active) > 0:
            for ci in active:
                is_pe = (pos_side[ci] == 'PE')
                curr_h = pe_h[idxs[ci], t] if is_pe else ce_h[idxs[ci], t]
                curr_l = pe_l[idxs[ci], t] if is_pe else ce_l[idxs[ci], t]
                
                if curr_l <= sl_price[ci]:
                    pts = sl_price[ci] - entry_price[ci]
                    daily_pts[ci] += pts
                    trades.append((trading_days[idxs[ci]], pos_side[ci], 'SL', float(entry_price[ci]), float(sl_price[ci]), pts * LOT - FEE))
                    in_pos[ci] = False
                elif curr_h >= tp_price[ci]:
                    pts = tp_price[ci] - entry_price[ci]
                    daily_pts[ci] += pts
                    trades.append((trading_days[idxs[ci]], pos_side[ci], 'TP', float(entry_price[ci]), float(tp_price[ci]), pts * LOT - FEE))
                    in_pos[ci] = False

        # Arming (S1 <= 25.0)
        pe_arm_mask = (pe_s1[idxs, t] <= ARM_S1) & (~in_pos) & (~cap_hit)
        ce_arm_mask = (ce_s1[idxs, t] <= ARM_S1) & (~in_pos) & (~cap_hit)
        
        pe_flag_armed[pe_arm_mask] = True; pe_flag_arm_t[pe_arm_mask] = t
        ce_flag_armed[ce_arm_mask] = True; ce_flag_arm_t[ce_arm_mask] = t
        pe_super_armed[pe_arm_mask] = True; pe_super_arm_t[pe_arm_mask] = t
        ce_super_armed[ce_arm_mask] = True; ce_super_arm_t[ce_arm_mask] = t

        if M_TRACE_DAY is not None and 160 <= t <= 175 and trading_days[idxs[0]] == M_TRACE_DAY:
            print(f"[ARM] t={t} ce_s1={ce_s1[idxs[0],t]:.2f} in_pos={in_pos[0]} ce_flag_armed={ce_flag_armed[0]} ce_arm_t={ce_flag_arm_t[0]} ce_m6_full={ce_m6_full[idxs[0],t]}", file=sys.stderr)

        # Arm expiration after 5 bars (dropped if price cannot reach/bounce above SR within 5 bars)
        pe_flag_armed[pe_flag_armed & (t - pe_flag_arm_t > ARM_WINDOW)] = False
        ce_flag_armed[ce_flag_armed & (t - ce_flag_arm_t > ARM_WINDOW)] = False
        pe_super_armed[pe_super_armed & (t - pe_super_arm_t > ARM_WINDOW)] = False
        ce_super_armed[ce_super_armed & (t - ce_super_arm_t > ARM_WINDOW)] = False

        if M_TRACE_DAY is not None:
            for ci in range(Cd):
                if trading_days[idxs[ci]] == M_TRACE_DAY:
                    M_POS_TRACE.append((t, int(in_pos[ci]), pos_side[ci]))

        # Trigger Evaluation:
        # A. Flag (M6) — OR across combined 2m/3m/5m TF stochastics
        pe_m6 = pe_flag_armed & (t - pe_flag_arm_t <= ARM_WINDOW) & pe_m6_full[idxs, t] & (~cap_hit)
        ce_m6 = ce_flag_armed & (t - ce_flag_arm_t <= ARM_WINDOW) & ce_m6_full[idxs, t] & (~cap_hit)

        # B. Super Signal (Same side) — OR across combined 2m/3m/5m TF stochastics
        pe_super = pe_super_armed & (t - pe_super_arm_t <= ARM_WINDOW) & pe_super_full[idxs, t] & (~cap_hit)
        ce_super = ce_super_armed & (t - ce_super_arm_t <= ARM_WINDOW) & ce_super_full[idxs, t] & (~cap_hit)

        # C. Reversal (enabled via --reversal): cross-side S4 oversold + Super on opposite chart
        rev_buy_pe = ce_rev_full[idxs, t]
        rev_buy_ce = pe_rev_full[idxs, t]

        # Check SR Bounce on Target Chart
        for ci in range(Cd):
            if in_pos[ci] or cap_hit[ci]:
                continue
            
            d_str = trading_days[idxs[ci]]
            hh = (SESSION_START + t) // 60; mm = (SESSION_START + t) % 60
            ts = f"{d_str} {hh:02d}:{mm:02d}:00"
            ec = elder_lookup.get(ts, 'blue') if USE_ELDER else 'blue'
            bull, bear = bias_lookup.get(ts, (False, False)) if USE_BIAS else (True, True)

            # Check PE (blocked if Elder green OR 15m bias not bearish OR RSI(3m) not < 40)
            if M_TRACE_DAY is not None and trading_days[idxs[ci]] == M_TRACE_DAY and 166 <= t <= 172:
                sys.stderr.write(f"[PEBLK-IN] t={t} in_pos={in_pos[ci]} pe_m6={bool(pe_m6[ci])} pe_super={bool(pe_super[ci])} rev={bool(rev_buy_pe[ci])}\n")
            if pe_m6[ci] or pe_super[ci] or rev_buy_pe[ci]:
                if M_TRACE_DAY is not None and trading_days[idxs[ci]] == M_TRACE_DAY and 166 <= t <= 172:
                    sys.stderr.write(f"[PEBLK] t={t} ec={ec} bear={bear} rev={bool(rev_buy_pe[ci])} m6={bool(pe_m6[ci])} sup={bool(pe_super[ci])}\n")
                if USE_ELDER and ec == 'green':
                    continue
                if USE_BIAS and not bear:
                    continue
                if USE_RSI and not (rsi_lookup.get(ts, 50.0) < RSI_PE_LO):
                    continue
                buf_pe = 1.0
                all_sr_pe = [lvl for _, lvl in pe_option_sr.get(d_str, [])]
                if not np.isnan(pe_ema20[idxs[ci], t]): all_sr_pe.append(pe_ema20[idxs[ci], t])
                if not np.isnan(pe_ema200[idxs[ci], t]): all_sr_pe.append(pe_ema200[idxs[ci], t])
                if not np.isnan(pe_vwap[idxs[ci], t]): all_sr_pe.append(pe_vwap[idxs[ci], t])
                # SR bounce on ANY combined-TF low/close (faithful port of COMBINED)
                pe_b = any(pe_l[idxs[ci], t] <= lvl + buf_pe and pe_c[idxs[ci], t] >= lvl - 0.5 for lvl in all_sr_pe)
                if not pe_b:
                    for c in pe_tf:
                        lo = c['lo'][idxs[ci], t]; cl = c['cl'][idxs[ci], t]
                        if any(lo <= lvl + buf_pe and cl >= lvl - 0.5 for lvl in all_sr_pe):
                            pe_b = True; break
                if pe_b:
                    in_pos[ci] = True
                    pos_side[ci] = 'PE'
                    if M_ENT_TRACE is not None:
                        M_ENT_TRACE.append((t, 'PE'))
                    entry_price[ci] = pe_c[idxs[ci], t]
                    _a = pe_atr[idxs[ci], t]
                    sl_pts = (_a * args.atr_mult) if (args.atr_sl and not np.isnan(_a)) else SL_PTS
                    tp_pts = (_a * args.atr_mult) if (args.atr_sl and not np.isnan(_a)) else TP_PTS
                    sl_pts = min(sl_pts, TP_PTS); tp_pts = min(tp_pts, TP_PTS)
                    sl_price[ci] = entry_price[ci] - sl_pts
                    tp_price[ci] = entry_price[ci] + tp_pts
                    pe_flag_armed[ci] = False; pe_super_armed[ci] = False
                    continue

            # Check CE (blocked if Elder red OR 15m bias not bullish)
            if M_TRACE_DAY is not None and trading_days[idxs[ci]] == M_TRACE_DAY and 165 <= t <= 175:
                sys.stderr.write(f"[TRIG] t={t} in_pos={in_pos[ci]} ce_m6={bool(ce_m6[ci])} ce_super={bool(ce_super[ci])} rev={bool(rev_buy_ce[ci])} flag={bool(ce_flag_armed[ci])} armt={ce_flag_arm_t[ci]}\n")
            if M_CE_CAND_TRACE is not None and trading_days[idxs[ci]] == M_TRACE_DAY:
                M_CE_CAND_TRACE.append((t, bool(ce_flag_armed[ci]), bool(ce_m6[ci]),
                    bool(ce_super[ci]), bool(rev_buy_ce[ci]),
                    bool(USE_ELDER and ec == 'red'), bool(USE_BIAS and not bull)))
            if ce_m6[ci] or ce_super[ci] or rev_buy_ce[ci]:
                if M_TRACE_DAY is not None and trading_days[idxs[ci]] == M_TRACE_DAY and 160 <= t <= 175:
                    sys.stderr.write(f"[CEBLK] t={t} ec={ec} bull={bull} rev={bool(rev_buy_ce[ci])} m6={bool(ce_m6[ci])} sup={bool(ce_super[ci])}\n")
                if USE_ELDER and ec == 'red':
                    if M_TRACE_DAY is not None and trading_days[idxs[ci]] == M_TRACE_DAY and 160 <= t <= 175:
                        sys.stderr.write(f"[CEBLK] t={t} BLOCK elder-red\n")
                    continue
                if USE_BIAS and not bull:
                    if M_TRACE_DAY is not None and trading_days[idxs[ci]] == M_TRACE_DAY and 160 <= t <= 175:
                        sys.stderr.write(f"[CEBLK] t={t} BLOCK bias-not-bull\n")
                    continue
                if USE_RSI and not (rsi_lookup.get(ts, 50.0) > RSI_CE_HI):
                    continue
                buf_ce = 1.0
                all_sr_ce = [lvl for _, lvl in ce_option_sr.get(d_str, [])]
                if not np.isnan(ce_ema20[idxs[ci], t]): all_sr_ce.append(ce_ema20[idxs[ci], t])
                if not np.isnan(ce_ema200[idxs[ci], t]): all_sr_ce.append(ce_ema200[idxs[ci], t])
                if not np.isnan(ce_vwap[idxs[ci], t]): all_sr_ce.append(ce_vwap[idxs[ci], t])
                ce_b = any(ce_l[idxs[ci], t] <= lvl + buf_ce and ce_c[idxs[ci], t] >= lvl - 0.5 for lvl in all_sr_ce)
                if not ce_b:
                    for c in ce_tf:
                        lo = c['lo'][idxs[ci], t]; cl = c['cl'][idxs[ci], t]
                        if any(lo <= lvl + buf_ce and cl >= lvl - 0.5 for lvl in all_sr_ce):
                            ce_b = True; break
                if M_TRACE_DAY is not None and trading_days[idxs[ci]] == M_TRACE_DAY and 160 <= t <= 175:
                    sys.stderr.write(f"[CEBLK] t={t} ce_b={ce_b} ce_l={ce_l[idxs[ci],t]:.2f} ce_c={ce_c[idxs[ci],t]:.2f} n_sr={len(all_sr_ce)}\n")
                if ce_b:
                    in_pos[ci] = True
                    pos_side[ci] = 'CE'
                    if M_ENT_TRACE is not None:
                        M_ENT_TRACE.append((t, 'CE'))
                    entry_price[ci] = ce_c[idxs[ci], t]
                    _a = ce_atr[idxs[ci], t]
                    sl_pts = (_a * args.atr_mult) if (args.atr_sl and not np.isnan(_a)) else SL_PTS
                    tp_pts = (_a * args.atr_mult) if (args.atr_sl and not np.isnan(_a)) else TP_PTS
                    sl_pts = min(sl_pts, TP_PTS); tp_pts = min(tp_pts, TP_PTS)
                    sl_price[ci] = entry_price[ci] - sl_pts
                    tp_price[ci] = entry_price[ci] + tp_pts
                    ce_flag_armed[ci] = False; ce_super_armed[ci] = False
                    if globals().get('M_CE_TRACE'):
                        M_CE_BARS.append((d_str, t))

    return trades

M_CE_BARS = []
M_CE_TRACE = False
M_POS_TRACE = []
M_TRACE_DAY = None
M_ENT_TRACE = None
M_CE_CAND_TRACE = None

def run_full_backtest(cap_val=0, workers=8):
    chunks = np.array_split(np.arange(D), workers)
    all_trades = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_days_chunk, ch, cap_val) for ch in chunks]
        for fu in as_completed(futs):
            all_trades.extend(fu.result())
    all_trades.sort(key=lambda x: x[0])
    return all_trades


def _metrics_from_trades(trades):
    n = len(trades)
    if n == 0:
        return dict(trades=0, wr=0.0, net_rs=0.0, net_pts=0.0,
                    avg_sl=0.0, avg_tp=0.0, avg_trades_day=0.0, max_dd=0.0)
    sl = [t for t in trades if t[2] == 'SL']
    tp = [t for t in trades if t[2] == 'TP']
    wins = sum(1 for t in trades if t[5] > 0)
    wr = wins / n if n else 0.0
    net_rs = sum(t[5] for t in trades)
    net_pts = sum((t[4] - t[3]) for t in trades)
    avg_sl = (sum(abs(t[4] - t[3]) for t in sl) / len(sl)) if sl else 0.0
    avg_tp = (sum((t[4] - t[3]) for t in tp) / len(tp)) if tp else 0.0
    avg_trades_day = n / float(D)
    daily = {}
    for t in trades:
        daily[t[0]] = daily.get(t[0], 0.0) + t[5]
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for d in sorted(daily.keys()):
        cum += daily[d]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return dict(trades=n, wr=wr, net_rs=net_rs, net_pts=net_pts,
                avg_sl=avg_sl, avg_tp=avg_tp, avg_trades_day=avg_trades_day, max_dd=max_dd)


def run_backtest_params(p):
    """Run the backtest for a parameter dict, reusing already-built GPU state.

    p keys: arm_window, use_elder, sl, tp, atr_sl, atr_mult, atr_period,
            bias_tf, cap, reversal, workers
    """
    global ARM_WINDOW, USE_ELDER, USE_RSI, SL_PTS, TP_PTS, ce_atr, pe_atr, pe_rev_full, ce_rev_full, args
    ARM_WINDOW = int(p.get('arm_window', ARM_WINDOW))
    USE_ELDER = bool(p.get('use_elder', USE_ELDER))
    USE_RSI = bool(p.get('use_rsi', USE_RSI))
    SL_PTS = float(p.get('sl', SL_PTS))
    TP_PTS = float(p.get('tp', TP_PTS))
    args.atr_sl = bool(p.get('atr_sl', getattr(args, 'atr_sl', False)))
    args.atr_mult = float(p.get('atr_mult', getattr(args, 'atr_mult', 1.0)))
    args.atr_period = int(p.get('atr_period', getattr(args, 'atr_period', 14)))
    args.bias_tf = p.get('bias_tf', getattr(args, 'bias_tf', '15m'))
    args.cap = int(p.get('cap', 0))
    rev = bool(p.get('reversal', False))
    pe_rev_full = pe_rev_on if rev else pe_rev_off
    ce_rev_full = ce_rev_on if rev else ce_rev_off
    if args.atr_sl:
        ce_atr = compute_atr(ce_h, ce_l, ce_c, args.atr_period)
        pe_atr = compute_atr(pe_h, pe_l, pe_c, args.atr_period)
    trades = run_full_backtest(cap_val=args.cap, workers=int(p.get('workers', 8)))
    return _metrics_from_trades(trades)


print("\n" + "=" * 85)
def run_engine():
    print(f" EXECUTING 7-YEAR NON-WALK FORWARD RUN (CAP = {args.cap})")
    print("=" * 85)
    t_start = time.time()
    _mark("->backtest_run")
    trades_res = run_full_backtest(cap_val=args.cap, workers=args.workers)
    t_elapsed = time.time() - t_start
    _mark("backtest_run_done")
    print(f"Backtest Completed in {t_elapsed:.2f}s | Total Trades: {len(trades_res)}")

    if trades_res:
        df_t = pd.DataFrame(trades_res, columns=["day", "side", "result", "entry", "exit", "pnl"])
        wins = (df_t["pnl"] > 0).sum()
        total = len(df_t)
        wr = wins / total * 100
        net_pnl = df_t["pnl"].sum()
        avg_pnl = net_pnl / total
    
        print("\n" + "-" * 85)
        print(f"OVERALL RESULTS (2020 - 2026):")
        print(f"  * Total Trades:   {total:,}")
        print(f"  * Wins / Losses:  {wins:,} / {total - wins:,}")
        print(f"  * Win Rate:       {wr:.2f}%")
        print(f"  * Total Net PnL:  Rs.{net_pnl:+,.0f}")
        print(f"  * Avg PnL / Trade:Rs.{avg_pnl:+,.1f}")
        print("-" * 85)
    
        print("\nYEARLY PERFORMANCE BREAKDOWN:")
        yr_df = df_t.groupby(df_t["day"].str[:4]).agg(
            trades=('pnl', 'count'),
            wins=('pnl', lambda x: (x > 0).sum()),
            win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
            net_pnl=('pnl', 'sum')
        )
        print(yr_df.to_string())
    
        print("\nSIDE PERFORMANCE BREAKDOWN (PE vs CE):")
        side_df = df_t.groupby("side").agg(
            trades=('pnl', 'count'),
            wins=('pnl', lambda x: (x > 0).sum()),
            win_rate=('pnl', lambda x: (x > 0).sum() / len(x) * 100),
            net_pnl=('pnl', 'sum')
        )
        print(side_df.to_string())
    
        out_csv = f"artifacts/f6_hybrid/trades_7y_v4_master_{args.bias_tf}_cap{args.cap}_sl{int(args.sl)}_tp{int(args.tp)}.csv"
        df_t.to_csv(out_csv, index=False)
        print(f"\nSaved trade ledger to: {out_csv}")

    # 6. Walk-Forward Validation (5 Folds)
    if args.walkforward or not args.smoke:
        print("\n" + "=" * 85)
        print(" EXECUTING 5-FOLD WALK-FORWARD OUT-OF-SAMPLE VALIDATION")
        print("=" * 85)
    
        # 5 Walk-Forward Splits
        splits = [
            ("Fold 1 (Train: 2020-2021 | Test: 2022)", "2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
            ("Fold 2 (Train: 2020-2022 | Test: 2023)", "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
            ("Fold 3 (Train: 2020-2023 | Test: 2024)", "2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
            ("Fold 4 (Train: 2020-2024 | Test: 2025)", "2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
            ("Fold 5 (Train: 2020-2025 | Test: 2026)", "2020-01-01", "2025-12-31", "2026-01-01", "2026-08-27"),
        ]
    
        wf_results = []
        for label, tr_start, tr_end, te_start, te_end in splits:
            # Filter test days
            te_idxs = [i for i, d in enumerate(trading_days) if te_start <= d <= te_end]
            if not te_idxs:
                continue
            
            test_trades = process_days_chunk(te_idxs, cap_val=args.cap)
            df_te = pd.DataFrame(test_trades, columns=["day", "side", "result", "entry", "exit", "pnl"])
        
            te_cnt = len(df_te)
            te_wins = (df_te["pnl"] > 0).sum() if te_cnt > 0 else 0
            te_wr = (te_wins / te_cnt * 100) if te_cnt > 0 else 0
            te_pnl = df_te["pnl"].sum() if te_cnt > 0 else 0
        
            wf_results.append({
                "Fold": label.split(":")[0].replace("Fold ", "F"),
                "Test Period": f"{te_start[:4]}",
                "OOS Trades": te_cnt,
                "OOS Win Rate": f"{te_wr:.1f}%",
                "OOS Net PnL": f"Rs.{te_pnl:+,.0f}"
            })
    
        print(pd.DataFrame(wf_results).to_string(index=False))
        print("=" * 85)


if __name__ == "__main__":
    run_engine()