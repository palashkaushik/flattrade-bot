"""F10 CERTIFIED RECORDER v3 — champion Full10 (S1 12,2 / S3 40,4 / S4 60,10).

Records raw arrays per day for the certified Full10 sweep:
  has(C,NT) u8, ohlc(C,NT,4), baridx(C,NT) i64, spec(NT,2) i32
  st(16,C,NT) f8    4 TFs x (s1, s3, s4, prev_s1), stochset f10
  ind(C,NT,10) f8   [atr, ema20, ema200, vwap(EQ-WEIGHT hlc3),
                     2m lo/cl, 3m lo/cl, 5m lo/cl]
  ema4(4,C,NT) f8   per-TF EMA200 (1m, 2m, 3m, 5m), carried between closes
  dorm(4,C,NT) i32  S4<20 run-length in COMPLETED TF candles
  s1a(C,NT) u8      any-TF S1<=30 (arming)
  pppF(C,10) f8     [PDH, PDL, C, CPR_r, BC_r, TC_r, fibR3, fibS3, camR3, camL3]
                    prior FULL session, half-up rounded CPR/BC/TC
  pdv(C) f8         prior-day eq-weight hlc3 mean (PDVWAP)
  vir(C,3) f8       virgin CPR/BC/TC values (family first day, r2)
  valive(C,3) u8    alive at day open (touched in prior days?)

WARMUP: 3 prior trading days seed EMA20/200, ATR, stochs, per-TF EMA200,
dormancy (all available history capped at 3 days; new families start fresh —
matches Upstox charts which only have the contract's own listed history).

MODES:
  weekly  — weekly_08SEP26 folder CSVs (Sep 2-7 2026; parity days)
  parquet — canonical parquet v2 (7-year certified run; virgin map from
            lh_f10_virgin.py pre-pass JSON)

Session: T0=555, NT=345 (09:15-15:00 certified window).
"""
import sys, os, json, time, argparse
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

os.environ["DISCORD_WEBHOOK_URL"] = ""
from flattrade_bot.strategies.last_hope_winner import (
    Bar1m, IncrementalStoch, IncrementalATR, IncrementalEMA, IncrementalVWAP, TFTracker,
)

S1K, S1D = 12, 2
S3K, S3D = 40, 4
S4K, S4D = 60, 10
T0, NT = 555, 345
DORM_S4 = 20.0

WEEKLY = r"C:\Users\user\Desktop\nifty50 data\weekly_08SEP26"
PARQ = r"C:\Users\user\AppData\Local\Temp\opencode\data\nifty50_options_master_v2.parquet"
IDXC = r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
VIRGIN_MAP = r"C:\Websites\FLATTRADE BOT\lh_f10_virgin_map.json"
OUTBASE = r"C:\Users\user\AppData\Local\Temp\opencode\lh_events_f10"

def r2(x):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def is_tf_close(m, tf):
    return ((m + 1 - T0) % tf) == 0

def mk_bar(m, o, h, l, c):
    return Bar1m(minute=int(m), open=float(o), high=float(h), low=float(l),
                 close=float(c), timestamp=pd.Timestamp("2000-01-01") + pd.Timedelta(minutes=int(m)))

def day_rows(g):
    """Full session rows (>= 09:15), sorted."""
    return g[(g.minute >= T0)].sort_values("minute")

def load_weekly(day):
    df = pd.read_csv(os.path.join(WEEKLY, f"options_{day}.csv"))
    tm = pd.to_datetime(df["time"], format="%H:%M:%S")
    df["minute"] = tm.dt.hour * 60 + tm.dt.minute
    import re
    pat = re.compile(r"NIFTY\d{2}[A-Z]{3}\d{2}(\d{5})(CE|PE)$")
    ks = [pat.match(str(s)) for s in df["symbol"]]
    df = df[[m is not None for m in ks]]
    ks = [m for m in ks if m is not None]
    df = df.assign(strike=[int(m.group(1)) for m in ks], side=[m.group(2) for m in ks])
    return {s: g for s, g in df.groupby("symbol")}

def load_spot_weekly(day):
    df = pd.read_csv(os.path.join(WEEKLY, f"index_1m_{day}.csv"))
    tm = pd.to_datetime(df["time"], format="%H:%M:%S")
    return {int(h*60+mi): float(c) for h, mi, c in zip(tm.dt.hour, tm.dt.minute, df["close"])}

def build_day(d_str, frames, prior_list, sp, virgin_map=None, official=None,
              pdv_vol=True):
    """frames: {sym: full-day df}; prior_list: [prior day frames, oldest..latest] (up to 3)."""
    elig = set()
    for m in range(T0, T0 + NT):
        s = sp.get(m)
        if s:
            atm = int(round(s / 50) * 50)
            elig.add(f"CE:{atm-100}"); elig.add(f"PE:{atm+100}")
    if not elig:
        return None
    key_sym = {}
    for sym, g in frames.items():
        k = f"{str(g.side.iloc[0])}:{int(g.strike.iloc[0])}"
        if k in elig:
            key_sym[k] = sym
    syms = sorted(key_sym.values())
    if not syms:
        return None
    C = len(syms)
    sym_idx = {s: i for i, s in enumerate(syms)}

    has = np.zeros((C, NT), dtype=np.uint8)
    ohlc = np.zeros((C, NT, 4), dtype=np.float64)
    baridx = np.zeros((C, NT), dtype=np.int64)
    ind = np.full((C, NT, 10), np.nan, dtype=np.float64)
    st = np.full((4, 4, C, NT), np.nan, dtype=np.float64)
    ema4 = np.full((4, C, NT), np.nan, dtype=np.float64)
    pppF = np.zeros((C, 10), dtype=np.float64)
    pdv = np.zeros(C, dtype=np.float64)
    vir = np.full((C, 3), np.nan, dtype=np.float64)
    valive = np.zeros((C, 3), dtype=np.uint8)
    spec = np.full((NT, 2), -1, dtype=np.int32)
    seed_dorm = np.zeros((C, 4), dtype=np.int32)   # carried S4<20 run per TF

    prior_frames = prior_list[-1] if prior_list else {}

    for i, sym in enumerate(syms):
        g = day_rows(frames[sym])
        # ---- trackers (fresh; seeded below with up to 3 prior days) ----
        atr = IncrementalATR(10)
        ema20 = IncrementalEMA(20); ema200 = IncrementalEMA(200)
        vwap = IncrementalVWAP()
        tf = {}
        for tfv in (1, 2, 3, 5):
            t = TFTracker(tfv)
            t.s1 = IncrementalStoch(S1K, S1D); t.s3 = IncrementalStoch(S3K, S3D)
            t.s4 = IncrementalStoch(S4K, S4D)
            tf[tfv] = t
        ema200tf = {tfv: IncrementalEMA(200) for tfv in (1, 2, 3, 5)}
        dorm = {tfv: 0 for tfv in (1, 2, 3, 5)}
        nseed = 0
        for pf in prior_list:            # oldest -> latest
            pg = pf.get(sym)
            if pg is None:
                continue
            for r in day_rows(pg).itertuples(index=False):
                m_ = int(r.minute)
                b = mk_bar(m_, r.open, r.high, r.low, r.close)
                atr.update(b.high, b.low, b.close)
                ema20.update(b.close); ema200.update(b.close)
                for tfv in (1, 2, 3, 5):
                    t = tf[tfv]
                    t.push_1m_bar(b)
                    if is_tf_close(m_, tfv):
                        ema200tf[tfv].update(b.close)
                        s4 = t.last_s4
                        if s4 is not None:
                            dorm[tfv] = dorm[tfv] + 1 if s4 < DORM_S4 else 0
                nseed += 1
        seed_dorm[i] = [dorm[1], dorm[2], dorm[3], dorm[5]]

        # PARITY: geoms resets TF fallback state (last_lo/last_cl) at the day
        # boundary — the first 2m/3m/5m "last completed bar" must be TODAY's,
        # never Friday's. Mirror that here before recording today's bars.
        for tfv in (2, 3, 5):
            tf[tfv].last_lo = None; tf[tfv].last_cl = None

        # ---- pivots: prior day FULL session H/L/C ----
        # WEEKLY PARITY: official Upstox closes (= NSE ClsPric) when provided;
        # PARQUET: vendor CSV last-bar close (documented approximation).
        pg = prior_frames.get(sym)
        if pg is not None:
            pr = day_rows(pg)
            if len(pr):
                # official = {sym: {...day's OHLC...}} ALREADY SLICED to the
                # prior trading day by the caller (weekly) — or None (parquet)
                off = (official or {}).get(sym)
                if off is not None:
                    ph = float(off["H"]); pl = float(off["L"]); pc = float(off["C"])
                    cpr = float(off["CPR"]); bc = float(off["BC"]); tc = float(off["TC"])
                else:
                    ph = float(pr.high.max()); pl = float(pr.low.min()); pc = float(pr.close.iloc[-1])
                    cpr = r2((ph + pl + pc) / 3.0); bc = r2((ph + pl) / 2.0); tc = r2(2.0 * cpr - bc)
                rng = ph - pl
                pppF[i] = [ph, pl, pc, cpr, bc, tc,
                           cpr + rng, cpr - rng, pc + rng * 1.1 / 4.0, pc - rng * 1.1 / 4.0]
                # PDVWAP: prior-day hlc3 mean — volume-weighted when volume
                # available (weekly), eq-weight otherwise (parquet has no vol)
                if pdv_vol and "volume" in pr.columns:
                    h3 = (pr.high + pr.low + pr.close) / 3.0
                    w = pr.volume.values.astype(float)
                    pdv[i] = float((h3.values * w).sum() / w.sum())
                else:
                    pdv[i] = float(((pr.high + pr.low + pr.close) / 3.0).mean())
        # ---- virgin CPR: family first day (from map or local prior walk) ----
        vm = (virgin_map or {}).get(sym)
        if vm is not None:
            vv = vm["vals"]; vir[i] = vv
            alive = vm["alive"]   # alive at day open (prior days walked)
            # walk TODAY at runtime in the kernel; here just record open state
            valive[i] = alive
        # ---- today ----
        g_today = g[(g.minute >= T0) & (g.minute < T0 + NT)]
        has_vol = "volume" in g_today.columns
        tfstate = {"baridx": -1}
        for r in g_today.itertuples(index=False):
            t = int(r.minute) - T0
            b = mk_bar(r.minute, r.open, r.high, r.low, r.close)
            atr.update(b.high, b.low, b.close)
            ema20.update(b.close); ema200.update(b.close)
            # VWAP: volume-weighted when volume exists (weekly parity), else
            # eq-weight (parquet has no volume column — documented difference)
            if has_vol:
                vwap.update(b.high, b.low, b.close, volume=float(r.volume))
            else:
                vwap.update(b.high, b.low, b.close)
            for tfv in (1, 2, 3, 5):
                tk = tf[tfv]
                tk.push_1m_bar(b)
                if is_tf_close(int(r.minute), tfv):
                    ema200tf[tfv].update(b.close)
                    s4 = tk.last_s4
                    if s4 is not None:
                        dorm[tfv] = dorm[tfv] + 1 if s4 < DORM_S4 else 0
            tfstate["baridx"] += 1
            has[i, t] = 1
            ohlc[i, t, 0] = b.open; ohlc[i, t, 1] = b.high
            ohlc[i, t, 2] = b.low; ohlc[i, t, 3] = b.close
            baridx[i, t] = nseed + tfstate["baridx"]
            ind[i, t, 0] = atr.value
            ind[i, t, 1] = ema20.value if ema20.value is not None else np.nan
            ind[i, t, 2] = ema200.value if ema200.value is not None else np.nan
            ind[i, t, 3] = vwap.value if vwap.value is not None else np.nan
            for j, tfv in enumerate((2, 3, 5)):
                tk = tf[tfv]
                ind[i, t, 4 + 2*j] = tk.last_lo if tk.last_lo is not None else np.nan
                ind[i, t, 5 + 2*j] = tk.last_cl if tk.last_cl is not None else np.nan
            for j, tfv in enumerate((1, 2, 3, 5)):
                tk = tf[tfv]
                st[j, 0, i, t] = tk.last_s1 if tk.last_s1 is not None else np.nan
                st[j, 1, i, t] = tk.last_s3 if tk.last_s3 is not None else np.nan
                st[j, 2, i, t] = tk.last_s4 if tk.last_s4 is not None else np.nan
                st[j, 3, i, t] = tk.prev_s1 if tk.prev_s1 is not None else np.nan
                e = ema200tf[tfv].value
                ema4[j, i, t] = e if e is not None else np.nan
        # dorm carry into bar array
        for j, tfv in enumerate((1, 2, 3, 5)):
            run = dorm[tfv]
            for t in range(NT):
                if has[i, t] and is_tf_close(T0 + t, tfv):
                    s4 = st[j, 2, i, t]
                    if s4 == s4:
                        run = run + 1 if s4 < DORM_S4 else 0
                # store
            # (loop above recomputes; simpler: store during today pass)
        # store dorm per bar (carried)
        # NOTE: recomputed below in vector pass for exactness
    # ---- dorm array: run-length per (tf, c, t) WITH seed carry ----
    dorm_arr = np.zeros((4, C, NT), dtype=np.int32)
    for j, tfv in enumerate((1, 2, 3, 5)):
        for i in range(C):
            run = int(seed_dorm[i, j])
            for t in range(NT):
                if has[i, t] and is_tf_close(T0 + t, tfv):
                    s4 = st[j, 2, i, t]
                    if s4 == s4:
                        run = run + 1 if s4 < DORM_S4 else 0
                dorm_arr[j, i, t] = run
    # ---- s1a: any-TF S1<=30 ----
    s1a = np.zeros((C, NT), dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        for j in range(4):
            s1 = st[j, 0]           # (C, NT)
            s1a |= ((s1 <= 30.0) & (s1 == s1)).astype(np.uint8)
    # ---- spec map (kept for compat; strategy scans all day-eligible c) ----
    for t in range(NT):
        s = sp.get(T0 + t)
        if not s:
            continue
        atm = int(round(s / 50) * 50)
        keys = {f"CE:{atm-100}", f"PE:{atm+100}"}
        slot = 0
        for sym in syms:
            k = f"{str(frames[sym].side.iloc[0])}:{int(frames[sym].strike.iloc[0])}"
            if k in keys and has[sym_idx[sym], t]:
                spec[t, slot] = sym_idx[sym]
                slot += 1
                if slot == 2:
                    break
    return dict(date=int(d_str.replace("-", "")), n_c=C, has=has, ohlc=ohlc,
                baridx=baridx, ind=ind, st=st.reshape(16, C, NT), ema4=ema4,
                dorm=dorm_arr, s1a=s1a, pppF=pppF, pdv=pdv, vir=vir,
                valive=valive, spec=spec)

def weekly_run(days, outdir):
    os.makedirs(outdir, exist_ok=True)
    all_days = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07"]
    frames = {d: load_weekly(d) for d in all_days}
    spots = {d: load_spot_weekly(d) for d in days}
    # official pivots (Upstox daily = NSE ClsPric) for prior days
    piv = json.load(open(r"C:\Websites\FLATTRADE BOT\lh_pivots_upstox.json"))
    # virgin map for weekly family (first day Sep 1, touched-walk over priors)
    vmap = build_weekly_virgin(frames, all_days, days)
    for d in days:
        di = all_days.index(d)
        pd_ = all_days[di - 1] if di > 0 else None
        prior_list = [frames[p] for p in all_days[max(0, di - 3):di]]
        vmap_day = {sym: per_day[d] for sym, per_day in vmap.items() if d in per_day}
        # official pivots for the PRIOR day, keyed by vendor sym
        official = {sym: pv[pd_] for sym, pv in piv.items()
                    if pd_ and pd_ in pv} if pd_ else None
        out = build_day(d, frames[d], prior_list, spots[d], virgin_map=vmap_day,
                        official=official)
        if out is not None:
            np.savez_compressed(os.path.join(outdir, f"ev_{d.replace('-','')}.npz"), **out)
            print(f"  built {d}: C={out['n_c']}")

def build_weekly_virgin(frames, all_days, target_days):
    """Virgin CPR map for the weekly family: values from family FIRST day
    (Sep 1), alive flags walked over days between first and each target."""
    first = all_days[0]
    vmap = {}
    # family first-day session stats per sym
    firstday_vals = {}
    f0 = frames[first]
    for sym, g in f0.items():
        pr = day_rows(g)
        if not len(pr):
            continue
        h = float(pr.high.max()); l = float(pr.low.min()); c = float(pr.close.iloc[-1])
        firstday_vals[sym] = [r2((h + l + c) / 3.0), r2((h + l) / 2.0),
                              r2(2.0 * r2((h + l + c) / 3.0) - r2((h + l) / 2.0))]
    for d in target_days:
        di = all_days.index(d)
        for sym, vals in firstday_vals.items():
            if sym not in frames[d]:
                continue
            alive = [1, 1, 1]
            for pd in all_days[1:di]:   # days after first, before target
                pg = frames[pd].get(sym)
                if pg is None:
                    continue
                for r in day_rows(pg).itertuples(index=False):
                    lo_, hi_ = float(r.low), float(r.high)
                    for li in range(3):
                        if alive[li] and lo_ <= vals[li] <= hi_:
                            alive[li] = 0
            vmap.setdefault(sym, {})[d] = dict(vals=vals, alive=alive)
    # restructure: {sym: {day: ...}} per day target — build_day wants per-day
    out = {}
    for sym, per_day in vmap.items():
        # use the LAST target day entry per sym? build_day is per-day; caller
        # passes per-day map. We return nested and let weekly_run slice.
        out[sym] = per_day
    return out

def parquet_run(first=None, last=None, workers=8, probe=None):
    """7-year certified recording from parquet v2."""
    alldays = sorted(pd.to_datetime(
        pd.read_parquet(PARQ, columns=["day"]).day.unique()))
    day_strs = [str(d.date()) for d in alldays]
    days = alldays
    if first:
        days = [d for d in days if d >= pd.Timestamp(first)]
    if last:
        days = [d for d in days if d <= pd.Timestamp(last)]
    d0 = alldays.index(days[0]); d1 = alldays.index(days[-1]) + 1
    if probe:
        d1 = min(d1, d0 + probe)
    virgin = json.load(open(VIRGIN_MAP))
    idx = pd.read_csv(IDXC)
    tc = idx.columns[0]
    idx[tc] = pd.to_datetime(idx[tc])
    idx_by_day = {}
    for d in days:
        i = idx[(idx[tc] >= d) & (idx[tc] < d + pd.Timedelta(days=1))]
        sp = {}
        for r in i.itertuples(index=False):
            dt = getattr(r, tc)
            m = dt.hour * 60 + dt.minute
            if T0 <= m < T0 + NT:
                sp[m] = float(getattr(r, "close"))
        idx_by_day[str(d.date())] = sp
    outdir = os.path.join(OUTBASE, "base")
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    done = 0
    for di in range(d0, d1):
        d = day_strs[di]
        prior_strs = day_strs[max(0, di - 3):di]
        pq = pd.read_parquet(PARQ, columns=["day", "minute", "symbol", "strike", "side",
                                            "open", "high", "low", "close"],
                             filters=[("day", "in", [d] + prior_strs)])
        pq["day"] = pd.to_datetime(pq["day"])
        frames = {s: g for s, g in pq[pq.day == pd.Timestamp(d)].groupby("symbol")}
        prior_list = []
        for p in prior_strs:
            pd_ = pq[pq.day == pd.Timestamp(p)]
            prior_list.append({s: g for s, g in pd_.groupby("symbol")})
        vmap_day = {sym: virgin[sym][d] for sym in set(virgin) & set(frames)
                    if d in virgin.get(sym, {})}
        out = build_day(d, frames, prior_list, idx_by_day[d], virgin_map=vmap_day)
        if out is not None:
            np.savez_compressed(os.path.join(outdir, f"ev_{d.replace('-','')}.npz"), **out)
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{d1-d0} days ({time.time()-t0:.0f}s)", flush=True)
    print(f"F10 BUILD DONE: {done} days in {time.time()-t0:.0f}s -> {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["weekly", "parquet"], default="weekly")
    ap.add_argument("--days", default="2026-09-04,2026-09-07")
    ap.add_argument("--first"); ap.add_argument("--last")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probe", type=int)
    a = ap.parse_args()
    if a.mode == "weekly":
        weekly_run(a.days.split(","), os.path.join(OUTBASE, "weekly"))
    else:
        parquet_run(a.first, a.last, a.workers, a.probe)
