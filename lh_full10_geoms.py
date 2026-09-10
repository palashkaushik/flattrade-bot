"""FULL10 — FINALIZED USER SPEC v5 — Sep 2/3/4/7, 10 exit geometries
+ DAILY CIRCUIT BREAKER (max daily cap): cumulative closed day P&L in
POINTS; touch +cap (profit locked) or -cap (loss capped) -> no more
entries, out of the market for the day.

v4 changes:
  - S4 DORMANCY counted in COMPLETED CANDLES on that chart's TF
    (increment only at TF bucket boundary), not 1m pushes. >14 candles
    under 20 -> ignore that TF's signals until S4 >= 20.
  - Days: Sep 4 (Fri) and Sep 7 (Mon) — Sep 5/6 2026 = weekend.

Geometries (per user):
  G1  SL = trigger 1m candle low - 2,      TP = entry + 15
  G2  SL = entry - 15,                     TP = entry + 15
  G3  SL = entry - 10,                     TP = entry + 10
  G4  SL = trigger 1m candle low - 2,     TP = entry + 10
  G5  SL = entry - 7,                      TP = entry + 10
  G6  SL = entry - 7,                      TP = entry + 15

All other spec = v3 (5-pt stoch buffers, 12+ levels incl virgin CPR +
PDVWAP, TB=2, no TF leak, single position, no BE, EOD close).
"""
import sys, os, re, json
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

os.environ["DISCORD_WEBHOOK_URL"] = ""
from flattrade_bot.strategies.last_hope_winner import (
    Bar1m, IncrementalStoch, IncrementalATR, IncrementalEMA, IncrementalRMA, IncrementalVWAP, TFTracker,
)

SRC = r"C:\Users\user\Desktop\nifty50 data\weekly_08SEP26"
PIVFILE = r"C:\Websites\FLATTRADE BOT\lh_pivots_upstox.json"
SESSION_START = 555

S1K, S1D = 12, 2
S3K, S3D = 40, 4
S4K, S4D = 60, 10
ARM_S1 = 30.0
ARM_WIN = 1
FLAG_S4 = 75.0
FLAG_S1 = 30.0
SUP_T = 30.0
TB = 2.0
DORM_S4 = 20.0
DORM_CANDLES = 14       # ignore TF if S4 under 20 for MORE THAN 14 completed candles
T0, TEND = 555, 915

DAYCFG = {
    "2026-09-02": dict(seed=["2026-09-01"], prev="2026-09-01",
                       vfrom="2026-09-01", vsince=[]),
    "2026-09-03": dict(seed=["2026-09-01", "2026-09-02"], prev="2026-09-02",
                       vfrom="2026-09-01", vsince=["2026-09-02"]),
    "2026-09-04": dict(seed=["2026-09-01", "2026-09-02", "2026-09-03"], prev="2026-09-03",
                       vfrom="2026-09-01", vsince=["2026-09-02", "2026-09-03"]),
    "2026-09-07": dict(seed=["2026-09-02", "2026-09-03", "2026-09-04"], prev="2026-09-04",
                       vfrom="2026-09-01", vsince=["2026-09-02", "2026-09-03", "2026-09-04"]),
    "2026-09-08": dict(seed=["2026-09-03", "2026-09-04", "2026-09-07"], prev="2026-09-07",
                       vfrom="2026-09-01", vsince=["2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07"]),
    "2026-09-09": dict(seed=["2026-09-04", "2026-09-07", "2026-09-08"], prev="2026-09-08",
                       vfrom="2026-09-03", vsince=["2026-09-04", "2026-09-07", "2026-09-08"]),
    "2026-09-10": dict(seed=["2026-09-07", "2026-09-08", "2026-09-09"], prev="2026-09-09",
                       vfrom="2026-09-03", vsince=["2026-09-04", "2026-09-07", "2026-09-08", "2026-09-09"]),
}

DAYS_ALL = ("2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08", "2026-09-09")

def r2(x):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def parse_sym(sym):
    m = re.match(r"NIFTY\d{2}[A-Z]{3}\d{2}(\d{5})(CE|PE)$", str(sym).strip())
    return (int(m.group(1)), m.group(2)) if m else (None, None)

_cache = {}
def load(day):
    if day not in _cache:
        df = pd.read_csv(os.path.join(SRC, f"options_{day}.csv"))
        tm = pd.to_datetime(df["time"], format="%H:%M:%S")
        df["minute"] = tm.dt.hour * 60 + tm.dt.minute
        ks = [parse_sym(s) for s in df["symbol"]]
        df["strike"] = [k[0] for k in ks]; df["side"] = [k[1] for k in ks]
        _cache[day] = {s: g.sort_values("minute") for s, g in df.groupby("symbol")}
    return _cache[day]

def load_spot(day):
    df = pd.read_csv(os.path.join(SRC, f"index_1m_{day}.csv"))
    tm = pd.to_datetime(df["time"], format="%H:%M:%S")
    return {int(h*60+mi): float(c) for h, mi, c in zip(tm.dt.hour, tm.dt.minute, df["close"])}

piv = json.load(open(PIVFILE))

def mk_bar(r, m):
    return Bar1m(minute=m, open=float(r.open), high=float(r.high), low=float(r.low),
                 close=float(r.close), timestamp=pd.Timestamp("2000-01-01") + pd.Timedelta(minutes=m))

def is_tf_close(m, tf):
    return ((m + 1 - SESSION_START) % tf) == 0

def build_states(day):
    cfg = DAYCFG[day]
    today = load(day); spot = load_spot(day)
    elig = set()
    for m in range(T0, TEND):
        s = spot.get(m)
        if s is None: continue
        atm = int(round(s / 50) * 50)
        elig.add(f"CE:{atm-100}"); elig.add(f"PE:{atm+100}")
    key_sym = {}
    for sym, g in today.items():
        k = f"{str(g.side.iloc[0])}:{int(g.strike.iloc[0])}"
        if k in elig:
            key_sym[k] = sym

    states = {}
    for k, sym in key_sym.items():
        cs = lambda: None
        cs.sym = sym; cs.key = k
        cs.atr = IncrementalATR(10); cs.ema20 = IncrementalRMA(20); cs.ema200 = IncrementalRMA(200)
        cs.vwap = IncrementalVWAP()
        cs.tf = {}
        for tf in (1, 2, 3, 5):
            t = TFTracker(tf)
            t.s1 = IncrementalStoch(S1K, S1D); t.s3 = IncrementalStoch(S3K, S3D); t.s4 = IncrementalStoch(S4K, S4D)
            cs.tf[tf] = t
        cs.baridx = -1; cs.anchor_bar = -10**9
        cs.dorm = {tf: 0 for tf in (1, 2, 3, 5)}
        cs.tf200 = {tf: IncrementalEMA(200) for tf in (1, 2, 3, 5)}

        pv = piv.get(sym, {}).get(cfg["prev"])
        if pv:
            cs.ph, cs.pl, cs.pc = float(pv["H"]), float(pv["L"]), float(pv["C"])
            cs.cpr, cs.bc, cs.tc = float(pv["CPR"]), float(pv["BC"]), float(pv["TC"])
        else:
            pg = load(cfg["prev"]).get(sym)
            if pg is not None and len(pg):
                rows = pg[pg.minute >= 555]
                cs.ph = float(rows.high.max()); cs.pl = float(rows.low.min()); cs.pc = float(rows.close.iloc[-1])
                cs.cpr = r2((cs.ph + cs.pl + cs.pc) / 3.0); cs.bc = r2((cs.ph + cs.pl) / 2.0)
                cs.tc = r2(2.0 * cs.cpr - cs.bc)
            else:
                cs.ph = cs.pl = cs.pc = np.nan; cs.cpr = cs.bc = cs.tc = np.nan

        cs.virgin = {}
        v1 = piv.get(sym, {}).get(cfg["vfrom"])
        if v1:
            cs.virgin = {"VCPR": float(v1["CPR"]), "VBC": float(v1["BC"]), "VTC": float(v1["TC"])}
            for d in cfg["vsince"]:
                pg = load(d).get(sym)
                if pg is None: continue
                for r in pg[pg.minute >= 555].itertuples(index=False):
                    lo_, hi_ = float(r.low), float(r.high)
                    for nm in list(cs.virgin):
                        if lo_ <= cs.virgin[nm] <= hi_:
                            del cs.virgin[nm]

        for d in cfg["seed"]:
            pg = load(d).get(sym)
            if pg is None: continue
            for r in pg[pg.minute >= 555].tail(1200).itertuples(index=False):
                m_ = int(r.minute)
                b = mk_bar(r, m_)
                cs.atr.update(b.high, b.low, b.close)
                cs.ema20.update(b.close); cs.ema200.update(b.close)
                for tf in (1, 2, 3, 5):
                    t = cs.tf[tf]
                    t.push_1m_bar(b)
                    if is_tf_close(m_, tf):
                        s4 = t.last_s4
                        if s4 is not None:
                            cs.dorm[tf] = cs.dorm[tf] + 1 if s4 < DORM_S4 else 0
                        cs.tf200[tf].update(b.close)

        pg = load(cfg["prev"]).get(sym)
        cs.pdvwap = None
        if pg is not None:
            vw = IncrementalVWAP()
            for r in pg[pg.minute >= 555].itertuples(index=False):
                vw.update(float(r.high), float(r.low), float(r.close), volume=float(r.volume))
            cs.pdvwap = vw.value

        for tf in (1, 2, 3, 5):
            cs.tf[tf].last_lo = None; cs.tf[tf].last_cl = None
        states[k] = cs
    return states, today, spot

def trig_of(cs):
    out = []
    for tf in (1, 2, 3, 5):
        t = cs.tf[tf]
        s1, s3, s4, p1 = t.last_s1, t.last_s3, t.last_s4, t.prev_s1
        if s1 is None or s4 is None: continue
        if s4 >= FLAG_S4 and s1 < FLAG_S1:
            out.append((tf, "FLAG"))
        if s3 is not None and p1 is not None:
            if s3 < SUP_T and s4 < SUP_T and s1 < SUP_T and s1 > p1:
                out.append((tf, "SUPER"))
    return out

def levels(cs):
    ph, pl, pc = cs.ph, cs.pl, cs.pc
    lv = [("EMA20", cs.ema20.value), ("EMA200", cs.ema200.value), ("VWAP", cs.vwap.value),
          ("CPR", cs.cpr), ("BC", cs.bc), ("TC", cs.tc)]
    if pc == pc:
        rng = ph - pl
        lv += [("PDH", ph), ("PDL", pl),
               ("fibR3", cs.cpr + rng), ("fibS3", cs.cpr - rng),
               ("camR3", pc + rng * 1.1 / 4.0), ("camL3", pc - rng * 1.1 / 4.0)]
    if cs.pdvwap is not None:
        lv.append(("PDVWAP", cs.pdvwap))
    lv += list(cs.virgin.items())
    return [(n, v) for n, v in lv if v is not None and v == v]

def bounce(cs, bar, tf):
    # v6 (user rule 2026-09-09): the bounce must occur on the SETUP's OWN TF
    # chart. 1m setup -> the trigger 1m bar; 2m/3m/5m setup -> that TF's last
    # COMPLETED (today's) bar. No cross-TF fallback, no 1m confirmation.
    if tf == 1:
        lo_, cl_ = bar.low, bar.close
    else:
        t = cs.tf[tf]
        if t.last_lo is None or t.last_cl is None:
            return None
        lo_, cl_ = t.last_lo, t.last_cl
    for name, v in levels(cs):
        if lo_ <= v + TB and cl_ >= v - 0.5:
            return name, f"{tf}m"
    return None

def hm(m):
    return f"{m//60:02d}:{m%60:02d}"

def collect_signals(day):
    states, today, spot = build_states(day)
    sigs = []
    bars = {}   # (key, minute) -> (low, high, close, lastbar_close)
    for m in range(T0, TEND):
        for k, cs in states.items():
            g = today[cs.sym]
            row = g[g.minute == m]
            if row.empty: continue
            r = row.iloc[0]
            bar = mk_bar(r, m)
            bars[(k, m)] = (bar.low, bar.high, bar.close, bar.low)
            cs.atr.update(bar.high, bar.low, bar.close)
            cs.ema20.update(bar.close); cs.ema200.update(bar.close)
            cs.vwap.update(bar.high, bar.low, bar.close, volume=float(r.volume))
            for tf in (1, 2, 3, 5):
                t = cs.tf[tf]
                t.push_1m_bar(bar)
                if is_tf_close(m, tf):
                    s4 = t.last_s4
                    if s4 is not None:
                        cs.dorm[tf] = cs.dorm[tf] + 1 if s4 < DORM_S4 else 0
                    cs.tf200[tf].update(bar.close)
            for nm in list(cs.virgin):
                if bar.low <= cs.virgin[nm] <= bar.high:
                    del cs.virgin[nm]
            cs.baridx += 1
            if any(cs.tf[tf].last_s1 is not None and cs.tf[tf].last_s1 <= ARM_S1 for tf in (1, 2, 3, 5)):
                cs.anchor_bar = cs.baridx
            if (cs.baridx - cs.anchor_bar) > ARM_WIN:
                continue
            trigs = trig_of(cs)
            if not trigs: continue
            tf, kind = trigs[0]
            b = bounce(cs, bar, tf)
            if b:
                e200 = {tf_: (cs.tf200[tf_].value if cs.tf200[tf_].value is not None else
                              (cs.ema200.value if tf_ == 1 else None)) for tf_ in (1, 2, 3, 5)}
                sigs.append(dict(minute=m, key=k, kind=kind, tf=tf,
                                 lvl=b[0], via=b[1], entry=float(bar.close),
                                 trig_lo=float(bar.low), atr=cs.atr.value,
                                 above200={t_: (e200[t_] is not None and bar.close >= e200[t_])
                                          for t_ in (1, 2, 3, 5)}))
    return states, today, sigs, bars

GEOMS = [
    ("G0 ATR*1.5 (default)", lambda s: (s["entry"] - round(min(max(s["atr"] * 1.5, 2.0), 15.0), 2),
                                        s["entry"] + round(min(max(s["atr"] * 1.5, 2.0), 15.0), 2))),
    ("G1 low-2 / TP15", lambda s: (s["trig_lo"] - 2.0, s["entry"] + 15.0)),
    ("G2 +-15",         lambda s: (s["entry"] - 15.0, s["entry"] + 15.0)),
    ("G3 +-10",         lambda s: (s["entry"] - 10.0, s["entry"] + 10.0)),
    ("G4 low-2 / TP10", lambda s: (s["trig_lo"] - 2.0, s["entry"] + 10.0)),
    ("G5 SL7 / TP10",   lambda s: (s["entry"] - 7.0,  s["entry"] + 10.0)),
    ("G6 SL7 / TP15",   lambda s: (s["entry"] - 7.0,  s["entry"] + 15.0)),
    ("G7 +-20",         lambda s: (s["entry"] - 20.0, s["entry"] + 20.0)),
    ("G8 +-25",         lambda s: (s["entry"] - 25.0, s["entry"] + 25.0)),
    ("G9 +-30",         lambda s: (s["entry"] - 30.0, s["entry"] + 30.0)),
    ("G10 SL10 / TP15", lambda s: (s["entry"] - 10.0, s["entry"] + 15.0)),
]

# EMA200 filter modes: which TFs' EMA200 the entry close must be above
FMODES = [
    ("none",        lambda s: True),
    ("1m",          lambda s: s["above200"][1]),
    ("trigTF",      lambda s: s["above200"][s["tf"]]),
    ("1m+trigTF",   lambda s: s["above200"][1] and s["above200"][s["tf"]]),
    ("all-TF",      lambda s: all(s["above200"].values())),
]

def simulate(sigs, bars, geom, cap=None):
    # cap = daily circuit breaker in cumulative closed POINTS
    trades = []
    pos = None
    cum = 0.0
    halt = None
    for m in range(T0, TEND + 30):
        if pos is not None:
            k, entry, sl, tp, etm = pos
            b = bars.get((k, m))
            if b is not None:
                lo_, hi_, cl_, _ = b
                if lo_ <= sl:
                    pts = sl - entry
                    trades[-1].extend([m, "SL", pts * 65 - 45])
                    cum += pts; pos = None
                elif hi_ >= tp:
                    pts = tp - entry
                    trades[-1].extend([m, "TP", pts * 65 - 45])
                    cum += pts; pos = None
        if pos is None and m <= TEND - 1:
            if cap is not None and (cum >= cap or cum <= -cap):
                if halt is None:
                    halt = (m, cum)
                continue
            for s in [x for x in sigs if x["minute"] == m]:
                sl, tp = geom(s)
                pos = (s["key"], s["entry"], round(sl, 2), round(tp, 2), m)
                trades.append([m, s["key"], s["kind"], f'{s["tf"]}m', s["lvl"], s["via"],
                               s["entry"], round(sl, 2), round(tp, 2)])
                break
    if pos is not None:
        k, entry, sl, tp, etm = pos
        last = max(mm for (kk, mm) in bars if kk == k)
        hi_, lo_, cl_, _ = bars.get((k, last), (None, None, None, None))
        if cl_ is not None:
            trades[-1].extend([last, "EOD", (cl_ - entry) * 65 - 45])
    return trades, halt

ALL = {}
for day in DAYS_ALL:
    states, today, sigs, bars = collect_signals(day)
    ALL[day] = (sigs, bars)
    print(f"{day}: {len(sigs)} raw signals (ARM_WIN=1, before filter)")

print()
print("=" * 112)
print("FULL GRID — EMA200 filter mode x geometry (net rs)")
print("=" * 112)
hdr = f"{'geometry':<20}" + "".join(f"{fm[0]:>12}" for fm in FMODES)
print(hdr)
results = {}
for gname, fn in GEOMS:
    row = f"{gname:<20}"
    for fmode, ffn in FMODES:
        tot_n = 0; tot_w = 0; net = 0.0
        for day in DAYS_ALL:
            sigs, bars = ALL[day]
            fsigs = [s for s in sigs if ffn(s)]
            trades, _ = simulate(fsigs, bars, fn)
            net += sum(t[11] for t in trades if len(t) > 11)
            tot_n += len(trades)
            tot_w += sum(1 for t in trades if len(t) > 11 and t[11] > 0)
        results[(gname, fmode)] = (tot_n, tot_w, net)
        row += f"{net:>+12.0f}"
    print(row)
print()
print("wins/trades per cell:")
print(hdr)
for gname, fn in GEOMS:
    row = f"{gname:<20}"
    for fmode, _ in FMODES:
        n, w, _ = results[(gname, fmode)]
        row += f"{w:>6}/{n:<5}"
    print(row)

best = sorted(results.items(), key=lambda kv: -kv[1][2])[:5]
print()
print("TOP 5 COMBOS (2-day net):")
for (g, f), (n, w, net) in best:
    print(f"  {g:<20} + {f:<10} net={net:>+8.0f}  ({w}W/{n})")

# ---- FULL TRADE LOGS — NO EMA200 filter (user request), ARM_WIN=1, Sep 2-7 ----
import json
none_fn = next(fn for nm, fn in FMODES if nm == "none")

def dump_trades(gname):
    g = next(fn for nm, fn in GEOMS if nm == gname)
    print()
    print("#" * 112)
    print(f"# {gname} — NO EMA200 FILTER | ARM_WIN=1 | Sep 2/3/4/7 2026")
    print("#" * 112)
    for day in DAYS_ALL:
        sigs, bars = ALL[day]
        fsigs = [s for s in sigs if none_fn(s)]
        trades, _ = simulate(fsigs, bars, g)
        net = sum(t[11] for t in trades if len(t) > 11)
        w = sum(1 for t in trades if len(t) > 11 and t[11] > 0)
        print(f"\n--- {day} | {len(trades)} trades | {w}W | net {net:+,.0f} ---")
        print(f"{'#':<3}{'time':<7}{'key':<10}{'trig':<7}{'TF':<4}{'level':<8}{'entry':>8}{'SL':>8}{'TP':>8}{'exit@':>7}{'res':<5}{'P&L':>8}")
        for i, t in enumerate(trades, 1):
            xt = hm(t[9]) if len(t) > 9 and t[9] else "open"
            res = t[10] if len(t) > 10 else "OPEN"
            pnl = t[11] if len(t) > 11 else 0.0
            print(f"{i:<3}{hm(t[0]):<7}{t[1]:<10}{t[2]:<7}{t[3]:<4}{str(t[4]):<8}{t[6]:>8.2f}{t[7]:>8.2f}{t[8]:>8.2f}{xt:>7}{res:<5}{pnl:>+8.0f}")

for gname in ("G10 SL10 / TP15",):
    dump_trades(gname)

# ---- v6 RAW SIGNAL DUMP (parity reference for the live engine) ----
sigdump = []
for day in DAYS_ALL:
    sigs, bars = ALL[day]
    for s in sigs:
        sigdump.append(dict(day=day, minute=s["minute"], key=s["key"], kind=s["kind"],
                            tf=s["tf"], lvl=s["lvl"], via=s["via"], entry=s["entry"]))
with open(r"C:\Websites\FLATTRADE BOT\lh_f10_v6_signals.json", "w") as f:
    json.dump(sigdump, f, indent=1)
print(f"v6 raw signals dumped: {len(sigdump)} -> lh_f10_v6_signals.json")

# per-day net for every geometry at none-filter
print()
print("=" * 112)
print("PER-DAY NET (no EMA200 filter, ARM_WIN=1)")
print("=" * 112)
print(f"{'geometry':<22}" + "".join(f"{d[-5:]:>10}" for d in DAYS_ALL) + f"{'TOTAL':>10}")
for gname, fn in GEOMS:
    row = f"{gname:<22}"; tot = 0.0
    for day in DAYS_ALL:
        sigs, bars = ALL[day]
        fsigs = [s for s in sigs if none_fn(s)]
        trades, _ = simulate(fsigs, bars, fn)
        net = sum(t[11] for t in trades if len(t) > 11)
        tot += net
        row += f"{net:>+10.0f}"
    print(row + f"{tot:>+10.0f}")

# ---- DAILY CIRCUIT BREAKER SWEEP (user spec) ----
# cumulative closed day P&L in POINTS: touch +/-cap -> out of the market
print()
print("=" * 112)
print("DAILY CAP SWEEP — cumulative closed pts; +/-cap -> stop trading for the day")
print("(no EMA200 filter, ARM_WIN=1; each geometry swept over cap values)")
print("=" * 112)
CAPS = (None, 50, 40, 30, 29, 25, 20, 15, 10)
print(f"{'cap':<6}" + "".join(f"{g[0]:>16}" for g in GEOMS))
capres = {}
for cap in CAPS:
    row = f"{'off' if cap is None else f'{cap}':<6}"
    for gname, fn in GEOMS:
        tot = 0.0; halts = 0
        for day in DAYS_ALL:
            sigs, bars = ALL[day]
            fsigs = [s for s in sigs if none_fn(s)]
            trades, halt = simulate(fsigs, bars, fn, cap=cap)
            tot += sum(t[11] for t in trades if len(t) > 11)
            if halt: halts += 1
        capres[(cap, gname)] = (tot, halts)
        row += f"{tot:>+16.0f}"
    print(row)
print()
print("halted days (of 4) per cap x geometry:")
print(f"{'cap':<6}" + "".join(f"{g[0]:>16}" for g in GEOMS))
for cap in CAPS:
    row = f"{'off' if cap is None else f'{cap}':<6}"
    for gname, _ in GEOMS:
        _, halts = capres[(cap, gname)]
        row += f"{'':>10}{halts}{'':<5}"
    print(row)

best_cap = max(capres.items(), key=lambda kv: kv[1][0])
print(f"\nBEST cap-geometry combo: cap={best_cap[0][0]} + {best_cap[0][1]}  net={best_cap[1][0]:>+,.0f} rs")

# json record dump (separate file — do NOT clobber the parity input JSON)
champ = []
champ30 = []
g4 = next(fn for nm, fn in GEOMS if nm.startswith("G4"))
for day in DAYS_ALL:
    sigs, bars = ALL[day]
    fsigs = [s for s in sigs if none_fn(s)]
    for t in simulate(fsigs, bars, g4)[0]:
        champ.append(dict(day=day, time=hm(t[0]), key=t[1], kind=t[2], tf=t[3],
                          lvl=t[4], entry=t[6], sl=t[7], tp=t[8],
                          exit=hm(t[9]) if len(t) > 9 and t[9] else None,
                          res=t[10] if len(t) > 10 else None,
                          pnl=t[11] if len(t) > 11 else None))
    for t in simulate(fsigs, bars, g4, cap=30)[0]:
        champ30.append(dict(day=day, time=hm(t[0]), key=t[1], kind=t[2], tf=t[3],
                           lvl=t[4], entry=t[6], sl=t[7], tp=t[8],
                           exit=hm(t[9]) if len(t) > 9 and t[9] else None,
                           res=t[10] if len(t) > 10 else None,
                           pnl=t[11] if len(t) > 11 else None))
with open(r"C:\Websites\FLATTRADE BOT\lh_f10_g4_nofilter_geoms.json", "w") as f:
    json.dump(champ, f, indent=1)
with open(r"C:\Websites\FLATTRADE BOT\lh_f10_g4_cap30_geoms.json", "w") as f:
    json.dump(champ30, f, indent=1)
print(f"\nG4 no-filter trades dumped: {len(champ)} -> lh_f10_g4_nofilter_geoms.json")
print(f"G4 cap=30 trades dumped: {len(champ30)} -> lh_f10_g4_cap30_geoms.json")
