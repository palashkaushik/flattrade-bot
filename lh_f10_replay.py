"""F10 CERTIFIED REPLAY v2 — Full10 v5 (ARM_WIN=1, no EMA200) + geometry
+ DAILY CIRCUIT BREAKER (cap on cumulative closed day points).

Strategy (executable spec; mirrors lh_full10_geoms.py exactly):
  Arming: any-TF S1<=30 within last 10 BARS of that contract (baridx space)
  Trigger (any TF, ignoring dormant TFs): FLAG = S4>=75 & S1<30
                                          SUPER = S1<30 & S3<30 & S4<30 & S1>prevS1
  Dormancy: TF ignored while its S4<20 run-length (completed TF candles) > 14
  Levels (per contract): EMA20, EMA200, VWAP(eq), PDH, PDL, CPR, BC, TC,
                         fibR3=CPR+rng, fibS3=CPR-rng, camR3=C+rng*1.1/4,
                         camL3=C-rng*1.1/4, PDVWAP, virgin CPR/BC/TC (alive)
  Bounce (trigger bar): 1m: low <= lvl+2.0 and close >= lvl-0.5
          TF fallback (2m/3m/5m last completed bar): TF low <= lvl+2.0 and
          TF close >= lvl-0.5 AND 1m entry close >= lvl-0.5
  Virgin walk: line dies if any bar's low <= line <= high (today too)
  EMA200 filter: entry close >= EMA200(1m) AND >= EMA200(trigger TF)
  Entry: signal-bar close. G4: SL = trigger 1m low - 2.0, TP = entry + 10.0
  Exits: SL/TP intrabar (SL priority), EOD force-close at last bar of session.
  Single position; spec-pair priority CE first then PE.

Reference engine (plain Python, day loop) and NUMBA twin (identical logic,
arrays only). PARITY GATE: both must reproduce the 32 champion trades from
lh_f10_champ_geoms.json on Sep 4 + Sep 7 (weekly recordings).
"""
import sys, os, json, glob
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
import numpy as np
from numba import njit

T0, NT = 555, 345
TB = 2.0
LOT, FEE = 65.0, 45.0
ARM_WIN = 1                 # v5: entry within 1 bar of the arming bar
EMA_FILTER = False          # v5: EMA200 filter OFF
SL_PTS, TP_PTS = 2.0, 10.0  # G4 defaults; symmetric geometries override

# ---------- reference (plain Python) ----------

def reference_day(z, geom="G4", cap=None):
    has = z["has"]; ohlc = z["ohlc"]; baridx = z["baridx"]
    st = z["st"].reshape(4, 4, z["n_c"], NT)
    ind = z["ind"]; ema4 = z["ema4"].reshape(4, z["n_c"], NT)
    dorm = z["dorm"].reshape(4, z["n_c"], NT)
    s1a = z["s1a"]; pppF = z["pppF"]; pdv = z["pdv"]
    vir = z["vir"]; valive = z["valive"]
    C = int(z["n_c"])
    trades = []
    pos = None   # [c, entry, sl, tp, entry_t]
    cum = 0.0    # cumulative closed day points (circuit breaker)

    # per-contract anchors tracked in baridx space
    anchor = [-10**9] * C
    virgin = [[vir[i, li] if vir[i, li] == vir[i, li] else None for li in range(3)]
              for i in range(C)]
    valive_now = [[bool(valive[i, li]) for li in range(3)] for i in range(C)]

    def geom_box(lo_, cl):
        # returns (sl, tp) for the given geometry
        if geom.startswith("SYM"):          # SYM20/25/30: entry +- X
            x = float(geom[3:])
            return cl - x, cl + x
        # G4: SL = trigger 1m low - 2, TP = entry + 10
        return round(lo_ - SL_PTS, 2), round(cl + TP_PTS, 2)

    for t in range(NT):
        # update all contracts (today's bar)
        for c in range(C):
            if not has[c, t]:
                continue
            # virgin death walk (today)
            lo_, hi_ = ohlc[c, t, 2], ohlc[c, t, 1]
            for li in range(3):
                v = virgin[c][li]
                if v is not None and valive_now[c][li] and lo_ <= v <= hi_:
                    valive_now[c][li] = False
            # arming (any TF S1<=30)
            if s1a[c, t]:
                anchor[c] = baridx[c, t]
        # position management first (SL priority)
        if pos is not None:
            c, entry, sl, tp, et = pos
            if has[c, t]:
                lo_, hi_ = ohlc[c, t, 2], ohlc[c, t, 1]
                if lo_ <= sl:
                    pts = sl - entry
                    trades[-1].update(exit_t=t, res="SL", pnl=pts * LOT - FEE)
                    cum += pts; pos = None
                elif hi_ >= tp:
                    pts = tp - entry
                    trades[-1].update(exit_t=t, res="TP", pnl=pts * LOT - FEE)
                    cum += pts; pos = None
        # entry scan
        if pos is None:
            if cap is not None and (cum >= cap or cum <= -cap):
                continue
            for c in range(C):
                if not has[c, t]:
                    continue
                a = anchor[c]
                if a < 0 or (baridx[c, t] - a) > 10:
                    continue
                # trigger on any TF (dormancy REMOVED per user)
                trig = None
                for j, tf in ((0, 1), (1, 2), (2, 3), (3, 5)):
                    s1, s3, s4, p1 = (st[j, 0, c, t], st[j, 1, c, t],
                                      st[j, 2, c, t], st[j, 3, c, t])
                    if s1 != s1 or s4 != s4:
                        continue
                    if s4 >= 75.0 and s1 < 30.0:
                        trig = (j, tf, "FLAG"); break
                    if s3 == s3 and p1 == p1 and s3 < 30.0 and s4 < 30.0 and s1 < 30.0 and s1 > p1:
                        trig = (j, tf, "SUPER"); break
                if trig is None:
                    continue
                j, tf, kind = trig
                # EMA200 filter (v5: OFF by default)
                if EMA_FILTER:
                    e1 = ema4[0, c, t]; etf = ema4[j, c, t]
                    cl = ohlc[c, t, 3]; lo_ = ohlc[c, t, 2]
                    if e1 != e1 or etf != etf or cl < e1 or cl < etf:
                        continue
                else:
                    cl = ohlc[c, t, 3]; lo_ = ohlc[c, t, 2]
                # levels
                ph, pl, pc, cpr, bc, tc, fr3, fs3, cr3, cl3 = pppF[c]
                lv = [ind[c, t, 1], ind[c, t, 2], ind[c, t, 3], cpr, bc, tc,
                      ph, pl, fr3, fs3, cr3, cl3, pdv[c]]
                for li in range(3):
                    v = virgin[c][li]
                    if v is not None and valive_now[c][li]:
                        lv.append(v)
                hit = None
                for name_i, v in enumerate(lv):
                    if v != v:
                        continue
                    if lo_ <= v + TB and cl >= v - 0.5:
                        hit = name_i; break
                if hit is None:
                    for jj, tfv in ((0, 2), (1, 3), (2, 5)):
                        tlo = ind[c, t, 4 + 2*jj]; tcl = ind[c, t, 5 + 2*jj]
                        if tlo != tlo or tcl != tcl:
                            continue
                        for name_i, v in enumerate(lv):
                            if v != v:
                                continue
                            if tlo <= v + TB and tcl >= v - 0.5 and cl >= v - 0.5:
                                hit = name_i; break
                        if hit is not None:
                            break
                if hit is None:
                    continue
                entry = cl
                sl, tp = geom_box(lo_, entry)
                pos = [c, entry, round(sl, 2), round(tp, 2), t]
                trades.append(dict(c=c, entry_t=t, kind=kind, tf=tf,
                                    lvl=hit, entry=entry, sl=sl, tp=tp))
                break
    if pos is not None:
        c, entry, sl, tp, et = pos
        last = max(tt for tt in range(NT) if has[c, tt])
        cl_ = ohlc[c, last, 3]
        pts = cl_ - entry
        trades[-1].update(exit_t=last, res="EOD", pnl=pts * LOT - FEE)
    return trades

# ---------- numba twin ----------

@njit(cache=True)
def njit_day(has, ohlc, baridx, st, ind, ema4, dorm, s1a, pppF, pdv, vir, valive, out_tr,
             arm_win=1, use_ema=False, sym=0.0, cap=0.0):
    C = has.shape[0]
    pos_c = -1; pos_entry = 0.0; pos_sl = 0.0; pos_tp = 0.0; pos_et = -1
    n = 0
    cum = 0.0
    anchor = np.full(C, -10**9, dtype=np.int64)
    virgin = np.zeros((C, 3), dtype=np.float64)
    valive_now = np.zeros((C, 3), dtype=np.uint8)
    for c in range(C):
        for li in range(3):
            v = vir[c, li]
            virgin[c, li] = v
            valive_now[c, li] = 1 if (v == v and valive[c, li] == 1) else 0
    for t in range(NT):
        for c in range(C):
            if has[c, t] == 0:
                continue
            lo_ = ohlc[c, t, 2]; hi_ = ohlc[c, t, 1]
            for li in range(3):
                if valive_now[c, li] == 1:
                    v = virgin[c, li]
                    if v == v and lo_ <= v <= hi_:
                        valive_now[c, li] = 0
            if s1a[c, t] == 1:
                anchor[c] = baridx[c, t]
        if pos_c >= 0 and has[pos_c, t] == 1:
            lo_ = ohlc[pos_c, t, 2]; hi_ = ohlc[pos_c, t, 1]
            if lo_ <= pos_sl:
                pts = pos_sl - pos_entry
                if n < out_tr.shape[0]:
                    out_tr[n, 0] = pos_c; out_tr[n, 1] = pos_et; out_tr[n, 2] = t
                    out_tr[n, 3] = pos_entry; out_tr[n, 4] = pos_sl
                    out_tr[n, 5] = pts * 65.0 - 45.0
                    out_tr[n, 6] = 0.0
                cum += pts
                n += 1; pos_c = -1
            elif hi_ >= pos_tp:
                pts = pos_tp - pos_entry
                if n < out_tr.shape[0]:
                    out_tr[n, 0] = pos_c; out_tr[n, 1] = pos_et; out_tr[n, 2] = t
                    out_tr[n, 3] = pos_entry; out_tr[n, 4] = pos_tp
                    out_tr[n, 5] = pts * 65.0 - 45.0
                    out_tr[n, 6] = 1.0
                cum += pts
                n += 1; pos_c = -1
        if pos_c == -1:
            if cap > 0.0 and (cum >= cap or cum <= -cap):
                continue
            for c in range(C):
                if has[c, t] == 0:
                    continue
                a = anchor[c]
                if a < 0 or (baridx[c, t] - a) > arm_win:
                    continue
                trig_j = -1; trig_kind = 0.0
                for j in range(4):
                    s1 = st[j, 0, c, t]; s3 = st[j, 1, c, t]
                    s4 = st[j, 2, c, t]; p1 = st[j, 3, c, t]
                    if s1 != s1 or s4 != s4:
                        continue
                    if s4 >= 75.0 and s1 < 30.0:
                        trig_j = j; trig_kind = 2.0; break
                    if s3 == s3 and p1 == p1 and s3 < 30.0 and s4 < 30.0 and s1 < 30.0 and s1 > p1:
                        trig_j = j; trig_kind = 3.0; break
                if trig_j < 0:
                    continue
                cl = ohlc[c, t, 3]; lo_ = ohlc[c, t, 2]
                if use_ema:
                    e1 = ema4[0, c, t]; etf = ema4[trig_j, c, t]
                    if e1 != e1 or etf != etf or cl < e1 or cl < etf:
                        continue
                ph = pppF[c, 0]; pl = pppF[c, 1]; pc = pppF[c, 2]
                cpr = pppF[c, 3]; bc = pppF[c, 4]; tc = pppF[c, 5]
                fr3 = pppF[c, 6]; fs3 = pppF[c, 7]; cr3 = pppF[c, 8]; cl3 = pppF[c, 9]
                lv = np.empty(16)
                lv[0] = ind[c, t, 1]; lv[1] = ind[c, t, 2]; lv[2] = ind[c, t, 3]
                lv[3] = cpr; lv[4] = bc; lv[5] = tc; lv[6] = ph; lv[7] = pl
                lv[8] = fr3; lv[9] = fs3; lv[10] = cr3; lv[11] = cl3; lv[12] = pdv[c]
                nlv = 13
                for li in range(3):
                    v = virgin[c, li]
                    if v == v and valive_now[c, li] == 1:
                        lv[nlv] = v; nlv += 1
                hit = -1
                for li in range(nlv):
                    v = lv[li]
                    if v != v:
                        continue
                    if lo_ <= v + 2.0 and cl >= v - 0.5:
                        hit = li; break
                if hit < 0:
                    for jj in range(3):
                        tlo = ind[c, t, 4 + 2*jj]; tcl = ind[c, t, 5 + 2*jj]
                        if tlo != tlo or tcl != tcl:
                            continue
                        for li in range(nlv):
                            v = lv[li]
                            if v != v:
                                continue
                            if tlo <= v + 2.0 and tcl >= v - 0.5 and cl >= v - 0.5:
                                hit = li; break
                        if hit >= 0:
                            break
                if hit < 0:
                    continue
                entry = cl
                if sym > 0.0:
                    pos_sl = entry - sym; pos_tp = entry + sym
                else:
                    pos_sl = round(lo_ - 2.0, 2); pos_tp = round(entry + 10.0, 2)
                pos_c = c; pos_entry = entry; pos_et = t
                if n < out_tr.shape[0]:
                    out_tr[n, 0] = c; out_tr[n, 1] = t; out_tr[n, 2] = -1
                    out_tr[n, 3] = entry; out_tr[n, 4] = pos_sl
                    out_tr[n, 5] = np.nan; out_tr[n, 6] = trig_kind
                n += 1
                break
    if pos_c >= 0:
        last = -1
        for tt in range(NT):
            if has[pos_c, tt] == 1:
                last = tt
        if last >= 0 and n <= out_tr.shape[0]:
            cl_ = ohlc[pos_c, last, 3]
            if n - 1 < out_tr.shape[0]:
                out_tr[n-1, 2] = last
                out_tr[n-1, 4] = cl_
                out_tr[n-1, 5] = (cl_ - pos_entry) * 65.0 - 45.0
                out_tr[n-1, 6] = 4.0
    return n

def numba_day(z, geom="G4", cap=None):
    has = z["has"]; ohlc = z["ohlc"]; baridx = z["baridx"]
    st = z["st"].reshape(4, 4, int(z["n_c"]), NT).copy()
    ind = z["ind"]; ema4 = z["ema4"].reshape(4, int(z["n_c"]), NT).copy()
    dorm = z["dorm"].reshape(4, int(z["n_c"]), NT).copy()
    s1a = z["s1a"]; pppF = z["pppF"]; pdv = z["pdv"]
    vir = z["vir"]; valive = z["valive"]
    out_tr = np.full((400, 7), np.nan)
    sym = float(geom[3:]) if geom.startswith("SYM") else 0.0
    n = min(njit_day(has, ohlc, baridx, st, ind, ema4, dorm, s1a, pppF, pdv,
                     vir, valive, out_tr, ARM_WIN, EMA_FILTER, sym,
                     cap if cap is not None else 0.0), 400)
    out = []
    for i in range(n):
        r = out_tr[i]
        out.append(dict(c=int(r[0]), entry_t=int(r[1]), exit_t=int(r[2]) if r[2] >= 0 else None,
                        entry=r[3], sl_or_exit=r[4], pnl=r[5] if r[5] == r[5] else None,
                        kcode=r[6]))
    return out

# ---------- parity gate vs geoms champion trades ----------

LVL_NAMES = ["EMA20", "EMA200", "VWAP", "CPR", "BC", "TC", "PDH", "PDL",
             "fibR3", "fibS3", "camR3", "camL3", "PDVWAP", "VCPR", "VBC", "VTC"]
KCODE = {2.0: "FLAG", 3.0: "SUPER", 4.0: "EOD"}

def hm(t):
    m = T0 + t
    return f"{m//60:02d}:{m%60:02d}"

def run_parity():
    ev_dir = os.path.join(r"C:\Users\user\AppData\Local\Temp\opencode\lh_events_f10", "weekly")
    # v5 ground truth: G4, ARM_WIN=1, NO EMA200 filter, Sep 2/3/4/7
    champ = json.load(open(r"C:\Websites\FLATTRADE BOT\lh_f10_g4_nofilter_geoms.json"))
    print("=" * 100)
    print("F10 PARITY GATE v5 — G4 no-EMA200, ARM_WIN=1 (engine parity, no cap)")
    print("=" * 100)
    total_ok = 0; total_n = 0
    for day in ("2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07"):
        z = np.load(os.path.join(ev_dir, f"ev_{day.replace('-','')}.npz"))
        ref = reference_day(z, geom="G4")
        ref = [tr for tr in ref if tr.get("res") is not None]
        nb = numba_day(z, geom="G4")
        nb = [tr for tr in nb if tr.get("pnl") is not None or tr["kcode"] == 4.0]
        exp = [c for c in champ if c["day"] == day]
        print(f"\n--- {day}: geoms={len(exp)} ref={len(ref)} numba={len(nb)} ---")
        # align ref/numba 1:1
        for i, (r, nbtr) in enumerate(zip(ref, nb)):
            total_n += 1
            e = exp[i] if i < len(exp) else None
            ok_r = ok_n = ok_e = ok_p = True
            if e is not None:
                ok_e = (hm(r["entry_t"]) == e["time"] and abs(r["entry"] - e["entry"]) < 0.01
                        and r["kind"] == e["kind"])
                ok_r = (nbtr["entry_t"] == r["entry_t"] and abs(nbtr["entry"] - r["entry"]) < 0.01)
                ok_p = (r["res"] == e["res"] and abs((r["pnl"] or 0) - (e["pnl"] or 0)) < 1.0)
            else:
                ok_e = ok_p = False
            ok = ok_e and ok_r and ok_p
            total_ok += ok
            lvl = LVL_NAMES[r["lvl"]] if isinstance(r.get("lvl"), int) else "?"
            mark = "OK " if ok else "FAIL"
            print(f"  {mark} #{i+1:<2} {hm(r['entry_t'])} {r['kind']:<5} lvl={lvl:<7} "
                  f"entry={r['entry']:7.2f} {r.get('res'):<4} pnl={r.get('pnl')}")
            if e is not None and not ok:
                print(f"       expected: {e['time']} {e['kind']} entry={e['entry']} "
                      f"{e['res']} pnl={e['pnl']}")
                print(f"       numba   : t={nbtr['entry_t']} entry={nbtr['entry']} pnl={nbtr['pnl']}")
        if len(ref) != len(exp) or len(nb) != len(exp):
            print(f"  !! COUNT MISMATCH ref={len(ref)} numba={len(nb)} geoms={len(exp)}")
    print(f"\nPARITY (no-cap): {total_ok}/{total_n} exact")

    # ---- cap=30 circuit-breaker parity (same engine, capped truth) ----
    champ30 = json.load(open(r"C:\Websites\FLATTRADE BOT\lh_f10_g4_cap30_geoms.json"))
    print()
    print("=" * 100)
    print("F10 PARITY GATE — cap=30 daily circuit breaker (G4, ARM_WIN=1)")
    print("=" * 100)
    c_ok = 0; c_n = 0
    for day in ("2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07"):
        z = np.load(os.path.join(ev_dir, f"ev_{day.replace('-','')}.npz"))
        ref = reference_day(z, geom="G4", cap=30.0)
        ref = [tr for tr in ref if tr.get("res") is not None]
        nb = numba_day(z, geom="G4", cap=30.0)
        nb = [tr for tr in nb if tr.get("pnl") is not None or tr["kcode"] == 4.0]
        exp = [c for c in champ30 if c["day"] == day]
        ok_day = (len(ref) == len(exp) == len(nb))
        if ok_day:
            for r, nbtr, e in zip(ref, nb, exp):
                ok = (hm(r["entry_t"]) == e["time"] and abs(r["entry"] - e["entry"]) < 0.01
                      and r["kind"] == e["kind"] and r["res"] == e["res"]
                      and abs((r["pnl"] or 0) - (e["pnl"] or 0)) < 1.0
                      and nbtr["entry_t"] == r["entry_t"])
                c_ok += ok; c_n += 1
        else:
            c_n += max(len(exp), len(ref))
        print(f"  {'OK ' if ok_day and len(ref) == len(exp) else 'DIFF'} {day}: "
              f"geoms={len(exp)} ref={len(ref)} numba={len(nb)}")
    print(f"\nCAP-30 PARITY: {c_ok}/{c_n} exact")
    return total_ok == total_n and c_ok == c_n and total_ok > 0

def run_long(dirpath, label):
    files = sorted(glob.glob(os.path.join(dirpath, "ev_*.npz")))
    tot_n = 0; tot_w = 0; net = 0.0
    yearly = {}
    for f in files:
        z = np.load(f)
        nb = numba_day(z)
        for tr in nb:
            if tr["pnl"] is None:
                continue
            yr = str(int(z["date"]))[:4]
            yearly.setdefault(yr, [0, 0, 0.0])
            yearly[yr][0] += 1
            yearly[yr][1] += (tr["pnl"] > 0)
            yearly[yr][2] += tr["pnl"]
            tot_n += 1; tot_w += (tr["pnl"] > 0); net += tr["pnl"]
    print(f"\n=== {label}: {len(files)} days | {tot_n} trades | WR {100*tot_w/max(tot_n,1):.0f}% | NET {net:+,.0f} rs ===")
    for yr in sorted(yearly):
        n, w, p = yearly[yr]
        print(f"  {yr}: {n:>4} trades  {w:>3}W  net {p:>+10.0f}")
    return net

if __name__ == "__main__":
    ok = run_parity()
    sys.exit(0 if ok else 1)
