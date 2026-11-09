"""ATR16K backtest harness (repo copy — survives temp cleanups).

Data: per-day option CSVs (date,time,symbol,open,high,low,close,volume) and
index CSVs (date/iso, open, high, low, close). Put them in DATA_DIR.

Usage:
    python bt.py matrix          # standing 4 x 2 matrix (deployed rule)
    python bt.py day 2026-11-06  # one day, trade list
    python bt.py range 2026-11-02 2026-11-06

Deployed rule: efirst trigger, twin 0.75, open>=level, correction-leg waves
(4>2>0, THR 5, BUF 1, SKIP 4), SL 1.1*ATR (min 3, max 20), TP 3.5*range,
entry cutoff 15:00, -40 day cap, levels = statics + EMA20/200 3m/5m + session
VWAP + PDVWAP + virgin CPR + ST 3m.
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from flattrade_bot.strategies.atr16k_engine import (  # noqa: E402
    swings, refine, ema_tf, st_map, atr_series, pat_any, twin_candles, THR,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PAT = re.compile(r"^NIFTY(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")
BUF, SKIP = 1.0, 4
WAVE_MODE = "full"
EXTRA_EMAS = False
LOT, FEE = 65, 45

# rule
SL_ATR, TP_MULT, MAX_SL, MIN_RISK = 1.2, 3.5, 20.0, 3.0
ENTRY_END, DAY_CAP = 900, 40.0
TWIN_TOL = 0.75


def load(p):
    df = pd.read_csv(p)
    df["min"] = df["time"].map(lambda t: int(t[:2]) * 60 + int(t[3:5]))
    df = df.drop_duplicates(["symbol", "date", "time"], keep="first")
    return df


def load_index(p):
    ix = pd.read_csv(p)
    col = "date" if "date" in ix.columns else ix.columns[0]
    ix["iso"] = ix[col].astype(str).str.slice(0, 10).map(
        lambda d: f"{d[6:10]}-{d[3:5]}-{d[0:2]}" if d[2] == "-" and d[4] != "-" else d)
    ix["min"] = (ix[col].astype(str).str.slice(11, 13).astype(int) * 60
                 + ix[col].astype(str).str.slice(14, 16).astype(int))
    return ix


class Data:
    def __init__(self, data_dir=DATA_DIR):
        self.opts, self.spot = [], {}
        for f in sorted(os.listdir(data_dir)):
            p = os.path.join(data_dir, f)
            if not f.endswith(".csv"):
                continue
            if "index" in f:
                ix = load_index(p)
                for d, m, c in zip(ix.iso, ix["min"], ix.close):
                    self.spot.setdefault(d, {})[int(m)] = float(c)
            else:
                self.opts.append(load(p))
        self.df = pd.concat(self.opts, ignore_index=True) if self.opts else pd.DataFrame()
        self.days = sorted(self.df.date.unique()) if len(self.df) else []

    def atm(self, day, minute):
        sp = self.spot.get(day, {})
        ks = [k for k in sp if k <= minute]
        return int(round(sp[max(ks)] / 50.0) * 50) if ks else None

    def contracts(self, day, nearest):
        t = self.df[(self.df.date == day) & self.df.symbol.str.startswith(nearest)]
        return sorted(t.symbol.unique())


def wave_ok(a0, a1, a2, a3, a4, a5):
    if WAVE_MODE == "correction":
        return a4 > a2 - BUF and a2 > a0 - BUF
    if WAVE_MODE == "impulse":
        return a5 > a3 - BUF and a3 > a1 - BUF
    return a5 > a3 - BUF and a3 > a1 - BUF and a4 > a2 - BUF and a2 > a0 - BUF


def precompute(data: Data, day, nearest):
    out = []
    for sym in data.contracts(day, nearest):
        m = PAT.match(sym)
        if not m:
            continue
        side, strike = m.group(3), int(m.group(2))
        gt = data.df[(data.df.symbol == sym) & (data.df.date == day)].sort_values("min")
        gt = gt[gt["min"] <= 930].reset_index(drop=True)
        if len(gt) < 40:
            continue
        prev_all = data.df[(data.df.symbol == sym) & (data.df.date < day)]
        if not len(prev_all):
            continue
        pdate = prev_all.date.max()
        gp = prev_all[prev_all.date == pdate].sort_values("min")
        gp = gp[gp["min"] <= 930].reset_index(drop=True)
        if len(gp) < 40:
            continue

        comb = []
        for tag, fr, dd in (("P", gp, pdate), ("T", gt, day)):
            for r in fr.itertuples():
                comb.append({"src": tag, "day": dd, "t": int(r.min), "o": float(r.open),
                             "h": float(r.high), "l": float(r.low), "c": float(r.close),
                             "v": float(r.volume)})
        t0 = next(i for i, b in enumerate(comb) if b["src"] == "T")
        pv = refine(comb, swings(comb, THR))
        NP = len(pv)
        waves = []
        for i0 in range(NP - 5):
            if pv[i0][1] != "L":
                continue
            found = []

            def dfs(pos, need, chosen, budget):
                if len(chosen) == 6:
                    w = [pv[q] for q in chosen]
                    if wave_ok(*(p[2] for p in w)):
                        found.append(list(chosen))
                    return
                for q in range(pos, min(NP, pos + 1 + budget)):
                    if pv[q][1] == need:
                        chosen.append(q)
                        dfs(q + 1, "H" if need == "L" else "L", chosen, budget - (q - pos))
                        chosen.pop()

            dfs(i0 + 1, "H", [i0], SKIP)
            if found:
                waves.append((i0, found[0]))

        ctxs = []
        for i0, idxs in waves:
            lw = idxs[-1]
            ai = next((q for q in range(lw + 1, NP) if pv[q][1] == "L"), None)
            bi = next((q for q in range(ai + 1, NP) if pv[q][1] == "H"), None) if ai is not None else None
            if ai is None or bi is None:
                continue
            bc = pv[bi][4]
            ci = next((q for q in range(bi + 1, NP) if pv[q][1] == "L"), None)
            cc = min((pv[ci][4] if ci is not None else len(comb) - 1) + 1, len(comb) - 1)
            if cc < t0:
                continue
            ctxs.append((i0, idxs, bc, cc))
        nonlap, last = [], -1
        for i0, idxs, bc, cc in ctxs:
            if i0 > last:
                nonlap.append((i0, idxs, bc, cc)); last = idxs[-1]
        if not nonlap:
            continue

        seed = prev_all.sort_values(["date", "min"]).tail(2000)
        lastp = seed[seed.date == pdate]
        bym = dict(zip(lastp["min"].astype(int), lastp.close))
        ph, pl = float(lastp.high.max()), float(lastp.low.min())
        pc = float(bym.get(932, lastp.iloc[-1]["close"]))
        rng = ph - pl
        P = (ph + pl + pc) / 3.0
        BC = (ph + pl) / 2.0
        r2 = lambda x: round(float(x), 2)
        base = {"PDH": ph, "PDL": pl, "CPR": r2(P), "BC": r2(BC), "TC": r2(2 * P - BC),
                "fibR3": r2(P + rng), "fibS3": r2(P - rng),
                "camR3": r2(pc + rng * 1.1 / 4), "camL3": r2(pc - rng * 1.1 / 4)}
        # previous-day VWAP close
        pv_sum = v_sum = 0.0
        for r in lastp.itertuples():
            if 555 <= int(r.min) <= 930 and float(r.high) >= float(r.low) > 0:
                pv_sum += ((float(r.high) + float(r.low) + float(r.close)) / 3.0) * max(float(r.volume), 1.0)
                v_sum += max(float(r.volume), 1.0)
        if v_sum > 0:
            base["PDVWAP"] = r2(pv_sum / v_sum)
        # virgin CPR
        days_ = sorted(prev_all.date.unique())
        if days_:
            fr = prev_all[prev_all.date == days_[0]]
            fh, fl, fc = float(fr.high.max()), float(fr.low.min()), float(fr.close.iloc[-1])
            vcpr, vbc = (fh + fl + fc) / 3.0, (fh + fl) / 2.0
            vir = {"VCPR": r2(vcpr), "VBC": r2(vbc), "VTC": r2(2 * vcpr - vbc)}
            for d_ in days_[1:]:
                dr = prev_all[prev_all.date == d_]
                lo_, hi_ = float(dr.low.min()), float(dr.high.max())
                for nm_ in list(vir):
                    if lo_ <= vir[nm_] <= hi_:
                        del vir[nm_]
            base.update(vir)

        HC = seed.close.tolist() + gt.close.tolist()
        HH = seed.high.tolist() + gt.high.tolist()
        HL = seed.low.tolist() + gt.low.tolist()
        HM = seed["min"].tolist() + gt["min"].tolist()
        k = len(seed)
        per_ti = {"EMA20": ema_tf(HC, HM, 5, 20)[k:], "EMA20_3": ema_tf(HC, HM, 3, 20)[k:],
                  "EMA200": ema_tf(HC, HM, 3, 200)[k:], "EMA200_5": ema_tf(HC, HM, 5, 200)[k:],
                  "ST": st_map(HH, HL, HC, HM)[k:]}
        if EXTRA_EMAS:
            per_ti.update({
                "EMA20_1": ema_tf(HC, HM, 1, 20)[k:], "EMA200_1": ema_tf(HC, HM, 1, 200)[k:],
                "EMA20_2": ema_tf(HC, HM, 2, 20)[k:], "EMA200_2": ema_tf(HC, HM, 2, 200)[k:],
            })
        atrs = atr_series(gt.high.tolist(), gt.low.tolist(), gt.close.tolist())
        pvs = vv = 0.0
        vw = []
        for i in range(len(gt)):
            v = float(gt.iloc[i].volume)
            tp = (gt.iloc[i].high + gt.iloc[i].low + gt.iloc[i].close) / 3.0
            pvs += tp * v; vv += v
            vw.append(pvs / vv if vv else tp)
        per_ti["VWAP"] = vw

        rows = [{"min": int(r.min), "o": float(r.open), "h": float(r.high),
                 "l": float(r.low), "c": float(r.close)} for r in gt.itertuples()]
        cinfo = []
        for i0, idxs, bc, cc in nonlap:
            wbar = comb[pv[idxs[0]][3]]
            atm = data.atm(wbar["day"], wbar["t"])
            if atm is None:
                continue
            if side == "CE" and strike != atm - 100:
                continue
            if side == "PE" and strike != atm + 100:
                continue
            cinfo.append({"bc": bc, "cc": cc, "w_day": wbar["day"], "w_min": wbar["t"], "atm": atm})
        if cinfo:
            out.append({"day": day, "sym": sym, "rows": rows, "t0": t0, "comb": comb,
                        "ctxs": cinfo, "base": base, "per_ti": per_ti, "atrs": atrs})
    return out


def evaluate(ctx, cap=DAY_CAP):
    sigs = []
    for c in ctx:
        rows, comb, t0 = c["rows"], c["comb"], c["t0"]
        for cx in c["ctxs"]:
            for gj in range(max(cx["bc"] + 1, t0), cx["cc"] + 1):
                ti = gj - t0
                if ti < 0 or ti >= len(rows) or rows[ti]["min"] > ENTRY_END:
                    break
                b, p = comb[gj], comb[gj - 1]
                if not pat_any(comb, gj) and not twin_candles(comb, gj, TWIN_TOL):
                    continue
                any_touch = not pat_any(comb, gj)
                lv = dict(c["base"])
                for nm, arr in c["per_ti"].items():
                    if arr[ti] is not None:
                        lv[nm] = arr[ti]
                cd = []
                for nm, v in lv.items():
                    if any_touch:
                        ok = b["l"] <= v <= b["h"]
                    else:
                        ok = b["l"] <= v <= b["h"] and b["o"] >= v
                    if ok:
                        cd.append((nm, v))
                if not cd:
                    continue
                nm, v = max(cd, key=lambda z: z[1])
                entry = b["c"]
                sl = v - SL_ATR * c["atrs"][ti]
                if entry - sl > MAX_SL:
                    sl = entry - MAX_SL
                if entry - sl < MIN_RISK:
                    sl = entry - MIN_RISK
                tp = v + TP_MULT * (b["h"] - b["l"])
                res = None
                for k2 in range(ti + 1, len(rows)):
                    if rows[k2]["min"] > 915:
                        res = (rows[k2]["c"] - entry, rows[k2]["min"], "EOD"); break
                    if rows[k2]["l"] <= sl - 1.0:
                        res = (sl - entry, rows[k2]["min"], "SL"); break
                    if rows[k2]["h"] >= tp:
                        res = (tp - entry, rows[k2]["min"], "TP"); break
                if res is None:
                    res = (rows[-1]["c"] - entry, rows[-1]["min"], "EOD")
                sigs.append({"j": rows[ti]["min"], "sym": c["sym"], "nm": nm, "entry": entry,
                             "sl": round(sl, 2), "risk": round(entry - sl, 2),
                             "exit": res[1], "pts": round(res[0], 2), "why": res[2]})
                break
    kept, last_exit = [], -1
    for s in sorted(sigs, key=lambda x: x["j"]):
        if s["j"] <= last_exit:
            continue
        kept.append(s); last_exit = s["exit"]
    if cap != float("inf"):
        out, cum = [], 0.0
        for s in kept:
            if cum <= -cap:
                break
            out.append(s); cum += s["pts"]
        kept = out
    return kept


def run_days(data, days, nearest_map):
    all_days = []
    for d in days:
        sigs = []
        for c in precompute(data, d, nearest_map.get(d, "")):
            sigs.extend(evaluate([c], cap=float("inf")))
        kept, last_exit = [], -1
        for s in sorted(sigs, key=lambda x: x["j"]):
            if s["j"] <= last_exit:
                continue
            kept.append(s); last_exit = s["exit"]
        if DAY_CAP != float("inf"):
            out, cum = [], 0.0
            for s in kept:
                if cum <= -DAY_CAP:
                    break
                out.append(s); cum += s["pts"]
            kept = out
        all_days.append((d, kept))
    return all_days


def summarise(all_days):
    seq = [t["pts"] for _, k in all_days for t in k]
    tp = round(sum(seq), 2)
    eq = peak = dd = 0.0
    for x in seq:
        eq += x; peak = max(peak, eq); dd = max(dd, peak - eq)
    wr = 100.0 * sum(1 for x in seq if x > 0) / max(1, len(seq))
    ann = tp / (len(all_days) / 252.0) if all_days else 0
    cal = ann / dd if dd > 0 else None
    return {"trades": len(seq), "wr": round(wr, 1), "pts": tp,
            "rs": round(tp * LOT - len(seq) * FEE), "dd": round(dd, 2),
            "calmar": round(cal, 2) if cal else None}


