"""ATR stop sweep: stop placed below ENTRY by k x ATR (not below the level).

Grid: triggers ob/obonly x ATR factors 0.5..2.0, TP 5.0, max stop 20, min 3,
day cap -40. Baseline (level-anchored deployed rule) included for comparison.

Modes:
  python sl_entry_sweep.py short   -> 2026-11-04 .. 2026-11-09 from data/ CSVs
  python sl_entry_sweep.py         -> six-year cache, day-level parallel
"""
import json
import os
import statistics
import sys
import threading
from datetime import datetime
from multiprocessing import Pool

import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sixyear_par as S  # noqa: E402
import bt  # noqa: E402

CHECK = os.path.join(HERE, "slentry_days.jsonl")
WORKERS = int(os.environ.get("SY_WORKERS", "10"))
CAP = 40.0
QTY, FEE = 65, 45
YEARS = 6.33

FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
TP_FACTORS = [3.0, 4.0, 5.0, 6.0, 8.0]
CONFIGS = []
for trig in ("ob", "obonly"):
    for k in FACTORS:
        for t in TP_FACTORS:
            CONFIGS.append((f"{trig} atr {k:g}/{t:g}", trig, k, t, 20.0, True, True))
CONFIGS.append(("ob level 1.0/5 deployed", "ob", 1.0, 5.0, 20.0, False, False))


def take(cfg, ctx):
    _n, trig, sl, tp, mx, from_entry, tp_from_entry = cfg
    bt.TRIGGER_MODE = trig
    bt.SL_ATR, bt.TP_MULT, bt.MAX_SL = sl, tp, mx
    bt.MIN_RISK, bt.ENTRY_END = 3.0, 900
    bt.SL_FROM_ENTRY = from_entry
    bt.TP_FROM_ENTRY = tp_from_entry
    sigs = []
    for c in ctx:
        sigs.extend(bt.evaluate([c], cap=float("inf")))
    kept, last = [], -1
    for s in sorted(sigs, key=lambda x: x["j"]):
        if s["j"] <= last:
            continue
        kept.append(s)
        last = s["exit"]
    out, cum = [], 0.0
    for s in kept:
        if cum <= -CAP:
            break
        out.append(s)
        cum += s["pts"]
    return out


def stats(trades):
    if not trades:
        return None
    pts = [t["pts"] for t in trades]
    sl = [t["entry"] - t["sl"] for t in trades]
    wins = [p for p in pts if p > 0]
    losses = [p for p in pts if p <= 0]
    eq = peak = dd = 0.0
    for p in pts:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    rs = sum(pts) * QTY - len(pts) * FEE
    rr = (statistics.mean(wins) / abs(statistics.mean(losses))) if wins and losses else 0.0
    return {"n": len(pts), "wr": 100 * len(wins) / len(pts), "pts": sum(pts), "rs": rs,
            "dd": dd, "cal": (sum(pts) / YEARS) / dd if dd else 0.0,
            "be": rs / (2 * len(pts) * QTY), "sl": statistics.mean(sl), "rr": rr}


def print_table(title, rows):
    print(f"\n=== {title}")
    print(f"{'config':30s} {'tr':>5s} {'WR':>6s} {'pts':>9s} {'Rs':>11s} {'DD':>8s} "
          f"{'Cal':>5s} {'BE':>5s} {'avgSL':>6s} {'RR':>5s}")
    for name, st in rows:
        if st is None:
            print(f"{name:30s}  no trades")
            continue
        print(f"{name:30s} {st['n']:5d} {st['wr']:6.1f} {st['pts']:9.2f} {st['rs']:+11,.0f} "
              f"{st['dd']:8.2f} {st['cal']:5.2f} {st['be']:5.2f} {st['sl']:6.2f} {st['rr']:5.2f}")


def nearest_map(data):
    import re
    exp = re.compile(r"^NIFTY(\d{2}[A-Z]{3}\d{2})")
    exps = set()
    for s in data.df.symbol.unique():
        m = exp.match(s)
        if m:
            try:
                exps.add(datetime.strptime(m.group(1), "%d%b%y").date())
            except ValueError:
                pass
    out = {}
    for d in data.days:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        later = sorted(e for e in exps if e >= dt)
        e = later[0] if later else (sorted(exps)[-1] if exps else None)
        out[d] = f"NIFTY{e:%d%b%y}".upper() if e else ""
    return out


def short():
    data = bt.Data()
    days = [d for d in data.days if "2026-11-04" <= d <= "2026-11-09"]
    nm = nearest_map(data)
    ctxs = {d: bt.precompute(data, d, nm.get(d, "")) for d in days}
    rows = []
    for cfg in CONFIGS:
        allt = []
        for d in days:
            allt.extend(take(cfg, ctxs[d]))
        rows.append((cfg[0], stats(allt)))
    rows.sort(key=lambda r: -(r[1]["cal"] if r[1] else -1))
    print_table(f"short window {days[0]} .. {days[-1]} ({len(days)} sessions)", rows)


def worker(task):
    day, prior = task
    d0 = prior[0] if prior else day
    tbl = pq.read_table(S.CACHE, filters=[("date", ">=", d0), ("date", "<=", day)])
    df = tbl.to_pandas(self_destruct=True)
    if df.empty:
        return day, {}
    df["date"] = pd.Categorical(df["date"], categories=sorted(df["date"].unique()), ordered=True)
    df["symbol"] = df["symbol"].astype("category")
    if day not in set(df.date.unique()):
        return day, {}
    data = S.FrameData(df, S.SPOT)
    ddf = df[df.date == day]
    exps = {S.expiry(s) for s in ddf.symbol.unique()}
    exps.discard(None)
    dd = datetime.strptime(day, "%Y-%m-%d").date()
    later = sorted(e for e in exps if e >= dd)
    nearest = f"NIFTY{later[0]:%d%b%y}".upper() if later else ""
    ctx = bt.precompute(data, day, nearest) if nearest else []
    res = {}
    for cfg in CONFIGS:
        res[cfg[0]] = [(s["j"], s["pts"], s["entry"], s["sl"]) for s in take(cfg, ctx)]
    return day, res


def load():
    done = {}
    if os.path.exists(CHECK):
        with open(CHECK, encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                done[o["day"]] = o["books"]
    return done


def long():
    days = S.all_days()
    done = load()
    todo = [(d, days[max(0, i - S.BUF_DAYS):i]) for i, d in enumerate(days) if d not in done]
    print(f"{len(done)} days done, {len(todo)} to run, {len(CONFIGS)} configs", flush=True)
    job = S.cap_cpu(S.CPU_CAP_PCT)
    stop_evt = threading.Event()
    wd = None
    if job is None:
        wd = threading.Thread(target=S._watchdog, args=(stop_evt,), daemon=True)
        wd.start()
    t0 = datetime.now()
    try:
        with Pool(WORKERS) as pool:
            for day, books in pool.imap_unordered(worker, todo, chunksize=1):
                done[day] = books
                with open(CHECK, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"day": day, "books": books}) + "\n")
    finally:
        stop_evt.set()
        if wd is not None:
            wd.join(timeout=3)
    print(f"done in {(datetime.now()-t0).seconds}s", flush=True)
    rows = []
    for cfg in CONFIGS:
        allt = [{"pts": p, "entry": e, "sl": sl}
                for d in sorted(done) for (j, p, e, sl) in done[d].get(cfg[0], [])]
        rows.append((cfg[0], stats(allt)))
    rows.sort(key=lambda r: -(r[1]["cal"] if r[1] else -1))
    print_table(f"six-year {len(done)} sessions", rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "short":
        short()
    else:
        long()
