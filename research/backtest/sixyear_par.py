"""Six-year run, day-level parallelism (10 workers), congruent by construction.

Each task rebuilds the exact 13-day window (12 priors + day) from the cache and
runs the same bt.precompute / bt.evaluate pipeline as the sequential runner, so
no buffer state crosses a task boundary.
"""
import os
import re
import sys
from datetime import datetime
from multiprocessing import Pool
import threading
try:
    import psutil
except ImportError:
    psutil = None

CPU_HIGH, CPU_LOW = 85.0, 70.0
CPU_CAP_PCT = float(os.environ.get("SY_CPU_CAP", "80"))


def cap_cpu(percent=80.0):
    """Hard-cap this process tree with a Windows Job Object (no oscillation)."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE

        class INFO(ctypes.Structure):
            _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = INFO()
        info.ControlFlags = 0x1 | 0x4          # ENABLE | HARD_CAP
        info.CpuRate = int(percent * 100)      # percent x 100 (80% -> 8000)
        if not k32.SetInformationJobObject(job, 15, ctypes.byref(info), ctypes.sizeof(info)):
            return None
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            return None
        return job
    except Exception:
        return None


def _watchdog(stop_evt):
    """Keep total CPU under CPU_HIGH by suspending the worker children."""
    if psutil is None:
        return
    me = psutil.Process()
    suspended = False
    while not stop_evt.is_set():
        cpu = psutil.cpu_percent(interval=1.0)
        kids = me.children(recursive=True)
        if cpu > CPU_HIGH and not suspended:
            for c in kids:
                try:
                    c.suspend()
                except Exception:
                    pass
            suspended = True
        elif cpu < CPU_LOW and suspended:
            for c in kids:
                try:
                    c.resume()
                except Exception:
                    pass
            suspended = False
    for c in me.children(recursive=True):
        try:
            c.resume()
        except Exception:
            pass

import json
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt  # noqa: E402

ROOT = r"C:\Users\user\Desktop\nifty50 data\nifty_options"
INDEX = r"C:\Users\user\Desktop\nifty50 data\index\NIFTY 50_minute.csv"
CACHE = r"C:\Websites\FLATTRADE BOT\research\backtest\sixyear_cache.parquet"
CHECK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sixyear_days.jsonl")
START, END = "2020-01-01", "2026-04-30"
BUF_DAYS = 12
WORKERS = int(os.environ.get("SY_WORKERS", "10"))

CONFIGS = [
    ("cal SL0.9 TP8", 0.9, 8.0, 20),
    ("wr  SL0.9 TP4 nocap", 0.9, 4.0, 99),
    ("dep SL1.0 TP5", 1.0, 5.0, 20),
    ("SL1.2 TP3.6", 1.2, 3.6, 20),
]

bt.WAVE_MODE = "full"
bt.TRIGGER_MODE = "ob"
bt.EXTRA_EMAS = False

_ix = pd.read_csv(INDEX, usecols=["date", "close"])
_ix["iso"] = _ix["date"].str.slice(0, 10)
_ix["min"] = _ix["date"].str.slice(11, 13).astype(int) * 60 + _ix["date"].str.slice(14, 16).astype(int)
SPOT = {}
for d, m, c in zip(_ix["iso"], _ix["min"], _ix["close"]):
    SPOT.setdefault(d, {})[int(m)] = float(c)
del _ix


def parse_day(p):
    m = re.search(r"nifty_options_(\d{2})_(\d{2})_(\d{4})\.csv$", p)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None


def all_days():
    days = set()
    for d, _, fs in os.walk(ROOT):
        for f in fs:
            if f.startswith("nifty_options_") and f.endswith(".csv"):
                day = parse_day(f)
                if day and START <= day <= END:
                    days.add(day)
    return sorted(days)


def expiry(sym):
    m = re.match(r"^NIFTY(\d{2}[A-Z]{3}\d{2})", sym)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%d%b%y").date()
    except ValueError:
        return None


class FrameData:
    def __init__(self, df, spot):
        self.df = df
        self.spot = spot

    def atm(self, day, minute):
        sp = self.spot.get(day, {})
        ks = [k for k in sp if k <= minute]
        return int(round(sp[max(ks)] / 50.0) * 50) if ks else None

    def contracts(self, day, nearest):
        t = self.df[(self.df.date == day) & self.df.symbol.str.startswith(nearest)]
        return sorted(t.symbol.unique())


def worker(task):
    day, prior = task
    d0 = prior[0] if prior else day
    tbl = pq.read_table(CACHE, filters=[("date", ">=", d0), ("date", "<=", day)])
    df = tbl.to_pandas(self_destruct=True)
    df["date"] = pd.Categorical(df["date"], categories=sorted(df["date"].unique()), ordered=True)
    df["symbol"] = df["symbol"].astype("category")
    if df.empty or day not in set(df.date.unique()):
        return day, {}
    data = FrameData(df, SPOT)
    ddf = df[df.date == day]
    exps = {expiry(s) for s in ddf.symbol.unique()}; exps.discard(None)
    dd = datetime.strptime(day, "%Y-%m-%d").date()
    later = sorted(e for e in exps if e >= dd)
    nearest = f"NIFTY{later[0]:%d%b%y}".upper() if later else ""
    ctx = bt.precompute(data, day, nearest) if nearest else []
    res = {}
    for name, sl, tp, mx in CONFIGS:
        bt.SL_ATR, bt.TP_MULT, bt.MAX_SL = sl, tp, mx
        bt.MIN_RISK, bt.ENTRY_END, bt.DAY_CAP = 3, 900, 40
        sigs = []
        for c in ctx:
            sigs.extend(bt.evaluate([c], cap=float("inf")))
        kept, last = [], -1
        for s in sorted(sigs, key=lambda x: x["j"]):
            if s["j"] <= last:
                continue
            kept.append(s); last = s["exit"]
        capped, cum = [], 0.0
        for s in kept:
            if cum <= -40:
                break
            capped.append(s); cum += s["pts"]
        res[name] = [(s["j"], s["pts"]) for s in capped]
    return day, res


def load_check():
    done = {}
    if os.path.exists(CHECK):
        with open(CHECK, encoding="utf-8") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                    done[o["day"]] = {k: [tuple(x) for x in v] for k, v in o["books"].items()}
                except Exception:
                    pass
    return done


def main(limit=None):
    days = all_days()
    if limit:
        days = days[:limit]
    done = load_check()
    tasks = [(d, days[max(0, i - BUF_DAYS):i]) for i, d in enumerate(days) if d not in done]
    print(f"checkpoint: {len(done)} days done, {len(tasks)} to run", flush=True)
    t0 = datetime.now()
    print(f"{len(tasks)} days, {WORKERS} workers", flush=True)
    job = cap_cpu(CPU_CAP_PCT)
    print(f"cpu cap: {CPU_CAP_PCT:.0f}% via job object" if job else "cpu cap: job object failed, using watchdog", flush=True)
    stop_evt = threading.Event()
    wd = None
    if job is None:
        wd = threading.Thread(target=_watchdog, args=(stop_evt,), daemon=True)
        wd.start()
    try:
        with Pool(WORKERS) as pool:
            for day, res in pool.imap_unordered(worker, tasks, chunksize=1):
                done[day] = res
                with open(CHECK, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"day": day, "books": {k: [[j, p] for j, p in v]
                                                               for k, v in res.items()}}) + "\n")
    finally:
        stop_evt.set()
        if wd is not None:
            wd.join(timeout=3)
    print(f"done in {(datetime.now()-t0).seconds}s", flush=True)
    stats = {c[0]: [] for c in CONFIGS}
    for day in sorted(done):
        for name in stats:
            pts = done[day].get(name, [])
            stats[name].append((day, [{"pts": p[1]} for p in pts]))
    print(f"\n=== {len(done)} sessions ({START} .. {END}) ===")
    for name, _, _, _ in CONFIGS:
        s = bt.summarise(stats[name])
        print(f"{name:22s} tr {s['trades']:5d} WR {s['wr']:5.1f} pts {s['pts']:10.2f} "
              f"Rs {s['rs']:+10,} DD {s['dd']:8.2f} Cal {s['calmar']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)





