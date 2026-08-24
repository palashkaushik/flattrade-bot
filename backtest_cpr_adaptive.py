"""Backtest: CPR-Adaptive ATR Multipliers (2020-2024)

CPR (Central Pivot Range) width from PREVIOUS day classifies the current day:
  Narrow CPR  (< 30 pts)  → Trending day   → Bigger moves → SL×2.5 / TP×5.0
  Moderate CPR(30-60 pts) → Standard day   → Normal moves → SL×2.0 / TP×4.0
  Wide CPR    (> 60 pts)  → Sideways day   → Small moves  → SL×1.5 / TP×3.0

CPR Formula (Standard):
  P  = (PDH + PDL + PDC) / 3
  BC = (PDH + PDL) / 2
  TC = 2*P - BC
  CPR Width = |TC - BC|

Compared against fixed ATR×2.0/×4.0 baseline (Rs +701,533).
"""

import time
from pathlib import Path
from collections import deque
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from backtest_5y_optimized import (
    load_spot, option_files, SYM_RE, to_minutes,
    latest_spot, TimeframeTracker, summarize, print_yearly_breakdown,
)
from flattrade_bot.indicators.patterns import Candle

# ── Constants ────────────────────────────────────────────────────────
LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_MAX_LOSS_RS  = -2000.0
DAILY_MAX_LOSS_PTS = DAILY_MAX_LOSS_RS / LOT_SIZE
CONSECUTIVE_LOSS_LIMIT = 6
ATR_PERIOD = 14

DATA_DIR  = Path("C:/Websites/ammu")
SPOT_PATH = DATA_DIR / "index" / "NIFTY 50_minute.csv"

# ── CPR thresholds (Nifty-specific, from research) ───────────────────
# Narrow  < NARROW_THRESH  → trending, bigger moves
# Moderate NARROW..WIDE   → standard
# Wide    > WIDE_THRESH    → sideways, smaller moves
NARROW_THRESH = 30.0   # points
WIDE_THRESH   = 60.0   # points

# ATR multipliers per regime
CPR_REGIMES = {
    "narrow":   {"atr_sl": 2.5, "atr_tp": 5.0, "label": "Narrow  (<30)"},
    "moderate": {"atr_sl": 2.0, "atr_tp": 4.0, "label": "Moderate(30-60)"},
    "wide":     {"atr_sl": 1.5, "atr_tp": 3.0, "label": "Wide    (>60)"},
}

TF_SPECS = {
    "1m": (1, 10, 6.0,  30.0),
    "2m": (2,  5, 10.0, 15.0),
    "3m": (3,  4, 8.0,  25.0),
    "5m": (5,  3, 10.0, 35.0),
}

GLOBAL_SPOT   = {}
GLOBAL_CPR    = {}   # day -> cpr_width (float or None)
GLOBAL_CONFIG = {}

def init_worker_cpr(sd, cpr_map, cfg):
    global GLOBAL_SPOT, GLOBAL_CPR, GLOBAL_CONFIG
    GLOBAL_SPOT = sd
    GLOBAL_CPR  = cpr_map
    GLOBAL_CONFIG = cfg


# ── ATR indicator ─────────────────────────────────────────────────────
class IncrementalATR:
    def __init__(self, period=14):
        self.period = period; self._buf = deque(maxlen=period)
        self.atr = None; self.prev_close = None; self._n = 0
    def update(self, h, l, c):
        tr = max(h-l, abs(h-self.prev_close), abs(l-self.prev_close)) if self.prev_close else h-l
        self._buf.append(tr); self._n += 1; self.prev_close = c
        if self._n < self.period:    self.atr = None
        elif self._n == self.period: self.atr = sum(self._buf) / self.period
        else:                        self.atr = (self.atr*(self.period-1)+tr) / self.period
        return self.atr


class MTFTrackerATR:
    """Multi-TF tracker with per-TF ATR computation."""
    def __init__(self):
        self.trackers = {tf: TimeframeTracker(tf, spec[1]) for tf, spec in TF_SPECS.items()}
        self.atrs     = {tf: IncrementalATR(ATR_PERIOD) for tf in TF_SPECS}
        self.bufs     = {tf: [] for tf in TF_SPECS}

    def push_1m(self, c1m: Candle):
        out = []
        for tf, spec in TF_SPECS.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf]) == spec[0]:
                buf = self.bufs[tf]; self.bufs[tf] = []
                ctf = Candle(open=buf[0].open, high=max(x.high for x in buf),
                             low=min(x.low for x in buf), close=buf[-1].close, minute=buf[-1].minute)
                atr_val = self.atrs[tf].update(ctf.high, ctf.low, ctf.close)
                trig, is_rev, stype, px = self.trackers[tf].push(ctf)
                if trig:
                    out.append((tf, is_rev, stype, px, atr_val, spec[2], spec[3]))
        return out


# ── CPR helpers ───────────────────────────────────────────────────────
def calculate_cpr_width(high, low, close):
    """Standard CPR: P=(H+L+C)/3, BC=(H+L)/2, TC=2P-BC, width=|TC-BC|"""
    pivot = (high + low + close) / 3
    bc    = (high + low) / 2
    tc    = 2 * pivot - bc
    return abs(tc - bc)


def load_daily_ohlc():
    """Aggregate 1-minute spot to daily OHLC."""
    df = pd.read_csv(SPOT_PATH, parse_dates=["date"], engine="c")
    df["day"] = df["date"].dt.strftime("%Y-%m-%d")
    daily = df.groupby("day").agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last")
    ).reset_index()
    return daily


def build_cpr_map(daily_ohlc, days):
    """For each trading day, compute previous day's CPR width."""
    daily_ohlc = daily_ohlc.sort_values("day").reset_index(drop=True)
    day_to_idx = {row["day"]: i for i, row in daily_ohlc.iterrows()}
    cpr_map = {}
    for day in days:
        idx = day_to_idx.get(day)
        if idx is None or idx == 0:
            cpr_map[day] = None
            continue
        prev = daily_ohlc.iloc[idx-1]
        w = calculate_cpr_width(prev["high"], prev["low"], prev["close"])
        cpr_map[day] = w
    return cpr_map


def get_regime(cpr_width):
    if cpr_width is None:    return "moderate"  # default if no prev data
    if cpr_width < NARROW_THRESH: return "narrow"
    if cpr_width > WIDE_THRESH:   return "wide"
    return "moderate"


# ── Per-day simulation ────────────────────────────────────────────────
def process_day(args):
    day, fpath, fprev = args
    cfg   = GLOBAL_CONFIG
    spot  = GLOBAL_SPOT.get(day)
    cpr_w = GLOBAL_CPR.get(day)

    # Determine today's ATR multipliers from CPR
    regime = get_regime(cpr_w)
    atr_sl_mult = CPR_REGIMES[regime]["atr_sl"]
    atr_tp_mult = CPR_REGIMES[regime]["atr_tp"]

    if spot is None or not fpath: return []
    fp = Path(fpath)
    if not fp.exists(): return []
    sp0 = latest_spot(spot, 555) or latest_spot(spot, 560)
    if sp0 is None: return []
    atm0 = int(round(sp0/50)*50)
    target_strikes = set(range(atm0-250, atm0+300, 50))
    try:
        dfc = pd.read_csv(fp, usecols=["time","symbol","open","high","low","close"], engine="c")
    except: return []
    if dfc.empty: return []
    fsym = dfc["symbol"].iloc[0]; mm = SYM_RE.match(fsym)
    if not mm: return []
    prefix = mm.group(1)
    dfc["min"] = np.array([to_minutes(t) for t in dfc["time"]])
    gc = {sym: g for sym, g in dfc.groupby("symbol")
          if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}

    gp = {}
    if fprev and Path(fprev).exists():
        try:
            dfp = pd.read_csv(fprev, usecols=["time","symbol","open","high","low","close"], engine="c")
            if not dfp.empty:
                dfp["min"] = np.array([to_minutes(t) for t in dfp["time"]])
                gp = {sym: g for sym, g in dfp.groupby("symbol")
                      if (m := SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}
        except: pass

    trk = {}
    for sym, g in gp.items():
        trk[sym] = MTFTrackerATR()
        mn, op, hi, lo, cl = (g[c].to_numpy() for c in ["min","open","high","low","close"])
        for i in range(len(mn)):
            trk[sym].push_1m(Candle(open=op[i], high=hi[i], low=lo[i], close=cl[i], minute=mn[i]))

    pmtrig = {}; slices = {}
    for sym, g in gc.items():
        if sym not in trk:
            trk[sym] = MTFTrackerATR()
        t = trk[sym]
        mn, op, hi, lo, cl = (g[c].to_numpy() for c in ["min","open","high","low","close"])
        slices[sym] = {"min": mn, "open": op, "high": hi, "low": lo, "close": cl}
        mm2 = SYM_RE.match(sym)
        if not mm2: continue
        sv, side = int(mm2.group(2)), mm2.group(3)
        for i in range(len(mn)):
            m = mn[i]
            for item in t.push_1m(Candle(open=op[i], high=hi[i], low=lo[i], close=cl[i], minute=m)):
                pmtrig.setdefault(m, []).append((side, sv, sym) + item)

    def bslice(sl, m):
        idx = np.searchsorted(sl["min"], m)
        if idx < len(sl["min"]) and sl["min"][idx] == m:
            return sl["open"][idx], sl["high"][idx], sl["low"][idx], sl["close"][idx]
        return None

    def ainfo(side, m):
        spx = latest_spot(spot, m)
        if spx is None: return None
        atm = int(round(spx/50)*50)
        stk = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        sym = f"{prefix}{stk}{side}"
        sl  = slices.get(sym)
        return (sym, sl, stk) if sl is not None else None

    trades = []; pos = None; dpnl = 0.0; closs = 0; shut = False

    for minute in range(SESSION_START, DAY_LAST+1):
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held
                pos["last_px"] = float(c); pos["duration_min"] += 1
                if dpnl*LOT_SIZE + (c-pos["entry"])*LOT_SIZE <= DAILY_MAX_LOSS_RS:
                    pts = round(c - pos["entry"], 2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":c,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"SHUTDOWN_LOSS",
                        "duration_min":pos["duration_min"],"tf":pos["tf"],
                        "regime":pos["regime"],"cpr_w":pos["cpr_w"]})
                    dpnl+=pts; pos=None; shut=True; continue
                ex=None; rsn=""
                has_tgt = pos.get("tgt") is not None
                if has_tgt and h>=pos["tgt"] and l<=pos["sl"]: ex, rsn = pos["sl"], "SL"
                elif has_tgt and h>=pos["tgt"]:                ex, rsn = pos["tgt"], "TP"
                elif l<=pos["sl"]:                             ex, rsn = pos["sl"], "SL"
                if ex is not None:
                    pts = round(ex - pos["entry"], 2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":ex,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":rsn,
                        "duration_min":pos["duration_min"],"tf":pos["tf"],
                        "regime":pos["regime"],"cpr_w":pos["cpr_w"]})
                    dpnl+=pts; closs=closs+1 if pts<=0 else 0
                    if closs>=CONSECUTIVE_LOSS_LIMIT or dpnl<=DAILY_MAX_LOSS_PTS: shut=True
                    pos=None

        if minute>=SESSION_END and pos is not None:
            pts = round(pos["last_px"] - pos["entry"], 2)
            trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":pos["last_px"],
                "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"EOD",
                "duration_min":pos["duration_min"],"tf":pos["tf"],
                "regime":pos["regime"],"cpr_w":pos["cpr_w"]})
            dpnl+=pts; pos=None; break

        if pos is not None or shut or minute>=SESSION_END: continue

        for item in pmtrig.get(minute, []):
            sig_side, sig_stk, sig_sym, tf, is_rev, stype, c_px, atr_val, tf_sl, tf_tp = item
            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                if is_rev:
                    as2 = "PE" if sig_side=="CE" else "CE"
                    ai2 = ainfo(as2, minute)
                    if ai2 is None: continue
                    asym, asl, _ = ai2
                else:
                    as2 = sig_side; asym = sig_sym; asl = ai[1]
                bar = bslice(asl, minute)
                if bar:
                    ep = float(bar[3])
                    sl_p = atr_val*atr_sl_mult if (atr_val and atr_val > 0.5) else tf_sl
                    tp_p = atr_val*atr_tp_mult if (atr_val and atr_val > 0.5) else tf_tp
                    pos = {"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                           "sl":ep-sl_p,"tgt":ep+tp_p,"entry_min":minute,
                           "last_px":ep,"duration_min":0,"tf":tf,
                           "regime":regime,"cpr_w":round(cpr_w,1) if cpr_w else 0}
                    break
    return trades


def main():
    print("Loading spot data...", flush=True)
    spot_all   = load_spot()
    daily_ohlc = load_daily_ohlc()
    files      = option_files("2020-01-01", "2024-12-31")
    days       = sorted(set(files.keys()) & set(spot_all.keys()))
    cpr_map    = build_cpr_map(daily_ohlc, days)

    # Print CPR statistics
    widths = [w for w in cpr_map.values() if w is not None]
    print(f"\nCPR Width Statistics (2020-2024, {len(days)} days):")
    print(f"  Mean:   {np.mean(widths):.1f} pts")
    print(f"  Median: {np.median(widths):.1f} pts")
    print(f"  P25:    {np.percentile(widths,25):.1f} pts")
    print(f"  P75:    {np.percentile(widths,75):.1f} pts")
    narrow  = sum(1 for w in widths if w < NARROW_THRESH)
    moderate= sum(1 for w in widths if NARROW_THRESH <= w <= WIDE_THRESH)
    wide    = sum(1 for w in widths if w > WIDE_THRESH)
    print(f"  Narrow  (<{NARROW_THRESH}pts): {narrow} days ({narrow/len(widths)*100:.1f}%)")
    print(f"  Moderate({NARROW_THRESH}-{WIDE_THRESH}pts): {moderate} days ({moderate/len(widths)*100:.1f}%)")
    print(f"  Wide    (>{WIDE_THRESH}pts): {wide} days ({wide/len(widths)*100:.1f}%)")

    print(f"\nRunning CPR-Adaptive ATR backtest...", flush=True)
    print(f"  Narrow  (<{NARROW_THRESH}): SL×{CPR_REGIMES['narrow']['atr_sl']} / TP×{CPR_REGIMES['narrow']['atr_tp']}")
    print(f"  Moderate({NARROW_THRESH}-{WIDE_THRESH}): SL×{CPR_REGIMES['moderate']['atr_sl']} / TP×{CPR_REGIMES['moderate']['atr_tp']}")
    print(f"  Wide    (>{WIDE_THRESH}): SL×{CPR_REGIMES['wide']['atr_sl']} / TP×{CPR_REGIMES['wide']['atr_tp']}\n", flush=True)

    tasks = [(day, str(files[day]),
              str(files[days[i-1]]) if i > 0 else "")
             for i, day in enumerate(days)]
    cfg = {}
    t0 = time.time()
    all_trades = []
    with Pool(processes=min(cpu_count(), 8),
              initializer=init_worker_cpr,
              initargs=(spot_all, cpr_map, cfg)) as pool:
        for res in pool.map(process_day, tasks):
            all_trades.extend(res)
    st = summarize(all_trades)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    BASELINE_ATR    = 701533
    BASELINE_TRAIL  = 739193

    print(f"\n{'='*90}")
    print(f"CPR-ADAPTIVE ATR BACKTEST RESULTS (2020-2024)")
    print(f"{'='*90}")
    print(f"Trades     : {st['trades']:,}")
    print(f"Win Rate   : {st['wr']:.1f}%")
    print(f"Net Points : {st['pts']:+,.2f}")
    print(f"Net Profit : Rs {st['rs']:+,d}")
    print(f"Profit Factor: {st['pf']:.2f}")
    print(f"\nvs Fixed ATR×2.0/×4.0   : {st['rs']-BASELINE_ATR:+,d} (baseline Rs +{BASELINE_ATR:,})")
    print(f"vs Trailing SL Unlimited : {st['rs']-BASELINE_TRAIL:+,d} (baseline Rs +{BASELINE_TRAIL:,})")

    print_yearly_breakdown(all_trades)

    # ── Per-regime breakdown ─────────────────────────────────────────
    df = pd.DataFrame(all_trades)
    if not df.empty and "regime" in df.columns:
        print(f"\n{'='*90}")
        print("REGIME BREAKDOWN")
        print(f"{'='*90}")
        print(f"{'REGIME':20s} | {'TRADES':7} | {'WR%':6} | {'WIN TRADES':10} | {'NET PTS':10} | {'NET Rs':12} | {'PF':6}")
        print(f"{'-'*90}")
        for reg in ["narrow","moderate","wide"]:
            sub = df[df["regime"]==reg]
            if sub.empty: continue
            wins = (sub["pts"] > 0).sum()
            wr   = wins/len(sub)*100
            net_pts = sub["pts"].sum()
            net_rs  = sub["rs"].sum()
            win_pts = sub[sub["pts"]>0]["pts"].sum()
            loss_pts= abs(sub[sub["pts"]<0]["pts"].sum())
            pf      = win_pts/loss_pts if loss_pts > 0 else 999
            cfg_label = CPR_REGIMES[reg]["label"]
            print(f"{cfg_label:20s} | {len(sub):7,d} | {wr:5.1f}% | {wins:10,d} | "
                  f"{net_pts:+10.1f} | Rs {net_rs:+10,d} | {pf:6.2f}")

        # ── CPR width distribution vs performance ────────────────────
        print(f"\n{'='*90}")
        print("CPR WIDTH BUCKETS (every 15 pts) vs TRADE PERFORMANCE")
        print(f"{'='*90}")
        print(f"{'CPR RANGE':15s} | {'DAYS':5} | {'TRADES':7} | {'WR%':6} | {'AVG PTS/TRADE':13} | {'NET Rs':12}")
        print(f"{'-'*90}")
        df["cpr_bucket"] = (df["cpr_w"] // 15) * 15
        for bkt in sorted(df["cpr_bucket"].unique()):
            sub = df[df["cpr_bucket"]==bkt]
            wins = (sub["pts"]>0).sum()
            wr   = wins/len(sub)*100 if len(sub)>0 else 0
            avg_pts = sub["pts"].mean()
            days_cnt = sub["date"].nunique()
            print(f"{bkt:.0f}-{bkt+15:.0f} pts       | {days_cnt:5d} | {len(sub):7,d} | "
                  f"{wr:5.1f}% | {avg_pts:+13.2f} | Rs {sub['rs'].sum():+10,d}")

    print(f"\nCPR Thresholds: Narrow<{NARROW_THRESH} | Moderate {NARROW_THRESH}-{WIDE_THRESH} | Wide>{WIDE_THRESH}")
    print(f"Adjust NARROW_THRESH and WIDE_THRESH at top of file to optimize.")


if __name__ == "__main__":
    main()
