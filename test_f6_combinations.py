"""F6 Flag No-Divergence × 4 Strategy Combinations

Tests whether adding F6 (immediate flag entry: S4>=80 + S1<=20, no div/pinbar)
improves win rate or net profit on top of these 4 strategies:

  #1  Trailing SL (+5/+10), Fixed Cap     — baseline Rs +736,001  WR 36.6%
  #2  ATR×2.0/×4.0, Unlimited             — baseline Rs +701,533  WR 45.9%
  #3  S1=(12,3) + ATR×2.0/×4.0           — baseline Rs +644,933  WR 45.9%
  #4  S1=(7,3)  + ATR×2.0/×4.0           — baseline Rs +601,381  WR 47.2%

Each strategy is run WITH F6 signals added alongside standard pin bar signals.
Smoke test first (5 days) to validate baseline reproduction before full run.
"""

import time
from pathlib import Path
from collections import deque
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from backtest_5y_optimized import (
    load_spot, option_files, SYM_RE, to_minutes,
    latest_spot, summarize, print_yearly_breakdown,
)
from flattrade_bot.indicators.patterns import Candle, BullishPinBarDetector
from flattrade_bot.indicators.stochastic import IncrementalStochastic
from flattrade_bot.indicators.divergence import DivergenceEngine

# ── Smoke test toggle ─────────────────────────────────────────────────
SMOKE_TEST = False   # ← set True to validate on 5 days

# ── Shared constants ──────────────────────────────────────────────────
LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
CONSECUTIVE_LOSS_LIMIT = 6
ATR_PERIOD = 14
ATR_SL_MULT = 2.0
ATR_TP_MULT = 4.0
TRAIL_STEP_PTS   = 10.0
TRAIL_AMOUNT_PTS = 5.0

# TF_SPECS: (tf_minutes, lookback, sl_fallback, tp_fallback)
TF_SPECS = {
    "1m": (1, 10, 6.0,  30.0),
    "2m": (2,  5, 10.0, 15.0),
    "3m": (3,  4, 8.0,  25.0),
    "5m": (5,  3, 10.0, 35.0),
}

# ── Strategy configs ──────────────────────────────────────────────────
STRATEGIES = [
    {
        "id": "trailing_fixed",
        "label": "Trailing SL Fixed Cap + F6",
        "s1_k": 9, "s1_d": 3,
        "mode": "trailing",
        "daily_profit_pts":  30.0,   # Fixed cap (Rs 1,950)
        "daily_loss_rs":  -2000.0,   # Rs 2,000
        "baseline_rs": 736001, "baseline_trades": 6248, "baseline_wr": 36.6,
    },
    {
        "id": "atr_unlimited",
        "label": "ATR×2/×4 Unlimited + F6",
        "s1_k": 9, "s1_d": 3,
        "mode": "atr",
        "daily_profit_pts": float("inf"),  # No cap
        "daily_loss_rs":  -2000.0,
        "baseline_rs": 701533, "baseline_trades": 6080, "baseline_wr": 45.9,
    },
    {
        "id": "s1_12_atr",
        "label": "S1=(12,3) + ATR×2/×4 + F6",
        "s1_k": 12, "s1_d": 3,
        "mode": "atr",
        "daily_profit_pts": 30.0,
        "daily_loss_rs":  -2000.0,
        "baseline_rs": 644933, "baseline_trades": 5277, "baseline_wr": 45.9,
    },
    {
        "id": "s1_7_atr",
        "label": "S1=(7,3) + ATR×2/×4 + F6",
        "s1_k": 7, "s1_d": 3,
        "mode": "atr",
        "daily_profit_pts": 30.0,
        "daily_loss_rs":  -2000.0,
        "baseline_rs": 601381, "baseline_trades": 4487, "baseline_wr": 47.2,
    },
]

GLOBAL_SPOT   = {}
GLOBAL_CONFIG = {}

def init_worker(sd, cfg):
    global GLOBAL_SPOT, GLOBAL_CONFIG
    GLOBAL_SPOT = sd; GLOBAL_CONFIG = cfg


# ── ATR indicator ─────────────────────────────────────────────────────
class IncrementalATR:
    def __init__(self, period=14):
        self.period = period; self._buf = deque(maxlen=period)
        self.atr = None; self.prev_close = None; self._n = 0
    def update(self, h, l, c):
        tr = max(h-l, abs(h-self.prev_close), abs(l-self.prev_close)) if self.prev_close else h-l
        self._buf.append(tr); self._n += 1; self.prev_close = c
        if self._n < self.period:    self.atr = None
        elif self._n == self.period: self.atr = sum(self._buf)/self.period
        else:                        self.atr = (self.atr*(self.period-1)+tr)/self.period
        return self.atr


# ── Parameterised stochastic ──────────────────────────────────────────
class ParamStoch:
    def __init__(self, s1_k, s1_d):
        self.s1 = IncrementalStochastic(s1_k, s1_d)
        self.s2 = IncrementalStochastic(14, 3)
        self.s3 = IncrementalStochastic(40, 4)
        self.s4 = IncrementalStochastic(60, 10)
    def push(self, h, l, c):
        return {"s1d": self.s1.push(h,l,c), "s2d": self.s2.push(h,l,c),
                "s3d": self.s3.push(h,l,c), "s4d": self.s4.push(h,l,c)}


# ── Per-TF signal tracker (pin bar + divergence) ──────────────────────
class TFTracker:
    def __init__(self, lb, s1_k, s1_d):
        self.lb = lb
        self.stoch = ParamStoch(s1_k, s1_d)
        self.div = DivergenceEngine()
        self.hist = []; self.setup = False; self.stype = ""
        self.prev_s1 = None; self.s4_emb = 0
        self.atr = IncrementalATR(ATR_PERIOD)

    def push(self, c: Candle):
        self.hist.append(c)
        if len(self.hist) > 40: self.hist.pop(0)
        sv = self.stoch.push(c.high, c.low, c.close)
        s1, s2, s3, s4 = sv["s1d"], sv["s2d"], sv["s3d"], sv["s4d"]
        atr_val = self.atr.update(c.high, c.low, c.close)
        self.prev_s1 = s1
        if s4 is not None: self.s4_emb = self.s4_emb+1 if s4 <= 20 else 0
        emb = self.s4_emb > 25
        self.div.update(c.close, s1)
        bull_div = self.div.has_bullish_trough_divergence()
        is_flag  = s4 is not None and s1 is not None and s4 >= 79.5 and s1 <= 20.5
        is_super = all(v is not None and v <= 20.5 for v in (s1, s2, s3, s4))
        if (is_flag or is_super) and bull_div:
            self.setup = True; self.stype = "super" if is_super else "flag"
        is_rev = emb and self.stype == "super"
        triggered = False
        if self.setup and len(self.hist) >= 2:
            if BullishPinBarDetector.check_vicinity_breakout(self.hist, self.lb):
                triggered = True; self.setup = False
        return triggered, is_rev, self.stype, c.close, atr_val


# ── F6: Flag no-divergence immediate scanner ──────────────────────────
class FlagNoDivScanner:
    """Emits True the first bar that S4>=79.5 AND S1<=20.5 (no div/pinbar)."""
    def __init__(self, s1_k, s1_d):
        self.s1 = IncrementalStochastic(s1_k, s1_d)
        self.s4 = IncrementalStochastic(60, 10)
        self._fired = False
    def push(self, h, l, c):
        s1v = self.s1.push(h, l, c)
        s4v = self.s4.push(h, l, c)
        if s1v is None or s4v is None: return False
        flag = s4v >= 79.5 and s1v <= 20.5
        if flag and not self._fired:
            self._fired = True; return True
        if not flag:
            self._fired = False
        return False


# ── Multi-TF tracker with F6 ──────────────────────────────────────────
class MTFTracker:
    def __init__(self, s1_k, s1_d):
        self.trackers = {tf: TFTracker(spec[1], s1_k, s1_d) for tf, spec in TF_SPECS.items()}
        self.f6scans  = {tf: FlagNoDivScanner(s1_k, s1_d) for tf in TF_SPECS}
        self.bufs = {tf: [] for tf in TF_SPECS}

    def push_1m(self, c1m: Candle):
        out = []
        for tf, spec in TF_SPECS.items():
            self.bufs[tf].append(c1m)
            if len(self.bufs[tf]) == spec[0]:
                buf = self.bufs[tf]; self.bufs[tf] = []
                ctf = Candle(open=buf[0].open, high=max(x.high for x in buf),
                             low=min(x.low for x in buf), close=buf[-1].close,
                             minute=buf[-1].minute)
                trig, is_rev, stype, px, atr_val = self.trackers[tf].push(ctf)
                if trig:
                    out.append((tf, is_rev, stype, px, atr_val))
                # F6: immediate flag entry (parallel scanner)
                if self.f6scans[tf].push(ctf.high, ctf.low, ctf.close):
                    out.append((tf, False, "flag_nodiv", ctf.close, atr_val))
        return out


# ── Process one day ───────────────────────────────────────────────────
def process_day(args):
    day, fpath, fprev = args
    cfg = GLOBAL_CONFIG
    spot = GLOBAL_SPOT.get(day)
    s1_k, s1_d = cfg["s1_k"], cfg["s1_d"]
    mode = cfg["mode"]
    daily_profit_pts = cfg["daily_profit_pts"]
    daily_loss_rs    = cfg["daily_loss_rs"]
    daily_loss_pts   = daily_loss_rs / LOT_SIZE

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
    gc = {sym:g for sym,g in dfc.groupby("symbol")
          if (m:=SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}
    gp = {}
    if fprev and Path(fprev).exists():
        try:
            dfp = pd.read_csv(fprev, usecols=["time","symbol","open","high","low","close"], engine="c")
            if not dfp.empty:
                dfp["min"] = np.array([to_minutes(t) for t in dfp["time"]])
                gp = {sym:g for sym,g in dfp.groupby("symbol")
                      if (m:=SYM_RE.match(sym)) and int(m.group(2)) in target_strikes}
        except: pass

    trk = {}
    for sym, g in gp.items():
        trk[sym] = MTFTracker(s1_k, s1_d)
        mn, op, hi, lo, cl = (g[c].to_numpy() for c in ["min","open","high","low","close"])
        for i in range(len(mn)):
            trk[sym].push_1m(Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=mn[i]))

    pmtrig = {}; slices = {}
    for sym, g in gc.items():
        if sym not in trk: trk[sym] = MTFTracker(s1_k, s1_d)
        t = trk[sym]
        mn, op, hi, lo, cl = (g[c].to_numpy() for c in ["min","open","high","low","close"])
        slices[sym] = {"min": mn, "open": op, "high": hi, "low": lo, "close": cl}
        mm2 = SYM_RE.match(sym)
        if not mm2: continue
        sv, side = int(mm2.group(2)), mm2.group(3)
        for i in range(len(mn)):
            m = mn[i]
            for (tf, is_rev, stype, px, atr_val) in t.push_1m(
                    Candle(open=op[i],high=hi[i],low=lo[i],close=cl[i],minute=m)):
                pmtrig.setdefault(m, []).append(
                    (side, sv, sym, px, is_rev, tf, TF_SPECS[tf][2], TF_SPECS[tf][3], atr_val))

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
        sym = f"{prefix}{stk}{side}"; sl = slices.get(sym)
        return (sym, sl, stk) if sl is not None else None

    trades = []; pos = None; dpnl = 0.0; closs = 0; shut = False
    for minute in range(SESSION_START, DAY_LAST+1):
        if pos is not None:
            held = bslice(pos["slice"], minute)
            if held:
                o, h, l, c = held; pos["last_px"] = float(c); pos["duration_min"] += 1
                # Trailing SL update
                if mode == "trailing":
                    gain = c - pos["entry"]
                    steps = int(gain / TRAIL_STEP_PTS)
                    if steps > pos["trail_steps"]:
                        pos["sl"] += (steps - pos["trail_steps"]) * TRAIL_AMOUNT_PTS
                        pos["trail_steps"] = steps
                # Daily max loss check (Rs-based)
                if dpnl*LOT_SIZE + (c-pos["entry"])*LOT_SIZE <= daily_loss_rs:
                    pts = round(c-pos["entry"], 2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":c,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"SHUTDOWN_LOSS",
                        "duration_min":pos["duration_min"],"tf":pos["tf"]})
                    dpnl += pts; pos = None; shut = True; continue
                ex, rsn = None, ""
                has_tgt = pos.get("tgt") is not None
                if has_tgt and h >= pos["tgt"] and l <= pos["sl"]: ex, rsn = pos["sl"], "SL"
                elif has_tgt and h >= pos["tgt"]: ex, rsn = pos["tgt"], "TP"
                elif l <= pos["sl"]: ex, rsn = pos["sl"], "SL"
                # Bearish peak divergence exit
                if ex is None:
                    t1 = trk.get(pos["symbol"])
                    if t1:
                        t1m = t1.trackers["1m"]
                        t1m.div.update(c, t1m.prev_s1)
                        if t1m.div.has_bearish_peak_divergence(): ex, rsn = c, "BEARISH_PEAK_REVERSAL"
                if ex is not None:
                    pts = round(ex-pos["entry"], 2)
                    trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                        "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":ex,
                        "pts":pts,"rs":round(pts*LOT_SIZE),"reason":rsn,
                        "duration_min":pos["duration_min"],"tf":pos["tf"]})
                    dpnl += pts; closs = closs+1 if pts <= 0 else 0
                    if dpnl >= daily_profit_pts or closs >= CONSECUTIVE_LOSS_LIMIT or dpnl <= daily_loss_pts:
                        shut = True
                    pos = None
        if minute >= SESSION_END and pos is not None:
            pts = round(pos["last_px"]-pos["entry"], 2)
            trades.append({"date":day,"entry_min":pos["entry_min"],"exit_min":minute,
                "side":pos["side"],"symbol":pos["symbol"],"entry":pos["entry"],"exit":pos["last_px"],
                "pts":pts,"rs":round(pts*LOT_SIZE),"reason":"EOD",
                "duration_min":pos["duration_min"],"tf":pos["tf"]})
            dpnl += pts; pos = None; break
        if pos is not None or shut or minute >= SESSION_END: continue

        for (sig_side, sig_stk, sig_sym, c_px, is_rev, tf, sl_pts, tp_pts, atr_val) in pmtrig.get(minute, []):
            ai = ainfo(sig_side, minute)
            if ai and ai[2] == sig_stk and pos is None:
                if is_rev:
                    as2 = "PE" if sig_side == "CE" else "CE"
                    ai2 = ainfo(as2, minute)
                    if ai2 is None: continue
                    asym, asl, _ = ai2
                else:
                    as2 = sig_side; asym = sig_sym; asl = ai[1]
                bar = bslice(asl, minute)
                if bar:
                    ep = float(bar[3])
                    if mode == "trailing":
                        pos = {"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                               "sl":ep-sl_pts,"tgt":None,
                               "entry_min":minute,"last_px":ep,"duration_min":0,
                               "tf":tf,"trail_steps":0}
                    else:  # atr
                        if atr_val and atr_val > 0.5:
                            sl_use = atr_val * ATR_SL_MULT
                            tp_use = atr_val * ATR_TP_MULT
                        else:
                            sl_use = sl_pts; tp_use = tp_pts
                        pos = {"side":as2,"symbol":asym,"slice":asl,"entry":ep,
                               "sl":ep-sl_use,"tgt":ep+tp_use,
                               "entry_min":minute,"last_px":ep,"duration_min":0,"tf":tf}
                    break
    return trades


def run_strategy(cfg, spot_all, files, days):
    label = cfg["label"]
    print(f"\nRunning [{label}]...", flush=True)
    tasks = [(day, str(files[day]), str(files[days[i-1]]) if i > 0 else "")
             for i, day in enumerate(days)]
    t0 = time.time()
    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker,
              initargs=(spot_all, cfg)) as pool:
        for res in pool.map(process_day, tasks):
            all_trades.extend(res)
    st = summarize(all_trades)
    elapsed = time.time() - t0
    baseline = cfg["baseline_rs"]
    delta_rs = st["rs"] - baseline
    delta_wr = st["wr"] - cfg["baseline_wr"]
    delta_tr = st["trades"] - cfg["baseline_trades"]
    print(f"  Trades:{st['trades']:6,d} ({delta_tr:+,d}) | WR:{st['wr']:.1f}% ({delta_wr:+.1f}%) | "
          f"Rs:{st['rs']:+,d} ({delta_rs:+,d}) | PF:{st['pf']:.2f} | {elapsed:.0f}s", flush=True)
    return st, all_trades


def main():
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    days = sorted(set(files.keys()) & set(spot_all.keys()))

    if SMOKE_TEST:
        days = days[:5]
        print(f"=== SMOKE TEST — {len(days)} DAYS ONLY ===")
    print(f"Loaded {len(days)} trading days.")
    print("Testing F6 Flag No-Div combined with 4 top strategies...", flush=True)

    results = []
    for cfg in STRATEGIES:
        st, all_trades = run_strategy(cfg, spot_all, files, days)
        results.append((cfg, st, all_trades))

    w = 110
    print(f"\n{'='*w}")
    print("F6 COMBINATION RESULTS  (F6 = immediate flag entry when S4>=80 + S1<=20)")
    print(f"{'='*w}")
    hdr = f"{'STRATEGY':42s} | {'TRADES':>7} | {'WR%':>6} | {'NET PROFIT':>13} | {'PF':>5} | {'vs BASELINE':>13} | {'WR CHG':>7}"
    print(hdr)
    print(f"{'-'*w}")
    for cfg, st, _ in results:
        delta = st["rs"] - cfg["baseline_rs"]
        dwr   = st["wr"] - cfg["baseline_wr"]
        flag  = "[WIN]" if delta > 0 else "[---]"
        print(f"{cfg['label']:42s} | {st['trades']:7,d} | {st['wr']:6.1f}% | "
              f"Rs {st['rs']:+10,d} | {st['pf']:5.2f} | "
              f"Rs {delta:+10,d} {flag} | {dwr:+6.1f}%")


    print(f"\n{'='*w}")
    print("BASELINES (for comparison)")
    print(f"{'-'*w}")
    for cfg in STRATEGIES:
        print(f"{cfg['label'].replace(' + F6',''):42s} | {cfg['baseline_trades']:7,d} | "
              f"{cfg['baseline_wr']:6.1f}% | Rs {cfg['baseline_rs']:+10,d} |")

    print(f"\n{'='*w}")
    print("YEARLY BREAKDOWN — strategies that BEAT baseline")
    for cfg, st, all_trades in results:
        if st["rs"] > cfg["baseline_rs"]:
            print(f"\n>>> {cfg['label']}  (+Rs {st['rs']-cfg['baseline_rs']:,})")
            print_yearly_breakdown(all_trades)


if __name__ == "__main__":
    main()
