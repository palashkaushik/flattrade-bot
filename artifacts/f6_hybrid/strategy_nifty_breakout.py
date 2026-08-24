"""
NIFTY TWO-CANDLE BREAKOUT  (translation of "NIFTY BANK INTRADAY OPTION BUYING
SINGLE SUCCESSFUL STRATEGY" by Akshay VG, applied to our Nifty 50 data)

Strategy core (from the book):
  - Trade the breakout of the FIRST K 5-minute candles after open.
  - Draw the range from those candles (body or wick).
  - First candle that closes/breaks outside the range -> entry:
        bullish breakout -> BUY slightly OTM CALL
        bearish breakout -> BUY slightly OTM PUT
  - Strike: slightly OTM (2-4 strikes from ATM).  Captured move scaled by a
    delta factor that falls with OTM distance.
  - Stop: opposite range line, or the nearest CPR level inside the "zone"
    (book cases 1 & 2), or the nearest CPR level just outside (book case 3).
  - Target: ride 60-75% of the distance to the next CPR level (level_ride),
    or a fixed % of the underlying move (pct).
  - Rules: trade only ONCE per day; no Fridays; strict SL.

Money model kept identical to the Optimus engine for ledger comparability:
    pnl_rs = move_points * capture_factor * 0.5 * LOT_SIZE(65) - FEE(30)

This is a SEPARATE price-action strategy -> its own parameter grid + honest
walk-forward (IS 2020-2023, OOS 2024-2026).
"""
import sys, json, time, os, itertools
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
sys.path.insert(0, str(ROOT))
import opt_futures_quad as source

LOT_SIZE = 65
FEE = 30.0
C = 5          # 5-minute candles
NC = 75        # candles per session (375 min / 5)
IS_END = "2023-12-31"
OOS_START = "2024-01-01"


def load_data():
    spot = source.load_spot()
    days = sorted(d for d in spot if d >= "2020-01-01")
    # candle ohlc per day: (D, NC), grouping consecutive bars into 5-min candles
    co = np.zeros((len(days), NC)); ch = np.zeros_like(co); cl = np.zeros_like(co); cc = np.zeros_like(co)
    dH = np.zeros(len(days)); dL = np.zeros(len(days)); dC = np.zeros(len(days))
    wk = np.zeros(len(days), dtype=int)
    for i, d in enumerate(days):
        g = spot[d]
        n = len(g["min"])
        ci = np.arange(n) // C
        nc = int(ci[-1]) + 1
        for c in range(min(nc, NC)):
            m = ci == c
            co[i, c] = g["open"][m][0]
            ch[i, c] = g["high"][m].max()
            cl[i, c] = g["low"][m].min()
            cc[i, c] = g["close"][m][-1]
        dH[i] = g["high"].max(); dL[i] = g["low"].min(); dC[i] = g["close"][-1]
        wk[i] = pd.Timestamp(d).dayofweek   # 0=Mon .. 4=Fri
    return days, co, ch, cl, cc, dH, dL, dC, wk


def cpr_levels(prevH, prevL, prevC):
    """Return dict of (D,) arrays for CPR levels from previous-day HLC."""
    p = (prevH + prevL + prevC) / 3.0
    R1 = 2 * p - prevL; S1 = 2 * p - prevH
    R2 = p + (R1 - S1); S2 = p - (R1 - S1)
    R3 = prevH + 2 * (p - prevL); S3 = prevL - 2 * (prevH - p)
    R4 = R3 + (R2 - R1); S4 = S3 - (R2 - R1)
    return {"pivot": p, "R1": R1, "R2": R2, "R3": R3, "R4": R4,
            "S1": S1, "S2": S2, "S3": S3, "S4": S4,
            "PDH": prevH, "PDL": prevL}


def run_cfg(cfg, days, co, ch, cl, cc, dH, dL, dC, wk, idx_mask):
    """Vectorized backtest for one config over the selected day subset."""
    K = cfg["open_candles"]
    rng_hi = np.zeros(len(days)); rng_lo = np.zeros(len(days))
    for c in range(K):
        if cfg["range_mode"] == "body":
            bh = np.maximum(co[:, c], cc[:, c]); bl = np.minimum(co[:, c], cc[:, c])
        else:
            bh = ch[:, c]; bl = cl[:, c]
        rng_hi = np.maximum(rng_hi, bh); rng_lo = np.minimum(rng_lo, bl)
    span = np.maximum(rng_hi - rng_lo, 1.0)
    buf = cfg["break_buf"]
    eu = cfg["entry_until"]

    # breakout detection per candidate candle c in [K, eu]
    cand = np.arange(NC)
    cand_mask = (cand >= K) & (cand <= eu)
    cand_idx = np.where(cand_mask)[0]
    bull_cond = np.zeros((len(days), NC), dtype=bool)
    bear_cond = np.zeros((len(days), NC), dtype=bool)
    for c in cand_idx:
        if cfg["break_mode"] == "close":
            b_up = cc[:, c] > rng_hi + buf
            b_dn = cc[:, c] < rng_lo - buf
        else:
            b_up = ch[:, c] > rng_hi + buf
            b_dn = cl[:, c] < rng_lo - buf
        bull_cond[:, c] = b_up
        bear_cond[:, c] = b_dn

    def first_true(m):
        f = np.where(m.any(1), m.argmax(1), 9999)
        return f
    bf = first_true(bull_cond); kf = first_true(bear_cond)
    if cfg["direction"] == "bull":
        kf = np.full_like(kf, 9999)
    elif cfg["direction"] == "bear":
        bf = np.full_like(bf, 9999)
    dirn = np.where(bf <= kf, 1, -1)
    b = np.minimum(bf, kf)            # breakout candle index (-1? no: 9999 if none)
    no_trade = (b >= 9999) | (b >= NC - 1)
    # Friday filter
    if not cfg["allow_friday"]:
        no_trade = no_trade | (wk == 4)

    D = len(days)
    b_safe = np.minimum(b, NC - 1)
    entry = np.take_along_axis(cc, b_safe[:, None], axis=1)[:, 0]
    entry = np.where(no_trade, np.nan, entry)
    bull = dirn == 1

    # ---- SL price ----
    prevH = np.roll(dH, 1); prevL = np.roll(dL, 1); prevC = np.roll(dC, 1)
    cpr = cpr_levels(prevH, prevL, prevC)
    levels = list(cpr.values())  # each (D,)
    sl = np.where(bull, rng_lo - cfg["sl_buf"], rng_hi + cfg["sl_buf"]).astype(float)
    if cfg["sl_mode"] in ("cpr", "cpr_respect"):
        for lv in levels:
            L = lv.astype(float)
            if cfg["sl_mode"] == "cpr":
                inside = (L >= rng_lo) & (L <= rng_hi)
                cand_bull = np.where(inside, L, -np.inf)   # highest support inside
                cand_bear = np.where(inside, L, +np.inf)   # lowest resistance inside
            else:  # cpr_respect: also nearest just outside on SL side
                dist = np.abs(L - entry)
                near = dist <= 1.5 * span
                cand_bull = np.where(near & (L < entry), L, -np.inf)
                cand_bear = np.where(near & (L > entry), L, +np.inf)
            sl_b = np.maximum(sl, cand_bull) if False else None
            # pick best: for bull we want max of (opposite line, inside levels)? use max valid
            sl_bull = np.where(cand_bull > -np.inf, np.maximum(cand_bull, rng_lo - cfg["sl_buf"]), sl)
            sl_bear = np.where(cand_bear < np.inf, np.minimum(cand_bear, rng_hi + cfg["sl_buf"]), sl)
            sl = np.where(bull, sl_bull, sl_bear)
    sl = np.where(no_trade, np.nan, sl)

    # ---- target price ----
    if cfg["target_mode"] == "level_ride":
        tgt_bull = np.full(D, np.nan); tgt_bear = np.full(D, np.nan)
        for lv in levels:
            L = lv.astype(float)
            above = np.where(L > entry, L, np.inf)
            below = np.where(L < entry, L, -np.inf)
            tgt_bull = np.minimum(tgt_bull, above) if False else np.fmin(tgt_bull, above)
            tgt_bear = np.maximum(tgt_bear, below)
        # fallback if no CPR level found: ride fraction of range span
        tgt_bull = np.where(np.isinf(tgt_bull), entry + cfg["ride_frac"] * span, tgt_bull)
        tgt_bear = np.where(np.isinf(tgt_bear), entry - cfg["ride_frac"] * span, tgt_bear)
        target = np.where(bull, entry + cfg["ride_frac"] * (tgt_bull - entry),
                                 entry - cfg["ride_frac"] * (entry - tgt_bear))
    else:
        target = np.where(bull, entry * (1 + cfg["pct_target"]),
                                 entry * (1 - cfg["pct_target"]))
    target = np.where(no_trade, np.nan, target)

    # ---- exit scan (vectorized) ----
    col = np.arange(NC)[None, :]
    after = col > b[:, None]
    tgt_hit = after & (ch >= target[:, None])
    sl_hit = after & (cl <= sl[:, None])
    t_first = np.where(tgt_hit.any(1), tgt_hit.argmax(1), 9999)
    s_first = np.where(sl_hit.any(1), sl_hit.argmax(1), 9999)
    win_t = (t_first < s_first) & (t_first < 9999)
    loss_t = (s_first <= t_first) & (s_first < 9999)
    last = np.take_along_axis(cc, np.full(D, NC - 1, dtype=int)[:, None], axis=1)[:, 0]
    exit_px = np.where(win_t, target, np.where(loss_t, sl, last))
    move = np.where(bull, exit_px - entry, entry - exit_px)
    cap = np.maximum(0.35, 0.5 - 0.02 * cfg["otm_strikes"])
    pnl = move * cap * 0.5 * LOT_SIZE - FEE
    pnl = np.where(no_trade, np.nan, pnl)

    return pnl[idx_mask]   # already float array; caller aggregates


def aggregate(pnl):
    p = pnl[~np.isnan(pnl)]
    if len(p) == 0:
        return dict(trades=0, win_rate=0.0, net_rs=0.0, pf=0.0, max_dd=0.0)
    wins = p[p > 0]; losses = p[p < 0]
    gross_w = wins.sum(); gross_l = -losses.sum()
    pf = (gross_w / gross_l) if gross_l > 0 else (float("inf") if gross_w > 0 else 0.0)
    eq = np.cumsum(p); peak = np.maximum.accumulate(eq); dd = (peak - eq).max()
    return dict(trades=int(len(p)), win_rate=100.0 * len(wins) / len(p),
                net_rs=float(p.sum()), pf=float(pf), max_dd=float(dd))


GRID = dict(
    open_candles=[2, 3],
    range_mode=["body", "wick"],
    break_mode=["close", "high_low"],
    break_buf=[0, 2, 5],
    entry_until=[6, 12],
    otm_strikes=[2, 3, 4],
    sl_mode=["opposite", "cpr", "cpr_respect"],
    sl_buf=[0, 3, 5],
    target_mode=["level_ride", "pct"],
    ride_frac=[0.6, 0.7, 0.75],
    pct_target=[0.06, 0.10],
    allow_friday=[False, True],
    direction=["both", "bull", "bear"],
)


def main():
    t0 = time.time()
    days, co, ch, cl, cc, dH, dL, dC, wk = load_data()
    is_mask = np.array([d <= IS_END for d in days])
    oos_mask = np.array([d >= OOS_START for d in days])
    print(f"Days: {len(days)} | IS {is_mask.sum()} | OOS {oos_mask.sum()} | "
          f"load {time.time()-t0:.1f}s", flush=True)

    keys = [k for k in GRID if k not in ("ride_frac", "pct_target")]

    def gen_combos():
        for vals in itertools.product(*(GRID[k] for k in keys)):
            base = dict(zip(keys, vals))
            for tm in ("level_ride", "pct"):
                cfg = dict(base); cfg["target_mode"] = tm
                if tm == "level_ride":
                    for rf in GRID["ride_frac"]:
                        c = dict(cfg); c["ride_frac"] = rf; c["pct_target"] = 0.0
                        yield c
                else:
                    for pt in GRID["pct_target"]:
                        c = dict(cfg); c["pct_target"] = pt; c["ride_frac"] = 0.0
                        yield c

    combos = list(gen_combos())
    print(f"Grid combos: {len(combos):,}", flush=True)

    smoke = int(os.environ.get("SMOKE", "0"))
    if smoke:
        days = days[:smoke]
        co, ch, cl, cc = co[:smoke], ch[:smoke], cl[:smoke], cc[:smoke]
        dH, dL, dC, wk = dH[:smoke], dL[:smoke], dC[:smoke], wk[:smoke]
        is_mask = np.ones(len(days), bool); oos_mask = is_mask.copy()
        print(f"*** SMOKE TEST: {len(days)} days ***", flush=True)

    full_mask = np.ones(len(days), bool)
    rows = []
    for i, cfg in enumerate(combos):
        # skip invalid: ride_frac only meaningful for level_ride, pct only for pct
        is_p = run_cfg(cfg, days, co, ch, cl, cc, dH, dL, dC, wk, is_mask)
        oos_p = run_cfg(cfg, days, co, ch, cl, cc, dH, dL, dC, wk, oos_mask)
        full_p = run_cfg(cfg, days, co, ch, cl, cc, dH, dL, dC, wk, full_mask)
        ai = aggregate(is_p); ao = aggregate(oos_p); af = aggregate(full_p)
        rows.append((cfg, ai, ao, af))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(combos)}  {time.time()-t0:.0f}s", flush=True)

    # top by FULL 7-year net
    rows.sort(key=lambda r: r[3]["net_rs"], reverse=True)
    print("\n=== TOP 15 BY FULL 7-YR NET (2020-2026) ===")
    print(f"{'cfg':<118}{'FULL_net':>12}{'F_WR':>7}{'F_PF':>7}{'OOS_net':>12}{'OOS_WR':>7}{'OOS_PF':>7}")
    for cfg, ai, ao, af in rows[:15]:
        cs = (f"K{cfg['open_candles']} {cfg['range_mode'][0]}{cfg['break_mode'][0]} "
              f"buf{cfg['break_buf']} eu{cfg['entry_until']} otm{cfg['otm_strikes']} "
              f"{cfg['sl_mode']} slb{cfg['sl_buf']} {cfg['target_mode'][:3]}"
              f"{cfg['ride_frac'] if cfg['target_mode']=='level_ride' else cfg['pct_target']} "
              f"fr{cfg['allow_friday']} {cfg['direction']}")
        print(f"{cs:<118}{af['net_rs']:>12,.0f}{af['win_rate']:>7.1f}{af['pf']:>7.2f}"
              f"{ao['net_rs']:>12,.0f}{ao['win_rate']:>7.1f}{ao['pf']:>7.2f}")

    # top by IS net
    rows.sort(key=lambda r: r[1]["net_rs"], reverse=True)
    print("\n=== TOP 15 BY IS NET (2020-2023) ===")
    print(f"{'cfg':<120}{'IS_net':>12}{'IS_WR':>7}{'IS_PF':>7}{'OOS_net':>12}{'OOS_WR':>7}{'OOS_PF':>7}")
    for cfg, ai, ao, af in rows[:15]:
        cs = (f"K{cfg['open_candles']} {cfg['range_mode'][0]}{cfg['break_mode'][0]} "
              f"buf{cfg['break_buf']} eu{cfg['entry_until']} otm{cfg['otm_strikes']} "
              f"{cfg['sl_mode']} slb{cfg['sl_buf']} {cfg['target_mode'][:3]}"
              f"{cfg['ride_frac'] if cfg['target_mode']=='level_ride' else cfg['pct_target']} "
              f"fr{cfg['allow_friday']} {cfg['direction']}")
        print(f"{cs:<120}{ai['net_rs']:>12,.0f}{ai['win_rate']:>7.1f}{ai['pf']:>7.2f}"
              f"{ao['net_rs']:>12,.0f}{ao['win_rate']:>7.1f}{ao['pf']:>7.2f}")

    # robustness over top-20 by IS
    top = rows[:20]
    is_pos = sum(1 for _, ai, _, _ in top if ai["net_rs"] > 0)
    oos_pos = sum(1 for _, _, ao, _ in top if ao["net_rs"] > 0)
    mean_is = np.mean([ai["net_rs"] for _, ai, _, _ in top])
    mean_oos = np.mean([ao["net_rs"] for _, _, ao, _ in top])
    wfe = mean_oos / mean_is if mean_is else 0
    print(f"\n=== ROBUSTNESS (top-20 by IS) ===")
    print(f"IS mean net  : {mean_is:,.0f}")
    print(f"OOS mean net : {mean_oos:,.0f}")
    print(f"WFE (OOS/IS) : {wfe:.3f}")
    print(f"IS-positive  : {is_pos}/20")
    print(f"OOS-positive : {oos_pos}/20")

    # best OOS
    rows_oos = sorted(rows, key=lambda r: r[2]["net_rs"], reverse=True)
    print("\n=== TOP 10 BY OOS NET (2024-2026) ===")
    for cfg, ai, ao, af in rows_oos[:10]:
        cs = (f"K{cfg['open_candles']} {cfg['range_mode'][0]}{cfg['break_mode'][0]} "
              f"otm{cfg['otm_strikes']} {cfg['sl_mode']} {cfg['target_mode'][:3]}"
              f"{cfg['ride_frac'] if cfg['target_mode']=='level_ride' else cfg['pct_target']} "
              f"{cfg['direction']}")
        print(f"{cs:<70}{'IS':>10}{ai['net_rs']:>10,.0f} | {'OOS':>10}{ao['net_rs']:>10,.0f} "
              f"WR{ao['win_rate']:.1f} PF{ao['pf']:.2f} T{ao['trades']}")

    # save full results
    out = []
    for cfg, ai, ao, af in rows:
        r = dict(cfg); r.update(dict(is_net=ai["net_rs"], is_wr=ai["win_rate"], is_pf=ai["pf"],
                                     is_t=ai["trades"], oos_net=ao["net_rs"], oos_wr=ao["win_rate"],
                                     oos_pf=ao["pf"], oos_t=ao["trades"],
                                     full_net=af["net_rs"], full_wr=af["win_rate"], full_pf=af["pf"],
                                     full_t=af["trades"]))
        out.append(r)
    pd.DataFrame(out).to_csv("breakout_sweep.csv", index=False)
    print(f"\nSaved breakout_sweep.csv ({len(out)} rows) | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
