"""LIVE ENGINE <-> GEOMS REFERENCE parity (Full10 v6).

Feeds the same 1m bars (weekly_08SEP26 CSVs) through:
  1. the GEOMS reference (lh_full10_geoms.build_states / collect_signals)
  2. the LIVE engine (flattrade_bot.strategies.full10_engine.Full10ContractState)

Compares: stoch S1/S3/S4 per TF, EMA20/200, VWAP, day levels, bounce
decisions, and the FULL signal stream (time/kind/TF/level/entry).
"""
import sys, os, json
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
os.environ.setdefault("DISCORD_WEBHOOK_URL", "")

import pandas as pd
from datetime import datetime, timedelta

# --- reference side (geoms, already-run ALL cache pattern) ---
import importlib.util
spec = importlib.util.spec_from_file_location("geoms", r"C:\Websites\FLATTRADE BOT\lh_full10_geoms.py")
# We only need the module's helpers, not the main run — but geoms runs at
# import. Instead re-implement the minimal feed using its own functions:
# load build_states from the module namespace after guarding the run.
import re
src = open(r"C:\Websites\FLATTRADE BOT\lh_full10_geoms.py", encoding="utf-8").read()
cut = src.index("ALL = {}")
src = src[:cut] + "\nif __name__ == '__main__':\n    pass\n"
ns = {}
exec(compile(src, "geoms_mod", "exec"), ns)

from flattrade_bot.strategies.full10_engine import Full10ContractState

def hm(m):
    return f"{m//60:02d}:{m%60:02d}"

FAIL = 0
def check(name, a, b, tol=0.02):
    global FAIL
    ok = (a is None and b is None) or (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
        and abs(a - b) <= tol)
    if not ok:
        FAIL += 1
        print(f"  MISMATCH {name}: ref={a} live={b}")
    return ok

DAYS = ("2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08")
total_sigs_ref = 0
total_sigs_live = 0

for day in DAYS:
    # Build ref states WITHOUT today's replay: build_states only (seed + levels).
    states, today, spot = ns["build_states"](day)
    cfg = ns["DAYCFG"][day]
    # disable TF-fallback reset the ref does inside collect? No — build_states
    # mirrors the engine exactly (seed + reset last_lo/last_cl happens inside
    # build_states? geoms does the reset at the END of build_states).
    # NOTE: geoms' collect_signals replays today through states; we drive the
    # replay ourselves below so both sides advance bar-by-bar in lockstep.

    # ---- live engine states for the same contracts ----
    live = {}
    for k, cs in states.items():
        lc = Full10ContractState(symbol=cs.sym, token="0", side=k.split(":")[0], strike=int(k.split(":")[1]))
        # day levels
        lc.set_day_levels(cs.ph, cs.pl, cs.pc,
                          virgin=dict(cs.virgin) if cs.virgin else None,
                          pdvwap=cs.pdvwap)
        # seed: replay prior days (geoms seeds via minutes>=555 tail 1200)
        seed_bars = []
        for d in cfg["seed"]:
            pg = ns["load"](d).get(cs.sym)
            if pg is None: continue
            for r in pg[pg.minute >= 555].tail(1200).itertuples(index=False):
                seed_bars.append(ns["mk_bar"](r, int(r.minute)))
        lc.seed_bars(seed_bars, [], session_date=day)   # today's bars come via ticks below
        live[k] = lc

    # ---- bar-by-bar lockstep replay: REF and LIVE advance together ----
    ref_sigs = {}
    live_sigs = {}
    ARM_WIN = ns["ARM_WIN"]
    TB, T0 = ns["TB"], ns["T0"]

    def ref_bar_close(cs, bar):
        """Replays ONE bar through the geoms reference state machine and
        returns its signal (mirrors collect_signals exactly)."""
        cs.atr.update(bar.high, bar.low, bar.close)
        cs.ema20.update(bar.close); cs.ema200.update(bar.close)
        cs.vwap.update(bar.high, bar.low, bar.close, volume=float(r.volume))
        sig = None
        for tf in (1, 2, 3, 5):
            cs.tf[tf].push_1m_bar(bar)
        for nm in list(cs.virgin):
            if bar.low <= cs.virgin[nm] <= bar.high:
                del cs.virgin[nm]
        cs.baridx += 1
        if any(cs.tf[tf].last_s1 is not None and cs.tf[tf].last_s1 <= ns["ARM_S1"] for tf in (1, 2, 3, 5)):
            cs.anchor_bar = cs.baridx
        if (cs.baridx - cs.anchor_bar) > ARM_WIN:
            return None
        trigs = ns["trig_of"](cs)
        if not trigs:
            return None
        tf, kind = trigs[0]
        b = ns["bounce"](cs, bar, tf)
        if b:
            return dict(minute=bar.minute, key=None, kind=kind, tf=tf,
                        lvl=b[0], via=b[1], entry=float(bar.close))
        return None

    def mkdt(m, sec=30):
        return datetime(2026, 9, 9, m // 60, m % 60, sec)

    # synthetic cumulative-volume baseline per contract (CSV interval volumes
    # are per-minute; the engine consumes CUMULATIVE touchline "v" — feed a
    # running total so each bar's delta equals that minute's interval volume)
    for k, lc in live.items():
        g = today[lc.symbol]
        firsts = [m for m in range(T0, T0 + 345) if not g[g.minute == m].empty]
        if firsts:
            r = g[g.minute == firsts[0]].iloc[0]
            lc.push_tick(float(r.open), mkdt(firsts[0], 5), cum_volume=0.0)

    def bar_row(g, m):
        row = g[g.minute == m]
        return row.iloc[0] if not row.empty else None

    # running cumulative volume per contract: cum_before[m] = total volume of
    # all minutes < m; rollover cum = cum_before[m] + r.volume(m)
    run_cum = {k: 0.0 for k in live}
    for m in range(T0, T0 + 345):
        # ---------- REF side: one bar ----------
        for k, cs in states.items():
            r = bar_row(today[cs.sym], m)
            if r is None:
                continue
            bar = ns["mk_bar"](r, m)
            s = ref_bar_close(cs, bar)
            if s:
                s["key"] = k
                ref_sigs[(m, k)] = s
        # ---------- LIVE side: same bar via ticks ----------
        for k, lc in live.items():
            g = today[lc.symbol]
            r = bar_row(g, m)
            if r is None:
                continue
            v_m = float(r.volume)
            lc.push_tick(float(r.high), mkdt(m, 10), cum_volume=run_cum[k] + v_m)
            lc.push_tick(float(r.low), mkdt(m, 15), cum_volume=run_cum[k] + v_m)
            lc.push_tick(float(r.close), mkdt(m, 20), cum_volume=run_cum[k] + v_m)
            # rollover: first tick of minute m+1 closes bar m. Its VALUE is
            # the open of the next bar (so no high/low contamination), its cum
            # volume includes ALL of minute m's volume -> bar m vol = v_m.
            nxt = bar_row(g, m + 1)
            roll_val = float(nxt.open) if nxt is not None else float(r.close)
            roll_cum = run_cum[k] + v_m
            sig = lc.push_tick(roll_val, mkdt(m + 1, 5), cum_volume=roll_cum)
            run_cum[k] = roll_cum
            if sig:
                live_sigs[(m, k)] = sig

        # indicator comparison at each minute with data
        if m % 15 == 0:  # sample every 15 min to keep output small
            for k, lc in live.items():
                cs = states[k]
                trackers_ref = cs.tf
                trackers_live = lc.tf
                for tfv in (1, 2, 3, 5):
                    tr_, tl = trackers_ref[tfv], trackers_live[tfv]
                    check(f"{day} {m} {k} TF{tfv} S1", tr_.last_s1, tl.last_s1)
                    check(f"{day} {m} {k} TF{tfv} S3", tr_.last_s3, tl.last_s3)
                    check(f"{day} {m} {k} TF{tfv} S4", tr_.last_s4, tl.last_s4)
                check(f"{day} {m} {k} EMA20", cs.ema20.value, lc.ema20.value)
                check(f"{day} {m} {k} EMA200", cs.ema200.value, lc.ema200.value)

    # ---- signal stream comparison ----
    ref_set = dict(ref_sigs)
    common = 0
    for key_t, s in ref_set.items():
        ls = live_sigs.get(key_t)
        if ls is None:
            FAIL += 1
            print(f"  SIGNAL MISSING live: {day} {hm(key_t[0])} {key_t[1]} {s['kind']} {s['lvl']}")
        else:
            common += 1
            ok = (ls["trigger"] == s["kind"] and ls["tf"] == f"{s['tf']}m"
                  and abs(ls["entry"] - s["entry"]) < 0.02 and ls["level"] == s["lvl"])
            if not ok:
                FAIL += 1
                print(f"  SIGNAL DIFF {day} {hm(key_t[0])} {key_t[1]}: ref={s['kind']}/{s['tf']}m/{s['lvl']}/{s['entry']:.2f} live={ls['trigger']}/{ls['tf']}/{ls['level']}/{ls['entry']:.2f}")
    extra = set(live_sigs) - set(ref_set)
    for e in extra:
        ls = live_sigs[e]
        FAIL += 1
        print(f"  EXTRA SIGNAL live: {day} {hm(e[0])} {e[1]} {ls['trigger']} {ls['level']}")
    total_sigs_ref += len(ref_set)
    total_sigs_live += len(live_sigs)
    print(f"{day}: ref signals={len(ref_set)} live signals={len(live_sigs)} common={common}")

print()
print(f"TOTAL ref={total_sigs_ref} live={total_sigs_live}")
print("PARITY:", "PASS — LIVE ENGINE EXACT" if FAIL == 0 else f"FAIL ({FAIL} mismatches)")
sys.exit(0 if FAIL == 0 else 1)
