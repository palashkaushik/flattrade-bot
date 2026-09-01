"""
WALK-FORWARD VALIDATION of the §41 seeded-mode candidates (GPU, causal).

Analysis A — Year-by-year consistency: each candidate's per-year net/WR/DD.
A real edge must be green across most years, not carried by one regime.

Analysis B — TRUE walk-forward selection: at each year boundary Y, the config
is chosen ONLY from prior years' seeded-sweep results (max net), then applied
UNCHANGED to the unseen year Y. The selection grid = full §41 grid (200 cfgs)
re-run here in seeded mode, trades bucketed by year. No look-ahead: year Y
metrics never touch years <= Y-1 selections.

Single GPU cost: run the grid ONCE, bucket trades by year host-side, then do
selection/aggregation in numpy — the sim is day-independent, so per-year
bucketing of a full-range run is exactly equivalent to running each year
separately (same state: days are independent in the sim core).
"""
import sys, time, os
sys.path.insert(0, r'C:\Websites\FLATTRADE BOT')
os.environ['LH_BIAS'] = '0'
# CPU hard cap: 8 workers everywhere (before any numeric import binds threads)
os.environ['NUMBA_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['POLARS_MAX_THREADS'] = '8'

import numpy as np
import torch

# GPU utilization: keep the device saturated — batch chunks sized so the
# eager core's (B, D) state tensors fill SMs on the RTX 3060 without
# spilling VRAM. 345-bar loop x 64-128 configs/launch is the sweet spot.
torch.set_float32_matmul_precision('high')
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

t0 = time.time()
import seeded_lib as SL        # builds master + GPU state + seeded tensors

import gpu_sim_last_hope as G

# Re-apply the seeded patches (seeded_lib already patched G's globals at import)
# Rebuild the FULL grid identical to §41 so selection matches the published sweep
CHAMPION = dict(SL.CHAMPION_BASE)
CHAMPION.update(kind='B', arm_window=10, atr_period=10, atr_mult=1.5,
                cap=0, be_trigger=0.70, be_buffer=1.0, tp_frac=1.0,
                touch_buffer=0.0, entry_start=0)

TOUCH_BUFFERS_SWEEP = [0.0, 0.5, 1.0]
grid = []
for arm in [5, 10, 15, 20]:
    for atr_p in [10, 14]:
        for atr_m in [1.25, 1.5, 2.0, 2.5]:
            for tb in TOUCH_BUFFERS_SWEEP:
                for be_t in [0.50, 0.70]:
                    cfg = dict(CHAMPION)
                    cfg.update(arm_window=arm, atr_period=atr_p, atr_mult=atr_m,
                               touch_buffer=tb, be_trigger=be_t, be_buffer=1.0,
                               entry_end=SL.T1, entry_start=0)
                    grid.append(cfg)
for arm in [10, 15]:
    for atr_m in [1.5, 2.0]:
        for tb in [0.0, 0.5]:
            cfg = dict(CHAMPION)
            cfg.update(arm_window=arm, atr_period=10, atr_mult=atr_m,
                       touch_buffer=tb, be_trigger=0.70, entry_start=75,
                       entry_end=SL.T1)
            grid.append(cfg)

print(f"[wf] grid: {len(grid)} configs (identical to §41)")

# Run the whole grid in seeded mode, one batch per touch-buffer
t_sweep = time.time()
all_results = []   # (cfg, trades_list)
for tb in TOUCH_BUFFERS_SWEEP:
    G.bounce_pe_stack = torch.stack([SL.bounce_pe_seeds[tb]] * len(G.TOUCH_BUFFERS), 0)
    G.bounce_ce_stack = torch.stack([SL.bounce_ce_seeds[tb]] * len(G.TOUCH_BUFFERS), 0)
    cfgs = [c for c in grid if c['touch_buffer'] == tb]
    CHUNK = 128   # full GPU saturation per launch (was 64)
    for i in range(0, len(cfgs), CHUNK):
        batch = cfgs[i:i + CHUNK]
        trades = G._eager_sim_core(batch)
        for cfg, tr in zip(batch, trades):
            all_results.append((cfg, tr))
print(f"[wf] seeded grid run in {time.time()-t_sweep:.0f}s")

# ---------------------------------------------------------------------------
# Analysis A: per-year consistency for the 4 named candidates
# ---------------------------------------------------------------------------
YEARS = sorted({str(d)[:4] for d in SL.trading_days})
print(f"\n[years] {YEARS}")

def cfg_key(c):
    return (c['arm_window'], c['atr_period'], c['atr_mult'], c['touch_buffer'], c['be_trigger'])

# Index grid results by key
res_by_key = {cfg_key(c): tr for c, tr in all_results if c.get('entry_start', 0) == 0}

print("\n" + "=" * 100)
print("ANALYSIS A — PER-YEAR CONSISTENCY (seeded mode)")
print("=" * 100)
header = f"{'config':<22} " + " ".join(f"{y:>10}" for y in YEARS) + f" | {'TOTAL':>12}"
print(header)
for cand in SL.CANDIDATES:
    key = (cand['arm_window'], cand['atr_period'], cand['atr_mult'],
           cand['touch_buffer'], cand['be_trigger'])
    tr = res_by_key[key]
    by_year = SL.trades_by_year(tr)
    row = f"{cand['label']:<22} "
    for y in YEARS:
        m = SL.year_metrics(by_year.get(y, []))
        row += f"{m['net']:>10,.0f} "
    total = SL.year_metrics(tr)
    row += f"| {total['net']:>12,.0f}"
    print(row)
    # per-year detail line: WR
    row2 = f"{'  (wr%)':<22} "
    for y in YEARS:
        m = SL.year_metrics(by_year.get(y, []))
        row2 += f"{m['wr']:>10.1f} "
    print(row2)

# ---------------------------------------------------------------------------
# Analysis B: TRUE walk-forward selection
#   For each year Y (from the 2nd onward): select max-net config using only
#   years < Y; apply to year Y. Report the OOS stitched equity.
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("ANALYSIS B — TRUE WALK-FORWARD SELECTION (select on past, trade next year)")
print("=" * 100)

grid_keys = list(res_by_key.keys())
# Precompute per-config per-year nets
per_cfg_year_nets = {}
for key, tr in res_by_key.items():
    by_year = SL.trades_by_year(tr)
    per_cfg_year_nets[key] = {y: sum(t[5] for t in by_year.get(y, [])) for y in YEARS}

wf_rows = []
for yi in range(1, len(YEARS)):
    y = YEARS[yi]
    prior_years = YEARS[:yi]
    # selection: max total net over prior years (with min-trades floor)
    best_key, best_prior_net = None, -float('inf')
    for key in grid_keys:
        prior_net = sum(per_cfg_year_nets[key].get(py, 0.0) for py in prior_years)
        tr = res_by_key[key]
        if len(tr) < 500:
            continue
        if prior_net > best_prior_net:
            best_prior_net = prior_net
            best_key = key
    oos_net = per_cfg_year_nets[best_key].get(y, 0.0)
    tr = res_by_key[best_key]
    by_year = SL.trades_by_year(tr)
    oos_m = SL.year_metrics(by_year.get(y, []))
    wf_rows.append((y, best_key, oos_net, oos_m))
    print(f"YEAR {y}: selected arm{best_key[0]}/atr{best_key[1]}/x{best_key[2]}/tb{best_key[3]}/be{best_key[4]} "
          f"(prior-net Rs {best_prior_net:>12,.0f}) -> OOS net Rs {oos_net:>10,.0f} | "
          f"wr {oos_m['wr']:.1f}% | trades {oos_m['trades']} | worst-day Rs {oos_m['worst_day']:,.0f}")

oos_total = sum(r[2] for r in wf_rows)
print(f"\nWALK-FORWARD OOS TOTAL (2021..{YEARS[-1]}): Rs {oos_total:,.0f}")

# WF equity curve maxDD on stitched OOS years
oos_trades = []
for y, key, oos_net, oos_m in wf_rows:
    oos_trades.extend(SL.trades_by_year(res_by_key[key]).get(y, []))
wf_m = SL.year_metrics(oos_trades)
print(f"WF OOS stitched: trades={wf_m['trades']} wr={wf_m['wr']:.1f}% net=Rs {wf_m['net']:,.0f} "
      f"maxDD=Rs {wf_m['max_dd']:,.0f} worst-day=Rs {wf_m['worst_day']:,.0f} calmar={wf_m['calmar']:.1f}")

# ---------------------------------------------------------------------------
# Analysis C: the 3 headline candidates evaluated year-by-year as FIXED configs
# (what you'd actually trade — no selection, just robustness view)
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("ANALYSIS C — FIXED-CANDIDATE YEAR-BY-YEAR (full metrics)")
print("=" * 100)
for cand in SL.CANDIDATES:
    key = (cand['arm_window'], cand['atr_period'], cand['atr_mult'],
           cand['touch_buffer'], cand['be_trigger'])
    tr = res_by_key[key]
    by_year = SL.trades_by_year(tr)
    print(f"\n--- {cand['label']} (arm{cand['arm_window']} atr{cand['atr_period']}x{cand['atr_mult']} tb{cand['touch_buffer']} be{cand['be_trigger']}) ---")
    print(f"{'year':<6} {'trades':>7} {'wr%':>6} {'net':>12} {'maxDD':>9} {'worst-day':>11}")
    for y in YEARS:
        m = SL.year_metrics(by_year.get(y, []))
        print(f"{y:<6} {m['trades']:>7} {m['wr']:>6.1f} {m['net']:>12,.0f} {m['max_dd']:>9,.0f} {m['worst_day']:>11,.0f}")
    tm = SL.year_metrics(tr)
    print(f"{'TOTAL':<6} {tm['trades']:>7} {tm['wr']:>6.1f} {tm['net']:>12,.0f} {tm['max_dd']:>9,.0f} {tm['worst_day']:>11,.0f}  calmar={tm['calmar']:.1f}")

print(f"\n[total] {time.time()-t0:.0f}s")
