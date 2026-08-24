"""Optimus Backtest — regression guard for SPEED and ACCURACY.

Anchors accuracy to the verified golden values from the hardening sessions and
enforces a speed budget. Run after any change to optimized_gpu_backtest.py or
cross_strategy_ensemble_gpu.py. Exit code 0 = no regression.

NOTE: HALF (mixed precision) is measured in an ISOLATED subprocess because
enable_half() mutates STOCH_CACHE in place (no disable), so fp32 runs must
happen before any HALF toggle in-process.
"""
import sys, time, json, torch, subprocess, os
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT\artifacts\f6_hybrid")
import optimized_gpu_backtest as base
import cross_strategy_ensemble_gpu as ens

HERE = r"C:\Websites\FLATTRADE BOT\artifacts\f6_hybrid"
BASE_B07 = dict(timeframe=3, s1_k=7, s4_k=50, s1_os=25.0, s4_ob=70.0, atr_p=10,
                sl_m=1.5, tp_m=5.0, daily_loss_pts=8, daily_profit_pts=50, moneyness=0.5,
                max_trade_loss_rs=1500, sess_start_off=0, sess_end_off=30, sess_end=315)
fails = []
def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)

print("="*72)
print("OPTIMUS BACKTEST - REGRESSION (accuracy + speed)")
print("="*72)

# ---------- SPEED (fp32, measured BEFORE any HALF toggle) ----------
print("\n-- SPEED (fp32) --")
ps100 = [dict(BASE_B07) for _ in range(100)]
t0 = time.time(); _ = base.evaluate_batch("B07", ps100, None); t_batch = time.time() - t0
t0 = time.time(); _ = base.evaluate_batch("B07", [BASE_B07], None); t_single = time.time() - t0
t0 = time.time(); par = ens.ensemble_parity_check(); t_parity = time.time() - t0
check("base evaluate_batch B=100 < 2.0s", t_batch < 2.0, f"{t_batch:.3f}s")
check("ensemble parity full < 120s", t_parity < 120, f"{t_parity:.1f}s")
print(f"     B=1={t_single:.3f}s  B=100={t_batch:.3f}s  parity={t_parity:.1f}s")

# ---------- ACCURACY: base engine golden anchors (fp32) ----------
print("\n-- ACCURACY: base engine --")
r = base.evaluate_batch("B07", [BASE_B07], None)[0]
# NOTE: golden values recomputed on the FIXED _finalize (2026-08-16). The old
# values (+429193.5 / T=678 / PF 4.13) were produced by the buggy engine, which
# truncated the whole backtest at the first daily-cap breach. The fixed engine
# honours the daily circuit breaker correctly (halts only the breaching day,
# counts the breach trade, resumes next day) -> honest, lower numbers.
check("B07 3m NW trades==1821", r["trades"] == 1821, f"T={r['trades']}")
check("B07 3m NW net==-331702.5", abs(r["net_rs"] - -331702.5) < 1.0, f"net={r['net_rs']:.1f}")
check("B07 3m NW pf==0.68", abs(r["pf"] - 0.68) < 0.02, f"pf={r['pf']:.2f}")

# ---------- ACCURACY: batching bit-exact ----------
print("\n-- ACCURACY: batching --")
ps = [dict(BASE_B07),
      dict(BASE_B07, timeframe=1, s1_k=12, s4_k=40, s1_os=20.0, s4_ob=80.0, atr_p=14, sl_m=1.2, tp_m=3.0),
      dict(BASE_B07, timeframe=2, s1_k=10, s4_k=50, s1_os=25.0, s4_ob=70.0, atr_p=14, sl_m=2.0, tp_m=4.0)]
b3 = base.evaluate_batch("B07", ps, None)
s1 = [base.evaluate_batch("B07", [p], None)[0] for p in ps]
ok_b = all(b3[i]["trades"] == s1[i]["trades"] and abs(b3[i]["net_rs"] - s1[i]["net_rs"]) < 1e-6 for i in range(3))
check("B=3 == 3xB=1 (trades+net exact)", ok_b, f"B3={[x['trades'] for x in b3]} S1={[x['trades'] for x in s1]}")

# ---------- ACCURACY: ensemble parity (meta path == base) ----------
print("\n-- ACCURACY: ensemble parity --")
check("ensemble parity PASS", par)

# ---------- ACCURACY: fp16 mixed precision (ISOLATED subprocess) ----------
print("\n-- ACCURACY: fp16 mixed precision (subprocess) --")
fp16_out = subprocess.run(
    [sys.executable, "-c",
     "import sys,json; sys.path.insert(0,r'C:\Websites\FLATTRADE BOT\artifacts\f6_hybrid');"
     "import os; os.environ['HALF']='1';"
     "import optimized_gpu_backtest as base;"
     "p=dict(timeframe=3,s1_k=7,s4_k=50,s1_os=25.0,s4_ob=70.0,atr_p=10,sl_m=1.5,tp_m=5.0,"
     "daily_loss_pts=8,daily_profit_pts=50,moneyness=0.5,max_trade_loss_rs=1500,"
     "sess_start_off=0,sess_end_off=30,sess_end=315);"
     "rr=base.evaluate_batch('B07',[p],None)[0];"
     "print(json.dumps({'trades':rr['trades'],'net':rr['net_rs']}))"],
    capture_output=True, text=True, cwd=HERE,
    env={**os.environ, "HALF": "1", "PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"})
try:
    rh = json.loads(fp16_out.stdout.strip().splitlines()[-1])
except Exception:
    rh = {"trades": -1, "net": 0}
    print("  fp16 subprocess error:", fp16_out.stderr[-300:])
check("fp16 trades within 2", abs(rh["trades"] - r["trades"]) <= 2, f"fp32={r['trades']} fp16={rh['trades']}")
check("fp16 net within 1.5%", abs(rh["net"] - r["net_rs"]) <= 0.015 * abs(r["net_rs"]),
      f"fp32={r['net_rs']:.1f} fp16={rh['net']:.1f}")

# ---------- persist measured snapshot ----------
snap = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_B07_NW": {"trades": r["trades"], "net": round(r["net_rs"], 1), "pf": round(r["pf"], 2)},
        "fp16_trades": rh["trades"], "fp16_net": round(rh["net"], 1),
        "speed_batch100_s": round(t_batch, 3), "speed_single_s": round(t_single, 3),
        "speed_parity_s": round(t_parity, 1)}
with open(HERE + r"\optimus_baseline.json", "w") as f:
    json.dump(snap, f, indent=2)
print(f"\n  baseline snapshot -> optimus_baseline.json")

print("\n" + "="*72)
print("REGRESSION:", "PASS" if not fails else f"FAIL ({len(fails)}): {fails}")
print("="*72)
sys.exit(1 if fails else 0)
