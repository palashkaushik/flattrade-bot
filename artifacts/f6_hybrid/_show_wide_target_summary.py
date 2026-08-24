import json

d = json.load(open("artifacts/f6_hybrid/smart_fib_wide_target_sweep_fixed40.json", encoding="utf-8"))
print("=== NON-WF ALL 30 (ranked) ===")
for r in d["full_window_ranked"]:
    p = r["params"]
    print(
        f"t={p['target_level']:<5} s={p['stop_level']:<5} "
        f"tr={r['trades']:<6} WR={r['win_rate']:>6.2f} "
        f"net={r['net_points']:>+10.2f} Rs={r['net_rs']:>+12,.2f} "
        f"DD={r['max_drawdown_points']:>8.2f} PF={r['profit_factor']:>8.4f}"
    )
print()
print("=== WFO FOLD SELECTIONS ===")
for w in d["train_selections"]:
    p = w["params"]
    print(
        f"fold={w['fold']} t={p['target_level']} s={p['stop_level']} "
        f"train_net={w['net_points']:+.2f} DD={w['max_drawdown_points']:.2f}"
    )
print()
print("=== WFO FOLD VALIDATIONS ===")
for v in d["folds"]:
    p = v["params"]
    print(
        f"fold={v['fold']} {v['validation_start']}..{v['validation_end']} "
        f"t={p['target_level']} s={p['stop_level']} tr={v['trades']} "
        f"net={v['net_points']:+.2f} Rs={v['net_rs']:+,.2f} "
        f"DD={v['max_drawdown_points']:.2f}"
    )
print()
s = d["stitched_oos"]
print(
    "STITCHED:",
    json.dumps(
        {k: s[k] for k in ("trades", "wins", "win_rate", "net_points", "net_rs", "max_drawdown_points", "fees_rs")}
    ),
)
