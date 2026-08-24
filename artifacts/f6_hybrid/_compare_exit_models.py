import json

old = json.load(open("artifacts/f6_hybrid/smart_fib_wide_target_sweep_fixed40.json", encoding="utf-8"))
new = json.load(open("artifacts/f6_hybrid/smart_fib_wide_target_sweep_fixed40_closetp.json", encoding="utf-8"))

old_by = {(r["params"]["target_level"], r["params"]["stop_level"]): r for r in old["full_window_ranked"]}
new_by = {(r["params"]["target_level"], r["params"]["stop_level"]): r for r in new["full_window_ranked"]}

print(f"{'TARGET':<8}{'STOP':<7}{'OLD tr':>8}{'NEW tr':>8}{'OLD pts':>11}{'NEW pts':>11}{'DELTA':>9}{'OLD avg/t':>10}{'NEW avg/t':>10}")
for key in sorted(old_by):
    o, n = old_by[key], new_by[key]
    avg_o = o["net_points"] / o["trades"]
    avg_n = n["net_points"] / n["trades"]
    print(
        f"{key[0]:<8}{key[1]:<7}{o['trades']:>8}{n['trades']:>8}"
        f"{o['net_points']:>+11.2f}{n['net_points']:>+11.2f}{n['net_points']-o['net_points']:>+9.2f}"
        f"{avg_o:>10.4f}{avg_n:>10.4f}"
    )

print()
print("=== NEW model: per-trade economics of new top configs ===")
for r in new["full_window_ranked"][:3]:
    p = r["params"]
    avg = r["net_points"] / r["trades"]
    wr = r["win_rate"] / 100.0
    pf = r["profit_factor"]
    win_loss_ratio = pf * (1 - wr) / wr
    avg_loss = avg / (wr * win_loss_ratio - (1 - wr))
    avg_win = win_loss_ratio * avg_loss
    print(
        f"t={p['target_level']} s={p['stop_level']}: tr={r['trades']} WR={r['win_rate']:.2f}% "
        f"avg_win={avg_win:.2f} pts avg_loss={avg_loss:.2f} pts avg_net={avg:.4f} pts/trade"
    )