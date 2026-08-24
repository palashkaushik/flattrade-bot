import json

out = {}
for t, lab in [(0, "NO_FILTER"), (5, "5M_HA_UTBOT"), (15, "15M_HA_UTBOT")]:
    d = json.load(open(f"optimus_hft_trend{t}.json"))
    top = d["top5"][0]
    out[lab] = {
        "nw_net": top["nw"]["net_rs"], "dd": top["nw"]["max_dd"], "pf": top["nw"]["pf"],
        "trades": top["nw"]["trades"], "per_day": top["nw"]["trades"] / 1574,
        "wr": top["nw"]["win_rate"], "oos_net": top["oos"]["net_rs"], "oos_wr": top["oos"]["win_rate"],
        "params": top["params"], "all_months_positive": top["all_months_positive"],
    }
    print(f"{lab:14s} net=Rs{top['nw']['net_rs']:>11,.0f}  PF={top['nw']['pf']:.2f}  "
          f"{top['nw']['trades']/1574:.2f}/day  WR={top['nw']['win_rate']:.1f}%  "
          f"OOS=Rs{top['oos']['net_rs']:>10,.0f} WR={top['oos']['win_rate']:.1f}%")
json.dump(out, open("optimus_hft_trend_comparison.json", "w"), indent=2)
print("Saved optimus_hft_trend_comparison.json")
