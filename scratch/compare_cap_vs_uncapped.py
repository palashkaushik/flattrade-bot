import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"c:\Websites\FLATTRADE BOT")

with open(ROOT / "artifacts" / "f6_hybrid" / "marni_vsa_tf_pairs_comparison.json", "r") as f:
    capped_data = json.load(f)

print(f"\n{'='*130}")
print(f"MARNI VSA ENGINE: WITH +/- 30 PT DAILY CAP vs WITHOUT CAP (7-YEAR MULTI-YEAR COMPARISON)")
print(f"{'='*130}")

# Uncapped baseline numbers from task-2662:
# 1m/15m: Trades 3334, WR 42.6%, Net Pts -6499.28p, Net Rs -474,881.34, PF 0.66, DD 478,006.11
# 1m/5m:  Trades 5016, WR 41.5%, Net Pts -11370.76p, Net Rs -816,492.03, PF 0.62, DD 816,746.80
# 3m/15m: Trades 1047, WR 43.2%, Net Pts -1720.10p, Net Rs -127,759.97, PF 0.78, DD 140,041.12
# 3m/5m:  Trades 1805, WR 41.3%, Net Pts -4890.75p, Net Rs -344,938.65, PF 0.67, DD 345,665.80
# 5m/15m: Trades 616,  WR 40.6%, Net Pts -1311.77p, Net Rs -94,759.92,   PF 0.76, DD 99,827.75
# 5m/30m: Trades 379,  WR 37.2%, Net Pts -1611.17p, Net Rs -110,825.03,  PF 0.59, DD 121,607.26

uncapped = {
    "1m_15m": {"trades": 3334, "wr": 42.6, "pts": -6499.28, "rs": -474881.34, "pf": 0.66, "dd": 478006.11},
    "1m_5m":  {"trades": 5016, "wr": 41.5, "pts": -11370.76, "rs": -816492.03, "pf": 0.62, "dd": 816746.80},
    "3m_15m": {"trades": 1047, "wr": 43.2, "pts": -1720.10, "rs": -127759.97, "pf": 0.78, "dd": 140041.12},
    "3m_5m":  {"trades": 1805, "wr": 41.3, "pts": -4890.75, "rs": -344938.65, "pf": 0.67, "dd": 345665.80},
    "5m_15m": {"trades": 616,  "wr": 40.6, "pts": -1311.77, "rs": -94759.92,   "pf": 0.76, "dd": 99827.75},
    "5m_30m": {"trades": 379,  "wr": 37.2, "pts": -1611.17, "rs": -110825.03,  "pf": 0.59, "dd": 121607.26},
}

print(f"{'Timeframe Bias Pair':35s} | {'Uncapped Net P&L':18s} | {'+/- 30pt Cap Net P&L':20s} | {'P&L Difference':16s} | {'Trades Pruned':15s} | {'Drawdown Saved':16s}")
print("-" * 130)

for k, cap_info in capped_data.items():
    lbl = cap_info["label"]
    c_st = cap_info["stats"]
    u_st = uncapped.get(k, {})
    
    u_rs = u_st.get("rs", 0.0)
    c_rs = c_st["net_rs"]
    diff_rs = c_rs - u_rs
    
    u_trades = u_st.get("trades", 0)
    c_trades = c_st["trades"]
    trades_pruned = u_trades - c_trades
    
    u_dd = u_st.get("dd", 0.0)
    c_dd = c_st["max_drawdown_rs"]
    dd_saved = u_dd - c_dd
    
    print(f"{lbl:35s} | Rs {u_rs:+14,.2f} | Rs {c_rs:+16,.2f} | Rs {diff_rs:+12,.2f} | {trades_pruned:5d} trades ({trades_pruned/u_trades*100:4.1f}%) | Rs {dd_saved:+12,.2f}")

print("-" * 130)
