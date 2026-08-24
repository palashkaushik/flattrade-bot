"""
MARNI VSA ENGINE — OFFICIAL TP = 0.290 (0.496x Option Span) ON AUG 12, 13, 14 (2026)
====================================================================================
Geometry:
  - Entry: Golden Pocket (0.786 / Volume Spike Trigger)
  - Target TP: 0.290 Fibonacci Level -> Entry + (0.496 * Option Span)
  - Stop Loss SL: 1.155 Extension Level -> Entry - (0.369 * Option Span)
  - Risk-to-Reward: 0.496 / 0.369 = 1.344 : 1
"""

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(r"c:\Websites\FLATTRADE BOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_walkforward_fees import BROKERAGE_PER_ORDER, SLIPPAGE_PTS, trade_cost

LOT_SIZE = 65

base_trades = [
    # August 12
    {"date": "2026-08-12", "time": "10:35", "side": "PE", "symbol": "NIFTY 24500 PE", "strike": 24500, "span": 49.50, "opt_span": 24.75, "entry": 164.20, "entry_m": 635, "range": "10:02 – 10:28"},
    {"date": "2026-08-12", "time": "11:44", "side": "PE", "symbol": "NIFTY 24400 PE", "strike": 24400, "span": 20.20, "opt_span": 10.10, "entry": 159.45, "entry_m": 704, "range": "11:32 – 11:41"},
    {"date": "2026-08-12", "time": "12:34", "side": "PE", "symbol": "NIFTY 24400 PE", "strike": 24400, "span": 28.50, "opt_span": 14.25, "entry": 157.30, "entry_m": 754, "range": "12:12 – 12:28"},
    # August 13
    {"date": "2026-08-13", "time": "10:49", "side": "PE", "symbol": "NIFTY 24450 PE", "strike": 24450, "span": 34.80, "opt_span": 17.40, "entry": 143.60, "entry_m": 649, "range": "10:15 – 10:32"},
    {"date": "2026-08-13", "time": "11:27", "side": "PE", "symbol": "NIFTY 24450 PE", "strike": 24450, "span": 33.30, "opt_span": 16.65, "entry": 145.55, "entry_m": 687, "range": "10:53 – 11:09"},
    {"date": "2026-08-13", "time": "12:51", "side": "CE", "symbol": "NIFTY 24250 CE", "strike": 24250, "span": 66.60, "opt_span": 33.30, "entry": 219.65, "entry_m": 771, "range": "11:47 – 12:09"},
    {"date": "2026-08-13", "time": "13:41", "side": "PE", "symbol": "NIFTY 24450 PE", "strike": 24450, "span": 24.00, "opt_span": 12.00, "entry": 106.75, "entry_m": 821, "range": "13:17 – 13:37"},
    {"date": "2026-08-13", "time": "15:13", "side": "PE", "symbol": "NIFTY 24450 PE", "strike": 24450, "span": 31.65, "opt_span": 15.83, "entry": 106.75, "entry_m": 913, "range": "14:43 – 14:59"},
    # August 14
    {"date": "2026-08-14", "time": "09:47", "side": "PE", "symbol": "NIFTY 24450 PE", "strike": 24450, "span": 57.60, "opt_span": 28.80, "entry": 134.40, "entry_m": 587, "range": "09:15 – 09:32"},
    {"date": "2026-08-14", "time": "10:55", "side": "PE", "symbol": "NIFTY 24450 PE", "strike": 24450, "span": 62.40, "opt_span": 31.20, "entry": 138.35, "entry_m": 655, "range": "09:38 – 10:48"},
    {"date": "2026-08-14", "time": "11:43", "side": "PE", "symbol": "NIFTY 24450 PE", "strike": 24450, "span": 28.00, "opt_span": 14.00, "entry": 139.65, "entry_m": 703, "range": "11:06 – 11:41"},
    {"date": "2026-08-14", "time": "14:17", "side": "CE", "symbol": "NIFTY 24350 CE", "strike": 24350, "span": 27.80, "opt_span": 13.90, "entry": 162.20, "entry_m": 857, "range": "13:30 – 14:14"},
]

def parse_time_min(t_str):
    if " " in t_str:
        t_str = t_str.split(" ")[1]
    parts = t_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])

day_caches = {}
for d_str in ["2026-08-12", "2026-08-13", "2026-08-14"]:
    cp = ROOT / "artifacts" / "flattrade_day_cache" / f"{d_str}.json.gz"
    with gzip.open(cp, "rt", encoding="utf-8") as f:
        day_caches[d_str] = json.load(f)

def simulate_trade_029(t):
    d_str = t["date"]
    contracts = day_caches[d_str]["contracts"]
    c_key = f"{t['side']}:{t['strike']}"
    rows = contracts[c_key]["rows"]
    
    bars = {}
    for r in rows:
        m = parse_time_min(r["time"])
        bars[m] = {
            "time": r["time"].split(" ")[1][:5] if " " in r["time"] else r["time"][:5],
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        }
        
    entry_m = t["entry_m"]
    opt_entry = t["entry"]
    opt_span = t["opt_span"]
    
    # Official 0.290 TP Level: Gain = 0.496 * Option Span
    tp_price = opt_entry + (opt_span * 0.496)
    # Stop Loss SL: 1.155 Ext -> Loss = 0.369 * Option Span
    sl_price = opt_entry - (opt_span * 0.369)
    
    exit_fill, exit_m, exit_t, rsn = None, None, "", ""
    
    for bar_m in range(entry_m + 1, 930 + 1):
        if bar_m not in bars:
            continue
        b = bars[bar_m]
        h, l, cl = b["high"], b["low"], b["close"]
        
        if l <= sl_price and h >= tp_price:
            exit_fill, exit_m, exit_t, rsn = sl_price, bar_m, b["time"], "SL"
            break
        elif h >= tp_price:
            exit_fill, exit_m, exit_t, rsn = tp_price, bar_m, b["time"], "TP (0.29)"
            break
        elif l <= sl_price:
            exit_fill, exit_m, exit_t, rsn = sl_price, bar_m, b["time"], "SL (1.155)"
            break
        elif bar_m >= 900:
            exit_fill, exit_m, exit_t, rsn = cl, bar_m, b["time"], "EOD"
            break
            
    pts = round(exit_fill - opt_entry, 2)
    fee = trade_cost(opt_entry, exit_fill, BROKERAGE_PER_ORDER)
    rs_net = round(pts * LOT_SIZE - fee, 2)
    return {
        "date": d_str,
        "time": t["time"],
        "side": t["side"],
        "symbol": t["symbol"],
        "span": t["span"],
        "opt_span": opt_span,
        "range": t["range"],
        "entry": opt_entry,
        "tp": tp_price,
        "sl": sl_price,
        "exit": exit_fill,
        "exit_time": exit_t,
        "reason": rsn,
        "points": pts,
        "fee": fee,
        "rs_net": rs_net,
    }

results = [simulate_trade_029(t) for t in base_trades]

by_day = {"2026-08-12": [], "2026-08-13": [], "2026-08-14": []}
for r in results:
    by_day[r["date"]].append(r)

def print_day_report(title, t_list):
    print(f"\n{'='*155}")
    print(f"{title} (TP = 0.290 FIB LEVEL | GAIN = 0.496x OPTION SPAN)")
    print(f"{'='*155}")
    print(f"{'Time':7s} | {'Side':4s} | {'Symbol':18s} | {'Impulse Time':15s} | {'Idx Span':9s} | {'Opt Span':9s} | {'Entry':8s} | {'TP (0.29)':9s} | {'SL (1.155)':10s} | {'Exit':8s} | {'Exit Time':9s} | {'Rsn':12s} | {'Points':8s} | {'Net Realized Rs':15s}")
    print("-" * 170)
    
    tot_pts = 0.0
    tot_rs = 0.0
    wins = 0
    
    for t in t_list:
        pts = t["points"]
        net_rs = t["rs_net"]
        tot_pts += pts
        tot_rs += net_rs
        if pts > 0: wins += 1
        print(f"{t['time']:7s} | {t['side']:4s} | {t['symbol']:18s} | {t['range']:15s} | {t['span']:8.1f}p | {t['opt_span']:8.1f}p | {t['entry']:8.2f} | {t['tp']:9.2f} | {t['sl']:10.2f} | {t['exit']:8.2f} | {t['exit_time']:9s} | {t['reason']:12s} | {pts:+7.2f} | Rs {net_rs:+12.2f}")
        
    wr = (wins / len(t_list)) * 100 if t_list else 0
    print("-" * 170)
    print(f"TRADES: {len(t_list)} | WIN RATE: {wr:.1f}% ({wins}/{len(t_list)}) | TOTAL POINTS: {tot_pts:+8.2f} pts | NET PROFIT: Rs {tot_rs:+12.2f}\n")
    return tot_pts, tot_rs, len(t_list), wins

p12, r12, n12, w12 = print_day_report("MARNI VSA (TP=0.290) — AUGUST 12, 2026", by_day["2026-08-12"])
p13, r13, n13, w13 = print_day_report("MARNI VSA (TP=0.290) — AUGUST 13, 2026", by_day["2026-08-13"])
p14, r14, n14, w14 = print_day_report("MARNI VSA (TP=0.290) — AUGUST 14, 2026", by_day["2026-08-14"])

tot_p = p12 + p13 + p14
tot_r = r12 + r13 + r14
tot_n = n12 + n13 + n14
tot_w = w12 + w13 + w14

print(f"{'='*155}")
print(f"MARNI VSA 3-DAY COMBINED (OFFICIAL TP = 0.290): {tot_n} Trades | {tot_w} Wins ({(tot_w/tot_n)*100:.1f}% WR) | {tot_p:+8.2f} pts | NET REALIZED PROFIT: Rs {tot_r:+12.2f}")
print(f"{'='*155}")
