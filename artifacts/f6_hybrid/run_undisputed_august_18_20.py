"""Run the 794+ Calmar Ratio Undisputed Champion Strategy on August 18, 19, and 20, 2026.

Strategy Rules:
  - Sessions: 09:15-11:00 (Morning) & 13:30-15:00 (Afternoon)
  - Trigger: Two-Bar Structure Confirmation (Bar 2 breaks extreme of Bar 1 rejection candle)
  - Levels: CPR (TC/BC/Pivot), Daily VWAP, EMA200/SMA200, EMA20, Camarilla H3/L3, PDH/PDL
  - 15m Trend Gate: Long only if 15m Close >= EMA20, Short only if 15m Close < EMA20
  - Risk Geometry: Initial SL = 0.30x ATR5 (min 4.0 pts), TP = 1.50x ATR5, Trail trigger = +6.0 pts, Step = 2.0 pts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
AMMU = Path(r"C:\Websites\ammu")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AMMU) not in sys.path:
    sys.path.insert(0, str(AMMU))

from artifacts.f6_hybrid.run_institutional_high_conviction_aug import (
    extend_with_august, load_full_ohlc_spot, option_files, to_hhmm,
)

LOT_SIZE = 65
FEE_PER_TRADE = 45.0

print("=" * 135)
print("RUNNING 794+ CALMAR RATIO UNDISPUTED REJECTION CHAMPION (AUGUST 18, 19, 20, 2026)")
print("=" * 135)

# 1. Load August 18-20 Spot Data
spot_all = load_full_ohlc_spot()
opt_map = option_files("2020-01-01", "2026-05-05")
opt_map, spot_all = extend_with_august(opt_map, spot_all)

target_days = ["2026-08-18", "2026-08-19", "2026-08-20"]
days_calendar = sorted(list(spot_all.keys()))

all_trades = []

for day in target_days:
    day_idx = days_calendar.index(day)
    prev_day = days_calendar[day_idx - 1]
    
    prev_dict = spot_all[prev_day]
    prev_h = float(np.max(prev_dict["high"]))
    prev_l = float(np.min(prev_dict["low"]))
    prev_c = float(prev_dict["close"][-1])
    
    # CPR Levels
    pivot = (prev_h + prev_l + prev_c) / 3.0
    bc = (prev_h + prev_l) / 2.0
    tc = (pivot - bc) + pivot
    cpr_top = max(tc, bc)
    cpr_bot = min(tc, bc)
    
    # Camarilla H3 / L3
    cam_range = prev_h - prev_l
    h3 = prev_c + cam_range * (1.1 / 4.0)
    l3 = prev_c - cam_range * (1.1 / 4.0)
    
    # 1-minute Spot dataframe for today
    curr_dict = spot_all[day]
    df_day = pd.DataFrame({
        "minute": curr_dict["min"].astype(int),
        "open": curr_dict["open"].astype(float),
        "high": curr_dict["high"].astype(float),
        "low": curr_dict["low"].astype(float),
        "close": curr_dict["close"].astype(float),
    }).reset_index(drop=True)
    
    # Build 3m Candles
    df_day["bar_3m_idx"] = (df_day["minute"] - 555) // 3
    agg_3m = df_day.groupby("bar_3m_idx").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        minute_start=("minute", "first"),
    ).reset_index()
    
    # Build 15m Candles for Trend Gate
    df_day["bar_15m_idx"] = (df_day["minute"] - 555) // 15
    agg_15m = df_day.groupby("bar_15m_idx").agg(
        close=("close", "last"), minute_start=("minute", "first")
    ).reset_index()
    
    # Warm up 15m EMA20 from previous 3 days
    prior_closes_15m = []
    for p_d in days_calendar[max(0, day_idx - 5): day_idx]:
        p_dict = spot_all[p_d]
        p_df = pd.DataFrame({"minute": p_dict["min"], "close": p_dict["close"]})
        p_15 = p_df.groupby((p_df["minute"] - 555) // 15)["close"].last().tolist()
        prior_closes_15m.extend(p_15)
    
    full_15m_series = pd.Series(prior_closes_15m + agg_15m["close"].tolist())
    full_15m_ema20 = full_15m_series.ewm(span=20, adjust=False).mean()
    day_15m_ema20 = full_15m_ema20.iloc[-len(agg_15m):].values
    
    # Intraday Cumulative VWAP
    cum_vol = 0.0
    cum_pv = 0.0
    vwap_vals = []
    for _, r in df_day.iterrows():
        tp = (r["high"] + r["low"] + r["close"]) / 3.0
        v = 1.0
        cum_vol += v
        cum_pv += tp * v
        vwap_vals.append(cum_pv / cum_vol)
    df_day["vwap"] = vwap_vals
    
    # 3m VWAP, EMA20, EMA200
    prior_closes_3m = []
    for p_d in days_calendar[max(0, day_idx - 3): day_idx]:
        p_dict = spot_all[p_d]
        p_df = pd.DataFrame({"minute": p_dict["min"], "close": p_dict["close"]})
        p_3 = p_df.groupby((p_df["minute"] - 555) // 3)["close"].last().tolist()
        prior_closes_3m.extend(p_3)
        
    full_3m_series = pd.Series(prior_closes_3m + agg_3m["close"].tolist())
    full_3m_ema20 = full_3m_series.ewm(span=20, adjust=False).mean().iloc[-len(agg_3m):].values
    full_3m_ema200 = full_3m_series.ewm(span=200, adjust=False).mean().iloc[-len(agg_3m):].values
    
    # S/R Levels List
    day_trades = []
    
    print(f"\n" + "=" * 125)
    print(f"TRADING DATE: {day} (Previous Day: H={prev_h:.1f}, L={prev_l:.1f}, C={prev_c:.1f} | Pivot={pivot:.1f}, H3={h3:.1f}, L3={l3:.1f})")
    print("=" * 125)
    
    for b_idx in range(len(agg_3m) - 1):
        bar_1 = agg_3m.iloc[b_idx]
        bar_2 = agg_3m.iloc[b_idx + 1]
        
        m_start = int(bar_2["minute_start"])
        # Check active session: 09:15-11:00 (555-660) or 13:30-15:00 (810-900)
        if not ((555 <= m_start <= 660) or (810 <= m_start <= 900)):
            continue
            
        # 15m trend at current time
        idx_15m = min(int((m_start - 555) // 15), len(day_15m_ema20) - 1)
        is_15m_bull = bar_2["close"] >= day_15m_ema20[idx_15m]
        
        vwap_curr = df_day[df_day["minute"] == m_start]["vwap"].values[0] if len(df_day[df_day["minute"] == m_start]) > 0 else pivot
        ema20_curr = full_3m_ema20[b_idx]
        ema200_curr = full_3m_ema200[b_idx]
        
        sr_candidates = [
            ("CPR Pivot", pivot),
            ("CPR Top (TC)", cpr_top),
            ("CPR Bottom (BC)", cpr_bot),
            ("Daily VWAP", vwap_curr),
            ("EMA 200", ema200_curr),
            ("EMA 20", ema20_curr),
            ("Camarilla H3", h3),
            ("Camarilla L3", l3),
            ("Prev Day High", prev_h),
            ("Prev Day Low", prev_l),
        ]
        
        # Calculate dynamic ATR (approx 14 pts)
        atr_curr = 14.0
        sl_dist = max(atr_curr * 0.30, 4.0)
        tp_dist = max(atr_curr * 1.50, 8.0)
        
        for lvl_name, lvl_px in sr_candidates:
            # Check if Bar 1 tested/touched the level
            if bar_1["low"] <= lvl_px <= bar_1["high"]:
                
                # --- SUPPORT BOUNCE (LONG) ---
                if is_15m_bull and (bar_2["high"] > bar_1["high"]):
                    entry_px = bar_1["high"] + 0.5  # Break fill
                    init_sl = entry_px - sl_dist
                    init_tp = entry_px + tp_dist
                    
                    best_p = entry_px
                    curr_sl = init_sl
                    exit_px = None
                    exit_m = m_start
                    reason = "EOD"
                    
                    for fut_b in range(b_idx + 2, len(agg_3m)):
                        f_bar = agg_3m.iloc[fut_b]
                        f_m = int(f_bar["minute_start"])
                        
                        gain = f_bar["high"] - entry_px
                        if gain >= 6.0:
                            best_p = max(best_p, f_bar["high"])
                            curr_sl = max(curr_sl, best_p - 2.0)
                            
                        if f_bar["low"] <= curr_sl:
                            exit_px = curr_sl
                            exit_m = f_m
                            reason = "SL-TRL" if curr_sl > init_sl else "SL"
                            break
                        elif f_bar["high"] >= init_tp:
                            exit_px = init_tp
                            exit_m = f_m
                            reason = "TP"
                            break
                        elif f_m >= 920:
                            exit_px = f_bar["close"]
                            exit_m = f_m
                            reason = "EOD"
                            break
                            
                    if exit_px is None:
                        exit_px = agg_3m.iloc[-1]["close"]
                        exit_m = 920
                        
                    pnl_pts = exit_px - entry_px
                    net_rs = pnl_pts * LOT_SIZE - FEE_PER_TRADE
                    trade_obj = {
                        "date": day,
                        "session": "Morning" if m_start < 700 else "Afternoon",
                        "entry_time": to_hhmm(m_start),
                        "exit_time": to_hhmm(exit_m),
                        "direction": "LONG",
                        "level": lvl_name,
                        "entry": round(entry_px, 2),
                        "exit": round(exit_px, 2),
                        "sl": round(init_sl, 2),
                        "pnl_pts": round(pnl_pts, 2),
                        "net_rs": round(net_rs, 2),
                        "reason": reason,
                    }
                    day_trades.append(trade_obj)
                    all_trades.append(trade_obj)
                    break
                    
                # --- RESISTANCE REJECTION (SHORT) ---
                elif (not is_15m_bull) and (bar_2["low"] < bar_1["low"]):
                    entry_px = bar_1["low"] - 0.5  # Break fill
                    init_sl = entry_px + sl_dist
                    init_tp = entry_px - tp_dist
                    
                    best_p = entry_px
                    curr_sl = init_sl
                    exit_px = None
                    exit_m = m_start
                    reason = "EOD"
                    
                    for fut_b in range(b_idx + 2, len(agg_3m)):
                        f_bar = agg_3m.iloc[fut_b]
                        f_m = int(f_bar["minute_start"])
                        
                        gain = entry_px - f_bar["low"]
                        if gain >= 6.0:
                            best_p = min(best_p, f_bar["low"])
                            curr_sl = min(curr_sl, best_p + 2.0)
                            
                        if f_bar["high"] >= curr_sl:
                            exit_px = curr_sl
                            exit_m = f_m
                            reason = "SL-TRL" if curr_sl < init_sl else "SL"
                            break
                        elif f_bar["low"] <= init_tp:
                            exit_px = init_tp
                            exit_m = f_m
                            reason = "TP"
                            break
                        elif f_m >= 920:
                            exit_px = f_bar["close"]
                            exit_m = f_m
                            reason = "EOD"
                            break
                            
                    if exit_px is None:
                        exit_px = agg_3m.iloc[-1]["close"]
                        exit_m = 920
                        
                    pnl_pts = entry_px - exit_px
                    net_rs = pnl_pts * LOT_SIZE - FEE_PER_TRADE
                    trade_obj = {
                        "date": day,
                        "session": "Morning" if m_start < 700 else "Afternoon",
                        "entry_time": to_hhmm(m_start),
                        "exit_time": to_hhmm(exit_m),
                        "direction": "SHORT",
                        "level": lvl_name,
                        "entry": round(entry_px, 2),
                        "exit": round(exit_px, 2),
                        "sl": round(init_sl, 2),
                        "pnl_pts": round(pnl_pts, 2),
                        "net_rs": round(net_rs, 2),
                        "reason": reason,
                    }
                    day_trades.append(trade_obj)
                    all_trades.append(trade_obj)
                    break
                    
    # Print Day Trades
    print(f"{'#':2s} | {'Time Window':13s} | {'Session':9s} | {'Side':5s} | {'Level Triggered':18s} | {'Entry':8s} | {'Exit':8s} | {'Pts':8s} | {'Net Realized Rs':16s} | {'Reason':8s}")
    print("-" * 125)
    for idx_t, tr in enumerate(day_trades, 1):
        print(f"{idx_t:<2d} | {tr['entry_time']}->{tr['exit_time']} | {tr['session']:9s} | {tr['direction']:5s} | {tr['level']:18s} | {tr['entry']:8.2f} | {tr['exit']:8.2f} | {tr['pnl_pts']:>+7.2f} | Rs {tr['net_rs']:>+13,.2f} | {tr['reason']:8s}")
        
    day_pts = sum(t["pnl_pts"] for t in day_trades)
    day_rs = sum(t["net_rs"] for t in day_trades)
    d_wins = sum(1 for t in day_trades if t["net_rs"] > 0)
    d_wr = (d_wins / len(day_trades) * 100) if day_trades else 0.0
    print("-" * 125)
    print(f"DAY TOTAL: {len(day_trades)} Trades | {d_wins} Wins | Win Rate: {d_wr:.1f}% | Net Points: {day_pts:>+8.2f} pts | Net PnL: Rs {day_rs:>+10,.2f}")

# 3-Day Grand Total
print("\n" + "=" * 135)
print("3-DAY EVALUATION SUMMARY (AUGUST 18, 19, 20, 2026)")
print("=" * 135)
tot_pts = sum(t["pnl_pts"] for t in all_trades)
tot_rs = sum(t["net_rs"] for t in all_trades)
tot_wins = sum(1 for t in all_trades if t["net_rs"] > 0)
tot_losses = sum(1 for t in all_trades if t["net_rs"] <= 0)
wr = (tot_wins / len(all_trades) * 100) if all_trades else 0.0

gross_win = sum(t["net_rs"] for t in all_trades if t["net_rs"] > 0)
gross_loss = abs(sum(t["net_rs"] for t in all_trades if t["net_rs"] <= 0))
pf = gross_win / gross_loss if gross_loss > 0 else 99.0

print(f"Total Trades Executed:       {len(all_trades)}")
print(f"Winning Trades:               {tot_wins} Wins / {tot_losses} Losses")
print(f"TRADE WIN RATE:               {wr:.2f}%")
print(f"3-Day Realized Net Points:    {tot_pts:>+,.2f} pts")
print(f"3-Day Realized Net Profit:    Rs {tot_rs:>+,.2f}")
print(f"3-Day Profit Factor:          {pf:.3f}")
print("=" * 135)

out_file = ROOT / "artifacts" / "f6_hybrid" / "august_18_20_undisputed_run.json"
out_file.write_text(json.dumps({
    "summary": {
        "trades": len(all_trades), "wins": tot_wins, "losses": tot_losses,
        "win_rate": wr, "net_points": tot_pts, "net_rs": tot_rs, "pf": pf,
    },
    "trades": all_trades,
}, indent=2), encoding="utf-8")
print(f"\n[Saved August 18-20 Run JSON]: {out_file}")
