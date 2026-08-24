"""Detailed Trade-by-Trade Execution of Combined Supreme Engine on August 18-20.

Prints full trade ledger with:
  - Date & Exact Entry Time
  - Rejection Level Triggered (Tier 1 Supreme / Tier 2 / Tier 3)
  - Direction (LONG / SHORT)
  - Entry Price, Initial SL, Target Price
  - Exit Time, Exit Price, Exit Reason (Trailing SL / TP Target / EOD)
  - Points Captured & Net Realized Rs (1 Lot / 65 qty)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Websites\FLATTRADE BOT")
sys.path.insert(0, str(ROOT))

DESKTOP_DATA = Path(r"C:\Users\user\Desktop\nifty50 data")
IDX_FILE = DESKTOP_DATA / "index" / "NIFTY 50_minute.csv"

LOT_SIZE = 65
FEE_PER_TRADE = 45.0


def run_aug_study(target_year: str = "2025"):
    df_raw = pd.read_csv(IDX_FILE)
    df_raw["dt"] = pd.to_datetime(df_raw["date"])
    df_raw["day"] = df_raw["dt"].dt.strftime("%Y-%m-%d")
    df_raw["minute"] = df_raw["dt"].dt.hour * 60 + df_raw["dt"].dt.minute
    df_raw = df_raw[(df_raw["minute"] >= 555) & (df_raw["minute"] <= 930)].reset_index(drop=True)

    all_days = sorted(list(df_raw["day"].unique()))
    aug_days = [d for d in all_days if d.startswith(f"{target_year}-08-18") or d.startswith(f"{target_year}-08-19") or d.startswith(f"{target_year}-08-20")]

    if not aug_days:
        print(f"No trading days found for {target_year}-08-18 to {target_year}-08-20")
        return

    print("=" * 135)
    print(f"COMBINED SUPREME ENGINE — TRADE-BY-TRADE AUDIT: AUGUST 18-20 ({target_year})")
    print(f"Active Trading Days: {', '.join(aug_days)}")
    print("=" * 135)

    # 3m Resampling
    df_raw["bar_3m_idx"] = (df_raw["minute"] - 555) // 3
    df_raw["bar_5m_idx"] = (df_raw["minute"] - 555) // 5
    df_raw["bar_15m_idx"] = (df_raw["minute"] - 555) // 15

    agg_3m = df_raw.groupby(["day", "bar_3m_idx"]).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        minute_start=("minute", "first")
    ).reset_index()

    agg_5m = df_raw.groupby(["day", "bar_5m_idx"]).agg(
        close=("close", "last")
    ).reset_index()

    agg_15m = df_raw.groupby(["day", "bar_15m_idx"]).agg(
        close=("close", "last")
    ).reset_index()

    daily_ohlc = df_raw.groupby("day").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).to_dict("index")

    # Fast Daily Level Lookup
    daily_levels = {}
    for i in range(1, len(all_days)):
        day = all_days[i]
        prev_day = all_days[i - 1]
        p_h = daily_ohlc[prev_day]["high"]
        p_l = daily_ohlc[prev_day]["low"]
        p_c = daily_ohlc[prev_day]["close"]

        pivot = (p_h + p_l + p_c) / 3.0
        bc = (p_h + p_l) / 2.0
        tc = (pivot - bc) + pivot
        c_top, c_bot = max(tc, bc), min(tc, bc)
        cam_rng = p_h - p_l
        h3 = p_c + cam_rng * (1.1 / 4.0)
        l3 = p_c - cam_rng * (1.1 / 4.0)
        fib_h3 = pivot + cam_rng * 1.000
        fib_l3 = pivot - cam_rng * 1.000

        cur_h = daily_ohlc[day]["high"]
        cur_l = daily_ohlc[day]["low"]
        is_virgin = not ((cur_l <= c_top) and (cur_h >= c_bot))

        daily_levels[day] = {
            "pivot": pivot, "tc": c_top, "bc": c_bot,
            "h3": h3, "l3": l3, "fib_h3": fib_h3, "fib_l3": fib_l3,
            "p_h": p_h, "p_l": p_l, "p_c": p_c, "is_virgin": is_virgin,
        }

    # Indicators
    agg_3m["vwap"] = agg_3m.groupby("day").apply(
        lambda g: (g["high"] + g["low"] + g["close"]).cumsum() / (3.0 * np.arange(1, len(g) + 1))
    ).reset_index(level=0, drop=True)

    agg_3m["ema20"] = agg_3m["close"].ewm(span=20, adjust=False).mean()
    agg_3m["ema200"] = agg_3m["close"].ewm(span=200, adjust=False).mean()
    agg_5m["ema20_5m"] = agg_5m["close"].ewm(span=20, adjust=False).mean()
    agg_5m["ema200_5m"] = agg_5m["close"].ewm(span=200, adjust=False).mean()
    agg_15m["ema20_15m"] = agg_15m["close"].ewm(span=20, adjust=False).mean()

    tr_series = np.maximum(
        agg_3m["high"] - agg_3m["low"],
        np.maximum(
            np.abs(agg_3m["high"] - agg_3m["close"].shift(1).fillna(agg_3m["open"])),
            np.abs(agg_3m["low"] - agg_3m["close"].shift(1).fillna(agg_3m["open"]))
        )
    )
    agg_3m["atr5"] = tr_series.rolling(5, min_periods=1).mean().clip(lower=8.0)

    day_to_3m = {d: g.reset_index(drop=True) for d, g in agg_3m.groupby("day")}
    day_to_5m = {d: g.reset_index(drop=True) for d, g in agg_5m.groupby("day")}
    day_to_15m = {d: g.reset_index(drop=True) for d, g in agg_15m.groupby("day")}

    active_virgin_list = []
    executed_trades = []

    # Track virgin CPRs up to target days
    for day in all_days:
        if day > aug_days[-1]:
            break
        dl = daily_levels.get(day)
        if dl is None:
            continue

        surviving = []
        cur_l_min = daily_ohlc[day]["low"]
        cur_h_max = daily_ohlc[day]["high"]
        for vp, vtc, vbc, o_day in active_virgin_list:
            if not ((cur_l_min <= vtc) and (cur_h_max >= vbc)):
                surviving.append((vp, vtc, vbc, o_day))
        active_virgin_list = surviving
        if dl["is_virgin"]:
            active_virgin_list.append((dl["pivot"], dl["tc"], dl["bc"], day))

        if day not in aug_days:
            continue

        d3 = day_to_3m.get(day)
        d5 = day_to_5m.get(day)
        d15 = day_to_15m.get(day)
        if d3 is None:
            continue

        op3m_h = float(d3.iloc[0]["high"])
        op3m_l = float(d3.iloc[0]["low"])

        touch_counts = {}

        for b_idx in range(len(d3) - 1):
            b1 = d3.iloc[b_idx]
            b2 = d3.iloc[b_idx + 1]
            m_start = int(b2["minute_start"])

            if not ((555 <= m_start <= 660) or (810 <= m_start <= 900)):
                continue

            idx_15m = min((m_start - 555) // 15, len(d15) - 1) if d15 is not None else 0
            is_bull = b2["close"] >= d15.iloc[idx_15m]["ema20_15m"] if d15 is not None else True
            idx_5m = min((m_start - 555) // 5, len(d5) - 1) if d5 is not None else 0

            cur_atr = float(b1["atr5"])
            sl_pts = max(cur_atr * 0.30, 4.0)
            tp_pts = max(cur_atr * 1.50, 8.0)

            levels = [
                ("Virgin CPR Pivot", active_virgin_list[-1][0] if active_virgin_list else dl["pivot"], 1, True),
                ("Daily CPR Pivot", dl["pivot"], 1, False),
                ("Daily CPR Top", dl["tc"], 1, False),
                ("Daily CPR Bot", dl["bc"], 1, False),
                ("Camarilla H3", dl["h3"], 1, False),
                ("Camarilla L3", dl["l3"], 1, False),
                ("5m EMA 20", float(d5.iloc[idx_5m]["ema20_5m"]) if d5 is not None else 0.0, 1, False),
                ("5m EMA 200", float(d5.iloc[idx_5m]["ema200_5m"]) if d5 is not None else 0.0, 1, False),
                ("Daily VWAP", float(b1["vwap"]), 1, False),
                ("3m EMA 200", float(b1["ema200"]), 1, False),
                ("Opening 3m High", op3m_h, 2, False),
                ("Opening 3m Low", op3m_l, 2, False),
                ("3m EMA 20", float(b1["ema20"]), 2, False),
                ("Prev Day High", dl["p_h"], 2, False),
                ("Prev Day Low", dl["p_l"], 2, False),
                ("Fibonacci H3", dl["fib_h3"], 3, False),
                ("Fibonacci L3", dl["fib_l3"], 3, False),
            ]

            sorted_lvls = sorted(levels, key=lambda x: (not x[3], x[2]))

            for lvl_name, lvl_px, prio, is_v in sorted_lvls:
                if touch_counts.get(lvl_name, 0) >= 2:
                    continue

                if b1["low"] <= lvl_px <= b1["high"]:
                    score = 40 + (25 if is_v else 20 if prio == 1 else 10 if prio == 2 else 5)
                    entry_dir = None
                    entry_px = 0.0

                    if is_bull and (b2["high"] > b1["high"]):
                        score += (15 if b1["close"] > lvl_px else 0) + 25
                        if score >= 50:
                            entry_dir = "LONG"
                            entry_px = float(b1["high"] + 0.5)
                    elif (not is_bull) and (b2["low"] < b1["low"]):
                        score += (15 if b1["close"] < lvl_px else 0) + 25
                        if score >= 50:
                            entry_dir = "SHORT"
                            entry_px = float(b1["low"] - 0.5)

                    if entry_dir is not None:
                        touch_counts[lvl_name] = touch_counts.get(lvl_name, 0) + 1

                        # Trade simulation across subsequent bars
                        init_sl = entry_px - sl_pts if entry_dir == "LONG" else entry_px + sl_pts
                        init_tp = entry_px + tp_pts if entry_dir == "LONG" else entry_px - tp_pts
                        cur_sl = init_sl
                        peak_px = entry_px

                        exit_px = float(d3.iloc[-1]["close"])
                        exit_time = f"{int(d3.iloc[-1]['minute_start'])//60:02d}:{int(d3.iloc[-1]['minute_start'])%60:02d}"
                        exit_reason = "EOD Squareoff"

                        for f_idx in range(b_idx + 1, len(d3)):
                            fb = d3.iloc[f_idx]
                            f_time = f"{int(fb['minute_start'])//60:02d}:{int(fb['minute_start'])%60:02d}"

                            if entry_dir == "LONG":
                                peak_px = max(peak_px, float(fb["high"]))
                                if (peak_px - entry_px) >= 6.0:
                                    cur_sl = max(cur_sl, peak_px - 2.0)

                                if float(fb["low"]) <= cur_sl:
                                    exit_px = cur_sl
                                    exit_time = f_time
                                    exit_reason = "Trailing SL Hit" if cur_sl > init_sl else "Initial SL Hit"
                                    break
                                elif float(fb["high"]) >= init_tp:
                                    exit_px = init_tp
                                    exit_time = f_time
                                    exit_reason = "Target TP Hit"
                                    break
                            else:
                                peak_px = min(peak_px, float(fb["low"]))
                                if (entry_px - peak_px) >= 6.0:
                                    cur_sl = min(cur_sl, peak_px + 2.0)

                                if float(fb["high"]) >= cur_sl:
                                    exit_px = cur_sl
                                    exit_time = f_time
                                    exit_reason = "Trailing SL Hit" if cur_sl < init_sl else "Initial SL Hit"
                                    break
                                elif float(fb["low"]) <= init_tp:
                                    exit_px = init_tp
                                    exit_time = f_time
                                    exit_reason = "Target TP Hit"
                                    break

                        raw_pts = (exit_px - entry_px) if entry_dir == "LONG" else (entry_px - exit_px)
                        opt_pts = raw_pts * 0.60
                        net_rs = opt_pts * LOT_SIZE - FEE_PER_TRADE

                        executed_trades.append({
                            "date": day,
                            "entry_time": f"{m_start//60:02d}:{m_start%60:02d}",
                            "level": lvl_name,
                            "direction": entry_dir,
                            "entry_px": entry_px,
                            "sl_px": init_sl,
                            "tp_px": init_tp,
                            "exit_time": exit_time,
                            "exit_px": exit_px,
                            "exit_reason": exit_reason,
                            "raw_pts": raw_pts,
                            "opt_pts": opt_pts,
                            "net_rs": net_rs,
                        })
                        break

    df_tr = pd.DataFrame(executed_trades)
    print(f"\n{'DATE':10s} | {'TIME':5s} | {'DIRECTION':7s} | {'LEVEL TRIGGERED':22s} | {'ENTRY':8s} | {'EXIT':8s} | {'EXIT REASON':16s} | {'PTS':8s} | {'NET P&L (Rs)':14s}")
    print("-" * 135)
    for _, r in df_tr.iterrows():
        pts_str = f"{r['opt_pts']:+.2f} pts"
        rs_str = f"Rs {r['net_rs']:>+10,.2f}"
        print(f"{r['date']:10s} | {r['entry_time']:5s} | {r['direction']:7s} | {r['level']:22s} | {r['entry_px']:<8.2f} | {r['exit_px']:<8.2f} | {r['exit_reason']:16s} | {pts_str:8s} | {rs_str:14s}")

    print("=" * 135)
    total_net = df_tr["net_rs"].sum()
    total_pts = df_tr["opt_pts"].sum()
    wins = len(df_tr[df_tr["net_rs"] > 0])
    print(f"SUMMARY FOR AUGUST 18-20 ({target_year}):")
    print(f"  - Total Trades:      {len(df_tr)} ({wins} Wins / {len(df_tr)-wins} Losses | {wins/len(df_tr)*100:.1f}% Win Rate)")
    print(f"  - Net Options Points:{total_pts:+.2f} pts")
    print(f"  - Total Realized Rs: Rs {total_net:>+12,.2f}")
    print("=" * 135)


if __name__ == "__main__":
    run_aug_study("2025")
