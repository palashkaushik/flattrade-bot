"""Macro Filters Backtest using real intraday VIX and Dow Jones 1m data.

Filter Logic:
  VIX Filter: At 09:15 IST, if VIX_open < previous_day_VIX_close -> CALMING -> CE allowed
                             if VIX_open >= previous_day_VIX_close -> EXPANDING -> PE allowed
  Dow Filter: If Dow previous-day return (close > prev_close) -> BULLISH -> CE allowed
                                                               -> BEARISH -> PE allowed

Evaluated only on exact overlapping trading days:
  - VIX Filter: Full 2020-2024 overlap.
  - Dow Filter: 2024-only overlap (Dow data starts 2024-01-02).
"""

import re
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from backtest_5y_optimized import load_spot, option_files, SYM_RE, to_minutes, latest_spot, TimeframeTracker, summarize, print_yearly_breakdown
from flattrade_bot.indicators.patterns import Candle

LOT_SIZE = 65
CE_OFFSET, PE_OFFSET = -100, 100
SESSION_START, SESSION_END, DAY_LAST = 560, 900, 930
DAILY_MAX_PROFIT_PTS = 30.0
DAILY_MAX_LOSS_PTS = -30.0
CONSECUTIVE_LOSS_LIMIT = 6

TF_SPECS = {
    "1m": (1, 10, 6.0, 30.0),
    "2m": (2, 5, 10.0, 15.0),
    "3m": (3, 4, 8.0, 25.0),
    "5m": (5, 3, 10.0, 35.0),
}

GLOBAL_SPOT = {}
DESKTOP = Path("C:/Users/user/Desktop/nifty50 data")


def init_worker_local(spot_dict):
    global GLOBAL_SPOT
    GLOBAL_SPOT = spot_dict


def load_vix_daily_sentiment() -> Dict[str, str]:
    """
    For each Nifty trading day:
      - Compare today's VIX open at 09:15 vs previous trading day's VIX close.
      - CALMING (VIX falling) -> CE trades allowed.
      - EXPANDING (VIX rising) -> PE trades allowed.
    """
    df = pd.read_csv(DESKTOP / "INDIA VIX_minute.csv", usecols=["date", "open", "close"])
    df["dt"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    # Per-day: get first bar open (09:15) and last bar close
    day_open = df.groupby("dt")["open"].first().to_dict()
    day_close = df.groupby("dt")["close"].last().to_dict()

    days_sorted = sorted(day_close.keys())
    sentiment = {}
    for i, day in enumerate(days_sorted):
        if i == 0:
            sentiment[day] = "NEUTRAL"
            continue
        prev_day = days_sorted[i - 1]
        today_open = day_open.get(day)
        prev_close = day_close.get(prev_day)
        if today_open is None or prev_close is None:
            sentiment[day] = "NEUTRAL"
        elif today_open < prev_close:
            sentiment[day] = "CALMING"   # VIX falling -> Nifty bullish -> CE
        else:
            sentiment[day] = "EXPANDING" # VIX rising  -> Nifty bearish -> PE

    return sentiment


def load_dow_daily_sentiment() -> Dict[str, str]:
    """
    Dow Jones 1m data is UTC timestamps. US market 09:30-16:00 EST = 14:30-21:00 UTC.
    We use the daily close-to-close return of the PREVIOUS US trading day to forecast next Nifty day.
    """
    df = pd.read_csv(DESKTOP / "DowJones1m.csv", usecols=["time", "close"])
    df["utc_dt"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["utc_dt"].dt.date.astype(str)

    # Daily last close per US trading day
    day_close = df.groupby("date")["close"].last().to_dict()
    us_days_sorted = sorted(day_close.keys())

    # Map: each US trading day's return applies to the NEXT Nifty trading day
    # US market closes ~21:30 IST, Nifty opens next morning 09:15 IST
    us_return = {}
    for i, day in enumerate(us_days_sorted):
        if i == 0:
            continue
        prev_day = us_days_sorted[i - 1]
        ret = day_close[day] - day_close[prev_day]
        us_return[day] = "BULLISH" if ret >= 0 else "BEARISH"

    # Map US date -> next calendar day's Nifty sentiment
    # Simple approach: for each Nifty trading day, use the most recent US close before it
    sentiment = {}
    for nifty_day in pd.date_range("2024-01-01", "2024-12-31").strftime("%Y-%m-%d"):
        us_before = [d for d in us_days_sorted if d < nifty_day]
        if not us_before:
            sentiment[nifty_day] = "NEUTRAL"
        else:
            latest_us = max(us_before)
            sentiment[nifty_day] = us_return.get(latest_us, "NEUTRAL")

    return sentiment


class MultiTimeframeTracker:
    def __init__(self):
        self.trackers = {
            tf: TimeframeTracker(tf, max_lookback=spec[1]) for tf, spec in TF_SPECS.items()
        }
        self.buffers = {tf: [] for tf in TF_SPECS}

    def push_1m(self, c1m: Candle) -> List[Tuple[str, bool, str, float]]:
        triggers = []
        for tf, spec in TF_SPECS.items():
            tf_m = spec[0]
            buf = self.buffers[tf]
            buf.append(c1m)
            if len(buf) == tf_m:
                c_tf = Candle(
                    open=buf[0].open, high=max(c.high for c in buf),
                    low=min(c.low for c in buf), close=buf[-1].close, minute=buf[-1].minute
                )
                self.buffers[tf] = []
                trig, is_rev, stype, px = self.trackers[tf].push(c_tf)
                if trig:
                    triggers.append((tf, is_rev, stype, px))
        return triggers


def process_single_day(args):
    day, file_path_str, prev_file_path_str, allowed_ce, allowed_pe, filter_mode = args
    spot = GLOBAL_SPOT.get(day)
    if spot is None or not file_path_str:
        return []

    file_path = Path(file_path_str)
    if not file_path.exists():
        return []

    spot_0915 = latest_spot(spot, 555) or latest_spot(spot, 560)
    if spot_0915 is None:
        return []

    atm_approx = int(round(spot_0915 / 50.0) * 50)
    target_strikes = set(range(atm_approx - 250, atm_approx + 300, 50))

    try:
        df_curr = pd.read_csv(file_path, usecols=["time", "symbol", "open", "high", "low", "close"], engine="c")
    except Exception:
        return []

    if df_curr.empty:
        return []

    first_sym = df_curr["symbol"].iloc[0]
    m_match = SYM_RE.match(first_sym)
    if not m_match:
        return []
    prefix = m_match.group(1)

    df_curr["min"] = np.array([to_minutes(t) for t in df_curr["time"]])

    groups_curr = {}
    for sym, g in df_curr.groupby("symbol"):
        m = SYM_RE.match(sym)
        if m and int(m.group(2)) in target_strikes:
            groups_curr[sym] = g

    groups_prev = {}
    if prev_file_path_str and Path(prev_file_path_str).exists():
        try:
            df_prev = pd.read_csv(prev_file_path_str, usecols=["time", "symbol", "open", "high", "low", "close"], engine="c")
            if not df_prev.empty:
                df_prev["min"] = np.array([to_minutes(t) for t in df_prev["time"]])
                for sym, g in df_prev.groupby("symbol"):
                    m = SYM_RE.match(sym)
                    if m and int(m.group(2)) in target_strikes:
                        groups_prev[sym] = g
        except Exception:
            pass

    mtf_trackers = {}
    for sym, g in groups_prev.items():
        mtf_trackers[sym] = MultiTimeframeTracker()
        mins = g["min"].to_numpy(); opens = g["open"].to_numpy()
        highs = g["high"].to_numpy(); lows = g["low"].to_numpy(); closes = g["close"].to_numpy()
        for i in range(len(mins)):
            mtf_trackers[sym].push_1m(Candle(open=opens[i], high=highs[i], low=lows[i], close=closes[i], minute=mins[i]))

    per_minute_triggers = {}
    slices = {}

    for sym, g in groups_curr.items():
        if sym not in mtf_trackers:
            mtf_trackers[sym] = MultiTimeframeTracker()
        tracker = mtf_trackers[sym]
        mins = g["min"].to_numpy(); opens = g["open"].to_numpy()
        highs = g["high"].to_numpy(); lows = g["low"].to_numpy(); closes = g["close"].to_numpy()
        slices[sym] = {"min": mins, "open": opens, "high": highs, "low": lows, "close": closes}
        m_match = SYM_RE.match(sym)
        if not m_match:
            continue
        strike_val = int(m_match.group(2))
        side_val = m_match.group(3)
        for i in range(len(mins)):
            m = mins[i]
            trig_list = tracker.push_1m(Candle(open=opens[i], high=highs[i], low=lows[i], close=closes[i], minute=m))
            for (tf_label, is_rev, stype, px) in trig_list:
                sl_pts = TF_SPECS[tf_label][2]; tp_pts = TF_SPECS[tf_label][3]
                per_minute_triggers.setdefault(m, []).append(
                    (side_val, strike_val, sym, px, is_rev, tf_label, sl_pts, tp_pts))

    trades = []
    pos = None
    daily_pnl_pts = 0.0
    consecutive_losses = 0
    shutdown = False

    def bar_at_slice(sl, minute):
        idx = np.searchsorted(sl["min"], minute)
        if idx < len(sl["min"]) and sl["min"][idx] == minute:
            return sl["open"][idx], sl["high"][idx], sl["low"][idx], sl["close"][idx]
        return None

    def get_active_info(side, minute):
        spot_px = latest_spot(spot, minute)
        if spot_px is None: return None
        atm = int(round(spot_px / 50.0) * 50)
        strike = atm + (CE_OFFSET if side == "CE" else PE_OFFSET)
        symbol = f"{prefix}{strike}{side}"
        sl = slices.get(symbol)
        return (symbol, sl, strike) if sl is not None else None

    for minute in range(SESSION_START, DAY_LAST + 1):
        if pos is not None:
            held = bar_at_slice(pos["slice"], minute)
            if held is not None:
                o_px, h_px, l_px, c_px = held
                pos["last_px"] = float(c_px)
                pos["duration_min"] += 1
                if daily_pnl_pts + (c_px - pos["entry"]) <= DAILY_MAX_LOSS_PTS:
                    pts = round(c_px - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": c_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": "SHUTDOWN_LOSS", "duration_min": pos["duration_min"],
                        "is_rev": pos["is_rev"], "tf": pos["tf"]})
                    daily_pnl_pts += pts; pos = None; shutdown = True; continue

                exit_px, reason = None, ""
                if h_px >= pos["tgt"] and l_px <= pos["sl"]: exit_px, reason = pos["sl"], "SL"
                elif h_px >= pos["tgt"]: exit_px, reason = pos["tgt"], "TP"
                elif l_px <= pos["sl"]: exit_px, reason = pos["sl"], "SL"

                if exit_px is None:
                    tr = mtf_trackers.get(pos["symbol"])
                    if tr:
                        t1 = tr.trackers["1m"]
                        t1.divergence.update(c_px, t1.prev_s1)
                        if t1.divergence.has_bearish_peak_divergence():
                            exit_px, reason = c_px, "BEARISH_PEAK_REVERSAL"

                if exit_px is not None:
                    pts = round(exit_px - pos["entry"], 2)
                    trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                        "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                        "exit": exit_px, "pts": pts, "rs": round(pts * LOT_SIZE),
                        "reason": reason, "duration_min": pos["duration_min"],
                        "is_rev": pos["is_rev"], "tf": pos["tf"]})
                    daily_pnl_pts += pts
                    consecutive_losses = consecutive_losses + 1 if pts <= 0 else 0
                    if daily_pnl_pts >= DAILY_MAX_PROFIT_PTS: shutdown = True
                    elif consecutive_losses >= CONSECUTIVE_LOSS_LIMIT or daily_pnl_pts <= DAILY_MAX_LOSS_PTS: shutdown = True
                    pos = None

            if minute >= SESSION_END and pos is not None:
                pts = round(pos["last_px"] - pos["entry"], 2)
                trades.append({"date": day, "entry_min": pos["entry_min"], "exit_min": minute,
                    "side": pos["side"], "symbol": pos["symbol"], "entry": pos["entry"],
                    "exit": pos["last_px"], "pts": pts, "rs": round(pts * LOT_SIZE),
                    "reason": "EOD", "duration_min": pos["duration_min"],
                    "is_rev": pos["is_rev"], "tf": pos["tf"]})
                daily_pnl_pts += pts; pos = None; break

        if pos is not None or shutdown or minute >= SESSION_END:
            continue

        trigs = per_minute_triggers.get(minute, [])
        for (signal_side, signal_strike, signal_symbol, c_px, is_rev, tf_label, sl_pts, tp_pts) in trigs:
            active_info = get_active_info(signal_side, minute)
            if active_info and active_info[2] == signal_strike and pos is None:
                if is_rev:
                    actual_side = "PE" if signal_side == "CE" else "CE"
                    actual_info = get_active_info(actual_side, minute)
                    if actual_info is None: continue
                    actual_symbol, actual_slice, _ = actual_info
                else:
                    actual_side = signal_side; actual_symbol = signal_symbol; actual_slice = active_info[1]

                # Macro filter gate
                if filter_mode != "none":
                    if actual_side == "CE" and not allowed_ce: continue
                    if actual_side == "PE" and not allowed_pe: continue

                bar = bar_at_slice(actual_slice, minute)
                if bar is not None:
                    entry_px = float(bar[3])
                    pos = {"side": actual_side, "symbol": actual_symbol, "slice": actual_slice,
                        "entry": entry_px, "sl": entry_px - sl_pts, "tgt": entry_px + tp_pts,
                        "entry_min": minute, "last_px": entry_px, "duration_min": 0,
                        "is_rev": is_rev, "tf": tf_label}
                    break

    return trades


def run_mode(filter_mode: str, label: str, days_subset=None):
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    all_days = sorted(set(files.keys()) & set(spot_all.keys()))

    vix_sentiment = load_vix_daily_sentiment()
    dow_sentiment = load_dow_daily_sentiment()

    if days_subset is not None:
        working_days = [d for d in all_days if d in days_subset]
    else:
        working_days = all_days

    tasks = []
    for i, day in enumerate(working_days):
        vs = vix_sentiment.get(day, "NEUTRAL")
        ds = dow_sentiment.get(day, "NEUTRAL")
        prev_day_file = str(files[all_days[all_days.index(day)-1]]) if all_days.index(day) > 0 else ""

        if filter_mode == "vix":
            allowed_ce = (vs == "CALMING")
            allowed_pe = (vs == "EXPANDING")
        elif filter_mode == "dow":
            allowed_ce = (ds == "BULLISH")
            allowed_pe = (ds == "BEARISH")
        else:
            allowed_ce = True
            allowed_pe = True

        tasks.append((day, str(files[day]), prev_day_file, allowed_ce, allowed_pe, filter_mode))

    all_trades = []
    with Pool(processes=min(cpu_count(), 8), initializer=init_worker_local, initargs=(spot_all,)) as pool:
        results = pool.map(process_single_day, tasks)
        for res in results:
            all_trades.extend(res)

    st = summarize(all_trades)
    print(f"\n{'='*115}")
    print(f"RESULTS ({len(working_days)} OVERLAPPING DAYS) FOR: {label.upper()}")
    print(f"{'='*115}")
    print(f"Total Trades : {st['trades']}")
    print(f"Win Rate     : {st['wr']:.1f}%")
    print(f"Net Points   : {st['pts']:+.2f} pts")
    print(f"Net Profit   : Rs {st['rs']:+,d}")
    pf_str = f"{st['pf']:.2f}" if st['pf'] != float("inf") else "INF"
    print(f"Profit Factor: {pf_str}")
    print_yearly_breakdown(all_trades)
    return st, len(working_days)


def main():
    spot_all = load_spot()
    files = option_files("2020-01-01", "2024-12-31")
    all_days = set(files.keys()) & set(spot_all.keys())

    vix_sentiment = load_vix_daily_sentiment()
    dow_sentiment = load_dow_daily_sentiment()

    # Days with valid VIX data: 2020-2024 full coverage
    vix_days = {d for d in all_days if vix_sentiment.get(d, "NEUTRAL") != "NEUTRAL"}
    # Days with valid Dow data: 2024 only
    dow_days = {d for d in all_days if dow_sentiment.get(d, "NEUTRAL") != "NEUTRAL"}

    print(f"Loaded {len(all_days)} total Nifty days | VIX overlap: {len(vix_days)} days | Dow overlap: {len(dow_days)} days", flush=True)

    # Backtest 1: VIX Filter (2020-2024 overlap)
    print("\nRunning Backtest 1: India VIX Intraday Filter on 2020-2024 overlap...", flush=True)
    st_vix_base, n1 = run_mode("none", "1. Baseline (No Filter, VIX-Overlap Days 2020-2024)", vix_days)
    st_vix_filt, n2 = run_mode("vix", "2. India VIX Filter (CE on Calming, PE on Expanding)", vix_days)

    # Backtest 2: Dow Filter (2024 only overlap)
    print("\nRunning Backtest 2: Dow Jones Previous-Day Filter on 2024 overlap...", flush=True)
    st_dow_base, n3 = run_mode("none", "3. Baseline (No Filter, Dow-Overlap Days 2024-only)", dow_days)
    st_dow_filt, n4 = run_mode("dow", "4. Dow Jones Previous-Day Filter (CE on Green Dow, PE on Red Dow)", dow_days)

    print(f"\n{'='*115}")
    print("MACRO FILTER COMPARISON SUMMARY (Intraday Data from Desktop)")
    print(f"{'='*115}")
    print(f"{'FILTER CONFIGURATION':45s} | {'DAYS':5s} | {'TRADES':7s} | {'WIN RATE':9s} | {'NET PTS':10s} | {'PROFIT (Rs)':13s} | PF")
    print(f"{'-'*115}")
    for name, st, n in [
        ("Baseline VIX Overlap (2020-2024)", st_vix_base, n1),
        ("India VIX Intraday Filter",       st_vix_filt, n2),
        ("Baseline Dow Overlap (2024-only)", st_dow_base, n3),
        ("Dow Jones Prev-Day Filter",        st_dow_filt, n4),
    ]:
        pf_s = f"{st['pf']:.2f}" if st['pf'] != float("inf") else "INF"
        print(f"{name:45s} | {n:5d} | {st['trades']:7d} | {st['wr']:8.1f}% | {st['pts']:+10.2f} | Rs {st['rs']:+10,d} | {pf_s}")


if __name__ == "__main__":
    main()
