"""Export and Display Detailed Trade Logs for Verification."""

import sys
from pathlib import Path
import pandas as pd

from backtest_5y_divergence import load_spot, option_files, load_day_options, run_day, summarize

def main():
    spot = load_spot()
    paths = option_files("2024-01-01", "2024-12-31")
    days = sorted(set(paths.keys()) & set(spot.keys()))
    
    print(f"Exporting trades for 2024 ({len(days)} trading days)...", flush=True)
    all_trades = []
    
    for day in days:
        opt_record = load_day_options(paths[day])
        trades = run_day(day, spot[day], opt_record, require_divergence=True, require_pinbar=True)
        all_trades.extend(trades)
        
    df = pd.DataFrame(all_trades)
    df.to_csv("trades_2024_verification.csv", index=False)
    print(f"\nSaved {len(df)} detailed trades to trades_2024_verification.csv\n")
    
    print("=" * 115)
    print(f"{'DATE':10s} | {'ENTRY':5s} | {'EXIT':5s} | {'SIDE':4s} | {'SYMBOL':22s} | {'ENTRY_PX':8s} | {'EXIT_PX':8s} | {'PTS':7s} | {'P&L (Rs)':10s} | {'REASON'}")
    print("=" * 115)
    
    # Print sample of last 25 trades
    sample = all_trades[-25:]
    for t in sample:
        entry_time = f"{t['entry_min'] // 60:02d}:{t['entry_min'] % 60:02d}"
        exit_time = f"{t['exit_min'] // 60:02d}:{t['exit_min'] % 60:02d}"
        pts_str = f"{t['pts']:+.2f}"
        rs_str = f"Rs {t['rs']:+,d}"
        print(f"{t['date']:10s} | {entry_time:5s} | {exit_time:5s} | {t['side']:4s} | {t['symbol']:22s} | {t['entry']:8.2f} | {t['exit']:8.2f} | {pts_str:>7s} | {rs_str:>10s} | {t['reason']}")

if __name__ == "__main__":
    main()
