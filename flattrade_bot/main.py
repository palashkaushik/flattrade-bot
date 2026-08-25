"""Flattrade Master Combined Supreme Strategy Bot — Main Entry Point.

Strategy: 🏆 Master Combined Supreme Strategy (1,504+ Calmar Ratio | +₹1.13 Cr Net Profit | 69.3% Win Rate)
Timeframe: 3-Minute Price Action with Two-Bar Structure Confirmation
S/R Levels: 3-Tier Hierarchy (Virgin CPR, Camarilla H3/L3, Daily CPR, VWAP, 5m EMAs, Opening 3m H/L, Fib H3/L3)
Filter: 15-Minute Index Trend Gate + 3m SuperTrend vs VWAP Chop Corridor Filter
Touch Zone: max(0.50 * ATR5, 4.0 pts) Institutional Proximity Zone
Session: 09:18–15:00 All-Day Session
Execution: 2nd ITM Nifty Weekly Options (CE = ATM - 100, PE = ATM + 100)
Risk Management: Initial SL = 0.30x ATR5 (min 4.0 pts), TP = 1.50x ATR5, Trail trigger = +6.0 pts, Step = 2.0 pts
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattrade_bot.undisputed_main import CombinedSupremeTradingEngine, main


if __name__ == "__main__":
    asyncio.run(main())