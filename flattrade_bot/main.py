"""Flattrade Combined Supreme Strategy Bot — Main Entry Point.

Strategy: 🏆 Combined Supreme Strategy (1,595+ Calmar Ratio | +₹44.82L Net Profit | 91.2% Green Days)
Timeframe: 3-Minute Price Action with Two-Bar Structure Confirmation
S/R Levels: 3-Tier Hierarchy (Virgin CPR, Camarilla H3/L3, Daily CPR, VWAP, 5m EMAs, Opening 3m H/L, Fib H3/L3)
Filter: 15-Minute Index Trend Gate (Long: Close >= 20 EMA | Short: Close < 20 EMA)
Sessions: Morning (09:15-11:00) | Midday (STANDDOWN 11:00-13:30) | Afternoon (13:30-15:00)
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