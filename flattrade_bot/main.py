"""Flattrade Last Hope GPU Winner Strategy Bot — Main Entry Point.

Strategy: 🏆 Last Hope GPU Winner (7-Year Net ₹2,108,703 | 63.89% Win Rate | Max DD ₹9,303 | Calmar 226.68)
Triggers: FLAG / SUPER stochastic setups on 2nd ITM strikes (CE = ATM - 100, PE = ATM + 100).
Gating:   10-bar arming window + strict S/R bounce (touch_buffer = 0.0) on option CPR/Camarilla/EMA/VWAP.
Exits:    Symmetric ATR(10)×1.5 distance + Breakeven stop hardening at +50% of distance to Entry + 1.0 pt.
Docs:     LAST_HOPE_WINNER.md
"""

import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flattrade_bot.last_hope_main import main


if __name__ == "__main__":
    asyncio.run(main())

