"""Risk Management Engine — Session Window, Daily Loss Shutdown & Consecutive Loss Controls."""

import logging
from dataclasses import dataclass
from typing import Tuple

from flattrade_bot.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    daily_pnl_rs: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    is_blocked: bool = False
    is_shutdown: bool = False


class RiskManager:
    """Controls trading risk limits and daily shutdown triggers."""

    def __init__(
        self,
        max_daily_loss_rs: float = None,
        max_daily_loss_points: float = settings.MAX_DAILY_LOSS_POINTS,
        quantity: int = settings.LOT_SIZE,
        consecutive_loss_limit: int = settings.CONSECUTIVE_LOSS_LIMIT,
        session_start_min: int = 9 * 60 + 20,   # 09:20
        session_end_min: int = 15 * 60 + 0,     # 15:00
    ):
        self.quantity = int(quantity)
        self.max_daily_loss_points = float(max_daily_loss_points)
        if max_daily_loss_rs is not None:
            self.max_daily_loss_rs = float(max_daily_loss_rs)
        else:
            self.max_daily_loss_rs = round(self.max_daily_loss_points * self.quantity, 2)
        self.consecutive_loss_limit = consecutive_loss_limit
        self.session_start_min = session_start_min
        self.session_end_min = session_end_min
        self.state = RiskState()

    def set_quantity(self, quantity: int):
        """Dynamically scales rupee loss limit when lot size / quantity changes."""
        self.quantity = int(quantity)
        self.max_daily_loss_rs = round(self.max_daily_loss_points * self.quantity, 2)
        logger.info(
            f"🔄 RiskManager scaled: Qty = {self.quantity} | Max Daily Loss = ₹{self.max_daily_loss_rs:,.2f} ({self.max_daily_loss_points} pts)"
        )

    def reset_day(self):
        self.state = RiskState()

    def is_session_active(self, current_min: int) -> bool:
        """Returns True if current minute is within trading window 09:20 - 15:00."""
        return self.session_start_min <= current_min < self.session_end_min

    def is_session_complete(self, current_min: int) -> bool:
        """Returns True once the configured session end minute has been reached."""
        return current_min >= self.session_end_min

    def can_open_trade(self, current_min: int, open_positions_count: int) -> Tuple[bool, str]:
        """Evaluates whether a new trade entry is permitted under risk rules."""
        if open_positions_count > 0:
            return False, "Position already open"

        if not self.is_session_active(current_min):
            return False, f"Outside session window (09:20 - 15:00)"

        if self.state.is_shutdown:
            return False, "Daily shutdown active (Max Loss hit)"

        if self.state.is_blocked:
            return False, "Trading blocked (Consecutive Loss Limit hit)"

        if self.state.daily_pnl_rs <= -self.max_daily_loss_rs:
            self.state.is_shutdown = True
            return False, f"Daily P&L ({self.state.daily_pnl_rs:.2f}) exceeded max loss (-{self.max_daily_loss_rs})"

        if self.state.consecutive_losses >= self.consecutive_loss_limit:
            self.state.is_blocked = True
            return False, f"Consecutive loss limit ({self.consecutive_loss_limit}) reached"

        return True, "Allowed"

    def record_trade_result(self, pnl_rs: float):
        """Updates risk state after a trade closes."""
        self.state.trades_today += 1
        self.state.daily_pnl_rs += pnl_rs

        if pnl_rs <= 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        if self.state.daily_pnl_rs <= -self.max_daily_loss_rs:
            self.state.is_shutdown = True
            logger.warning(f"🚨 DAILY SHUTDOWN TRIGGERED: P&L = ₹{self.state.daily_pnl_rs:,.2f}")

        if self.state.consecutive_losses >= self.consecutive_loss_limit:
            self.state.is_blocked = True
            logger.warning(f"⚠️ CONSECUTIVE LOSS BLOCK TRIGGERED: {self.state.consecutive_losses} losses in a row")

    def sync_broker_state(self, daily_pnl_rs: float, trades_today: int) -> None:
        """Makes broker-reported realized P&L authoritative after restarts/manual exits."""
        self.state.daily_pnl_rs = round(float(daily_pnl_rs), 2)
        self.state.trades_today = max(0, int(trades_today))
        if self.state.daily_pnl_rs <= -self.max_daily_loss_rs:
            self.state.is_shutdown = True
            logger.warning(
                "🚨 DAILY SHUTDOWN FROM BROKER P&L: ₹%.2f",
                self.state.daily_pnl_rs,
            )
