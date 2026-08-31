"""Flattrade Trading Bot Configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


PROJECT_DIR = Path(__file__).parent.parent


@dataclass
class Settings:
    # ── Flattrade API Auth ──
    FLATTRADE_USER_ID: str = os.getenv("FLATTRADE_USER_ID", "")
    FLATTRADE_API_KEY: str = os.getenv("FLATTRADE_API_KEY", "")
    FLATTRADE_API_SECRET: str = os.getenv("FLATTRADE_API_SECRET", "")
    FLATTRADE_TOTP_KEY: str = os.getenv("FLATTRADE_TOTP_KEY", "")
    FLATTRADE_PASSWORD: str = os.getenv("FLATTRADE_PASSWORD", "")
    FLATTRADE_API_URL: str = os.getenv("FLATTRADE_API_URL", "https://piconnect.flattrade.in/PiConnectAPI/")
    FLATTRADE_WS_URL: str = os.getenv("FLATTRADE_WS_URL", "wss://piconnect.flattrade.in/PiConnectWSTp/")
    
    # ── Discord Webhook ──
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    DISCORD_BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "")
    DISCORD_GUILD_ID: str = os.getenv("DISCORD_GUILD_ID", "")
    DISCORD_CONTROL_CHANNEL_ID: str = os.getenv("DISCORD_CONTROL_CHANNEL_ID", "")
    DISCORD_ALLOWED_USER_IDS: str = os.getenv("DISCORD_ALLOWED_USER_IDS", "")
    DISCORD_CONTROL_ENABLED: bool = _env_bool("DISCORD_CONTROL_ENABLED", False)
    DISCORD_APPLICATION_ID: str = os.getenv("DISCORD_APPLICATION_ID", "")

    # ── Nifty Option Contracts ──
    NIFTY_SPOT_TOKEN: str = "NSE|26000"  # Nifty 50 Index token
    LOT_SIZE: int = 65
    STRIKE_STEP: int = 50
    CE_STRIKE_OFFSET: int = 0    # ATM Exact (0 offset)
    PE_STRIKE_OFFSET: int = 0    # ATM Exact (0 offset)

    # ── Live Strategy: Pocket Money (10s FLAG/SUPER scalper) ──
    # 2nd ITM strikes (CE ATM-100 / PE ATM+100), index UT Bot + LinReg side filter.
    # SL/TP = ±7.0 premium pts (SL priority), EOD flat at 15:00 IST, 4-loss day block.
    # Verified congruent with artifacts/f6_hybrid/pocket_money_backtest.py (9/9 trades).
    STRATEGY_NAME: str = "POCKET_MONEY"
    UNDISPUTED_MIN_SCORE: int = 50
    UNDISPUTED_SL_MULT: float = 0.30
    UNDISPUTED_TP_MULT: float = 1.50
    UNDISPUTED_MIN_SL_PTS: float = 4.0
    UNDISPUTED_MAX_SL_PTS: float = 15.0
    UNDISPUTED_TRAIL_TRIGGER: float = 6.0
    UNDISPUTED_TRAIL_STEP: float = 2.0
    UNDISPUTED_MAX_TOUCHES: int = 2
    UNDISPUTED_CONSECUTIVE_LOSS_LIMIT: int = 4
    UNDISPUTED_SESSION_1_START: int = 555   # 09:15 IST
    UNDISPUTED_SESSION_1_END: int = 660     # 11:00 IST
    UNDISPUTED_SESSION_2_START: int = 810   # 13:30 IST
    UNDISPUTED_SESSION_2_END: int = 900     # 15:00 IST
    UNDISPUTED_EOD_MINUTE: int = 920        # 15:20 IST Hard Exit

    # ── Legacy Option A Parameters (Quad Pinbar F6 Reference) ──
    S1_SPEC: tuple = (12, 3)
    S2_SPEC: tuple = (14, 3)
    S3_SPEC: tuple = (40, 4)
    S4_SPEC: tuple = (50, 10)
    F6_S4_THRESH: float = 79.5
    F6_S1_THRESH: float = 20.5
    ATR_PERIOD: int = 10
    ATR_SL_MULT: float = 3.0
    ATR_TP_MULT: float = 6.0
    SL_POINTS: float = 10.0   # Fallback default
    TP_POINTS: float = 30.0   # Fallback default

    # ── Session & Risk Management ──
    SESSION_START: str = "09:20"  # B17 Session Start (matches backtest)
    SESSION_END: str = "15:00"    # B17 EOD Exit at 15:00
    BOT_START_TIME: str = os.getenv("BOT_START_TIME", "09:15")
    BOT_STOP_TIME: str = os.getenv("BOT_STOP_TIME", "15:00")
    BOT_START_GRACE_MINUTES: int = int(os.getenv("BOT_START_GRACE_MINUTES", "5"))
    TRADING_TIMEZONE: str = os.getenv("TRADING_TIMEZONE", "Asia/Kolkata")
    BOT_STOP_FILE: Path = Path(os.getenv("BOT_STOP_FILE", str(PROJECT_DIR / "logs" / "stop.requested")))
    BOT_RUNTIME_FILE: Path = Path(os.getenv("BOT_RUNTIME_FILE", str(PROJECT_DIR / "logs" / "bot.runtime.json")))
    BOT_POSITION_FILE: Path = Path(os.getenv("BOT_POSITION_FILE", str(PROJECT_DIR / "logs" / "bot.position.json")))
    BOT_VISIBLE_CONSOLE: bool = _env_bool("BOT_VISIBLE_CONSOLE", False)
    BOT_VISIBLE_TASK_NAME: str = os.getenv("BOT_VISIBLE_TASK_NAME", "\\Flattrade Bot Visible")
    LIVE_TRADING: bool = _env_bool("LIVE_TRADING", False)
    MAX_DAILY_LOSS_POINTS: float = float(os.getenv("MAX_DAILY_LOSS_POINTS", "30.0"))
    MAX_DAILY_LOSS_RS: float = float(os.getenv("MAX_DAILY_LOSS_RS", str(MAX_DAILY_LOSS_POINTS * LOT_SIZE)))
    CONSECUTIVE_LOSS_LIMIT: int = 8

    # ── Paths ──
    DATA_DIR: Path = Path(os.getenv("AMMU_DIR", str(PROJECT_DIR / "data")))
    OPTS_DIR: Path = DATA_DIR / "nifty_options"
    SPOT_PATH: Path = DATA_DIR / "index" / "NIFTY 50_minute.csv"
    LOGS_DIR: Path = PROJECT_DIR / "logs"


settings = Settings()
