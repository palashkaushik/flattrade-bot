# Flattrade Quad Rotation Options Trading Bot & 5-Year Backtester

A high-performance, automated trading bot and backtest engine built for **FlatTrade** (`https://pi.flattrade.in/docs`).

## Strategy Features
- **Chart**: 1-Minute Option Charts Only (CE & PE).
- **Strike Selection**: Nifty Spot resolves 2nd In-The-Money options twice per cycle ($\text{CE} = \text{ATM} - 100$, $\text{PE} = \text{ATM} + 100$).
- **Indicators**: 4-Stochastic Engine ($S1: 9/3, S2: 14/3, S3: 40/4, S4: 60/10$).
- **Setup Filter**: Quad Flag / SuperSignal setup + **Bullish Trough Divergence** filter ($\text{Price}(T_2) < \text{Price}(T_1)$ and $S1(T_2) > S1(T_1)$).
- **Entry Trigger**: **Bullish Pin Bar candle**, confirmed when the next 1-minute candle breaks and closes above the Pin Bar's High ($\text{Close}_{\text{next}} > \text{High}_{\text{pinbar}}$).
- **Position Exits**:
  - **Stop Loss**: 10 Points in option premium.
  - **Take Profit**: 15 Points in option premium.
  - **Bearish Peak Reversal Exit**: Exits immediately at market if a Bearish Peak Divergence forms ($\text{Price}(P_2) > \text{Price}(P_1)$ and $S1(P_2) < S1(P_1)$) while holding trade.
  - **EOD Exit**: Auto square-off at 15:00.
- **Risk Management**: Session window (09:20 - 15:00), Max Daily Loss (-₹2,000), Consecutive Loss Limit (6 losses).
- **Discord Notifications**: Real-time Discord Webhook embeds for trade entries and exits with complete trade details and exact reasons.

## Quick Start

### 1. Run 5-Year Backtest
```bash
python backtest_5y_divergence.py
```

### 2. Run Verification Unit Tests
```bash
python verify_backtest.py
```

### 3. Launch Flattrade Live Trading Bot
```bash
python -m flattrade_bot.main
```
Or double-click `START_BOT.bat`.

The bot is terminal-bound: closing its terminal requests a clean shutdown and
stops market polling. `START_BOT.bat` must be run directly; do not launch it
with `start`, `Start-Process`, or another detached process wrapper.

## Discord Trading Control

The local control process connects outbound to Discord's Gateway and exposes
three allowlisted slash commands:

- `/trading start` starts the live Flattrade process with `--auto-login --live-orders`.
- `/trading stop` requests a graceful stop and closes a confirmed open position first.
- `/trading status` reports the child process state.

Install the controller dependency with:

```bash
python -m pip install -r requirements-discord-control.txt
```

Create a Discord application in the Developer Portal, add a bot, and install
it to the server with the `bot` and `applications.commands` scopes. Give it
only the permissions needed to respond in the control channel, normally
`Send Messages`.

Add these values to the local `.env` file. Never put the bot token in source
control or chat:

```text
DISCORD_CONTROL_ENABLED=true
DISCORD_BOT_TOKEN=your_bot_token
DISCORD_APPLICATION_ID=your_application_id
DISCORD_GUILD_ID=1527242045268164758
DISCORD_CONTROL_CHANNEL_ID=1527242045796651100
DISCORD_ALLOWED_USER_IDS=your_discord_user_id
BOT_START_TIME=09:15
BOT_STOP_TIME=15:00
BOT_START_GRACE_MINUTES=5
TRADING_TIMEZONE=Asia/Kolkata
```

The IDs above identify the existing `SCALPER` guild and channel found in the
`ammu` configuration. Verify them in Discord before activation. The existing
webhook remains outbound-only; it cannot receive control commands.

Run the controller manually first:

```bash
python -m flattrade_bot.discord_control
```

Available authorized Discord commands:

- `/trading start` starts the managed live bot.
- `/trading start-visible` starts the same live bot with a visible terminal.
- `/trading stop` requests a graceful stop.
- `/trading close` requests a graceful close and stop.
- `/trading status` reports the managed process state.

Trade-entry embeds include the triggering timeframe for verification.

For a visible verification session, double-click
`RESTART_DISCORD_CONTROL_VISIBLE.bat`. It stops the scheduled controller,
optionally requests a graceful bot stop, runs Discord control in the visible
terminal, and restores the scheduled task when the session exits. It never
force-kills the bot.

After `/trading status` works, schedule `START_DISCORD_CONTROL.bat` in Windows
Task Scheduler with an **At startup** trigger and the project directory as
the working directory. Do not enable the old `KBot_AutoStart_0915` task.
