#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " 🚀 FLATTRADE 24/7 AUTONOMOUS VPS INSTALLATION SCRIPT"
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_DIR="${SCRIPT_DIR}/systemd"

# 1. Set System Timezone to Asia/Kolkata
echo "🕒 Setting system timezone to Asia/Kolkata..."
sudo timedatectl set-timezone Asia/Kolkata

# 2. Install Headless Automation Engines (Playwright & Core Dependencies)
echo "⚡ Installing Playwright & Core Dependencies for 100% Autonomous Zero-Touch Auto-Login..."
pip3 install httpx pyotp rich discord.py python-dotenv playwright --break-system-packages || true
playwright install chromium --with-deps || true

# 3. Ensure log directory exists
mkdir -p /home/ubuntu/FLATTRADE_BOT/logs

# 3. Copy systemd service and timer units
echo "📂 Installing systemd services and timers..."
sudo cp "${SYSTEMD_DIR}/flattrade-bot.service" /etc/systemd/system/
sudo cp "${SYSTEMD_DIR}/flattrade-bot-start.timer" /etc/systemd/system/
sudo cp "${SYSTEMD_DIR}/flattrade-bot-stop.service" /etc/systemd/system/
sudo cp "${SYSTEMD_DIR}/flattrade-bot-stop.timer" /etc/systemd/system/
sudo cp "${SYSTEMD_DIR}/flattrade-discord.service" /etc/systemd/system/

# 4. Reload systemd daemon
echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

# 5. Enable timers and 24/7 Discord control
echo "⚡ Enabling 24/7 Autonomous Timers & Discord Supervisor..."
sudo systemctl enable --now flattrade-bot-start.timer
sudo systemctl enable --now flattrade-bot-stop.timer
sudo systemctl enable --now flattrade-discord.service

echo ""
echo "=========================================================="
echo " ✅ 24/7 AUTONOMOUS SETUP COMPLETED SUCCESSFULLY!"
echo "=========================================================="
echo "• Morning Auto-Start:  09:05 AM IST (Mon-Fri) via flattrade-bot-start.timer"
echo "• EOD Auto-Stop:       15:15 PM IST (Mon-Fri) via flattrade-bot-stop.timer"
echo "• Weekends / Holidays: Automatically Skipped (Zero Trades)"
echo "• 24/7 Discord Control: Active (/trading status, /trading logs, etc.)"
echo "• Security:            Non-root user (ubuntu), dynamic points-loss guard"
echo "=========================================================="
