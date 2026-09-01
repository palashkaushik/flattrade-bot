#!/usr/bin/env bash
# Full-stack autonomous setup: systemd-owned bot lifecycle + Discord remote control.
# Run ON the VPS:  bash deploy/aws-lightsail/setup-automation.sh
set -euo pipefail

echo "=========================================================="
echo " 🚀 FLATTRADE FULL AUTOMATION STACK SETUP"
echo "=========================================================="

# 1. Timezone
echo "🕒 Setting system timezone to Asia/Kolkata..."
sudo timedatectl set-timezone Asia/Kolkata

# 2. tmux (screen-socket sessions aren't shareable with systemd cgroups)
if ! command -v tmux >/dev/null 2>&1; then
    echo "📦 Installing tmux..."
    sudo apt-get update -qq && sudo apt-get install -y tmux
fi

# 3. Passwordless systemctl for the ubuntu user (Discord control bridge)
echo "🔐 Granting ubuntu passwordless systemctl rights on flattrade-bot units..."
sudo tee /etc/sudoers.d/flattrade-bot > /dev/null <<'EOF'
ubuntu ALL=(root) NOPASSWD: /usr/bin/systemctl start flattrade-bot.service, /usr/bin/systemctl stop flattrade-bot.service, /usr/bin/systemctl restart flattrade-bot.service, /usr/bin/systemctl is-active flattrade-bot.service
EOF
sudo chmod 440 /etc/sudoers.d/flattrade-bot

# 4. Ensure log dir
mkdir -p /home/ubuntu/FLATTRADE_BOT/logs

# 5. Install units
echo "📂 Installing systemd units (tmux bot + timers + Discord supervisor)..."
sudo cp deploy/aws-lightsail/systemd/flattrade-bot.service /etc/systemd/system/
sudo cp deploy/aws-lightsail/systemd/flattrade-bot-start.timer /etc/systemd/system/
sudo cp deploy/aws-lightsail/systemd/flattrade-bot-stop.service /etc/systemd/system/
sudo cp deploy/aws-lightsail/systemd/flattrade-bot-stop.timer /etc/systemd/system/
sudo cp deploy/aws-lightsail/systemd/flattrade-discord.service /etc/systemd/system/

# 6. Kill legacy screen bot + wipe stale sessions
pkill -f "flattrade_bot.main" 2>/dev/null || true
sleep 2
screen -wipe 2>/dev/null || true
rm -f /home/ubuntu/FLATTRADE_BOT/logs/trading_bot.lock

# 7. Reload + enable
sudo systemctl daemon-reload
sudo systemctl enable flattrade-bot-start.timer flattrade-bot-stop.timer
sudo systemctl restart flattrade-discord.service

# Start the bot now if market window, else leave it to the timer
IST_MIN=$(date +%H%M)
if [ "$IST_MIN" -ge "0905" ] && [ "$IST_MIN" -le "1515" ] && [ "$(date +%u)" -le "5" ]; then
    echo "⏰ Within market hours — starting bot now..."
    sudo systemctl start flattrade-bot.service
else
    echo "⏰ Outside market hours — bot will start via 09:05 IST timer."
fi

echo ""
echo "=========================================================="
echo " ✅ FULL AUTOMATION STACK ACTIVE"
echo "=========================================================="
echo "• systemd flattrade-bot.service — bot in tmux session 'bot'"
echo "• Auto-start 09:05 IST / auto-stop 15:15 IST (Mon-Fri)"
echo "• Restart=always — crash auto-recovers in 10s"
echo "• Discord: /trading status|start|stop|restart|logs (systemctl bridge)"
echo "• Dashboard: ssh in, then:  tmux attach -t bot"
echo "• Bot logs:  journalctl -u flattrade-bot -f  OR  tail -f logs/last_hope_bot.log"
echo "• Heartbeat: logs/bot.runtime.json (refreshed every tick)"
echo "=========================================================="
